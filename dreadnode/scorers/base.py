import asyncio
import inspect
import typing as t
from copy import deepcopy
from datetime import datetime, timezone

import typing_extensions as te

from dreadnode.common_types import UNSET, JsonDict, Unset
from dreadnode.meta import Component, ConfigInfo, Context
from dreadnode.meta.context import TaskInput, TaskOutput
from dreadnode.metric import Metric
from dreadnode.util import get_callable_name, warn_at_user_stacklevel

T = t.TypeVar("T")
T_contra = t.TypeVar("T_contra", contravariant=True)

OuterT = te.TypeVar("OuterT", default=t.Any)
UnusedP = te.ParamSpec("UnusedP", default=...)


class ScorerWarning(UserWarning):
    """Warning related to scorer mechanics."""


ScorerResult = float | int | bool | Metric
"""The result of a scorer function, which can be a numeric value or a Metric object."""


@t.runtime_checkable
class ScorerCallable(t.Protocol, t.Generic[T_contra]):
    """
    A callable that takes an object and returns a compatible score result.
    - Can take just the object or additional args/kwargs
    - Can return single result or sequence
    - Can be sync or async
    """

    def __call__(
        self, obj: T_contra, /, *args: t.Any, **kwargs: t.Any
    ) -> (
        t.Awaitable[ScorerResult]
        | ScorerResult
        | t.Awaitable[t.Sequence[ScorerResult]]
        | t.Sequence[ScorerResult]
    ): ...


class Scorer(Component[te.Concatenate[T, ...], t.Any], t.Generic[T]):
    """
    A stateful, configurable, and composable wrapper for a scoring function.

    A Scorer is a specialized Component that evaluates an object and produces a Metric.
    It inherits the configuration and context-awareness of a Component, allowing
    scorers to be defined with `dn.Config` and `dn.Context` parameters.
    """

    def __init__(
        self,
        func: ScorerCallable[T],
        *,
        name: str | None = None,
        attributes: JsonDict | None = None,
        catch: bool = False,
        step: int = 0,
        auto_increment_step: bool = False,
        log_all: bool = True,
        bound_obj: t.Any | Unset = UNSET,
        config: dict[str, ConfigInfo] | None = None,
        context: dict[str, Context] | None = None,
        wraps: t.Callable[..., t.Any] | None = None,
    ):
        if isinstance(func, Scorer):
            func = func.func

        super().__init__(func, name=name, config=config, context=context, wraps=wraps)

        self.name = self.name
        "The name of the scorer, used for reporting metrics."
        self.attributes = attributes or {}
        "A dictionary of attributes for metrics produced by this Scorer."
        self.catch = catch
        "Catch exceptions in the scorer function and return a 0 Metric with error information."
        self.step = step
        "The step value to attach to metrics produced by this Scorer."
        self.auto_increment_step = auto_increment_step
        "Automatically increment an internal step counter every time this scorer is called."
        self.log_all = log_all
        "Log all sub-metrics from nested composition, or just the final resulting metric."
        self.bound_obj = bound_obj
        "If set, the scorer will always be called with this object instead of the caller-provided object."

    def __repr__(self) -> str:
        func_name = get_callable_name(self.func, short=True)

        parts: list[str] = [
            f"name='{self.name}'",
            f"func={func_name}",
            f"catch={self.catch}",
            f"log_all={self.log_all}",
            f"auto_increment_step={self.auto_increment_step}",
            f"step={self.step}",
        ]

        return f"{self.__class__.__name__}({', '.join(parts)})"

    @classmethod
    def fit(cls, scorer: "ScorerLike[T]") -> "Scorer[T]":
        """
        Fit a scorer to the given attributes.

        Args:
            scorer: The scorer to fit.

        Returns:
            A Scorer instance.
        """
        return scorer if isinstance(scorer, Scorer) else cls(scorer)

    @classmethod
    def fit_many(cls, scorers: "ScorersLike[T] | None") -> list["Scorer[T]"]:
        """
        Convert a collection of scorer-like objects into a list of Scorer instances.

        This method provides a flexible way to handle different input formats for scorers,
        automatically converting callables to Scorer objects and applying consistent naming
        and attributes across all scorers.

        Args:
            scorers: A collection of scorer-like objects. Can be:
                - A dictionary mapping names to scorer objects or callables
                - A sequence of scorer objects or callables
                - None (returns empty list)

        Returns:
            A list of Scorer instances with consistent configuration.
        """
        if isinstance(scorers, t.Mapping):
            return [
                scorer.with_(name=name) if isinstance(scorer, Scorer) else cls(scorer, name=name)
                for name, scorer in scorers.items()
            ]

        return [scorer if isinstance(scorer, Scorer) else cls(scorer) for scorer in scorers or []]

    def __deepcopy__(self, memo: dict[int, t.Any]) -> "Scorer[T]":
        return Scorer(
            func=self.func,
            name=self.name,
            attributes=self.attributes.copy(),
            catch=self.catch,
            step=self.step,
            auto_increment_step=self.auto_increment_step,
            log_all=self.log_all,
            bound_obj=self.bound_obj,
            config=deepcopy(self.__dn_param_config__, memo),
            context=deepcopy(self.__dn_context__, memo),
        )

    def clone(self) -> "Scorer[T]":
        """Clone the scorer."""
        return self.__deepcopy__({})

    def with_(
        self,
        *,
        name: str | None = None,
        attributes: JsonDict | None = None,
        step: int | None = None,
        auto_increment_step: bool | None = None,
        catch: bool | None = None,
        log_all: bool | None = None,
    ) -> "Scorer[T]":
        """
        Create a new Scorer with updated properties.

        Args:
            name: New name for the scorer.
            attributes: New attributes for the scorer.
            step: New step value for the scorer.
            auto_increment_step: Automatically increment the step for each time this scorer is called.
            catch: Catch exceptions in the scorer function.
            log_all: Log all sub-metrics from nested composition.

        Returns:
            A new Scorer with the updated properties
        """
        new = self.clone()
        new.name = name or self.name
        new.attributes = {**self.attributes, **(attributes or {})}
        new.func = self.func
        new.step = step if step is not None else self.step
        new.auto_increment_step = (
            auto_increment_step if auto_increment_step is not None else self.auto_increment_step
        )
        new.catch = catch if catch is not None else self.catch
        new.log_all = log_all if log_all is not None else self.log_all
        return new

    def bind(self, obj: t.Any) -> "Scorer[t.Any]":
        """
        Bind the scorer to a specific object. Any time the scorer is executed,
        the bound object will be passed instead of the caller-provided object.

        This is useful for building scoring patterns that are not directly
        tied to the output of a task.

        Examples:
            ```
            @dn.task(scorers=[
                dn.scorers.image_distance(reference).bind(dn.TaskInput("image"))
            ])
            async def classify(image: dn.Image) -> str:
                ...
            ```

        Args:
            obj: The object to bind the scorer to.

        Returns:
            A new Scorer bound to the specified object.
        """
        new = self.clone()
        new.bound_obj = obj
        return new

    def rename(self, new_name: str) -> "Scorer[T]":
        """
        Rename the scorer.

        Args:
            new_name: The new name for the scorer.

        Returns:
            A new Scorer with the updated name.
        """
        return self.with_(name=new_name)

    def adapt(
        self: "Scorer[T]",
        adapt: t.Callable[[OuterT], T],
        type: type[OuterT] = t.Any,  # type: ignore[assignment] # noqa: ARG002
        *,
        name: str | None = None,
    ) -> "Scorer[OuterT]":
        """
        Adapts a scorer to operate with some other type

        This is a powerful wrapper that allows a generic scorer (e.g., one that
        refines a string) to be used with a complex candidate object (e.g., a
        Pydantic model containing that string).

        Args:
            adapt: A function to convert from some outer type to the scorer's expected type.
            type: The outer type which is being adapted (used for type hinting - particularly with lambdas)
            name: An optional new name for the adapted scorer.

        Returns:
            A new Scorer instance that operates on the `OuterT`.
        """
        original = self

        async def evaluate(obj: OuterT, *args: t.Any, **kwargs: t.Any) -> list[Metric]:
            return await original.normalize_and_score(adapt(obj), *args, **kwargs)

        return Scorer(evaluate, name=name or self.name, wraps=original)

    async def normalize_and_score(self, obj: T, *args: t.Any, **kwargs: t.Any) -> list[Metric]:
        """
        Executes the scorer and returns all generated metrics,
        including from nested compositions.

        Args:
            obj: The object to score.

        Returns:
            All metrics generated by the scorer.
        """
        result: (
            ScorerResult
            | t.Sequence[ScorerResult]
            | t.Awaitable[ScorerResult]
            | t.Awaitable[t.Sequence[ScorerResult]]
        )

        if not isinstance(self.bound_obj, Unset):
            if isinstance(self.bound_obj, Context):
                obj = self.bound_obj.resolve()
            else:
                obj = self.bound_obj

        try:
            bound_args = self._bind_args(obj, *args, **kwargs)
            result = self.func(*bound_args.args, **bound_args.kwargs)
            if inspect.isawaitable(result):
                result = await result
        except Exception as e:
            if not self.catch:
                raise

            warn_at_user_stacklevel(
                f"Error executing scorer {self.name!r} for object {obj.__class__.__name__}: {e}",
                ScorerWarning,
            )
            result = Metric(value=0.0, step=self.step, attributes={"error": str(e)})

        if not isinstance(result, list | tuple):
            result = t.cast("list[ScorerResult]", [result])

        metrics = [
            _result
            if isinstance(_result, Metric)
            else Metric(
                float(_result),
                step=self.step,
                timestamp=datetime.now(timezone.utc),
                attributes=self.attributes,
            )
            for _result in result
        ]

        if self.auto_increment_step:
            self.step += 1

        metrics[0]._scorer_name = self.name  # type: ignore [attr-defined] # noqa: SLF001
        metrics[0]._scorer = self  # type: ignore [attr-defined] # noqa: SLF001
        metrics[0].attributes.update(self.attributes)

        # Strip any metrics from the composition stack which would
        # be confused because of name collisions. Otherwise we could
        # get duplicate metrics reported despite them coming from different
        # places in the stack.
        #
        # TODO(nick): Should look at a more explicit warning here for users.
        metrics = [
            metrics[0],
            *[m for m in metrics[1:] if getattr(m, "_scorer_name", None) != self.name],
        ]

        if not self.log_all:
            metrics = metrics[:1]  # Only return the primary metric if log_all is False

        return metrics

    async def score_composite(
        self, obj: T, *args: t.Any, **kwargs: t.Any
    ) -> tuple[Metric, list[Metric]]:
        """
        Executes the scorer and returns both the primary Metric and a list of any
        additional metrics from nested compositions.

        Args:
            obj: The object to score.

        Returns:
            A tuple of the primary Metric and a list of all metrics generated.
        """
        metrics = await self.normalize_and_score(obj, *args, **kwargs)
        return metrics[0], metrics[1:]

    async def score(self, obj: T, *args: t.Any, **kwargs: t.Any) -> Metric:
        """
        Execute the scorer and return the metric. If the scorer is a composition of other scorers,
        it will return the "highest-priority" metric, typically the first in the list.

        Any output value will be converted to a Metric object if not already one.

        Args:
            obj: The object to score.

        Returns:
            A Metric object.
        """
        all_metrics = await self.normalize_and_score(obj, *args, **kwargs)
        return all_metrics[0]

    @te.override
    async def __call__(self, obj: T, *args: t.Any, **kwargs: t.Any) -> Metric:
        return await self.score(obj, *args, **kwargs)

    def __gt__(self, value: float) -> "Scorer[T]":
        return threshold(self, gt=value)

    def __lt__(self, value: float) -> "Scorer[T]":
        return threshold(self, lt=value)

    def __ge__(self, value: float) -> "Scorer[T]":
        return threshold(self, gte=value)

    def __le__(self, value: float) -> "Scorer[T]":
        return threshold(self, lte=value)

    def __and__(self, other: "Scorer[T]") -> "Scorer[T]":
        return and_(self, other)

    def __or__(self, other: "Scorer[T]") -> "Scorer[T]":
        return or_(self, other)

    def __invert__(self) -> "Scorer[T]":
        return not_(self)  # ~ operator

    def __add__(self, other: "Scorer[T]") -> "Scorer[T]":
        return add(self, other)

    def __sub__(self, other: "Scorer[T]") -> "Scorer[T]":
        return subtract(self, other)

    def __mul__(self, weight: float) -> "Scorer[T]":
        return scale(self, weight)

    def __rmul__(self, weight: float) -> "Scorer[T]":
        return scale(self, weight)

    def __truediv__(self, weight: float) -> "Scorer[T]":
        return scale(self, 1.0 / weight)

    def __rshift__(self, name: str) -> "Scorer[T]":
        return self.with_(name=name, log_all=True)

    def __floordiv__(self, name: str) -> "Scorer[T]":
        return self.with_(name=name, log_all=False)


ScorerLike = Scorer[T] | ScorerCallable[T]
"""A Scorer instance or compatible callable."""
ScorersLike = t.Sequence[ScorerLike[T]] | t.Mapping[str, ScorerLike[T]]
"""A list of scorer-like objects or mapping or name/scorer pairs."""

# Inversion


def invert(scorer: Scorer[T], *, known_max: float = 1.0, name: str | None = None) -> Scorer[T]:
    """
    Invert the result of a scorer.

    The new score is calculated as `max_value - original_score`.

    Examples:
        ```
        @scorer
        def harmful(data: T) -> float:
            ... # 0 (safe) to 1 (harmful)

        safety = invert(harmful)
        # 0 (harmful) to 1 (safe)
        ```

    Args:
        scorer: The Scorer instance to wrap.
        known_max: The maximum value of the original score, used for inversion.
        name: Optional name for the new scorer. If None, it will be derived from the original scorer's name.
    """

    async def evaluate(data: T, *args: t.Any, **kwargs: t.Any) -> list[Metric]:
        original, others = await scorer.score_composite(data, *args, **kwargs)
        metric = Metric(max(0, known_max - original.value), step=original.step)
        return [metric, original, *others]

    return Scorer[T](evaluate, name=name or f"{scorer.name}_inverted", wraps=scorer)


# Range remapping and normalization


def remap_range(
    scorer: Scorer[T],
    *,
    known_min: float,
    known_max: float,
    new_min: float,
    new_max: float,
    name: str | None = None,
) -> Scorer[T]:
    """
    Remap the output of a scorer from one range to another.

    Examples:
        ```
        @scorer
        def harmful(data: T) -> float:
            ... # 0 (safe) to 1 (harmful)

        remapped = remap_range(
            harmful,
            known_min=0, known_max=1,
            new_min=0, new_max=100
        )
        # 0 (safe) to 100 (harmful)
        ```

    Args:
        scorer: The Scorer instance to wrap.
        known_min: The assumed minimum of the original score
        known_max: The assumed maximum of the original score.
        new_min: The minimum value of the new range.
        new_max: The maximum value of the new range.
        name: Optional name for the new scorer. If None, it will be derived from the original scorer's name.
    """
    if known_min >= known_max or new_min >= new_max:
        raise ValueError("Min values must be less than max values.")

    original_range = known_max - known_min
    new_range = new_max - new_min

    async def evaluate(data: T, *args: t.Any, **kwargs: t.Any) -> list[Metric]:
        original, others = await scorer.score_composite(data, *args, **kwargs)

        if original.value > known_max:
            warn_at_user_stacklevel(
                f"Scorer '{scorer.name}' returned {original.value}, which is greater than supplied known_max of {known_max}.",
                ScorerWarning,
            )
        elif original.value < known_min:
            warn_at_user_stacklevel(
                f"Scorer '{scorer.name}' returned {original.value}, which is less than supplied known_min of {known_min}.",
                ScorerWarning,
            )

        if original_range == 0:  # Avoid division by zero
            scaled_value = new_min
        else:
            # Normalize original score to 0-1
            normalized = (original.value - known_min) / original_range
            # Scale to new range
            scaled_value = new_min + (normalized * new_range)

        # Clamp the value to the new range to handle potential floating point errors
        final_value = max(new_min, min(new_max, scaled_value))

        metric = Metric(value=final_value, step=original.step)
        return [metric, original, *others]

    return Scorer[T](evaluate, name=name or f"{scorer.name}_remapped", wraps=scorer)


def normalize(
    scorer: Scorer[T], known_max: float, known_min: float = 0.0, *, name: str | None = None
) -> Scorer[T]:
    """
    Normalize the output of a scorer to a range of `[0.0, 1.0]`.

    Uses `remap_range` internally with `new_min = 0.0` and `new_max = 1.0`.

    Examples:
        ```
        @scorer
        def confidence(data: T) -> float:
            ... # 0 (low) to 50 (high)

        normalized = normalize(confidence, known_max=50)
        # 0 (low) to 1 (high)
        ```

    Args:
        scorer: The Scorer instance to wrap.
        known_max: The maximum value of the original score.
        known_min: The minimum value of the original score (default is 0.0).
        name: Optional name for the new scorer. If None, it will be derived from the original scorer's name.
    """
    return remap_range(
        scorer,
        known_min=known_min,
        known_max=known_max,
        new_min=0.0,
        new_max=1.0,
        name=name or f"{scorer.name}_normalized",
    )


# Binary thresholding


def threshold(
    scorer: Scorer[T],
    *,
    gt: float | None = None,
    gte: float | None = None,
    lt: float | None = None,
    lte: float | None = None,
    eq: float | None = None,
    ne: float | None = None,
    pass_value: float = 1.0,
    fail_value: float = 0.0,
    name: str | None = None,
) -> Scorer[T]:
    """
    Perform a threshold check on the output of a scorer and treat the result as a binary pass/fail.

    Examples:
        ```
        @scorer
        def confidence(data: T) -> float:
            ... # 0 (low) to 50 (high)

        strong_confidence = threshold(confidence, gte=40)
        # 0.0 (weak) and 1.0 (strong)
        ```

    Args:
        scorer: The Scorer instance to wrap.
        gt: Passes if score is greater than this value.
        gte: Passes if score is greater than or equal to this value.
        lt: Passes if score is less than this value.
        lte: Passes if score is less than or equal to this value.
        eq: Passes if score is equal to this value.
        ne: Passes if score is not equal to this value.
        pass_value: The score to return on a successful threshold check.
        fail_value: The score to return on a failed threshold check.
        name: Optional name for the new scorer. If None, it will be derived from the original scorer's name.
    """

    async def evaluate(data: T, *args: t.Any, **kwargs: t.Any) -> list[Metric]:
        original, others = await scorer.score_composite(data, *args, **kwargs)
        score = original.value

        passed = False
        if gt is not None and score > gt:
            passed = True
        if gte is not None and score >= gte:
            passed = True
        if lt is not None and score < lt:
            passed = True
        if lte is not None and score <= lte:
            passed = True
        if eq is not None and score == eq:
            passed = True
        if ne is not None and score != ne:
            passed = True

        metric = Metric(value=pass_value if passed else fail_value, step=original.step)
        return [metric, original, *others]

    operators = [
        "gt" if gt is not None else "",
        "gte" if gte is not None else "",
        "lt" if lt is not None else "",
        "lte" if lte is not None else "",
        "eq" if eq is not None else "",
        "ne" if ne is not None else "",
    ]
    operators = [op for op in operators if op]
    operator_str = ("_" + "_".join(operators)) if operators else ""

    return Scorer[T](evaluate, name=name or f"{scorer.name}{operator_str}", wraps=scorer)


# Logical combinations


def and_(scorer: Scorer[T], other: Scorer[T], *, name: str | None = None) -> Scorer[T]:
    """
    Create a scorer that performs logical AND between two scorers.

    The resulting scorer returns 1.0 if both input scorers produce truthy values
    (greater than 0), and 0.0 otherwise.

    Args:
        scorer: The first Scorer instance to combine.
        other: The second Scorer instance to combine.
        name: Optional name for the composed scorer. If None, combines the names
            of the input scorers as "scorer_name_and_other_name".

    Returns:
        A new Scorer that applies logical AND to the two input scorers.
    """

    async def evaluate(data: T, *args: t.Any, **kwargs: t.Any) -> list[Metric]:
        (original, previous), (original_other, previous_other) = await asyncio.gather(
            *[scorer.score_composite(data, *args, **kwargs), other.score_composite(data)]
        )
        passed = original.value > 0 and original_other.value > 0
        metric = Metric(float(passed), step=original.step)
        return [metric, original, original_other, *previous, *previous_other]

    return Scorer[T](evaluate, name=name or f"{scorer.name}_and_{other.name}", wraps=scorer)


def or_(scorer: Scorer[T], other: Scorer[T], *, name: str | None = None) -> Scorer[T]:
    """
    Create a scorer that performs logical OR between two scorers.

    The resulting scorer returns 1.0 if either input scorer produces a truthy value
    (greater than 0), and 0.0 only if both scorers produce falsy values (0 or negative).

    Args:
        scorer: The first Scorer instance to combine.
        other: The second Scorer instance to combine.
        name: Optional name for the composed scorer. If None, combines the names
            of the input scorers as "scorer_name_or_other_name".

    Returns:
        A new Scorer that applies logical OR to the two input scorers.
    """

    async def evaluate(data: T, *args: t.Any, **kwargs: t.Any) -> list[Metric]:
        (original, previous), (original_other, previous_other) = await asyncio.gather(
            *[scorer.score_composite(data, *args, **kwargs), other.score_composite(data)]
        )
        passed = original.value > 0 or original_other.value > 0
        metric = Metric(float(passed), step=original.step)
        return [metric, original, original_other, *previous, *previous_other]

    return Scorer[T](evaluate, name=name or f"{scorer.name}_or_{other.name}", wraps=scorer)


def not_(scorer: Scorer[T], *, name: str | None = None) -> Scorer[T]:
    """
    Apply a logical NOT operation to a scorer - inverting its truthiness (non-zero).

    Args:
        scorer: The Scorer instance to invert.
        name: Optional name for the new scorer. If None, it will be derived from the original scorer's name.
    """

    async def evaluate(data: T, *args: t.Any, **kwargs: t.Any) -> list[Metric]:
        original, others = await scorer.score_composite(data, *args, **kwargs)
        passed = original.value <= 0
        metric = Metric(float(passed), step=original.step)
        return [metric, original, *others]

    return Scorer[T](evaluate, name=name or f"not_{scorer.name}", wraps=scorer)


# Arithmetic operations


def add(
    scorer: Scorer[T], *others: Scorer[T], average: bool = False, name: str | None = None
) -> Scorer[T]:
    """
    Create a scorer that adds the values multiple scorers together.

    This composition performs arithmetic addition of the scorer values,
    with an optional averaging mode.

    Args:
        scorer: The first Scorer instance to combine.
        others: The additional Scorer instances to combine.
        average: If True, divides the sum by 2 to compute the average instead
            of the raw sum. Defaults to False.
        name: Optional name for the composed scorer. If None, combines the names
            of the input scorers as "scorer_name_add_other_name".

    Returns:
        A new Scorer that adds (or averages) the values of the two input scorers.
    """
    if len(others) == 0:
        raise ValueError("At least one other scorer must be provided for addition.")

    async def evaluate(data: T, *args: t.Any, **kwargs: t.Any) -> list[Metric]:
        (original, previous), (original_other, previous_other) = await asyncio.gather(
            *[
                scorer.score_composite(data, *args, **kwargs),
                *[other.score_composite(data) for other in others],
            ]
        )
        value = original.value + original_other.value
        metric = Metric(
            value / (len(others) + 1) if average else value,
            step=original.step,
        )
        return [metric, original, original_other, *previous, *previous_other]

    generated_name = (
        f"{scorer.name}_add_{len(others)}"
        if len(others) > 1
        else f"{scorer.name}_add_{others[0].name}"
    )
    return Scorer[T](evaluate, name=name or generated_name, wraps=scorer)


def subtract(scorer: Scorer[T], other: Scorer[T], *, name: str | None = None) -> Scorer[T]:
    """
    Create a scorer that subtracts one scorer's value from another's.

    This composition performs arithmetic subtraction (scorer - other), which can be
    useful for penalty systems, relative scoring, or creating difference metrics.

    Args:
        scorer: The Scorer instance to subtract from (minuend).
        other: The Scorer instance to subtract (subtrahend).
        name: Optional name for the composed scorer. If None, combines the names
            of the input scorers as "scorer_name_sub_other_name".

    Returns:
        A new Scorer that subtracts the second scorer's value from the first.
    """

    async def evaluate(data: T, *args: t.Any, **kwargs: t.Any) -> list[Metric]:
        (original, previous), (original_other, previous_other) = await asyncio.gather(
            *[scorer.score_composite(data, *args, **kwargs), other.score_composite(data)]
        )
        value = original.value - original_other.value
        metric = Metric(value, step=original.step)
        return [metric, original, original_other, *previous, *previous_other]

    return Scorer[T](evaluate, name=name or f"{scorer.name}_sub_{other.name}", wraps=scorer)


def avg(scorer: Scorer[T], *others: Scorer[T], name: str | None = None) -> Scorer[T]:
    """
    Average multiple scorers together.

    This is a convenience function that uses the `add` function with `average=True`.

    Args:
        scorer: The Scorer instance.
        others: Additional Scorer instances to include in the average.
        name: Optional name for the new scorer. If None, it will be derived from the original scorers' names.
    """
    return add(scorer, *others, average=True, name=name or f"{scorer.name}_avg")


def weighted_avg(*scorers: tuple[Scorer[T], float], name: str | None = None) -> Scorer[T]:
    """
    Create a scorer that computes a weighted average of multiple scorers.

    This composition allows for sophisticated scoring schemes where different
    metrics have different importance levels. The final score is calculated as
    the sum of (score * weight) for each scorer, divided by the total weight.

    Examples:
        ```
        # Safety is most important, then accuracy, then speed
        composite = weighted_avg(
            (safety, 1.0),
            (accuracy, 0.7),
            (speed, 0.3)
        )
        # (safety * 1.0 + accuracy * 0.7 + speed * 0.3) / 2.0
        ```

    Args:
        *scorers: Variable number of (Scorer, weight) tuples. Each tuple contains
            a Scorer instance and its corresponding weight (float). At least one
            scorer must be provided.
        name: Optional name for the composed scorer. Defaults to "weighted_avg".
    """

    if not scorers:
        raise ValueError("At least one scorer must be provided.")

    async def evaluate(data: T) -> list[Metric]:
        total_weight = sum(weight for _, weight in scorers)
        weighted_sum = 0.0
        all_metrics: list[Metric] = []

        for scorer, weight in scorers:
            original, previous = await scorer.score_composite(data)
            weighted_sum += original.value * weight
            all_metrics.append(original)
            all_metrics.extend(previous)

        weighted_avg_value = weighted_sum / total_weight if total_weight > 0 else 0.0
        metric = Metric(weighted_avg_value, step=max(m.step for m in all_metrics))
        return [metric, *all_metrics]

    return Scorer[T](evaluate, name=name or "weighted_avg")


def scale(scorer: Scorer[T], factor: float, *, name: str | None = None) -> Scorer[T]:
    """
    Create a scorer that scales the output of another scorer by a constant factor.

    This composition multiplies the scorer's output by the specified factor,
    which is useful for adjusting score ranges, applying importance weights,
    or inverting scores (with negative factors). The original metric is
    preserved alongside the scaled result.

    Args:
        scorer: The Scorer instance to scale.
        factor: The multiplier to apply to the scorer's output. Can be positive,
            negative, or fractional.
        name: Optional name for the scaled scorer. If None, derives the name
            from the original scorer as "scorer_name_scaled".

    Returns:
        A new Scorer that returns the scaled value of the input scorer.
    """

    async def evaluate(data: T, *args: t.Any, **kwargs: t.Any) -> list[Metric]:
        original, others = await scorer.score_composite(data, *args, **kwargs)
        metric = Metric(original.value * factor, step=original.step)
        return [metric, original, *others]

    return Scorer[T](evaluate, name=name or f"{scorer.name}_scaled", wraps=scorer)


def clip(
    scorer: Scorer[T],
    min_val: float,
    max_val: float,
    *,
    name: str | None = None,
) -> Scorer[T]:
    """
    Create a scorer that clips the output of another scorer to a specified range.

    This composition constrains the scorer's output to lie within [min_val, max_val],
    clamping values that exceed the bounds. This is useful for ensuring scores
    remain within expected ranges, preventing outliers from skewing results,
    or enforcing score normalization bounds.

    Args:
        scorer: The Scorer instance to clip.
        min_val: The minimum value to clip to. Values below this will be set to min_val.
        max_val: The maximum value to clip to. Values above this will be set to max_val.
        name: Optional name for the clipped scorer. If None, derives the name
            from the original scorer as "scorer_name_clipped".

    Returns:
        A new Scorer that returns the clipped value of the input scorer.
    """

    async def evaluate(data: T, *args: t.Any, **kwargs: t.Any) -> list[Metric]:
        original, others = await scorer.score_composite(data, *args, **kwargs)
        clipped_value = max(min_val, min(max_val, original.value))
        metric = Metric(clipped_value, step=original.step)
        return [metric, original, *others]

    return Scorer[T](evaluate, name=name or f"{scorer.name}_clipped", wraps=scorer)


# Core Scorers
#
# TODO(nick): Lots of odd overlap here between intents, would
# like to come back and do a full pass on these.


def equals(reference: t.Any, *, name: str = "equals") -> Scorer[t.Any]:
    """
    Create a scorer that checks for equality between the object and a reference value.

    Returns a 1.0 if they are equal, and 0.0 otherwise.

    Args:
        reference: The value to compare against.
        name: Optional name for the equality scorer. If None, derives the name
            from the reference value.
    """

    async def evaluate(data: t.Any, *, reference: t.Any = reference) -> Metric:
        return Metric(1.0 if data == reference else 0.0)

    return Scorer[t.Any](evaluate, name=name)


def forward(value: t.Any, *, name: str = "forward") -> Scorer[t.Any]:
    """
    Create a scorer that forwards a known value as the score.

    This is useful for patterns where you want to fix a score value,
    or use some portion of the task input/output as the score.

    Examples:
        ```
        # Always return a score of 0.75
        fixed = forward(0.75)

        # Use the length of the input text as the score
        length_scorer = forward(dn.TaskInput("text").adapt(len))
        ```

    Args:
        value: The value to forward.
        name: Optional name for the forward scorer. If None, derives the name
            from the value.
    """

    async def evaluate(data: t.Any, *, value: float = value) -> ScorerResult:  # noqa: ARG001
        return value

    return Scorer[t.Any](evaluate, name=name or f"forward_{value}")


def task_output(
    adapt: t.Callable[[t.Any], float] | None = None, *, name: str = "task_output"
) -> Scorer[t.Any]:
    """
    Create a scorer that forwards from the output of a task with an optional adapter.

    This is useful when you want to use (and process) the output of a task
    as the score value.

    Examples:
        ```
        @dn.task(scorers=[
            dn.scorers.task_output(lambda output: len(output) / 100)  # Score based on length of output
        ])
        async def summarize(text: str) -> str:
            ...
        ```

    Args:
        adapt: An optional function to adapt the task output to a float score.
        name: Optional name for the scorer. If None, defaults to "task_output".
    """
    context = TaskOutput()
    if adapt is not None:
        context.adapt(adapt)
    return forward(context, name=name)


def task_input(
    input_name: str, adapt: t.Callable[[t.Any], float] | None = None, *, name: str = "task_input"
) -> Scorer[t.Any]:
    """
    Create a scorer that forwards from a named input to a task with an optional adapter.

    This is useful when you want to use (and process) one of the inputs
    to a task as the score value.

    Examples:
        ```
        @dn.task(scorers=[
            dn.scorers.task_input("text", lambda text: len(text) / 100)  # Score based on length of input text
        ])
        async def summarize(text: str) -> str:
            ...
        ```

    Args:
        input_name: The name of the task input to use as the score.
        adapt: An optional function to adapt the task input to a float score.
    """
    context = TaskInput(input_name)
    if adapt is not None:
        context.adapt(adapt)
    return forward(context, name=name)

import asyncio
import inspect
import typing as t
from dataclasses import dataclass
from datetime import datetime, timezone

from logfire._internal.stack_info import warn_at_user_stacklevel
from logfire._internal.utils import safe_repr

from dreadnode.configurable import clone_config_attrs
from dreadnode.metric import Metric
from dreadnode.types import JsonDict

T = t.TypeVar("T")


class ScorerWarning(UserWarning):
    pass


ScorerResult = float | int | bool | Metric
"""The result of a scorer function, which can be a numeric value or a Metric object."""
ScorerCallable = (
    t.Callable[[T], t.Awaitable[ScorerResult]]
    | t.Callable[[T], ScorerResult]
    | t.Callable[[T], t.Awaitable[t.Sequence[ScorerResult]]]
    | t.Callable[[T], t.Sequence[ScorerResult]]
)
"""A callable that takes an object of type T and returns a ScorerResult or a sequence of ScorerResults."""
ScorerLike = t.Union["Scorer[T]", ScorerCallable[T]]
ScorersLike = t.Sequence[ScorerLike[T]] | dict[str, ScorerLike[T]]


@dataclass
class Scorer(t.Generic[T]):
    name: str
    "The name of the scorer, used for reporting metrics."
    func: ScorerCallable[T]
    "The function to call to get the metric."
    attributes: JsonDict
    "A dictionary of attributes for metrics produced by this Scorer."
    step: int = 0
    "The step value to attach to metrics produced by this Scorer."
    auto_increment_step: bool = False
    "Whether to automatically increment the step for each time this scorer is called."
    catch: bool = False
    "Whether to catch exceptions in the scorer function and return a 0 Metric with error information."
    log_all: bool = False
    "Whether to log all sub-metrics from nested composition, or just the final resulting metric."

    @classmethod
    def from_callable(
        cls,
        func: "ScorerCallable[T] | Scorer[T]",
        *,
        name: str | None = None,
        attributes: JsonDict | None = None,
        catch: bool = False,
        auto_increment_step: bool = False,
        log_all: bool = False,
    ) -> "Scorer[T]":
        """
        Create a scorer from a callable function.

        Args:
            func: The function to call to get the metric.
            name: The name of the scorer, used for reporting metrics.
            attributes: A dictionary of attributes to attach to the metric.
            catch: Whether to catch exceptions in the scorer function and return a 0 Metric with error information.
            auto_increment_step: Whether to automatically increment the step for each time this scorer is called.
            log_all: Whether to log all sub-metrics from nested composition, or just the final resulting metric.

        Returns:
            A Scorer object.
        """
        if isinstance(func, Scorer):
            return func

        # if isinstance(func, Task):
        #     raise TypeError(
        #         f"Cannot create a Scorer from a @dn.task object ('{func.name}'). "
        #         "Scorer functions should be simple, undecorated callables. "
        #         "If you need to configure your scorer, create a factory function that returns a Scorer object."
        #     )

        # if inspect.iscoroutine(func):
        #     raise TypeError(
        #         "Received a coroutine when creating a Scorer. This can happen if you apply "
        #         "@dn.task to a scorer factory function. Please remove the @dn.task decorator "
        #         "from your scorer factory."
        #     )

        unwrapped = inspect.unwrap(func)
        func_name = getattr(
            unwrapped,
            "__qualname__",
            getattr(func, "__name__", safe_repr(unwrapped)),
        )
        name = name or func_name
        return clone_config_attrs(
            func,
            cls(
                name=name,
                func=func,
                catch=catch,
                auto_increment_step=auto_increment_step,
                log_all=log_all,
                attributes=attributes or {},
            ),
        )

    @classmethod
    def fit_like(
        cls, scorers: ScorersLike[T] | None, *, attributes: JsonDict | None = None
    ) -> list["Scorer[T]"]:
        if isinstance(scorers, dict):
            return [
                scorer.with_(name=name, attributes=attributes)
                if isinstance(scorer, Scorer)
                else cls.from_callable(scorer, name=name, attributes=attributes)
                for name, scorer in scorers.items()
            ]

        return [
            scorer.with_(attributes=attributes)
            if isinstance(scorer, Scorer)
            else cls.from_callable(scorer, attributes=attributes)
            for scorer in scorers or []
        ]

    def __post_init__(self) -> None:
        self.__signature__ = inspect.signature(self.func)
        self.__name__ = self.name

    def clone(self) -> "Scorer[T]":
        """
        Clone the scorer.

        Returns:
            A new Scorer.
        """
        return clone_config_attrs(
            self,
            Scorer(
                name=self.name,
                attributes=self.attributes,
                func=self.func,
                step=self.step,
                auto_increment_step=self.auto_increment_step,
                log_all=self.log_all,
                catch=self.catch,
            ),
        )

    def with_(
        self,
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
            auto_increment_step: Whether to auto-increment the step.
            catch: Whether to catch exceptions in the scorer function.
            log_all: Whether to log all sub-metrics from nested composition.

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

    def rename(self, new_name: str) -> "Scorer[T]":
        """
        Rename the scorer.

        Args:
            new_name: The new name for the scorer.

        Returns:
            A new Scorer with the updated name.
        """
        return self.with_(name=new_name)

    async def normalize_and_score(self, object: T) -> list[Metric]:
        """
        Executes the scorer and returns all generated metrics,
        including from nested compositions.

        Args:
            object: The object to score.

        Returns:
            All metrics generated by the scorer.
        """
        result: (
            ScorerResult
            | t.Sequence[ScorerResult]
            | t.Awaitable[ScorerResult]
            | t.Awaitable[t.Sequence[ScorerResult]]
        )

        try:
            result = self.func(object)
            if inspect.isawaitable(result):
                result = await result
        except Exception as e:
            if not self.catch:
                raise

            warn_at_user_stacklevel(
                f"Error executing scorer {self.name!r} for object {object!r}: {e}",
                ScorerWarning,
            )
            result = Metric(value=0.0, step=self.step, attributes={"error": str(e)})

        if not isinstance(result, (list, tuple)):
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

        for metric in metrics:
            # Add an origin in case this metric gets rolled up in composition.
            if not hasattr(metric, "_scorer_name"):
                metric._scorer_name = self.name  # type: ignore [attr-defined] # noqa: SLF001
            if not hasattr(metric, "_scorer"):
                metric._scorer = self  # type: ignore [attr-defined] # noqa: SLF001

            # Update our attributes
            metric.attributes.update(self.attributes)

        if not self.log_all:
            metrics = metrics[:1]  # Only return the primary metric if log_all is False

        return metrics

    async def score_composite(self, object: T) -> tuple[Metric, list[Metric]]:
        """
        Executes the scorer and returns both the primary Metric and a list of any
        additional metrics from nested compositions.

        Args:
            object: The object to score.

        Returns:
            A tuple of the primary Metric and a list of all metrics generated.
        """
        metrics = await self.normalize_and_score(object)
        return metrics[0], metrics[1:]

    async def score(self, obj: T) -> Metric:
        """
        Execute the scorer and return the metric. If the scorer is a composition of other scorers,
        it will return the "highest-priority" metric, typically the first in the list.

        Any output value will be converted to a Metric object if not already one.

        Args:
            obj: The object to score.

        Returns:
            A Metric object.
        """
        all_metrics = await self.normalize_and_score(obj)
        return all_metrics[0]

    async def __call__(self, object: T) -> Metric:
        """
        Execute the scorer and return the metric. If the scorer is a composition of other scorers,
        it will return the "highest-priority" metric, typically the first in the list.

        Any output value will be converted to a Metric object if not already one.

        Args:
            object: The object to score.

        Returns:
            A Metric object.
        """
        return await self.score(object)

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
        return self.with_(name=name, log_all=False)

    def __floordiv__(self, name: str) -> "Scorer[T]":
        return self.with_(name=name, log_all=True)


def named(name: str, scorer: Scorer[T]) -> Scorer[T]:
    """
    Give a scorer a name.

    Args:
        name: The name to assign to the scorer.
        scorer: The Scorer instance to rename.

    Returns:
        A new Scorer with the updated name.
    """
    return scorer.rename(name)


# Inversion


def invert(scorer: Scorer[T], *, known_max: float = 1.0, name: str | None = None) -> Scorer[T]:
    """
    Invert the result of a scorer.

    The new score is calculated as `max_value - original_score`.

    Args:
        scorer: The Scorer instance to wrap.
        known_max: The maximum value of the original score, used for inversion.
        name: Optional name for the new scorer. If None, it will be derived from the original scorer's name.
    """

    async def evaluate(data: t.Any) -> list[Metric]:
        original, others = await scorer.score_composite(data)
        metric = Metric(max(0, known_max - original.value), step=original.step)
        return [metric, original, *others]

    return Scorer[T].from_callable(evaluate, name=name or f"{scorer.name}_inverted")


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

    async def evaluate(data: t.Any) -> list[Metric]:
        original, others = await scorer.score_composite(data)

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

    return Scorer[T].from_callable(evaluate, name=name or f"{scorer.name}_remapped")


def normalize(
    scorer: Scorer[T], known_max: float, known_min: float = 0.0, *, name: str | None = None
) -> Scorer[T]:
    """
    Normalize the output of a scorer to a range of [0.0, 1.0].

    Uses `remap_range` internally.

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

    async def evaluate(data: T) -> list[Metric]:
        original, others = await scorer.score_composite(data)
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

    return Scorer[T].from_callable(evaluate, name=name or f"{scorer.name}{operator_str}")


# Logical combinations


def and_(scorer: Scorer[T], other: Scorer[T], *, name: str | None = None) -> Scorer[T]:
    """
    Apply a logical AND operation between two scorers - testing their values as truthy (non-zero).

    Args:
        scorer: The first Scorer instance.
        other: The second Scorer instance.
        name: Optional name for the new scorer. If None, it will be derived from the original scorers' names.
    """

    async def evaluate(data: T) -> list[Metric]:
        (original, previous), (original_other, previous_other) = await asyncio.gather(
            *[scorer.score_composite(data), other.score_composite(data)]
        )
        passed = original.value > 0 and original_other.value > 0
        metric = Metric(float(passed), step=original.step)
        return [metric, original, original_other, *previous, *previous_other]

    return Scorer[T].from_callable(evaluate, name=name or f"{scorer.name}_and_{other.name}")


def or_(scorer: Scorer[T], other: Scorer[T], *, name: str | None = None) -> Scorer[T]:
    """
    Apply a logical OR operation between two scorers - testing their values as truthy (non-zero).

    Args:
        scorer: The first Scorer instance.
        other: The second Scorer instance.
        name: Optional name for the new scorer. If None, it will be derived from the original scorers' names.
    """

    async def evaluate(data: T) -> list[Metric]:
        (original, previous), (original_other, previous_other) = await asyncio.gather(
            *[scorer.score_composite(data), other.score_composite(data)]
        )
        passed = original.value > 0 or original_other.value > 0
        metric = Metric(float(passed), step=original.step)
        return [metric, original, original_other, *previous, *previous_other]

    return Scorer[T].from_callable(evaluate, name=name or f"{scorer.name}_or_{other.name}")


def not_(scorer: Scorer[T], *, name: str | None = None) -> Scorer[T]:
    """
    Apply a logical NOT operation to a scorer - inverting its truthiness (non-zero).

    Args:
        scorer: The Scorer instance to invert.
        name: Optional name for the new scorer. If None, it will be derived from the original scorer's name.
    """

    async def evaluate(data: T) -> list[Metric]:
        original, others = await scorer.score_composite(data)
        passed = original.value <= 0
        metric = Metric(float(passed), step=original.step)
        return [metric, original, *others]

    return Scorer[T].from_callable(evaluate, name=name or f"not_{scorer.name}")


# Arithmetic operations


def add(
    scorer: Scorer[T], other: Scorer[T], *, average: bool = False, name: str | None = None
) -> Scorer[T]:
    """
    Add two scorers together.

    Args:
        scorer: The first Scorer instance.
        other: The second Scorer instance.
        average: If True, the average of the two scores will be divided by 2.
        name: Optional name for the new scorer. If None, it will be derived from the original scorers' names.
    """

    async def evaluate(data: T) -> list[Metric]:
        (original, previous), (original_other, previous_other) = await asyncio.gather(
            *[scorer.score_composite(data), other.score_composite(data)]
        )
        value = original.value + original_other.value
        metric = Metric(
            value / 2 if average else value,
            step=original.step,
        )
        return [metric, original, original_other, *previous, *previous_other]

    return Scorer[T].from_callable(evaluate, name=name or f"{scorer.name}_add_{other.name}")


def subtract(scorer: Scorer[T], other: Scorer[T], *, name: str | None = None) -> Scorer[T]:
    """
    Subtract one scorer from another.

    Args:
        scorer: The first Scorer instance.
        other: The second Scorer instance.
        name: Optional name for the new scorer. If None, it will be derived from the original scorers' names.
    """

    async def evaluate(data: T) -> list[Metric]:
        (original, previous), (original_other, previous_other) = await asyncio.gather(
            *[scorer.score_composite(data), other.score_composite(data)]
        )
        value = original.value - original_other.value
        metric = Metric(value, step=original.step)
        return [metric, original, original_other, *previous, *previous_other]

    return Scorer[T].from_callable(evaluate, name=name or f"{scorer.name}_sub_{other.name}")


def avg(scorer: Scorer[T], other: Scorer[T], *, name: str | None = None) -> Scorer[T]:
    """
    Average two scorers together.

    This is a convenience function that uses the `add` function with `average=True`.

    Args:
        scorer: The first Scorer instance.
        other: The second Scorer instance.
        name: Optional name for the new scorer. If None, it will be derived from the original scorers' names.
    """
    return add(scorer, other, average=True, name=name or f"{scorer.name}_{other.name}_avg")


def weighted_avg(*scorers: tuple[Scorer[T], float], name: str | None = None) -> Scorer[T]:
    """
    Combine multiple scorers with specified weights.

    Args:
        *scorers: A variable number of tuples, each containing a Scorer and its weight.
        name: Optional name for the new scorer. If None, it will be derived from the names of the scorers.

    Returns:
        A new Scorer that combines the weighted scores of the input scorers.
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

    return Scorer[T].from_callable(evaluate, name=name or "weighted_avg")


def scale(scorer: Scorer[T], factor: float, *, name: str | None = None) -> Scorer[T]:
    """
    Scale the output of a scorer by some factor.

    Args:
        scorer: The Scorer instance to wrap.
        factor: The factor to scale the score by.
        name: Optional name for the new scorer. If None, it will be derived from the original scorer's name.
    """

    async def evaluate(data: T) -> list[Metric]:
        original, others = await scorer.score_composite(data)
        metric = Metric(original.value * factor, step=original.step)
        return [metric, original, *others]

    return Scorer[T].from_callable(evaluate, name=name or f"{scorer.name}_scaled")


def clip(
    scorer: Scorer[T],
    min_val: float,
    max_val: float,
    *,
    name: str | None = None,
) -> Scorer[T]:
    """
    Clip the result of a scorer to a specified range.

    Args:
        scorer: The Scorer instance to wrap.
        min_val: The minimum value to clip to.
        max_val: The maximum value to clip to.
        name: Optional name for the new scorer. If None, it will be derived from the original scorer's name.
    """

    async def evaluate(data: T) -> list[Metric]:
        original, others = await scorer.score_composite(data)
        clipped_value = max(min_val, min(max_val, original.value))
        metric = Metric(clipped_value, step=original.step)
        return [metric, original, *others]

    return Scorer[T].from_callable(evaluate, name=name or f"{scorer.name}_clipped")

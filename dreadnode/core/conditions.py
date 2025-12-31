import inspect
import typing as t
from copy import deepcopy

import typing_extensions as te

from dreadnode.core.exceptions import warn_at_user_stacklevel
from dreadnode.core.meta import Component, ConfigInfo, Context
from dreadnode.core.scorer import Scorer

if t.TYPE_CHECKING:
    from dreadnode.core.agents.events import AgentError, AgentEvent, AgentStep, ToolEnd, ToolStart
    from dreadnode.core.generators.models import XMLModel

AgentEventT = te.TypeVar("AgentEventT", bound="AgentEvent", default="AgentEvent")
AgentEventT_contra = te.TypeVar(
    "AgentEventT_contra", bound="AgentEvent", contravariant=True, default="AgentEvent"
)

# Type alias for events that have tool_call
ToolEvent = t.Union["ToolStart", "ToolEnd"]


class ConditionWarning(UserWarning):
    """Warning issued for non-critical issues during condition evaluation."""


@t.runtime_checkable
class ConditionCallable(t.Protocol, t.Generic[AgentEventT_contra]):
    """A callable that takes an event and returns a boolean."""

    def __call__(
        self, event: AgentEventT_contra, /, *args: t.Any, **kwargs: t.Any
    ) -> t.Awaitable[bool] | bool: ...


class Condition(Component[te.Concatenate[AgentEventT, ...], bool], t.Generic[AgentEventT]):
    """Represents a condition that evaluates an agent event and returns a boolean."""

    def __init__(
        self,
        func: ConditionCallable[AgentEventT],
        *,
        name: str | None = None,
        catch: bool = False,
        default: bool = False,
        config: dict[str, ConfigInfo] | None = None,
        context: dict[str, Context] | None = None,
        wraps: t.Callable[..., t.Any] | None = None,
    ):
        super().__init__(
            t.cast("t.Callable[[AgentEventT], bool]", func),
            name=name,
            config=config,
            context=context,
            wraps=wraps,
        )

        self.name = self.name
        "The name of the condition, used for reporting and logging."
        self.catch = catch
        """
        If True, catches exceptions during evaluation and returns the default value.
        If False, exceptions are raised.
        """
        self.default = default
        """The default value to return if an exception is caught."""

    def __repr__(self) -> str:
        return f"Condition(name='{self.name}', catch={self.catch}, default={self.default})"

    @classmethod
    def fit(cls, condition: "ConditionLike[AgentEventT]") -> "Condition[AgentEventT]":
        """Ensures that the provided condition is a Condition instance."""
        if isinstance(condition, Condition):
            return condition
        if callable(condition):
            return Condition(condition)
        raise TypeError("Condition must be a Condition instance or a callable.")

    @classmethod
    def fit_many(
        cls, conditions: "ConditionsLike[AgentEventT] | None"
    ) -> list["Condition[AgentEventT]"]:
        """
        Convert a collection of condition-like objects into a list of Condition instances.

        Args:
            conditions: A collection of condition-like objects. Can be:
                - A dictionary mapping names to condition objects or callables
                - A sequence of condition objects or callables
                - None (returns empty list)

        Returns:
            A list of Condition instances with consistent configuration.
        """
        if isinstance(conditions, t.Mapping):
            return [
                condition.with_(name=name)
                if isinstance(condition, Condition)
                else cls(condition, name=name)
                for name, condition in conditions.items()
            ]

        return [
            condition if isinstance(condition, Condition) else cls(condition)
            for condition in conditions or []
        ]

    def __deepcopy__(self, memo: dict[int, t.Any]) -> "Condition[AgentEventT]":
        return Condition(
            func=self.func,
            name=self.name,
            catch=self.catch,
            default=self.default,
            config=deepcopy(self.__dn_param_config__, memo),
            context=deepcopy(self.__dn_context__, memo),
        )

    def clone(self) -> "Condition[AgentEventT]":
        """Clone the condition."""
        return self.__deepcopy__({})

    def with_(
        self,
        *,
        name: str | None = None,
        catch: bool | None = None,
        default: bool | None = None,
    ) -> "Condition[AgentEventT]":
        """
        Create a new Condition with updated properties.

        Args:
            name: New name for the condition.
            catch: Catch exceptions in the condition function.
            default: Default value to return on exception.

        Returns:
            A new Condition with the updated properties.
        """
        new = self.clone()
        new.name = name or self.name
        new.catch = catch if catch is not None else self.catch
        new.default = default if default is not None else self.default
        return new

    def rename(self, new_name: str) -> "Condition[AgentEventT]":
        """
        Rename the condition.

        Args:
            new_name: The new name for the condition.

        Returns:
            A new Condition with the updated name.
        """
        return self.with_(name=new_name)

    async def evaluate(self, event: AgentEventT, *args: t.Any, **kwargs: t.Any) -> bool:
        """
        Evaluate the condition for a given event.

        Args:
            event: The event to evaluate.

        Returns:
            True if the condition is met, False otherwise.
        """
        try:
            bound_args = self._bind_args(event, *args, **kwargs)
            result = t.cast(
                "bool | t.Awaitable[bool]", self.func(*bound_args.args, **bound_args.kwargs)
            )
            if inspect.isawaitable(result):
                result = await result

        except Exception as e:
            if not self.catch:
                raise

            warn_at_user_stacklevel(
                f"Error evaluating condition {self.name!r} for event {event!r}: {e}",
                ConditionWarning,
            )
            return self.default

        return result

    @te.override
    async def __call__(self, event: AgentEventT, *args: t.Any, **kwargs: t.Any) -> bool:
        return await self.evaluate(event, *args, **kwargs)

    def __and__(self, other: "ConditionLike[AgentEventT]") -> "Condition[AgentEventT]":
        return and_(self, Condition.fit(other))

    def __or__(self, other: "ConditionLike[AgentEventT]") -> "Condition[AgentEventT]":
        return or_(self, Condition.fit(other))

    def __invert__(self) -> "Condition[AgentEventT]":
        return not_(self)


ConditionLike = Condition[AgentEventT] | ConditionCallable[AgentEventT]
"""A condition or compatible callable."""
ConditionsLike = t.Sequence[ConditionLike[AgentEventT]] | t.Mapping[str, ConditionLike[AgentEventT]]
"""A sequence of condition-like objects or mapping of name/condition pairs."""

# Backwards compatibility
EventCondition = ConditionLike


# =============================================================================
# Conversion Functions
# =============================================================================


def from_scorer(
    scorer: Scorer[AgentEventT],
    *,
    threshold: float = 0,
    name: str | None = None,
) -> Condition[AgentEventT]:
    """
    Create a Condition from a Scorer.

    The condition evaluates to True if the scorer's value exceeds the threshold.

    Args:
        scorer: The Scorer to convert.
        threshold: The minimum value for the condition to be True.
        name: Optional name for the condition.

    Returns:
        A Condition that is True if the scorer's value > threshold.

    Example:
        ```python
        quality_scorer = Scorer(lambda e: len(e.content) / 100, name="quality")

        # Condition that passes if quality > 0.5
        high_quality = from_scorer(quality_scorer, threshold=0.5)

        @hook(GenerationEnd, condition=high_quality)
        async def on_high_quality(self, event: GenerationEnd) -> Reaction | None:
            ...
        ```
    """

    async def evaluate(event: AgentEventT) -> bool:
        metric = await scorer.score(event)
        return metric.value > threshold

    return Condition(evaluate, name=name or f"scorer:{scorer.name}(>{threshold})")


# =============================================================================
# Logical Combinators
# =============================================================================


def and_(
    condition: Condition[AgentEventT],
    other: Condition[AgentEventT],
    *,
    name: str | None = None,
) -> Condition[AgentEventT]:
    """
    Combine two conditions with AND logic.

    Args:
        condition: The first condition.
        other: The second condition.
        name: Optional name for the combined condition.

    Returns:
        A new Condition that is True only if both conditions are True.
    """

    async def evaluate(event: AgentEventT, *args: t.Any, **kwargs: t.Any) -> bool:
        return await condition.evaluate(event, *args, **kwargs) and await other.evaluate(
            event, *args, **kwargs
        )

    return Condition(evaluate, name=name or f"({condition.name} & {other.name})", wraps=condition)


def or_(
    condition: Condition[AgentEventT],
    other: Condition[AgentEventT],
    *,
    name: str | None = None,
) -> Condition[AgentEventT]:
    """
    Combine two conditions with OR logic.

    Args:
        condition: The first condition.
        other: The second condition.
        name: Optional name for the combined condition.

    Returns:
        A new Condition that is True if either condition is True.
    """

    async def evaluate(event: AgentEventT, *args: t.Any, **kwargs: t.Any) -> bool:
        return await condition.evaluate(event, *args, **kwargs) or await other.evaluate(
            event, *args, **kwargs
        )

    return Condition(evaluate, name=name or f"({condition.name} | {other.name})", wraps=condition)


def not_(
    condition: Condition[AgentEventT],
    *,
    name: str | None = None,
) -> Condition[AgentEventT]:
    """
    Negate a condition.

    Args:
        condition: The condition to negate.
        name: Optional name for the negated condition.

    Returns:
        A new Condition that is True when the original is False.
    """

    async def evaluate(event: AgentEventT, *args: t.Any, **kwargs: t.Any) -> bool:
        return not await condition.evaluate(event, *args, **kwargs)

    return Condition(evaluate, name=name or f"~{condition.name}", wraps=condition)


# =============================================================================
# Core Condition Factories
# =============================================================================


def always(*, name: str = "always") -> Condition[AgentEventT]:
    """
    A condition that always returns True.

    Args:
        name: Optional name for the condition.

    Returns:
        A Condition that always evaluates to True.
    """

    def evaluate(event: AgentEventT) -> bool:  # noqa: ARG001
        return True

    return Condition(evaluate, name=name)


def never(*, name: str = "never") -> Condition[AgentEventT]:
    """
    A condition that always returns False.

    Args:
        name: Optional name for the condition.

    Returns:
        A Condition that always evaluates to False.
    """

    def evaluate(event: AgentEventT) -> bool:  # noqa: ARG001
        return False

    return Condition(evaluate, name=name)


def has_attr(attr: str, *, name: str | None = None) -> Condition[AgentEventT]:
    """
    Check if the event has the specified attribute.

    Args:
        attr: The attribute name to check for.
        name: Optional name for the condition.

    Returns:
        A Condition that is True if the event has the attribute.
    """

    def evaluate(event: AgentEventT, *, attr: str = attr) -> bool:
        return hasattr(event, attr)

    return Condition(evaluate, name=name or f"has_{attr}")


def attr_equals(attr: str, value: t.Any, *, name: str | None = None) -> Condition[AgentEventT]:
    """
    Check if the event attribute equals the specified value.

    Args:
        attr: The attribute name to check.
        value: The value to compare against.
        name: Optional name for the condition.

    Returns:
        A Condition that is True if the attribute equals the value.
    """

    def evaluate(event: AgentEventT, *, attr: str = attr, value: t.Any = value) -> bool:
        return getattr(event, attr, None) == value

    return Condition(evaluate, name=name or f"{attr}_equals")


# =============================================================================
# Event-Specific Condition Factories
# =============================================================================


def is_file_not_found_error(*, name: str = "is_file_not_found") -> "Condition[AgentError]":
    """
    Returns a Condition that checks if the agent's error is a FileNotFoundError.

    Args:
        name: Optional name for the condition.

    Returns:
        A Condition that is True if the error is a FileNotFoundError.
    """
    from dreadnode.core.agents.trajectory import AgentError

    def evaluate(event: AgentError) -> bool:
        return isinstance(event.error, FileNotFoundError)

    return Condition(evaluate, name=name)


def tool_called(
    tool_name: str | t.Collection[str], *, name: str | None = None
) -> Condition[ToolEvent]:
    """
    Returns a Condition that checks if the event is for a specific tool.

    Args:
        tool_name: A single tool name or a collection of tool names to match.
        name: Optional name for the condition.

    Returns:
        A Condition that is True if the tool matches.
    """
    names = {tool_name} if isinstance(tool_name, str) else set(tool_name)

    def evaluate(event: ToolEvent) -> bool:
        return event.tool_call.name in names

    return Condition(evaluate, name=name or f"tool_is_{'_or_'.join(sorted(names))}")


def error_contains(
    text: str, *, case_sensitive: bool = False, name: str | None = None
) -> "Condition[AgentError]":
    """
    Returns a Condition that checks if the event's error message contains specific text.

    Args:
        text: The text to search for in the error message.
        case_sensitive: Whether the search should be case-sensitive.
        name: Optional name for the condition.

    Returns:
        A Condition that is True if the error message contains the text.
    """
    from dreadnode.core.agents.trajectory import AgentError

    def evaluate(event: AgentError) -> bool:
        error_str = str(event.error)
        if case_sensitive:
            return text in error_str
        return text.lower() in error_str.lower()

    return Condition(evaluate, name=name or f"error_contains_{text.replace(' ', '_').lower()}")


# =============================================================================
# Fluent Builder Pattern for Model-Based Conditions
# =============================================================================


class FieldConditionBuilder(t.Generic[AgentEventT]):
    """Represents a specific field of a model, ready for comparison."""

    def __init__(self, model_type: type, field_name: str, parser: "t.Any"):
        self.model_type = model_type
        self.field_name = field_name
        self.parser = parser

    def __hash__(self) -> int:
        return hash((self.model_type, self.field_name))

    def equals(self, value: object, *, name: str | None = None) -> Condition[AgentEventT]:
        """
        Creates a Condition for an equality check.

        Args:
            value: The value to compare against.
            name: Optional name for the condition.

        Returns:
            A Condition that is True if the field equals the value.
        """
        model_type = self.model_type
        field_name = self.field_name
        parser = self.parser

        def evaluate(event: AgentEventT) -> bool:
            model = parser.try_parse(event, model_type)
            if model is None:
                return False
            return getattr(model, field_name, None) == value

        return Condition(evaluate, name=name or f"{field_name}_equals_{value}")

    def contains(
        self, substring: str, *, case_sensitive: bool = False, name: str | None = None
    ) -> Condition[AgentEventT]:
        """
        Creates a Condition that checks if the field contains a substring.

        Args:
            substring: The substring to search for.
            case_sensitive: Whether the search should be case-sensitive.
            name: Optional name for the condition.

        Returns:
            A Condition that is True if the field contains the substring.
        """
        model_type = self.model_type
        field_name = self.field_name
        parser = self.parser

        def evaluate(event: AgentEventT) -> bool:
            model = parser.try_parse(event, model_type)
            if model is None:
                return False
            field_value = getattr(model, field_name, None)
            if not isinstance(field_value, str):
                return False
            if case_sensitive:
                return substring in field_value
            return substring.lower() in field_value.lower()

        return Condition(evaluate, name=name or f"{field_name}_contains_{substring}")

    def is_not_none(self, *, name: str | None = None) -> Condition[AgentEventT]:
        """
        Creates a Condition that checks if the field is not None.

        Args:
            name: Optional name for the condition.

        Returns:
            A Condition that is True if the field is not None.
        """
        model_type = self.model_type
        field_name = self.field_name
        parser = self.parser

        def evaluate(event: AgentEventT) -> bool:
            model = parser.try_parse(event, model_type)
            if model is None:
                return False
            return getattr(model, field_name, None) is not None

        return Condition(evaluate, name=name or f"{field_name}_is_not_none")

    def one_of(self, *values: object, name: str | None = None) -> Condition[AgentEventT]:
        """
        Creates a Condition that checks if the field is one of the specified values.

        Args:
            *values: The values to check against.
            name: Optional name for the condition.

        Returns:
            A Condition that is True if the field is one of the values.
        """
        model_type = self.model_type
        field_name = self.field_name
        parser = self.parser
        value_set = set(values)

        def evaluate(event: AgentEventT) -> bool:
            model = parser.try_parse(event, model_type)
            if model is None:
                return False
            return getattr(model, field_name, None) in value_set

        return Condition(evaluate, name=name or f"{field_name}_one_of")


class ModelConditionBuilder(t.Generic[AgentEventT]):
    """Represents the start of a fluent condition chain for a model."""

    def __init__(self, model_type: "XMLModel", parser: "t.Any"):
        self._model_type = model_type
        self._parser = parser

    def __getattr__(self, name: str) -> FieldConditionBuilder[AgentEventT]:
        if name.startswith("_"):
            raise AttributeError(name)
        return FieldConditionBuilder(self._model_type, name, self._parser)

    def exists(self, *, name: str | None = None) -> Condition[AgentEventT]:
        """
        Creates a Condition that passes if the model can be parsed from the event.

        Args:
            name: Optional name for the condition.

        Returns:
            A Condition that is True if the model exists.
        """
        model_type = self._model_type
        parser = self._parser

        def evaluate(event: AgentEventT) -> bool:
            return parser.try_parse(event, model_type) is not None

        return Condition(evaluate, name=name or f"{model_type.__name__}_exists")

    def satisfies(
        self, scorer: Scorer["XMLModel"], *, name: str | None = None
    ) -> Condition[AgentEventT]:
        """
        Creates a Condition that passes if the extracted model satisfies the scorer.

        Args:
            scorer: The scorer to evaluate the model against.
            name: Optional name for the condition.

        Returns:
            A Condition that is True if the model satisfies the scorer.
        """
        model_type = self._model_type
        parser = self._parser

        async def evaluate(event: AgentEventT) -> bool:
            model = parser.try_parse(event, model_type)
            if model is None:
                return False
            metric = await scorer.score(model)
            return metric.value > 0.0

        return Condition(evaluate, name=name or f"{model_type.__name__}_satisfies_{scorer.name}")


class ListFieldConditionBuilder(t.Generic[AgentEventT]):
    """Represents a specific field on a list of models, ready for comparison."""

    def __init__(
        self,
        parent: "ListConditionBuilder[AgentEventT]",
        field_name: str,
    ):
        self.parent = parent
        self.field_name = field_name

    def __hash__(self) -> int:
        return hash((id(self.parent), self.field_name))

    def equals(self, value: object, *, name: str | None = None) -> Condition[AgentEventT]:
        """
        Creates a Condition for an equality check on a list of models.

        Args:
            value: The value to compare against.
            name: Optional name for the condition.

        Returns:
            A Condition based on the aggregator (any/all).
        """
        parent = self.parent
        field_name = self.field_name

        def evaluate(event: AgentEventT) -> bool:
            models = parent._parser.try_parse_many(event, parent._model_type)
            if not models:
                return False
            results = (getattr(model, field_name, None) == value for model in models)
            return parent._aggregator(results)

        return Condition(
            evaluate,
            name=name or f"{parent._aggregator_name}_{field_name}_equals_{value}",
        )

    def contains(
        self, substring: str, *, case_sensitive: bool = False, name: str | None = None
    ) -> Condition[AgentEventT]:
        """
        Creates a Condition that checks if the field contains a substring on a list of models.

        Args:
            substring: The substring to search for.
            case_sensitive: Whether the search should be case-sensitive.
            name: Optional name for the condition.

        Returns:
            A Condition based on the aggregator (any/all).
        """
        parent = self.parent
        field_name = self.field_name

        def check_contains(field_value: t.Any) -> bool:
            if not isinstance(field_value, str):
                return False
            if case_sensitive:
                return substring in field_value
            return substring.lower() in field_value.lower()

        def evaluate(event: AgentEventT) -> bool:
            models = parent._parser.try_parse_many(event, parent._model_type)
            if not models:
                return False
            results = (check_contains(getattr(model, field_name, None)) for model in models)
            return parent._aggregator(results)

        return Condition(
            evaluate,
            name=name or f"{parent._aggregator_name}_{field_name}_contains_{substring}",
        )

    def is_not_none(self, *, name: str | None = None) -> Condition[AgentEventT]:
        """
        Creates a Condition that checks if the field is not None on a list of models.

        Args:
            name: Optional name for the condition.

        Returns:
            A Condition based on the aggregator (any/all).
        """
        parent = self.parent
        field_name = self.field_name

        def evaluate(event: AgentEventT) -> bool:
            models = parent._parser.try_parse_many(event, parent._model_type)
            if not models:
                return False
            results = (getattr(model, field_name, None) is not None for model in models)
            return parent._aggregator(results)

        return Condition(
            evaluate,
            name=name or f"{parent._aggregator_name}_{field_name}_is_not_none",
        )


class ListConditionBuilder(t.Generic[AgentEventT]):
    """Builds a condition by applying a check to a list of models."""

    def __init__(
        self,
        model_type: "XMLModel",
        aggregator: t.Callable[[t.Iterable[bool]], bool],
        parser: "t.Any",
    ):
        self._model_type = model_type
        self._aggregator = aggregator
        self._aggregator_name = aggregator.__name__
        self._parser = parser

    def __getattr__(self, name: str) -> ListFieldConditionBuilder[AgentEventT]:
        if name.startswith("_"):
            raise AttributeError(name)
        return ListFieldConditionBuilder(self, name)

    def exists(self, *, name: str | None = None) -> Condition[AgentEventT]:
        """
        Creates a Condition that passes based on whether models exist.

        For `when_any`: True if at least one model exists.
        For `when_all`: True if at least one model exists (same behavior).

        Args:
            name: Optional name for the condition.

        Returns:
            A Condition that is True if models exist.
        """
        model_type = self._model_type
        parser = self._parser

        def evaluate(event: AgentEventT) -> bool:
            models = parser.try_parse_many(event, model_type)
            return len(models) > 0

        return Condition(evaluate, name=name or f"{model_type.__name__}_exists")

    def satisfies(
        self, scorer: Scorer["XMLModel"], *, name: str | None = None
    ) -> Condition[AgentEventT]:
        """
        Applies a Scorer to each model and aggregates the boolean results.

        Args:
            scorer: The scorer to evaluate each model against.
            name: Optional name for the condition.

        Returns:
            A Condition based on the aggregator (any/all).
        """
        model_type = self._model_type
        aggregator = self._aggregator
        aggregator_name = self._aggregator_name
        parser = self._parser

        async def evaluate(event: AgentEventT) -> bool:
            models = parser.try_parse_many(event, model_type)
            if not models:
                return False
            results = [(await scorer.score(model)).value > 0.0 for model in models]
            return aggregator(results)

        return Condition(
            evaluate,
            name=name or f"{aggregator_name}_{model_type.__name__}_satisfies_{scorer.name}",
        )

    def count_equals(self, count: int, *, name: str | None = None) -> Condition[AgentEventT]:
        """
        Creates a Condition that checks if the number of models equals the count.

        Args:
            count: The expected number of models.
            name: Optional name for the condition.

        Returns:
            A Condition that is True if the count matches.
        """
        model_type = self._model_type
        parser = self._parser

        def evaluate(event: AgentEventT) -> bool:
            models = parser.try_parse_many(event, model_type)
            return len(models) == count

        return Condition(evaluate, name=name or f"{model_type.__name__}_count_equals_{count}")

    def count_at_least(self, count: int, *, name: str | None = None) -> Condition[AgentEventT]:
        """
        Creates a Condition that checks if the number of models is at least the count.

        Args:
            count: The minimum number of models.
            name: Optional name for the condition.

        Returns:
            A Condition that is True if there are at least `count` models.
        """
        model_type = self._model_type
        parser = self._parser

        def evaluate(event: AgentEventT) -> bool:
            models = parser.try_parse_many(event, model_type)
            return len(models) >= count

        return Condition(evaluate, name=name or f"{model_type.__name__}_count_at_least_{count}")


# =============================================================================
# Entry Points for Fluent Builder Pattern
# =============================================================================


def when(
    model_type: "XMLModel", *, parser: "t.Any | None" = None
) -> ModelConditionBuilder["AgentStep"]:
    """
    Entry point for conditions that operate on a single model instance.

    Args:
        model_type: The model type to parse from the event.
        parser: Optional parser module. If None, imports from rigging.

    Returns:
        A ModelConditionBuilder for fluent condition construction.

    Example:
        ```python
        # Check if Thinking model exists
        has_thinking = when(Thinking).exists()

        # Check if Thinking status equals "complete"
        thinking_done = when(Thinking).status.equals("complete")

        # Check if Thinking satisfies a scorer
        thinking_confident = when(Thinking).satisfies(confidence_scorer)
        ```
    """
    if parser is None:
        from rigging import parse

        parser = parse

    return ModelConditionBuilder(model_type, parser)


def when_any(
    model_type: "XMLModel", *, parser: "t.Any | None" = None
) -> ListConditionBuilder["AgentStep"]:
    """
    Entry point for conditions that are true if ANY model in a set matches.

    Args:
        model_type: The model type to parse from the event.
        parser: Optional parser module. If None, imports from rigging.

    Returns:
        A ListConditionBuilder with `any` aggregation.

    Example:
        ```python
        # True if any ToolResult has status "error"
        any_error = when_any(ToolResult).status.equals("error")
        ```
    """
    if parser is None:
        from rigging import parse

        parser = parse

    return ListConditionBuilder(model_type, any, parser)


def when_all(
    model_type: "XMLModel", *, parser: "t.Any | None" = None
) -> ListConditionBuilder["AgentStep"]:
    """
    Entry point for conditions that are true only if ALL models in a set match.

    Args:
        model_type: The model type to parse from the event.
        parser: Optional parser module. If None, imports from rigging.

    Returns:
        A ListConditionBuilder with `all` aggregation.

    Example:
        ```python
        # True only if all ToolResults have status "success"
        all_success = when_all(ToolResult).status.equals("success")
        ```
    """
    if parser is None:
        from rigging import parse

        parser = parse

    return ListConditionBuilder(model_type, all, parser)


# =============================================================================
# Decorator
# =============================================================================


@t.overload
def condition(
    func: None = None,
    /,
    *,
    name: str | None = None,
    catch: bool = False,
    default: bool = False,
) -> t.Callable[[ConditionCallable[AgentEventT]], Condition[AgentEventT]]: ...


@t.overload
def condition(
    func: ConditionCallable[AgentEventT],
    /,
) -> Condition[AgentEventT]: ...


def condition(
    func: ConditionCallable[AgentEventT] | None = None,
    *,
    name: str | None = None,
    catch: bool = False,
    default: bool = False,
) -> t.Callable[[ConditionCallable[AgentEventT]], Condition[AgentEventT]] | Condition[AgentEventT]:
    """
    Make a condition from a callable function.

    This decorator can be used with or without arguments.

    Example:
        ```python
        @condition
        def has_tool_calls(event: ToolEvent) -> bool:
            return len(event.tool_calls) > 0

        @condition(name="is_search", catch=True)
        def is_search_tool(event: ToolEvent) -> bool:
            return event.tool.name == "search"
        ```

    Args:
        func: The callable to wrap.
        name: Optional name for the condition.
        catch: If True, catch exceptions and return the default value.
        default: The default value to return if an exception is caught.

    Returns:
        A Condition instance.
    """
    if isinstance(func, Condition):
        return func

    def make_condition(func: ConditionCallable[AgentEventT]) -> Condition[AgentEventT]:
        if isinstance(func, Condition):
            return func.with_(name=name, catch=catch, default=default)
        return Condition(func, name=name, catch=catch, default=default)

    return make_condition if func is None else make_condition(func)

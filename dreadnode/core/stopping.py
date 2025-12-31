import inspect
import typing as t

from dreadnode.core.util import get_callable_name

InputT = t.TypeVar("InputT")
InputT_contra = t.TypeVar("InputT_contra", contravariant=True)


class StopConditionWarning(UserWarning):
    """Warning issued for non-critical issues during stop condition evaluation."""


@t.runtime_checkable
class StopConditionCallable(t.Protocol, t.Generic[InputT_contra]):
    """A callable that takes an input and returns a boolean."""

    def __call__(self, input: InputT_contra, /, *args: t.Any, **kwargs: t.Any) -> bool: ...


class StopCondition(t.Generic[InputT]):
    """
    A condition that determines when a process should stop.

    Conditions can be combined using & (AND), | (OR), and ~ (NOT).
    Generic over the input type that the condition evaluates.

    Examples:
        ```python
        # Create a simple stop condition
        max_steps = StopCondition(lambda events: len(events) >= 10, name="max_steps")

        # Combine conditions
        stop_early = max_steps | timeout_condition

        # Use the decorator
        @stop_condition
        def custom_stop(events: list[Event]) -> bool:
            return events[-1].is_terminal
        ```
    """

    def __init__(
        self,
        func: StopConditionCallable[InputT],
        *,
        name: str | None = None,
        catch: bool = False,
        default: bool = False,
    ):
        """
        Initialize a StopCondition.

        Args:
            func: The callable that evaluates the condition.
            name: Optional name for the condition. If None, derived from the function name.
            catch: If True, catches exceptions and returns the default value.
            default: The default value to return if an exception is caught.
        """
        if name is None:
            unwrapped = inspect.unwrap(func)
            name = get_callable_name(unwrapped, short=True)

        self.func = func
        self.name = name
        self.catch = catch
        self.default = default

    def __repr__(self) -> str:
        return f"StopCondition(name='{self.name}', catch={self.catch}, default={self.default})"

    @classmethod
    def fit(cls, condition: "StopConditionLike[InputT]") -> "StopCondition[InputT]":
        """
        Ensures that the provided condition is a StopCondition instance.

        Args:
            condition: A StopCondition or callable to convert.

        Returns:
            A StopCondition instance.

        Raises:
            TypeError: If the condition is not a StopCondition or callable.
        """
        if isinstance(condition, StopCondition):
            return condition
        if callable(condition):
            return StopCondition(condition)
        raise TypeError("StopCondition must be a StopCondition instance or a callable.")

    @classmethod
    def fit_many(
        cls, conditions: "StopConditionsLike[InputT] | None"
    ) -> list["StopCondition[InputT]"]:
        """
        Convert a collection of condition-like objects into a list of StopCondition instances.

        Args:
            conditions: A collection of condition-like objects. Can be:
                - A dictionary mapping names to condition objects or callables
                - A sequence of condition objects or callables
                - None (returns empty list)

        Returns:
            A list of StopCondition instances.
        """
        if isinstance(conditions, t.Mapping):
            return [
                condition.with_(name=name)
                if isinstance(condition, StopCondition)
                else cls(condition, name=name)
                for name, condition in conditions.items()
            ]

        return [
            condition if isinstance(condition, StopCondition) else cls(condition)
            for condition in conditions or []
        ]

    def __deepcopy__(self, memo: dict[int, t.Any]) -> "StopCondition[InputT]":
        return StopCondition(
            func=self.func,
            name=self.name,
            catch=self.catch,
            default=self.default,
        )

    def clone(self) -> "StopCondition[InputT]":
        """
        Clone the stop condition.

        Returns:
            A new StopCondition with the same properties.
        """
        return self.__deepcopy__({})

    def with_(
        self,
        *,
        name: str | None = None,
        catch: bool | None = None,
        default: bool | None = None,
    ) -> "StopCondition[InputT]":
        """
        Create a new StopCondition with updated properties.

        Args:
            name: New name for the condition.
            catch: Whether to catch exceptions.
            default: Default value to return on exception.

        Returns:
            A new StopCondition with the updated properties.
        """
        new = self.clone()
        new.name = name or self.name
        new.catch = catch if catch is not None else self.catch
        new.default = default if default is not None else self.default
        return new

    def rename(self, new_name: str) -> "StopCondition[InputT]":
        """
        Rename the stop condition.

        Args:
            new_name: The new name for the condition.

        Returns:
            A new StopCondition with the updated name.
        """
        return self.with_(name=new_name)

    def evaluate(self, input: InputT, *args: t.Any, **kwargs: t.Any) -> bool:
        """
        Evaluate the stop condition.

        Args:
            input: The input to evaluate.
            *args: Additional positional arguments passed to the function.
            **kwargs: Additional keyword arguments passed to the function.

        Returns:
            True if the condition is met (should stop), False otherwise.
        """
        try:
            return self.func(input, *args, **kwargs)
        except Exception as e:
            if not self.catch:
                raise

            from dreadnode.core.exceptions import warn_at_user_stacklevel

            warn_at_user_stacklevel(
                f"Error evaluating stop condition {self.name!r}: {e}",
                StopConditionWarning,
            )
            return self.default

    def __call__(self, input: InputT, *args: t.Any, **kwargs: t.Any) -> bool:
        """Evaluate the stop condition (alias for evaluate)."""
        return self.evaluate(input, *args, **kwargs)

    def __and__(self, other: "StopConditionLike[InputT]") -> "StopCondition[InputT]":
        """Combine with another condition using AND logic."""
        return and_(self, StopCondition.fit(other))

    def __or__(self, other: "StopConditionLike[InputT]") -> "StopCondition[InputT]":
        """Combine with another condition using OR logic."""
        return or_(self, StopCondition.fit(other))

    def __invert__(self) -> "StopCondition[InputT]":
        """Negate this condition."""
        return not_(self)


StopConditionLike = StopCondition[InputT] | StopConditionCallable[InputT]
"""A stop condition or compatible callable."""

StopConditionsLike = (
    t.Sequence[StopConditionLike[InputT]] | t.Mapping[str, StopConditionLike[InputT]]
)
"""A sequence of stop condition-like objects or mapping of name/condition pairs."""


def and_(
    condition: StopCondition[InputT],
    other: StopCondition[InputT],
    *,
    name: str | None = None,
) -> StopCondition[InputT]:
    """
    Combine two stop conditions with AND logic.

    The resulting condition returns True only if both conditions return True.

    Args:
        condition: The first condition.
        other: The second condition.
        name: Optional name for the combined condition.

    Returns:
        A new StopCondition that is True only if both conditions are True.
    """

    def evaluate(input: InputT) -> bool:
        return condition.evaluate(input) and other.evaluate(input)

    return StopCondition(evaluate, name=name or f"({condition.name} & {other.name})")


def or_(
    condition: StopCondition[InputT],
    other: StopCondition[InputT],
    *,
    name: str | None = None,
) -> StopCondition[InputT]:
    """
    Combine two stop conditions with OR logic.

    The resulting condition returns True if either condition returns True.

    Args:
        condition: The first condition.
        other: The second condition.
        name: Optional name for the combined condition.

    Returns:
        A new StopCondition that is True if either condition is True.
    """

    def evaluate(input: InputT) -> bool:
        return condition.evaluate(input) or other.evaluate(input)

    return StopCondition(evaluate, name=name or f"({condition.name} | {other.name})")


def not_(
    condition: StopCondition[InputT],
    *,
    name: str | None = None,
) -> StopCondition[InputT]:
    """
    Negate a stop condition.

    The resulting condition returns True when the original returns False.

    Args:
        condition: The condition to negate.
        name: Optional name for the negated condition.

    Returns:
        A new StopCondition that is True when the original is False.
    """

    def evaluate(input: InputT) -> bool:
        return not condition.evaluate(input)

    return StopCondition(evaluate, name=name or f"~{condition.name}")


def always(*, name: str = "always") -> StopCondition[InputT]:
    """
    A stop condition that always returns True (always stops).

    Args:
        name: Optional name for the condition.

    Returns:
        A StopCondition that always evaluates to True.
    """

    def evaluate(input: InputT) -> bool:  # noqa: ARG001
        return True

    return StopCondition(evaluate, name=name)


def never(*, name: str = "never") -> StopCondition[InputT]:
    """
    A stop condition that always returns False (never stops).

    Args:
        name: Optional name for the condition.

    Returns:
        A StopCondition that always evaluates to False.
    """

    def evaluate(input: InputT) -> bool:  # noqa: ARG001
        return False

    return StopCondition(evaluate, name=name)


@t.overload
def stop_condition(
    func: None = None,
    /,
    *,
    name: str | None = None,
    catch: bool = False,
    default: bool = False,
) -> t.Callable[[StopConditionCallable[InputT]], StopCondition[InputT]]: ...


@t.overload
def stop_condition(
    func: StopConditionCallable[InputT],
    /,
) -> StopCondition[InputT]: ...


def stop_condition(
    func: StopConditionCallable[InputT] | None = None,
    *,
    name: str | None = None,
    catch: bool = False,
    default: bool = False,
) -> t.Callable[[StopConditionCallable[InputT]], StopCondition[InputT]] | StopCondition[InputT]:
    """
    Make a stop condition from a callable function.

    This decorator can be used with or without arguments.

    Examples:
        ```python
        @stop_condition
        def max_steps(events: Sequence[AgentStep]) -> bool:
            return len(events) >= 10

        @stop_condition(name="custom_stop", catch=True)
        def risky_check(events: Sequence[AgentStep]) -> bool:
            return events[-1].risky_field > 100
        ```

    Args:
        func: The callable to wrap.
        name: Optional name for the condition.
        catch: If True, catch exceptions and return the default value.
        default: The default value to return if an exception is caught.

    Returns:
        A StopCondition instance.
    """
    if isinstance(func, StopCondition):
        return func

    def make_condition(func: StopConditionCallable[InputT]) -> StopCondition[InputT]:
        if isinstance(func, StopCondition):
            return func.with_(name=name, catch=catch, default=default)
        return StopCondition(func, name=name, catch=catch, default=default)

    return make_condition if func is None else make_condition(func)

import inspect
import re
import typing as t
from collections.abc import Sequence

from dreadnode.agent.events import AgentEvent, GenerationEnd, ToolEnd
from dreadnode.meta import Config
from dreadnode.util import get_callable_name


class StopCondition:
    """
    A condition that determines when an agent's run should stop, defined by a callable.
    Conditions can be combined using & (AND) and | (OR).
    """

    def __init__(self, func: t.Callable[[Sequence[AgentEvent]], bool], name: str | None = None):
        """
        Initializes the StopCondition.

        Args:
            func: A callable that takes a sequence of events and returns True if the run should stop.
            name: An optional name for the condition for representation.
        """

        if name is None:
            unwrapped = inspect.unwrap(func)
            name = get_callable_name(unwrapped, short=True)

        self.func = func
        """The function that defines the stop condition."""
        self.name = name
        """A human-readable name for the condition."""

    def __repr__(self) -> str:
        return f"StopCondition(name='{self.name}')"

    def __call__(self, events: Sequence[AgentEvent]) -> bool:
        return self.func(events)

    def __and__(self, other: "StopCondition") -> "StopCondition":
        """Combines this condition with another using AND logic."""
        return and_(self, other)

    def __or__(self, other: "StopCondition") -> "StopCondition":
        """Combines this condition with another using OR logic."""
        return or_(self, other)


def and_(
    condition: StopCondition, other: StopCondition, *, name: str | None = None
) -> StopCondition:
    """Perform a logical AND with two conditions."""

    def stop(events: Sequence[AgentEvent]) -> bool:
        return condition(events) and other(events)

    return StopCondition(stop, name=name or f"({condition.name}_and_{other.name})")


def or_(
    condition: StopCondition, other: StopCondition, *, name: str | None = None
) -> StopCondition:
    """Perform a logical OR with two conditions."""

    def stop(events: Sequence[AgentEvent]) -> bool:
        return condition(events) or other(events)

    return StopCondition(stop, name=name or f"({condition.name}_or_{other.name})")


def stop_never() -> StopCondition:
    """A condition that never stops the agent."""

    def stop(_: Sequence[AgentEvent]) -> bool:
        return False

    return StopCondition(stop, name="stop_never")


def stop_after_steps(max_steps: int) -> StopCondition:
    """Terminates after a maximum number of LLM calls (steps)."""

    def stop(events: Sequence[AgentEvent], *, max_steps: int = Config(max_steps)) -> bool:
        step_count = sum(1 for event in events if isinstance(event, GenerationEnd))
        return step_count >= max_steps

    return StopCondition(stop, name="stop_after_steps")


def stop_on_tool_use(tool_name: str) -> StopCondition:
    """Terminates after a specific tool has been successfully used."""

    def stop(events: Sequence[AgentEvent]) -> bool:
        return any(isinstance(e, ToolEnd) and e.tool_call.name == tool_name for e in events)

    return StopCondition(stop, name="stop_on_tool_use")


def stop_on_text(
    pattern: str | re.Pattern[str],
    *,
    case_sensitive: bool = False,
    exact: bool = False,
    regex: bool = False,
) -> StopCondition:
    """
    Terminates if a specific string or pattern is mentioned in the last generated message.

    Args:
        pattern: The string or compiled regex pattern to search for.
        case_sensitive: If True, the match is case-sensitive. Defaults to False.
        exact: If True, performs an exact string match instead of containment. Defaults to False.
        regex: If True, treats the `pattern` string as a regular expression. Defaults to False.
    """

    def stop(events: Sequence[AgentEvent]) -> bool:
        if not events:
            return False

        last_generation = next((e for e in reversed(events) if isinstance(e, GenerationEnd)), None)
        if not last_generation:
            return False

        text = last_generation.message.content
        found = False

        if isinstance(pattern, re.Pattern) or regex:
            compiled = pattern
            if isinstance(pattern, str):
                flags = 0 if case_sensitive else re.IGNORECASE
                compiled = re.compile(pattern, flags)

            if isinstance(compiled, re.Pattern):  # Make type checker happy
                found = bool(compiled.search(text))
        elif exact:
            found = text == pattern if case_sensitive else text.lower() == str(pattern).lower()
        else:  # Default to substring containment
            search_text = text if case_sensitive else text.lower()
            search_pattern = str(pattern) if case_sensitive else str(pattern).lower()
            found = search_pattern in search_text

        return found

    return StopCondition(stop, name="stop_on_text")

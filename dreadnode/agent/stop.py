import inspect
import re
import typing as t
from collections.abc import Sequence

from dreadnode.agent.events import AgentEvent, GenerationEnd, StepStart, ToolEnd
from dreadnode.meta import Config
from dreadnode.meta.config import component
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
    """Perform a logical AND with two stop conditions."""

    def stop(events: Sequence[AgentEvent]) -> bool:
        return condition(events) and other(events)

    return StopCondition(stop, name=name or f"({condition.name}_and_{other.name})")


def or_(
    condition: StopCondition, other: StopCondition, *, name: str | None = None
) -> StopCondition:
    """Perform a logical OR with two stop conditions."""

    def stop(events: Sequence[AgentEvent]) -> bool:
        return condition(events) or other(events)

    return StopCondition(stop, name=name or f"({condition.name}_or_{other.name})")


def never() -> StopCondition:
    """
    A condition that never stops the agent.

    This is generally useful for triggering stalling
    conditions when an agent does not issue any tool
    calls, and a hook reaction will be used.
    """

    def stop(_: Sequence[AgentEvent]) -> bool:
        return False

    return StopCondition(stop, name="stop_never")


def generation_count(max_generations: int) -> StopCondition:
    """
    Stop after a maximum number of LLM generations (inference calls).

    This is slightly more robust than using `max_steps` as
    retry calls to the LLM will also count towards this limit.

    Args:
        max_generations: The maximum number of LLM generations to allow.
    """

    @component
    def stop(
        events: Sequence[AgentEvent], *, max_generations: int = Config(max_generations)
    ) -> bool:
        generation_count = sum(1 for event in events if isinstance(event, GenerationEnd))
        return generation_count >= max_generations

    return StopCondition(stop, name="stop_on_generation_count")


def tool_use(tool_name: str, *, count: int = 1) -> StopCondition:
    """
    Stop after a specific tool has been successfully used.

    Args:
        tool_name: The name of the tool to monitor.
        count: The number of times the tool must be used to trigger stopping. Defaults to 1.
    """

    def stop(events: Sequence[AgentEvent]) -> bool:
        tool_count = sum(
            1 for e in events if isinstance(e, ToolEnd) and e.tool_call.name == tool_name
        )
        return tool_count >= count

    return StopCondition(stop, name="stop_on_tool_use")


def output(
    pattern: str | re.Pattern[str],
    *,
    case_sensitive: bool = False,
    exact: bool = False,
    regex: bool = False,
) -> StopCondition:
    """
    Stop if a specific string or pattern is mentioned in the last generated message.

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

    return StopCondition(stop, name="stop_on_output")


def tool_output(
    pattern: str | re.Pattern[str],
    *,
    tool_name: str | None = None,
    case_sensitive: bool = False,
    exact: bool = False,
    regex: bool = False,
) -> StopCondition:
    """
    Stop if a specific string or pattern is found in the output of a tool call.

    Args:
        pattern: The string or compiled regex pattern to search for.
        tool_name: If specified, only considers outputs from this tool.
        case_sensitive: If True, the match is case-sensitive. Defaults to False.
        exact: If True, performs an exact string match instead of containment. Defaults to False.
        regex: If True, treats the `pattern` string as a regular expression. Defaults to False.
    """

    def stop(events: Sequence[AgentEvent]) -> bool:
        for event in reversed(events):
            if isinstance(event, ToolEnd):
                if tool_name and event.tool_call.name != tool_name:
                    continue

                output = event.message.content
                if output is None:
                    continue

                text = str(output)
                found = False

                if isinstance(pattern, re.Pattern) or regex:
                    compiled = pattern
                    if isinstance(pattern, str):
                        flags = 0 if case_sensitive else re.IGNORECASE
                        compiled = re.compile(pattern, flags)

                    if isinstance(compiled, re.Pattern):  # Make type checker happy
                        found = bool(compiled.search(text))
                elif exact:
                    found = (
                        text == pattern if case_sensitive else text.lower() == str(pattern).lower()
                    )
                else:  # Default to substring containment
                    search_text = text if case_sensitive else text.lower()
                    search_pattern = str(pattern) if case_sensitive else str(pattern).lower()
                    found = search_pattern in search_text

                if found:
                    return True

        return False

    return StopCondition(stop, name="stop_on_tool_output")


def tool_error(tool_name: str | None = None) -> StopCondition:
    """
    Stop if any tool call results in an gracefully handled error.

    Args:
        tool_name: If specified, only considers errors from this tool.
    """

    def stop(events: Sequence[AgentEvent]) -> bool:
        for event in reversed(events):
            if isinstance(event, ToolEnd):
                if tool_name and event.tool_call.name != tool_name:
                    continue

                if "error" in event.message.metadata:
                    return True

        return False

    return StopCondition(stop, name="stop_on_tool_error")


def no_new_tool_used(for_steps: int) -> StopCondition:
    """
    Stop if the agent goes for a number of steps without using a new tool.

    Args:
        for_steps: The number of consecutive steps without a new tool use
            before the agent should stop.
    """

    def stop(events: Sequence[AgentEvent]) -> bool:
        step_starts = [e for e in events if isinstance(e, StepStart)]
        if len(step_starts) < for_steps:
            return False

        # Get events from the last `for_steps` steps
        relevant_events = events[events.index(step_starts[-for_steps]) :]

        used_tools_in_period = {e.tool_call.name for e in relevant_events if isinstance(e, ToolEnd)}

        # Find tools used before this period
        prior_events = events[: events.index(step_starts[-for_steps])]
        prior_tools = {e.tool_call.name for e in prior_events if isinstance(e, ToolEnd)}

        # If any tool used in the current period is new, don't stop
        return used_tools_in_period - prior_tools != set()

    return StopCondition(stop, name="stop_on_no_new_tool")


def token_usage(limit: int, *, mode: t.Literal["total", "in", "out"] = "total") -> StopCondition:
    """
    Stop if the token usage exceeds a specified limit.

    Args:
        limit: The maximum number of tokens allowed.
        mode: Which token count to consider:
            - "total": Total tokens (default)
            - "in": Input tokens only
            - "out": Output tokens only
    """

    def stop(events: Sequence[AgentEvent]) -> bool:
        last_event = next((e for e in reversed(events)), None)
        if not last_event:
            return False

        usage = last_event.total_usage
        token_count = (
            usage.total_tokens
            if mode == "total"
            else (usage.input_tokens if mode == "in" else usage.output_tokens)
        )

        return token_count > limit

    return StopCondition(stop, name="stop_on_token_usage")


def elapsed_time(max_seconds: int) -> StopCondition:
    """
    Stop if the total execution time exceeds a given duration.

    Args:
        max_seconds: The maximum number of seconds the agent is allowed to run.
    """

    def stop(events: Sequence[AgentEvent]) -> bool:
        if len(events) < 2:
            return False

        first_event = events[0]
        last_event = events[-1]

        delta = last_event.timestamp - first_event.timestamp
        return delta.total_seconds() > max_seconds

    return StopCondition(stop, name="stop_on_elapsed_time")


def estimated_cost(limit: float) -> StopCondition:
    """
    Stop if the estimated cost of LLM generations exceeds a limit.

    Args:
        limit: The maximum cost allowed (USD).
    """

    def stop(events: Sequence[AgentEvent]) -> bool:
        last_event = next((e for e in reversed(events)), None)
        if not last_event:
            return False
        estimated_cost = last_event.estimated_cost
        return estimated_cost > limit if estimated_cost else False

    return StopCondition(stop, name="stop_on_estimated_cost")

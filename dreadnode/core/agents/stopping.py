"""
Agent-specific stop conditions.

This module provides stop conditions for controlling when an agent should stop execution.
All conditions operate on a sequence of AgentStep events.
"""

import re
import typing as t
from collections.abc import Sequence

from dreadnode.core.agents.events import AgentStep, GenerationEnd, ToolEnd
from dreadnode.core.meta import Config
from dreadnode.core.meta.config import component
from dreadnode.core.stopping import StopCondition
from dreadnode.core.stopping import never as base_never

AgentStopCondition = StopCondition[Sequence[AgentStep]]
"""Type alias for agent stop conditions."""


def never() -> AgentStopCondition:
    """
    A condition that never stops the agent.

    This is generally useful for triggering stalling conditions when an agent
    does not issue any tool calls, and a hook reaction will be used.

    Returns:
        A StopCondition that always returns False.
    """
    return base_never(name="stop_never")


def generation_count(max_generations: int, *, name: str | None = None) -> AgentStopCondition:
    """
    Stop after a maximum number of LLM generations (inference calls).

    This is slightly more robust than using `max_steps` as retry calls
    to the LLM will also count towards this limit.

    Args:
        max_generations: The maximum number of LLM generations to allow.
        name: Optional name for the condition.

    Returns:
        A StopCondition that triggers after the specified number of generations.
    """

    @component
    def evaluate(
        events: Sequence[AgentStep], *, max_generations: int = Config(max_generations)
    ) -> bool:
        count = sum(1 for event in events if isinstance(event, GenerationEnd))
        return count >= max_generations

    return StopCondition(evaluate, name=name or f"stop_on_generation_count({max_generations})")


def step_count(max_steps: int, *, name: str | None = None) -> AgentStopCondition:
    """
    Stop after a maximum number of agent steps.

    Args:
        max_steps: The maximum number of steps to allow.
        name: Optional name for the condition.

    Returns:
        A StopCondition that triggers after the specified number of steps.
    """

    def evaluate(events: Sequence[AgentStep]) -> bool:
        return len(events) >= max_steps

    return StopCondition(evaluate, name=name or f"stop_on_step_count({max_steps})")


def tool_use(tool_name: str, *, count: int = 1, name: str | None = None) -> AgentStopCondition:
    """
    Stop after a specific tool has been successfully used.

    Args:
        tool_name: The name of the tool to monitor.
        count: The number of times the tool must be used to trigger stopping.
        name: Optional name for the condition.

    Returns:
        A StopCondition that triggers after the tool is used the specified number of times.
    """

    def evaluate(events: Sequence[AgentStep]) -> bool:
        tool_count = sum(
            1 for e in events if isinstance(e, ToolEnd) and e.tool_call.name == tool_name
        )
        return tool_count >= count

    return StopCondition(evaluate, name=name or f"stop_on_tool_use({tool_name}, {count})")


def any_tool_use(*, count: int = 1, name: str | None = None) -> AgentStopCondition:
    """
    Stop after any tool has been used a specified number of times.

    Args:
        count: The total number of tool uses to trigger stopping.
        name: Optional name for the condition.

    Returns:
        A StopCondition that triggers after any tools are used the specified number of times.
    """

    def evaluate(events: Sequence[AgentStep]) -> bool:
        tool_count = sum(1 for e in events if isinstance(e, ToolEnd))
        return tool_count >= count

    return StopCondition(evaluate, name=name or f"stop_on_any_tool_use({count})")


def output(
    pattern: str | re.Pattern[str],
    *,
    case_sensitive: bool = False,
    exact: bool = False,
    regex: bool = False,
    name: str | None = None,
) -> AgentStopCondition:
    """
    Stop if a specific string or pattern is mentioned in the last generated message.

    Args:
        pattern: The string or compiled regex pattern to search for.
        case_sensitive: If True, the match is case-sensitive.
        exact: If True, performs an exact string match instead of containment.
        regex: If True, treats the `pattern` string as a regular expression.
        name: Optional name for the condition.

    Returns:
        A StopCondition that triggers when the pattern is found in the output.
    """

    def evaluate(events: Sequence[AgentStep]) -> bool:
        if not events:
            return False

        last_generation = next((e for e in reversed(events) if isinstance(e, GenerationEnd)), None)
        if not last_generation:
            return False

        text = last_generation.message.content
        return _match_pattern(text, pattern, case_sensitive, exact, regex)

    pattern_str = pattern.pattern if isinstance(pattern, re.Pattern) else pattern
    return StopCondition(evaluate, name=name or f"stop_on_output({pattern_str!r})")


def tool_output(
    pattern: str | re.Pattern[str],
    *,
    tool_name: str | None = None,
    case_sensitive: bool = False,
    exact: bool = False,
    regex: bool = False,
    name: str | None = None,
) -> AgentStopCondition:
    """
    Stop if a specific string or pattern is found in the output of a tool call.

    Args:
        pattern: The string or compiled regex pattern to search for.
        tool_name: If specified, only considers outputs from this tool.
        case_sensitive: If True, the match is case-sensitive.
        exact: If True, performs an exact string match instead of containment.
        regex: If True, treats the `pattern` string as a regular expression.
        name: Optional name for the condition.

    Returns:
        A StopCondition that triggers when the pattern is found in tool output.
    """

    def evaluate(events: Sequence[AgentStep]) -> bool:
        for event in reversed(events):
            if not isinstance(event, ToolEnd):
                continue
            if tool_name and event.tool_call.name != tool_name:
                continue

            output = event.message.content
            if output is None:
                continue

            if _match_pattern(str(output), pattern, case_sensitive, exact, regex):
                return True

        return False

    pattern_str = pattern.pattern if isinstance(pattern, re.Pattern) else pattern
    tool_suffix = f", tool={tool_name}" if tool_name else ""
    return StopCondition(
        evaluate, name=name or f"stop_on_tool_output({pattern_str!r}{tool_suffix})"
    )


def tool_error(tool_name: str | None = None, *, name: str | None = None) -> AgentStopCondition:
    """
    Stop if any tool call results in a gracefully handled error.

    Args:
        tool_name: If specified, only considers errors from this tool.
        name: Optional name for the condition.

    Returns:
        A StopCondition that triggers when a tool error occurs.
    """

    def evaluate(events: Sequence[AgentStep]) -> bool:
        for event in reversed(events):
            if not isinstance(event, ToolEnd):
                continue
            if tool_name and event.tool_call.name != tool_name:
                continue
            if "error" in event.message.metadata:
                return True

        return False

    return StopCondition(evaluate, name=name or f"stop_on_tool_error({tool_name or 'any'})")


def no_new_tool_used(for_steps: int, *, name: str | None = None) -> AgentStopCondition:
    """
    Stop if the agent goes for a number of steps without using a new tool.

    Args:
        for_steps: The number of consecutive steps without a new tool use
            before the agent should stop.
        name: Optional name for the condition.

    Returns:
        A StopCondition that triggers when no new tools are used for the specified steps.
    """

    def evaluate(events: Sequence[AgentStep]) -> bool:
        step_starts = [e for e in events if isinstance(e, AgentStep)]
        if len(step_starts) < for_steps:
            return False

        relevant_events = events[events.index(step_starts[-for_steps]) :]
        used_tools_in_period = {e.tool_call.name for e in relevant_events if isinstance(e, ToolEnd)}

        prior_events = events[: events.index(step_starts[-for_steps])]
        prior_tools = {e.tool_call.name for e in prior_events if isinstance(e, ToolEnd)}

        return used_tools_in_period - prior_tools == set()

    return StopCondition(evaluate, name=name or f"stop_on_no_new_tool({for_steps})")


def no_tool_calls(for_steps: int = 1, *, name: str | None = None) -> AgentStopCondition:
    """
    Stop if the agent goes for a number of steps without making any tool calls.

    Args:
        for_steps: The number of consecutive steps without any tool calls.
        name: Optional name for the condition.

    Returns:
        A StopCondition that triggers when no tool calls are made for the specified steps.
    """

    def evaluate(events: Sequence[AgentStep]) -> bool:
        generation_events = [e for e in events if isinstance(e, GenerationEnd)]
        if len(generation_events) < for_steps:
            return False

        # Check the last `for_steps` generations
        recent_generations = generation_events[-for_steps:]
        first_idx = events.index(recent_generations[0])
        relevant_events = events[first_idx:]

        tool_calls = sum(1 for e in relevant_events if isinstance(e, ToolEnd))
        return tool_calls == 0

    return StopCondition(evaluate, name=name or f"stop_on_no_tool_calls({for_steps})")


def token_usage(
    limit: int,
    *,
    mode: t.Literal["total", "in", "out"] = "total",
    name: str | None = None,
) -> AgentStopCondition:
    """
    Stop if the token usage exceeds a specified limit.

    Args:
        limit: The maximum number of tokens allowed.
        mode: Which token count to consider: "total", "in", or "out".
        name: Optional name for the condition.

    Returns:
        A StopCondition that triggers when token usage exceeds the limit.
    """

    def evaluate(events: Sequence[AgentStep]) -> bool:
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

    return StopCondition(evaluate, name=name or f"stop_on_token_usage({limit}, {mode})")


def elapsed_time(max_seconds: float, *, name: str | None = None) -> AgentStopCondition:
    """
    Stop if the total execution time exceeds a given duration.

    Args:
        max_seconds: The maximum number of seconds the agent is allowed to run.
        name: Optional name for the condition.

    Returns:
        A StopCondition that triggers when elapsed time exceeds the limit.
    """

    def evaluate(events: Sequence[AgentStep]) -> bool:
        if len(events) < 2:
            return False

        first_event = events[0]
        last_event = events[-1]

        delta = last_event.timestamp - first_event.timestamp
        return delta.total_seconds() > max_seconds

    return StopCondition(evaluate, name=name or f"stop_on_elapsed_time({max_seconds}s)")


def estimated_cost(limit: float, *, name: str | None = None) -> AgentStopCondition:
    """
    Stop if the estimated cost of LLM generations exceeds a limit.

    Args:
        limit: The maximum cost allowed (USD).
        name: Optional name for the condition.

    Returns:
        A StopCondition that triggers when estimated cost exceeds the limit.
    """

    def evaluate(events: Sequence[AgentStep]) -> bool:
        last_event = next((e for e in reversed(events)), None)
        if not last_event:
            return False
        cost = last_event.estimated_cost
        return cost > limit if cost else False

    return StopCondition(evaluate, name=name or f"stop_on_estimated_cost(${limit:.2f})")


def consecutive_errors(count: int, *, name: str | None = None) -> AgentStopCondition:
    """
    Stop if there are consecutive tool errors.

    Args:
        count: The number of consecutive errors before stopping.
        name: Optional name for the condition.

    Returns:
        A StopCondition that triggers after consecutive errors.
    """

    def evaluate(events: Sequence[AgentStep]) -> bool:
        consecutive = 0
        for event in reversed(events):
            if isinstance(event, ToolEnd):
                if "error" in event.message.metadata:
                    consecutive += 1
                    if consecutive >= count:
                        return True
                else:
                    consecutive = 0

        return False

    return StopCondition(evaluate, name=name or f"stop_on_consecutive_errors({count})")


def _match_pattern(
    text: str,
    pattern: str | re.Pattern[str],
    *,
    case_sensitive: bool,
    exact: bool,
    regex: bool,
) -> bool:
    """
    Helper to match text against a pattern with various options.

    Args:
        text: The text to search in.
        pattern: The pattern to search for.
        case_sensitive: Whether the match should be case-sensitive.
        exact: Whether to do an exact match.
        regex: Whether to treat the pattern as a regex.

    Returns:
        True if the pattern matches, False otherwise.
    """
    if isinstance(pattern, re.Pattern) or regex:
        compiled = pattern
        if isinstance(pattern, str):
            flags = 0 if case_sensitive else re.IGNORECASE
            compiled = re.compile(pattern, flags)

        if isinstance(compiled, re.Pattern):
            return bool(compiled.search(text))
        return False

    if exact:
        return text == pattern if case_sensitive else text.lower() == pattern.lower()

    # Default to substring containment
    search_text = text if case_sensitive else text.lower()
    search_pattern = pattern if case_sensitive else pattern.lower()
    return search_pattern in search_text

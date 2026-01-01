"""Explicit span factories for type-safe tracing."""

from __future__ import annotations

import json
import typing as t

from dreadnode.core.tracing.constants import (
    GENERATION_ATTRIBUTE_CONTENT,
    GENERATION_ATTRIBUTE_EXTRA,
    GENERATION_ATTRIBUTE_FAILED,
    GENERATION_ATTRIBUTE_INPUT_TOKENS,
    GENERATION_ATTRIBUTE_MODEL,
    GENERATION_ATTRIBUTE_OUTPUT_TOKENS,
    GENERATION_ATTRIBUTE_ROLE,
    GENERATION_ATTRIBUTE_STOP_REASON,
    GENERATION_ATTRIBUTE_TOOL_CALLS,
    GENERATION_ATTRIBUTE_TOTAL_TOKENS,
    SCORER_ATTRIBUTE_NAME,
    SCORER_ATTRIBUTE_PASSED,
    SCORER_ATTRIBUTE_RATIONALE,
    SCORER_ATTRIBUTE_SCORE,
    SCORER_ATTRIBUTE_STEP,
    TOOL_ATTRIBUTE_ARGUMENTS,
    TOOL_ATTRIBUTE_CALL_ID,
    TOOL_ATTRIBUTE_ERROR,
    TOOL_ATTRIBUTE_NAME,
    TOOL_ATTRIBUTE_RESULT,
    TOOL_ATTRIBUTE_STOPPED,
    TRIAL_ATTRIBUTE_CANDIDATE,
    TRIAL_ATTRIBUTE_ID,
    TRIAL_ATTRIBUTE_IS_PROBE,
    TRIAL_ATTRIBUTE_SCORES,
    TRIAL_ATTRIBUTE_STATUS,
    TRIAL_ATTRIBUTE_STEP,
)
from dreadnode.core.tracing.span import TaskSpan

if t.TYPE_CHECKING:
    from dreadnode.core.types.common import AnyDict


def generation_span(
    *,
    step: int | None = None,
    model: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    content: str | None = None,
    role: str = "assistant",
    tool_calls: list[dict[str, t.Any]] | None = None,
    stop_reason: str | None = None,
    failed: bool = False,
    extra: AnyDict | None = None,
    label: str | None = None,
    tags: list[str] | None = None,
) -> TaskSpan[t.Any]:
    """
    Create a span for LLM generation steps.

    Args:
        step: The step number in the agent execution.
        model: The model identifier used for generation.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        content: The generated content (truncated to 4000 chars).
        role: The message role (usually "assistant").
        tool_calls: List of tool calls with name, id, arguments.
        stop_reason: Why generation stopped (end_turn, tool_use, max_tokens).
        failed: Whether the generation failed.
        extra: Additional metadata from the generator.
        label: Human-readable label for the span.
        tags: Tags for filtering.

    Returns:
        A configured TaskSpan for generation.
    """
    from dreadnode import task_span

    span = task_span(
        name="generation",
        type="generation",
        label=label or (f"step_{step}" if step is not None else None),
        tags=["llm", "generation", *(tags or [])],
    )

    if model:
        span.set_attribute(GENERATION_ATTRIBUTE_MODEL, model)
    if content:
        span.set_attribute(GENERATION_ATTRIBUTE_CONTENT, content[:4000])
    if role:
        span.set_attribute(GENERATION_ATTRIBUTE_ROLE, role)
    if tool_calls:
        span.set_attribute(GENERATION_ATTRIBUTE_TOOL_CALLS, tool_calls)
    if stop_reason:
        span.set_attribute(GENERATION_ATTRIBUTE_STOP_REASON, stop_reason)
    if extra:
        span.set_attribute(GENERATION_ATTRIBUTE_EXTRA, extra)

    span.set_attribute(GENERATION_ATTRIBUTE_INPUT_TOKENS, input_tokens)
    span.set_attribute(GENERATION_ATTRIBUTE_OUTPUT_TOKENS, output_tokens)
    span.set_attribute(GENERATION_ATTRIBUTE_TOTAL_TOKENS, input_tokens + output_tokens)
    span.set_attribute(GENERATION_ATTRIBUTE_FAILED, failed)

    return span


def tool_span(
    name: str,
    *,
    call_id: str,
    arguments: dict[str, t.Any] | str | None = None,
    result: str | None = None,
    error: str | None = None,
    stopped: bool = False,
    label: str | None = None,
    tags: list[str] | None = None,
) -> TaskSpan[t.Any]:
    """
    Create a span for tool execution.

    Args:
        name: The tool name.
        call_id: The unique tool call ID.
        arguments: Tool arguments (parsed dict or raw JSON string).
        result: Tool execution result (truncated to 4000 chars).
        error: Error message if tool failed.
        stopped: Whether this tool requested agent stop.
        label: Human-readable label.
        tags: Tags for filtering.

    Returns:
        A configured TaskSpan for tool execution.
    """
    from dreadnode import task_span

    span = task_span(
        name=f"tool:{name}",
        type="tool",
        label=label or call_id,
        tags=["tool", name, *(tags or [])],
    )

    span.set_attribute(TOOL_ATTRIBUTE_NAME, name)
    span.set_attribute(TOOL_ATTRIBUTE_CALL_ID, call_id)

    if arguments is not None:
        # Parse if string
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        span.set_attribute(TOOL_ATTRIBUTE_ARGUMENTS, arguments)
    if result is not None:
        span.set_attribute(TOOL_ATTRIBUTE_RESULT, result[:4000])
    if error is not None:
        span.set_attribute(TOOL_ATTRIBUTE_ERROR, error[:1000])

    span.set_attribute(TOOL_ATTRIBUTE_STOPPED, stopped)

    return span


def scorer_span(
    name: str,
    *,
    score: float | None = None,
    passed: bool | None = None,
    rationale: str | None = None,
    step: int | None = None,
    label: str | None = None,
    tags: list[str] | None = None,
) -> TaskSpan[t.Any]:
    """
    Create a span for scorer execution.

    Args:
        name: The scorer name.
        score: The numeric score value.
        passed: Whether the score passed assertions.
        rationale: Explanation for the score.
        step: The step number (for agent context).
        label: Human-readable label.
        tags: Tags for filtering.

    Returns:
        A configured TaskSpan for scorer execution.
    """
    from dreadnode import task_span

    span = task_span(
        name=f"scorer:{name}",
        type="scorer",
        label=label or name,
        tags=["scorer", name, *(tags or [])],
    )

    span.set_attribute(SCORER_ATTRIBUTE_NAME, name)

    if score is not None:
        span.set_attribute(SCORER_ATTRIBUTE_SCORE, score)
    if passed is not None:
        span.set_attribute(SCORER_ATTRIBUTE_PASSED, passed)
    if rationale:
        span.set_attribute(SCORER_ATTRIBUTE_RATIONALE, rationale[:1000])
    if step is not None:
        span.set_attribute(SCORER_ATTRIBUTE_STEP, step)

    return span


def trial_span(
    trial_id: str,
    *,
    step: int,
    candidate: dict[str, t.Any],
    task_name: str | None = None,
    is_probe: bool = False,
    status: str | None = None,
    scores: dict[str, float] | None = None,
    label: str | None = None,
    tags: list[str] | None = None,
) -> TaskSpan[t.Any]:
    """
    Create a span for optimization trial.

    Args:
        trial_id: Unique trial identifier.
        step: Trial number in the study.
        candidate: The trial's hyperparameter candidate.
        task_name: Name of the task being evaluated.
        is_probe: Whether this is a probe trial.
        status: Trial status (running, finished, failed, pruned).
        scores: Objective scores for the trial.
        label: Human-readable label.
        tags: Tags for filtering.

    Returns:
        A configured TaskSpan for trial execution.
    """
    from dreadnode import task_span

    probe_or_trial = "probe" if is_probe else "trial"
    span = task_span(
        name=f"trial:{trial_id[:8]}",
        type="trial",
        label=label or (f"{task_name} [{step}]" if task_name else f"trial_{step}"),
        tags=[probe_or_trial, *(tags or [])],
    )

    span.set_attribute(TRIAL_ATTRIBUTE_ID, trial_id)
    span.set_attribute(TRIAL_ATTRIBUTE_STEP, step)
    span.set_attribute(TRIAL_ATTRIBUTE_CANDIDATE, candidate)
    span.set_attribute(TRIAL_ATTRIBUTE_IS_PROBE, is_probe)

    if status:
        span.set_attribute(TRIAL_ATTRIBUTE_STATUS, status)
    if scores:
        span.set_attribute(TRIAL_ATTRIBUTE_SCORES, scores)

    return span

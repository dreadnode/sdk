"""Explicit span factories for type-safe tracing."""

from __future__ import annotations

import json
import typing as t

from dreadnode.core.tracing.constants import (
    # Agent attributes
    AGENT_ATTRIBUTE_GOAL,
    AGENT_ATTRIBUTE_ID,
    AGENT_ATTRIBUTE_MODEL,
    AGENT_ATTRIBUTE_NAME,
    AGENT_ATTRIBUTE_SESSION_ID,
    AGENT_ATTRIBUTE_TOOLS,
    # Evaluation attributes
    EVALUATION_ATTRIBUTE_ASSERT_SCORES,
    EVALUATION_ATTRIBUTE_DATASET_SIZE,
    EVALUATION_ATTRIBUTE_ITERATIONS,
    EVALUATION_ATTRIBUTE_SCORERS,
    EVALUATION_ATTRIBUTE_TASK_NAME,
    # Generation attributes
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
    # Sample attributes
    SAMPLE_ATTRIBUTE_CONTEXT,
    SAMPLE_ATTRIBUTE_INDEX,
    SAMPLE_ATTRIBUTE_ITERATION,
    SAMPLE_ATTRIBUTE_SCENARIO_PARAMS,
    SAMPLE_ATTRIBUTE_TOTAL,
    # Scorer attributes
    SCORER_ATTRIBUTE_NAME,
    SCORER_ATTRIBUTE_PASSED,
    SCORER_ATTRIBUTE_RATIONALE,
    SCORER_ATTRIBUTE_SCORE,
    SCORER_ATTRIBUTE_STEP,
    # Study attributes
    STUDY_ATTRIBUTE_DIRECTIONS,
    STUDY_ATTRIBUTE_MAX_TRIALS,
    STUDY_ATTRIBUTE_OBJECTIVES,
    STUDY_ATTRIBUTE_SEARCH_STRATEGY,
    # Tool attributes
    TOOL_ATTRIBUTE_ARGUMENTS,
    TOOL_ATTRIBUTE_CALL_ID,
    TOOL_ATTRIBUTE_ERROR,
    TOOL_ATTRIBUTE_NAME,
    TOOL_ATTRIBUTE_RESULT,
    TOOL_ATTRIBUTE_STOPPED,
    # Trial attributes
    TRIAL_ATTRIBUTE_CANDIDATE,
    TRIAL_ATTRIBUTE_ID,
    TRIAL_ATTRIBUTE_IS_PROBE,
    TRIAL_ATTRIBUTE_SCORES,
    TRIAL_ATTRIBUTE_STATUS,
    TRIAL_ATTRIBUTE_STEP,
)

if t.TYPE_CHECKING:
    from dreadnode.core.tracing.span import TaskSpan
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


def evaluation_span(
    name: str,
    *,
    task_name: str,
    dataset_size: int,
    iterations: int = 1,
    scorers: list[str] | None = None,
    assert_scores: list[str] | None = None,
    label: str | None = None,
    tags: list[str] | None = None,
) -> TaskSpan[t.Any]:
    """
    Create a span for evaluation execution.

    Args:
        name: The evaluation name.
        task_name: Name of the task being evaluated.
        dataset_size: Number of samples in the dataset.
        iterations: Number of iterations per sample.
        scorers: List of scorer names applied.
        assert_scores: List of score assertions.
        label: Human-readable label.
        tags: Tags for filtering.

    Returns:
        A configured TaskSpan for evaluation.
    """
    from dreadnode import task_span

    span = task_span(
        name=f"evaluation:{name}",
        type="evaluation",
        label=label or name,
        tags=["evaluation", *(tags or [])],
    )

    span.set_attribute(EVALUATION_ATTRIBUTE_TASK_NAME, task_name)
    span.set_attribute(EVALUATION_ATTRIBUTE_DATASET_SIZE, dataset_size)
    span.set_attribute(EVALUATION_ATTRIBUTE_ITERATIONS, iterations)

    if scorers:
        span.set_attribute(EVALUATION_ATTRIBUTE_SCORERS, scorers)
    if assert_scores:
        span.set_attribute(EVALUATION_ATTRIBUTE_ASSERT_SCORES, assert_scores)

    return span


def agent_span(
    agent_id: str,
    *,
    session_id: str | None = None,
    agent_name: str | None = None,
    model: str | None = None,
    tools: list[str] | None = None,
    goal: str | None = None,
    label: str | None = None,
    tags: list[str] | None = None,
) -> TaskSpan[t.Any]:
    """
    Create a span for agent execution.

    Args:
        agent_id: Unique agent identifier (ULID string).
        session_id: Session identifier for multi-turn conversations.
        agent_name: Human-readable agent name.
        model: The model identifier used by the agent.
        tools: List of tool names available to the agent.
        goal: The goal/input provided to the agent.
        label: Human-readable label for the span.
        tags: Tags for filtering.

    Returns:
        A configured TaskSpan for agent execution.
    """
    from dreadnode import task_span

    display_name = agent_name or agent_id[:8]
    span = task_span(
        name=f"agent:{display_name}",
        type="agent",
        label=label or display_name,
        tags=["agent", *(tags or [])],
    )

    span.set_attribute(AGENT_ATTRIBUTE_ID, agent_id)

    if session_id:
        span.set_attribute(AGENT_ATTRIBUTE_SESSION_ID, session_id)
    if agent_name:
        span.set_attribute(AGENT_ATTRIBUTE_NAME, agent_name)
    if model:
        span.set_attribute(AGENT_ATTRIBUTE_MODEL, model)
    if tools:
        span.set_attribute(AGENT_ATTRIBUTE_TOOLS, tools)
    if goal:
        span.set_attribute(AGENT_ATTRIBUTE_GOAL, goal[:2000])

    return span


def study_span(
    name: str,
    *,
    search_strategy: str,
    objectives: list[str],
    max_trials: int,
    directions: list[str] | None = None,
    label: str | None = None,
    tags: list[str] | None = None,
) -> TaskSpan[t.Any]:
    """
    Create a span for optimization study execution.

    Args:
        name: The study name.
        search_strategy: The search strategy class name (e.g., "GridSearch").
        objectives: List of objective scorer names.
        max_trials: Maximum number of trials for the study.
        directions: Optimization directions ("maximize"/"minimize") per objective.
        label: Human-readable label.
        tags: Tags for filtering.

    Returns:
        A configured TaskSpan for study execution.
    """
    from dreadnode import task_span

    span = task_span(
        name=f"study:{name}",
        type="study",
        label=label or name,
        tags=["study", "optimization", *(tags or [])],
    )

    span.set_attribute(STUDY_ATTRIBUTE_SEARCH_STRATEGY, search_strategy)
    span.set_attribute(STUDY_ATTRIBUTE_OBJECTIVES, objectives)
    span.set_attribute(STUDY_ATTRIBUTE_MAX_TRIALS, max_trials)

    if directions:
        span.set_attribute(STUDY_ATTRIBUTE_DIRECTIONS, directions)

    return span


def sample_span(
    *,
    index: int,
    total: int,
    iteration: int = 1,
    scenario_params: dict[str, t.Any] | None = None,
    context: dict[str, t.Any] | None = None,
    label: str | None = None,
    tags: list[str] | None = None,
) -> TaskSpan[t.Any]:
    """
    Create a span for a single evaluation sample.

    Args:
        index: The sample index in the dataset (0-based).
        total: Total number of samples in the dataset.
        iteration: The iteration number (1-based).
        scenario_params: Parameters defining the scenario.
        context: Extra dataset fields not used as task inputs.
        label: Human-readable label.
        tags: Tags for filtering.

    Returns:
        A configured TaskSpan for sample execution.
    """
    from dreadnode import task_span

    span = task_span(
        name=f"sample:{index + 1}/{total}",
        type="sample",
        label=label or f"sample_{index + 1}",
        tags=["sample", *(tags or [])],
    )

    span.set_attribute(SAMPLE_ATTRIBUTE_INDEX, index)
    span.set_attribute(SAMPLE_ATTRIBUTE_TOTAL, total)
    span.set_attribute(SAMPLE_ATTRIBUTE_ITERATION, iteration)

    if scenario_params:
        span.set_attribute(SAMPLE_ATTRIBUTE_SCENARIO_PARAMS, scenario_params)
    if context:
        span.set_attribute(SAMPLE_ATTRIBUTE_CONTEXT, context)

    return span

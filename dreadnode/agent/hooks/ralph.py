import typing as t
from dataclasses import dataclass, field

from loguru import logger

from dreadnode.agent.events import AgentEvent, GenerationEnd, StepStart
from dreadnode.agent.reactions import Fail, Finish, Reaction, RetryWithFeedback
from dreadnode.scorers import Scorer, ScorerCallable, avg

if t.TYPE_CHECKING:
    from ulid import ULID

    from dreadnode.agent.hooks.base import Hook


@dataclass
class RalphState:
    """
    Tracks the state of a Ralph iteration loop for a single agent session.

    Attributes:
        iteration: Current iteration number (1-indexed).
        score_history: List of average scores from each iteration.
        last_step_seen: The last step number observed (for detecting new steps).
    """

    iteration: int = 0
    score_history: list[float] = field(default_factory=list)
    last_step_seen: int = -1

    def reset(self, step: int = -1) -> None:
        """Reset state to initial values."""
        self.iteration = 0
        self.score_history = []
        self.last_step_seen = step


def _is_completion_attempt(message: t.Any) -> bool:
    """
    Check if a generation looks like a completion attempt.

    A generation is considered a completion attempt if it has text content
    and no tool calls (meaning the agent is trying to provide a final answer).
    """
    # Check if message has content and no tool calls
    has_content = message.content and isinstance(message.content, str)
    has_no_tools = not message.tool_calls or len(message.tool_calls) == 0

    return has_content and has_no_tools


async def _score_output(
    output: str,
    scorer: Scorer[t.Any],
    iteration: int,
) -> float:
    """Score the output using the composed scorer."""
    try:
        metric = await scorer(output)
        score_value = float(metric.value)
        logger.debug(f"Ralph iteration {iteration}: score = {score_value:.3f}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Ralph hook: Scoring failed: {e}")
        score_value = 0.0

    return score_value


def _generate_feedback(
    iteration: int,
    max_iterations: int,
    current_score: float,
    min_score: float,
    feedback_template: str | None,
) -> str:
    """Generate feedback message for the next iteration."""
    if feedback_template:
        return feedback_template.format(
            iteration=iteration,
            max_iterations=max_iterations,
            current_score=current_score,
            min_score=min_score,
        )

    return (
        f"Iteration {iteration}/{max_iterations}: Score {current_score:.3f} "
        f"(target: {min_score:.3f})\n\n"
        f"Your output did not meet the quality threshold. "
        f"Review and improve your work."
    )


def ralph_hook(
    completion_scorers: list[Scorer[t.Any]] | list[ScorerCallable[t.Any]],
    *,
    min_score: float = 0.8,
    max_iterations: int = 10,
    feedback_template: str | None = None,
) -> "Hook":
    """
    Create a hook that implements iterative agent refinement based on scorer thresholds.

    Intercepts agent generations and scores final answers (non-tool-calling responses).
    When score is below threshold, provides feedback and retries. Continues until
    minimum score achieved or max iterations reached.

    Args:
        completion_scorers: Scorers to evaluate output. Multiple scorers are averaged.
        min_score: Minimum score (0.0-1.0) to accept output.
        max_iterations: Maximum retry attempts before failure.
        feedback_template: Optional feedback template with {iteration}, {max_iterations},
            {current_score}, {min_score} placeholders.

    Returns:
        Hook that implements iteration logic.

    Raises:
        ValueError: If max_iterations <= 0 or min_score not in [0.0, 1.0].

    Example:
        >>> hook = ralph_hook(
        ...     completion_scorers=[contains(["critical"]), length_in_range(min_length=100)],
        ...     min_score=0.9,
        ...     max_iterations=15
        ... )
        >>> agent = dn.Agent(instructions="...", tools=[...], hooks=[hook])
    """
    if max_iterations <= 0:
        msg = f"max_iterations must be > 0, got {max_iterations}"
        raise ValueError(msg)

    if not 0.0 <= min_score <= 1.0:
        msg = f"min_score must be in [0.0, 1.0], got {min_score}"
        raise ValueError(msg)

    # Compose scorers into single averaged scorer with error catching
    scorers: list[Scorer[t.Any]] = [
        (s if isinstance(s, Scorer) else Scorer(s)).with_(catch=True) for s in completion_scorers
    ]
    composed_scorer = avg(*scorers) if len(scorers) > 1 else scorers[0]

    # Session-based state tracking
    session_states: dict[ULID, RalphState] = {}

    async def ralph_iteration_hook(event: AgentEvent) -> Reaction | None:
        """Hook implementation that handles Ralph iteration logic."""
        state = session_states.setdefault(event.session_id, RalphState())

        # Reset state on new step (agent progressed naturally)
        if isinstance(event, StepStart):
            if event.step > state.last_step_seen:
                state.reset(event.step)
            return None

        # Only intercept GenerationEnd events with valid completion attempts
        if not isinstance(event, GenerationEnd) or not _is_completion_attempt(event.message):
            return None

        state.iteration += 1

        # Extract output text for scoring
        output = event.message.content
        if not output or not isinstance(output, str):
            logger.warning(
                f"Ralph hook: No text content in generation for session {event.session_id}"
            )
            return None

        # Score the output
        score_value = await _score_output(output, composed_scorer, state.iteration)
        state.score_history.append(score_value)

        logger.info(
            f"Ralph iteration {state.iteration}/{max_iterations}: "
            f"score {score_value:.3f} (target: {min_score:.3f})"
        )

        # Check convergence
        if score_value >= min_score:
            logger.success(
                f"Ralph hook: Converged after {state.iteration} iteration(s) "
                f"(score: {score_value:.3f} >= {min_score:.3f})"
            )
            session_states.pop(event.session_id, None)
            return Finish(reason=f"Ralph loop converged (score: {score_value:.3f})")

        # Check max iterations
        if state.iteration >= max_iterations:
            best_score = max(state.score_history) if state.score_history else 0.0
            logger.warning(
                f"Ralph hook: Max iterations ({max_iterations}) reached without convergence. "
                f"Best score: {best_score:.3f} (target: {min_score:.3f})"
            )
            session_states.pop(event.session_id, None)
            return Fail(
                f"Ralph loop did not converge after {max_iterations} iterations. "
                f"Best score: {best_score:.3f} (target: {min_score:.3f})"
            )

        # Generate feedback and retry
        feedback = _generate_feedback(
            state.iteration, max_iterations, score_value, min_score, feedback_template
        )
        return RetryWithFeedback(feedback=feedback)

    return ralph_iteration_hook

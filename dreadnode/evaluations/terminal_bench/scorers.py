"""Scorers for Terminal-Bench evaluation."""

from dataclasses import dataclass

from dreadnode.core.meta import DatasetField
from dreadnode.core.scorer import scorer


@dataclass
class TerminalResult:
    """Result of a terminal task execution."""

    trajectory_steps: int
    test_exit_code: int
    test_output: str
    success: bool


@scorer(name="task_success")
async def task_success_scorer(
    result: TerminalResult,
) -> float:
    """
    Score based on whether the test script passed.

    Returns 1.0 if test script exited with code 0, else 0.0.
    """
    return 1.0 if result.success else 0.0


@scorer(name="efficiency")
async def efficiency_scorer(
    result: TerminalResult,
) -> float:
    """
    Score based on how efficiently the agent completed the task.

    Rewards fewer steps, penalizes excessive steps.
    """
    steps = result.trajectory_steps

    if not result.success:
        return 0.0

    # Optimal: 1-5 steps
    if steps <= 5:
        return 1.0
    # Good: 6-10 steps
    if steps <= 10:
        return 0.8
    # Acceptable: 11-15 steps
    if steps <= 15:
        return 0.6
    # Inefficient: 16-20 steps
    if steps <= 20:
        return 0.4
    # Very inefficient: 20+ steps
    return max(0.1, 0.4 - (steps - 20) * 0.02)


@scorer(name="terminal_composite")
async def terminal_composite_scorer(
    result: TerminalResult,
    difficulty: str = DatasetField("difficulty"),
) -> list:
    """
    Composite score weighing success and efficiency.

    Difficulty affects weighting:
    - easy: 70% success, 30% efficiency
    - medium: 80% success, 20% efficiency
    - hard: 90% success, 10% efficiency

    Returns list of metrics to include child scorer metrics.
    """

    success_value = 1.0 if result.success else 0.0

    # Use score_composite to get the Metric and extract value
    efficiency_metric, _ = await efficiency_scorer.score_composite(result)
    efficiency_value = efficiency_metric.value

    weights = {
        "easy": (0.7, 0.3),
        "medium": (0.8, 0.2),
        "hard": (0.9, 0.1),
    }

    success_weight, efficiency_weight = weights.get(difficulty, (0.8, 0.2))

    composite_value = success_value * success_weight + efficiency_value * efficiency_weight

    # Return primary metric plus child metrics for logging
    return [composite_value, efficiency_metric]

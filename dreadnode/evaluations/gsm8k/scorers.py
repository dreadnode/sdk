"""GSM8K scorers for evaluating math problem solutions."""

import json

import dreadnode as dn
from dreadnode.core.agents.events import ToolStep
from dreadnode.core.agents.trajectory import Trajectory
from dreadnode.core.meta import DatasetField
from dreadnode.core.scorer import scorer, weighted_avg


def extract_submitted_answer(trajectory: Trajectory) -> float | None:
    """Extract the submitted answer from the agent's trajectory."""
    for step in reversed(trajectory.steps):
        if isinstance(step, ToolStep) and step.tool_call.name == "submit_answer":
            try:
                args = step.tool_call.arguments
                # Arguments can be a JSON string or a dict
                if isinstance(args, str):
                    args = json.loads(args)
                if isinstance(args, dict) and "answer" in args:
                    return float(args["answer"])
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
    return None


@scorer(name="answer_correct")
async def answer_correct_scorer(
    trajectory: Trajectory,
    expected_answer: float = DatasetField("numeric_answer"),
) -> float:
    """
    Score whether the agent's submitted answer matches the expected answer.

    Uses DatasetField to automatically pull the expected answer from the dataset.
    Returns 1.0 for correct, 0.0 for incorrect or no answer.
    """
    submitted = extract_submitted_answer(trajectory)
    if submitted is None:
        return 0.0

    tolerance = abs(expected_answer) * 0.001 if expected_answer != 0 else 0.001
    return 1.0 if abs(submitted - expected_answer) <= tolerance else 0.0


@scorer(name="reasoning_quality")
async def reasoning_quality_scorer(trajectory: Trajectory) -> float:
    """
    Score the quality of the agent's reasoning process.

    Rewards using the calculate tool and submitting an answer.
    """
    calculation_count = 0
    has_answer = False

    for step in trajectory.steps:
        if isinstance(step, ToolStep):
            if step.tool_call.name == "calculate":
                calculation_count += 1
            elif step.tool_call.name == "submit_answer":
                has_answer = True

    calc_score = min(1.0, calculation_count / 5.0)
    answer_score = 1.0 if has_answer else 0.0

    return calc_score * 0.5 + answer_score * 0.5


@scorer(name="efficiency")
async def efficiency_scorer(trajectory: Trajectory) -> float:
    """
    Score the efficiency of the agent's problem-solving.

    Rewards concise solutions, penalizes overly long ones.
    """
    step_count = len(trajectory.steps)

    if step_count < 3:
        return 0.5
    if step_count <= 8:
        return 1.0
    if step_count <= 15:
        return 0.7
    return max(0.2, 1.0 - (step_count - 15) * 0.05)


# Composite scorer using the library's built-in weighted_avg
gsm8k_composite_scorer = weighted_avg(
    (answer_correct_scorer, 0.6),
    (reasoning_quality_scorer, 0.3),
    (efficiency_scorer, 0.1),
    name="gsm8k_composite",
)

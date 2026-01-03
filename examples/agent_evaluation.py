"""Agent Evaluation Example - Evaluate Agents with Benchmarks.

This example demonstrates:
- Creating an agent with tools
- Converting agent to task with agent.as_task()
- Evaluating agent on a dataset
- Custom scorers for agent trajectories

Run with:
    python examples/agent_evaluation.py

Note: Requires a configured LLM provider (e.g., GROQ_API_KEY).
"""

import asyncio
import re

import dreadnode as dn
from dreadnode.core.agents.stopping import tool_use


# Define tools
@dn.tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters"
        return str(eval(expression))  # noqa: S307
    except Exception as e:
        return f"Error: {e}"


@dn.tool
def submit_answer(answer: float) -> str:
    """Submit the final answer."""
    return f"SUBMITTED: {answer}"


# Create the agent to evaluate
agent = dn.Agent(
    name="MathAgent",
    model="groq/llama-3.3-70b-versatile",
    instructions="Solve math problems. Use calculate() for arithmetic, then submit_answer() with the result.",
    tools=[calculate, submit_answer],
    stop_conditions=[tool_use("submit_answer")],
    max_steps=8,
)


# Define scorers for evaluation
@dn.scorer
def answer_correct(trajectory, expected: float) -> float:
    """Check if the agent's submitted answer matches expected.

    Args:
        trajectory: The agent's execution trajectory.
        expected: The expected numerical answer.

    Returns:
        1.0 if correct, 0.0 otherwise.
    """
    # Convert trajectory to string and look for SUBMITTED pattern
    text = str(trajectory)
    match = re.search(r"SUBMITTED:\s*(-?[\d.]+)", text)

    if match:
        submitted = float(match.group(1))
        return 1.0 if abs(submitted - expected) < 0.01 else 0.0
    return 0.0


@dn.scorer
def used_calculator(trajectory) -> float:
    """Check if the agent used the calculator tool."""
    text = str(trajectory)
    return 1.0 if "calculate" in text.lower() else 0.0


@dn.scorer
def efficiency(trajectory) -> float:
    """Score based on number of steps (fewer is better)."""
    # Prefer solutions in 3-4 steps
    num_steps = len(trajectory.steps) if hasattr(trajectory, "steps") else 5
    if num_steps <= 3:
        return 1.0
    elif num_steps <= 5:
        return 0.7
    elif num_steps <= 7:
        return 0.4
    return 0.1


# Create evaluation
@dn.evaluation(
    dataset=[
        {"question": "What is 5 + 3?", "expected": 8.0},
        {"question": "What is 10 * 4?", "expected": 40.0},
        {"question": "What is 100 / 5?", "expected": 20.0},
        {"question": "What is 7 + 8 - 3?", "expected": 12.0},
        {"question": "What is 25 * 4?", "expected": 100.0},
    ],
    dataset_input_mapping={"question": "goal"},  # Map "question" to agent's "goal" param
    scorers=[answer_correct, used_calculator, efficiency],
    concurrency=2,
)
async def evaluate_math_agent(goal: str):
    """Run the agent on a math problem."""
    return await agent.run(goal)


async def main():
    dn.configure(server="local")

    print("Agent Evaluation Example")
    print("=" * 50)
    print("\nEvaluating math agent on benchmark...")

    # Run evaluation
    result = await evaluate_math_agent.run()

    # Print results
    print(f"\nResults:")
    print(f"  Accuracy: {result.metrics_aggregated.get('answer_correct', 0):.1%}")
    print(f"  Used calculator: {result.metrics_aggregated.get('used_calculator', 0):.1%}")
    print(f"  Efficiency: {result.metrics_aggregated.get('efficiency', 0):.2f}")

    print(f"\nPer-sample results:")
    for sample in result.samples:
        status = "PASS" if sample.passed else "FAIL"
        question = sample.input.get("question", "?")[:40]
        print(f"  [{status}] {question}")

    # Summary
    print(f"\nSummary:")
    print(f"  Total: {len(result.samples)}")
    print(f"  Passed: {sum(1 for s in result.samples if s.passed)}")
    print(f"  Failed: {sum(1 for s in result.samples if not s.passed)}")


if __name__ == "__main__":
    asyncio.run(main())

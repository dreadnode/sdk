"""
GSM8K Evaluation

This module provides a complete GSM8K evaluation using the Evaluation class.
The Agent.as_task() method bridges Agent and Evaluation cleanly.
"""

from dreadnode.agents import Agent
from dreadnode.core.agents.trajectory import Trajectory
from dreadnode.evaluations import Evaluation
from dreadnode.evaluations.gsm8k.agent import create_math_agent
from dreadnode.evaluations.gsm8k.dataset import load_gsm8k_sample
from dreadnode.evaluations.gsm8k.scorers import (
    answer_correct_scorer,
    efficiency_scorer,
    gsm8k_composite_scorer,
    reasoning_quality_scorer,
)


def create_gsm8k_evaluation(
    agent: Agent | None = None,
    *,
    concurrency: int = 2,
    iterations: int = 1,
    use_composite_scorer: bool = True,
) -> Evaluation[str, Trajectory]:
    """
    Create a GSM8K evaluation.

    Uses Agent.as_task() to bridge the agent with the Evaluation system.

    Args:
        agent: The agent to evaluate. If None, creates a default math agent.
        concurrency: Number of concurrent evaluations.
        iterations: Number of iterations per problem.
        use_composite_scorer: Whether to use the composite scorer or individual scorers.

    Returns:
        An Evaluation instance ready to run.

    Example:
        ```python
        # Using default agent
        evaluation = create_gsm8k_evaluation()
        result = await evaluation.run()

        # Using custom agent
        my_agent = create_math_agent(model="openai/gpt-4o")
        evaluation = create_gsm8k_evaluation(agent=my_agent)
        result = await evaluation.run()
        ```
    """
    if agent is None:
        agent = create_math_agent()

    # Select scorers
    if use_composite_scorer:
        scorers = [gsm8k_composite_scorer]
    else:
        scorers = [answer_correct_scorer, reasoning_quality_scorer, efficiency_scorer]

    return Evaluation(
        name="GSM8K Evaluation",
        description="Evaluate math problem solving on the GSM8K benchmark.",
        task=agent.as_task(),
        dataset=load_gsm8k_sample(),
        # Map dataset field "question" to task parameter "goal"
        dataset_input_mapping={"question": "goal"},
        scorers=scorers,
        assert_scores=["answer_correct"] if not use_composite_scorer else [],
        concurrency=concurrency,
        iterations=iterations,
        tags=["math", "gsm8k", "benchmark"],
    )


# Create a default evaluation instance for CLI discovery
GSM8KEvaluation = create_gsm8k_evaluation()


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        evaluation = create_gsm8k_evaluation()
        result = await evaluation.run()
        print(f"Pass rate: {result.pass_rate:.1%}")
        print(f"Samples: {len(result.samples)}")

    asyncio.run(main())

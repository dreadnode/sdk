"""Study Example - Hyperparameter Optimization.

This example demonstrates:
- dn.study() - Hyperparameter optimization decorator
- Search strategies - Grid search, random search
- Objectives - Metrics to optimize
- Directions - Maximize or minimize
- Constraints - Hard constraints on candidates

Run with:
    python examples/study.py
"""

import asyncio

import dreadnode as dn

# Import search strategies
try:
    from dreadnode.search import grid
except ImportError:
    # Fallback if search module not available
    def grid(params: dict) -> list[dict]:
        """Simple grid search implementation."""
        import itertools
        keys = list(params.keys())
        values = list(params.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


# Define objective scorers
@dn.scorer
def accuracy(prediction: str, expected: str) -> float:
    """Check if prediction matches expected answer."""
    pred = prediction.strip().lower()
    exp = expected.strip().lower()
    return 1.0 if pred == exp else 0.0


@dn.scorer
def efficiency(prediction: str) -> float:
    """Shorter, more concise responses are more efficient."""
    length = len(prediction)
    if length <= 20:
        return 1.0
    elif length <= 50:
        return 0.7
    elif length <= 100:
        return 0.4
    return 0.1


# Define the study with hyperparameter search
@dn.study(
    search_strategy=grid({
        "temperature": [0.0, 0.5, 1.0],
        "style": ["concise", "verbose"],
    }),
    dataset=[
        {"question": "What is 2+2?", "expected": "4"},
        {"question": "Capital of France?", "expected": "paris"},
        {"question": "Largest planet?", "expected": "jupiter"},
    ],
    objectives=[accuracy, efficiency],
    directions=["maximize", "maximize"],
    max_trials=6,  # 3 temps x 2 styles = 6 combinations
    concurrency=2,
)
def create_qa_task(candidate: dict):
    """Task factory that creates a task with the given hyperparameters.

    Args:
        candidate: Dict with "temperature" and "style" keys.

    Returns:
        A task function configured with the candidate's parameters.
    """
    temperature = candidate["temperature"]
    style = candidate["style"]

    @dn.task
    async def answer_question(question: str) -> str:
        """Answer a question with the configured style."""
        await asyncio.sleep(0.05)  # Simulate API call

        # Simulate different behaviors based on parameters
        answers = {
            "What is 2+2?": "4",
            "Capital of France?": "Paris",
            "Largest planet?": "Jupiter",
        }

        base_answer = answers.get(question, "unknown")

        if style == "verbose":
            return f"The answer to '{question}' is: {base_answer}. I hope this helps!"
        else:  # concise
            return base_answer

    return answer_question


async def main():
    dn.configure(server="local")

    print("Running hyperparameter study...")
    print("-" * 50)

    # Run the study
    result = await create_qa_task.run()

    # Print best trial
    if result.best_trial:
        print(f"\nBest trial:")
        print(f"  Candidate: {result.best_trial.candidate}")
        print(f"  Score: {result.best_trial.score:.3f}")
        print(f"  All scores: {result.best_trial.scores}")

    # Print all trials
    print(f"\nAll trials ({len(result.trials)}):")
    for trial in sorted(result.trials, key=lambda t: t.score, reverse=True):
        print(f"  {trial.candidate} -> score={trial.score:.3f}")

    # Export results
    # result.to_jsonl("study_results.jsonl")
    # df = result.to_dataframe()


if __name__ == "__main__":
    asyncio.run(main())

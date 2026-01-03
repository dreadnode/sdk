"""Evaluation Example - Batch Evaluation with Scorers.

This example demonstrates:
- Evaluation class - Batch evaluation
- Dataset loading - Inline list or from file
- Scorers - Automatic scoring of outputs
- Concurrency - Parallel evaluation

Run with:
    python examples/evaluation.py
"""

import asyncio

import dreadnode as dn
from dreadnode.evaluations import Evaluation


# Define scorers for evaluation
@dn.scorer
def has_greeting(response: str) -> float:
    """Check if response contains a greeting."""
    greetings = ["hello", "hi", "hey", "greetings", "welcome"]
    response_lower = response.lower()
    return 1.0 if any(g in response_lower for g in greetings) else 0.0


@dn.scorer
def appropriate_length(response: str) -> float:
    """Check if response is appropriate length (50-200 chars ideal)."""
    length = len(response)
    if 50 <= length <= 200:
        return 1.0
    elif 20 <= length < 50 or 200 < length <= 300:
        return 0.5
    return 0.0


@dn.scorer
def no_errors(response: str) -> float:
    """Check response doesn't contain error messages."""
    error_indicators = ["error", "sorry", "cannot", "unable", "failed"]
    response_lower = response.lower()
    return 0.0 if any(e in response_lower for e in error_indicators) else 1.0


# Define the task to evaluate
@dn.task
async def respond(prompt: str) -> str:
    """Generate a response to a prompt.

    In a real application, this would call an LLM.
    """
    await asyncio.sleep(0.05)  # Simulate API latency
    return f"Hello! In response to '{prompt}': This is a helpful and friendly answer that addresses your question."


async def main():
    dn.configure(server="local")

    print("Running evaluation...")
    print("-" * 50)

    # Create evaluation using the Evaluation class
    evaluation = Evaluation(
        name="Response Evaluation",
        description="Evaluate response quality",
        task=respond,
        dataset=[
            {"prompt": "How are you today?"},
            {"prompt": "What's the weather like?"},
            {"prompt": "Tell me a joke"},
            {"prompt": "What time is it?"},
            {"prompt": "Help me with my homework"},
        ],
        scorers=[has_greeting, appropriate_length, no_errors],
        concurrency=2,
        iterations=1,
    )

    # Run the evaluation
    result = await evaluation.run()

    # Print summary
    print(f"\nResults:")
    print(f"  Pass rate: {result.pass_rate:.1%}")
    print(f"  Total samples: {len(result.samples)}")

    # Print per-scorer metrics
    if result.metrics_aggregated:
        print(f"\nMetrics (aggregated):")
        for name, value in result.metrics_aggregated.items():
            print(f"  {name}: {value:.3f}")

    # Print individual samples
    print(f"\nSamples:")
    for sample in result.samples:
        status = "PASS" if sample.passed else "FAIL"
        input_str = str(sample.input)[:40] if sample.input else "?"
        print(f"  [{status}] {input_str}...")


if __name__ == "__main__":
    asyncio.run(main())

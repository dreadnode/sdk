"""Quickstart Example - SDK Basics.

This example demonstrates the core Dreadnode SDK primitives:
- dn.configure() - Configure the SDK
- dn.run() - Create a tracked run
- dn.task() - Define a tracked task
- dn.scorer() - Define a scorer function
- dn.log_metric() / dn.log_params() - Log metrics and parameters

Run with:
    python examples/quickstart.py
"""

import asyncio

import dreadnode as dn


# Define a scorer to evaluate task outputs
@dn.scorer
def length_score(text: str) -> float:
    """Score based on response length.

    Returns a score between 0 and 1 based on text length.
    Longer responses (up to 100 chars) get higher scores.
    """
    return min(len(text) / 100, 1.0)


@dn.scorer
def has_punctuation(text: str) -> float:
    """Check if text has proper punctuation."""
    return 1.0 if text.rstrip().endswith((".", "!", "?")) else 0.0


# Define a task (scorers can be added via @dn.task(scorers=[...]))
@dn.task
async def generate_story(topic: str) -> str:
    """Generate a short story about a topic.

    In a real application, this would call an LLM.
    """
    # Simulate LLM generation
    await asyncio.sleep(0.1)
    return f"Once upon a time, there was a brave {topic} who went on an adventure. After many trials, the {topic} found what they were looking for!"


async def main():
    # Configure the SDK (use "local" for local-only mode, no server needed)
    dn.configure(server="local")

    # Create a tracked run
    with dn.run("story-generation", params={"version": "1.0", "model": "mock"}):
        # Log additional parameters
        dn.log_params(temperature=0.7, max_tokens=100)

        # Run tasks within the run context
        topics = ["dragon", "robot", "wizard"]

        for i, topic in enumerate(topics):
            result = await generate_story(topic)

            # Log metrics for this iteration
            dn.log_metric("story_length", len(result), step=i)
            dn.log_metric("word_count", len(result.split()), step=i)

            # Manually apply scorers and log their results
            dn.log_metric("length_score", await length_score(result), step=i)
            dn.log_metric("punctuation_score", await has_punctuation(result), step=i)

            print(f"[{topic}] Generated {len(result)} chars")

        print("\nRun completed!")


if __name__ == "__main__":
    asyncio.run(main())

"""Full Workflow Example - End-to-End Pipeline.

This example demonstrates a complete workflow:
1. Create an agent with tools
2. Evaluate its baseline performance
3. Train the underlying model with GRPO
4. Re-evaluate to measure improvement

Run with:
    python examples/full_workflow.py

Note: Requires GPU and configured LLM provider.
"""

import os

# Set environment before imports
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import asyncio
import re

import dreadnode as dn
from dreadnode.core.agents.stopping import tool_use

from examples.utils import extract_gsm8k_answer, load_gsm8k


# =============================================================================
# Step 1: Define Tools
# =============================================================================


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


# =============================================================================
# Step 2: Create Agent
# =============================================================================


def create_math_agent(model: str = "groq/llama-3.3-70b-versatile") -> dn.Agent:
    """Create a math-solving agent."""
    return dn.Agent(
        name="MathAgent",
        model=model,
        instructions="""You are a math problem solver.

1. Read the problem carefully
2. Use calculate() for arithmetic
3. Use submit_answer() with the final number

Be concise and accurate.""",
        tools=[calculate, submit_answer],
        stop_conditions=[tool_use("submit_answer")],
        max_steps=8,
    )


# =============================================================================
# Step 3: Define Scorers
# =============================================================================


@dn.scorer
def answer_correct(trajectory, expected: float) -> float:
    """Check if submitted answer matches expected."""
    text = str(trajectory)
    match = re.search(r"SUBMITTED:\s*(-?[\d.]+)", text)
    if match:
        submitted = float(match.group(1))
        return 1.0 if abs(submitted - expected) < 0.01 else 0.0
    return 0.0


@dn.scorer
def has_reasoning(completion: str) -> float:
    """Reward showing reasoning steps."""
    keywords = ["step", "first", "then", "because", "calculate"]
    matches = sum(1 for k in keywords if k in completion.lower())
    return min(matches / 3, 1.0)


@dn.scorer
def has_answer_format(completion: str) -> float:
    """Reward proper answer format."""
    if re.search(r"(answer|result|=)\s*\d+", completion, re.I):
        return 1.0
    return 0.0


# =============================================================================
# Step 4: Evaluation Function
# =============================================================================


async def evaluate_agent(agent: dn.Agent, dataset: list[dict], name: str) -> dict:
    """Evaluate an agent on a dataset.

    Args:
        agent: The agent to evaluate.
        dataset: List of dicts with "question" and "answer" keys.
        name: Name for the evaluation.

    Returns:
        Dict with accuracy and metrics.
    """
    print(f"\nEvaluating: {name}")
    print("-" * 40)

    correct = 0
    total = len(dataset)

    for i, item in enumerate(dataset):
        question = item["question"]
        expected = item["answer"]

        try:
            trajectory = await agent.run(question)
            text = str(trajectory)

            # Check if answer is correct
            match = re.search(r"SUBMITTED:\s*(-?[\d.]+)", text)
            if match:
                submitted = float(match.group(1))
                if abs(submitted - expected) < 0.01:
                    correct += 1
                    status = "CORRECT"
                else:
                    status = f"WRONG (got {submitted}, expected {expected})"
            else:
                status = "NO ANSWER"
        except Exception as e:
            status = f"ERROR: {e}"

        print(f"  [{i + 1}/{total}] {status}")

    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {accuracy:.1%} ({correct}/{total})")

    return {"accuracy": accuracy, "correct": correct, "total": total}


# =============================================================================
# Step 5: Training Function
# =============================================================================


def train_model(dataset: list[dict], model_name: str, max_steps: int = 50):
    """Train a model with GRPO.

    Args:
        dataset: Training dataset with questions.
        model_name: Model to train.
        max_steps: Number of training steps.
    """
    print(f"\nTraining model: {model_name}")
    print(f"Steps: {max_steps}")
    print("-" * 40)

    # Prepare prompts
    prompts = [
        f"Solve this math problem: {item['question']}\n\nAnswer:"
        for item in dataset
    ]

    # Train with GRPO
    result = dn.train(
        {
            "trainer": "grpo",
            "model_name": model_name,
            "max_steps": max_steps,
            "num_prompts_per_step": 4,
            "num_generations_per_prompt": 4,
            "learning_rate": 1e-6,
            "temperature": 0.7,
            "checkpoint_dir": "./checkpoints/full_workflow",
        },
        prompts=prompts,
        scorers=[has_reasoning, has_answer_format],
    )

    print(f"\nTraining complete!")
    return result


# =============================================================================
# Main Workflow
# =============================================================================


async def main():
    dn.configure(server="local")

    print("=" * 60)
    print("Full Workflow Example")
    print("Agent → Evaluate → Train → Re-evaluate")
    print("=" * 60)

    # Load datasets
    print("\n[1/5] Loading datasets...")
    train_data = load_gsm8k(split="train", max_samples=100)
    eval_data = load_gsm8k(split="test", max_samples=20)
    print(f"  Train: {len(train_data)} samples")
    print(f"  Eval: {len(eval_data)} samples")

    # Create agent
    print("\n[2/5] Creating agent...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Use a smaller model for training
    agent = create_math_agent(model=f"local/{model_name}")
    print(f"  Agent: {agent.name}")
    print(f"  Model: {model_name}")

    # Baseline evaluation
    print("\n[3/5] Baseline evaluation...")
    with dn.run("baseline-eval"):
        baseline_metrics = await evaluate_agent(agent, eval_data[:5], "Baseline")
        dn.log_metric("baseline_accuracy", baseline_metrics["accuracy"])

    # Training
    print("\n[4/5] Training with GRPO...")
    with dn.run("training"):
        train_result = train_model(train_data, model_name, max_steps=20)
        dn.log_params(model=model_name, steps=20)

    # Post-training evaluation
    print("\n[5/5] Post-training evaluation...")
    with dn.run("post-training-eval"):
        # Note: In a real scenario, you'd load the trained checkpoint
        post_metrics = await evaluate_agent(agent, eval_data[:5], "Post-training")
        dn.log_metric("post_accuracy", post_metrics["accuracy"])

    # Summary
    print("\n" + "=" * 60)
    print("Workflow Complete!")
    print("=" * 60)
    print(f"\nBaseline accuracy: {baseline_metrics['accuracy']:.1%}")
    print(f"Post-training accuracy: {post_metrics['accuracy']:.1%}")

    improvement = post_metrics["accuracy"] - baseline_metrics["accuracy"]
    if improvement > 0:
        print(f"Improvement: +{improvement:.1%}")
    else:
        print(f"Change: {improvement:.1%}")


if __name__ == "__main__":
    asyncio.run(main())

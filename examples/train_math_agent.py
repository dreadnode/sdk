#!/usr/bin/env python3
"""
Train a math reasoning agent using GRPO.

This example demonstrates the full workflow:
1. Create an agent with tools (calculator, submit_answer)
2. Load a math dataset (GSM8K)
3. Define reward functions based on correctness
4. Train using Ray-based GRPO
5. Evaluate improvement

Usage:
    # Quick test (10 steps)
    python examples/train_math_agent.py --quick

    # Full training
    python examples/train_math_agent.py --steps 500

    # With specific model
    python examples/train_math_agent.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

import os
import re
import argparse
import asyncio
from dataclasses import dataclass

# Set environment before CUDA imports
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch
from datasets import load_dataset

# Dreadnode imports
import dreadnode as dn
from dreadnode.agents import Agent
from dreadnode.core.agents.stopping import tool_use


# =============================================================================
# Tools
# =============================================================================


@dn.tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Args:
        expression: Math expression like "2 + 3 * 4" or "(10 - 5) / 2"

    Returns:
        The result of the calculation.
    """
    try:
        # Only allow safe characters
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return f"Error: Invalid characters in '{expression}'"

        result = eval(expression)  # noqa: S307

        # Format nicely
        if isinstance(result, float) and result.is_integer():
            result = int(result)

        return f"{result}"
    except Exception as e:
        return f"Error: {e}"


@dn.tool
def submit_answer(answer: float) -> str:
    """
    Submit your final answer to the math problem.

    Args:
        answer: The numeric answer to submit.

    Returns:
        Confirmation message.
    """
    return f"SUBMITTED: {answer}"


# =============================================================================
# Agent Definition
# =============================================================================


AGENT_INSTRUCTIONS = """You are a math problem solver. Solve step by step.

RULES:
1. Break down the problem into steps
2. Use calculate() for each arithmetic operation
3. Show your reasoning
4. Use submit_answer() with the final numeric answer

Example:
Problem: If John has 5 apples and buys 3 more, how many does he have?
Thought: John starts with 5 apples and gets 3 more. I need to add them.
Action: calculate("5 + 3")
Result: 8
Thought: The answer is 8.
Action: submit_answer(8)
"""


def create_agent(model: str) -> Agent:
    """Create the math agent."""
    return Agent(
        name="MathAgent",
        description="Solves math word problems step by step",
        model=model,
        instructions=AGENT_INSTRUCTIONS,
        max_steps=10,
        tools=[calculate, submit_answer],
        stop_conditions=[tool_use("submit_answer")],
    )


# =============================================================================
# Dataset & Rewards
# =============================================================================


def extract_answer(text: str) -> float | None:
    """Extract numeric answer from text."""
    # GSM8K format: #### <number>
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass

    # Try "answer is X" format
    match = re.search(r"answer\s+is\s+(-?[\d,]+(?:\.\d+)?)", text, re.I)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass

    # Last number in text
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass

    return None


def extract_submitted_answer(completion: str) -> float | None:
    """Extract the answer submitted by the agent."""
    # Look for submit_answer call with various formats
    patterns = [
        r"submit_answer\s*\(\s*(-?[\d.]+)\s*\)",  # submit_answer(42)
        r"submit_answer\s*\(\s*\"?\s*(-?[\d.]+)\s*\"?\s*\)",  # submit_answer("42")
        r"SUBMITTED:\s*(-?[\d.]+)",  # SUBMITTED: 42
        r"final answer[:\s]+(-?[\d.]+)",  # final answer: 42
        r"answer is[:\s]+(-?[\d.]+)",  # answer is 42
        r"\*\*(-?[\d.]+)\*\*",  # **42** (markdown bold)
        r"####\s*(-?[\d.]+)",  # #### 42 (GSM8K format)
    ]

    for pattern in patterns:
        match = re.search(pattern, completion, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    # Last resort: find the last number in the completion
    numbers = re.findall(r"(?<![a-zA-Z])(-?\d+\.?\d*)(?![a-zA-Z])", completion)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass

    return None


TOOL_PROMPT_TEMPLATE = """You are a math problem solver with access to a calculator.

Available tools:
- calculate(expression): Evaluate a math expression like "2 + 3 * 4"
- submit_answer(answer): Submit your final numeric answer

You MUST use submit_answer(X) to provide your final answer, where X is a number.

Example:
Problem: If John has 5 apples and buys 3 more, how many does he have?

Let me solve this step by step.
First, I'll calculate: calculate("5 + 3") = 8
The answer is 8.
submit_answer(8)
SUBMITTED: 8

Now solve this problem:

Problem: {question}

"""


def load_gsm8k_prompts(split: str = "train", max_samples: int = 1000) -> list[dict]:
    """Load GSM8K dataset and format as prompts."""
    dataset = load_dataset("openai/gsm8k", "main", split=split)

    prompts = []
    for i, item in enumerate(dataset):
        if i >= max_samples:
            break

        prompts.append({
            "prompt": TOOL_PROMPT_TEMPLATE.format(question=item['question']),
            "answer": extract_answer(item["answer"]),
            "raw_answer": item["answer"],
        })

    return prompts


@dataclass
class MathRewardConfig:
    """Configuration for math rewards."""
    correct_reward: float = 1.0
    incorrect_reward: float = -0.5
    no_answer_reward: float = -1.0
    partial_credit: bool = True
    tolerance: float = 0.01


def create_reward_fn(
    dataset: list[dict],
    config: MathRewardConfig | None = None
):
    """Create reward function for math problems."""
    config = config or MathRewardConfig()

    # Map prompts to answers
    prompt_to_answer = {d["prompt"]: d["answer"] for d in dataset}

    def reward_fn(prompts: list[str], completions: list[str]) -> list[float]:
        """Compute rewards for completions."""
        rewards = []

        for prompt, completion in zip(prompts, completions):
            expected = prompt_to_answer.get(prompt)
            submitted = extract_submitted_answer(completion)

            if expected is None:
                # Unknown prompt, give neutral reward
                rewards.append(0.0)
                continue

            if submitted is None:
                # No answer submitted
                rewards.append(config.no_answer_reward)
                continue

            # Check correctness
            if abs(submitted - expected) < config.tolerance:
                rewards.append(config.correct_reward)
            elif config.partial_credit:
                # Partial credit based on relative error
                rel_error = abs(submitted - expected) / max(abs(expected), 1e-10)
                if rel_error < 0.1:  # Within 10%
                    rewards.append(config.correct_reward * 0.5)
                elif rel_error < 0.5:  # Within 50%
                    rewards.append(0.0)
                else:
                    rewards.append(config.incorrect_reward)
            else:
                rewards.append(config.incorrect_reward)

        return rewards

    return reward_fn


# =============================================================================
# Evaluation
# =============================================================================


async def evaluate_agent(
    agent: Agent,
    prompts: list[dict],
    num_samples: int = 50,
) -> dict:
    """Evaluate agent accuracy on a subset of problems."""
    correct = 0
    total = 0
    no_answer = 0

    samples = prompts[:num_samples]

    for item in samples:
        try:
            # Run agent
            result = await agent.run(item["prompt"])

            # Extract answer from trajectory
            trajectory_text = str(result)
            submitted = extract_submitted_answer(trajectory_text)

            if submitted is None:
                no_answer += 1
            elif item["answer"] is not None:
                if abs(submitted - item["answer"]) < 0.01:
                    correct += 1

            total += 1

        except Exception as e:
            print(f"Error evaluating: {e}")
            total += 1

    return {
        "accuracy": correct / total if total > 0 else 0,
        "no_answer_rate": no_answer / total if total > 0 else 0,
        "correct": correct,
        "total": total,
    }


# =============================================================================
# Training
# =============================================================================


def train_with_grpo(
    model_name: str,
    prompts: list[str],
    reward_fn,
    max_steps: int = 100,
    batch_size: int = 4,
    num_generations: int = 4,
):
    """Train model using Ray GRPO."""
    from dreadnode.core.training.ray import (
        RayGRPOConfig,
        RayGRPOTrainer,
        VLLMConfig,
        GRPOLossConfig,
    )

    config = RayGRPOConfig(
        model_name=model_name,
        max_steps=max_steps,
        num_prompts_per_step=batch_size,
        num_generations_per_prompt=num_generations,
        max_new_tokens=256,
        temperature=0.7,
        learning_rate=1e-6,
        log_interval=10,
        checkpoint_interval=50,
        vllm=VLLMConfig(
            gpu_memory_utilization=0.8,
            enable_prefix_caching=True,
        ),
        loss=GRPOLossConfig(
            kl_coef=0.05,
            use_leave_one_out_baseline=True,
        ),
    )

    trainer = RayGRPOTrainer(config)

    print(f"\nStarting GRPO training:")
    print(f"  Model: {model_name}")
    print(f"  Steps: {max_steps}")
    print(f"  Batch: {batch_size} prompts x {num_generations} generations")
    print(f"  Total samples/step: {batch_size * num_generations}")
    print()

    result = trainer.train(prompts=prompts, reward_fn=reward_fn)

    return trainer, result


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train math agent with GRPO")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model to train")
    parser.add_argument("--steps", type=int, default=100, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Prompts per step")
    parser.add_argument("--num-gen", type=int, default=4, help="Generations per prompt")
    parser.add_argument("--max-samples", type=int, default=500, help="Max training samples")
    parser.add_argument("--eval-samples", type=int, default=50, help="Evaluation samples")
    parser.add_argument("--quick", action="store_true", help="Quick test (10 steps)")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate, don't train")
    args = parser.parse_args()

    if args.quick:
        args.steps = 10
        args.max_samples = 100
        args.eval_samples = 20

    print("=" * 60)
    print("Math Agent Training with GRPO")
    print("=" * 60)

    # Load dataset
    print("\n[1/4] Loading GSM8K dataset...")
    dataset = load_gsm8k_prompts(split="train", max_samples=args.max_samples)
    prompts = [d["prompt"] for d in dataset]
    print(f"  Loaded {len(prompts)} problems")

    # Create reward function
    print("\n[2/4] Setting up reward function...")
    reward_fn = create_reward_fn(dataset)
    print("  Reward: +1.0 correct, -0.5 incorrect, -1.0 no answer")

    if args.eval_only:
        print("\n[3/4] Skipping training (--eval-only)")
        print("\n[4/4] Evaluating base agent...")
        agent = create_agent(args.model)
        eval_prompts = load_gsm8k_prompts(split="test", max_samples=args.eval_samples)
        metrics = asyncio.run(evaluate_agent(agent, eval_prompts, args.eval_samples))
        print(f"\nBase Model Results:")
        print(f"  Accuracy: {metrics['accuracy']:.1%}")
        print(f"  No Answer: {metrics['no_answer_rate']:.1%}")
        return

    # Train
    print(f"\n[3/4] Training with GRPO ({args.steps} steps)...")
    trainer, result = train_with_grpo(
        model_name=args.model,
        prompts=prompts,
        reward_fn=reward_fn,
        max_steps=args.steps,
        batch_size=args.batch_size,
        num_generations=args.num_gen,
    )

    print("\nTraining complete!")
    print(f"  Final metrics: {result}")

    # Note: Full evaluation would require loading the trained weights
    # into a dreadnode Agent, which needs the checkpoint path
    print("\n[4/4] Training finished!")
    print(f"  Checkpoints saved to: {trainer.config.checkpoint_dir}")
    print("\nTo evaluate the trained model:")
    print(f"  1. Load checkpoint from {trainer.config.checkpoint_dir}")
    print("  2. Create agent with trained model path")
    print("  3. Run evaluation on test set")


if __name__ == "__main__":
    main()

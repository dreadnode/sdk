#!/usr/bin/env python3
"""
Train a language model on GSM8K using Ray-based GRPO.

This example demonstrates:
1. Loading GSM8K dataset
2. Defining a math answer reward function
3. Training with native Ray GRPO

Usage:
    python examples/train_gsm8k_ray_grpo.py

Requirements:
    pip install datasets ray vllm transformers torch
"""

# Must be first before any CUDA imports
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  # Force spawn for vLLM workers

import re
import argparse
from datasets import load_dataset

from dreadnode.core.training.ray import RayGRPOTrainer, RayGRPOConfig, AsyncRayGRPOTrainer
from dreadnode.core.training.ray.config import VLLMConfig, TrainingConfig, GRPOLossConfig


def extract_answer(text: str) -> str | None:
    """
    Extract the final numerical answer from a solution.

    Looks for patterns like:
    - "#### 42"
    - "The answer is 42"
    - "= 42"
    """
    # Try GSM8K format first: #### <number>
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")

    # Try "answer is" format
    match = re.search(r"answer\s+is\s+(-?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")

    # Try finding the last number in the text
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def normalize_answer(answer: str | None) -> str | None:
    """Normalize answer for comparison."""
    if answer is None:
        return None

    # Remove commas and whitespace
    answer = answer.replace(",", "").strip()

    try:
        # Convert to float and back to handle scientific notation
        num = float(answer)
        # If it's a whole number, return as int
        if num == int(num):
            return str(int(num))
        return str(num)
    except ValueError:
        return answer


def create_reward_fn(dataset):
    """
    Create a reward function for GSM8K.

    Returns 1.0 if the model's answer matches the ground truth,
    0.0 otherwise.
    """
    # Build a mapping from question to answer
    question_to_answer = {}
    for item in dataset:
        question = item["question"]
        answer = item["answer"]
        # Extract the numerical answer from the full solution
        final_answer = extract_answer(answer)
        if final_answer:
            question_to_answer[question] = normalize_answer(final_answer)

    def reward_fn(prompts: list[str], completions: list[str]) -> list[float]:
        """Score completions based on answer correctness."""
        rewards = []

        for prompt, completion in zip(prompts, completions):
            # Extract the question from the prompt
            # Prompt format: "Question: {question}\n\nAnswer: Let me solve this step by step."
            if "Question: " in prompt and "\n\nAnswer:" in prompt:
                # Split on "\n\nAnswer:" and take the question part
                question_part = prompt.split("\n\nAnswer:")[0]
                question = question_part.replace("Question: ", "").strip()
            else:
                question = prompt.strip()

            # Get ground truth
            ground_truth = question_to_answer.get(question)

            if ground_truth is None:
                # Unknown question - give 0 reward (will create learning signal)
                rewards.append(0.0)
                continue

            # Extract model's answer from completion
            model_answer = normalize_answer(extract_answer(completion))

            # Compare answers
            if model_answer is not None and model_answer == ground_truth:
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        return rewards

    return reward_fn


def format_prompt(question: str) -> str:
    """Format a GSM8K question as a prompt."""
    return f"""Question: {question}

Answer: Let me solve this step by step."""


def main():
    parser = argparse.ArgumentParser(description="Train on GSM8K with Ray GRPO")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=8,
        help="Number of prompts per step",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="Generations per prompt",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens per generation",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints/gsm8k_grpo",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Use async training (requires 2+ GPUs)",
    )
    args = parser.parse_args()

    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    eval_dataset = load_dataset("openai/gsm8k", "main", split="test")

    print(f"  Train samples: {len(dataset)}")
    print(f"  Test samples: {len(eval_dataset)}")

    # Create prompts
    train_prompts = [format_prompt(item["question"]) for item in dataset]
    eval_prompts = [format_prompt(q) for q in eval_dataset[:100]["question"]]

    # Create reward function
    print("Creating reward function...")
    reward_fn = create_reward_fn(dataset)

    # Configure GRPO
    config = RayGRPOConfig(
        model_name=args.model,
        num_prompts_per_step=args.num_prompts,
        num_generations_per_prompt=args.num_generations,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        max_new_tokens=args.max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=1,
        eval_interval=10,
        checkpoint_interval=50,
        vllm=VLLMConfig(
            tensor_parallel_size=1,
            gpu_memory_utilization=0.4,  # Leave room for training
            enforce_eager=True,  # For debugging
        ),
        training=TrainingConfig(
            num_workers=1,
            backend="deepspeed",
            deepspeed_stage=2,
            gradient_checkpointing=True,
            mixed_precision="bf16",
        ),
        loss=GRPOLossConfig(
            kl_coef=0.0,  # Disable KL to save memory (no reference model)
            clip_ratio=0.2,
            normalize_advantages=True,
            use_leave_one_out_baseline=True,
        ),
    )

    print("\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Prompts per step: {config.num_prompts_per_step}")
    print(f"  Generations per prompt: {config.num_generations_per_prompt}")
    print(f"  Batch size: {config.train_batch_size}")
    print(f"  Max steps: {config.max_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  KL coefficient: {config.loss.kl_coef}")

    # Create trainer
    print(f"\nInitializing {'async ' if args.async_mode else ''}trainer...")

    if args.async_mode:
        trainer = AsyncRayGRPOTrainer(config)
    else:
        trainer = RayGRPOTrainer(config)

    # Train
    print("\nStarting training...")
    try:
        state = trainer.train(
            prompts=train_prompts,
            reward_fn=reward_fn,
            num_steps=args.max_steps,
        )

        print(f"\nTraining complete!")
        print(f"  Steps: {state.step}")
        print(f"  Samples seen: {state.samples_seen}")
        print(f"  Duration: {state.elapsed_seconds():.1f}s")

    finally:
        trainer.shutdown()


if __name__ == "__main__":
    main()

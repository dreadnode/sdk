#!/usr/bin/env python3
"""
Train a language model on GSM8K using the new async GRPO coordinator.

This example demonstrates:
1. Loading GSM8K dataset
2. Optional SFT warmup on ground truth solutions
3. Async GRPO training with ExperienceBuffer
4. Integration of SFT → GRPO pipeline

Usage:
    # GRPO only (uses AsyncGRPOCoordinator)
    python examples/train_gsm8k_async.py

    # SFT warmup then GRPO
    python examples/train_gsm8k_async.py --sft-warmup

    # Distributed training (requires multiple GPUs)
    python examples/train_gsm8k_async.py --distributed --num-workers 4

Requirements:
    pip install datasets ray vllm transformers torch
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse
import asyncio
import re
from typing import Callable

from datasets import load_dataset


# =============================================================================
# Answer Extraction and Reward Function
# =============================================================================


def extract_answer(text: str) -> str | None:
    """Extract numerical answer from solution text."""
    # GSM8K format: #### <number>
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")

    # "answer is" format
    match = re.search(r"answer\s+is\s+(-?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")

    # Last number in text
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def normalize_answer(answer: str | None) -> str | None:
    """Normalize answer for comparison."""
    if answer is None:
        return None

    answer = answer.replace(",", "").strip()

    try:
        num = float(answer)
        if num == int(num):
            return str(int(num))
        return str(num)
    except ValueError:
        return answer


def create_reward_fn(dataset) -> Callable[[list[str], list[str]], list[float]]:
    """Create reward function that scores correctness."""
    # Build question -> answer mapping
    question_to_answer = {}
    for item in dataset:
        question = item["question"]
        answer = item["answer"]
        final_answer = extract_answer(answer)
        if final_answer:
            question_to_answer[question] = normalize_answer(final_answer)

    def reward_fn(prompts: list[str], completions: list[str]) -> list[float]:
        """Score completions based on answer correctness."""
        rewards = []

        for prompt, completion in zip(prompts, completions):
            # Extract question from prompt
            if "Question: " in prompt and "\n\nAnswer:" in prompt:
                question_part = prompt.split("\n\nAnswer:")[0]
                question = question_part.replace("Question: ", "").strip()
            else:
                question = prompt.strip()

            ground_truth = question_to_answer.get(question)

            if ground_truth is None:
                rewards.append(0.0)
                continue

            model_answer = normalize_answer(extract_answer(completion))

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


def format_sft_example(item: dict) -> dict:
    """Format GSM8K item for SFT training."""
    return {
        "text": f"""Question: {item["question"]}

Answer: {item["answer"]}"""
    }


# =============================================================================
# Training Functions
# =============================================================================


def run_sft_warmup(
    model_name: str,
    dataset,
    max_steps: int = 100,
    checkpoint_dir: str = "./checkpoints/gsm8k_sft",
):
    """Run SFT warmup on ground truth solutions."""
    from dreadnode.core.training.ray.sft import SFTConfig, SFTTrainer

    print("\n" + "=" * 50)
    print("Starting SFT Warmup")
    print("=" * 50)

    # Format dataset for SFT
    sft_data = [format_sft_example(item) for item in dataset]

    config = SFTConfig(
        model_name=model_name,
        max_seq_length=2048,
        use_packing=True,
        learning_rate=2e-5,
        max_steps=max_steps,
        batch_size=4,
        checkpoint_dir=checkpoint_dir,
        log_interval=10,
    )

    trainer = SFTTrainer(config)
    metrics = trainer.train(sft_data)

    print(f"\nSFT complete! Final loss: {metrics.get('loss', 'N/A'):.4f}")
    return checkpoint_dir


def run_sync_grpo(
    model_name: str,
    prompts: list[str],
    reward_fn: Callable,
    max_steps: int = 100,
    checkpoint_dir: str = "./checkpoints/gsm8k_grpo",
    fast_mode: bool = False,
):
    """Run sync GRPO training (single GPU).

    Args:
        fast_mode: If True, keeps both vLLM and training model loaded (higher memory,
                   but much faster due to no model reload overhead).
    """
    from dreadnode.core.training.ray import RayGRPOTrainer, RayGRPOConfig
    from dreadnode.core.training.ray.config import VLLMConfig, GRPOLossConfig, TrainingConfig

    print("\n" + "=" * 50)
    mode_str = "Fast" if fast_mode else "Memory-Efficient"
    print(f"Starting Sync GRPO Training ({mode_str})")
    print("=" * 50)

    if fast_mode:
        # Fast mode: higher batch, CUDA graphs, more GPU memory for vLLM
        # Note: uses reload-based weight sync (colocate mode not supported with vLLM v1)
        config = RayGRPOConfig(
            model_name=model_name,
            num_prompts_per_step=8,  # Larger batch
            num_generations_per_prompt=4,
            max_steps=max_steps,
            learning_rate=1e-6,
            max_new_tokens=512,
            temperature=0.7,
            checkpoint_dir=checkpoint_dir,
            log_interval=1,
            vllm=VLLMConfig(
                tensor_parallel_size=1,
                gpu_memory_utilization=0.85,  # High utilization when not colocated
                enforce_eager=False,  # Enable CUDA graphs for 2-3x speedup
                max_model_len=2048,
            ),
            training=TrainingConfig(
                gradient_checkpointing=True,
                mixed_precision="bf16",
            ),
            loss=GRPOLossConfig(
                kl_coef=0.0,
                clip_ratio=0.2,
                use_leave_one_out_baseline=True,
            ),
        )
        colocate = False  # vLLM v1 doesn't support in-place weight updates
    else:
        # Memory-efficient mode: lower memory, model loading/unloading each step
        config = RayGRPOConfig(
            model_name=model_name,
            num_prompts_per_step=4,
            num_generations_per_prompt=4,
            max_steps=max_steps,
            learning_rate=1e-6,
            max_new_tokens=256,
            temperature=0.7,
            checkpoint_dir=checkpoint_dir,
            log_interval=1,
            vllm=VLLMConfig(
                tensor_parallel_size=1,
                gpu_memory_utilization=0.3,
                enforce_eager=True,
                max_model_len=2048,
            ),
            training=TrainingConfig(
                gradient_checkpointing=True,
                mixed_precision="bf16",
            ),
            loss=GRPOLossConfig(
                kl_coef=0.0,
                clip_ratio=0.2,
                use_leave_one_out_baseline=True,
            ),
        )
        colocate = False

    trainer = RayGRPOTrainer(config, colocate=colocate)

    try:
        state = trainer.train(
            prompts=prompts,
            reward_fn=reward_fn,
            num_steps=max_steps,
        )
        print(f"\nGRPO complete!")
        print(f"  Steps: {state.step}")
        print(f"  Best reward: {state.best_reward:.4f}")
        print(f"  Duration: {state.elapsed_seconds():.1f}s")
        return state
    finally:
        trainer.shutdown()


async def run_async_grpo(
    model_name: str,
    prompts: list[str],
    reward_fn: Callable,
    max_steps: int = 100,
    num_rollout_workers: int = 1,
    buffer_size: int = 2,
    checkpoint_dir: str = "./checkpoints/gsm8k_grpo",
    packed: bool = False,
):
    """Run async GRPO training.

    Args:
        packed: If True, pack multiple vLLM workers per GPU for higher throughput.
    """
    import ray
    from dreadnode.core.training.ray import (
        AsyncGRPOCoordinator,
        AsyncGRPOConfig,
        RayGRPOConfig,
    )
    from dreadnode.core.training.ray.config import VLLMConfig, GRPOLossConfig

    print("\n" + "=" * 50)
    print("Starting Async GRPO Training")
    print("=" * 50)
    print(f"  Rollout workers: {num_rollout_workers}")
    print(f"  Buffer size: {buffer_size}")
    print(f"  Packed mode: {packed}")

    # Initialize Ray with excludes for large files
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "excludes": [
                    "checkpoints/",
                    ".git/objects/",
                    "*.bin",
                    "*.pt",
                    "*.safetensors",
                    "__pycache__/",
                    ".venv/",
                ]
            }
        )

    if packed:
        # Packed mode: multiple vLLM workers per GPU, higher throughput
        # For 80GB A100: can fit 2 vLLM workers (40% each) + training on remaining space
        vllm_config = VLLMConfig(
            tensor_parallel_size=1,
            gpu_memory_utilization=0.35,  # ~28GB each, fits 2 per 80GB GPU
            enforce_eager=False,  # Enable CUDA graphs
            max_model_len=2048,
        )
        num_prompts = 16  # Larger batch with more workers
    else:
        # Standard mode: 1 vLLM worker per GPU
        vllm_config = VLLMConfig(
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            enforce_eager=False,  # Enable CUDA graphs
            max_model_len=2048,
        )
        num_prompts = 8

    config = RayGRPOConfig(
        model_name=model_name,
        num_prompts_per_step=num_prompts,
        num_generations_per_prompt=4,
        max_steps=max_steps,
        learning_rate=1e-6,
        max_new_tokens=512,  # More tokens for better accuracy
        temperature=0.7,
        checkpoint_dir=checkpoint_dir,
        vllm=vllm_config,
        loss=GRPOLossConfig(
            kl_coef=0.0,
            clip_ratio=0.2,
            use_leave_one_out_baseline=True,
        ),
    )

    async_config = AsyncGRPOConfig(
        num_rollout_workers=num_rollout_workers,
        buffer_size=buffer_size,
        weight_sync_interval=5,
        max_steps=max_steps,
        log_interval=1,
        checkpoint_interval=50,
    )

    coordinator = AsyncGRPOCoordinator(
        config=config,
        prompts=prompts,
        reward_fn=reward_fn,
        async_config=async_config,
        packed=packed,
    )

    try:
        state = await coordinator.train(num_steps=max_steps)
        print(f"\nGRPO complete!")
        print(f"  Steps: {state.step}")
        print(f"  Best reward: {state.best_reward:.4f}")
        print(f"  Duration: {state.elapsed_seconds():.1f}s")
        return state
    finally:
        coordinator.shutdown()


def run_distributed_grpo(
    model_name: str,
    prompts: list[str],
    reward_fn: Callable,
    num_workers: int = 4,
    max_steps: int = 100,
):
    """Run distributed GRPO training with Ray Train."""
    from dreadnode.core.training.ray import (
        RayGRPOConfig,
        train_grpo_distributed,
        DistributedConfig,
    )
    from dreadnode.core.training.ray.config import VLLMConfig, GRPOLossConfig

    print("\n" + "=" * 50)
    print(f"Starting Distributed GRPO Training ({num_workers} workers)")
    print("=" * 50)

    config = RayGRPOConfig(
        model_name=model_name,
        num_prompts_per_step=4,  # Per worker
        num_generations_per_prompt=4,
        max_steps=max_steps,
        learning_rate=1e-6,
        vllm=VLLMConfig(
            gpu_memory_utilization=0.3,  # Lower for colocated training
            enforce_eager=True,
        ),
        loss=GRPOLossConfig(
            kl_coef=0.0,
            use_leave_one_out_baseline=True,
        ),
    )

    distributed_config = DistributedConfig(
        num_workers=num_workers,
        checkpoint_interval=50,
    )

    result = train_grpo_distributed(
        config=config,
        prompts=prompts,
        reward_fn=reward_fn,
        distributed_config=distributed_config,
    )

    print(f"\nDistributed training complete!")
    print(f"  Final loss: {result.metrics.get('loss', 'N/A')}")
    return result


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train on GSM8K with async GRPO coordinator"
    )
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
        "--sft-warmup",
        action="store_true",
        help="Run SFT warmup before GRPO",
    )
    parser.add_argument(
        "--sft-steps",
        type=int,
        default=50,
        help="SFT warmup steps",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: keeps vLLM + training model loaded (needs ~20GB VRAM for 1.5B model)",
    )
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Use async training (requires separate GPUs for inference/training)",
    )
    parser.add_argument(
        "--packed",
        action="store_true",
        help="Pack multiple vLLM workers per GPU (for 80GB+ GPUs)",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use distributed training with Ray Train",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of workers for distributed/async training",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=2,
        help="Experience buffer size (off-policy degree)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints/gsm8k",
        help="Checkpoint directory",
    )
    args = parser.parse_args()

    # Load dataset
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    print(f"  Samples: {len(dataset)}")

    # Format prompts
    train_prompts = [format_prompt(item["question"]) for item in dataset]

    # Create reward function
    print("Creating reward function...")
    reward_fn = create_reward_fn(dataset)

    # Determine starting model
    model_name = args.model

    # Optional SFT warmup
    if args.sft_warmup:
        sft_checkpoint = run_sft_warmup(
            model_name=model_name,
            dataset=dataset,
            max_steps=args.sft_steps,
            checkpoint_dir=f"{args.checkpoint_dir}/sft",
        )
        model_name = f"{sft_checkpoint}/step_{args.sft_steps}"
        print(f"\nUsing SFT checkpoint: {model_name}")

    # Run GRPO training
    if args.distributed:
        run_distributed_grpo(
            model_name=model_name,
            prompts=train_prompts,
            reward_fn=reward_fn,
            num_workers=args.num_workers,
            max_steps=args.max_steps,
        )
    elif args.use_async:
        asyncio.run(
            run_async_grpo(
                model_name=model_name,
                prompts=train_prompts,
                reward_fn=reward_fn,
                max_steps=args.max_steps,
                num_rollout_workers=args.num_workers,
                buffer_size=args.buffer_size,
                checkpoint_dir=f"{args.checkpoint_dir}/grpo",
                packed=args.packed,
            )
        )
    else:
        # Default: sync training (single GPU)
        run_sync_grpo(
            model_name=model_name,
            prompts=train_prompts,
            reward_fn=reward_fn,
            max_steps=args.max_steps,
            checkpoint_dir=f"{args.checkpoint_dir}/grpo",
            fast_mode=args.fast,
        )


if __name__ == "__main__":
    main()

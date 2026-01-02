#!/usr/bin/env python3
"""
Train a model on GSM8K using GRPO with the dreadnode training module.

This script demonstrates how to use the dreadnode training infrastructure
to fine-tune a model on the GSM8K math word problems dataset.

The dreadnode training module wraps NeMo RL, providing:
- GRPOTrainer for policy optimization
- ScorerReward to use dreadnode scorers as reward signals
- AgentEnvironment for NeMo RL-compatible rollout evaluation

Usage:
    python -m dreadnode.training.gsm8k
    python -m dreadnode.training.gsm8k --model Qwen/Qwen2.5-1.5B-Instruct --steps 200
"""

import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import torch


def init_ray_simple() -> None:
    """Initialize Ray with a simple runtime environment."""
    import ray

    if ray.is_initialized():
        return

    env_vars = dict(os.environ)
    env_vars.pop("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", None)

    runtime_env = {"env_vars": env_vars}

    ray.init(
        log_to_driver=True,
        include_dashboard=True,
        runtime_env=runtime_env,
    )


# Initialize Ray before importing dreadnode training components
# to avoid auto-packaging of the local module
init_ray_simple()

import ray  # noqa: E402

from dreadnode.core.training import (  # noqa: E402
    GRPOTrainer,
    RewardAggregator,
    ScorerReward,
    SuccessReward,
    AgentEnvironment,
    AgentEnvConfig,
)
from dreadnode.evaluations.gsm8k.agent import create_math_agent  # noqa: E402
from dreadnode.evaluations.gsm8k.scorers import (  # noqa: E402
    answer_correct_scorer,
    reasoning_quality_scorer,
    efficiency_scorer,
)


@dataclass
class GSM8KTrainingConfig:
    """Configuration for GSM8K GRPO training."""

    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # Training hyperparameters
    num_prompts_per_step: int = 8
    num_generations_per_prompt: int = 4
    max_num_steps: int = 100
    learning_rate: float = 5e-6
    kl_penalty_coeff: float = 0.01

    # Rollout settings
    max_rollout_turns: int = 15
    max_seq_length: int = 1024

    # Validation
    val_period: int = 10
    val_at_start: bool = True

    # Data
    max_train_samples: int | None = 1000

    # Checkpointing
    checkpoint_dir: str = "results/gsm8k_grpo"
    checkpoint_interval: int = 20

    # Logging
    log_dir: str = "logs/gsm8k"
    tensorboard_enabled: bool = True

    # Cluster
    num_nodes: int = 1
    gpus_per_node: int | None = None  # Auto-detect


def create_gsm8k_rewards() -> RewardAggregator:
    """
    Create reward aggregator for GSM8K training.

    Uses dreadnode scorers bridged to reward functions:
    - answer_correct_scorer: Primary signal (correct vs incorrect)
    - reasoning_quality_scorer: Encourages showing work
    - efficiency_scorer: Penalizes overly long solutions
    """
    return RewardAggregator(
        rewards=[
            # Primary reward: correct answer (weight=0.6)
            ScorerReward(
                scorer=answer_correct_scorer,
                weight=0.6,
                source_min=0.0,
                source_max=1.0,
                target_min=-0.2,  # Penalty for wrong answer
                target_max=1.0,   # Full reward for correct
            ),
            # Reasoning quality (weight=0.25)
            ScorerReward(
                scorer=reasoning_quality_scorer,
                weight=0.25,
                source_min=0.0,
                source_max=1.0,
                target_min=0.0,
                target_max=0.5,
            ),
            # Efficiency bonus (weight=0.15)
            ScorerReward(
                scorer=efficiency_scorer,
                weight=0.15,
                source_min=0.0,
                source_max=1.0,
                target_min=0.0,
                target_max=0.3,
            ),
        ]
    )


def create_gsm8k_environment(num_workers: int = 4) -> Any:
    """
    Create NeMo RL-compatible environment for GSM8K.

    The AgentEnvironment evaluates rollouts using the dreadnode scorers
    and provides rewards to the training loop.
    """
    env_config: AgentEnvConfig = {
        "num_workers": num_workers,
        "success_reward": 1.0,
        "failure_reward": 0.0,
        "partial_reward": 0.5,
        "eval_mode": "offline",  # Compare to ground truth
        "max_tool_calls": 20,
    }

    return AgentEnvironment.remote(env_config)


def load_gsm8k_prompts(split: str = "train", max_samples: int | None = None) -> list[str]:
    """Load GSM8K prompts from HuggingFace datasets."""
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split=split)

    prompts = []
    for i, sample in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        prompts.append(sample["question"])

    return prompts


async def train_gsm8k(config: GSM8KTrainingConfig) -> bool:
    """
    Run GSM8K GRPO training using dreadnode infrastructure.

    Args:
        config: Training configuration.

    Returns:
        True if training completed successfully.
    """
    print("=" * 60)
    print("GSM8K GRPO Training with Dreadnode")
    print("=" * 60)

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return False

    gpus = config.gpus_per_node or torch.cuda.device_count()
    print(f"\nCUDA devices: {gpus}")
    for i in range(gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    # Create the GSM8K math agent
    print(f"\nCreating GSM8K agent with model: {config.model_name}")
    agent = create_math_agent(
        model=config.model_name,
        max_steps=config.max_rollout_turns,
    )

    # Create reward aggregator using dreadnode scorers
    print("\nSetting up reward aggregator...")
    rewards = create_gsm8k_rewards()
    print(f"  Reward components: {len(rewards.rewards)}")

    # Create GRPOTrainer with dreadnode infrastructure
    print("\nInitializing GRPO trainer...")
    trainer = GRPOTrainer(
        agent=agent,
        rewards=rewards,
        config={
            "model_name": config.model_name,
            "num_prompts_per_step": config.num_prompts_per_step,
            "num_generations_per_prompt": config.num_generations_per_prompt,
            "max_num_steps": config.max_num_steps,
            "max_rollout_turns": config.max_rollout_turns,
            "max_seq_length": config.max_seq_length,
            "learning_rate": config.learning_rate,
            "kl_penalty_coeff": config.kl_penalty_coeff,
            "val_period": config.val_period,
            "val_at_start": config.val_at_start,
            "checkpoint_dir": config.checkpoint_dir,
            "checkpoint_interval": config.checkpoint_interval,
            "log_dir": config.log_dir,
            "tensorboard_enabled": config.tensorboard_enabled,
            "use_leave_one_out_baseline": True,
            "normalize_rewards": True,
            "colocated_inference": True,
            "generation_backend": "vllm",
        },
    )

    # Create environment for rollouts
    print("\nCreating GSM8K environment...")
    environment = create_gsm8k_environment(num_workers=4)

    # Load training prompts
    print("\nLoading GSM8K dataset...")
    train_prompts = load_gsm8k_prompts("train", max_samples=config.max_train_samples)
    print(f"  Training prompts: {len(train_prompts)}")

    # Setup trainer with environment
    print("\nSetting up trainer...")
    await trainer.setup(
        environment=environment,
        cluster_config={
            "num_nodes": config.num_nodes,
            "gpus_per_node": gpus,
        },
    )

    # Run training
    print("\n" + "=" * 60)
    print("Starting GRPO Training")
    print("=" * 60)
    print(f"  Model: {config.model_name}")
    print(f"  Prompts/step: {config.num_prompts_per_step}")
    print(f"  Generations/prompt: {config.num_generations_per_prompt}")
    print(f"  Max steps: {config.max_num_steps}")
    print(f"  Learning rate: {config.learning_rate}")

    try:
        state = await trainer.train(
            prompts=train_prompts,
            environment=environment,
            num_steps=config.max_num_steps,
        )

        # Print final results
        print("\n" + "=" * 60)
        print("Training Complete")
        print("=" * 60)
        print(f"  Total steps: {state.step}")
        print(f"  Samples seen: {state.samples_seen}")
        print(f"  Best reward: {state.best_reward:.4f}")
        print(f"  Final metrics: {state.metrics}")

        return True

    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        print("\nShutting down...")
        trainer.shutdown()
        if ray.is_initialized():
            ray.shutdown()


def parse_args() -> GSM8KTrainingConfig:
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Train on GSM8K with GRPO")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Model to train")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of training steps")
    parser.add_argument("--prompts-per-step", type=int, default=8,
                        help="Number of prompts per training step")
    parser.add_argument("--generations-per-prompt", type=int, default=4,
                        help="Number of generations per prompt")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Maximum training samples")
    parser.add_argument("--checkpoint-dir", type=str, default="results/gsm8k_grpo",
                        help="Checkpoint directory")

    args = parser.parse_args()

    return GSM8KTrainingConfig(
        model_name=args.model,
        max_num_steps=args.steps,
        num_prompts_per_step=args.prompts_per_step,
        num_generations_per_prompt=args.generations_per_prompt,
        learning_rate=args.lr,
        max_train_samples=args.max_samples,
        checkpoint_dir=args.checkpoint_dir,
    )


async def main() -> int:
    """Main entry point."""
    config = parse_args()

    try:
        success = await train_gsm8k(config)
        if success:
            print("\nTraining completed successfully!")
            return 0
        else:
            print("\nTraining failed!")
            return 1
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

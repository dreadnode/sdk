"""
Trainers for DN SDK agents.

Provides high-level training interfaces wrapping NeMo RL algorithms.

Key components:
- GRPOTrainer: Group Relative Policy Optimization
- SFTTrainer: Supervised Fine-Tuning
- TrainingConfig: Common configuration

Usage:
    from dreadnode import Agent, scorer
    from dreadnode.core.training import GRPOTrainer, RewardAggregator, ScorerReward

    agent = Agent(model="meta-llama/...", tools=[...])

    @scorer
    async def task_success(trajectory):
        return 1.0 if trajectory.success else 0.0

    rewards = RewardAggregator([ScorerReward(task_success)])

    trainer = GRPOTrainer(
        agent=agent,
        rewards=rewards,
        config={...}
    )
    await trainer.train(prompts, environment)
"""

from dreadnode.core.training.trainers.base import (
    BaseTrainer,
    TrainingCallback,
    TrainingConfig,
    TrainingState,
)
from dreadnode.core.training.trainers.grpo import GRPOConfig, GRPOTrainer
from dreadnode.core.training.trainers.sft import SFTConfig, SFTTrainer

__all__ = [
    "BaseTrainer",
    "GRPOConfig",
    "GRPOTrainer",
    "SFTConfig",
    "SFTTrainer",
    "TrainingCallback",
    "TrainingConfig",
    "TrainingState",
]

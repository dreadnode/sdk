"""
Dreadnode Training Examples and Utilities.

This module provides training examples and utilities for fine-tuning models
using the dreadnode training infrastructure.

The core training module is in dreadnode.core.training, which provides:
- GRPOTrainer: Group Relative Policy Optimization trainer
- SFTTrainer: Supervised Fine-Tuning trainer
- RewardAggregator: Combine multiple reward sources
- ScorerReward: Bridge dreadnode scorers to training rewards

This module provides example training scripts and configurations:
- gsm8k: Train on grade school math problems
"""

from dreadnode.core.training import (
    GRPOTrainer,
    GRPOConfig,
    SFTTrainer,
    SFTConfig,
    RewardAggregator,
    ScorerReward,
    SuccessReward,
    ToolPenalty,
    LengthPenalty,
    TurnPenalty,
)

# Ray-based trainers (new)
from dreadnode.core.training.ray import (
    RayGRPOConfig,
    RayGRPOTrainer,
    AsyncRayGRPOTrainer,
    DPOConfig,
    DPOTrainer,
    PPOConfig,
    PPOTrainer,
    RMConfig,
    RewardModelTrainer,
    SFTConfig as RaySFTConfig,
    SFTTrainer as RaySFTTrainer,
)

__all__ = [
    # Legacy trainers
    "GRPOTrainer",
    "GRPOConfig",
    "SFTTrainer",
    "SFTConfig",
    # Rewards
    "RewardAggregator",
    "ScorerReward",
    "SuccessReward",
    "ToolPenalty",
    "LengthPenalty",
    "TurnPenalty",
    # Ray-based trainers
    "RayGRPOConfig",
    "RayGRPOTrainer",
    "AsyncRayGRPOTrainer",
    "DPOConfig",
    "DPOTrainer",
    "PPOConfig",
    "PPOTrainer",
    "RMConfig",
    "RewardModelTrainer",
    "RaySFTConfig",
    "RaySFTTrainer",
]

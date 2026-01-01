"""
Reward functions for agent training.

Bridges DN SDK Scorers with NeMo RL training infrastructure.

Key components:
- RewardFunction: Protocol for computing rewards from rollout results
- RewardAggregator: Combines multiple reward sources
- ScorerReward: Wraps DN SDK Scorers for training
- Built-in rewards: SuccessReward, ToolPenalty, LengthPenalty

Usage:
    from dreadnode.core.training.rewards import (
        RewardAggregator,
        ScorerReward,
        SuccessReward,
        ToolPenalty,
    )

    # Create composite reward
    reward_fn = RewardAggregator(
        rewards=[
            SuccessReward(weight=1.0),
            ToolPenalty(weight=0.1),
            ScorerReward(my_scorer, weight=0.5),
        ]
    )

    # Compute reward from rollout
    result = reward_fn.compute(rollout_result)
"""

from dreadnode.core.training.rewards.types import (
    RewardConfig,
    RewardResult,
    RewardComponent,
    RewardMetrics,
)
from dreadnode.core.training.rewards.functions import (
    RewardFunction,
    BaseRewardFunction,
    SuccessReward,
    ToolPenalty,
    LengthPenalty,
    TurnPenalty,
    FormatReward,
)
from dreadnode.core.training.rewards.aggregator import (
    RewardAggregator,
    AggregationStrategy,
    create_standard_reward,
)
from dreadnode.core.training.rewards.scorer_bridge import (
    ScorerReward,
    ScorerHookReward,
    MetricReward,
    create_scorer_reward,
)
from dreadnode.core.training.rewards.shaping import (
    RewardShaper,
    apply_dapo_penalty,
    apply_scaling,
    clip_reward,
    shape_rewards_for_grpo,
)

__all__ = [
    # Types
    "RewardConfig",
    "RewardResult",
    "RewardComponent",
    "RewardMetrics",
    # Core Functions
    "RewardFunction",
    "BaseRewardFunction",
    "SuccessReward",
    "ToolPenalty",
    "LengthPenalty",
    "TurnPenalty",
    "FormatReward",
    # Aggregation
    "RewardAggregator",
    "AggregationStrategy",
    "create_standard_reward",
    # DN SDK Integration
    "ScorerReward",
    "ScorerHookReward",
    "MetricReward",
    "create_scorer_reward",
    # Shaping (NeMo RL compatible)
    "RewardShaper",
    "apply_dapo_penalty",
    "apply_scaling",
    "clip_reward",
    "shape_rewards_for_grpo",
]

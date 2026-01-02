"""
Reward shaping utilities.

Matches NeMo RL's reward shaping patterns for compatibility with GRPO training.

Key functions:
- apply_dapo_penalty: DAPO-style length penalty
- apply_scaling: Linear reward scaling with clamping
- clip_reward: Clamp rewards to a range
- RewardShaper: Composable shaping pipeline
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from dreadnode.core.training.rewards.types import (
    RewardConfig,
    RewardResult,
)


def apply_dapo_penalty(
    reward: float,
    response_length: int,
    max_response_length: int,
    buffer_length: int,
    penalty: float,
) -> float:
    """
    Apply DAPO-style penalty for responses exceeding max length.

    This matches NeMo RL's apply_reward_shaping() function.

    From DAPO paper: https://arxiv.org/pdf/2503.14476

    Args:
        reward: Original reward value.
        response_length: Length of the response in tokens.
        max_response_length: Maximum allowed response length.
        buffer_length: Buffer before full penalty applies.
        penalty: Maximum penalty value.

    Returns:
        Shaped reward with penalty applied.
    """
    expected_length = max_response_length - buffer_length

    if response_length <= expected_length:
        return reward

    # Calculate exceed and penalty
    exceed_length = response_length - expected_length
    overlong_reward = min(
        -exceed_length / buffer_length * penalty,
        0.0,
    )

    return reward + overlong_reward


def apply_scaling(
    reward: float,
    source_min: float = 0.0,
    source_max: float = 1.0,
    target_min: float = 0.0,
    target_max: float = 1.0,
) -> float:
    """
    Apply linear scaling with clamping.

    Matches NeMo RL's RewardScalingConfig behavior.

    Args:
        reward: Original reward value.
        source_min: Minimum of source range.
        source_max: Maximum of source range.
        target_min: Minimum of target range.
        target_max: Maximum of target range.

    Returns:
        Scaled reward.
    """
    # Clamp to source range
    clamped = max(source_min, min(source_max, reward))

    # Normalize to [0, 1]
    if source_max == source_min:
        normalized = 0.0
    else:
        normalized = (clamped - source_min) / (source_max - source_min)

    # Scale to target range
    return target_min + normalized * (target_max - target_min)


def clip_reward(
    reward: float,
    min_val: float = -float("inf"),
    max_val: float = float("inf"),
) -> float:
    """
    Clip reward to a range.

    Args:
        reward: Original reward value.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.

    Returns:
        Clipped reward.
    """
    return max(min_val, min(max_val, reward))


@dataclass
class RewardShaper:
    """
    Composable reward shaping pipeline.

    Applies multiple shaping operations in sequence to produce
    the final reward. Compatible with NeMo RL's BatchedDataDict.

    Example:
        shaper = RewardShaper(
            scaling_enabled=True,
            source_min=0.0,
            source_max=1.0,
            target_min=0.0,
            target_max=1.0,
            dapo_enabled=True,
            max_response_length=4096,
            buffer_length=512,
            penalty=0.5,
        )

        # Shape a single result
        shaped = shaper.shape(result, response_length=3000)

        # Shape a batch for training
        batch = shaper.shape_batch(results, response_lengths)
    """

    # Scaling config (matches NeMo RL RewardScalingConfig)
    scaling_enabled: bool = False
    source_min: float = 0.0
    source_max: float = 1.0
    target_min: float = 0.0
    target_max: float = 1.0

    # DAPO config (matches NeMo RL RewardShapingConfig)
    dapo_enabled: bool = False
    max_response_length: int = 4096
    buffer_length: int = 512
    penalty: float = 0.5

    # Clipping
    clip_enabled: bool = False
    clip_min: float = -float("inf")
    clip_max: float = float("inf")

    # Normalization (for GRPO)
    normalize_enabled: bool = False
    running_mean: float = 0.0
    running_std: float = 1.0
    normalize_epsilon: float = 1e-8

    def shape(
        self,
        result: RewardResult,
        response_length: int | None = None,
    ) -> RewardResult:
        """
        Apply shaping to a single reward result.

        Args:
            result: Original RewardResult.
            response_length: Response length in tokens (for DAPO).

        Returns:
            New RewardResult with shaped reward.
        """
        reward = result.reward

        # Apply scaling
        if self.scaling_enabled:
            reward = apply_scaling(
                reward,
                self.source_min,
                self.source_max,
                self.target_min,
                self.target_max,
            )
            result.metrics.scaling_applied = True

        # Apply DAPO penalty
        if self.dapo_enabled and response_length is not None:
            reward = apply_dapo_penalty(
                reward,
                response_length,
                self.max_response_length,
                self.buffer_length,
                self.penalty,
            )
            result.metrics.penalty_applied = True

        # Apply clipping
        if self.clip_enabled:
            reward = clip_reward(reward, self.clip_min, self.clip_max)

        # Update result
        result.metrics.shaped_reward = reward
        result.reward = reward

        return result

    def shape_batch(
        self,
        results: Sequence[RewardResult],
        response_lengths: Sequence[int] | None = None,
    ) -> list[RewardResult]:
        """
        Apply shaping to a batch of results.

        Args:
            results: List of RewardResults.
            response_lengths: Optional list of response lengths.

        Returns:
            List of shaped RewardResults.
        """
        if response_lengths is None:
            response_lengths = [None] * len(results)  # type: ignore

        shaped = []
        for result, length in zip(results, response_lengths):
            shaped.append(self.shape(result, length))

        # Apply normalization across batch if enabled
        if self.normalize_enabled and shaped:
            rewards = [r.reward for r in shaped]
            mean = sum(rewards) / len(rewards)
            std = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
            std = max(std, self.normalize_epsilon)

            for result in shaped:
                result.reward = (result.reward - mean) / std
                result.metrics.baseline = mean

        return shaped

    def to_nemo_config(self) -> dict:
        """
        Export as NeMo RL compatible config.

        Returns dict matching RewardShapingConfig + RewardScalingConfig.
        """
        return {
            "reward_shaping": {
                "enabled": self.dapo_enabled,
                "overlong_buffer_length": self.buffer_length,
                "overlong_buffer_penalty": self.penalty,
                "max_response_length": self.max_response_length,
            },
            "reward_scaling": {
                "enabled": self.scaling_enabled,
                "source_min": self.source_min,
                "source_max": self.source_max,
                "target_min": self.target_min,
                "target_max": self.target_max,
            },
        }

    @classmethod
    def from_nemo_config(cls, config: dict) -> "RewardShaper":
        """
        Create RewardShaper from NeMo RL config.

        Args:
            config: Dict with reward_shaping and/or reward_scaling keys.

        Returns:
            Configured RewardShaper.
        """
        shaping = config.get("reward_shaping", {})
        scaling = config.get("reward_scaling", {})

        return cls(
            # Scaling
            scaling_enabled=scaling.get("enabled", False),
            source_min=scaling.get("source_min", 0.0),
            source_max=scaling.get("source_max", 1.0),
            target_min=scaling.get("target_min", 0.0),
            target_max=scaling.get("target_max", 1.0),
            # DAPO
            dapo_enabled=shaping.get("enabled", False),
            buffer_length=shaping.get("overlong_buffer_length", 512),
            penalty=shaping.get("overlong_buffer_penalty", 0.5),
            max_response_length=shaping.get("max_response_length", 4096),
        )


def shape_rewards_for_grpo(
    results: Sequence[RewardResult],
    response_lengths: Sequence[int],
    config: RewardConfig | None = None,
    leave_one_out_baseline: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Shape rewards for GRPO training.

    Matches NeMo RL's reward processing in grpo.py:
    1. Apply scaling
    2. Apply DAPO penalty
    3. Compute per-prompt baseline
    4. Return rewards and advantages

    Args:
        results: List of RewardResults.
        response_lengths: Token lengths for DAPO penalty.
        config: Optional RewardConfig.
        leave_one_out_baseline: Use leave-one-out baseline per prompt.

    Returns:
        Tuple of (rewards tensor, advantages tensor).
    """
    # Apply shaping
    if config:
        shaper = RewardShaper(
            scaling_enabled=config.get("enabled", False),
            source_min=config.get("source_min", 0.0),
            source_max=config.get("source_max", 1.0),
            target_min=config.get("target_min", 0.0),
            target_max=config.get("target_max", 1.0),
            dapo_enabled=config.get("overlong_penalty_enabled", False),
            buffer_length=config.get("overlong_buffer_length", 512),
            penalty=config.get("overlong_buffer_penalty", 0.5),
            max_response_length=config.get("max_response_length", 4096),
            clip_enabled=config.get("clip_min") is not None,
            clip_min=config.get("clip_min", -float("inf")),
            clip_max=config.get("clip_max", float("inf")),
        )
        shaped = shaper.shape_batch(list(results), list(response_lengths))
    else:
        shaped = list(results)

    # Extract rewards
    rewards = torch.tensor([r.reward for r in shaped], dtype=torch.float32)

    # Compute advantages
    if leave_one_out_baseline:
        # Leave-one-out: for each sample, baseline is mean of others
        n = len(rewards)
        total = rewards.sum()
        baselines = (total - rewards) / (n - 1) if n > 1 else torch.zeros_like(rewards)
        advantages = rewards - baselines
    else:
        # Simple baseline: mean of all
        baseline = rewards.mean()
        advantages = rewards - baseline

    return rewards, advantages

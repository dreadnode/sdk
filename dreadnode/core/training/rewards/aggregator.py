"""
Reward aggregation for combining multiple reward sources.

Supports various aggregation strategies matching NeMo RL patterns.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from dreadnode.core.training.rewards.types import (
    RewardComponent,
    RewardResult,
    RewardMetrics,
)
from dreadnode.core.training.rewards.functions import RewardFunction
from dreadnode.core.training.rollouts.types import RolloutResult


class AggregationStrategy(str, Enum):
    """Strategy for combining reward components."""

    SUM = "sum"
    """Simple sum of weighted components."""

    WEIGHTED_AVERAGE = "weighted_average"
    """Weighted average (normalize by total weight)."""

    MIN = "min"
    """Take minimum component (pessimistic)."""

    MAX = "max"
    """Take maximum component (optimistic)."""

    PRODUCT = "product"
    """Multiply all components (all must be positive)."""

    GATED = "gated"
    """Gate by success: if failed, return 0 regardless of components."""


@dataclass
class RewardAggregator:
    """
    Combines multiple reward functions into a single reward signal.

    This is the main entry point for reward computation. It:
    1. Runs each reward function on the rollout
    2. Aggregates components according to the strategy
    3. Produces a RewardResult for training

    Example:
        aggregator = RewardAggregator(
            rewards=[
                SuccessReward(weight=1.0),
                ToolPenalty(weight=0.1),
                LengthPenalty(weight=0.1),
            ],
            strategy=AggregationStrategy.SUM,
        )

        result = aggregator.compute(rollout)
        print(f"Total reward: {result.reward}")

    For async scorers (like LLM-as-judge):
        result = await aggregator.compute_async(rollout)
    """

    rewards: list[RewardFunction] = field(default_factory=list)
    strategy: AggregationStrategy = AggregationStrategy.SUM

    # Optional base reward
    base_reward: float = 0.0

    # Success gating
    success_gate: bool = False
    """If True and rollout failed, return 0 regardless of components."""

    failure_reward: float = 0.0
    """Reward to return on failure when success_gate is True."""

    def add(self, reward_fn: RewardFunction) -> "RewardAggregator":
        """Add a reward function (fluent interface)."""
        self.rewards.append(reward_fn)
        return self

    def compute(self, rollout: RolloutResult) -> RewardResult:
        """
        Compute aggregated reward synchronously.

        Args:
            rollout: Complete rollout result.

        Returns:
            RewardResult with final reward and all components.
        """
        # Check success gate
        if self.success_gate and not rollout.success:
            return RewardResult(
                reward=self.failure_reward,
                success=False,
                terminated=rollout.metrics.natural_termination,
                components=[],
                metrics=RewardMetrics(
                    raw_total=self.failure_reward,
                    weighted_total=self.failure_reward,
                ),
            )

        # Compute all components
        components: list[RewardComponent] = []
        for reward_fn in self.rewards:
            component = reward_fn.compute(rollout)
            components.append(component)

        # Aggregate
        final_reward = self._aggregate(components)

        # Build metrics
        metrics = self._build_metrics(components, final_reward)

        return RewardResult(
            reward=final_reward,
            success=rollout.success,
            terminated=rollout.metrics.natural_termination,
            components=components,
            metrics=metrics,
        )

    async def compute_async(self, rollout: RolloutResult) -> RewardResult:
        """
        Compute aggregated reward asynchronously.

        Use this when you have async scorers (e.g., LLM-as-judge).

        Args:
            rollout: Complete rollout result.

        Returns:
            RewardResult with final reward and all components.
        """
        import asyncio

        # Check success gate
        if self.success_gate and not rollout.success:
            return RewardResult(
                reward=self.failure_reward,
                success=False,
                terminated=rollout.metrics.natural_termination,
                components=[],
                metrics=RewardMetrics(
                    raw_total=self.failure_reward,
                    weighted_total=self.failure_reward,
                ),
            )

        # Compute all components concurrently
        tasks = [reward_fn.compute_async(rollout) for reward_fn in self.rewards]
        components = await asyncio.gather(*tasks)

        # Aggregate
        final_reward = self._aggregate(list(components))

        # Build metrics
        metrics = self._build_metrics(list(components), final_reward)

        return RewardResult(
            reward=final_reward,
            success=rollout.success,
            terminated=rollout.metrics.natural_termination,
            components=list(components),
            metrics=metrics,
        )

    def _aggregate(self, components: list[RewardComponent]) -> float:
        """Aggregate components according to strategy."""
        if not components:
            return self.base_reward

        match self.strategy:
            case AggregationStrategy.SUM:
                return self.base_reward + sum(c.weighted_value() for c in components)

            case AggregationStrategy.WEIGHTED_AVERAGE:
                total_weight = sum(c.weight for c in components)
                if total_weight == 0:
                    return self.base_reward
                weighted_sum = sum(c.weighted_value() for c in components)
                return self.base_reward + weighted_sum / total_weight

            case AggregationStrategy.MIN:
                return self.base_reward + min(c.weighted_value() for c in components)

            case AggregationStrategy.MAX:
                return self.base_reward + max(c.weighted_value() for c in components)

            case AggregationStrategy.PRODUCT:
                result = 1.0
                for c in components:
                    result *= c.weighted_value()
                return self.base_reward + result

            case AggregationStrategy.GATED:
                # Use first component as gate
                if components[0].value <= 0:
                    return self.base_reward
                return self.base_reward + sum(
                    c.weighted_value() for c in components[1:]
                )

            case _:
                return self.base_reward + sum(c.weighted_value() for c in components)

    def _build_metrics(
        self, components: list[RewardComponent], final_reward: float
    ) -> RewardMetrics:
        """Build RewardMetrics from components."""
        metrics = RewardMetrics()

        for comp in components:
            metrics.component_values[comp.name] = comp.value
            metrics.component_weights[comp.name] = comp.weight

        metrics.raw_total = sum(c.value for c in components)
        metrics.weighted_total = final_reward

        return metrics

    def compute_batch(
        self, rollouts: Sequence[RolloutResult]
    ) -> list[RewardResult]:
        """
        Compute rewards for a batch of rollouts.

        Args:
            rollouts: List of rollout results.

        Returns:
            List of reward results.
        """
        return [self.compute(r) for r in rollouts]

    async def compute_batch_async(
        self, rollouts: Sequence[RolloutResult]
    ) -> list[RewardResult]:
        """
        Compute rewards for a batch of rollouts asynchronously.

        Args:
            rollouts: List of rollout results.

        Returns:
            List of reward results.
        """
        import asyncio

        return await asyncio.gather(*[self.compute_async(r) for r in rollouts])


def create_standard_reward(
    success_weight: float = 1.0,
    tool_penalty_weight: float = 0.1,
    length_penalty_weight: float = 0.0,
    turn_penalty_weight: float = 0.0,
) -> RewardAggregator:
    """
    Create a standard reward aggregator with common components.

    Args:
        success_weight: Weight for binary success reward.
        tool_penalty_weight: Weight for tool usage penalty.
        length_penalty_weight: Weight for response length penalty.
        turn_penalty_weight: Weight for turn count penalty.

    Returns:
        Configured RewardAggregator.
    """
    from dreadnode.core.training.rewards.functions import (
        SuccessReward,
        ToolPenalty,
        LengthPenalty,
        TurnPenalty,
    )

    rewards: list[Any] = [SuccessReward(weight=success_weight)]

    if tool_penalty_weight > 0:
        rewards.append(ToolPenalty(weight=tool_penalty_weight))

    if length_penalty_weight > 0:
        rewards.append(LengthPenalty(weight=length_penalty_weight))

    if turn_penalty_weight > 0:
        rewards.append(TurnPenalty(weight=turn_penalty_weight))

    return RewardAggregator(rewards=rewards)

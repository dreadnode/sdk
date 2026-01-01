"""
Types for reward computation.

Designed to bridge DN SDK Scorers with NeMo RL training.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, NotRequired, TypedDict

import torch


class RewardConfig(TypedDict):
    """Configuration for reward computation."""

    # Scaling
    enabled: NotRequired[bool]
    source_min: NotRequired[float]
    source_max: NotRequired[float]
    target_min: NotRequired[float]
    target_max: NotRequired[float]

    # DAPO-style penalty (matches NeMo RL)
    overlong_penalty_enabled: NotRequired[bool]
    overlong_buffer_length: NotRequired[int]
    overlong_buffer_penalty: NotRequired[float]
    max_response_length: NotRequired[int]

    # Clipping
    clip_min: NotRequired[float]
    clip_max: NotRequired[float]

    # Normalization
    normalize: NotRequired[bool]
    normalize_mean: NotRequired[float]
    normalize_std: NotRequired[float]


@dataclass
class RewardComponent:
    """
    A single reward component with metadata.

    Analogous to DN SDK's Metric but for rewards.
    """

    name: str
    """Name of the reward source."""

    value: float
    """The reward value."""

    weight: float = 1.0
    """Weight for aggregation."""

    step: int | None = None
    """Rollout step where this was computed."""

    rationale: str | None = None
    """Optional explanation (from LLM-as-judge scorers)."""

    attributes: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    timestamp: datetime | None = None
    """When this reward was computed."""

    def weighted_value(self) -> float:
        """Return value * weight."""
        return self.value * self.weight

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "weight": self.weight,
            "weighted_value": self.weighted_value(),
            "step": self.step,
            "rationale": self.rationale,
            "attributes": self.attributes,
        }


@dataclass
class RewardMetrics:
    """Aggregated metrics from reward computation."""

    # Per-component
    component_values: dict[str, float] = field(default_factory=dict)
    component_weights: dict[str, float] = field(default_factory=dict)

    # Aggregated
    raw_total: float = 0.0
    weighted_total: float = 0.0

    # After shaping
    shaped_reward: float | None = None
    scaling_applied: bool = False
    penalty_applied: bool = False

    # For GRPO
    baseline: float | None = None
    advantage: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "components": self.component_values,
            "weights": self.component_weights,
            "raw_total": self.raw_total,
            "weighted_total": self.weighted_total,
            "shaped_reward": self.shaped_reward,
            "baseline": self.baseline,
            "advantage": self.advantage,
        }


@dataclass
class RewardResult:
    """
    Complete result from reward computation.

    Contains the final reward value plus all components and metrics.
    """

    # Final reward (after all shaping/aggregation)
    reward: float

    # Components that contributed
    components: list[RewardComponent] = field(default_factory=list)

    # Aggregated metrics
    metrics: RewardMetrics = field(default_factory=RewardMetrics)

    # For training integration
    success: bool = False
    """Whether the task was successful (binary reward signal)."""

    terminated: bool = False
    """Whether the rollout terminated naturally."""

    # Tensor format for NeMo RL
    _reward_tensor: torch.Tensor | None = None

    @property
    def reward_tensor(self) -> torch.Tensor:
        """Get reward as tensor for training."""
        if self._reward_tensor is None:
            self._reward_tensor = torch.tensor(self.reward, dtype=torch.float32)
        return self._reward_tensor

    def component_by_name(self, name: str) -> RewardComponent | None:
        """Get a specific component by name."""
        for comp in self.components:
            if comp.name == name:
                return comp
        return None

    def total_weight(self) -> float:
        """Sum of all component weights."""
        return sum(c.weight for c in self.components)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "reward": self.reward,
            "success": self.success,
            "terminated": self.terminated,
            "components": [c.to_dict() for c in self.components],
            "metrics": self.metrics.to_dict(),
        }

    def to_training_dict(self) -> dict[str, Any]:
        """
        Convert to format expected by NeMo RL.

        Compatible with BatchedDataDict structure.
        """
        return {
            "total_reward": self.reward,
            "success": float(self.success),
            "terminated": float(self.terminated),
            "reward_components": {
                c.name: c.weighted_value() for c in self.components
            },
        }

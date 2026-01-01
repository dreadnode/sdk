"""
Base trainer infrastructure.

Provides common interfaces and utilities for all trainers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, NotRequired, Protocol, Sequence, TypedDict

from dreadnode.core.training.rewards.aggregator import RewardAggregator


class TrainingConfig(TypedDict):
    """Common configuration for all trainers."""

    # Model
    model_name: str
    """Model name or path."""

    tokenizer_name: NotRequired[str]
    """Tokenizer name (defaults to model_name)."""

    # Training parameters
    learning_rate: NotRequired[float]
    """Learning rate (default: 1e-5)."""

    batch_size: NotRequired[int]
    """Batch size per device."""

    gradient_accumulation_steps: NotRequired[int]
    """Gradient accumulation steps."""

    max_steps: NotRequired[int]
    """Maximum training steps."""

    warmup_steps: NotRequired[int]
    """Warmup steps."""

    weight_decay: NotRequired[float]
    """Weight decay."""

    # Rollout parameters
    max_rollout_turns: NotRequired[int]
    """Maximum turns per rollout."""

    max_tokens_per_turn: NotRequired[int]
    """Maximum tokens per generation."""

    # Generation
    temperature: NotRequired[float]
    """Sampling temperature."""

    top_p: NotRequired[float]
    """Top-p sampling."""

    # Logging
    log_dir: NotRequired[str]
    """Directory for logs."""

    log_interval: NotRequired[int]
    """Steps between logging."""

    eval_interval: NotRequired[int]
    """Steps between evaluation."""

    # Checkpointing
    checkpoint_dir: NotRequired[str]
    """Directory for checkpoints."""

    checkpoint_interval: NotRequired[int]
    """Steps between checkpoints."""

    save_total_limit: NotRequired[int]
    """Maximum checkpoints to keep."""

    # Distributed
    num_workers: NotRequired[int]
    """Number of data loading workers."""

    seed: NotRequired[int]
    """Random seed."""


@dataclass
class TrainingState:
    """Current state of training."""

    step: int = 0
    """Current training step."""

    epoch: int = 0
    """Current epoch."""

    samples_seen: int = 0
    """Total samples processed."""

    best_reward: float = float("-inf")
    """Best reward achieved."""

    total_reward: float = 0.0
    """Cumulative reward."""

    started_at: datetime | None = None
    """Training start time."""

    metrics: dict[str, float] = field(default_factory=dict)
    """Current metrics."""

    def elapsed_seconds(self) -> float:
        """Seconds since training started."""
        if self.started_at is None:
            return 0.0
        return (datetime.now() - self.started_at).total_seconds()


class TrainingCallback(Protocol):
    """Protocol for training callbacks."""

    def on_step_start(self, state: TrainingState) -> None:
        """Called at the start of each step."""
        ...

    def on_step_end(self, state: TrainingState, metrics: dict[str, float]) -> None:
        """Called at the end of each step."""
        ...

    def on_epoch_start(self, state: TrainingState) -> None:
        """Called at the start of each epoch."""
        ...

    def on_epoch_end(self, state: TrainingState, metrics: dict[str, float]) -> None:
        """Called at the end of each epoch."""
        ...

    def on_evaluation(self, state: TrainingState, metrics: dict[str, float]) -> None:
        """Called after evaluation."""
        ...


class BaseTrainer(ABC):
    """
    Base class for all trainers.

    Provides common infrastructure for training DN agents.
    """

    def __init__(
        self,
        config: TrainingConfig,
        rewards: RewardAggregator | None = None,
        callbacks: Sequence[TrainingCallback] | None = None,
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration.
            rewards: Reward aggregator for RL training.
            callbacks: Optional training callbacks.
        """
        self.config = config
        self.rewards = rewards
        self.callbacks = list(callbacks) if callbacks else []
        self.state = TrainingState()

    @abstractmethod
    async def train(
        self,
        prompts: Sequence[str],
        environment: Any | None = None,
        num_steps: int | None = None,
    ) -> TrainingState:
        """
        Run training.

        Args:
            prompts: Training prompts.
            environment: Optional environment for rollouts.
            num_steps: Number of training steps.

        Returns:
            Final training state.
        """
        ...

    @abstractmethod
    async def evaluate(
        self,
        prompts: Sequence[str],
        environment: Any | None = None,
    ) -> dict[str, float]:
        """
        Run evaluation.

        Args:
            prompts: Evaluation prompts.
            environment: Optional environment.

        Returns:
            Evaluation metrics.
        """
        ...

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        ...

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        ...

    def add_callback(self, callback: TrainingCallback) -> None:
        """Add a training callback."""
        self.callbacks.append(callback)

    def _notify_step_start(self) -> None:
        """Notify callbacks of step start."""
        for callback in self.callbacks:
            callback.on_step_start(self.state)

    def _notify_step_end(self, metrics: dict[str, float]) -> None:
        """Notify callbacks of step end."""
        for callback in self.callbacks:
            callback.on_step_end(self.state, metrics)

    def _notify_epoch_start(self) -> None:
        """Notify callbacks of epoch start."""
        for callback in self.callbacks:
            callback.on_epoch_start(self.state)

    def _notify_epoch_end(self, metrics: dict[str, float]) -> None:
        """Notify callbacks of epoch end."""
        for callback in self.callbacks:
            callback.on_epoch_end(self.state, metrics)

    def _notify_evaluation(self, metrics: dict[str, float]) -> None:
        """Notify callbacks of evaluation."""
        for callback in self.callbacks:
            callback.on_evaluation(self.state, metrics)

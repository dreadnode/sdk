"""
Async GRPO Coordinator for orchestrating distributed training.

This module provides the AsyncGRPOCoordinator that manages the async
training pipeline following the OpenRLHF architecture:
- RolloutWorkers generate experiences continuously
- ExperienceBuffer decouples generation and training
- Learner trains on buffered experiences
- Weight sync keeps generation on-policy (configurable staleness)

Usage:
    from dreadnode.core.training.ray.coordinator import AsyncGRPOCoordinator

    coordinator = AsyncGRPOCoordinator(
        config=grpo_config,
        prompts=training_prompts,
        reward_fn=my_reward_fn,
        num_rollout_workers=2,
        buffer_size=2,
    )

    state = await coordinator.train(num_steps=1000)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import ray

from dreadnode.core.training.ray.config import RayGRPOConfig
from dreadnode.core.training.ray.experience import ExperienceBatch, ExperienceBuffer
from dreadnode.core.training.ray.fsdp2_learner import FSDP2Config, FSDP2Learner
from dreadnode.core.training.ray.rollout_worker import (
    RewardFunction,
    RolloutWorker,
    RolloutWorkerPool,
)

if TYPE_CHECKING:
    from dreadnode.core.storage.storage import Storage


@dataclass
class TrainingState:
    """Current state of async training."""

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

    # Async-specific
    generation_step: int = 0
    """Current generation step."""

    buffer_fill_ratio: float = 0.0
    """Experience buffer fill ratio."""

    weight_sync_count: int = 0
    """Number of weight syncs performed."""

    def elapsed_seconds(self) -> float:
        """Seconds since training started."""
        if self.started_at is None:
            return 0.0
        return (datetime.now() - self.started_at).total_seconds()

    def samples_per_second(self) -> float:
        """Training throughput."""
        elapsed = self.elapsed_seconds()
        if elapsed <= 0:
            return 0.0
        return self.samples_seen / elapsed


@dataclass
class AsyncGRPOConfig:
    """Configuration for async GRPO training."""

    # Async parameters
    num_rollout_workers: int = 1
    """Number of parallel rollout workers."""

    buffer_size: int = 1
    """Experience buffer size (off-policy degree)."""

    weight_sync_interval: int = 1
    """Steps between weight syncs to rollout workers."""

    packed: bool = False
    """If True, multiple vLLM workers share the same GPU(s)."""

    # Training control
    max_steps: int = 1000
    """Maximum training steps."""

    max_epochs: int = 10
    """Maximum training epochs."""

    prompts_per_step: int = 8
    """Prompts to sample per generation step."""

    # Logging
    log_interval: int = 10
    """Steps between logging."""

    eval_interval: int = 100
    """Steps between evaluation."""

    checkpoint_interval: int = 100
    """Steps between checkpoints."""

    checkpoint_dir: str = "./checkpoints"
    """Directory for checkpoints."""


class AsyncGRPOCoordinator:
    """
    Coordinates async rollout generation and training.

    Based on OpenRLHF's async architecture, this coordinator:
    1. Manages RolloutWorkerPool for experience generation
    2. Uses ExperienceBuffer for async decoupling
    3. Runs FSDP2Learner for distributed training
    4. Handles weight synchronization between learner and workers

    The async design allows overlapped generation and training:
    - Workers generate experiences continuously
    - Buffer stores N batches (configurable off-policy degree)
    - Learner trains on buffered experiences
    - Weight sync keeps generation approximately on-policy

    Attributes:
        config: GRPO configuration
        async_config: Async-specific configuration
        prompts: Training prompts
        reward_fn: Reward function
        buffer: Experience buffer
        worker_pool: Rollout worker pool
        learner: FSDP2 learner
    """

    def __init__(
        self,
        config: RayGRPOConfig,
        prompts: Sequence[str],
        reward_fn: RewardFunction,
        async_config: AsyncGRPOConfig | None = None,
        fsdp_config: FSDP2Config | None = None,
        num_rollout_workers: int | None = None,
        buffer_size: int | None = None,
        weight_sync_interval: int | None = None,
        packed: bool | None = None,
        storage: Storage | None = None,
        checkpoint_name: str | None = None,
    ) -> None:
        """
        Initialize async coordinator.

        Args:
            config: GRPO configuration
            prompts: Training prompts
            reward_fn: Reward function (prompts, completions) -> rewards
            async_config: Async configuration (overrides other params)
            fsdp_config: FSDP2 configuration
            num_rollout_workers: Number of workers (overrides async_config)
            buffer_size: Buffer size (overrides async_config)
            weight_sync_interval: Sync interval (overrides async_config)
            packed: If True, multiple vLLM workers share GPU(s) (overrides async_config)
            storage: Optional Storage for CAS-based checkpointing
            checkpoint_name: Name for checkpoints (defaults to model name)
        """
        self.config = config
        self.prompts = list(prompts)
        self.reward_fn = reward_fn
        self.fsdp_config = fsdp_config or FSDP2Config()
        self.storage = storage
        self.checkpoint_name = checkpoint_name

        # Merge async config with explicit parameters
        self.async_config = async_config or AsyncGRPOConfig()
        if num_rollout_workers is not None:
            self.async_config.num_rollout_workers = num_rollout_workers
        if buffer_size is not None:
            self.async_config.buffer_size = buffer_size
        if weight_sync_interval is not None:
            self.async_config.weight_sync_interval = weight_sync_interval
        if packed is not None:
            self.async_config.packed = packed

        # Initialize components
        self._init_components()

        # Training state
        self.state = TrainingState()

        # Control flags
        self._running = False
        self._generation_task: asyncio.Task | None = None

    def _init_components(self) -> None:
        """Initialize buffer, workers, and learner."""
        # Experience buffer
        self.buffer = ExperienceBuffer(
            max_size=self.async_config.buffer_size,
            batch_size=self.config.train_batch_size,
        )

        # Rollout worker pool
        self.worker_pool = RolloutWorkerPool(
            config=self.config,
            reward_fn=self.reward_fn,
            num_workers=self.async_config.num_rollout_workers,
            packed=self.async_config.packed,
        )

        # Calculate learner GPU
        gpus_per_worker = self.config.vllm.tensor_parallel_size
        if self.async_config.packed:
            # Packed mode: all workers share GPU(s) 0..TP-1, learner uses next GPU
            learner_gpu = gpus_per_worker
        else:
            # Normal mode: workers use GPUs 0..N*TP-1, learner uses next GPU
            learner_gpu = self.async_config.num_rollout_workers * gpus_per_worker

        # FSDP2 learner uses GPU after rollout workers
        self.learner = FSDP2Learner(
            model_name=self.config.model_name,
            config=self.config,
            fsdp_config=self.fsdp_config,
            world_size=1,
            rank=learner_gpu,  # Use GPU index after rollout workers
            storage=self.storage,
            checkpoint_name=self.checkpoint_name,
        )

    async def train(
        self,
        num_steps: int | None = None,
        callbacks: list[Callable[[TrainingState], None]] | None = None,
    ) -> TrainingState:
        """
        Run async training loop.

        Args:
            num_steps: Number of training steps (overrides config)
            callbacks: Optional callbacks called after each step

        Returns:
            Final training state
        """
        num_steps = num_steps or self.async_config.max_steps
        callbacks = callbacks or []

        self.state.started_at = datetime.now()
        self._running = True

        # Create scheduler
        self.learner.create_scheduler(total_steps=num_steps)

        # Start generation task
        self._generation_task = asyncio.create_task(self._generation_loop())

        try:
            for step in range(num_steps):
                self.state.step = step

                # Get batch from buffer (blocks if empty)
                batch = await self.buffer.get()

                # Training step
                metrics = self.learner.train_step(batch)
                self._update_state(metrics, batch)

                # Weight sync to rollout workers
                if step > 0 and step % self.async_config.weight_sync_interval == 0:
                    await self._sync_weights()

                # Logging
                if step % self.async_config.log_interval == 0:
                    self._log_metrics()

                # Checkpointing
                if step > 0 and step % self.async_config.checkpoint_interval == 0:
                    self._save_checkpoint()

                # Callbacks
                for callback in callbacks:
                    callback(self.state)

                # Check for early stopping or other conditions
                if not self._running:
                    break

        finally:
            self._running = False
            if self._generation_task is not None:
                self._generation_task.cancel()
                try:
                    await self._generation_task
                except asyncio.CancelledError:
                    pass

        return self.state

    async def _generation_loop(self) -> None:
        """Continuously generate experiences in background."""
        prompt_idx = 0

        while self._running:
            # Get batch of prompts
            prompts = self._get_prompt_batch(prompt_idx)
            prompt_idx = (prompt_idx + len(prompts)) % len(self.prompts)

            # Generate experiences
            try:
                batch = self.worker_pool.generate_batch(
                    prompts=prompts,
                    num_generations_per_prompt=self.config.num_generations_per_prompt,
                )

                # Add to buffer (blocks if full)
                await self.buffer.put(batch)
                self.state.generation_step += 1

            except Exception as e:
                print(f"Generation error: {e}")
                await asyncio.sleep(0.1)

            # Update buffer stats
            self.state.buffer_fill_ratio = self.buffer.size / self.buffer.max_size

            # Yield control to allow training
            await asyncio.sleep(0)

    def _get_prompt_batch(self, start_idx: int) -> list[str]:
        """Get a batch of prompts starting from index."""
        batch_size = self.async_config.prompts_per_step
        end_idx = start_idx + batch_size

        if end_idx <= len(self.prompts):
            return self.prompts[start_idx:end_idx]
        else:
            # Wrap around
            batch = self.prompts[start_idx:]
            batch += self.prompts[: end_idx - len(self.prompts)]
            return batch

    async def _sync_weights(self) -> None:
        """Sync learner weights to rollout workers.

        Uses checkpoint-based sync for reliability with vLLM v1.
        Falls back to state_dict transfer if checkpoint fails.
        """
        import os
        import tempfile

        # Save checkpoint for weight sync (more reliable than state_dict transfer)
        checkpoint_dir = os.path.join(
            tempfile.gettempdir(),
            f"grpo_weight_sync_{id(self)}",
        )

        try:
            # Save model for vLLM to load
            self.learner.save_for_vllm(checkpoint_dir)

            # Update workers via checkpoint path
            success = self.worker_pool.update_weights(checkpoint_path=checkpoint_dir)
            if success:
                self.state.weight_sync_count += 1
                return

        except Exception as e:
            print(f"Checkpoint-based weight sync failed: {e}, trying state_dict...")

        # Fall back to state dict method
        state_dict = self.learner.get_state_dict()
        if state_dict:
            success = self.worker_pool.update_weights(state_dict=state_dict)
            if success:
                self.state.weight_sync_count += 1

    def _update_state(self, metrics: dict[str, float], batch: ExperienceBatch) -> None:
        """Update training state with metrics."""
        self.state.metrics.update(metrics)
        self.state.samples_seen += len(batch)

        # Update reward tracking
        if batch.rewards is not None:
            reward_mean = batch.rewards.mean().item()
        elif batch.experiences:
            # Compute from individual experiences if tensor not available
            rewards = [exp.reward for exp in batch.experiences]
            reward_mean = sum(rewards) / len(rewards)
        else:
            reward_mean = None

        if reward_mean is not None:
            self.state.total_reward += reward_mean * len(batch)
            if reward_mean > self.state.best_reward:
                self.state.best_reward = reward_mean

    def _log_metrics(self) -> None:
        """Log training metrics."""
        metrics = {
            "step": self.state.step,
            "generation_step": self.state.generation_step,
            "samples_seen": self.state.samples_seen,
            "samples_per_second": self.state.samples_per_second(),
            "buffer_fill": self.state.buffer_fill_ratio,
            "weight_syncs": self.state.weight_sync_count,
            "best_reward": self.state.best_reward,
            **self.state.metrics,
        }

        # Simple print logging (can be replaced with wandb, etc.)
        metric_str = " | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items())
        print(f"[Step {self.state.step}] {metric_str}")

    def _save_checkpoint(self) -> None:
        """Save training checkpoint."""
        import os

        # Save to CAS if storage is available
        if self.storage is not None:
            local_model = self.learner.save_checkpoint_to_storage()
            if local_model:
                print(f"Saved checkpoint to CAS: {local_model.name} v{local_model.version}")
                return

        # Fall back to file-based checkpoint
        checkpoint_path = os.path.join(
            self.async_config.checkpoint_dir,
            f"step_{self.state.step}",
        )
        self.learner.save_checkpoint(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    def stop(self) -> None:
        """Stop training gracefully."""
        self._running = False

    def shutdown(self) -> None:
        """Shutdown all components."""
        self.stop()
        self.worker_pool.shutdown()

    def get_model(self) -> Any:
        """Get the trained model."""
        return self.learner.model

    def get_state_dict(self) -> dict:
        """Get model state dict."""
        return self.learner.get_state_dict()


class SyncGRPOCoordinator:
    """
    Synchronous GRPO coordinator (simpler, for debugging).

    Unlike AsyncGRPOCoordinator, this runs generation and training
    sequentially in each step. Useful for debugging and small-scale
    experiments.
    """

    def __init__(
        self,
        config: RayGRPOConfig,
        prompts: Sequence[str],
        reward_fn: RewardFunction,
        fsdp_config: FSDP2Config | None = None,
        storage: Storage | None = None,
        checkpoint_name: str | None = None,
    ) -> None:
        """Initialize sync coordinator.

        Args:
            config: GRPO configuration
            prompts: Training prompts
            reward_fn: Reward function
            fsdp_config: FSDP2 configuration
            storage: Optional Storage for CAS-based checkpointing
            checkpoint_name: Name for checkpoints
        """
        self.config = config
        self.prompts = list(prompts)
        self.reward_fn = reward_fn
        self.fsdp_config = fsdp_config or FSDP2Config()
        self.storage = storage
        self.checkpoint_name = checkpoint_name

        # Single worker for generation
        self.worker = RolloutWorker.options(
            num_gpus=config.vllm.tensor_parallel_size,
            name="rollout_worker",
        ).remote(
            config=config,
            reward_fn=reward_fn,
            worker_id=0,
        )

        # Learner
        self.learner = FSDP2Learner(
            model_name=config.model_name,
            config=config,
            fsdp_config=self.fsdp_config,
            world_size=1,
            rank=0,
            storage=storage,
            checkpoint_name=checkpoint_name,
        )

        self.state = TrainingState()

    def train(
        self,
        num_steps: int | None = None,
        callbacks: list[Callable[[TrainingState], None]] | None = None,
    ) -> TrainingState:
        """
        Run synchronous training loop.

        Args:
            num_steps: Number of training steps
            callbacks: Optional callbacks

        Returns:
            Final training state
        """
        num_steps = num_steps or self.config.max_steps
        callbacks = callbacks or []

        self.state.started_at = datetime.now()
        self.learner.create_scheduler(total_steps=num_steps)

        prompt_idx = 0

        for step in range(num_steps):
            self.state.step = step

            # Get prompts
            prompts = self.prompts[
                prompt_idx : prompt_idx + self.config.num_prompts_per_step
            ]
            prompt_idx = (
                prompt_idx + self.config.num_prompts_per_step
            ) % len(self.prompts)

            # Generate experiences
            batch = ray.get(
                self.worker.generate_batch.remote(
                    prompts=prompts,
                    num_generations_per_prompt=self.config.num_generations_per_prompt,
                )
            )

            # Train
            metrics = self.learner.train_step(batch)
            self.state.metrics.update(metrics)
            self.state.samples_seen += len(batch)

            # Update rewards
            if batch.rewards is not None:
                reward_mean = batch.rewards.mean().item()
                if reward_mean > self.state.best_reward:
                    self.state.best_reward = reward_mean

            # Weight sync every step for on-policy
            ray.get(self.worker.update_weights.remote(self.learner.get_state_dict()))
            self.state.weight_sync_count += 1

            # Logging
            if step % 10 == 0:
                print(
                    f"[Step {step}] loss: {metrics.get('loss', 0):.4f} | "
                    f"reward: {self.state.best_reward:.4f}"
                )

            # Callbacks
            for callback in callbacks:
                callback(self.state)

        return self.state

    def shutdown(self) -> None:
        """Shutdown components."""
        ray.get(self.worker.shutdown.remote())
        ray.kill(self.worker)

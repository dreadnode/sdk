"""
Experience data structures for async RL training.

This module provides the core data structures for managing training experiences
in async GRPO/RL pipelines, following the OpenRLHF pattern.

Key components:
- Experience: Single training experience from rollout
- ExperienceBatch: Batch of experiences ready for training
- ExperienceBuffer: Async queue for decoupling generation and training

Usage:
    from dreadnode.core.training.ray.experience import (
        Experience,
        ExperienceBatch,
        ExperienceBuffer,
    )

    # Create experiences during rollout
    exp = Experience(
        prompt="What is 2+2?",
        prompt_ids=prompt_ids,
        completion="4",
        completion_ids=completion_ids,
        reward=1.0,
    )

    # Batch and buffer for async training
    buffer = ExperienceBuffer(max_size=2)
    await buffer.put(ExperienceBatch(experiences=[exp]))
    batch = await buffer.get()
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class Experience:
    """
    Single training experience from a rollout.

    Contains all information needed to compute policy gradient loss:
    - Token IDs for prompt and completion
    - Log probabilities from the generation model
    - Reward and advantage for the completion
    - Optional group ID for GRPO leave-one-out baseline
    """

    # Text
    prompt: str
    """Original prompt text."""

    completion: str
    """Generated completion text."""

    # Token IDs
    prompt_ids: torch.Tensor
    """Token IDs for the prompt."""

    completion_ids: torch.Tensor
    """Token IDs for the completion."""

    full_ids: torch.Tensor | None = None
    """Full sequence (prompt + completion) token IDs."""

    # Masks
    completion_mask: torch.Tensor | None = None
    """Mask indicating completion tokens (1 for completion, 0 for prompt)."""

    attention_mask: torch.Tensor | None = None
    """Attention mask for the full sequence."""

    # Log probabilities
    logprobs: torch.Tensor | None = None
    """Per-token log probabilities from generation model."""

    ref_logprobs: torch.Tensor | None = None
    """Per-token log probabilities from reference model (for KL penalty)."""

    # Rewards and advantages
    reward: float = 0.0
    """Total reward for this completion."""

    advantage: float | None = None
    """Computed advantage (after baseline subtraction)."""

    # GRPO grouping
    group_id: int | None = None
    """Group ID for GRPO leave-one-out baseline computation."""

    # Metadata
    metadata: dict[str, Any] | None = None
    """Optional metadata (e.g., tool calls, intermediate rewards)."""

    def __post_init__(self) -> None:
        """Compute derived fields if not provided."""
        if self.full_ids is None and self.prompt_ids is not None and self.completion_ids is not None:
            self.full_ids = torch.cat([self.prompt_ids, self.completion_ids], dim=-1)

        if self.completion_mask is None and self.full_ids is not None and self.prompt_ids is not None:
            prompt_len = self.prompt_ids.shape[-1]
            total_len = self.full_ids.shape[-1]
            self.completion_mask = torch.zeros(total_len, dtype=torch.bool)
            self.completion_mask[prompt_len:] = True

    def to_device(self, device: str | torch.device) -> Experience:
        """Move all tensors to the specified device."""

        def move(t: torch.Tensor | None) -> torch.Tensor | None:
            return t.to(device) if t is not None else None

        return Experience(
            prompt=self.prompt,
            completion=self.completion,
            prompt_ids=move(self.prompt_ids),  # type: ignore
            completion_ids=move(self.completion_ids),  # type: ignore
            full_ids=move(self.full_ids),
            completion_mask=move(self.completion_mask),
            attention_mask=move(self.attention_mask),
            logprobs=move(self.logprobs),
            ref_logprobs=move(self.ref_logprobs),
            reward=self.reward,
            advantage=self.advantage,
            group_id=self.group_id,
            metadata=self.metadata,
        )


@dataclass
class ExperienceBatch:
    """
    Batch of experiences ready for training.

    Contains both the raw experiences and optionally the tensorized batch
    for efficient GPU training. Tensorization is done lazily via `to_tensors()`.
    """

    experiences: list[Experience]
    """List of individual experiences."""

    # Tensorized batch (created via to_tensors)
    input_ids: torch.Tensor | None = None
    """Padded input IDs [batch_size, seq_len]."""

    attention_mask: torch.Tensor | None = None
    """Attention mask [batch_size, seq_len]."""

    completion_mask: torch.Tensor | None = None
    """Completion mask [batch_size, seq_len]."""

    logprobs: torch.Tensor | None = None
    """Per-token log probabilities [batch_size, seq_len]."""

    ref_logprobs: torch.Tensor | None = None
    """Reference model log probabilities [batch_size, seq_len]."""

    advantages: torch.Tensor | None = None
    """Advantages [batch_size]."""

    rewards: torch.Tensor | None = None
    """Rewards [batch_size]."""

    group_ids: torch.Tensor | None = None
    """Group IDs for GRPO [batch_size]."""

    def __len__(self) -> int:
        return len(self.experiences)

    @property
    def is_tensorized(self) -> bool:
        """Check if batch has been tensorized."""
        return self.input_ids is not None

    def to_tensors(
        self,
        tokenizer: Any,
        device: str | torch.device = "cuda",
        max_length: int | None = None,
        padding_side: str = "right",
    ) -> ExperienceBatch:
        """
        Convert experiences to padded tensors for training.

        Args:
            tokenizer: Tokenizer for padding
            device: Target device
            max_length: Maximum sequence length (truncates if needed)
            padding_side: Padding side ('left' or 'right')

        Returns:
            New ExperienceBatch with tensorized data
        """
        if not self.experiences:
            return self

        # Collect all full_ids
        all_ids = [exp.full_ids for exp in self.experiences]
        all_completion_masks = [exp.completion_mask for exp in self.experiences]
        all_logprobs = [exp.logprobs for exp in self.experiences if exp.logprobs is not None]
        all_ref_logprobs = [exp.ref_logprobs for exp in self.experiences if exp.ref_logprobs is not None]

        # Find max length
        lengths = [ids.shape[-1] for ids in all_ids if ids is not None]
        if not lengths:
            return self

        pad_length = max(lengths)
        if max_length is not None:
            pad_length = min(pad_length, max_length)

        batch_size = len(self.experiences)
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

        # Initialize tensors
        input_ids = torch.full((batch_size, pad_length), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, pad_length), dtype=torch.long)
        completion_mask = torch.zeros((batch_size, pad_length), dtype=torch.bool)

        # Pad sequences
        for i, exp in enumerate(self.experiences):
            if exp.full_ids is None:
                continue

            seq_len = min(exp.full_ids.shape[-1], pad_length)

            if padding_side == "right":
                input_ids[i, :seq_len] = exp.full_ids[:seq_len]
                attention_mask[i, :seq_len] = 1
                if exp.completion_mask is not None:
                    completion_mask[i, :seq_len] = exp.completion_mask[:seq_len]
            else:
                start = pad_length - seq_len
                input_ids[i, start:] = exp.full_ids[:seq_len]
                attention_mask[i, start:] = 1
                if exp.completion_mask is not None:
                    completion_mask[i, start:] = exp.completion_mask[:seq_len]

        # Pad logprobs if available
        logprobs_tensor = None
        if all_logprobs and len(all_logprobs) == batch_size:
            logprobs_tensor = torch.zeros((batch_size, pad_length), dtype=torch.float32)
            for i, lp in enumerate(all_logprobs):
                if lp is not None:
                    seq_len = min(lp.shape[-1], pad_length)
                    if padding_side == "right":
                        logprobs_tensor[i, :seq_len] = lp[:seq_len]
                    else:
                        start = pad_length - seq_len
                        logprobs_tensor[i, start:] = lp[:seq_len]

        # Pad ref_logprobs if available
        ref_logprobs_tensor = None
        if all_ref_logprobs and len(all_ref_logprobs) == batch_size:
            ref_logprobs_tensor = torch.zeros((batch_size, pad_length), dtype=torch.float32)
            for i, lp in enumerate(all_ref_logprobs):
                if lp is not None:
                    seq_len = min(lp.shape[-1], pad_length)
                    if padding_side == "right":
                        ref_logprobs_tensor[i, :seq_len] = lp[:seq_len]
                    else:
                        start = pad_length - seq_len
                        ref_logprobs_tensor[i, start:] = lp[:seq_len]

        # Collect scalar values
        advantages = torch.tensor(
            [exp.advantage if exp.advantage is not None else 0.0 for exp in self.experiences],
            dtype=torch.float32,
        )
        rewards = torch.tensor([exp.reward for exp in self.experiences], dtype=torch.float32)
        group_ids = torch.tensor(
            [exp.group_id if exp.group_id is not None else -1 for exp in self.experiences],
            dtype=torch.long,
        )

        return ExperienceBatch(
            experiences=self.experiences,
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            completion_mask=completion_mask.to(device),
            logprobs=logprobs_tensor.to(device) if logprobs_tensor is not None else None,
            ref_logprobs=ref_logprobs_tensor.to(device) if ref_logprobs_tensor is not None else None,
            advantages=advantages.to(device),
            rewards=rewards.to(device),
            group_ids=group_ids.to(device),
        )

    def compute_advantages(self, use_leave_one_out: bool = True) -> ExperienceBatch:
        """
        Compute GRPO advantages with optional leave-one-out baseline.

        For each group of completions from the same prompt, computes:
        - Standard: advantage = reward - mean(group_rewards)
        - Leave-one-out: advantage = reward - mean(other_rewards)

        Args:
            use_leave_one_out: Use leave-one-out baseline (GRPO default)

        Returns:
            Self with computed advantages
        """
        # Group experiences by group_id
        groups: dict[int, list[Experience]] = {}
        for exp in self.experiences:
            gid = exp.group_id if exp.group_id is not None else 0
            groups.setdefault(gid, []).append(exp)

        for group_exps in groups.values():
            rewards = torch.tensor([e.reward for e in group_exps])
            n = len(rewards)

            if n <= 1:
                # Single sample, no baseline
                for exp in group_exps:
                    exp.advantage = 0.0
                continue

            if use_leave_one_out:
                # Leave-one-out baseline: baseline_i = mean(rewards except i)
                total = rewards.sum()
                for i, exp in enumerate(group_exps):
                    baseline = (total - rewards[i]) / (n - 1)
                    exp.advantage = (rewards[i] - baseline).item()
            else:
                # Standard baseline: mean of all rewards
                baseline = rewards.mean()
                for i, exp in enumerate(group_exps):
                    exp.advantage = (rewards[i] - baseline).item()

        return self

    @staticmethod
    def merge(batches: Sequence[ExperienceBatch]) -> ExperienceBatch:
        """Merge multiple batches into one."""
        all_experiences = []
        for batch in batches:
            all_experiences.extend(batch.experiences)
        return ExperienceBatch(experiences=all_experiences)


class ExperienceBuffer:
    """
    Async experience buffer for decoupling generation and training.

    Based on OpenRLHF's async architecture, this buffer allows:
    - RolloutWorkers to generate experiences continuously
    - Learner to train on buffered experiences
    - Configurable off-policy degree via max_size

    The buffer controls the trade-off between throughput and staleness:
    - max_size=1: Nearly on-policy (minimal staleness)
    - max_size=N: More off-policy (better throughput, more staleness)

    Usage:
        buffer = ExperienceBuffer(max_size=2, batch_size=32)

        # Producer (RolloutWorker)
        async def producer():
            while True:
                batch = await generate_experiences()
                await buffer.put(batch)

        # Consumer (Learner)
        async def consumer():
            while True:
                batch = await buffer.get()
                train_step(batch)

    Attributes:
        max_size: Maximum batches to buffer (controls off-policy degree)
        batch_size: Expected batch size (for metrics)
    """

    def __init__(
        self,
        max_size: int = 1,
        batch_size: int = 32,
    ) -> None:
        """
        Initialize experience buffer.

        Args:
            max_size: Maximum batches in buffer (off-policy degree)
            batch_size: Expected batch size for metrics
        """
        self.max_size = max_size
        self.batch_size = batch_size
        self._buffer: asyncio.Queue[ExperienceBatch] = asyncio.Queue(maxsize=max_size)
        self._stats: dict[str, int] = {
            "produced": 0,
            "consumed": 0,
            "dropped": 0,
        }
        self._total_experiences = 0

    async def put(self, batch: ExperienceBatch) -> None:
        """
        Add batch to buffer (blocks if full).

        Args:
            batch: Experience batch to add
        """
        await self._buffer.put(batch)
        self._stats["produced"] += 1
        self._total_experiences += len(batch)

    async def get(self) -> ExperienceBatch:
        """
        Get batch from buffer (blocks if empty).

        Returns:
            Next experience batch
        """
        batch = await self._buffer.get()
        self._stats["consumed"] += 1
        self._buffer.task_done()
        return batch

    def try_put(self, batch: ExperienceBatch) -> bool:
        """
        Non-blocking put (returns False if full).

        Args:
            batch: Experience batch to add

        Returns:
            True if batch was added, False if buffer is full
        """
        try:
            self._buffer.put_nowait(batch)
            self._stats["produced"] += 1
            self._total_experiences += len(batch)
            return True
        except asyncio.QueueFull:
            self._stats["dropped"] += 1
            return False

    def try_get(self) -> ExperienceBatch | None:
        """
        Non-blocking get (returns None if empty).

        Returns:
            Next experience batch or None if empty
        """
        try:
            batch = self._buffer.get_nowait()
            self._stats["consumed"] += 1
            self._buffer.task_done()
            return batch
        except asyncio.QueueEmpty:
            return None

    @property
    def size(self) -> int:
        """Current number of batches in buffer."""
        return self._buffer.qsize()

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self._buffer.empty()

    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self._buffer.full()

    @property
    def stats(self) -> dict[str, int | float]:
        """
        Get buffer statistics.

        Returns:
            Dict with produced, consumed, dropped counts and fill ratio
        """
        return {
            **self._stats,
            "current_size": self.size,
            "fill_ratio": self.size / self.max_size if self.max_size > 0 else 0.0,
            "total_experiences": self._total_experiences,
        }

    async def drain(self) -> list[ExperienceBatch]:
        """
        Drain all batches from buffer without blocking.

        Returns:
            List of all batches currently in buffer
        """
        batches = []
        while not self.is_empty:
            batch = self.try_get()
            if batch is not None:
                batches.append(batch)
            else:
                break
        return batches

    async def wait_until_empty(self) -> None:
        """Wait until all items have been processed."""
        await self._buffer.join()

    def reset_stats(self) -> None:
        """Reset buffer statistics."""
        self._stats = {"produced": 0, "consumed": 0, "dropped": 0}
        self._total_experiences = 0

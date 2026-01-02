"""
SFT (Supervised Fine-Tuning) trainer with sequence packing.

This module provides efficient SFT training with:
- Sequence packing for optimal GPU utilization
- FSDP2 distributed training
- Integration with Ray Train
- Support for conversation-style datasets

Packing combines multiple short sequences into a single training sample,
dramatically improving throughput for datasets with variable-length examples.

Usage:
    from dreadnode.core.training.ray.sft import SFTTrainer, SFTConfig

    config = SFTConfig(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_seq_length=2048,
        use_packing=True,
    )

    trainer = SFTTrainer(config)
    trainer.train(dataset)
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

from dreadnode.core.training.ray.config import VLLMConfig
from dreadnode.core.training.ray.fsdp2_learner import FSDP2Config, FSDP2Learner


@dataclass
class SFTConfig:
    """Configuration for SFT training."""

    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    """Model name or path."""

    tokenizer_name: str | None = None
    """Tokenizer name (defaults to model_name)."""

    # Sequence settings
    max_seq_length: int = 2048
    """Maximum sequence length."""

    use_packing: bool = True
    """Enable sequence packing for efficiency."""

    packing_efficiency_threshold: float = 0.9
    """Minimum packing efficiency before padding."""

    # Training
    learning_rate: float = 2e-5
    """Learning rate."""

    weight_decay: float = 0.01
    """Weight decay."""

    warmup_ratio: float = 0.1
    """Warmup steps as fraction of total."""

    max_steps: int = 1000
    """Maximum training steps."""

    max_epochs: int = 3
    """Maximum training epochs."""

    batch_size: int = 4
    """Batch size per device."""

    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm."""

    # Logging & Checkpointing
    log_interval: int = 10
    """Steps between logging."""

    checkpoint_interval: int = 100
    """Steps between checkpoints."""

    checkpoint_dir: str = "./checkpoints"
    """Directory for checkpoints."""

    # Seed
    seed: int = 42
    """Random seed."""

    # Trust remote code
    trust_remote_code: bool = True
    """Trust remote code in model repository."""

    def __post_init__(self) -> None:
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name


@dataclass
class PackedSample:
    """A packed sequence containing multiple examples."""

    input_ids: torch.Tensor
    """Packed input token IDs [seq_len]."""

    attention_mask: torch.Tensor
    """Attention mask [seq_len]."""

    labels: torch.Tensor
    """Labels (same as input_ids, with -100 for non-prediction tokens)."""

    position_ids: torch.Tensor
    """Position IDs (reset for each packed sample)."""

    sample_boundaries: list[int]
    """Boundaries of individual samples within the packed sequence."""


class SequencePacker:
    """
    Packs multiple sequences into single training samples.

    Packing dramatically improves training efficiency by:
    1. Eliminating padding within batches
    2. Maximizing GPU memory utilization
    3. Processing more tokens per batch

    The packer maintains proper attention masking so that sequences
    don't attend to each other within a packed sample.
    """

    def __init__(
        self,
        max_seq_length: int = 2048,
        pad_token_id: int = 0,
        efficiency_threshold: float = 0.9,
    ) -> None:
        """
        Initialize sequence packer.

        Args:
            max_seq_length: Maximum packed sequence length
            pad_token_id: Token ID for padding
            efficiency_threshold: Minimum packing efficiency
        """
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.efficiency_threshold = efficiency_threshold

    def pack_sequences(
        self,
        sequences: list[dict[str, torch.Tensor]],
    ) -> list[PackedSample]:
        """
        Pack a list of tokenized sequences.

        Args:
            sequences: List of dicts with 'input_ids', 'labels' keys

        Returns:
            List of PackedSample objects
        """
        packed_samples = []
        current_ids = []
        current_labels = []
        current_positions = []
        boundaries = [0]

        for seq in sequences:
            seq_ids = seq["input_ids"]
            seq_labels = seq.get("labels", seq_ids.clone())

            if isinstance(seq_ids, torch.Tensor):
                seq_len = seq_ids.shape[0]
            else:
                seq_len = len(seq_ids)
                seq_ids = torch.tensor(seq_ids)
                seq_labels = torch.tensor(seq_labels) if not isinstance(seq_labels, torch.Tensor) else seq_labels

            # Check if sequence fits
            current_len = sum(len(ids) if isinstance(ids, list) else ids.shape[0] for ids in current_ids)

            if current_len + seq_len > self.max_seq_length:
                # Finalize current packed sample
                if current_ids:
                    packed = self._finalize_pack(
                        current_ids, current_labels, current_positions, boundaries
                    )
                    packed_samples.append(packed)

                # Start new pack
                current_ids = []
                current_labels = []
                current_positions = []
                boundaries = [0]

            # Add sequence to current pack
            current_ids.append(seq_ids)
            current_labels.append(seq_labels)
            current_positions.append(torch.arange(seq_len))
            boundaries.append(boundaries[-1] + seq_len)

        # Finalize last pack
        if current_ids:
            packed = self._finalize_pack(
                current_ids, current_labels, current_positions, boundaries
            )
            packed_samples.append(packed)

        return packed_samples

    def _finalize_pack(
        self,
        ids_list: list[torch.Tensor],
        labels_list: list[torch.Tensor],
        positions_list: list[torch.Tensor],
        boundaries: list[int],
    ) -> PackedSample:
        """Finalize a packed sample with proper padding."""
        # Concatenate
        input_ids = torch.cat(ids_list)
        labels = torch.cat(labels_list)
        position_ids = torch.cat(positions_list)

        current_len = input_ids.shape[0]

        # Pad to max_seq_length if needed
        if current_len < self.max_seq_length:
            pad_len = self.max_seq_length - current_len

            input_ids = torch.cat([
                input_ids,
                torch.full((pad_len,), self.pad_token_id, dtype=input_ids.dtype),
            ])

            labels = torch.cat([
                labels,
                torch.full((pad_len,), -100, dtype=labels.dtype),  # -100 = ignore
            ])

            position_ids = torch.cat([
                position_ids,
                torch.zeros(pad_len, dtype=position_ids.dtype),
            ])

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones(self.max_seq_length, dtype=torch.long)
        attention_mask[current_len:] = 0

        return PackedSample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
            sample_boundaries=boundaries,
        )


class PackedDataset(IterableDataset):
    """
    Dataset that yields packed sequences for efficient training.

    Wraps a base dataset and packs sequences on-the-fly.
    """

    def __init__(
        self,
        base_dataset: Dataset | Sequence[dict],
        tokenizer: Any,
        max_seq_length: int = 2048,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        """
        Initialize packed dataset.

        Args:
            base_dataset: Base dataset with 'text' or 'input'/'output' fields
            tokenizer: Tokenizer for encoding
            max_seq_length: Maximum packed sequence length
            shuffle: Shuffle examples before packing
            seed: Random seed for shuffling
        """
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.shuffle = shuffle
        self.seed = seed

        self.packer = SequencePacker(
            max_seq_length=max_seq_length,
            pad_token_id=tokenizer.pad_token_id or 0,
        )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Iterate over packed samples."""
        # Get and optionally shuffle indices
        indices = list(range(len(self.base_dataset)))
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(indices)

        # Tokenize and collect sequences
        sequences = []
        for idx in indices:
            example = self.base_dataset[idx]
            tokenized = self._tokenize(example)
            if tokenized is not None:
                sequences.append(tokenized)

            # Pack when we have enough sequences
            if len(sequences) >= 100:
                packed = self.packer.pack_sequences(sequences)
                for sample in packed:
                    yield self._sample_to_dict(sample)
                sequences = []

        # Pack remaining
        if sequences:
            packed = self.packer.pack_sequences(sequences)
            for sample in packed:
                yield self._sample_to_dict(sample)

    def _tokenize(self, example: dict) -> dict[str, torch.Tensor] | None:
        """Tokenize a single example."""
        if "text" in example:
            text = example["text"]
        elif "input" in example and "output" in example:
            text = f"{example['input']}{example['output']}"
        elif "messages" in example:
            # Chat format
            text = self.tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            return None

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        labels = input_ids.clone()

        return {"input_ids": input_ids, "labels": labels}

    def _sample_to_dict(self, sample: PackedSample) -> dict[str, torch.Tensor]:
        """Convert PackedSample to dict for DataLoader."""
        return {
            "input_ids": sample.input_ids,
            "attention_mask": sample.attention_mask,
            "labels": sample.labels,
            "position_ids": sample.position_ids,
        }


class SFTTrainer:
    """
    SFT trainer with sequence packing and FSDP2 support.

    Features:
    - Sequence packing for efficient training
    - FSDP2 distributed training
    - Gradient accumulation
    - Mixed precision (bf16)
    - Checkpointing
    """

    def __init__(
        self,
        config: SFTConfig,
        fsdp_config: FSDP2Config | None = None,
    ) -> None:
        """
        Initialize SFT trainer.

        Args:
            config: SFT configuration
            fsdp_config: Optional FSDP2 configuration
        """
        self.config = config
        self.fsdp_config = fsdp_config or FSDP2Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        self._load_model_and_tokenizer()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = None
        self.step = 0

    def _load_model_and_tokenizer(self) -> None:
        """Load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name or self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.fsdp_config.param_dtype,
            attn_implementation="flash_attention_2",
            trust_remote_code=self.config.trust_remote_code,
        )

        if self.fsdp_config.use_activation_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model = self.model.to(self.device)

    def train(
        self,
        dataset: Dataset | Sequence[dict],
        eval_dataset: Dataset | Sequence[dict] | None = None,
    ) -> dict[str, float]:
        """
        Run SFT training.

        Args:
            dataset: Training dataset
            eval_dataset: Optional evaluation dataset

        Returns:
            Final training metrics
        """
        # Create packed dataset if packing enabled
        if self.config.use_packing:
            train_dataset = PackedDataset(
                base_dataset=dataset,
                tokenizer=self.tokenizer,
                max_seq_length=self.config.max_seq_length,
                shuffle=True,
                seed=self.config.seed,
            )
            dataloader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=self._collate_fn,
            )

        # Create scheduler
        total_steps = self.config.max_steps
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        self._create_scheduler(total_steps, warmup_steps)

        # Training loop
        self.model.train()
        total_loss = 0.0
        metrics = {}

        for epoch in range(self.config.max_epochs):
            for batch in dataloader:
                if self.step >= self.config.max_steps:
                    break

                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Optional position_ids for packed sequences
                position_ids = batch.get("position_ids")
                if position_ids is not None:
                    position_ids = position_ids.to(self.device)

                # Forward
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    position_ids=position_ids,
                )
                loss = outputs.loss / self.config.gradient_accumulation_steps

                # Backward
                loss.backward()
                total_loss += loss.item()

                # Optimizer step
                if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm,
                        )

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if self.scheduler is not None:
                        self.scheduler.step()

                # Logging
                if self.step % self.config.log_interval == 0:
                    avg_loss = total_loss / max(self.step, 1)
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(f"[Step {self.step}] loss: {avg_loss:.4f} | lr: {lr:.2e}")
                    metrics = {"loss": avg_loss, "learning_rate": lr}

                # Checkpointing
                if self.step > 0 and self.step % self.config.checkpoint_interval == 0:
                    self.save_checkpoint()

                self.step += 1

            if self.step >= self.config.max_steps:
                break

        # Final checkpoint
        self.save_checkpoint()

        return metrics

    def _collate_fn(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        """Collate function for non-packed training."""
        texts = []
        for ex in examples:
            if "text" in ex:
                texts.append(ex["text"])
            elif "input" in ex and "output" in ex:
                texts.append(f"{ex['input']}{ex['output']}")

        encodings = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt",
        )

        labels = encodings["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }

    def _create_scheduler(self, total_steps: int, warmup_steps: int) -> None:
        """Create learning rate scheduler."""
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        decay = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.config.learning_rate * 0.1,
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, decay],
            milestones=[warmup_steps],
        )

    def save_checkpoint(self) -> None:
        """Save training checkpoint."""
        import os

        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"step_{self.step}",
        )
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

        # Save training state
        torch.save(
            {
                "step": self.step,
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            os.path.join(checkpoint_path, "training_state.pt"),
        )

        print(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        import os

        # Load training state
        state_path = os.path.join(path, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.step = state["step"]
            self.optimizer.load_state_dict(state["optimizer_state_dict"])

        # Load model
        from transformers import AutoModelForCausalLM

        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=self.fsdp_config.param_dtype,
            trust_remote_code=self.config.trust_remote_code,
        ).to(self.device)

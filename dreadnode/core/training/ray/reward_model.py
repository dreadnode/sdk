"""
Reward Model trainer for RLHF.

This module provides training for reward models used in RLHF pipelines.
The reward model learns to predict scalar rewards from preference pairs
using the Bradley-Terry model.

The reward model architecture adds a value head (linear projection) on top
of a pretrained language model. The model outputs a scalar reward for each
input sequence.

The Bradley-Terry loss is:
    L = -log(sigmoid(r_chosen - r_rejected))

where r_chosen and r_rejected are the reward model's predictions.

References:
- https://arxiv.org/abs/2203.02155 (InstructGPT)
- https://arxiv.org/abs/1906.01488 (Bradley-Terry model)
- https://github.com/OpenRLHF/OpenRLHF

Usage:
    from dreadnode.core.training.ray.reward_model import RewardModelTrainer, RMConfig

    config = RMConfig(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        learning_rate=1e-5,
    )

    trainer = RewardModelTrainer(config)
    trainer.train(preference_dataset)

    # Use the trained model for inference
    rewards = trainer.compute_rewards(["prompt 1", "prompt 2"])
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dreadnode.core.training.ray.fsdp2_learner import FSDP2Config

if TYPE_CHECKING:
    from dreadnode.core.storage.storage import Storage
    from dreadnode.models.local import LocalModel


@dataclass
class RMConfig:
    """Configuration for Reward Model training."""

    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    """Base model name or path."""

    tokenizer_name: str | None = None
    """Tokenizer name (defaults to model_name)."""

    # Architecture
    value_head_hidden_size: int | None = None
    """Hidden size for value head. None = match model hidden size."""

    value_head_dropout: float = 0.1
    """Dropout for value head."""

    pooling: str = "last"
    """Pooling method: 'last' (last non-pad token), 'mean', 'max'."""

    # Sequence settings
    max_seq_length: int = 2048
    """Maximum sequence length."""

    max_prompt_length: int = 512
    """Maximum prompt length."""

    # Training
    learning_rate: float = 1e-5
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

    gradient_accumulation_steps: int = 4
    """Gradient accumulation steps."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm."""

    # Regularization
    margin: float = 0.0
    """Margin for Bradley-Terry loss (0 = no margin)."""

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


class RewardModelHead(nn.Module):
    """
    Value head for reward model.

    Projects the hidden state to a scalar reward.
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, hidden_size]

        Returns:
            Scalar rewards: [batch_size]
        """
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output.squeeze(-1)


class RewardModel(nn.Module):
    """
    Reward model combining base LLM with value head.

    The model takes a sequence and outputs a scalar reward.
    """

    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int,
        dropout: float = 0.1,
        pooling: str = "last",
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.value_head = RewardModelHead(hidden_size, dropout)
        self.pooling = pooling

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            Scalar rewards: [batch_size]
        """
        # Get hidden states from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Get last hidden state
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]

        # Pool hidden states
        if self.pooling == "last":
            # Get last non-padding token
            batch_size = input_ids.shape[0]
            seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
            pooled = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                seq_lengths,
            ]
        elif self.pooling == "mean":
            # Mean pool over non-padding tokens
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif self.pooling == "max":
            # Max pool over non-padding tokens
            mask = attention_mask.unsqueeze(-1).float()
            hidden_states = hidden_states.masked_fill(mask == 0, float("-inf"))
            pooled = hidden_states.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # Get scalar reward
        reward = self.value_head(pooled)
        return reward

    def save_pretrained(self, path: str, **kwargs: Any) -> None:
        """Save model to path."""
        os.makedirs(path, exist_ok=True)

        # Save base model
        self.base_model.save_pretrained(path, **kwargs)

        # Save value head separately
        value_head_path = os.path.join(path, "value_head.pt")
        torch.save(self.value_head.state_dict(), value_head_path)

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        dropout: float = 0.1,
        pooling: str = "last",
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> "RewardModel":
        """Load model from path."""
        from transformers import AutoModelForCausalLM, AutoConfig

        config = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
        base_model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        model = cls(
            base_model=base_model,
            hidden_size=config.hidden_size,
            dropout=dropout,
            pooling=pooling,
        )

        # Load value head if exists
        value_head_path = os.path.join(path, "value_head.pt")
        if os.path.exists(value_head_path):
            model.value_head.load_state_dict(torch.load(value_head_path, weights_only=True))

        return model


class RewardModelTrainer:
    """
    Reward Model trainer using Bradley-Terry loss.

    Trains a model to predict scalar rewards from preference pairs.
    The trained model can then be used in RLHF pipelines (PPO, GRPO, etc.).

    Attributes:
        config: Reward model configuration
        model: The reward model (base LLM + value head)
        tokenizer: Tokenizer
    """

    def __init__(
        self,
        config: RMConfig,
        fsdp_config: FSDP2Config | None = None,
        storage: Storage | None = None,
        checkpoint_name: str | None = None,
    ) -> None:
        """
        Initialize Reward Model trainer.

        Args:
            config: Reward model configuration
            fsdp_config: Optional FSDP2 configuration
            storage: Optional storage for CAS checkpointing
            checkpoint_name: Name for checkpoints
        """
        self.config = config
        self.fsdp_config = fsdp_config or FSDP2Config()
        self.storage = storage
        self.checkpoint_name = checkpoint_name or config.model_name.replace("/", "-")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._checkpoint_version = 0
        self.step = 0

        # Load model and tokenizer
        self._load_model_and_tokenizer()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = None

    def _load_model_and_tokenizer(self) -> None:
        """Load reward model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name or self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.fsdp_config.param_dtype,
            attn_implementation="sdpa",
            trust_remote_code=self.config.trust_remote_code,
        )

        # Get hidden size
        model_config = AutoConfig.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        hidden_size = self.config.value_head_hidden_size or model_config.hidden_size

        # Create reward model
        self.model = RewardModel(
            base_model=base_model,
            hidden_size=hidden_size,
            dropout=self.config.value_head_dropout,
            pooling=self.config.pooling,
        )

        if self.fsdp_config.use_activation_checkpointing:
            self.model.base_model.gradient_checkpointing_enable()

        self.model = self.model.to(self.device)

    def train(
        self,
        dataset: Dataset | list[dict],
    ) -> dict[str, float]:
        """
        Run reward model training.

        Args:
            dataset: Training dataset with preference pairs.
                     Each item should have 'prompt', 'chosen', 'rejected' keys.

        Returns:
            Final training metrics
        """
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
        total_acc = 0.0
        metrics = {}

        for epoch in range(self.config.max_epochs):
            for batch in dataloader:
                if self.step >= self.config.max_steps:
                    break

                # Compute Bradley-Terry loss
                loss, batch_metrics = self._compute_loss(batch)

                # Scale loss for gradient accumulation
                scaled_loss = loss / self.config.gradient_accumulation_steps
                scaled_loss.backward()
                total_loss += loss.item()
                total_acc += batch_metrics["accuracy"]

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
                    avg_acc = total_acc / max(self.step, 1)
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"[Step {self.step}] loss: {avg_loss:.4f} | "
                        f"accuracy: {avg_acc:.4f} | "
                        f"chosen_reward: {batch_metrics['chosen_reward']:.4f} | "
                        f"rejected_reward: {batch_metrics['rejected_reward']:.4f} | "
                        f"margin: {batch_metrics['margin']:.4f} | lr: {lr:.2e}"
                    )
                    metrics = {
                        "loss": avg_loss,
                        "accuracy": avg_acc,
                        "learning_rate": lr,
                        **batch_metrics,
                    }

                # Checkpointing
                if self.step > 0 and self.step % self.config.checkpoint_interval == 0:
                    self.save_checkpoint()

                self.step += 1

            if self.step >= self.config.max_steps:
                break

        # Final checkpoint
        self.save_checkpoint()

        return metrics

    def _collate_fn(self, examples: list[dict]) -> dict[str, Any]:
        """Collate preference pairs into a batch."""
        prompts = []
        chosen_responses = []
        rejected_responses = []

        for ex in examples:
            prompts.append(ex["prompt"])
            chosen_responses.append(ex["chosen"])
            rejected_responses.append(ex["rejected"])

        # Tokenize chosen (prompt + chosen response)
        chosen_texts = [p + c for p, c in zip(prompts, chosen_responses)]
        chosen_encodings = self.tokenizer(
            chosen_texts,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt",
        )

        # Tokenize rejected (prompt + rejected response)
        rejected_texts = [p + r for p, r in zip(prompts, rejected_responses)]
        rejected_encodings = self.tokenizer(
            rejected_texts,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_encodings["input_ids"],
            "chosen_attention_mask": chosen_encodings["attention_mask"],
            "rejected_input_ids": rejected_encodings["input_ids"],
            "rejected_attention_mask": rejected_encodings["attention_mask"],
        }

    def _compute_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict]:
        """Compute Bradley-Terry loss for a batch of preference pairs."""
        # Move to device
        chosen_ids = batch["chosen_input_ids"].to(self.device)
        chosen_mask = batch["chosen_attention_mask"].to(self.device)
        rejected_ids = batch["rejected_input_ids"].to(self.device)
        rejected_mask = batch["rejected_attention_mask"].to(self.device)

        # Get rewards
        chosen_rewards = self.model(chosen_ids, chosen_mask)
        rejected_rewards = self.model(rejected_ids, rejected_mask)

        # Bradley-Terry loss with optional margin
        # L = -log(sigmoid(r_chosen - r_rejected - margin))
        logits = chosen_rewards - rejected_rewards - self.config.margin
        loss = -nn.functional.logsigmoid(logits).mean()

        # Metrics
        margin = (chosen_rewards - rejected_rewards).mean().item()
        accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

        metrics = {
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
            "margin": margin,
            "accuracy": accuracy,
        }

        return loss, metrics

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

    def compute_rewards(
        self,
        texts: list[str],
        batch_size: int = 8,
    ) -> list[float]:
        """
        Compute rewards for a list of texts.

        Args:
            texts: List of text sequences
            batch_size: Batch size for inference

        Returns:
            List of scalar rewards
        """
        self.model.eval()
        rewards = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                encodings = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_length,
                    return_tensors="pt",
                )

                input_ids = encodings["input_ids"].to(self.device)
                attention_mask = encodings["attention_mask"].to(self.device)

                batch_rewards = self.model(input_ids, attention_mask)
                rewards.extend(batch_rewards.cpu().tolist())

        self.model.train()
        return rewards

    def save_checkpoint(self) -> None:
        """Save training checkpoint."""
        # Save to storage if available
        if self.storage is not None:
            local_model = self._save_to_storage()
            if local_model:
                print(f"Saved checkpoint to CAS: {local_model.name}")
                return

        # Fall back to file-based
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"reward_model_step_{self.step}",
        )
        os.makedirs(checkpoint_path, exist_ok=True)

        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

        torch.save(
            {
                "step": self.step,
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            os.path.join(checkpoint_path, "training_state.pt"),
        )

        print(f"Saved checkpoint to {checkpoint_path}")

    def _save_to_storage(self) -> LocalModel | None:
        """Save checkpoint to CAS."""
        if self.storage is None:
            return None

        try:
            from dreadnode.models.local import LocalModel
            import tempfile

            self._checkpoint_version += 1
            version = f"0.{self._checkpoint_version}.0"
            name = f"{self.checkpoint_name}-rm-step{self.step}"

            # Save to temp dir first
            with tempfile.TemporaryDirectory() as tmpdir:
                self.model.save_pretrained(tmpdir)
                self.tokenizer.save_pretrained(tmpdir)

                return LocalModel.from_hf(
                    model=self.model.base_model,
                    name=name,
                    storage=self.storage,
                    tokenizer=self.tokenizer,
                    format="safetensors",
                    task="text-generation",
                    version=version,
                )
        except Exception as e:
            print(f"Failed to save to CAS: {e}")
            return None

    def get_model(self) -> RewardModel:
        """Get the trained reward model."""
        return self.model

    def get_reward_fn(self) -> callable:
        """
        Get a reward function for use with GRPO/PPO.

        Returns:
            A callable that takes texts and returns rewards
        """
        model = self.model
        tokenizer = self.tokenizer
        device = self.device
        max_length = self.config.max_seq_length

        def reward_fn(texts: list[str]) -> list[float]:
            model.eval()
            with torch.no_grad():
                encodings = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                input_ids = encodings["input_ids"].to(device)
                attention_mask = encodings["attention_mask"].to(device)
                rewards = model(input_ids, attention_mask)
                return rewards.cpu().tolist()

        return reward_fn

"""
GRPO Learner for distributed policy gradient training.

Uses Ray Train with DeepSpeed/FSDP for efficient distributed training.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from dreadnode.core.training.ray.config import GRPOLossConfig, RayGRPOConfig


@dataclass
class TrainingBatch:
    """Batch for GRPO training."""

    input_ids: torch.Tensor
    """Full sequence token IDs [batch_size, seq_len]."""

    attention_mask: torch.Tensor
    """Attention mask [batch_size, seq_len]."""

    completion_mask: torch.Tensor
    """Mask for completion tokens [batch_size, seq_len]."""

    advantages: torch.Tensor
    """Advantage values [batch_size]."""

    generation_logprobs: torch.Tensor
    """Log probs from generation policy [batch_size, seq_len]."""

    reference_logprobs: torch.Tensor | None = None
    """Log probs from reference policy [batch_size, seq_len]."""


class GRPODataset(Dataset):
    """Dataset for GRPO training batches."""

    def __init__(self, batches: list[TrainingBatch]):
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int) -> TrainingBatch:
        return self.batches[idx]


def compute_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log probabilities for tokens.

    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size].
        input_ids: Token IDs [batch_size, seq_len].
        mask: Mask for valid tokens [batch_size, seq_len].

    Returns:
        Log probabilities [batch_size, seq_len].
    """
    # Shift for next-token prediction
    logits = logits[:, :-1, :]  # [B, S-1, V]
    targets = input_ids[:, 1:]  # [B, S-1]
    mask = mask[:, 1:]  # [B, S-1]

    # Compute log softmax
    log_probs = F.log_softmax(logits.float(), dim=-1)

    # Gather log probs for actual tokens
    token_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [B, S-1]

    return token_log_probs * mask


def compute_kl_divergence(
    curr_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask: torch.Tensor,
    kl_type: str = "forward",
) -> torch.Tensor:
    """
    Compute KL divergence between current and reference policies.

    Args:
        curr_logprobs: Current policy log probs [batch_size, seq_len].
        ref_logprobs: Reference policy log probs [batch_size, seq_len].
        mask: Mask for valid tokens [batch_size, seq_len].
        kl_type: Type of KL divergence ("forward" or "reverse").

    Returns:
        KL divergence per sample [batch_size].
    """
    if kl_type == "forward":
        # KL(ref || curr) = sum(ref * (log_ref - log_curr))
        log_ratio = ref_logprobs - curr_logprobs
        kl = torch.exp(ref_logprobs) * log_ratio
    else:
        # KL(curr || ref) = sum(curr * (log_curr - log_ref))
        log_ratio = curr_logprobs - ref_logprobs
        kl = torch.exp(curr_logprobs) * log_ratio

    # Mask and sum
    kl = (kl * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    return kl


def grpo_loss(
    logits: torch.Tensor,
    batch: TrainingBatch,
    config: GRPOLossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute GRPO loss.

    GRPO loss combines:
    1. Policy gradient loss with advantages
    2. KL divergence penalty against reference policy
    3. PPO-style ratio clipping

    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size].
        batch: Training batch with advantages and masks.
        config: Loss configuration.

    Returns:
        Tuple of (loss tensor, metrics dict).
    """
    # Compute current log probabilities
    curr_logprobs = compute_log_probs(logits, batch.input_ids, batch.completion_mask)

    # Compute probability ratio (PPO-style)
    # ratio = exp(curr_logprobs - generation_logprobs)
    gen_logprobs = batch.generation_logprobs[:, 1:]  # Shift to align
    mask = batch.completion_mask[:, 1:]

    log_ratio = curr_logprobs - gen_logprobs
    ratio = torch.exp(log_ratio)

    # Clipped ratio
    ratio_clipped = torch.clamp(
        ratio,
        1.0 - config.clip_ratio,
        1.0 + config.clip_ratio,
    )

    # Advantages [batch_size] -> [batch_size, 1] for broadcasting
    advantages = batch.advantages.unsqueeze(-1)

    # Policy gradient loss
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * ratio_clipped

    # Take max (pessimistic bound)
    pg_loss = torch.max(pg_loss1, pg_loss2)

    # Mask and average
    if config.token_level_loss:
        # Average over tokens, then samples
        pg_loss = (pg_loss * mask).sum() / mask.sum().clamp(min=1)
    else:
        # Average over samples (sequence-level)
        seq_loss = (pg_loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        pg_loss = seq_loss.mean()

    # KL penalty
    kl_loss = torch.tensor(0.0, device=logits.device)
    if config.kl_coef > 0 and batch.reference_logprobs is not None:
        ref_logprobs = batch.reference_logprobs[:, 1:]  # Shift to align
        kl = compute_kl_divergence(curr_logprobs, ref_logprobs, mask)
        kl_loss = config.kl_coef * kl.mean()

    # Total loss
    loss = pg_loss + kl_loss

    # Metrics
    with torch.no_grad():
        metrics = {
            "loss": loss.item(),
            "pg_loss": pg_loss.item(),
            "kl_loss": kl_loss.item(),
            "ratio_mean": ratio.mean().item(),
            "ratio_min": ratio.min().item(),
            "ratio_max": ratio.max().item(),
            "advantages_mean": batch.advantages.mean().item(),
            "advantages_std": batch.advantages.std().item(),
        }

    return loss, metrics


class GRPOLearner:
    """
    GRPO Learner for distributed policy gradient training.

    Handles:
    - Model loading and management
    - Reference policy for KL penalty
    - Distributed training with DeepSpeed/FSDP
    - Gradient accumulation and optimization
    """

    def __init__(
        self,
        config: RayGRPOConfig,
        model: nn.Module | None = None,
        reference_model: nn.Module | None = None,
    ):
        """
        Initialize learner.

        Args:
            config: GRPO configuration.
            model: Optional pre-loaded model.
            reference_model: Optional reference model for KL penalty.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model if not provided
        if model is None:
            self.model = self._load_model()
        else:
            self.model = model.to(self.device)

        # Reference model (frozen copy for KL penalty)
        if reference_model is None and config.loss.kl_coef > 0:
            self.reference_model = self._load_model()
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False
        else:
            self.reference_model = reference_model

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        self.scheduler = None  # Created in train() when we know total steps

        self.step = 0

    def _load_model(self) -> nn.Module:
        """Load model from HuggingFace."""
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16
            if self.config.training.mixed_precision == "bf16"
            else torch.float16,
            trust_remote_code=self.config.vllm.trust_remote_code,
        )

        if self.config.training.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        return model.to(self.device)

    def compute_reference_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probabilities from reference policy.

        Args:
            input_ids: Token IDs [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            completion_mask: Completion token mask [batch_size, seq_len].

        Returns:
            Log probabilities [batch_size, seq_len].
        """
        if self.reference_model is None:
            return None

        with torch.no_grad():
            outputs = self.reference_model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
            )
            logprobs = compute_log_probs(
                outputs.logits,
                input_ids.to(self.device),
                completion_mask.to(self.device),
            )
            # Pad to match input shape
            padded = torch.zeros_like(input_ids, dtype=logprobs.dtype, device=self.device)
            padded[:, 1:] = logprobs
            return padded

    def train_step(self, batch: TrainingBatch) -> dict[str, float]:
        """
        Execute one training step.

        Args:
            batch: Training batch.

        Returns:
            Metrics dictionary.
        """
        self.model.train()

        # Move batch to device
        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)
        completion_mask = batch.completion_mask.to(self.device)
        advantages = batch.advantages.to(self.device)
        generation_logprobs = batch.generation_logprobs.to(self.device)

        # Compute reference log probs if needed
        reference_logprobs = None
        if self.reference_model is not None:
            reference_logprobs = self.compute_reference_logprobs(
                input_ids, attention_mask, completion_mask
            )

        # Create batch with device tensors
        device_batch = TrainingBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            completion_mask=completion_mask,
            advantages=advantages,
            generation_logprobs=generation_logprobs,
            reference_logprobs=reference_logprobs,
        )

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Compute loss
        loss, metrics = grpo_loss(outputs.logits, device_batch, self.config.loss)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.scheduler is not None:
            self.scheduler.step()
            metrics["learning_rate"] = self.scheduler.get_last_lr()[0]

        self.step += 1
        metrics["step"] = self.step

        return metrics

    def get_state_dict(self) -> dict[str, torch.Tensor]:
        """Get model state dict for weight sync."""
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        import os

        os.makedirs(path, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step": self.step,
                "config": self.config.to_dict(),
            },
            os.path.join(path, "checkpoint.pt"),
        )

        self.model.save_pretrained(os.path.join(path, "model"))

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        import os

        checkpoint = torch.load(os.path.join(path, "checkpoint.pt"))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint["step"]


def create_deepspeed_config(config: RayGRPOConfig) -> dict[str, Any]:
    """
    Create DeepSpeed configuration.

    Args:
        config: GRPO configuration.

    Returns:
        DeepSpeed config dictionary.
    """
    ds_config = {
        "train_batch_size": config.train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
                "betas": [0.9, 0.999],
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": config.learning_rate,
                "warmup_num_steps": int(config.max_steps * config.warmup_ratio),
                "total_num_steps": config.max_steps,
            },
        },
        "gradient_clipping": config.max_grad_norm,
    }

    # ZeRO configuration
    stage = config.training.deepspeed_stage
    if stage == 1:
        ds_config["zero_optimization"] = {
            "stage": 1,
        }
    elif stage == 2:
        ds_config["zero_optimization"] = {
            "stage": 2,
            "offload_optimizer": {"device": "none"},
            "contiguous_gradients": True,
        }
    elif stage == 3:
        ds_config["zero_optimization"] = {
            "stage": 3,
            "offload_optimizer": {"device": "cpu"},
            "offload_param": {"device": "cpu"},
            "contiguous_gradients": True,
            "overlap_comm": True,
        }

    # Mixed precision
    if config.training.mixed_precision == "bf16":
        ds_config["bf16"] = {"enabled": True}
    elif config.training.mixed_precision == "fp16":
        ds_config["fp16"] = {"enabled": True}

    return ds_config

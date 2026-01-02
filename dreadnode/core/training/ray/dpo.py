"""
DPO (Direct Preference Optimization) trainer.

This module provides DPO training for language models, which is a simpler
alternative to RLHF that doesn't require a reward model or PPO.

DPO directly optimizes the policy using preference pairs (chosen vs rejected),
making it much simpler to implement while achieving comparable results.

The DPO loss is:
    L = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))

where log_ratio = log(pi(y|x)) - log(pi_ref(y|x))

References:
- https://arxiv.org/abs/2305.18290 (DPO paper)
- https://huggingface.co/docs/trl/dpo_trainer

Usage:
    from dreadnode.core.training.ray.dpo import DPOTrainer, DPOConfig

    config = DPOConfig(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        beta=0.1,
    )

    trainer = DPOTrainer(config)
    trainer.train(preference_dataset)
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dreadnode.core.training.ray.fsdp2_learner import FSDP2Config

if TYPE_CHECKING:
    from dreadnode.core.storage.storage import Storage
    from dreadnode.models.local import LocalModel


@dataclass
class DPOConfig:
    """Configuration for DPO training."""

    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    """Model name or path."""

    tokenizer_name: str | None = None
    """Tokenizer name (defaults to model_name)."""

    # DPO parameters
    beta: float = 0.1
    """Temperature parameter for DPO loss. Higher = more conservative updates."""

    label_smoothing: float = 0.0
    """Label smoothing for DPO loss (0 = no smoothing)."""

    loss_type: str = "sigmoid"
    """Loss type: 'sigmoid' (standard DPO), 'hinge', 'ipo'."""

    # Sequence settings
    max_seq_length: int = 2048
    """Maximum sequence length."""

    max_prompt_length: int = 512
    """Maximum prompt length."""

    # Training
    learning_rate: float = 5e-7
    """Learning rate (DPO typically uses lower LR than SFT)."""

    weight_decay: float = 0.01
    """Weight decay."""

    warmup_ratio: float = 0.1
    """Warmup steps as fraction of total."""

    max_steps: int = 1000
    """Maximum training steps."""

    max_epochs: int = 1
    """Maximum training epochs."""

    batch_size: int = 4
    """Batch size per device."""

    gradient_accumulation_steps: int = 4
    """Gradient accumulation steps."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm."""

    # Reference model
    ref_model_offload: bool = True
    """Keep reference model on CPU to save GPU memory."""

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
class PreferencePair:
    """A single preference pair for DPO training."""

    prompt: str
    """The prompt/query."""

    chosen: str
    """The preferred (winning) response."""

    rejected: str
    """The rejected (losing) response."""


class DPOTrainer:
    """
    DPO (Direct Preference Optimization) trainer.

    DPO directly optimizes the policy using preference pairs without needing
    a separate reward model or PPO. This makes it much simpler than RLHF.

    The training process:
    1. Load policy model and frozen reference model
    2. For each preference pair (chosen, rejected):
       - Compute log probabilities for both under policy and reference
       - Compute DPO loss to prefer chosen over rejected
    3. Update policy via gradient descent

    Attributes:
        config: DPO configuration
        model: Training policy model
        ref_model: Frozen reference model
        tokenizer: Tokenizer
    """

    def __init__(
        self,
        config: DPOConfig,
        fsdp_config: FSDP2Config | None = None,
        storage: Storage | None = None,
        checkpoint_name: str | None = None,
    ) -> None:
        """
        Initialize DPO trainer.

        Args:
            config: DPO configuration
            fsdp_config: Optional FSDP2 configuration
            storage: Optional storage for CAS checkpointing
            checkpoint_name: Name for checkpoints
        """
        self.config = config
        self.fsdp_config = fsdp_config or FSDP2Config(
            ref_model_offload=config.ref_model_offload
        )
        self.storage = storage
        self.checkpoint_name = checkpoint_name or config.model_name.replace("/", "-")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._checkpoint_version = 0
        self.step = 0

        # Load model and tokenizer
        self._load_model_and_tokenizer()

        # Load reference model
        self._load_reference_model()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = None

    def _load_model_and_tokenizer(self) -> None:
        """Load policy model and tokenizer."""
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
            attn_implementation="sdpa",
            trust_remote_code=self.config.trust_remote_code,
        )

        if self.fsdp_config.use_activation_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model = self.model.to(self.device)

    def _load_reference_model(self) -> None:
        """Load frozen reference model."""
        from transformers import AutoModelForCausalLM

        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.fsdp_config.param_dtype,
            attn_implementation="sdpa",
            trust_remote_code=self.config.trust_remote_code,
        )

        # Keep on CPU if offload enabled
        if self.fsdp_config.ref_model_offload:
            self.ref_model = self.ref_model.to("cpu")
            print("Reference model loaded on CPU (offload enabled)")
        else:
            self.ref_model = self.ref_model.to(self.device)

        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def train(
        self,
        dataset: Dataset | list[PreferencePair] | list[dict],
    ) -> dict[str, float]:
        """
        Run DPO training.

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
        metrics = {}

        for epoch in range(self.config.max_epochs):
            for batch in dataloader:
                if self.step >= self.config.max_steps:
                    break

                # Compute DPO loss
                loss, batch_metrics = self._compute_dpo_loss(batch)

                # Scale loss for gradient accumulation
                scaled_loss = loss / self.config.gradient_accumulation_steps
                scaled_loss.backward()
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
                    print(
                        f"[Step {self.step}] loss: {avg_loss:.4f} | "
                        f"chosen_reward: {batch_metrics['chosen_reward']:.4f} | "
                        f"rejected_reward: {batch_metrics['rejected_reward']:.4f} | "
                        f"margin: {batch_metrics['margin']:.4f} | lr: {lr:.2e}"
                    )
                    metrics = {
                        "loss": avg_loss,
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

    def _collate_fn(self, examples: list[dict | PreferencePair]) -> dict[str, Any]:
        """Collate preference pairs into a batch."""
        prompts = []
        chosen_responses = []
        rejected_responses = []

        for ex in examples:
            if isinstance(ex, PreferencePair):
                prompts.append(ex.prompt)
                chosen_responses.append(ex.chosen)
                rejected_responses.append(ex.rejected)
            else:
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

        # Tokenize prompts to get prompt lengths
        prompt_encodings = self.tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=self.config.max_prompt_length,
        )
        prompt_lengths = [len(ids) for ids in prompt_encodings["input_ids"]]

        return {
            "chosen_input_ids": chosen_encodings["input_ids"],
            "chosen_attention_mask": chosen_encodings["attention_mask"],
            "rejected_input_ids": rejected_encodings["input_ids"],
            "rejected_attention_mask": rejected_encodings["attention_mask"],
            "prompt_lengths": prompt_lengths,
        }

    def _compute_dpo_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict]:
        """Compute DPO loss for a batch of preference pairs."""
        # Move to device
        chosen_ids = batch["chosen_input_ids"].to(self.device)
        chosen_mask = batch["chosen_attention_mask"].to(self.device)
        rejected_ids = batch["rejected_input_ids"].to(self.device)
        rejected_mask = batch["rejected_attention_mask"].to(self.device)
        prompt_lengths = batch["prompt_lengths"]

        # Compute policy log probs
        policy_chosen_logps = self._compute_logprobs(
            self.model, chosen_ids, chosen_mask, prompt_lengths
        )
        policy_rejected_logps = self._compute_logprobs(
            self.model, rejected_ids, rejected_mask, prompt_lengths
        )

        # Compute reference log probs
        with torch.no_grad():
            # Move ref model to GPU if offloaded
            if self.fsdp_config.ref_model_offload:
                self.ref_model = self.ref_model.to(self.device)

            try:
                ref_chosen_logps = self._compute_logprobs(
                    self.ref_model, chosen_ids, chosen_mask, prompt_lengths
                )
                ref_rejected_logps = self._compute_logprobs(
                    self.ref_model, rejected_ids, rejected_mask, prompt_lengths
                )
            finally:
                # Move ref model back to CPU
                if self.fsdp_config.ref_model_offload:
                    self.ref_model = self.ref_model.to("cpu")
                    torch.cuda.empty_cache()

        # Compute log ratios
        chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
        rejected_log_ratio = policy_rejected_logps - ref_rejected_logps

        # DPO loss
        if self.config.loss_type == "sigmoid":
            # Standard DPO loss
            logits = self.config.beta * (chosen_log_ratio - rejected_log_ratio)
            if self.config.label_smoothing > 0:
                # Label smoothing
                losses = (
                    -nn.functional.logsigmoid(logits) * (1 - self.config.label_smoothing)
                    - nn.functional.logsigmoid(-logits) * self.config.label_smoothing
                )
            else:
                losses = -nn.functional.logsigmoid(logits)
        elif self.config.loss_type == "hinge":
            # Hinge loss variant
            losses = torch.relu(1 - self.config.beta * (chosen_log_ratio - rejected_log_ratio))
        elif self.config.loss_type == "ipo":
            # IPO (Identity Preference Optimization) loss
            losses = (chosen_log_ratio - rejected_log_ratio - 1 / (2 * self.config.beta)) ** 2
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")

        loss = losses.mean()

        # Metrics
        chosen_rewards = self.config.beta * chosen_log_ratio.detach()
        rejected_rewards = self.config.beta * rejected_log_ratio.detach()
        margin = (chosen_rewards - rejected_rewards).mean().item()

        metrics = {
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
            "margin": margin,
            "accuracy": (chosen_rewards > rejected_rewards).float().mean().item(),
        }

        return loss, metrics

    def _compute_logprobs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: list[int],
    ) -> torch.Tensor:
        """Compute log probabilities for response tokens only."""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Get log probs
        logits = outputs.logits[:, :-1, :]  # Shift for next token prediction
        labels = input_ids[:, 1:]  # Shift labels

        log_probs = nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask out prompt tokens and padding
        batch_size, seq_len = labels.shape
        response_mask = torch.zeros_like(labels, dtype=torch.float)
        for i, prompt_len in enumerate(prompt_lengths):
            # Response starts after prompt
            response_start = min(prompt_len - 1, seq_len - 1)  # -1 for shift
            response_mask[i, response_start:] = attention_mask[i, prompt_len:prompt_len + seq_len - response_start]

        # Sum log probs for response tokens
        masked_log_probs = token_log_probs * response_mask
        sum_log_probs = masked_log_probs.sum(dim=-1)

        # Normalize by response length
        response_lengths = response_mask.sum(dim=-1).clamp(min=1)
        avg_log_probs = sum_log_probs / response_lengths

        return avg_log_probs

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

        # Save to storage if available
        if self.storage is not None:
            local_model = self._save_to_storage()
            if local_model:
                print(f"Saved checkpoint to CAS: {local_model.name}")
                return

        # Fall back to file-based
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"dpo_step_{self.step}",
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

            self._checkpoint_version += 1
            version = f"0.{self._checkpoint_version}.0"
            name = f"{self.checkpoint_name}-dpo-step{self.step}"

            return LocalModel.from_hf(
                model=self.model,
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

    def get_model(self) -> nn.Module:
        """Get the trained model."""
        return self.model

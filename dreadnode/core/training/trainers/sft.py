"""
SFT Trainer for DN SDK agents.

Supervised Fine-Tuning on agent trajectories using NeMo RL.

This module provides integration with NeMo RL's SFT algorithm for training
agent models on expert trajectories before RL training.

Key Features:
- Full NeMo RL Policy integration
- NLL loss for causal language modeling
- Response-only loss option for chat models
- LoRA support for efficient fine-tuning
- Checkpoint saving with model weights
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, NotRequired, TypedDict

import torch

from dreadnode.core.training.trainers.base import (
    BaseTrainer,
    TrainingCallback,
    TrainingConfig,
    TrainingState,
)


class SFTSaveState(TypedDict, total=False):
    """Training state for checkpointing (matches NeMo RL)."""

    consumed_samples: int
    current_step: int
    current_epoch: int
    total_valid_tokens: int


class SFTConfig(TrainingConfig):
    """Configuration for SFT training (matches NeMo RL)."""

    # Data
    train_file: NotRequired[str]
    """Path to training data (JSONL)."""

    val_file: NotRequired[str]
    """Path to validation data (JSONL)."""

    # Training
    max_num_epochs: NotRequired[int]
    """Maximum number of training epochs."""

    max_num_steps: NotRequired[int]
    """Maximum number of training steps."""

    max_seq_length: NotRequired[int]
    """Maximum sequence length."""

    # Batch sizes
    train_global_batch_size: NotRequired[int]
    """Global batch size for training."""

    train_micro_batch_size: NotRequired[int]
    """Micro batch size per GPU."""

    val_global_batch_size: NotRequired[int]
    """Global batch size for validation."""

    val_micro_batch_size: NotRequired[int]
    """Micro batch size for validation."""

    # Loss
    loss_type: NotRequired[str]
    """Loss type: 'causal_lm' or 'response_only'."""

    # Validation
    val_period: NotRequired[int]
    """Steps between validation."""

    val_batches: NotRequired[int]
    """Number of validation batches."""

    val_at_start: NotRequired[bool]
    """Run validation at start."""

    # LoRA (optional)
    use_lora: NotRequired[bool]
    """Whether to use LoRA."""

    lora_r: NotRequired[int]
    """LoRA rank."""

    lora_alpha: NotRequired[int]
    """LoRA alpha."""

    lora_dropout: NotRequired[float]
    """LoRA dropout."""

    lora_target_modules: NotRequired[list[str]]
    """LoRA target modules."""

    # Legacy aliases
    num_epochs: NotRequired[int]
    batch_size: NotRequired[int]


def _default_sft_save_state() -> SFTSaveState:
    """Create default save state."""
    return {
        "consumed_samples": 0,
        "current_step": 0,
        "current_epoch": 0,
        "total_valid_tokens": 0,
    }


@dataclass
class SFTTrainer(BaseTrainer):
    """
    Supervised Fine-Tuning trainer.

    Provides full integration with NeMo RL's SFT algorithm for training
    models on expert agent trajectories before RL training.

    This trainer:
    - Initializes NeMo RL Policy with optimizer
    - Loads and processes JSONL trajectory data
    - Computes NLL loss for causal language modeling
    - Supports response-only loss for chat models
    - Handles checkpoint saving with model weights

    Example:
        from dreadnode.core.training import SFTTrainer

        trainer = SFTTrainer(
            config={
                "model_name": "meta-llama/Llama-3.1-8B-Instruct",
                "train_file": "trajectories.jsonl",
                "max_num_epochs": 3,
                "learning_rate": 1e-5,
            }
        )

        state = await trainer.train()
    """

    config: SFTConfig = field(default_factory=dict)  # type: ignore
    callbacks: list[TrainingCallback] = field(default_factory=list)

    # NeMo RL components (initialized during setup)
    _policy: Any = None
    _tokenizer: Any = None
    _loss_fn: Any = None
    _cluster: Any = None
    _logger: Any = None
    _checkpointer: Any = None
    _save_state: SFTSaveState = field(default_factory=_default_sft_save_state)

    # State tracking
    _initialized: bool = False

    def __post_init__(self):
        self.state = TrainingState()
        self.rewards = None  # SFT doesn't use rewards

    async def setup(
        self,
        cluster_config: dict[str, Any] | None = None,
    ) -> None:
        """
        Set up NeMo RL components for training.

        Initializes:
        - Ray virtual cluster for training
        - Policy model with optimizer
        - NLL loss function
        - Logger and checkpointer

        Args:
            cluster_config: Optional cluster configuration override.
        """
        import ray
        from nemo_rl.algorithms.loss_functions import NLLLoss
        from nemo_rl.algorithms.utils import get_tokenizer, set_seed
        from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
        from nemo_rl.models.policy.lm_policy import Policy
        from nemo_rl.utils.checkpoint import CheckpointManager
        from nemo_rl.utils.logger import Logger

        if self._initialized:
            return

        # Initialize Ray if not already
        if not ray.is_initialized():
            init_ray()

        # Set seed
        seed = self.config.get("seed", 42)
        set_seed(seed)

        # Get tokenizer
        tokenizer_name = self.config.get("tokenizer_name") or self.config.get("model_name")
        if tokenizer_name:
            self._tokenizer = get_tokenizer({"name_or_path": tokenizer_name})

        # Set up cluster
        cluster_cfg = cluster_config or {}
        num_nodes = cluster_cfg.get("num_nodes", 1)
        gpus_per_node = cluster_cfg.get("gpus_per_node", torch.cuda.device_count() if torch.cuda.is_available() else 1)

        self._cluster = RayVirtualCluster(
            name="sft_train_cluster",
            bundle_ct_per_node_list=[gpus_per_node] * num_nodes,
            use_gpus=True,
            num_gpus_per_node=gpus_per_node,
            max_colocated_worker_groups=1,
        )

        # Initialize policy
        model_name = self.config.get("model_name")
        if not model_name:
            raise ValueError("model_name is required in config")

        policy_config = {
            "model_name": model_name,
            "tokenizer": {"name_or_path": model_name},
            "train_global_batch_size": self.config.get("train_global_batch_size", self.config.get("batch_size", 8)),
            "train_micro_batch_size": self.config.get("train_micro_batch_size", 1),
            "max_seq_length": self.config.get("max_seq_length", 4096),
            "learning_rate": self.config.get("learning_rate", 1e-5),
        }

        self._policy = Policy(
            cluster=self._cluster,
            config=policy_config,
            tokenizer=self._tokenizer,
            init_optimizer=True,
        )

        # Initialize loss function
        self._loss_fn = NLLLoss()

        # Initialize logger
        log_dir = self.config.get("log_dir", "./logs")
        self._logger = Logger({
            "log_dir": log_dir,
            "wandb_enabled": self.config.get("wandb_enabled", False),
            "tensorboard_enabled": self.config.get("tensorboard_enabled", True),
        })

        # Initialize checkpointer
        checkpoint_dir = self.config.get("checkpoint_dir", "./checkpoints")
        self._checkpointer = CheckpointManager({
            "enabled": self.config.get("checkpoint_dir") is not None,
            "checkpoint_dir": checkpoint_dir,
            "save_period": self.config.get("checkpoint_interval", 500),
        })

        self._initialized = True

    async def train(
        self,
        prompts: Sequence[str] | None = None,
        environment: Any | None = None,
        num_steps: int | None = None,
    ) -> TrainingState:
        """
        Run SFT training.

        This implements the full SFT training loop:
        1. Load training data from JSONL file
        2. Process data into NeMo RL format
        3. Train policy with NLL loss
        4. Run validation periodically

        Args:
            prompts: Optional prompts (unused if train_file provided).
            environment: Unused for SFT.
            num_steps: Maximum training steps.

        Returns:
            Final training state.
        """
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict

        if not self._initialized:
            await self.setup()

        self.state.started_at = datetime.now()

        train_file = self.config.get("train_file")
        max_epochs = self.config.get("max_num_epochs", self.config.get("num_epochs", 3))
        max_steps = num_steps or self.config.get("max_num_steps")
        val_period = self.config.get("val_period", 0)

        # Load training data
        train_data = self._load_training_data(train_file, prompts)
        if not train_data:
            raise ValueError("No training data provided")

        batch_size = self.config.get("train_global_batch_size", self.config.get("batch_size", 8))
        total_samples = len(train_data)

        current_step = self._save_state.get("current_step", 0)
        current_epoch = self._save_state.get("current_epoch", 0)
        total_valid_tokens = self._save_state.get("total_valid_tokens", 0)

        # Validation at start if configured
        if self.config.get("val_at_start") and current_step == 0:
            val_metrics = await self.evaluate(None, None)
            self._logger.log_metrics(val_metrics, 0, prefix="validation")
            self._notify_evaluation(val_metrics)

        while current_epoch < max_epochs:
            self.state.epoch = current_epoch
            self._notify_epoch_start()

            epoch_loss = 0.0
            epoch_tokens = 0

            for i in range(0, total_samples, batch_size):
                if max_steps and current_step >= max_steps:
                    break

                self.state.step = current_step
                self._notify_step_start()

                # Get batch
                batch_data = train_data[i : i + batch_size]

                # Prepare batch for NeMo RL
                train_batch = self._prepare_batch(batch_data)

                # Train step
                self._policy.prepare_for_training()
                train_results = self._policy.train(train_batch, self._loss_fn)

                # Extract metrics
                batch_loss = float(train_results["loss"])
                grad_norm = float(train_results["grad_norm"])
                valid_tokens = int(train_results.get("global_valid_toks", len(batch_data)))

                epoch_loss += batch_loss * valid_tokens
                epoch_tokens += valid_tokens
                total_valid_tokens += valid_tokens
                self.state.samples_seen += len(batch_data)

                metrics = {
                    "loss": batch_loss,
                    "grad_norm": grad_norm,
                    "epoch": current_epoch,
                    "samples_seen": self.state.samples_seen,
                    "tokens_seen": total_valid_tokens,
                }
                self.state.metrics = metrics
                self._logger.log_metrics(metrics, current_step, prefix="train")
                self._notify_step_end(metrics)

                # Validation
                if val_period > 0 and (current_step + 1) % val_period == 0:
                    val_metrics = await self.evaluate(None, None)
                    self._logger.log_metrics(val_metrics, current_step + 1, prefix="validation")
                    self._notify_evaluation(val_metrics)

                # Checkpointing
                checkpoint_interval = self.config.get("checkpoint_interval", 500)
                checkpoint_dir = self.config.get("checkpoint_dir")
                if checkpoint_dir and (current_step + 1) % checkpoint_interval == 0:
                    self._save_state["current_step"] = current_step + 1
                    self._save_state["current_epoch"] = current_epoch
                    self._save_state["total_valid_tokens"] = total_valid_tokens
                    self.save_checkpoint(f"{checkpoint_dir}/step_{current_step + 1}")

                current_step += 1

            if max_steps and current_step >= max_steps:
                break

            # Epoch metrics
            epoch_metrics = {
                "epoch_loss": epoch_loss / epoch_tokens if epoch_tokens > 0 else 0.0,
                "epoch": current_epoch,
                "epoch_tokens": epoch_tokens,
            }
            self._notify_epoch_end(epoch_metrics)

            current_epoch += 1

        return self.state

    def _load_training_data(
        self,
        train_file: str | None,
        prompts: Sequence[str] | None,
    ) -> list[dict[str, Any]]:
        """Load training data from file or prompts."""
        import json

        train_data = []

        if train_file:
            with open(train_file) as f:
                for line in f:
                    train_data.append(json.loads(line))

        if not train_data and prompts:
            # Use prompts as simple training data
            train_data = [{"messages": [{"role": "user", "content": p}]} for p in prompts]

        return train_data

    def _prepare_batch(self, batch_data: list[dict[str, Any]]) -> Any:
        """Prepare a batch for NeMo RL training."""
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict

        input_ids_list = []
        input_lengths = []
        token_masks = []

        loss_type = self.config.get("loss_type", "causal_lm")
        max_seq_length = self.config.get("max_seq_length", 4096)

        for item in batch_data:
            messages = item.get("messages", [])

            # Apply chat template
            text = self._tokenizer.apply_chat_template(messages, tokenize=False)
            tokens = self._tokenizer.encode(text, add_special_tokens=True, max_length=max_seq_length, truncation=True)

            input_ids_list.append(torch.tensor(tokens))
            input_lengths.append(len(tokens))

            # Create token mask
            if loss_type == "response_only":
                # Only compute loss on assistant responses
                mask = torch.zeros(len(tokens))
                # Find assistant response tokens (simplified)
                for msg in messages:
                    if msg.get("role") == "assistant":
                        # Mark assistant tokens for loss
                        mask[-len(self._tokenizer.encode(msg.get("content", ""))) :] = 1
                token_masks.append(mask)
            else:
                # Compute loss on all tokens (causal LM)
                token_masks.append(torch.ones(len(tokens)))

        # Pad to max length in batch
        max_len = max(input_lengths)
        padded_ids = torch.full((len(batch_data), max_len), self._tokenizer.pad_token_id)
        padded_masks = torch.zeros((len(batch_data), max_len))

        for i, (ids, mask) in enumerate(zip(input_ids_list, token_masks)):
            padded_ids[i, : len(ids)] = ids
            padded_masks[i, : len(mask)] = mask

        batch = BatchedDataDict({
            "input_ids": padded_ids,
            "input_lengths": torch.tensor(input_lengths),
            "token_mask": padded_masks,
            "sample_mask": torch.ones(len(batch_data)),
        })
        batch.to("cpu")

        return batch

    async def evaluate(
        self,
        prompts: Sequence[str] | None = None,
        environment: Any | None = None,
    ) -> dict[str, float]:
        """
        Run evaluation on validation set.

        Args:
            prompts: Unused (uses val_file from config).
            environment: Unused for SFT.

        Returns:
            Evaluation metrics including loss and perplexity.
        """
        import json

        val_file = self.config.get("val_file")
        if not val_file:
            return {"loss": 0.0}

        val_data = []
        with open(val_file) as f:
            for line in f:
                val_data.append(json.loads(line))

        if not val_data:
            return {"loss": 0.0}

        val_batch_size = self.config.get("val_global_batch_size", self.config.get("batch_size", 8))
        val_batches = self.config.get("val_batches", len(val_data) // val_batch_size or 1)

        total_loss = 0.0
        total_tokens = 0

        for i in range(min(val_batches, len(val_data) // val_batch_size)):
            batch_data = val_data[i * val_batch_size : (i + 1) * val_batch_size]
            val_batch = self._prepare_batch(batch_data)

            # Evaluate (no gradient updates)
            val_results = self._policy.train(val_batch, self._loss_fn, eval_mode=True)

            batch_loss = float(val_results["loss"])
            valid_tokens = int(val_results.get("global_valid_toks", len(batch_data)))

            total_loss += batch_loss * valid_tokens
            total_tokens += valid_tokens

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "num_samples": min(val_batches * val_batch_size, len(val_data)),
            "num_tokens": total_tokens,
        }

    def save_checkpoint(self, path: str) -> None:
        """
        Save model checkpoint including policy weights.

        Args:
            path: Directory path for checkpoint.
        """
        import json

        os.makedirs(path, exist_ok=True)

        state_dict = {
            "step": self.state.step,
            "epoch": self.state.epoch,
            "samples_seen": self.state.samples_seen,
            "metrics": self.state.metrics,
            "save_state": dict(self._save_state),
        }

        with open(f"{path}/training_state.json", "w") as f:
            json.dump(state_dict, f, indent=2)

        # Save policy weights via NeMo RL
        if self._policy is not None:
            self._policy.prepare_for_training()
            self._policy.save_checkpoint(
                weights_path=os.path.join(path, "policy", "weights"),
                optimizer_path=os.path.join(path, "policy", "optimizer"),
                tokenizer_path=os.path.join(path, "policy", "tokenizer"),
                checkpointing_cfg={"save_optimizer": True},
            )

        if self._checkpointer is not None:
            self._checkpointer.finalize_checkpoint(path)

    def load_checkpoint(self, path: str) -> None:
        """
        Load model checkpoint including policy weights.

        Args:
            path: Directory path for checkpoint.
        """
        import json

        with open(f"{path}/training_state.json") as f:
            state_dict = json.load(f)

        self.state.step = state_dict["step"]
        self.state.epoch = state_dict["epoch"]
        self.state.samples_seen = state_dict["samples_seen"]
        self.state.metrics = state_dict.get("metrics", {})
        self._save_state = state_dict.get("save_state", _default_sft_save_state())

    def to_nemo_config(self) -> dict[str, Any]:
        """
        Convert to NeMo RL MasterConfig format.

        Returns config suitable for nemo_rl.algorithms.sft.sft_train().
        """
        return {
            "policy": {
                "model_name": self.config.get("model_name"),
                "tokenizer": {
                    "name_or_path": self.config.get("tokenizer_name") or self.config.get("model_name"),
                },
                "train_global_batch_size": self.config.get("train_global_batch_size", self.config.get("batch_size", 8)),
                "train_micro_batch_size": self.config.get("train_micro_batch_size", 1),
                "max_seq_length": self.config.get("max_seq_length", 4096),
                "learning_rate": self.config.get("learning_rate", 1e-5),
            },
            "sft": {
                "max_num_epochs": self.config.get("max_num_epochs", self.config.get("num_epochs", 3)),
                "max_num_steps": self.config.get("max_num_steps"),
                "val_period": self.config.get("val_period", 0),
                "val_batches": self.config.get("val_batches", 10),
                "val_at_start": self.config.get("val_at_start", False),
                "seed": self.config.get("seed", 42),
            },
            "data": {
                "train_file": self.config.get("train_file"),
                "val_file": self.config.get("val_file"),
                "max_seq_length": self.config.get("max_seq_length", 4096),
            },
            "logger": {
                "log_dir": self.config.get("log_dir", "./logs"),
                "wandb_enabled": self.config.get("wandb_enabled", False),
                "tensorboard_enabled": self.config.get("tensorboard_enabled", True),
            },
            "checkpointing": {
                "enabled": self.config.get("checkpoint_dir") is not None,
                "checkpoint_dir": self.config.get("checkpoint_dir", "./checkpoints"),
                "save_period": self.config.get("checkpoint_interval", 500),
            },
            "lora": {
                "enabled": self.config.get("use_lora", False),
                "r": self.config.get("lora_r", 16),
                "alpha": self.config.get("lora_alpha", 32),
                "dropout": self.config.get("lora_dropout", 0.05),
                "target_modules": self.config.get("lora_target_modules", ["q_proj", "v_proj"]),
            },
        }

    def shutdown(self) -> None:
        """Shutdown trainer and release resources."""
        if self._policy is not None:
            self._policy.shutdown()
            self._policy = None

        if self._cluster is not None:
            self._cluster.shutdown()
            self._cluster = None

        self._initialized = False

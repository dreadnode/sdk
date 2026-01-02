"""
FSDP2 Learner for distributed policy gradient training.

This module provides a learner using PyTorch 2.x FSDP2 (`fully_shard` API)
for efficient distributed training with:
- DTensor-based parameter sharding
- Better memory management (no recordStream)
- Communication-free sharded state dicts
- Native mixed precision support

References:
- https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- https://docs.ray.io/en/latest/train/examples/pytorch/pytorch-fsdp/README.html

Usage:
    from dreadnode.core.training.ray.fsdp2_learner import FSDP2Learner

    learner = FSDP2Learner(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        config=grpo_config,
        world_size=4,
        rank=0,
    )

    for batch in experience_buffer:
        metrics = learner.train_step(batch)
        print(metrics)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.distributed as dist
from torch import nn

from dreadnode.core.training.ray.config import GRPOLossConfig, RayGRPOConfig
from dreadnode.core.training.ray.experience import ExperienceBatch
from dreadnode.core.training.ray.learner import compute_log_probs, grpo_loss

if TYPE_CHECKING:
    from dreadnode.core.storage.storage import Storage
    from dreadnode.models.local import LocalModel


@dataclass
class FSDP2Config:
    """Configuration for FSDP2 training."""

    # Sharding
    sharding_strategy: Literal["full", "shard_grad_op", "no_shard"] = "full"
    """Sharding strategy (full=ZeRO-3, shard_grad_op=ZeRO-2, no_shard=DDP)."""

    # Mixed precision
    param_dtype: torch.dtype = torch.bfloat16
    """Data type for parameters."""

    reduce_dtype: torch.dtype = torch.float32
    """Data type for gradient reduction (higher precision for stability)."""

    # Memory optimization
    use_cpu_offload: bool = False
    """Offload parameters to CPU when not in use."""

    ref_model_offload: bool = False
    """Keep reference model on CPU, move to GPU only for KL computation.
    Saves ~50% GPU memory when KL penalty is used, at cost of slower KL computation."""

    # Checkpointing
    use_activation_checkpointing: bool = True
    """Enable activation checkpointing to save memory."""

    # State dict
    use_sharded_state_dict: bool = True
    """Use FSDP2's communication-free sharded state dicts."""


class FSDP2Learner:
    """
    Training learner using FSDP2 (fully_shard API).

    FSDP2 advantages over FSDP1:
    - DTensor-based parameter sharding for cleaner abstraction
    - Better memory management (no recordStream)
    - Simpler meta-device initialization
    - Communication-free sharded state dicts

    This learner:
    1. Loads model and applies FSDP2 sharding
    2. Manages reference model for KL penalty
    3. Executes training steps with gradient accumulation
    4. Provides state dict for weight sync to vLLM workers
    5. Saves/loads distributed checkpoints

    Attributes:
        model: The FSDP2-wrapped training model
        ref_model: Optional frozen reference model for KL
        optimizer: AdamW optimizer
        config: GRPO configuration
    """

    def __init__(
        self,
        model_name: str,
        config: RayGRPOConfig,
        fsdp_config: FSDP2Config | None = None,
        world_size: int = 1,
        rank: int = 0,
        storage: Storage | None = None,
        checkpoint_name: str | None = None,
    ) -> None:
        """
        Initialize FSDP2 learner.

        Args:
            model_name: HuggingFace model name or path, or LocalModel name (if storage provided)
            config: GRPO training configuration
            fsdp_config: FSDP2 configuration (uses defaults if None)
            world_size: Total number of training workers
            rank: This worker's rank
            storage: Optional Storage instance for CAS-based checkpointing
            checkpoint_name: Name for saving checkpoints (defaults to sanitized model_name)
        """
        self.model_name = model_name
        self.config = config
        self.fsdp_config = fsdp_config or FSDP2Config()
        self.world_size = world_size
        self.rank = rank
        self.storage = storage

        # Checkpoint naming
        if checkpoint_name:
            self.checkpoint_name = checkpoint_name
        else:
            # Sanitize model name for use as checkpoint name
            self.checkpoint_name = model_name.replace("/", "-").replace(":", "-")

        self.device = torch.device(f"cuda:{rank}")
        self.step = 0
        self._checkpoint_version = 0  # Track checkpoint versions

        # Load tokenizer
        self._load_tokenizer()

        # Load and shard model
        self.model = self._load_and_shard_model()

        # Reference model for KL penalty (optional)
        self.ref_model = None
        if config.loss.kl_coef > 0:
            self.ref_model = self._load_reference_model()

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Scheduler (optional, created in train loop)
        self.scheduler = None

    def _load_tokenizer(self) -> None:
        """Load tokenizer."""
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name or self.model_name,
            trust_remote_code=self.config.vllm.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_and_shard_model(self) -> nn.Module:
        """Load model and apply FSDP2 sharding."""
        from transformers import AutoModelForCausalLM

        # Try flash_attention_2, fall back to sdpa or eager
        attn_impl = "sdpa"  # Default to PyTorch SDPA
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            pass

        # Try loading from LocalModel in storage first
        model = None
        if self.storage is not None:
            model = self._try_load_from_storage(attn_impl)

        # Fall back to HuggingFace loading
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.fsdp_config.param_dtype,
                attn_implementation=attn_impl,
                trust_remote_code=self.config.vllm.trust_remote_code,
            )

        # Enable gradient checkpointing if requested
        if self.fsdp_config.use_activation_checkpointing:
            model.gradient_checkpointing_enable()

        # Move to device
        model = model.to(self.device)

        # Apply FSDP2 sharding if distributed
        if self.world_size > 1:
            model = self._apply_fsdp2(model)

        return model

    def _try_load_from_storage(self, attn_impl: str) -> nn.Module | None:
        """Try to load model from LocalModel in storage."""
        if self.storage is None:
            return None

        try:
            from dreadnode.models.local import LocalModel

            # Check if model exists in storage
            local_model = LocalModel(self.model_name, self.storage)
            print(f"Loading model from CAS: {local_model.name} v{local_model.version}")

            return local_model.to_hf(
                torch_dtype=self.fsdp_config.param_dtype,
                trust_remote_code=self.config.vllm.trust_remote_code,
            )
        except FileNotFoundError:
            # Model not in storage, will load from HuggingFace
            return None
        except Exception as e:
            print(f"Failed to load from storage: {e}, falling back to HuggingFace")
            return None

    def _apply_fsdp2(self, model: nn.Module) -> nn.Module:
        """Apply FSDP2 sharding to model using fully_shard API."""
        try:
            from torch.distributed.fsdp import (
                MixedPrecisionPolicy,
                fully_shard,
            )
            from torch.distributed.device_mesh import init_device_mesh
        except ImportError:
            # Fall back to older FSDP if FSDP2 not available
            return self._apply_fsdp1_fallback(model)

        # Initialize device mesh for data parallelism
        # For simple DP, use a 1D mesh across all ranks
        device_mesh = init_device_mesh("cuda", (self.world_size,))

        # Mixed precision policy
        mp_policy = MixedPrecisionPolicy(
            param_dtype=self.fsdp_config.param_dtype,
            reduce_dtype=self.fsdp_config.reduce_dtype,
        )

        # FSDP2 pattern: shard each transformer layer first
        # This matches how the model is structured
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # Standard HuggingFace transformer structure
            for layer in model.model.layers:
                fully_shard(
                    layer,
                    mesh=device_mesh,
                    mp_policy=mp_policy,
                )
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # GPT-2 style structure
            for layer in model.transformer.h:
                fully_shard(
                    layer,
                    mesh=device_mesh,
                    mp_policy=mp_policy,
                )

        # Then shard root model
        fully_shard(
            model,
            mesh=device_mesh,
            mp_policy=mp_policy,
        )

        return model

    def _apply_fsdp1_fallback(self, model: nn.Module) -> nn.Module:
        """Fall back to FSDP1 if FSDP2 not available."""
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
        )

        mp_policy = MixedPrecision(
            param_dtype=self.fsdp_config.param_dtype,
            reduce_dtype=self.fsdp_config.reduce_dtype,
            buffer_dtype=self.fsdp_config.param_dtype,
        )

        return FSDP(
            model,
            mixed_precision=mp_policy,
            device_id=self.rank,
        )

    def _load_reference_model(self) -> nn.Module:
        """Load frozen reference model for KL penalty.

        If ref_model_offload is enabled, keeps model on CPU to save GPU memory.
        """
        from transformers import AutoModelForCausalLM

        # Try flash_attention_2, fall back to sdpa
        # Note: flash_attention_2 requires GPU, so use sdpa for CPU offload
        if self.fsdp_config.ref_model_offload:
            attn_impl = "sdpa"  # SDPA works on both CPU and GPU
        else:
            attn_impl = "sdpa"
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
            except ImportError:
                pass

        ref_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.fsdp_config.param_dtype,
            attn_implementation=attn_impl,
            trust_remote_code=self.config.vllm.trust_remote_code,
        )

        # Keep on CPU if offload is enabled
        if self.fsdp_config.ref_model_offload:
            ref_model = ref_model.to("cpu")
            print(f"Reference model loaded on CPU (offload enabled)")
        else:
            ref_model = ref_model.to(self.device)

        ref_model.eval()

        # Freeze parameters
        for param in ref_model.parameters():
            param.requires_grad = False

        return ref_model

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
        )

    def train_step(self, batch: ExperienceBatch) -> dict[str, float]:
        """
        Execute single training step.

        Args:
            batch: Experience batch (will be tensorized if not already)

        Returns:
            Dictionary of training metrics
        """
        # Tensorize batch if needed
        if not batch.is_tensorized:
            batch = batch.to_tensors(self.tokenizer, device=self.device)

        self.model.train()

        # Forward pass
        outputs = self.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        )

        # Compute reference logprobs if needed
        ref_logprobs = None
        if self.ref_model is not None and batch.ref_logprobs is None:
            ref_logprobs = self._compute_reference_logprobs(batch)

        # Create training batch for loss computation
        from dreadnode.core.training.ray.learner import TrainingBatch

        training_batch = TrainingBatch(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            completion_mask=batch.completion_mask,
            advantages=batch.advantages,
            generation_logprobs=batch.logprobs if batch.logprobs is not None else torch.zeros_like(batch.input_ids, dtype=torch.float32),
            reference_logprobs=ref_logprobs or batch.ref_logprobs,
        )

        # Compute GRPO loss
        loss, metrics = grpo_loss(outputs.logits, training_batch, self.config.loss)

        # Backward pass
        loss.backward()

        # Gradient clipping
        # Note: FSDP2 (composable) uses standard PyTorch gradient clipping
        # The gradients are already unsharded during backward pass
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()
            metrics["learning_rate"] = self.scheduler.get_last_lr()[0]

        self.step += 1
        metrics["step"] = self.step

        # Add reward metrics
        if batch.rewards is not None:
            metrics["reward_mean"] = batch.rewards.mean().item()
            metrics["reward_std"] = batch.rewards.std().item()

        return metrics

    def _compute_reference_logprobs(self, batch: ExperienceBatch) -> torch.Tensor:
        """Compute log probabilities from reference model.

        If ref_model_offload is enabled, temporarily moves model to GPU.
        """
        with torch.no_grad():
            # Move ref model to GPU if offloaded
            if self.fsdp_config.ref_model_offload:
                self.ref_model = self.ref_model.to(self.device)

            try:
                outputs = self.ref_model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                )

                logprobs = compute_log_probs(
                    outputs.logits,
                    batch.input_ids,
                    batch.completion_mask,
                )

                # Pad to match input shape
                padded = torch.zeros_like(batch.input_ids, dtype=torch.float32)
                padded[:, 1:] = logprobs

            finally:
                # Move ref model back to CPU if offloaded
                if self.fsdp_config.ref_model_offload:
                    self.ref_model = self.ref_model.to("cpu")
                    torch.cuda.empty_cache()

            return padded

    def get_state_dict(self) -> dict[str, torch.Tensor]:
        """
        Get model state dict for weight sync to vLLM.

        For FSDP2, this returns the full unsharded state dict
        (communication required). For single GPU, returns directly.

        Returns:
            Model state dict with all parameters on CPU
        """
        if self.world_size > 1:
            # FSDP2: get full state dict (requires communication)
            try:
                from torch.distributed.fsdp import FullStateDictConfig, StateDictType
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

                # Configure full state dict
                full_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

                with FSDP.state_dict_type(
                    self.model,
                    StateDictType.FULL_STATE_DICT,
                    full_config,
                ):
                    state_dict = self.model.state_dict()

                # Only rank 0 has the full state dict
                if self.rank == 0:
                    return {k: v.cpu() for k, v in state_dict.items()}
                else:
                    return {}

            except Exception:
                # Fall back to direct state_dict
                return {k: v.cpu() for k, v in self.model.state_dict().items()}
        else:
            return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def get_sharded_state_dict(self) -> dict[str, torch.Tensor]:
        """
        Get sharded state dict (FSDP2 communication-free).

        This returns the local shard of the state dict, which can be
        saved without communication for distributed checkpointing.

        Returns:
            Local shard of model state dict
        """
        if self.world_size > 1:
            try:
                from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

                sharded_config = ShardedStateDictConfig(offload_to_cpu=True)

                with FSDP.state_dict_type(
                    self.model,
                    StateDictType.SHARDED_STATE_DICT,
                    sharded_config,
                ):
                    return self.model.state_dict()

            except Exception:
                return self.model.state_dict()
        else:
            return self.model.state_dict()

    def save_checkpoint(self, path: str) -> None:
        """
        Save distributed checkpoint using PyTorch DCP.

        Args:
            path: Directory to save checkpoint
        """
        import os

        os.makedirs(path, exist_ok=True)

        if self.world_size > 1:
            # Use distributed checkpoint for FSDP
            try:
                from torch.distributed.checkpoint import save

                save(
                    state_dict={
                        "model": self.get_sharded_state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "step": self.step,
                    },
                    checkpoint_id=path,
                )
            except ImportError:
                # Fall back to regular save on rank 0
                if self.rank == 0:
                    self._save_regular_checkpoint(path)
        else:
            self._save_regular_checkpoint(path)

    def _save_regular_checkpoint(self, path: str) -> None:
        """Save regular (non-distributed) checkpoint."""
        import os

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step": self.step,
                "config": self.config.to_dict(),
            },
            os.path.join(path, "checkpoint.pt"),
        )

        # Also save HuggingFace format
        self.model.save_pretrained(os.path.join(path, "model"))
        self.tokenizer.save_pretrained(os.path.join(path, "model"))

        # Save to CAS if storage is available
        if self.storage is not None:
            self._save_to_storage()

    def save_for_vllm(self, path: str) -> str:
        """Save model in format loadable by vLLM.

        Saves just the model weights and tokenizer, optimized for vLLM loading.

        Args:
            path: Directory to save to

        Returns:
            Path to saved model directory
        """
        import os

        os.makedirs(path, exist_ok=True)

        # Save model and tokenizer in HuggingFace format
        self.model.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)

        return path

    def _save_to_storage(self, version: str | None = None) -> LocalModel:
        """Save model checkpoint to CAS using LocalModel.

        Args:
            version: Version string. If None, auto-increments.

        Returns:
            LocalModel instance for the saved checkpoint.
        """
        from dreadnode.models.local import LocalModel

        if self.storage is None:
            raise ValueError("Storage not configured")

        # Auto-increment version if not specified
        if version is None:
            self._checkpoint_version += 1
            version = f"0.{self._checkpoint_version}.0"

        checkpoint_name = f"{self.checkpoint_name}-step{self.step}"

        print(f"Saving checkpoint to CAS: {checkpoint_name} v{version}")

        # Save model to CAS
        local_model = LocalModel.from_hf(
            model=self.model,
            name=checkpoint_name,
            storage=self.storage,
            tokenizer=self.tokenizer,
            format="safetensors",
            task="text-generation",
            version=version,
        )

        print(f"Saved to CAS: {local_model}")
        return local_model

    def save_checkpoint_to_storage(self, version: str | None = None) -> LocalModel | None:
        """Public method to save checkpoint to CAS.

        Args:
            version: Version string. If None, auto-increments.

        Returns:
            LocalModel instance if storage is configured, None otherwise.
        """
        if self.storage is None:
            print("Storage not configured, skipping CAS checkpoint")
            return None

        return self._save_to_storage(version)

    def load_checkpoint(self, path: str) -> None:
        """
        Load distributed checkpoint.

        Args:
            path: Directory containing checkpoint
        """
        import os

        if self.world_size > 1:
            try:
                from torch.distributed.checkpoint import load

                state_dict = {
                    "model": self.get_sharded_state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "step": 0,
                }

                load(state_dict, checkpoint_id=path)

                self.step = state_dict["step"]

            except ImportError:
                self._load_regular_checkpoint(path)
        else:
            self._load_regular_checkpoint(path)

    def _load_regular_checkpoint(self, path: str) -> None:
        """Load regular checkpoint."""
        import os

        checkpoint_path = os.path.join(path, "checkpoint.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.step = checkpoint["step"]

    def create_scheduler(
        self,
        total_steps: int,
        warmup_ratio: float | None = None,
    ) -> None:
        """
        Create learning rate scheduler.

        Args:
            total_steps: Total training steps
            warmup_ratio: Warmup steps as fraction (defaults to config)
        """
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup_ratio = warmup_ratio or self.config.warmup_ratio
        warmup_steps = int(total_steps * warmup_ratio)

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        decay_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.config.learning_rate * 0.1,
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps],
        )


def init_distributed(rank: int, world_size: int, backend: str = "nccl") -> None:
    """
    Initialize distributed training.

    Args:
        rank: This process's rank
        world_size: Total number of processes
        backend: Communication backend (nccl, gloo)
    """
    import os

    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()

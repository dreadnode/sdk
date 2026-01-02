"""
Ray-based GRPO Trainer.

Coordinates distributed inference and training for GRPO.
Implements colocated design where inference and training time-share GPUs.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dreadnode.core.training.ray.callbacks import CallbackHandler, TrainerCallback, TrainerControl
from dreadnode.core.training.ray.config import RayGRPOConfig
from dreadnode.core.training.ray.learner import TrainingBatch, compute_log_probs, grpo_loss

if TYPE_CHECKING:
    from dreadnode.core.storage.storage import Storage
    from dreadnode.models.local import LocalModel


@dataclass
class TrainingMetrics:
    """Metrics from training."""

    step: int = 0
    loss: float = 0.0
    pg_loss: float = 0.0
    kl_loss: float = 0.0
    reward_mean: float = 0.0
    reward_std: float = 0.0
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    generation_time: float = 0.0
    training_time: float = 0.0
    tokens_per_second: float = 0.0
    samples_per_second: float = 0.0


@dataclass
class TrainingState:
    """Current state of GRPO training."""

    step: int = 0
    epoch: int = 0
    samples_seen: int = 0
    best_reward: float = float("-inf")
    metrics_history: list[TrainingMetrics] = field(default_factory=list)
    started_at: datetime | None = None

    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        return (datetime.now() - self.started_at).total_seconds()


RewardFn = Callable[[list[str], list[str]], list[float]]
"""Reward function type: (prompts, completions) -> rewards"""


def clear_gpu_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class RayGRPOTrainer:
    """
    Native Ray-based GRPO trainer with colocated inference/training.

    Supports two modes:
    1. Memory-efficient mode (default): Time-shares GPU between vLLM and training
       - Lower memory, but slower due to model loading/unloading
    2. Fast mode (colocate=True): Keeps both models loaded
       - Higher memory usage, but much faster (no reload overhead)
       - Uses in-place vLLM weight updates

    Example:
        >>> config = RayGRPOConfig(
        ...     model_name="Qwen/Qwen2.5-1.5B-Instruct",
        ...     num_generations_per_prompt=4,
        ... )
        >>> trainer = RayGRPOTrainer(config, colocate=True)  # Fast mode
        >>>
        >>> def reward_fn(prompts, completions):
        ...     return [1.0 if is_correct(c) else 0.0 for c in completions]
        >>>
        >>> trainer.train(prompts, reward_fn)
    """

    def __init__(
        self,
        config: RayGRPOConfig,
        colocate: bool = False,
        storage: Storage | None = None,
        checkpoint_name: str | None = None,
        callbacks: list[TrainerCallback] | None = None,
    ):
        """
        Initialize GRPO trainer.

        Args:
            config: GRPO configuration.
            colocate: If True, keep both vLLM and training model loaded (faster but more memory).
            storage: Optional Storage for CAS-based checkpointing.
            checkpoint_name: Name for checkpoints (defaults to sanitized model name).
            callbacks: List of TrainerCallback instances for customizing training behavior.
        """
        self.config = config
        self.colocate = colocate
        self.storage = storage
        self.state = TrainingState()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Callback handler
        self.callback_handler = CallbackHandler(callbacks or [])

        # Checkpoint naming
        if checkpoint_name:
            self.checkpoint_name = checkpoint_name
        else:
            self.checkpoint_name = config.model_name.replace("/", "-").replace(":", "-")

        self._checkpoint_version = 0  # Track checkpoint versions

        # Load tokenizer (shared between inference and training)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name or config.model_name,
            trust_remote_code=config.vllm.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # These are loaded/unloaded as needed
        self._vllm_engine = None
        self._train_model = None
        self._ref_model = None
        self._optimizer = None

        # Saved weights path for weight sync
        self._weights_path = "/tmp/grpo_weights.pt"

        # Track if we need to sync weights to vLLM
        self._weights_dirty = False

        # Checkpoint dir for vLLM reload (set when in-place sync fails)
        self._checkpoint_dir_for_vllm: str | None = None

    def _load_vllm(self):
        """Load vLLM engine for inference."""
        if self._vllm_engine is not None:
            return

        from vllm import LLM

        # Use saved checkpoint if available (after weight sync)
        if self._checkpoint_dir_for_vllm is not None:
            model_path = self._checkpoint_dir_for_vllm
            print(f"  Loading vLLM from checkpoint...", flush=True)
        else:
            model_path = self.config.model_name
            print("  Loading vLLM engine...", flush=True)

        self._vllm_engine = LLM(
            model=model_path,
            tensor_parallel_size=self.config.vllm.tensor_parallel_size,
            gpu_memory_utilization=self.config.vllm.gpu_memory_utilization,
            max_model_len=self.config.vllm.max_model_len,
            enforce_eager=self.config.vllm.enforce_eager,
            dtype=self.config.vllm.dtype,
            trust_remote_code=self.config.vllm.trust_remote_code,
        )

    def _unload_vllm(self):
        """Unload vLLM engine to free GPU memory."""
        if self._vllm_engine is None:
            return

        print("  Unloading vLLM engine...", flush=True)
        del self._vllm_engine
        self._vllm_engine = None
        clear_gpu_memory()

    def _sync_weights_to_vllm(self):
        """
        Sync training weights to vLLM.

        Tries in-place sync first, falls back to checkpoint-based reload if needed.
        """
        if self._train_model is None:
            return

        if not self._weights_dirty:
            return

        print("  Syncing weights to vLLM...", flush=True)
        t0 = time.time()

        # Try to update vLLM weights in-place (vLLM v0 engine only)
        if self._vllm_engine is not None:
            try:
                model_runner = (
                    self._vllm_engine.llm_engine.model_executor.driver_worker.model_runner
                )
                state_dict = self._train_model.state_dict()
                model_runner.model.load_weights(state_dict.items())
                self._weights_dirty = False
                print(f"  Weight sync took {time.time() - t0:.2f}s")
                return
            except (AttributeError, RuntimeError):
                # vLLM v1 engine doesn't support in-place weight updates
                # Fall back to non-colocate mode for remaining training
                print("  vLLM v1 detected - switching to reload mode")
                self.colocate = False

        # Non-colocate mode: save checkpoint, unload training model
        # The vLLM reload will happen at the start of the next generation phase
        print("  Saving checkpoint for vLLM reload...")
        checkpoint_dir = "/tmp/grpo_vllm_checkpoint"
        self._save_vllm_checkpoint(checkpoint_dir)
        self._checkpoint_dir_for_vllm = checkpoint_dir

        # Unload both models
        self._unload_vllm()
        self._unload_train_model()
        self._weights_dirty = False
        print(f"  Weight sync (checkpoint save) took {time.time() - t0:.2f}s")

    def _save_vllm_checkpoint(self, path: str):
        """Save a checkpoint that vLLM can load."""
        import os

        os.makedirs(path, exist_ok=True)

        # Use transformers save_pretrained which handles tied weights correctly
        self._train_model.save_pretrained(path, safe_serialization=True)

        # Copy config and tokenizer from original model
        from transformers import AutoConfig, AutoTokenizer

        config = AutoConfig.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.vllm.trust_remote_code,
        )
        config.save_pretrained(path)

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.vllm.trust_remote_code,
        )
        tokenizer.save_pretrained(path)

    def _load_vllm_from_checkpoint(self, checkpoint_path: str):
        """Load vLLM from a saved checkpoint."""
        from vllm import LLM

        print(f"  Loading vLLM from checkpoint...", flush=True)
        self._vllm_engine = LLM(
            model=checkpoint_path,
            tensor_parallel_size=self.config.vllm.tensor_parallel_size,
            gpu_memory_utilization=self.config.vllm.gpu_memory_utilization,
            max_model_len=self.config.vllm.max_model_len,
            enforce_eager=self.config.vllm.enforce_eager,
            dtype=self.config.vllm.dtype,
            trust_remote_code=True,  # Trust our saved checkpoint
        )

    def _load_train_model(self):
        """Load model for training."""
        if self._train_model is not None:
            return

        import os

        # Determine dtype
        if self.config.training.mixed_precision == "bf16":
            dtype = torch.bfloat16
        elif self.config.training.mixed_precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Load from checkpoint if available (after weight sync)
        if self._checkpoint_dir_for_vllm is not None and os.path.exists(
            self._checkpoint_dir_for_vllm
        ):
            model_path = self._checkpoint_dir_for_vllm
            print("  Loading training model from checkpoint...", flush=True)
        else:
            model_path = self.config.model_name
            print("  Loading training model...", flush=True)

        self._train_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=self.config.vllm.trust_remote_code,
        ).to(self.device)

        if self.config.training.gradient_checkpointing:
            self._train_model.gradient_checkpointing_enable()

        # Create optimizer
        self._optimizer = torch.optim.AdamW(
            self._train_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Reference model for KL penalty (frozen)
        if self.config.loss.kl_coef > 0:
            print("  Loading reference model...", flush=True)
            self._ref_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=dtype,
                trust_remote_code=self.config.vllm.trust_remote_code,
            ).to(self.device)
            self._ref_model.eval()
            for p in self._ref_model.parameters():
                p.requires_grad = False

    def _unload_train_model(self):
        """Unload training model to free GPU memory."""
        if self._train_model is None:
            return

        # Save weights before unloading
        print("  Saving weights...", flush=True)
        torch.save(self._train_model.state_dict(), self._weights_path)

        print("  Unloading training model...", flush=True)
        del self._train_model
        del self._optimizer
        if self._ref_model is not None:
            del self._ref_model
        self._train_model = None
        self._optimizer = None
        self._ref_model = None
        clear_gpu_memory()

    def _generate(
        self,
        prompts: list[str],
    ) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate completions using vLLM.

        Returns:
            Tuple of (completions, full_ids, completion_mask, gen_logprobs)
        """
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            logprobs=1,
        )

        outputs = self._vllm_engine.generate(prompts, sampling_params)

        # Process outputs
        completions = []
        all_prompt_ids = []
        all_completion_ids = []
        all_logprobs = []

        for output in outputs:
            prompt_ids = list(output.prompt_token_ids)
            completion = output.outputs[0]
            comp_ids = list(completion.token_ids)

            completions.append(completion.text)
            all_prompt_ids.append(prompt_ids)
            all_completion_ids.append(comp_ids)

            # Extract logprobs
            lps = []
            if completion.logprobs:
                for lp_dict in completion.logprobs:
                    if lp_dict:
                        token_id = list(lp_dict.keys())[0]
                        lps.append(lp_dict[token_id].logprob)
                    else:
                        lps.append(0.0)
            else:
                lps = [0.0] * len(comp_ids)
            all_logprobs.append(lps)

        # Pad and create tensors
        max_prompt_len = max(len(ids) for ids in all_prompt_ids)
        max_comp_len = max(len(ids) for ids in all_completion_ids)
        max_seq_len = max_prompt_len + max_comp_len
        batch_size = len(prompts)

        full_ids = torch.full(
            (batch_size, max_seq_len), self.tokenizer.pad_token_id, dtype=torch.long
        )
        completion_mask = torch.zeros(batch_size, max_seq_len)
        gen_logprobs = torch.zeros(batch_size, max_seq_len)

        for i in range(batch_size):
            p_len = len(all_prompt_ids[i])
            c_len = len(all_completion_ids[i])

            full_ids[i, :p_len] = torch.tensor(all_prompt_ids[i])
            full_ids[i, p_len : p_len + c_len] = torch.tensor(all_completion_ids[i])
            completion_mask[i, p_len : p_len + c_len] = 1.0
            gen_logprobs[i, p_len : p_len + c_len] = torch.tensor(all_logprobs[i])

        return completions, full_ids, completion_mask, gen_logprobs

    def _train_step_on_batch(
        self,
        full_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        gen_logprobs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> dict[str, float]:
        """Execute training step on a batch."""
        self._train_model.train()

        # Move to device
        full_ids = full_ids.to(self.device)
        completion_mask = completion_mask.to(self.device)
        gen_logprobs = gen_logprobs.to(self.device)
        advantages = advantages.to(self.device)
        attention_mask = (full_ids != self.tokenizer.pad_token_id).long()

        # Forward pass
        outputs = self._train_model(
            input_ids=full_ids,
            attention_mask=attention_mask,
        )

        # Compute reference logprobs if needed
        ref_logprobs = None
        if self._ref_model is not None:
            with torch.no_grad():
                ref_outputs = self._ref_model(
                    input_ids=full_ids,
                    attention_mask=attention_mask,
                )
                ref_logprobs_raw = compute_log_probs(ref_outputs.logits, full_ids, completion_mask)
                # Pad to full sequence length
                ref_logprobs = torch.zeros_like(gen_logprobs)
                ref_logprobs[:, 1:] = ref_logprobs_raw

        # Create training batch
        batch = TrainingBatch(
            input_ids=full_ids,
            attention_mask=attention_mask,
            completion_mask=completion_mask,
            advantages=advantages,
            generation_logprobs=gen_logprobs,
            reference_logprobs=ref_logprobs,
        )

        # Compute loss
        loss, metrics = grpo_loss(outputs.logits, batch, self.config.loss)

        # Backward
        self._optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self._train_model.parameters(),
                self.config.max_grad_norm,
            )

        self._optimizer.step()

        return metrics

    def train(
        self,
        prompts: Sequence[str],
        reward_fn: RewardFn,
        eval_prompts: Sequence[str] | None = None,
        num_steps: int | None = None,
    ) -> TrainingState:
        """
        Run GRPO training.

        Args:
            prompts: Training prompts.
            reward_fn: Function to score completions.
            eval_prompts: Optional evaluation prompts.
            num_steps: Optional number of steps (overrides config).

        Returns:
            Final training state.
        """
        self.state.started_at = datetime.now()
        max_steps = num_steps or self.config.max_steps

        mode_name = "Fast Colocated" if self.colocate else "Memory-Efficient"
        print(f"\n{'=' * 60}")
        print(f" GRPO Training ({mode_name} Mode)")
        print(f"{'=' * 60}")
        print(f"  Model: {self.config.model_name}")
        print(f"  Prompts: {len(prompts)}")
        print(f"  Generations per prompt: {self.config.num_generations_per_prompt}")
        print(f"  Max steps: {max_steps}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"{'=' * 60}\n")

        prompt_list = list(prompts)
        batch_size = self.config.num_prompts_per_step

        # Notify callbacks of training start
        control = self.callback_handler.on_train_begin(self.state)

        # In colocate mode, load both models upfront
        if self.colocate:
            print("Loading models for colocated training...")
            self._load_train_model()  # Load training model first (smaller memory footprint)
            self._load_vllm()  # Then vLLM

        try:
            for step in range(max_steps):
                self.state.step = step

                # Step begin callback
                control = self.callback_handler.on_step_begin(self.state)
                if control.should_stop:
                    print("Early stopping triggered by callback")
                    break

                print(f"\n--- Step {step + 1}/{max_steps} ---")

                metrics = TrainingMetrics(step=step)

                # === GENERATION PHASE ===
                if not self.colocate:
                    self._load_vllm()
                elif self._weights_dirty:
                    # Fast weight sync in colocate mode
                    self._sync_weights_to_vllm()

                # Sample prompts and repeat for multiple generations
                indices = torch.randint(0, len(prompt_list), (batch_size,))
                batch_prompts = [prompt_list[i] for i in indices]
                repeated_prompts = []
                for p in batch_prompts:
                    repeated_prompts.extend([p] * self.config.num_generations_per_prompt)

                print(f"  Generating {len(repeated_prompts)} completions...")
                t0 = time.time()
                completions, full_ids, completion_mask, gen_logprobs = self._generate(
                    repeated_prompts
                )
                metrics.generation_time = time.time() - t0

                # Compute tokens generated for throughput
                num_gen_tokens = int(completion_mask.sum().item())
                gen_tok_per_sec = num_gen_tokens / metrics.generation_time if metrics.generation_time > 0 else 0
                print(f"  Generation: {metrics.generation_time:.2f}s ({gen_tok_per_sec:.0f} tok/s)")

                # Compute rewards
                rewards = reward_fn(repeated_prompts, completions)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
                metrics.reward_mean = rewards_tensor.mean().item()
                metrics.reward_std = rewards_tensor.std().item()
                print(f"  Rewards: {metrics.reward_mean:.4f} +/- {metrics.reward_std:.4f}")

                # Generation end callback
                self.callback_handler.on_generation_end(
                    self.state,
                    prompts=repeated_prompts,
                    completions=completions,
                    rewards=rewards,
                )

                # Compute advantages
                advantages = self._compute_advantages(
                    rewards_tensor, self.config.num_generations_per_prompt
                )
                metrics.advantage_mean = advantages.mean().item()
                metrics.advantage_std = advantages.std().item()

                # Unload vLLM in memory-efficient mode
                if not self.colocate:
                    self._unload_vllm()

                # === TRAINING PHASE ===
                if not self.colocate:
                    self._load_train_model()

                print("  Training...")
                t0 = time.time()
                train_metrics = self._train_step_on_batch(
                    full_ids, completion_mask, gen_logprobs, advantages
                )
                metrics.training_time = time.time() - t0
                print(f"  Training took {metrics.training_time:.2f}s")

                metrics.loss = train_metrics["loss"]
                metrics.pg_loss = train_metrics["pg_loss"]
                metrics.kl_loss = train_metrics["kl_loss"]

                # Mark weights as dirty (need sync to vLLM)
                self._weights_dirty = True

                # Unload training model in memory-efficient mode
                if not self.colocate:
                    self._unload_train_model()

                # Throughput
                total_tokens = full_ids.numel()
                total_time = metrics.generation_time + metrics.training_time
                metrics.tokens_per_second = total_tokens / total_time
                metrics.samples_per_second = len(repeated_prompts) / total_time

                # Update state
                self.state.samples_seen += len(repeated_prompts)
                self.state.metrics_history.append(metrics)

                # Log
                print(
                    f"  Loss: {metrics.loss:.4f} | "
                    f"Reward: {metrics.reward_mean:.4f} | "
                    f"Tok/s: {metrics.tokens_per_second:.0f}"
                )

                # Prepare metrics dict for callbacks
                step_metrics = {
                    "loss": metrics.loss,
                    "pg_loss": metrics.pg_loss,
                    "kl_loss": metrics.kl_loss,
                    "reward_mean": metrics.reward_mean,
                    "reward_std": metrics.reward_std,
                    "advantage_mean": metrics.advantage_mean,
                    "advantage_std": metrics.advantage_std,
                    "generation_time": metrics.generation_time,
                    "training_time": metrics.training_time,
                    "tokens_per_second": metrics.tokens_per_second,
                    "samples_per_second": metrics.samples_per_second,
                }

                # Step end callback
                control = self.callback_handler.on_step_end(self.state, step_metrics)

                # Handle callback control signals
                if control.should_save or (step + 1) % self.config.checkpoint_interval == 0:
                    checkpoint_path = self._save_checkpoint(f"step_{step + 1}")
                    self.callback_handler.on_save(self.state, checkpoint_path=checkpoint_path or "")

                # Log callback
                if control.should_log or (step + 1) % self.config.log_interval == 0:
                    self.callback_handler.on_log(self.state, logs=step_metrics)

                if control.should_stop:
                    print("Early stopping triggered by callback")
                    break

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            self._save_checkpoint("final")
            self._cleanup()

            # Training end callback
            self.callback_handler.on_train_end(self.state)

        print(f"\nTraining complete! Best reward: {self.state.best_reward:.4f}")
        return self.state

    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        group_size: int,
    ) -> torch.Tensor:
        """Compute group-relative advantages (GRPO)."""
        num_groups = len(rewards) // group_size
        rewards = rewards.view(num_groups, group_size)

        group_mean = rewards.mean(dim=1, keepdim=True)
        group_std = rewards.std(dim=1, keepdim=True).clamp(min=1e-8)

        if self.config.loss.use_leave_one_out_baseline:
            total = rewards.sum(dim=1, keepdim=True)
            loo_mean = (total - rewards) / (group_size - 1)
            advantages = rewards - loo_mean
        else:
            advantages = rewards - group_mean

        if self.config.loss.normalize_advantages:
            advantages = advantages / group_std

        return advantages.view(-1)

    def _save_checkpoint(self, name: str) -> str | None:
        """Save checkpoint and return the path."""
        import os
        import shutil

        # Save to CAS if storage is available
        if self.storage is not None and self._train_model is not None:
            local_model = self._save_to_storage()
            if local_model:
                print(f"  Checkpoint saved to CAS: {local_model.name} v{local_model.version}")
                return f"cas://{local_model.name}@{local_model.version}"

        # Fall back to file-based checkpoint
        path = os.path.join(self.config.checkpoint_dir, name)
        os.makedirs(path, exist_ok=True)

        # Copy weights
        if os.path.exists(self._weights_path):
            shutil.copy(self._weights_path, os.path.join(path, "pytorch_model.bin"))

        # Save config
        import json

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        print(f"  Checkpoint saved to {path}")
        return path

    def _save_to_storage(self, version: str | None = None) -> LocalModel | None:
        """Save model checkpoint to CAS using LocalModel.

        Args:
            version: Version string. If None, auto-increments.

        Returns:
            LocalModel instance for the saved checkpoint, or None if failed.
        """
        if self.storage is None or self._train_model is None:
            return None

        try:
            from dreadnode.models.local import LocalModel

            # Auto-increment version if not specified
            if version is None:
                self._checkpoint_version += 1
                version = f"0.{self._checkpoint_version}.0"

            checkpoint_name = f"{self.checkpoint_name}-step{self.state.step}"

            print(f"  Saving checkpoint to CAS: {checkpoint_name} v{version}")

            # Save model to CAS
            local_model = LocalModel.from_hf(
                model=self._train_model,
                name=checkpoint_name,
                storage=self.storage,
                tokenizer=self.tokenizer,
                format="safetensors",
                task="text-generation",
                version=version,
            )

            return local_model
        except Exception as e:
            print(f"  Failed to save to CAS: {e}")
            return None

    def save_checkpoint_to_storage(self, version: str | None = None) -> LocalModel | None:
        """Public method to save checkpoint to CAS.

        Args:
            version: Version string. If None, auto-increments.

        Returns:
            LocalModel instance if storage is configured, None otherwise.
        """
        return self._save_to_storage(version)

    def _cleanup(self):
        """Cleanup resources."""
        self._unload_vllm()
        self._unload_train_model()

        import os

        if os.path.exists(self._weights_path):
            os.remove(self._weights_path)

    def add_callback(self, callback: TrainerCallback) -> None:
        """Add a callback to the trainer."""
        self.callback_handler.add_callback(callback)

    def remove_callback(self, callback_type: type) -> None:
        """Remove all callbacks of a given type."""
        self.callback_handler.remove_callback(callback_type)

    def shutdown(self) -> None:
        """Shutdown trainer."""
        self._cleanup()

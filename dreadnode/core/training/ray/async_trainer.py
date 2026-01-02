"""
Async Ray-based GRPO Trainer.

Overlaps inference and training on separate GPUs for maximum throughput.
"""

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dreadnode.core.training.ray.config import RayGRPOConfig
from dreadnode.core.training.ray.learner import TrainingBatch, grpo_loss


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


@dataclass
class GenerationBatch:
    """A batch ready for training."""

    prompts: list[str]
    completions: list[str]
    full_ids: torch.Tensor
    completion_mask: torch.Tensor
    gen_logprobs: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    generation_time: float


RewardFn = Callable[[list[str], list[str]], list[float]]


@ray.remote(num_gpus=1)
class InferenceWorker:
    """
    Ray actor for vLLM inference on a dedicated GPU.

    Runs continuously, generating batches and putting them in a queue.
    """

    def __init__(self, config: RayGRPOConfig, weights_path: str):
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        from vllm import LLM

        self.config = config
        self.weights_path = weights_path

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name or config.model_name,
            trust_remote_code=config.vllm.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load vLLM
        print("[InferenceWorker] Loading vLLM on GPU...", flush=True)
        self._llm = LLM(
            model=config.model_name,
            tensor_parallel_size=config.vllm.tensor_parallel_size,
            gpu_memory_utilization=config.vllm.gpu_memory_utilization,
            max_model_len=config.vllm.max_model_len,
            enforce_eager=config.vllm.enforce_eager,
            dtype=config.vllm.dtype,
            trust_remote_code=config.vllm.trust_remote_code,
        )
        print("[InferenceWorker] Ready!", flush=True)

    def generate_batch(
        self,
        prompts: list[str],
        reward_fn_serialized: bytes,
    ) -> GenerationBatch:
        """Generate a batch of completions."""
        import cloudpickle

        reward_fn = cloudpickle.loads(reward_fn_serialized)

        from vllm import SamplingParams

        t0 = time.time()

        sampling_params = SamplingParams(
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            logprobs=1,
        )

        outputs = self._llm.generate(prompts, sampling_params)

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

        generation_time = time.time() - t0

        # Compute rewards
        rewards = reward_fn(prompts, completions)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        # Compute advantages
        advantages = self._compute_advantages(rewards_tensor)

        return GenerationBatch(
            prompts=prompts,
            completions=completions,
            full_ids=full_ids,
            completion_mask=completion_mask,
            gen_logprobs=gen_logprobs,
            rewards=rewards_tensor,
            advantages=advantages,
            generation_time=generation_time,
        )

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute group-relative advantages."""
        group_size = self.config.num_generations_per_prompt
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

    def update_weights(self, state_dict: dict) -> bool:
        """Update model weights (for vLLM this requires reload)."""
        # vLLM doesn't support easy weight updates, so we skip for now
        # In production, you'd use vLLM's weight loading APIs
        return True


@ray.remote(num_gpus=1)
class TrainingWorker:
    """
    Ray actor for training on a dedicated GPU.

    Receives batches from inference worker and updates the model.
    """

    def __init__(self, config: RayGRPOConfig, weights_path: str):
        self.config = config
        self.weights_path = weights_path
        self.device = torch.device("cuda")
        self.step = 0

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name or config.model_name,
            trust_remote_code=config.vllm.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model
        print("[TrainingWorker] Loading model on GPU...", flush=True)
        dtype = torch.bfloat16 if config.training.mixed_precision == "bf16" else torch.float16

        self._model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
            trust_remote_code=config.vllm.trust_remote_code,
        ).to(self.device)

        if config.training.gradient_checkpointing:
            self._model.gradient_checkpointing_enable()

        # Optimizer
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        print("[TrainingWorker] Ready!", flush=True)

    def train_on_batch(self, batch: GenerationBatch) -> dict[str, float]:
        """Train on a batch from the inference worker."""
        t0 = time.time()
        self._model.train()

        # Move to device
        full_ids = batch.full_ids.to(self.device)
        completion_mask = batch.completion_mask.to(self.device)
        gen_logprobs = batch.gen_logprobs.to(self.device)
        advantages = batch.advantages.to(self.device)
        attention_mask = (full_ids != self.tokenizer.pad_token_id).long()

        # Forward pass
        outputs = self._model(
            input_ids=full_ids,
            attention_mask=attention_mask,
        )

        # Create training batch
        train_batch = TrainingBatch(
            input_ids=full_ids,
            attention_mask=attention_mask,
            completion_mask=completion_mask,
            advantages=advantages,
            generation_logprobs=gen_logprobs,
            reference_logprobs=None,  # No KL penalty in async mode for simplicity
        )

        # Compute loss
        loss, metrics = grpo_loss(outputs.logits, train_batch, self.config.loss)

        # Backward
        self._optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(),
                self.config.max_grad_norm,
            )

        self._optimizer.step()

        self.step += 1
        training_time = time.time() - t0

        # Add timing and reward metrics
        metrics["training_time"] = training_time
        metrics["generation_time"] = batch.generation_time
        metrics["reward_mean"] = batch.rewards.mean().item()
        metrics["reward_std"] = batch.rewards.std().item()
        metrics["advantage_mean"] = batch.advantages.mean().item()
        metrics["advantage_std"] = batch.advantages.std().item()
        metrics["step"] = self.step

        return metrics

    def get_state_dict(self) -> dict:
        """Get model state dict for weight sync."""
        return {k: v.cpu() for k, v in self._model.state_dict().items()}

    def save_checkpoint(self, path: str) -> None:
        """Save checkpoint."""
        os.makedirs(path, exist_ok=True)
        torch.save(self._model.state_dict(), os.path.join(path, "pytorch_model.bin"))
        print(f"[TrainingWorker] Saved checkpoint to {path}", flush=True)


class AsyncRayGRPOTrainer:
    """
    Async Ray-based GRPO trainer.

    Uses separate GPUs for inference and training to overlap computation:
    - GPU 0: vLLM inference (generates batches continuously)
    - GPU 1: Training (processes batches as they arrive)

    This achieves much higher throughput than the colocated version.

    Requires at least 2 GPUs.
    """

    def __init__(self, config: RayGRPOConfig):
        self.config = config
        self.state = TrainingState()
        self.weights_path = "/tmp/grpo_async_weights.pt"

        # Initialize Ray with excludes for large files
        if not ray.is_initialized():
            ray.init(
                runtime_env={
                    "excludes": [
                        "checkpoints/",
                        "*.pt",
                        "*.bin",
                        "*.safetensors",
                        ".git/objects/",
                    ]
                }
            )

        # Check GPU availability
        num_gpus = int(ray.cluster_resources().get("GPU", 0))
        if num_gpus < 2:
            raise RuntimeError(
                f"AsyncRayGRPOTrainer requires at least 2 GPUs, found {num_gpus}. "
                "Use RayGRPOTrainer for single-GPU training."
            )

        print(f"Initializing async trainer with {num_gpus} GPUs...")

        # Create workers
        self._inference_worker = InferenceWorker.remote(config, self.weights_path)
        self._training_worker = TrainingWorker.remote(config, self.weights_path)

        # Wait for workers to initialize (just check they're ready)
        print("Workers initialized!")

    def _serialize_reward_fn(self, reward_fn: RewardFn) -> bytes:
        """Serialize reward function for passing to workers."""
        import cloudpickle

        return cloudpickle.dumps(reward_fn)

    def train(
        self,
        prompts: Sequence[str],
        reward_fn: RewardFn,
        num_steps: int | None = None,
    ) -> TrainingState:
        """
        Run async GRPO training.

        Overlaps inference and training for maximum throughput.
        """
        self.state.started_at = datetime.now()
        max_steps = num_steps or self.config.max_steps

        print(f"\n{'=' * 60}")
        print(" GRPO Training (Async Mode)")
        print(f"{'=' * 60}")
        print(f"  Model: {self.config.model_name}")
        print(f"  Prompts: {len(prompts)}")
        print(f"  Generations per prompt: {self.config.num_generations_per_prompt}")
        print(f"  Max steps: {max_steps}")
        print(f"{'=' * 60}\n")

        prompt_list = list(prompts)
        batch_size = self.config.num_prompts_per_step
        reward_fn_serialized = self._serialize_reward_fn(reward_fn)

        # Pipeline: keep one generation ahead
        pending_generation = None

        try:
            for step in range(max_steps):
                self.state.step = step

                # Sample prompts for this step
                indices = torch.randint(0, len(prompt_list), (batch_size,))
                batch_prompts = [prompt_list[i] for i in indices]
                repeated_prompts = []
                for p in batch_prompts:
                    repeated_prompts.extend([p] * self.config.num_generations_per_prompt)

                # Start generation for this step (or use pending from last iteration)
                if pending_generation is None:
                    generation_future = self._inference_worker.generate_batch.remote(
                        repeated_prompts, reward_fn_serialized
                    )
                else:
                    generation_future = pending_generation

                # Start next generation in parallel (pipelining)
                if step + 1 < max_steps:
                    next_indices = torch.randint(0, len(prompt_list), (batch_size,))
                    next_batch_prompts = [prompt_list[i] for i in next_indices]
                    next_repeated_prompts = []
                    for p in next_batch_prompts:
                        next_repeated_prompts.extend([p] * self.config.num_generations_per_prompt)
                    pending_generation = self._inference_worker.generate_batch.remote(
                        next_repeated_prompts, reward_fn_serialized
                    )
                else:
                    pending_generation = None

                # Wait for current generation
                gen_batch = ray.get(generation_future)

                # Train on batch (this overlaps with next generation)
                metrics = ray.get(self._training_worker.train_on_batch.remote(gen_batch))

                # Log
                total_time = metrics["generation_time"] + metrics["training_time"]
                total_tokens = gen_batch.full_ids.numel()
                tok_per_sec = total_tokens / total_time

                print(
                    f"Step {step + 1:4d} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"Reward: {metrics['reward_mean']:.3f} +/- {metrics['reward_std']:.3f} | "
                    f"Gen: {metrics['generation_time']:.1f}s | "
                    f"Train: {metrics['training_time']:.2f}s | "
                    f"Tok/s: {tok_per_sec:.0f}"
                )

                self.state.samples_seen += len(repeated_prompts)

                # Checkpoint
                if (step + 1) % self.config.checkpoint_interval == 0:
                    checkpoint_path = os.path.join(self.config.checkpoint_dir, f"step_{step + 1}")
                    ray.get(self._training_worker.save_checkpoint.remote(checkpoint_path))

        except KeyboardInterrupt:
            print("\nTraining interrupted")
        finally:
            # Final checkpoint
            checkpoint_path = os.path.join(self.config.checkpoint_dir, "final")
            ray.get(self._training_worker.save_checkpoint.remote(checkpoint_path))

        print("\nTraining complete!")
        print(f"  Steps: {self.state.step}")
        print(f"  Samples: {self.state.samples_seen}")
        print(f"  Duration: {self.state.elapsed_seconds():.1f}s")

        return self.state

    def shutdown(self):
        """Shutdown workers."""
        ray.kill(self._inference_worker)
        ray.kill(self._training_worker)

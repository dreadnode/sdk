"""
RolloutWorker Ray actor for async experience generation.

This module provides the RolloutWorker, a Ray actor that implements the
RolloutEnvironment protocol for continuous experience generation.

Key features:
- vLLM-based fast inference
- Async experience generation
- Weight synchronization via NCCL/Ray ObjectRef
- Support for multi-turn conversations (optional)

Usage:
    import ray
    from dreadnode.core.training.ray.rollout_worker import RolloutWorker

    worker = RolloutWorker.remote(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        reward_fn=my_reward_fn,
    )

    # Generate experiences
    batch = ray.get(worker.generate_batch.remote(
        prompts=["What is 2+2?"],
        num_generations_per_prompt=4,
    ))

    # Update weights from learner
    ray.get(worker.update_weights.remote(state_dict_ref))
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import ray
import torch

from dreadnode.core.training.ray.config import RayGRPOConfig, VLLMConfig
from dreadnode.core.training.ray.experience import Experience, ExperienceBatch


@dataclass
class RolloutConfig:
    """Configuration for rollout worker."""

    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    """Model name or path."""

    tokenizer_name: str | None = None
    """Tokenizer name (defaults to model_name)."""

    # Generation
    max_new_tokens: int = 512
    """Maximum tokens to generate per completion."""

    temperature: float = 0.7
    """Sampling temperature."""

    top_p: float = 0.9
    """Top-p (nucleus) sampling."""

    # GRPO
    num_generations_per_prompt: int = 4
    """Default number of completions per prompt."""

    use_leave_one_out_baseline: bool = True
    """Use leave-one-out baseline for advantages."""

    # vLLM
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    """vLLM configuration."""

    def __post_init__(self) -> None:
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name

    @classmethod
    def from_grpo_config(cls, config: RayGRPOConfig) -> RolloutConfig:
        """Create from RayGRPOConfig."""
        return cls(
            model_name=config.model_name,
            tokenizer_name=config.tokenizer_name,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            num_generations_per_prompt=config.num_generations_per_prompt,
            use_leave_one_out_baseline=config.loss.use_leave_one_out_baseline,
            vllm=config.vllm,
        )


# Type alias for reward functions
RewardFunction = Callable[[list[str], list[str]], list[float]]
AsyncRewardFunction = Callable[[list[str], list[str]], "asyncio.Future[list[float]]"]


@ray.remote
class RolloutWorker:
    """
    Ray actor for async experience generation.

    Implements the RolloutEnvironment protocol as a Ray actor for
    distributed and continuous experience generation.

    The worker:
    1. Manages a vLLM instance for fast inference
    2. Generates completions for batches of prompts
    3. Computes rewards using the provided reward function
    4. Constructs Experience objects with all training data
    5. Computes GRPO advantages with leave-one-out baseline
    6. Supports weight updates for on-policy training

    Attributes:
        config: Rollout configuration
        reward_fn: Function to compute rewards for completions
        worker_id: Unique identifier for this worker
    """

    def __init__(
        self,
        config: RolloutConfig | RayGRPOConfig,
        reward_fn: RewardFunction,
        worker_id: int = 0,
        gpu_ids: list[int] | None = None,
    ) -> None:
        """
        Initialize rollout worker.

        Args:
            config: Configuration (RolloutConfig or RayGRPOConfig)
            reward_fn: Reward function (prompts, completions) -> rewards
            worker_id: Unique worker ID for logging
            gpu_ids: Specific GPU IDs to use
        """
        import os

        # Convert config if needed
        if isinstance(config, RayGRPOConfig):
            self.config = RolloutConfig.from_grpo_config(config)
        else:
            self.config = config

        self.reward_fn = reward_fn
        self.worker_id = worker_id

        # Set CUDA devices if specified
        if gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        # Initialize vLLM
        self._init_vllm()

        # Statistics
        self._stats = {
            "batches_generated": 0,
            "experiences_generated": 0,
            "weight_updates": 0,
        }

    def _init_vllm(self, model_path: str | None = None) -> None:
        """Initialize vLLM engine and tokenizer.

        Args:
            model_path: Optional override for model path (used during weight reload)
        """
        from vllm import LLM

        model = model_path or self.config.model_name

        self.llm = LLM(
            model=model,
            tensor_parallel_size=self.config.vllm.tensor_parallel_size,
            gpu_memory_utilization=self.config.vllm.gpu_memory_utilization,
            max_model_len=self.config.vllm.max_model_len,
            enforce_eager=self.config.vllm.enforce_eager,
            dtype=self.config.vllm.dtype,
            trust_remote_code=self.config.vllm.trust_remote_code,
            enable_prefix_caching=self.config.vllm.enable_prefix_caching,
            enable_chunked_prefill=self.config.vllm.enable_chunked_prefill,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def generate_batch(
        self,
        prompts: list[str],
        num_generations_per_prompt: int | None = None,
        compute_advantages: bool = True,
    ) -> ExperienceBatch:
        """
        Generate an ExperienceBatch for a batch of prompts.

        Args:
            prompts: List of prompt strings
            num_generations_per_prompt: Completions per prompt (defaults to config)
            compute_advantages: Whether to compute GRPO advantages

        Returns:
            ExperienceBatch with all experiences and computed advantages
        """
        from vllm import SamplingParams

        num_gen = num_generations_per_prompt or self.config.num_generations_per_prompt

        # Expand prompts for multiple generations
        expanded_prompts = []
        group_ids = []
        for i, prompt in enumerate(prompts):
            for _ in range(num_gen):
                expanded_prompts.append(prompt)
                group_ids.append(i)

        # Generate with vLLM
        sampling_params = SamplingParams(
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            logprobs=1,  # Need logprobs for training
        )

        outputs = self.llm.generate(expanded_prompts, sampling_params)

        # Build experiences
        experiences = []
        all_completions = []

        for i, output in enumerate(outputs):
            prompt = expanded_prompts[i]
            prompt_ids = torch.tensor(output.prompt_token_ids, dtype=torch.long)

            completion = output.outputs[0]
            completion_text = completion.text
            completion_ids = torch.tensor(completion.token_ids, dtype=torch.long)

            # Extract logprobs
            logprobs = self._extract_logprobs(completion)

            all_completions.append(completion_text)

            experiences.append(
                Experience(
                    prompt=prompt,
                    completion=completion_text,
                    prompt_ids=prompt_ids,
                    completion_ids=completion_ids,
                    logprobs=logprobs,
                    group_id=group_ids[i],
                    reward=0.0,  # Will be set after reward computation
                )
            )

        # Compute rewards
        all_prompts = [exp.prompt for exp in experiences]
        rewards = self.reward_fn(all_prompts, all_completions)

        for exp, reward in zip(experiences, rewards):
            exp.reward = reward

        # Create batch
        batch = ExperienceBatch(experiences=experiences)

        # Compute advantages
        if compute_advantages:
            batch.compute_advantages(
                use_leave_one_out=self.config.use_leave_one_out_baseline
            )

        # Update stats
        self._stats["batches_generated"] += 1
        self._stats["experiences_generated"] += len(experiences)

        return batch

    def generate_experiences(
        self,
        prompts: list[str],
        num_generations_per_prompt: int | None = None,
    ) -> list[Experience]:
        """
        Generate experiences for a batch of prompts.

        Args:
            prompts: List of prompt strings
            num_generations_per_prompt: Completions per prompt

        Returns:
            List of Experience objects
        """
        batch = self.generate_batch(
            prompts,
            num_generations_per_prompt,
            compute_advantages=True,
        )
        return batch.experiences

    def update_weights(
        self,
        state_dict_ref: ray.ObjectRef | dict | None = None,
        checkpoint_path: str | None = None,
    ) -> bool:
        """
        Update vLLM weights from training model.

        Supports multiple weight sync strategies:
        1. Direct state_dict loading (fast, may not work on all vLLM versions)
        2. Checkpoint path reload (reliable, works on all versions)

        Args:
            state_dict_ref: Ray ObjectRef to state dict, or state dict directly
            checkpoint_path: Path to saved checkpoint (alternative to state_dict)

        Returns:
            True if update succeeded
        """
        import time

        t0 = time.time()

        # Strategy 1: Checkpoint path (most reliable for vLLM v1)
        if checkpoint_path is not None:
            success = self._update_via_checkpoint(checkpoint_path)
            if success:
                print(f"[Worker {self.worker_id}] Checkpoint reload took {time.time() - t0:.2f}s")
                return True

        # Strategy 2: Direct state dict (for vLLM v0 or compatible versions)
        if state_dict_ref is not None:
            success = self._update_via_state_dict(state_dict_ref)
            if success:
                print(f"[Worker {self.worker_id}] State dict sync took {time.time() - t0:.2f}s")
                return True

        print(f"[Worker {self.worker_id}] Weight update failed - no valid method")
        return False

    def _update_via_checkpoint(self, checkpoint_path: str) -> bool:
        """Update weights by reloading from checkpoint."""
        import gc

        try:
            # Release current vLLM
            del self.llm
            gc.collect()
            torch.cuda.empty_cache()

            # Reload from new checkpoint (with prefix caching preserved)
            self._init_vllm(model_path=checkpoint_path)

            self._stats["weight_updates"] += 1
            return True

        except Exception as e:
            print(f"[Worker {self.worker_id}] Checkpoint reload failed: {e}")
            # Try to reinitialize with original model
            try:
                self._init_vllm()
            except Exception:
                pass
            return False

    def _update_via_state_dict(self, state_dict_ref: ray.ObjectRef | dict) -> bool:
        """Update weights via direct state dict loading (vLLM v0 style)."""
        try:
            # Get state dict if it's a reference
            if isinstance(state_dict_ref, ray.ObjectRef):
                state_dict = ray.get(state_dict_ref)
            else:
                state_dict = state_dict_ref

            # Try vLLM v0 style direct access
            try:
                model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                model.load_weights(weights=list(state_dict.items()))
                self._stats["weight_updates"] += 1
                return True
            except AttributeError:
                pass

            # Try v1 collective_rpc (may not be implemented)
            try:
                weights = [(k, v.cpu()) for k, v in state_dict.items()]
                self.llm.collective_rpc("load_weights", args=(weights,))
                self._stats["weight_updates"] += 1
                return True
            except (NotImplementedError, Exception):
                pass

            return False

        except Exception as e:
            print(f"[Worker {self.worker_id}] State dict update failed: {e}")
            return False

    def get_stats(self) -> dict[str, int]:
        """Get worker statistics."""
        return self._stats.copy()

    def get_tokenizer(self) -> Any:
        """Get the tokenizer."""
        return self.tokenizer

    def shutdown(self) -> None:
        """Shutdown vLLM engine and release resources."""
        del self.llm
        torch.cuda.empty_cache()

    def _extract_logprobs(self, completion: Any) -> torch.Tensor:
        """Extract log probabilities from vLLM completion output."""
        if completion.logprobs is None:
            return torch.zeros(len(completion.token_ids))

        lps = []
        for lp_dict in completion.logprobs:
            if lp_dict:
                # Get the logprob of the sampled token
                # In newer vLLM versions, logprobs is a dict[token_id, Logprob]
                token_id = list(lp_dict.keys())[0]
                lp_obj = lp_dict[token_id]
                # Handle both Logprob object and raw float
                if hasattr(lp_obj, "logprob"):
                    lps.append(lp_obj.logprob)
                else:
                    lps.append(float(lp_obj))
            else:
                lps.append(0.0)

        return torch.tensor(lps, dtype=torch.float32)


class RolloutWorkerPool:
    """
    Pool of RolloutWorker actors for distributed generation.

    Manages multiple workers and distributes generation requests.
    """

    def __init__(
        self,
        config: RayGRPOConfig,
        reward_fn: RewardFunction,
        num_workers: int = 1,
        packed: bool = False,
    ) -> None:
        """
        Initialize worker pool.

        Args:
            config: GRPO configuration
            reward_fn: Reward function for all workers
            num_workers: Number of workers to create
            packed: If True, multiple workers share the same GPU(s)
        """
        self.config = config
        self.num_workers = num_workers
        self.packed = packed
        self.workers: list[ray.ActorHandle] = []

        # Determine GPU allocation
        gpus_per_worker = config.vllm.tensor_parallel_size

        if packed:
            # Packed mode: all workers share GPU 0 (or first N GPUs for TP)
            # Each worker gets fractional GPU allocation
            fractional_gpus = gpus_per_worker / num_workers
            gpu_ids = list(range(gpus_per_worker))  # All workers use same GPUs

            for i in range(num_workers):
                worker = RolloutWorker.options(
                    num_gpus=fractional_gpus,
                    name=f"rollout_worker_{i}",
                ).remote(
                    config=config,
                    reward_fn=reward_fn,
                    worker_id=i,
                    gpu_ids=gpu_ids,
                )
                self.workers.append(worker)
        else:
            # Normal mode: each worker gets dedicated GPU(s)
            for i in range(num_workers):
                gpu_start = i * gpus_per_worker
                gpu_ids = list(range(gpu_start, gpu_start + gpus_per_worker))

                worker = RolloutWorker.options(
                    num_gpus=gpus_per_worker,
                    name=f"rollout_worker_{i}",
                ).remote(
                    config=config,
                    reward_fn=reward_fn,
                    worker_id=i,
                    gpu_ids=gpu_ids,
                )
                self.workers.append(worker)

    def generate_batch(
        self,
        prompts: list[str],
        num_generations_per_prompt: int | None = None,
    ) -> ExperienceBatch:
        """
        Generate experiences by distributing prompts across workers.

        Args:
            prompts: List of prompts
            num_generations_per_prompt: Completions per prompt

        Returns:
            Merged ExperienceBatch from all workers
        """
        # Split prompts across workers
        chunks = self._split_prompts(prompts)

        # Launch parallel generation
        futures = []
        for worker, chunk in zip(self.workers, chunks):
            if chunk:
                future = worker.generate_batch.remote(
                    prompts=chunk,
                    num_generations_per_prompt=num_generations_per_prompt,
                )
                futures.append(future)

        # Gather and merge results
        batches = ray.get(futures)
        return ExperienceBatch.merge(batches)

    def update_weights(
        self,
        state_dict: dict | None = None,
        checkpoint_path: str | None = None,
    ) -> bool:
        """
        Update weights on all workers.

        Supports two strategies:
        1. checkpoint_path: Save model to disk, workers reload (reliable)
        2. state_dict: Direct state dict transfer via Ray (faster, may not work)

        Args:
            state_dict: Model state dict (optional)
            checkpoint_path: Path to saved checkpoint (optional, preferred)

        Returns:
            True if all updates succeeded
        """
        # Prefer checkpoint path (more reliable for vLLM v1)
        if checkpoint_path is not None:
            futures = [
                w.update_weights.remote(checkpoint_path=checkpoint_path)
                for w in self.workers
            ]
            results = ray.get(futures)
            return all(results)

        # Fall back to state dict
        if state_dict is not None:
            state_dict_ref = ray.put(state_dict)
            futures = [
                w.update_weights.remote(state_dict_ref=state_dict_ref)
                for w in self.workers
            ]
            results = ray.get(futures)
        return all(results)

    def get_stats(self) -> dict[str, Any]:
        """Get combined statistics from all workers."""
        futures = [w.get_stats.remote() for w in self.workers]
        all_stats = ray.get(futures)

        combined = {
            "batches_generated": 0,
            "experiences_generated": 0,
            "weight_updates": 0,
            "per_worker": all_stats,
        }

        for stats in all_stats:
            combined["batches_generated"] += stats["batches_generated"]
            combined["experiences_generated"] += stats["experiences_generated"]
            combined["weight_updates"] = max(
                combined["weight_updates"], stats["weight_updates"]
            )

        return combined

    def shutdown(self) -> None:
        """Shutdown all workers."""
        futures = [w.shutdown.remote() for w in self.workers]
        ray.get(futures)
        for worker in self.workers:
            ray.kill(worker)
        self.workers = []

    def _split_prompts(self, prompts: list[str]) -> list[list[str]]:
        """Split prompts evenly across workers."""
        n = len(prompts)
        k = len(self.workers)
        chunk_size = (n + k - 1) // k

        chunks = []
        for i in range(k):
            start = i * chunk_size
            end = min(start + chunk_size, n)
            chunks.append(prompts[start:end])

        return chunks


# Convenience function for creating a single worker
def create_rollout_worker(
    config: RayGRPOConfig,
    reward_fn: RewardFunction,
    worker_id: int = 0,
) -> ray.ActorHandle:
    """
    Create a single RolloutWorker actor.

    Args:
        config: GRPO configuration
        reward_fn: Reward function
        worker_id: Worker ID

    Returns:
        Ray actor handle to RolloutWorker
    """
    return RolloutWorker.options(
        num_gpus=config.vllm.tensor_parallel_size,
        name=f"rollout_worker_{worker_id}",
    ).remote(
        config=config,
        reward_fn=reward_fn,
        worker_id=worker_id,
    )

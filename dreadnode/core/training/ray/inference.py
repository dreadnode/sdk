"""
vLLM-based inference pool for distributed generation.

Uses Ray actors to manage vLLM instances across multiple GPUs.
"""

from dataclasses import dataclass
from typing import Any

import ray
import torch
from transformers import AutoTokenizer

from dreadnode.core.training.ray.config import RayGRPOConfig, VLLMConfig


@dataclass
class GenerationOutput:
    """Output from generation."""

    prompt_ids: torch.Tensor
    """Input prompt token IDs [batch_size, prompt_len]."""

    completion_ids: torch.Tensor
    """Generated completion token IDs [batch_size, completion_len]."""

    full_ids: torch.Tensor
    """Full sequence (prompt + completion) [batch_size, seq_len]."""

    logprobs: torch.Tensor
    """Log probabilities for completions [batch_size, completion_len]."""

    completion_mask: torch.Tensor
    """Mask indicating completion tokens [batch_size, seq_len]."""


@ray.remote(num_gpus=0)  # Don't reserve GPUs at actor level, vLLM manages its own
class VLLMInferenceActor:
    """
    Ray actor wrapping a vLLM engine for fast LLM inference.

    Each actor manages a single vLLM instance that can use tensor
    parallelism across multiple GPUs.
    """

    def __init__(
        self,
        model_name: str,
        config: VLLMConfig,
        gpu_ids: list[int] | None = None,
    ):
        """
        Initialize vLLM engine.

        Args:
            model_name: HuggingFace model name or path.
            config: vLLM configuration.
            gpu_ids: GPU IDs to use (for tensor parallelism).
        """
        import os

        # Set CUDA visible devices if specified
        if gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        from vllm import LLM

        self.model_name = model_name
        self.config = config

        # Initialize vLLM engine
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            enforce_eager=config.enforce_eager,
            dtype=config.dtype,
            trust_remote_code=config.trust_remote_code,
        )

        # Get tokenizer from vLLM
        self.tokenizer = self.llm.get_tokenizer()

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_logprobs: bool = True,
    ) -> dict[str, Any]:
        """
        Generate completions for prompts.

        Args:
            prompts: List of prompt strings.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            return_logprobs: Whether to return log probabilities.

        Returns:
            Dictionary with generated outputs.
        """
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=1 if return_logprobs else None,
        )

        outputs = self.llm.generate(prompts, sampling_params)

        # Process outputs
        completions = []
        all_logprobs = []
        prompt_token_ids = []
        completion_token_ids = []

        for output in outputs:
            prompt_token_ids.append(list(output.prompt_token_ids))
            completion = output.outputs[0]
            completions.append(completion.text)
            completion_token_ids.append(list(completion.token_ids))

            if return_logprobs and completion.logprobs:
                # Extract log probabilities for generated tokens
                lps = []
                for lp_dict in completion.logprobs:
                    if lp_dict:
                        # Get the logprob of the sampled token
                        token_id = list(lp_dict.keys())[0]
                        lps.append(lp_dict[token_id].logprob)
                    else:
                        lps.append(0.0)
                all_logprobs.append(lps)
            else:
                all_logprobs.append([0.0] * len(completion.token_ids))

        return {
            "completions": completions,
            "prompt_token_ids": prompt_token_ids,
            "completion_token_ids": completion_token_ids,
            "logprobs": all_logprobs,
        }

    def get_tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer."""
        return self.tokenizer

    def update_weights(self, state_dict: dict[str, torch.Tensor]) -> bool:
        """
        Update model weights from state dict.

        Args:
            state_dict: New model weights.

        Returns:
            True if successful.
        """
        # vLLM weight update using the model runner
        try:
            # Get the model from vLLM
            model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model

            # Load new weights
            model.load_state_dict(state_dict, strict=False)

            return True
        except Exception as e:
            print(f"Failed to update weights: {e}")
            return False

    def shutdown(self):
        """Shutdown the vLLM engine."""
        del self.llm
        torch.cuda.empty_cache()


class VLLMInferencePool:
    """
    Pool of vLLM inference actors for distributed generation.

    Manages multiple vLLM instances and distributes generation
    requests across them.
    """

    def __init__(
        self,
        config: RayGRPOConfig,
        num_replicas: int = 1,
    ):
        """
        Initialize inference pool.

        Args:
            config: GRPO configuration.
            num_replicas: Number of vLLM replicas.
        """
        self.config = config
        self.num_replicas = num_replicas
        self.actors: list[ray.ObjectRef] = []

        # Determine GPU allocation
        num_gpus = ray.cluster_resources().get("GPU", 0)
        gpus_per_replica = config.vllm.tensor_parallel_size

        if num_gpus < num_replicas * gpus_per_replica:
            raise ValueError(
                f"Not enough GPUs. Need {num_replicas * gpus_per_replica}, have {num_gpus}"
            )

        # Create actors
        for i in range(num_replicas):
            gpu_start = i * gpus_per_replica
            gpu_ids = list(range(gpu_start, gpu_start + gpus_per_replica))

            actor = VLLMInferenceActor.options(
                num_gpus=gpus_per_replica,
                name=f"vllm_inference_{i}",
            ).remote(
                model_name=config.model_name,
                config=config.vllm,
                gpu_ids=gpu_ids,
            )
            self.actors.append(actor)

        print(f"Created {num_replicas} vLLM inference actors")

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> GenerationOutput:
        """
        Generate completions for prompts.

        Distributes prompts across actors for parallel generation.

        Args:
            prompts: List of prompt strings.
            max_new_tokens: Maximum tokens (defaults to config).
            temperature: Sampling temperature (defaults to config).
            top_p: Top-p sampling (defaults to config).

        Returns:
            GenerationOutput with all results.
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p

        # Split prompts across actors
        chunks = self._split_prompts(prompts)

        # Launch parallel generation
        futures = []
        for actor, chunk in zip(self.actors, chunks):
            if chunk:
                future = actor.generate.remote(
                    prompts=chunk,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    return_logprobs=True,
                )
                futures.append(future)

        # Gather results
        results = ray.get(futures)

        # Merge results
        return self._merge_results(results, prompts)

    def update_weights(self, state_dict: dict[str, torch.Tensor]) -> bool:
        """
        Update weights on all inference actors.

        Args:
            state_dict: New model weights.

        Returns:
            True if all updates succeeded.
        """
        # Broadcast weights to all actors
        futures = [actor.update_weights.remote(state_dict) for actor in self.actors]
        results = ray.get(futures)
        return all(results)

    def shutdown(self):
        """Shutdown all actors."""
        futures = [actor.shutdown.remote() for actor in self.actors]
        ray.get(futures)
        for actor in self.actors:
            ray.kill(actor)
        self.actors = []

    def _split_prompts(self, prompts: list[str]) -> list[list[str]]:
        """Split prompts evenly across actors."""
        n = len(prompts)
        k = len(self.actors)
        chunk_size = (n + k - 1) // k

        chunks = []
        for i in range(k):
            start = i * chunk_size
            end = min(start + chunk_size, n)
            chunks.append(prompts[start:end])

        return chunks

    def _merge_results(self, results: list[dict[str, Any]], prompts: list[str]) -> GenerationOutput:
        """Merge results from multiple actors."""
        all_completions = []
        all_prompt_ids = []
        all_completion_ids = []
        all_logprobs = []

        for result in results:
            all_completions.extend(result["completions"])
            all_prompt_ids.extend(result["prompt_token_ids"])
            all_completion_ids.extend(result["completion_token_ids"])
            all_logprobs.extend(result["logprobs"])

        # Pad sequences for batching
        max_prompt_len = max(len(ids) for ids in all_prompt_ids)
        max_completion_len = max(len(ids) for ids in all_completion_ids)
        max_seq_len = max_prompt_len + max_completion_len

        batch_size = len(prompts)
        prompt_ids = torch.zeros(batch_size, max_prompt_len, dtype=torch.long)
        completion_ids = torch.zeros(batch_size, max_completion_len, dtype=torch.long)
        full_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        logprobs = torch.zeros(batch_size, max_completion_len)
        completion_mask = torch.zeros(batch_size, max_seq_len)

        for i in range(batch_size):
            p_len = len(all_prompt_ids[i])
            c_len = len(all_completion_ids[i])

            prompt_ids[i, :p_len] = torch.tensor(all_prompt_ids[i])
            completion_ids[i, :c_len] = torch.tensor(all_completion_ids[i])
            full_ids[i, :p_len] = torch.tensor(all_prompt_ids[i])
            full_ids[i, p_len : p_len + c_len] = torch.tensor(all_completion_ids[i])
            logprobs[i, :c_len] = torch.tensor(all_logprobs[i])
            completion_mask[i, p_len : p_len + c_len] = 1.0

        return GenerationOutput(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            full_ids=full_ids,
            logprobs=logprobs,
            completion_mask=completion_mask,
        )

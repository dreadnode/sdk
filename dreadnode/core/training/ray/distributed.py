"""
Ray Train integration for distributed GRPO/SFT training.

This module provides distributed training using Ray Train with FSDP2,
enabling multi-node training with fault tolerance and checkpointing.

Key features:
- Ray Train integration for distributed orchestration
- FSDP2 for efficient model sharding
- Fault-tolerant training with automatic recovery
- Distributed checkpointing
- Hyperparameter optimization with Ray Tune

Usage:
    from dreadnode.core.training.ray.distributed import train_grpo_distributed

    result = train_grpo_distributed(
        config=grpo_config,
        prompts=training_prompts,
        reward_fn=my_reward_fn,
        num_workers=4,
    )

    # Access trained model
    checkpoint = result.checkpoint
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

# Enable Ray Train v2 API
os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

from dreadnode.core.training.ray.config import RayGRPOConfig
from dreadnode.core.training.ray.experience import ExperienceBatch
from dreadnode.core.training.ray.fsdp2_learner import FSDP2Config, FSDP2Learner
from dreadnode.core.training.ray.rollout_worker import RewardFunction, RolloutConfig


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    # Workers
    num_workers: int = 1
    """Number of training workers (GPUs)."""

    use_gpu: bool = True
    """Use GPUs for training."""

    resources_per_worker: dict[str, float] = field(default_factory=lambda: {"GPU": 1})
    """Resources per worker."""

    # Checkpointing
    checkpoint_interval: int = 100
    """Steps between checkpoints."""

    num_checkpoints_to_keep: int = 3
    """Maximum checkpoints to keep."""

    # Fault tolerance
    max_failures: int = 3
    """Maximum worker failures before abort."""

    # Storage
    storage_path: str | None = None
    """Path for checkpoints and logs."""


def train_grpo_distributed(
    config: RayGRPOConfig,
    prompts: Sequence[str],
    reward_fn: RewardFunction,
    distributed_config: DistributedConfig | None = None,
    fsdp_config: FSDP2Config | None = None,
    num_workers: int | None = None,
) -> Any:  # ray.train.Result
    """
    Distributed GRPO training with Ray Train + FSDP2.

    This function launches a distributed training job using Ray Train,
    with FSDP2 for model sharding across workers. Each worker:
    1. Initializes FSDP2Learner with proper rank/world_size
    2. Runs local experience generation
    3. Executes training steps
    4. Reports metrics and saves checkpoints

    Args:
        config: GRPO configuration
        prompts: Training prompts
        reward_fn: Reward function
        distributed_config: Distributed training configuration
        fsdp_config: FSDP2 configuration
        num_workers: Number of workers (overrides distributed_config)

    Returns:
        Ray Train Result with checkpoint and metrics
    """
    from ray import train
    from ray.train import CheckpointConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    # Merge configs
    distributed_config = distributed_config or DistributedConfig()
    if num_workers is not None:
        distributed_config.num_workers = num_workers

    fsdp_config = fsdp_config or FSDP2Config()

    # Convert prompts to list for serialization
    prompts_list = list(prompts)

    def train_loop_per_worker() -> None:
        """Training loop executed on each worker."""
        import ray.train as train_module

        # Get worker context
        context = train_module.get_context()
        world_size = context.get_world_size()
        rank = context.get_local_rank()

        # Initialize distributed (Ray Train handles this)
        torch.cuda.set_device(rank)

        # Initialize learner with FSDP2
        learner = FSDP2Learner(
            model_name=config.model_name,
            config=config,
            fsdp_config=fsdp_config,
            world_size=world_size,
            rank=rank,
        )

        # Create scheduler
        learner.create_scheduler(total_steps=config.max_steps)

        # Shard prompts across workers
        worker_prompts = prompts_list[rank::world_size]

        # Local experience generation (simplified for distributed)
        # In production, you'd use RolloutWorkerPool on separate GPUs
        from vllm import LLM, SamplingParams

        # Use colocated generation if on same GPU
        # This is a simplified approach - for large scale, use separate inference nodes
        local_generation = _create_local_generator(config, rank)

        for step in range(config.max_steps):
            # Get prompts for this step
            prompt_idx = (step * config.num_prompts_per_step) % len(worker_prompts)
            step_prompts = worker_prompts[
                prompt_idx : prompt_idx + config.num_prompts_per_step
            ]

            if not step_prompts:
                step_prompts = worker_prompts[:config.num_prompts_per_step]

            # Generate experiences locally
            batch = local_generation(step_prompts, reward_fn, config)

            # Training step
            metrics = learner.train_step(batch)

            # Report metrics
            train_module.report(
                {
                    "loss": metrics.get("loss", 0.0),
                    "reward_mean": metrics.get("reward_mean", 0.0),
                    "step": step,
                    **{k: v for k, v in metrics.items() if k not in ["loss", "reward_mean", "step"]},
                }
            )

            # Checkpoint
            if step > 0 and step % distributed_config.checkpoint_interval == 0:
                with tempfile.TemporaryDirectory() as tmpdir:
                    learner.save_checkpoint(tmpdir)
                    train_module.report(
                        {},
                        checkpoint=train_module.Checkpoint.from_directory(tmpdir),
                    )

        # Final checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            learner.save_checkpoint(tmpdir)
            train_module.report(
                {"final": True},
                checkpoint=train_module.Checkpoint.from_directory(tmpdir),
            )

    # Configure trainer
    scaling_config = ScalingConfig(
        num_workers=distributed_config.num_workers,
        use_gpu=distributed_config.use_gpu,
        resources_per_worker=distributed_config.resources_per_worker,
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=distributed_config.num_checkpoints_to_keep,
        ),
        storage_path=distributed_config.storage_path,
        failure_config=train.FailureConfig(
            max_failures=distributed_config.max_failures,
        ),
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    return trainer.fit()


def _create_local_generator(
    config: RayGRPOConfig,
    rank: int,
) -> Callable:
    """
    Create local generation function for colocated training.

    This is a simplified approach for distributed training where
    generation happens on the same GPU as training. For large-scale
    training, use separate inference nodes with RolloutWorkerPool.
    """
    # Lazy initialization of vLLM
    _llm = None
    _tokenizer = None

    def generate(
        prompts: list[str],
        reward_fn: RewardFunction,
        config: RayGRPOConfig,
    ) -> ExperienceBatch:
        nonlocal _llm, _tokenizer

        from vllm import LLM, SamplingParams
        from dreadnode.core.training.ray.experience import Experience, ExperienceBatch

        # Initialize vLLM on first call
        if _llm is None:
            _llm = LLM(
                model=config.model_name,
                tensor_parallel_size=1,  # Single GPU per worker
                gpu_memory_utilization=config.vllm.gpu_memory_utilization * 0.5,  # Share with training
                enforce_eager=True,  # For memory efficiency with training
                trust_remote_code=config.vllm.trust_remote_code,
            )
            _tokenizer = _llm.get_tokenizer()

        # Expand prompts
        expanded_prompts = []
        group_ids = []
        for i, prompt in enumerate(prompts):
            for _ in range(config.num_generations_per_prompt):
                expanded_prompts.append(prompt)
                group_ids.append(i)

        # Generate
        sampling_params = SamplingParams(
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            logprobs=1,
        )

        outputs = _llm.generate(expanded_prompts, sampling_params)

        # Build experiences
        experiences = []
        completions = []

        for i, output in enumerate(outputs):
            prompt = expanded_prompts[i]
            prompt_ids = torch.tensor(output.prompt_token_ids, dtype=torch.long)

            completion = output.outputs[0]
            completion_text = completion.text
            completion_ids = torch.tensor(completion.token_ids, dtype=torch.long)

            # Extract logprobs
            logprobs = _extract_logprobs(completion)

            completions.append(completion_text)

            experiences.append(
                Experience(
                    prompt=prompt,
                    completion=completion_text,
                    prompt_ids=prompt_ids,
                    completion_ids=completion_ids,
                    logprobs=logprobs,
                    group_id=group_ids[i],
                    reward=0.0,
                )
            )

        # Compute rewards
        all_prompts = [exp.prompt for exp in experiences]
        rewards = reward_fn(all_prompts, completions)

        for exp, reward in zip(experiences, rewards):
            exp.reward = reward

        # Create and process batch
        batch = ExperienceBatch(experiences=experiences)
        batch.compute_advantages(use_leave_one_out=config.loss.use_leave_one_out_baseline)

        return batch

    return generate


def _extract_logprobs(completion: Any) -> torch.Tensor:
    """Extract log probabilities from vLLM completion output."""
    if completion.logprobs is None:
        return torch.zeros(len(completion.token_ids))

    lps = []
    for lp_dict in completion.logprobs:
        if lp_dict:
            token_id = list(lp_dict.keys())[0]
            lp_obj = lp_dict[token_id]
            if hasattr(lp_obj, "logprob"):
                lps.append(lp_obj.logprob)
            else:
                lps.append(float(lp_obj))
        else:
            lps.append(0.0)

    return torch.tensor(lps, dtype=torch.float32)


def train_sft_distributed(
    config: RayGRPOConfig,
    dataset: Any,  # HuggingFace Dataset or similar
    distributed_config: DistributedConfig | None = None,
    fsdp_config: FSDP2Config | None = None,
    num_workers: int | None = None,
) -> Any:  # ray.train.Result
    """
    Distributed SFT training with Ray Train + FSDP2.

    Args:
        config: Training configuration
        dataset: Training dataset with 'input' and 'output' fields
        distributed_config: Distributed training configuration
        fsdp_config: FSDP2 configuration
        num_workers: Number of workers

    Returns:
        Ray Train Result
    """
    from ray import train
    from ray.train import CheckpointConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    distributed_config = distributed_config or DistributedConfig()
    if num_workers is not None:
        distributed_config.num_workers = num_workers

    fsdp_config = fsdp_config or FSDP2Config()

    def train_loop_per_worker() -> None:
        """SFT training loop per worker."""
        import ray.train as train_module
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from torch.utils.data import DataLoader, DistributedSampler

        context = train_module.get_context()
        world_size = context.get_world_size()
        rank = context.get_local_rank()

        torch.cuda.set_device(rank)

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name or config.model_name,
            trust_remote_code=config.vllm.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=fsdp_config.param_dtype,
            attn_implementation="flash_attention_2",
            trust_remote_code=config.vllm.trust_remote_code,
        ).to(f"cuda:{rank}")

        if fsdp_config.use_activation_checkpointing:
            model.gradient_checkpointing_enable()

        # Apply FSDP2 if distributed
        if world_size > 1:
            from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
            from torch.distributed.device_mesh import init_device_mesh

            device_mesh = init_device_mesh("cuda", (world_size,))
            mp_policy = MixedPrecisionPolicy(
                param_dtype=fsdp_config.param_dtype,
                reduce_dtype=fsdp_config.reduce_dtype,
            )

            if hasattr(model, "model") and hasattr(model.model, "layers"):
                for layer in model.model.layers:
                    fully_shard(layer, mesh=device_mesh, mp_policy=mp_policy)

            fully_shard(model, mesh=device_mesh, mp_policy=mp_policy)

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Create dataloader
        # Assume dataset has 'text' field or 'input'/'output' fields
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=config.num_prompts_per_step,
            sampler=sampler,
            collate_fn=lambda x: _collate_sft(x, tokenizer),
        )

        # Training loop
        step = 0
        for epoch in range(config.max_epochs):
            sampler.set_epoch(epoch)

            for batch in dataloader:
                if step >= config.max_steps:
                    break

                input_ids = batch["input_ids"].to(f"cuda:{rank}")
                attention_mask = batch["attention_mask"].to(f"cuda:{rank}")
                labels = batch["labels"].to(f"cuda:{rank}")

                # Forward
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                # Backward
                loss.backward()

                if config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config.max_grad_norm,
                    )

                optimizer.step()
                optimizer.zero_grad()

                # Report
                train_module.report({"loss": loss.item(), "step": step})

                # Checkpoint
                if step > 0 and step % distributed_config.checkpoint_interval == 0:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        model.save_pretrained(tmpdir)
                        tokenizer.save_pretrained(tmpdir)
                        train_module.report(
                            {},
                            checkpoint=train_module.Checkpoint.from_directory(tmpdir),
                        )

                step += 1

    scaling_config = ScalingConfig(
        num_workers=distributed_config.num_workers,
        use_gpu=distributed_config.use_gpu,
        resources_per_worker=distributed_config.resources_per_worker,
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=distributed_config.num_checkpoints_to_keep,
        ),
        storage_path=distributed_config.storage_path,
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    return trainer.fit()


def _collate_sft(
    examples: list[dict],
    tokenizer: Any,
    max_length: int = 2048,
) -> dict[str, torch.Tensor]:
    """Collate function for SFT training."""
    texts = []
    for ex in examples:
        if "text" in ex:
            texts.append(ex["text"])
        elif "input" in ex and "output" in ex:
            texts.append(f"{ex['input']}{ex['output']}")
        else:
            raise ValueError("Dataset must have 'text' or 'input'/'output' fields")

    # Tokenize
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Labels = input_ids (causal LM)
    labels = encodings["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
    }


# Convenience function for hyperparameter tuning
def tune_grpo(
    config: RayGRPOConfig,
    prompts: Sequence[str],
    reward_fn: RewardFunction,
    param_space: dict[str, Any],
    num_samples: int = 10,
    num_workers: int = 1,
) -> Any:  # ray.tune.ResultGrid
    """
    Hyperparameter optimization for GRPO training using Ray Tune.

    Args:
        config: Base GRPO configuration
        prompts: Training prompts
        reward_fn: Reward function
        param_space: Parameter search space (Ray Tune format)
        num_samples: Number of trials
        num_workers: Workers per trial

    Returns:
        Ray Tune ResultGrid with best hyperparameters
    """
    from ray import tune

    def trainable(tune_config: dict) -> None:
        """Trainable function for Ray Tune."""
        # Merge tune config into base config
        merged_config = RayGRPOConfig(
            model_name=config.model_name,
            learning_rate=tune_config.get("learning_rate", config.learning_rate),
            temperature=tune_config.get("temperature", config.temperature),
            num_generations_per_prompt=tune_config.get(
                "num_generations_per_prompt",
                config.num_generations_per_prompt,
            ),
            max_steps=config.max_steps // 10,  # Shorter for tuning
        )

        # Run training
        result = train_grpo_distributed(
            config=merged_config,
            prompts=prompts,
            reward_fn=reward_fn,
            num_workers=num_workers,
        )

        # Report final reward
        tune.report(
            reward=result.metrics.get("reward_mean", 0.0),
            loss=result.metrics.get("loss", float("inf")),
        )

    # Run tuning
    tuner = tune.Tuner(
        trainable,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            metric="reward",
            mode="max",
        ),
    )

    return tuner.fit()

"""
GRPO Trainer for DN SDK agents.

Wrapper around NeMo RL's GRPO algorithm for training agents.

This module provides a complete integration with NeMo RL for
Group Relative Policy Optimization (GRPO) training of agent models.

Key Features:
- Full NeMo RL Policy and Generation integration
- Advantage computation with leave-one-out baseline
- Reward scaling and DAPO penalty support
- vLLM-based inference with weight synchronization
- Distributed training support via Ray
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

import torch

from dreadnode.core.training.rewards.aggregator import RewardAggregator
from dreadnode.core.training.trainers.base import (
    BaseTrainer,
    TrainingCallback,
    TrainingConfig,
    TrainingState,
)

if TYPE_CHECKING:
    from dreadnode.core.agents import Agent


class RewardScalingConfig(TypedDict, total=False):
    """Configure linear reward scaling with clamping (matches NeMo RL)."""

    enabled: bool
    source_min: float
    source_max: float
    target_min: float
    target_max: float


class RewardShapingConfig(TypedDict, total=False):
    """Configure DAPO-style reward shaping (matches NeMo RL)."""

    enabled: bool
    overlong_buffer_length: int
    overlong_buffer_penalty: float
    max_response_length: int


class GRPOConfig(TrainingConfig):
    """Configuration for GRPO training (matches NeMo RL's GRPOConfig)."""

    # GRPO-specific parameters
    num_prompts_per_step: NotRequired[int]
    """Number of prompts per training step."""

    num_generations_per_prompt: NotRequired[int]
    """Number of generations per prompt (for advantage estimation)."""

    max_num_epochs: NotRequired[int]
    """Maximum number of epochs."""

    max_num_steps: NotRequired[int]
    """Maximum number of training steps."""

    max_rollout_turns: NotRequired[int]
    """Maximum turns per rollout."""

    # Policy gradient settings
    kl_penalty_coeff: NotRequired[float]
    """KL penalty coefficient."""

    entropy_bonus: NotRequired[float]
    """Entropy bonus coefficient."""

    clip_ratio: NotRequired[float]
    """PPO-style clip ratio for importance sampling."""

    use_importance_sampling_correction: NotRequired[bool]
    """Enable importance sampling correction."""

    # Advantage computation
    normalize_rewards: NotRequired[bool]
    """Whether to normalize advantages."""

    use_leave_one_out_baseline: NotRequired[bool]
    """Use leave-one-out baseline for advantage computation."""

    # Dynamic sampling (DAPO)
    use_dynamic_sampling: NotRequired[bool]
    """Enable dynamic sampling (discard zero-std prompts)."""

    dynamic_sampling_max_gen_batches: NotRequired[int]
    """Maximum batches for dynamic sampling."""

    batch_multiplier: NotRequired[float]
    """Batch size multiplier for dynamic sampling."""

    overlong_filtering: NotRequired[bool]
    """Filter overlong sequences from training."""

    # Reward shaping (matches NeMo RL)
    reward_scaling: NotRequired[RewardScalingConfig]
    """Reward scaling configuration."""

    reward_shaping: NotRequired[RewardShapingConfig]
    """DAPO-style reward shaping configuration."""

    # Validation
    val_period: NotRequired[int]
    """Steps between validation."""

    val_batch_size: NotRequired[int]
    """Validation batch size."""

    val_at_start: NotRequired[bool]
    """Run validation at start."""

    max_val_samples: NotRequired[int]
    """Maximum validation samples."""

    # vLLM generation settings
    generation_backend: NotRequired[str]
    """Generation backend: 'vllm' or 'megatron'."""

    colocated_inference: NotRequired[bool]
    """Colocate inference with training (share GPU memory)."""

    # Legacy support
    reward_scaling_enabled: NotRequired[bool]
    reward_scaling_source_min: NotRequired[float]
    reward_scaling_source_max: NotRequired[float]
    reward_scaling_target_min: NotRequired[float]
    reward_scaling_target_max: NotRequired[float]
    dapo_enabled: NotRequired[bool]
    dapo_max_length: NotRequired[int]
    dapo_buffer_length: NotRequired[int]
    dapo_penalty: NotRequired[float]


class GRPOSaveState(TypedDict, total=False):
    """Training state for checkpointing (matches NeMo RL)."""

    consumed_samples: int
    current_step: int
    current_epoch: int
    total_steps: int
    total_valid_tokens: int
    val_reward: float


def _default_grpo_save_state() -> GRPOSaveState:
    """Create default save state."""
    return {
        "consumed_samples": 0,
        "current_step": 0,
        "current_epoch": 0,
        "total_steps": 0,
        "total_valid_tokens": 0,
    }


def calculate_baseline_and_std_per_prompt(
    input_ids: torch.Tensor,
    rewards: torch.Tensor,
    mask: torch.Tensor,
    leave_one_out_baseline: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate per-prompt baseline and standard deviation.

    This matches NeMo RL's calculate_baseline_and_std_per_prompt function.

    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        rewards: Reward values [batch_size]
        mask: Valid sample mask [batch_size]
        leave_one_out_baseline: Use leave-one-out baseline

    Returns:
        Tuple of (baseline, std) tensors [batch_size]
    """
    n = rewards.numel()

    if n == 0:
        return torch.zeros_like(rewards), torch.ones_like(rewards)

    if n == 1:
        # Single sample: baseline is the reward itself, std is 1 to avoid division issues
        return rewards.clone(), torch.ones_like(rewards)

    if leave_one_out_baseline:
        # Leave-one-out: baseline for each sample is mean of all others
        total = rewards.sum()
        baseline = (total - rewards) / (n - 1)

        # Compute std excluding each sample using proper leave-one-out variance
        # For sample i: var_i = sum_{j != i} (r_j - mean_i)^2 / (n - 2) if n > 2, else 0
        std = torch.zeros_like(rewards)
        if n > 2:
            for i in range(n):
                # Create mask for all samples except i
                others_mask = torch.ones(n, dtype=torch.bool, device=rewards.device)
                others_mask[i] = False
                others = rewards[others_mask]
                std_val = others.std(unbiased=True)
                # Handle NaN case (all values same -> std is NaN with unbiased=True)
                if torch.isnan(std_val):
                    std_val = torch.tensor(0.0, device=rewards.device)
                std[i] = std_val
        else:
            # With only 2 samples, std of 1 sample is undefined, use overall std
            overall_std = rewards.std()
            if torch.isnan(overall_std):
                overall_std = torch.tensor(0.0, device=rewards.device)
            std = overall_std.expand_as(rewards)

        # Ensure std is never zero or NaN to avoid division issues
        std = torch.where(torch.isnan(std), torch.zeros_like(std), std)
        std = std.clamp(min=1e-8)
    else:
        # Simple mean baseline
        baseline = rewards.mean().expand_as(rewards)
        std = rewards.std().clamp(min=1e-8).expand_as(rewards)

    return baseline, std


def scale_rewards(
    rewards: torch.Tensor,
    config: RewardScalingConfig,
) -> torch.Tensor:
    """
    Apply linear reward scaling with clamping (matches NeMo RL).

    Args:
        rewards: Original rewards [batch_size]
        config: Scaling configuration

    Returns:
        Scaled rewards
    """
    if not config.get("enabled", False):
        return rewards

    source_min = config.get("source_min", 0.0)
    source_max = config.get("source_max", 1.0)
    target_min = config.get("target_min", 0.0)
    target_max = config.get("target_max", 1.0)

    # Clamp and scale
    clamped = torch.clamp(rewards, min=source_min, max=source_max)
    scaled = target_min + (clamped - source_min) / (source_max - source_min) * (
        target_max - target_min
    )

    return scaled


@dataclass
class GRPOTrainer(BaseTrainer):
    """
    Group Relative Policy Optimization trainer.

    Provides full integration with NeMo RL's GRPO algorithm for training
    DN agents on multi-turn tasks with tool use.

    This trainer:
    - Initializes NeMo RL Policy and vLLM Generation objects
    - Runs multi-turn rollouts with environment feedback
    - Computes advantages with leave-one-out baseline
    - Performs policy gradient updates with clipped loss
    - Handles weight synchronization between training and inference

    Example:
        from dreadnode import Agent
        from dreadnode.core.training import GRPOTrainer, RewardAggregator, SuccessReward

        agent = Agent(model="meta-llama/...", tools=[...])
        rewards = RewardAggregator([SuccessReward(weight=1.0)])

        trainer = GRPOTrainer(
            agent=agent,
            rewards=rewards,
            config={
                "model_name": "meta-llama/Llama-3.1-8B-Instruct",
                "num_prompts_per_step": 32,
                "num_generations_per_prompt": 4,
                "max_rollout_turns": 10,
                "learning_rate": 1e-5,
            }
        )

        state = await trainer.train(prompts, environment, num_steps=1000)
    """

    agent: Agent | None = None
    """DN SDK Agent to train."""

    config: GRPOConfig = field(default_factory=dict)  # type: ignore[assignment]
    """GRPO training configuration."""

    rewards: RewardAggregator | None = None
    """Reward aggregator for computing training rewards."""

    callbacks: list[TrainingCallback] = field(default_factory=list)
    """Training callbacks for logging, checkpointing, etc."""

    # NeMo RL components (initialized during setup)
    # These are typed as Any because they come from NeMo RL which may not be installed
    _policy: Any = None
    """NeMo RL Policy object."""

    _policy_generation: Any = None
    """NeMo RL Generation object (vLLM or Megatron)."""
    _tokenizer: Any = None
    _loss_fn: Any = None
    _train_cluster: Any = None
    _inference_cluster: Any = None
    _logger: Any = None
    _checkpointer: Any = None
    _save_state: GRPOSaveState = field(default_factory=_default_grpo_save_state)

    # State tracking
    _environment: Any = None
    _initialized: bool = False
    _colocated_inference: bool = False
    _generation_stale: bool = True

    def __post_init__(self):
        self.state = TrainingState()

    async def setup(
        self,
        environment: Any | None = None,
        cluster_config: dict[str, Any] | None = None,
    ) -> None:
        """
        Set up NeMo RL components for training.

        Initializes:
        - Ray virtual clusters for training and inference
        - Policy model with optimizer
        - vLLM generation backend (if using vLLM)
        - Loss function
        - Logger and checkpointer

        Args:
            environment: Environment for rollouts (NeMo RL EnvironmentInterface).
            cluster_config: Optional cluster configuration override.
        """
        import ray
        from nemo_rl.algorithms.loss_functions import ClippedPGLossConfig, ClippedPGLossFn
        from nemo_rl.algorithms.utils import get_tokenizer, set_seed
        from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
        from nemo_rl.utils.checkpoint import CheckpointManager
        from nemo_rl.utils.logger import Logger

        if self._initialized:
            return

        # Initialize Ray if not already
        # Use a simple runtime_env with only env_vars to avoid package uploads
        if not ray.is_initialized():
            import os

            env_vars = dict(os.environ)
            env_vars.pop("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", None)
            runtime_env = {"env_vars": env_vars}

            ray.init(
                log_to_driver=True,
                include_dashboard=True,
                runtime_env=runtime_env,
            )

        # Set seed
        seed = self.config.get("seed", 42)
        set_seed(seed)

        # Get tokenizer
        tokenizer_name = self.config.get("tokenizer_name") or self.config.get("model_name")
        if tokenizer_name:
            self._tokenizer = get_tokenizer({"name": tokenizer_name})

        # Store environment
        self._environment = environment

        # Set up clusters
        cluster_cfg = cluster_config or {}
        num_nodes = cluster_cfg.get("num_nodes", 1)
        gpus_per_node = cluster_cfg.get(
            "gpus_per_node", torch.cuda.device_count() if torch.cuda.is_available() else 1
        )

        self._colocated_inference = self.config.get("colocated_inference", True)

        if self._colocated_inference:
            # Single cluster for colocated training and inference
            self._train_cluster = RayVirtualCluster(
                name="grpo_train_cluster",
                bundle_ct_per_node_list=[gpus_per_node] * num_nodes,
                use_gpus=True,
                num_gpus_per_node=gpus_per_node,
                max_colocated_worker_groups=1,
            )
            self._inference_cluster = self._train_cluster
        else:
            # Separate clusters
            train_nodes = max(1, num_nodes - 1)
            inference_nodes = 1

            self._train_cluster = RayVirtualCluster(
                name="grpo_train_cluster",
                bundle_ct_per_node_list=[gpus_per_node] * train_nodes,
                use_gpus=True,
                num_gpus_per_node=gpus_per_node,
                max_colocated_worker_groups=1,
            )
            self._inference_cluster = RayVirtualCluster(
                name="grpo_inference_cluster",
                bundle_ct_per_node_list=[gpus_per_node] * inference_nodes,
                use_gpus=True,
                num_gpus_per_node=gpus_per_node,
                max_colocated_worker_groups=1,
            )

        # Initialize policy
        backend = self.config.get("generation_backend", "vllm")

        if backend == "vllm":
            await self._setup_vllm_backend()
        else:
            await self._setup_megatron_backend()

        # Initialize loss function
        loss_config: ClippedPGLossConfig = {
            "kl_penalty_coeff": self.config.get("kl_penalty_coeff", 0.01),
            "entropy_bonus_coeff": self.config.get("entropy_bonus", 0.0),
            "clip_ratio": self.config.get("clip_ratio", 0.2),
            "use_importance_sampling_correction": self.config.get(
                "use_importance_sampling_correction", True
            ),
        }
        self._loss_fn = ClippedPGLossFn(loss_config)

        # Initialize logger
        log_dir = self.config.get("log_dir", "./logs")
        self._logger = Logger(
            {
                "log_dir": log_dir,
                "wandb_enabled": self.config.get("wandb_enabled", False),
                "tensorboard_enabled": self.config.get("tensorboard_enabled", True),
            }
        )

        # Initialize checkpointer
        checkpoint_dir = self.config.get("checkpoint_dir", "./checkpoints")
        self._checkpointer = CheckpointManager(
            {
                "enabled": self.config.get("checkpoint_dir") is not None,
                "checkpoint_dir": checkpoint_dir,
                "save_period": self.config.get("checkpoint_interval", 500),
            }
        )

        self._initialized = True
        self._generation_stale = True

    async def _setup_vllm_backend(self) -> None:
        """Set up vLLM-based generation backend."""
        from nemo_rl.models.generation.vllm import VllmGeneration
        from nemo_rl.models.policy.lm_policy import Policy

        model_name = self.config.get("model_name")
        if not model_name:
            raise ValueError("model_name is required in config")

        max_seq_length = self.config.get("max_seq_length", 4096)

        # Build NeMo RL-compatible vLLM config
        # Dreadnode wraps the complexity of NeMo RL config formats
        vllm_cfg = {
            "tensor_parallel_size": self.config.get("tensor_parallel_size", 1),
            "pipeline_parallel_size": self.config.get("pipeline_parallel_size", 1),
            "expert_parallel_size": self.config.get("expert_parallel_size", 1),
            "gpu_memory_utilization": self.config.get("gpu_memory_utilization", 0.8),
            "max_model_len": max_seq_length,
            "skip_tokenizer_init": True,
            "async_engine": False,
            "precision": self.config.get("precision", "bfloat16"),
            "kv_cache_dtype": "auto",
            "enforce_eager": True,
            "load_format": self.config.get("load_format", "auto"),
        }

        generation_config = {
            "backend": "vllm",
            "model_name": model_name,
            "max_new_tokens": self.config.get("max_tokens_per_turn", 512),
            "temperature": self.config.get("temperature", 0.7),
            "top_p": self.config.get("top_p", 1.0),
            "top_k": self.config.get("top_k", None),
            "stop_strings": self.config.get("stop_strings"),
            "stop_token_ids": self.config.get("stop_token_ids"),
            "vllm_cfg": vllm_cfg,
            "colocated": {"enabled": self._colocated_inference},
        }

        # Build policy config
        policy_config = {
            "model_name": model_name,
            "tokenizer": {"name": model_name},
            "train_global_batch_size": self.config.get("batch_size", 8),
            "train_micro_batch_size": self.config.get("micro_batch_size", 1),
            "max_total_sequence_length": max_seq_length,
            "precision": self.config.get("precision", "bfloat16"),
            "max_grad_norm": self.config.get("max_grad_norm", 1.0),
            "optimizer": {
                "name": "torch.optim.AdamW",
                "kwargs": {
                    "lr": self.config.get("learning_rate", 1e-5),
                    "weight_decay": 0.01,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                },
            },
            "generation": generation_config,
            "dtensor_cfg": {
                "_v2": False,
                "enabled": True,
                "cpu_offload": False,
                "sequence_parallel": False,
                "activation_checkpointing": False,
                "tensor_parallel_size": 1,
                "context_parallel_size": 1,
            },
            "dynamic_batching": {
                "enabled": False,
                "sequence_length_round": 128,
            },
            "sequence_packing": {
                "enabled": False,
            },
        }

        # Initialize vLLM generation first (prefers clean GPU memory)
        print("Creating VllmGeneration...")
        self._policy_generation = VllmGeneration(
            cluster=self._inference_cluster,
            config=generation_config,
        )
        print("Calling finish_generation...")
        self._policy_generation.finish_generation()

        # Initialize policy
        print("Creating Policy...")
        self._policy = Policy(
            cluster=self._train_cluster,
            config=policy_config,
            tokenizer=self._tokenizer,
            init_optimizer=True,
        )
        print("Policy created successfully!", flush=True)

        # Set up weight synchronization for non-colocated inference
        if not self._colocated_inference:
            print("Setting up weight sync for non-colocated inference...")
            ip, port = self._train_cluster.get_master_address_and_port()
            train_world_size = self._train_cluster.world_size()
            inference_world_size = self._inference_cluster.world_size()
            world_size = train_world_size + inference_world_size

            import ray

            futures_train = self._policy.init_collective(
                ip, port, world_size, train_world_size=train_world_size
            )
            futures_inference = self._policy_generation.init_collective(
                ip, port, world_size, train_world_size=train_world_size
            )
            ray.get(futures_train + futures_inference)
            print("Weight sync initialized!")

        # Prepare refit info
        print("Preparing refit info from policy...")
        state_dict_info = self._policy.prepare_refit_info()
        print(f"Got state_dict_info with {len(state_dict_info) if state_dict_info else 0} entries")
        print("Preparing refit info in generation...")
        self._policy_generation.prepare_refit_info(state_dict_info)
        print("Refit info prepared!")

    async def _setup_megatron_backend(self) -> None:
        """Set up Megatron-based generation backend."""
        from nemo_rl.models.policy.lm_policy import Policy

        model_name = self.config.get("model_name")
        if not model_name:
            raise ValueError("model_name is required in config")

        policy_config = {
            "model_name": model_name,
            "tokenizer": {"name_or_path": model_name},
            "train_global_batch_size": self.config.get("batch_size", 8),
            "train_micro_batch_size": self.config.get("micro_batch_size", 1),
            "max_seq_length": self.config.get("max_seq_length", 4096),
            "learning_rate": self.config.get("learning_rate", 1e-5),
            "generation": {
                "backend": "megatron",
                "temperature": self.config.get("temperature", 0.7),
                "top_p": self.config.get("top_p", 0.95),
                "max_new_tokens": self.config.get("max_tokens_per_turn", 4096),
            },
        }

        self._policy = Policy(
            cluster=self._train_cluster,
            config=policy_config,
            tokenizer=self._tokenizer,
            init_optimizer=True,
        )

        # For megatron backend, policy is also the generation interface
        self._policy_generation = self._policy

    def _refit_policy_generation(self) -> None:
        """Synchronize weights from training policy to generation backend."""
        import ray

        if self._policy_generation is self._policy:
            # Megatron backend - no refit needed
            return

        if self._colocated_inference:
            self._policy.offload_before_refit()
            self._policy_generation.prepare_for_generation(tags=["weights"])

            # Stream weights via IPC
            buffer_size_bytes = int(self._policy.get_free_memory_bytes() * 0.3)
            futures_train = self._policy.stream_weights_via_ipc_zmq(
                buffer_size_bytes=buffer_size_bytes
            )
            futures_inference = self._policy_generation.update_weights_via_ipc_zmq()
            ray.get(futures_train)
            ray.get(futures_inference)

            self._policy.offload_after_refit()
            self._policy_generation.prepare_for_generation(tags=["kv_cache"])
        else:
            # Broadcast via NCCL
            futures_train = self._policy.broadcast_weights_for_collective()
            futures_inference = self._policy_generation.update_weights_from_collective()
            ray.get(futures_train)
            ray.get(futures_inference)

        self._generation_stale = False

    async def train(
        self,
        prompts: Sequence[str],
        environment: Any | None = None,
        num_steps: int | None = None,
    ) -> TrainingState:
        """
        Run GRPO training.

        This implements the full GRPO training loop:
        1. Generate rollouts using vLLM/policy generation
        2. Compute rewards using reward aggregator
        3. Scale rewards and apply DAPO penalty
        4. Compute advantages with leave-one-out baseline
        5. Get logprobs from policy
        6. Train policy with clipped PG loss
        7. Sync weights to generation backend

        Args:
            prompts: Training prompts.
            environment: Environment for rollouts (NeMo RL EnvironmentInterface).
            num_steps: Number of training steps.

        Returns:
            Final training state.
        """
        from nemo_rl.algorithms.reward_functions import apply_reward_shaping
        from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict
        from nemo_rl.experience.rollouts import run_multi_turn_rollout

        # Setup if needed
        if not self._initialized:
            await self.setup(environment)

        self.state.started_at = datetime.now()
        max_steps = num_steps or self.config.get("max_num_steps", 1000)
        max_epochs = self.config.get("max_num_epochs", 1)

        num_prompts_per_step = self.config.get("num_prompts_per_step", 32)
        num_generations = self.config.get("num_generations_per_prompt", 4)
        max_rollout_turns = self.config.get("max_rollout_turns", 10)
        use_leave_one_out = self.config.get("use_leave_one_out_baseline", True)
        normalize_rewards = self.config.get("normalize_rewards", False)

        # Get reward configs
        reward_scaling_config = self.config.get("reward_scaling", {})
        reward_shaping_config = self.config.get("reward_shaping", {})

        # Handle legacy config format
        if self.config.get("reward_scaling_enabled"):
            reward_scaling_config = {
                "enabled": True,
                "source_min": self.config.get("reward_scaling_source_min", 0.0),
                "source_max": self.config.get("reward_scaling_source_max", 1.0),
                "target_min": self.config.get("reward_scaling_target_min", 0.0),
                "target_max": self.config.get("reward_scaling_target_max", 1.0),
            }

        if self.config.get("dapo_enabled"):
            reward_shaping_config = {
                "enabled": True,
                "max_response_length": self.config.get("dapo_max_length", 4096),
                "overlong_buffer_length": self.config.get("dapo_buffer_length", 512),
                "overlong_buffer_penalty": self.config.get("dapo_penalty", 0.5),
            }

        # Build task_to_env mapping
        task_to_env = {}
        if environment is not None:
            task_to_env["default"] = environment
        if self._environment is not None:
            task_to_env["default"] = self._environment

        current_step = self._save_state.get("current_step", 0)
        total_steps = self._save_state.get("total_steps", 0)
        current_epoch = self._save_state.get("current_epoch", 0)

        # Validation at start if configured
        if self.config.get("val_at_start") and current_step == 0:
            if self._generation_stale:
                self._refit_policy_generation()
            val_metrics = await self.evaluate(
                prompts[: self.config.get("max_val_samples", 100)], environment
            )
            self._logger.log_metrics(val_metrics, 0, prefix="validation")
            self._notify_evaluation(val_metrics)

        # Training loop
        while current_epoch < max_epochs and total_steps < max_steps:
            self.state.epoch = current_epoch

            # Create batches from prompts
            for batch_start in range(0, len(prompts), num_prompts_per_step):
                if total_steps >= max_steps:
                    break

                self.state.step = total_steps
                self._notify_step_start()

                # Get batch of prompts
                batch_prompts = list(prompts[batch_start : batch_start + num_prompts_per_step])

                # Prepare batch for NeMo RL
                batch_data = self._prepare_batch(batch_prompts, num_generations)

                # Refit generation weights if stale
                if self._generation_stale:
                    self._refit_policy_generation()
                else:
                    if self._colocated_inference:
                        self._policy.offload_after_refit()
                    self._policy_generation.prepare_for_generation()

                # Generate rollouts
                repeated_batch, rollout_metrics = run_multi_turn_rollout(
                    policy_generation=self._policy_generation,
                    input_batch=batch_data,
                    tokenizer=self._tokenizer,
                    task_to_env=task_to_env,
                    max_seq_len=self.config.get("max_seq_length", 4096),
                    max_rollout_turns=max_rollout_turns,
                    greedy=False,
                )
                self._policy_generation.finish_generation()

                # Extract rewards from rollouts
                rewards = repeated_batch["total_reward"]

                # Apply reward scaling
                if reward_scaling_config.get("enabled"):
                    rewards = scale_rewards(rewards, reward_scaling_config)
                    repeated_batch["total_reward"] = rewards

                # Apply DAPO reward shaping
                if reward_shaping_config.get("enabled"):
                    repeated_batch = apply_reward_shaping(repeated_batch, reward_shaping_config)
                    rewards = repeated_batch["total_reward"]

                # Compute advantages
                input_ids = repeated_batch.get("input_ids", torch.zeros(len(rewards)))
                baseline, std = calculate_baseline_and_std_per_prompt(
                    input_ids,
                    rewards,
                    torch.ones_like(rewards),
                    leave_one_out_baseline=use_leave_one_out,
                )

                advantages = (rewards - baseline).unsqueeze(-1)

                # Normalize advantages if configured
                if normalize_rewards:
                    non_zero_mask = std > 0
                    advantages[non_zero_mask] = advantages[non_zero_mask] / (
                        std.unsqueeze(-1)[non_zero_mask] + 1e-6
                    )

                # Add advantages to message log for training
                for i, message_log in enumerate(repeated_batch["message_log"]):
                    for message in message_log:
                        if message["role"] == "assistant":
                            message["token_loss_mask"] = torch.ones_like(message["token_ids"])
                        else:
                            message["token_loss_mask"] = torch.zeros_like(message["token_ids"])
                        if "generation_logprobs" not in message:
                            message["generation_logprobs"] = torch.zeros_like(
                                message["token_ids"], dtype=torch.float32
                            )
                        message["advantages"] = advantages[i].expand(message["token_ids"].shape)

                # Convert to flat format for training
                flat_messages, input_lengths = batched_message_log_to_flat_message(
                    repeated_batch["message_log"],
                    pad_value_dict={"token_ids": self._tokenizer.pad_token_id},
                )

                # Create training data
                train_data = BatchedDataDict(
                    {
                        "input_ids": flat_messages["token_ids"],
                        "input_lengths": input_lengths,
                        "advantages": flat_messages["advantages"],
                        "generation_logprobs": flat_messages["generation_logprobs"],
                        "token_mask": flat_messages["token_loss_mask"],
                        "sample_mask": repeated_batch.get(
                            "loss_multiplier", torch.ones(len(rewards))
                        ),
                    }
                )
                train_data.to("cpu")

                # Get logprobs from policy
                self._policy.prepare_for_lp_inference()
                fprop_logprobs = self._policy.get_logprobs(train_data)["logprobs"]
                reference_logprobs = self._policy.get_reference_policy_logprobs(train_data)[
                    "reference_logprobs"
                ]
                train_data["prev_logprobs"] = fprop_logprobs
                train_data["reference_policy_logprobs"] = reference_logprobs

                # Train policy
                self._policy.prepare_for_training()
                self._generation_stale = True
                train_results = self._policy.train(train_data, self._loss_fn)

                # Update state
                mean_reward = float(rewards.mean())
                self.state.total_reward += mean_reward
                self.state.best_reward = max(self.state.best_reward, mean_reward)
                self.state.samples_seen += len(batch_prompts) * num_generations
                self.state.metrics["mean_reward"] = mean_reward
                self.state.metrics["loss"] = float(train_results["loss"])
                self.state.metrics["grad_norm"] = float(train_results["grad_norm"])
                self.state.metrics.update(rollout_metrics)

                # Log metrics
                metrics = dict(self.state.metrics)
                self._logger.log_metrics(metrics, total_steps, prefix="train")
                self._notify_step_end(metrics)

                # Validation
                val_period = self.config.get("val_period", 0)
                if val_period > 0 and (total_steps + 1) % val_period == 0:
                    if self._generation_stale:
                        self._refit_policy_generation()
                    val_metrics = await self.evaluate(
                        prompts[: self.config.get("max_val_samples", 100)], environment
                    )
                    self._logger.log_metrics(val_metrics, total_steps + 1, prefix="validation")
                    self._notify_evaluation(val_metrics)
                    self._policy_generation.finish_generation()

                # Checkpointing
                checkpoint_interval = self.config.get("checkpoint_interval", 500)
                checkpoint_dir = self.config.get("checkpoint_dir")
                if checkpoint_dir and (total_steps + 1) % checkpoint_interval == 0:
                    self._save_state["current_step"] = current_step + 1
                    self._save_state["total_steps"] = total_steps + 1
                    self._save_state["current_epoch"] = current_epoch
                    self.save_checkpoint(f"{checkpoint_dir}/step_{total_steps + 1}")

                total_steps += 1
                current_step += 1

            current_epoch += 1
            current_step = 0

        return self.state

    def _prepare_batch(
        self,
        prompts: list[str],
        num_generations: int,
    ) -> Any:
        """Prepare a batch of prompts for NeMo RL rollouts."""
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict

        # Tokenize prompts
        message_logs = []
        for prompt in prompts:
            # Create message log in NeMo RL format
            tokens = self._tokenizer.encode(prompt, add_special_tokens=True)
            message_log = [
                {
                    "role": "user",
                    "content": prompt,
                    "token_ids": torch.tensor(tokens),
                }
            ]
            message_logs.append(message_log)

        # Create batch
        batch = BatchedDataDict(
            {
                "message_log": message_logs,
                "length": torch.tensor([len(self._tokenizer.encode(p)) for p in prompts]),
                "task": ["default"] * len(prompts),
            }
        )

        # Repeat for multiple generations per prompt
        if num_generations > 1:
            batch = batch.repeat_interleave(num_generations)

        return batch

    async def evaluate(
        self,
        prompts: Sequence[str],
        environment: Any | None = None,
    ) -> dict[str, float]:
        """
        Run evaluation using policy generation.

        Args:
            prompts: Evaluation prompts.
            environment: Environment for rollouts.

        Returns:
            Evaluation metrics including accuracy and mean reward.
        """
        from nemo_rl.experience.rollouts import run_multi_turn_rollout

        if self._policy_generation is None:
            return {"error": 1.0}

        # Prepare evaluation batch
        val_batch_size = self.config.get("val_batch_size", 32)
        eval_prompts = list(prompts[:val_batch_size])

        batch_data = self._prepare_batch(eval_prompts, num_generations=1)

        # Build task_to_env mapping
        task_to_env = {}
        if environment is not None:
            task_to_env["default"] = environment
        if self._environment is not None:
            task_to_env["default"] = self._environment

        # Generate rollouts
        repeated_batch, rollout_metrics = run_multi_turn_rollout(
            policy_generation=self._policy_generation,
            input_batch=batch_data,
            tokenizer=self._tokenizer,
            task_to_env=task_to_env,
            max_seq_len=self.config.get("max_seq_length", 4096),
            max_rollout_turns=self.config.get("max_rollout_turns", 10),
            greedy=True,  # Use greedy decoding for evaluation
        )

        # Extract rewards
        rewards = repeated_batch.get("total_reward", torch.zeros(len(eval_prompts)))

        # Compute metrics
        mean_reward = float(rewards.mean()) if len(rewards) > 0 else 0.0
        success_rate = float((rewards > 0.5).float().mean()) if len(rewards) > 0 else 0.0

        metrics = {
            "accuracy": success_rate,
            "reward": mean_reward,
            "mean_turns": rollout_metrics.get("mean_turns", 0.0),
            "num_samples": len(eval_prompts),
        }

        return metrics

    def save_checkpoint(self, path: str) -> None:
        """
        Save model checkpoint including policy weights.

        Args:
            path: Directory path for checkpoint.
        """
        import json

        os.makedirs(path, exist_ok=True)

        # Save training state
        state_dict = {
            "step": self.state.step,
            "epoch": self.state.epoch,
            "samples_seen": self.state.samples_seen,
            "best_reward": self.state.best_reward,
            "total_reward": self.state.total_reward,
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

        # Finalize checkpoint if using checkpointer
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
        self.state.best_reward = state_dict["best_reward"]
        self.state.total_reward = state_dict["total_reward"]
        self.state.metrics = state_dict.get("metrics", {})
        self._save_state = state_dict.get("save_state", _default_grpo_save_state())

        # Note: Policy weights are loaded during setup when weights_path is provided

    def to_nemo_config(self) -> dict[str, Any]:
        """
        Convert to NeMo RL MasterConfig format.

        Returns config suitable for nemo_rl.algorithms.grpo.grpo_train().
        """
        # Get reward configs with legacy support
        reward_scaling = self.config.get("reward_scaling", {})
        if self.config.get("reward_scaling_enabled"):
            reward_scaling = {
                "enabled": True,
                "source_min": self.config.get("reward_scaling_source_min", 0.0),
                "source_max": self.config.get("reward_scaling_source_max", 1.0),
                "target_min": self.config.get("reward_scaling_target_min", 0.0),
                "target_max": self.config.get("reward_scaling_target_max", 1.0),
            }

        reward_shaping = self.config.get("reward_shaping", {})
        if self.config.get("dapo_enabled"):
            reward_shaping = {
                "enabled": True,
                "max_response_length": self.config.get("dapo_max_length", 4096),
                "overlong_buffer_length": self.config.get("dapo_buffer_length", 512),
                "overlong_buffer_penalty": self.config.get("dapo_penalty", 0.5),
            }

        return {
            "policy": {
                "model_name": self.config.get("model_name"),
                "tokenizer": {
                    "name_or_path": self.config.get("tokenizer_name")
                    or self.config.get("model_name")
                },
                "train_global_batch_size": self.config.get("batch_size", 8),
                "train_micro_batch_size": self.config.get("micro_batch_size", 1),
                "max_seq_length": self.config.get("max_seq_length", 4096),
                "max_total_sequence_length": self.config.get("max_seq_length", 4096),
                "make_sequence_length_divisible_by": 8,
                "learning_rate": self.config.get("learning_rate", 1e-5),
                "generation": {
                    "backend": self.config.get("generation_backend", "vllm"),
                    "temperature": self.config.get("temperature", 0.7),
                    "top_p": self.config.get("top_p", 0.95),
                    "max_new_tokens": self.config.get("max_tokens_per_turn", 4096),
                    "colocated": {
                        "enabled": self.config.get("colocated_inference", True),
                    },
                },
            },
            "grpo": {
                "num_prompts_per_step": self.config.get("num_prompts_per_step", 32),
                "num_generations_per_prompt": self.config.get("num_generations_per_prompt", 4),
                "max_num_epochs": self.config.get("max_num_epochs", 1),
                "max_num_steps": self.config.get("max_num_steps", 1000),
                "max_rollout_turns": self.config.get("max_rollout_turns", 10),
                "kl_penalty_coeff": self.config.get("kl_penalty_coeff", 0.01),
                "entropy_bonus": self.config.get("entropy_bonus", 0.0),
                "clip_ratio": self.config.get("clip_ratio", 0.2),
                "normalize_rewards": self.config.get("normalize_rewards", False),
                "use_leave_one_out_baseline": self.config.get("use_leave_one_out_baseline", True),
                "use_dynamic_sampling": self.config.get("use_dynamic_sampling", False),
                "dynamic_sampling_max_gen_batches": self.config.get(
                    "dynamic_sampling_max_gen_batches", 10
                ),
                "batch_multiplier": self.config.get("batch_multiplier", 1.0),
                "overlong_filtering": self.config.get("overlong_filtering", False),
                "reward_scaling": reward_scaling,
                "reward_shaping": reward_shaping,
                "val_period": self.config.get("val_period", 0),
                "val_at_start": self.config.get("val_at_start", False),
                "seed": self.config.get("seed", 42),
            },
            "data": {
                "batch_size": self.config.get("batch_size", 8),
                "max_input_seq_length": self.config.get("max_tokens_per_turn", 4096),
            },
            "logger": {
                "log_dir": self.config.get("log_dir", "./logs"),
                "log_interval": self.config.get("log_interval", 10),
                "wandb_enabled": self.config.get("wandb_enabled", False),
                "tensorboard_enabled": self.config.get("tensorboard_enabled", True),
            },
            "checkpointing": {
                "enabled": self.config.get("checkpoint_dir") is not None,
                "checkpoint_dir": self.config.get("checkpoint_dir", "./checkpoints"),
                "save_period": self.config.get("checkpoint_interval", 500),
            },
        }

    def shutdown(self) -> None:
        """Shutdown trainer and release resources."""

        if self._policy is not None:
            self._policy.shutdown()
            self._policy = None

        if self._policy_generation is not None and self._policy_generation is not self._policy:
            self._policy_generation.shutdown()
            self._policy_generation = None

        if self._train_cluster is not None:
            self._train_cluster.shutdown()
            self._train_cluster = None

        if (
            self._inference_cluster is not None
            and self._inference_cluster is not self._train_cluster
        ):
            self._inference_cluster.shutdown()
            self._inference_cluster = None

        self._initialized = False

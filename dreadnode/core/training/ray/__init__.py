"""
Native Ray-based GRPO/SFT/DPO/PPO/RM training framework.

This module provides a production-ready Ray-native implementation of GRPO,
SFT, DPO, PPO, and Reward Model training for language models. It uses:

- Ray actors for distributed inference (vLLM)
- Ray Train for distributed training with FSDP2
- Native Ray coordination (no RLlib dependency)
- OpenRLHF-style async architecture for overlapped generation/training

Key components:
- RayGRPOConfig: Configuration for GRPO training
- RayGRPOTrainer: Colocated trainer (single GPU, time-shared)
- AsyncRayGRPOTrainer: Async trainer (existing)
- AsyncGRPOCoordinator: New async coordinator with ExperienceBuffer
- FSDP2Learner: Distributed training with PyTorch FSDP2
- RolloutWorker: vLLM-based experience generation
- train_grpo_distributed: Ray Train integration

Architecture modes:
1. Colocated: Time-share GPU between inference/training (RayGRPOTrainer)
2. Async: Overlapped generation/training (AsyncGRPOCoordinator)
3. Distributed: Multi-node FSDP2 (train_grpo_distributed)

Usage:
    # Mode 1: Colocated (single GPU)
    from dreadnode.core.training.ray import RayGRPOConfig, RayGRPOTrainer

    config = RayGRPOConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct")
    trainer = RayGRPOTrainer(config)
    trainer.train(prompts, reward_fn)

    # Mode 2: Async (overlapped)
    from dreadnode.core.training.ray import AsyncGRPOCoordinator

    coordinator = AsyncGRPOCoordinator(
        config=config,
        prompts=prompts,
        reward_fn=reward_fn,
        num_rollout_workers=2,
        buffer_size=2,
    )
    state = await coordinator.train(num_steps=1000)

    # Mode 3: Distributed (multi-node FSDP2)
    from dreadnode.core.training.ray import train_grpo_distributed

    result = train_grpo_distributed(
        config=config,
        prompts=prompts,
        reward_fn=reward_fn,
        num_workers=8,
    )
"""

# Configuration
from dreadnode.core.training.ray.config import (
    GRPOLossConfig,
    RayGRPOConfig,
    TrainingConfig,
    VLLMConfig,
)

# Existing trainers
from dreadnode.core.training.ray.trainer import RayGRPOTrainer
from dreadnode.core.training.ray.async_trainer import AsyncRayGRPOTrainer

# Experience management
from dreadnode.core.training.ray.experience import (
    Experience,
    ExperienceBatch,
    ExperienceBuffer,
)

# Rollout environment protocol
from dreadnode.core.training.ray.rollout_env import (
    RolloutEnvironment,
    StreamingRolloutEnvironment,
    MultiTurnRolloutEnvironment,
)

# Rollout workers
from dreadnode.core.training.ray.rollout_worker import (
    RolloutConfig,
    RolloutWorker,
    RolloutWorkerPool,
    create_rollout_worker,
)

# Multi-turn rollouts
from dreadnode.core.training.ray.multi_turn import (
    ConversationResult,
    ConversationTurn,
    MultiTurnConfig,
    MultiTurnRolloutWorker,
    ToolExecutor,
)

# FSDP2 learner
from dreadnode.core.training.ray.fsdp2_learner import (
    FSDP2Config,
    FSDP2Learner,
    init_distributed,
    cleanup_distributed,
)

# Coordinators
from dreadnode.core.training.ray.coordinator import (
    AsyncGRPOConfig,
    AsyncGRPOCoordinator,
    SyncGRPOCoordinator,
    TrainingState,
)

# Distributed training
from dreadnode.core.training.ray.distributed import (
    DistributedConfig,
    train_grpo_distributed,
    train_sft_distributed,
    tune_grpo,
)

# SFT
from dreadnode.core.training.ray.sft import (
    PackedDataset,
    PackedSample,
    SequencePacker,
    SFTConfig,
    SFTTrainer,
)

# DPO
from dreadnode.core.training.ray.dpo import (
    DPOConfig,
    DPOTrainer,
    PreferencePair,
)

# Reward Model
from dreadnode.core.training.ray.reward_model import (
    RMConfig,
    RewardModel,
    RewardModelHead,
    RewardModelTrainer,
)

# PPO
from dreadnode.core.training.ray.ppo import (
    PPOConfig,
    PPOTrainer,
    ValueHead,
)

# Callbacks
from dreadnode.core.training.ray.callbacks import (
    CallbackHandler,
    CheckpointCallback,
    EarlyStoppingCallback,
    GradientClippingCallback,
    JSONLoggingCallback,
    MetricHistoryCallback,
    PrintCallback,
    ProgressCallback,
    SampleLoggingCallback,
    TensorBoardCallback,
    ToolCallInfo,
    ToolLoggingCallback,
    TrainerCallback,
    TrainerControl,
    WandbCallback,
)

# Learner utilities
from dreadnode.core.training.ray.learner import (
    GRPOLearner,
    TrainingBatch,
    compute_log_probs,
    grpo_loss,
)

# Inference
from dreadnode.core.training.ray.inference import (
    GenerationOutput,
    VLLMInferenceActor,
    VLLMInferencePool,
)

__all__ = [
    # Configuration
    "GRPOLossConfig",
    "RayGRPOConfig",
    "TrainingConfig",
    "VLLMConfig",
    "RolloutConfig",
    "FSDP2Config",
    "AsyncGRPOConfig",
    "DistributedConfig",
    # Trainers
    "RayGRPOTrainer",
    "AsyncRayGRPOTrainer",
    # Experience
    "Experience",
    "ExperienceBatch",
    "ExperienceBuffer",
    # Protocols
    "RolloutEnvironment",
    "StreamingRolloutEnvironment",
    "MultiTurnRolloutEnvironment",
    # Workers
    "RolloutWorker",
    "RolloutWorkerPool",
    "create_rollout_worker",
    # Multi-turn
    "ConversationResult",
    "ConversationTurn",
    "MultiTurnConfig",
    "MultiTurnRolloutWorker",
    "ToolExecutor",
    # Learners
    "GRPOLearner",
    "FSDP2Learner",
    "TrainingBatch",
    "compute_log_probs",
    "grpo_loss",
    # Coordinators
    "AsyncGRPOCoordinator",
    "SyncGRPOCoordinator",
    "TrainingState",
    # Distributed
    "train_grpo_distributed",
    "train_sft_distributed",
    "tune_grpo",
    "init_distributed",
    "cleanup_distributed",
    # Inference
    "GenerationOutput",
    "VLLMInferenceActor",
    "VLLMInferencePool",
    # SFT
    "PackedDataset",
    "PackedSample",
    "SequencePacker",
    "SFTConfig",
    "SFTTrainer",
    # DPO
    "DPOConfig",
    "DPOTrainer",
    "PreferencePair",
    # Reward Model
    "RMConfig",
    "RewardModel",
    "RewardModelHead",
    "RewardModelTrainer",
    # PPO
    "PPOConfig",
    "PPOTrainer",
    "ValueHead",
    # Callbacks
    "CallbackHandler",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "GradientClippingCallback",
    "JSONLoggingCallback",
    "MetricHistoryCallback",
    "PrintCallback",
    "ProgressCallback",
    "SampleLoggingCallback",
    "TensorBoardCallback",
    "ToolCallInfo",
    "ToolLoggingCallback",
    "TrainerCallback",
    "TrainerControl",
    "WandbCallback",
]

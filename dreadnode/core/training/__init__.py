"""
Training module for DN SDK agents.

This module provides reinforcement learning and supervised fine-tuning
capabilities for Dreadnode agents, built on Ray and NeMo RL.

Install with: pip install dreadnode[training]

Requires:
- Ray (distributed computing, serving)
- NeMo RL (GRPO, policy optimization)
- vLLM (fast inference, online training)

Components:

    Rollouts:
        RolloutOrchestrator - Multi-turn agent execution for training
        RolloutConfig - Configuration for rollout execution
        RolloutResult - Result from completed rollout

    Rewards:
        RewardFunction - Protocol for reward computation
        RewardAggregator - Combine multiple reward sources
        ScorerReward - Bridge DN Scorer → RewardFunction
        SuccessReward, ToolPenalty, etc. - Built-in rewards

    Trainers:
        GRPOTrainer - Group Relative Policy Optimization
        SFTTrainer - Supervised Fine-Tuning

    Serving:
        VllmClient - Client for vLLM HTTP server

Example:
    ```python
    from dreadnode import Agent, scorer
    from dreadnode.core.training import (
        GRPOTrainer,
        RewardAggregator,
        ScorerReward,
        SuccessReward,
    )

    # Define agent
    agent = Agent(model="meta-llama/...", tools=[...])

    # Define rewards using DN scorers
    @scorer
    async def task_success(result) -> float:
        return 1.0 if result.success else 0.0

    rewards = RewardAggregator([
        ScorerReward(task_success, weight=1.0),
        SuccessReward(weight=0.5),
    ])

    # Train
    trainer = GRPOTrainer(agent=agent, rewards=rewards)
    await trainer.train(prompts, num_steps=1000)
    ```
"""

from __future__ import annotations

import importlib
import typing as t

__all__ = [
    # Rollouts
    "RolloutOrchestrator",
    "RolloutConfig",
    "RolloutResult",
    "RolloutMetrics",
    "MessageLog",
    "Message",
    "TurnResult",
    "GenerationInterface",
    "EnvironmentInterface",
    # Adapters
    "DNAgentAdapter",
    "AgentAdapter",  # Alias for DNAgentAdapter
    "VllmAdapter",
    # Rewards
    "RewardFunction",
    "RewardAggregator",
    "RewardResult",
    "RewardComponent",
    "AggregationStrategy",
    # Reward Functions
    "SuccessReward",
    "ToolPenalty",
    "LengthPenalty",
    "TurnPenalty",
    "FormatReward",
    # Scorer Bridge
    "ScorerReward",
    "ScorerHookReward",
    "MetricReward",
    # Shaping
    "RewardShaper",
    "apply_dapo_penalty",
    "apply_scaling",
    "scale_rewards",
    # Trainers
    "GRPOTrainer",
    "GRPOConfig",
    "SFTTrainer",
    "SFTConfig",
    "BaseTrainer",
    "TrainingConfig",
    "TrainingState",
    "TrainingCallback",
    # Utility functions
    "calculate_baseline_and_std_per_prompt",
    # Serving
    "VllmClient",
    "VllmTrainingContext",
    "create_openai_client",
    "wait_for_vllm_ready",
]

# Dependency state
_DEPS_CHECKED = False
_VLLM_AVAILABLE = False


class TrainingDependencyError(ImportError):
    """Raised when training dependencies are not installed."""

    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(
            f"Training requires: {', '.join(missing)}. "
            f"Install with: pip install dreadnode[training]"
        )


def _check_dependencies() -> None:
    """
    Check required dependencies on first import.

    Raises:
        TrainingDependencyError: If required dependencies are missing.
    """
    global _DEPS_CHECKED, _VLLM_AVAILABLE

    if _DEPS_CHECKED:
        return

    missing: list[str] = []

    # Ray is required
    try:
        import ray  # noqa: F401
    except ImportError:
        missing.append("ray")

    # NeMo RL is required
    try:
        import nemo_rl  # noqa: F401
    except ImportError:
        missing.append("nemo-rl")

    # PyTorch is required
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")

    # vLLM is optional but recommended
    try:
        import vllm  # noqa: F401

        _VLLM_AVAILABLE = True
    except ImportError:
        import warnings

        warnings.warn(
            "vLLM not installed. Model serving features will not be available. "
            "Install with: pip install vllm",
            stacklevel=3,
        )
        _VLLM_AVAILABLE = False

    if missing:
        raise TrainingDependencyError(missing)

    _DEPS_CHECKED = True


def is_vllm_available() -> bool:
    """Check if vLLM is available."""
    _check_dependencies()
    return _VLLM_AVAILABLE


# Lazy import mapping
# Maps exported names to their source modules
_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
    # Rollouts - (module, attribute or None for same name)
    "RolloutOrchestrator": (".rollouts", None),
    "RolloutConfig": (".rollouts", None),
    "RolloutResult": (".rollouts", None),
    "RolloutMetrics": (".rollouts", None),
    "MessageLog": (".rollouts", None),
    "Message": (".rollouts", None),
    "TurnResult": (".rollouts", None),
    "GenerationInterface": (".rollouts", None),
    "EnvironmentInterface": (".rollouts", None),
    # Adapters
    "DNAgentAdapter": (".rollouts", None),
    "AgentAdapter": (".rollouts", "DNAgentAdapter"),  # Alias
    "VllmAdapter": (".rollouts", None),
    # Rewards
    "RewardFunction": (".rewards", None),
    "RewardAggregator": (".rewards", None),
    "RewardResult": (".rewards", None),
    "RewardComponent": (".rewards", None),
    "AggregationStrategy": (".rewards", None),
    # Reward Functions
    "SuccessReward": (".rewards", None),
    "ToolPenalty": (".rewards", None),
    "LengthPenalty": (".rewards", None),
    "TurnPenalty": (".rewards", None),
    "FormatReward": (".rewards", None),
    # Scorer Bridge
    "ScorerReward": (".rewards", None),
    "ScorerHookReward": (".rewards", None),
    "MetricReward": (".rewards", None),
    # Shaping
    "RewardShaper": (".rewards", None),
    "apply_dapo_penalty": (".rewards", None),
    "apply_scaling": (".rewards", None),
    "scale_rewards": (".trainers.grpo", None),
    # Trainers
    "GRPOTrainer": (".trainers", None),
    "GRPOConfig": (".trainers.grpo", None),
    "SFTTrainer": (".trainers", None),
    "SFTConfig": (".trainers.sft", None),
    "BaseTrainer": (".trainers", None),
    "TrainingConfig": (".trainers", None),
    "TrainingState": (".trainers", None),
    "TrainingCallback": (".trainers", None),
    # Utility functions
    "calculate_baseline_and_std_per_prompt": (".trainers.grpo", None),
    # Serving
    "VllmClient": (".serving", None),
    "VllmTrainingContext": (".serving", None),
    "create_openai_client": (".serving", None),
    "wait_for_vllm_ready": (".serving", None),
}

# Cache for imported modules
_MODULE_CACHE: dict[str, t.Any] = {}


def __getattr__(name: str) -> t.Any:
    """
    Lazy import handler.

    Imports components only when first accessed, after checking dependencies.
    """
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module 'dreadnode.training' has no attribute '{name}'")

    # Check dependencies on first import
    _check_dependencies()

    module_path, attr_name = _LAZY_IMPORTS[name]
    attr_name = attr_name or name

    # Check module cache
    cache_key = f"{module_path}.{attr_name}"
    if cache_key in _MODULE_CACHE:
        return _MODULE_CACHE[cache_key]

    # Import the module
    try:
        module = importlib.import_module(module_path, package="dreadnode.core.training")
        component = getattr(module, attr_name)
        _MODULE_CACHE[cache_key] = component
        return component
    except ImportError as e:
        raise ImportError(
            f"Failed to import {name} from dreadnode.core.training{module_path}: {e}"
        ) from e
    except AttributeError as e:
        raise AttributeError(
            f"Module dreadnode.core.training{module_path} has no attribute '{attr_name}'"
        ) from e


def __dir__() -> list[str]:
    """Return available attributes for autocomplete."""
    return list(__all__) + [
        "is_vllm_available",
        "TrainingDependencyError",
    ]

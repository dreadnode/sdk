"""
Rollout orchestration for agent training.

This module provides multi-turn rollout execution matching NeMo RL's pattern,
integrated with DN SDK agents.

Key components:
- RolloutOrchestrator: Manages multi-turn agent-environment interaction
- RolloutConfig: Configuration for rollout execution
- RolloutResult: Result from completed rollout with metrics

Usage:
    from dreadnode.core.training.rollouts import RolloutOrchestrator, RolloutConfig

    orchestrator = RolloutOrchestrator(config)

    # Single rollout
    result = await orchestrator.run_single(agent, goal, environment)

    # Batch rollouts
    results = await orchestrator.run_batch(agents, goals, environment)
"""

from dreadnode.core.training.rollouts.types import (
    RolloutConfig,
    RolloutResult,
    RolloutMetrics,
    MessageLog,
    Message,
    MessageRole,
    TurnResult,
)
from dreadnode.core.training.rollouts.orchestrator import (
    RolloutOrchestrator,
    GenerationInterface,
    EnvironmentInterface,
    EnvironmentReturn,
)
from dreadnode.core.training.rollouts.adapters import (
    DNAgentAdapter,
    VllmAdapter,
)

__all__ = [
    # Core
    "RolloutOrchestrator",
    "RolloutConfig",
    "RolloutResult",
    "RolloutMetrics",
    "MessageLog",
    "Message",
    "MessageRole",
    "TurnResult",
    # Interfaces
    "GenerationInterface",
    "EnvironmentInterface",
    "EnvironmentReturn",
    # Adapters
    "DNAgentAdapter",
    "VllmAdapter",
]

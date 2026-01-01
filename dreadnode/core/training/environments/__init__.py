"""
NeMo RL environment wrappers for agent training.

Provides EnvironmentInterface implementations for training DN agents.

Key components:
- AgentEnvironment: Multi-turn agent evaluation environment
- AgentEnvConfig: Environment configuration

Usage:
    from dreadnode.core.training import AgentEnvironment, AgentEnvConfig

    config: AgentEnvConfig = {
        "num_workers": 4,
        "success_reward": 1.0,
        "eval_mode": "offline",
    }

    env = AgentEnvironment.remote(config)
"""

from dreadnode.core.training.environments.agent_env import (
    AgentEnvironment,
    AgentEnvConfig,
    AgentEnvMetadata,
    AgentRewardWorker,
)

__all__ = [
    "AgentEnvironment",
    "AgentEnvConfig",
    "AgentEnvMetadata",
    "AgentRewardWorker",
]

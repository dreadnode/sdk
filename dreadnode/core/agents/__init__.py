from pydantic.dataclasses import rebuild_dataclass

from dreadnode.core import stopping
from dreadnode.core.agents import events, result
from dreadnode.core.agents.agent import (
    Agent,
    agent,
)
from dreadnode.core.agents.result import AgentResult
from dreadnode.core.agents.trajectory import Trajectory
from dreadnode.core.hook import Hook

Agent.model_rebuild()
Trajectory.model_rebuild()

rebuild_dataclass(AgentResult)  # type: ignore[arg-type]

__all__ = [
    "Agent",
    "AgentResult",
    "Continue",
    "Fail",
    "Finish",
    "Hook",
    "Reaction",
    "Retry",
    "RetryWithFeedback",
    "TaskAgent",
    "Thread",
    "agent",
    "events",
    "exceptions",
    "hooks",
    "reactions",
    "result",
    "stopping",
]

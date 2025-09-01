from pydantic.dataclasses import rebuild_dataclass

from dreadnode.agent import error, events, hooks, reactions, result, stop, tools
from dreadnode.agent.agent import Agent
from dreadnode.agent.result import AgentResult
from dreadnode.agent.thread import Thread

Agent.model_rebuild()
Thread.model_rebuild()

rebuild_dataclass(AgentResult)  # type: ignore[arg-type]

__all__ = [
    "Agent",
    "AgentResult",
    "Thread",
    "error",
    "events",
    "hooks",
    "reactions",
    "result",
    "stop",
    "tools",
]

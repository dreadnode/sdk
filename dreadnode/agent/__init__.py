from pydantic.dataclasses import rebuild_dataclass

from dreadnode.agent import error, events, reactions, result, stop
from dreadnode.agent.agent import Agent
from dreadnode.agent.events import rebuild_event_models
from dreadnode.agent.result import AgentResult
from dreadnode.agent.thread import Thread

Agent.model_rebuild()
Thread.model_rebuild()

rebuild_event_models()

rebuild_dataclass(AgentResult)  # type: ignore[arg-type]

__all__ = [
    "Agent",
    "AgentResult",
    "Thread",
    "error",
    "events",
    "reactions",
    "result",
    "stop",
]

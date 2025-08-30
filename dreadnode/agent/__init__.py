from pydantic.dataclasses import rebuild_dataclass

from dreadnode.agent.agent import Agent
from dreadnode.agent.events import rebuild_event_models
from dreadnode.agent.result import AgentResult
from dreadnode.agent.state import State

Agent.model_rebuild()
State.model_rebuild()

# rebuild_event_models()

rebuild_dataclass(AgentResult)  # type: ignore[arg-type]

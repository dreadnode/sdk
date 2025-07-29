import typing as t

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from rigging.generator.base import Usage
from rigging.message import Message

if t.TYPE_CHECKING:
    from dreadnode.agent.agent import Agent


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class AgentResult:
    agent: "Agent"
    messages: list[Message]
    usage: Usage
    steps: int
    failed: bool
    error: Exception | str | None

    def __repr__(self) -> str:
        return (
            f"AgentResult(agent={self.agent.name}, "
            f"messages={len(self.messages)}, "
            f"usage={self.usage}, "
            f"steps={self.steps}, "
            f"failed={self.failed}, "
            f"error={self.error})"
        )

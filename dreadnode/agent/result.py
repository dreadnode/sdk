import typing as t
from dataclasses import dataclass, field

from rigging.generator.base import Usage
from rigging.message import Message
from rigging.model import Model

AgentStatus = t.Literal["success", "error"]


@dataclass
class AgentResult:
    messages: list[Message]
    agent: "Agent"
    output: Model | Message | None = None
    failed: bool = False
    error: Exception | None = None
    usage: Usage
    turns: int
    sub_tasks: dict[str, "AgentResult"] = field(default_factory=dict)

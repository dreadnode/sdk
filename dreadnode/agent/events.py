import typing as t
from dataclasses import dataclass

from pydantic import ValidationError

if t.TYPE_CHECKING:
    from rigging import Message, Model
    from rigging.generator import Usage
    from rigging.tools import ToolCall

    from dreadnode.agent.agent import Agent
    from dreadnode.agent.thread import Thread


AgentEventT = t.TypeVar("AgentEventT", bound="AgentEvent")


@dataclass
class AgentEvent:
    agent: "Agent"
    thread: "Thread"
    messages: "list[Message]"
    events: "list[AgentEvent]"


@dataclass
class AgentStart(AgentEvent):
    user_input: str


@dataclass
class StepStart(AgentEvent):
    step: int


@dataclass
class GenerationEnd(AgentEvent):
    message: "Message"
    usage: "Usage | None"


@dataclass
class ToolCallEnd(AgentEvent):
    tool_call: "ToolCall"
    tool_response: "Message"


@dataclass
class OutputParsingFailed(AgentEvent):
    error: ValidationError
    message: "Message"


@dataclass
class AgentRunEnd(AgentEvent):
    result: "AgentResult"


@dataclass
class AgentRunError(AgentEvent):
    error: Exception


@dataclass
class AgentResult:
    output: "Model | Message"
    history: "list[Message]"
    usage: "Usage"
    turns: int

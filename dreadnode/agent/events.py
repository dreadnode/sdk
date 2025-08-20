import typing as t
from dataclasses import field  # Some odities with repr=False, otherwise I would use pydantic.Field

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass, rebuild_dataclass

from dreadnode.agent.types import Message, ToolCall, Usage
from dreadnode.util import shorten_string

if t.TYPE_CHECKING:
    from dreadnode.agent.agent import Agent
    from dreadnode.agent.reactions import Reaction
    from dreadnode.agent.result import AgentResult
    from dreadnode.agent.thread import Thread


EventT = t.TypeVar("EventT", bound="Event")


@dataclass
class Event:
    agent: "Agent" = field(repr=False)
    thread: "Thread" = field(repr=False)
    messages: "list[Message]" = field(repr=False)
    events: "list[Event]" = field(repr=False)

    def get_latest_event_by_type(self, event_type: type[EventT]) -> EventT | None:
        """
        Returns the latest event of the specified type from the thread's events.

        Args:
            event_type: The type of event to search for.
        """
        for event in reversed(self.events):
            if isinstance(event, event_type):
                return event
        return None

    def get_events_by_type(self, event_type: type[EventT]) -> list[EventT]:
        """
        Returns all events of the specified type from the thread's events.

        Args:
            event_type: The type of event to search for.
        """
        return [event for event in self.events if isinstance(event, event_type)]


@dataclass
class AgentStart(Event): ...


@dataclass
class StepStart(Event):
    step: int


@dataclass
class GenerationEnd(Event):
    message: Message
    usage: "Usage | None"

    def __repr__(self) -> str:
        message_content = shorten_string(str(self.message.content), 50)
        tool_call_count = len(self.message.tool_calls) if self.message.tool_calls else 0
        message = f"Message(role={self.message.role}, content='{message_content}', tool_calls={tool_call_count})"
        return f"GenerationEnd(message={message})"


@dataclass
class AgentStalled(Event): ...


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class AgentError(Event):
    error: Exception


@dataclass
class ToolStart(Event):
    tool_call: ToolCall

    def __repr__(self) -> str:
        return f"ToolStart(tool_call={self.tool_call})"


@dataclass
class ToolEnd(Event):
    tool_call: ToolCall
    message: Message
    stop: bool

    def __repr__(self) -> str:
        message_content = shorten_string(str(self.message.content), 50)
        message = f"Message(role={self.message.role}, content='{message_content}')"
        return f"ToolEnd(tool_call={self.tool_call}, message={message}, stop={self.stop})"


@dataclass
class Reacted(Event):
    hook_name: str
    reaction: "Reaction"


@dataclass
class AgentEnd(Event):
    result: "AgentResult"


def rebuild_event_models() -> None:
    from dreadnode.agent.agent import Agent  # noqa: F401,PLC0415
    from dreadnode.agent.reactions import Reaction  # noqa: F401,PLC0415
    from dreadnode.agent.result import AgentResult  # noqa: F401,PLC0415
    from dreadnode.agent.thread import Thread  # noqa: F401,PLC0415

    rebuild_dataclass(Event)  # type: ignore[arg-type]
    rebuild_dataclass(AgentStart)  # type: ignore[arg-type]
    rebuild_dataclass(StepStart)  # type: ignore[arg-type]
    rebuild_dataclass(GenerationEnd)  # type: ignore[arg-type]
    rebuild_dataclass(AgentStalled)  # type: ignore[arg-type]
    rebuild_dataclass(AgentError)  # type: ignore[arg-type]
    rebuild_dataclass(ToolStart)  # type: ignore[arg-type]
    rebuild_dataclass(ToolEnd)  # type: ignore[arg-type]
    rebuild_dataclass(Reacted)  # type: ignore[arg-type]
    rebuild_dataclass(AgentEnd)  # type: ignore[arg-type]

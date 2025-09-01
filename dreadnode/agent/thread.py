from copy import deepcopy

from pydantic import BaseModel, Field
from rigging.generator import Usage
from rigging.message import Message

from dreadnode.agent.events import (
    AgentEvent,
    GenerationEnd,
    _total_usage_from_events,
)


class Thread(BaseModel):
    messages: list[Message] = Field(default_factory=list)
    """The current messages for this thread."""
    events: list[AgentEvent] = Field(default_factory=list)
    """All events that have occurred during the use of this thread."""

    def __repr__(self) -> str:
        parts = []
        if self.messages:
            parts.append(f"messages={len(self.messages)}")
        if self.events:
            parts.append(f"events={len(self.events)}")
            parts.append(f"last_event={self.events[-1]}")
            parts.append(f"total_usage={self.total_usage}")

        return f"Thread({', '.join(parts)})"

    @property
    def total_usage(self) -> Usage:
        """Aggregates the usage from all events in the thread."""
        return _total_usage_from_events(self.events)

    @property
    def last_usage(self) -> Usage | None:
        """Returns the usage from the last generation event, if available."""
        if not self.events:
            return None
        last_event = self.events[-1]
        if isinstance(last_event, GenerationEnd):
            return last_event.usage
        return None

    def fork(self) -> "Thread":
        return Thread(messages=deepcopy(self.messages), events=deepcopy(self.events))

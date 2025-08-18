import typing as t

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from dreadnode.agent.agent import Agent
from dreadnode.agent.types import Message

if t.TYPE_CHECKING:
    from dreadnode.agent.events import Event


@dataclass
class Reaction(Exception): ...  # noqa: N818


@dataclass
class Continue(Reaction):
    messages: list[Message] = Field(repr=False)


# New Reaction Type
@dataclass
class DelegateTask(Reaction):
    """A reaction that delegates a task to another agent."""

    target_agent: "Agent"
    task_input: str


@dataclass
class Retry(Reaction):
    messages: list[Message] | None = Field(None, repr=False)


@dataclass
class RetryWithFeedback(Reaction):
    feedback: str


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class Fail(Reaction):
    error: Exception | str


@dataclass
class Finish(Reaction):
    reason: str | None = None


@dataclass
class GiveUp(Reaction):
    reason: str | None = None


@dataclass
class MarkComplete(Reaction):
    complete: bool = False


@t.runtime_checkable
class Hook(t.Protocol):
    def __call__(self, event: "Event") -> "t.Awaitable[Reaction | None]": ...

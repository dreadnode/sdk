import typing as t

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from dreadnode.agent.types import Message

if t.TYPE_CHECKING:
    from dreadnode.agent.events import Event


@dataclass
class Reaction(Exception): ...


@dataclass
class Continue(Reaction):
    messages: list[Message] = Field(repr=False)


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


@t.runtime_checkable
class Hook(t.Protocol):
    def __call__(self, event: "Event") -> "t.Awaitable[Reaction | None]": ...

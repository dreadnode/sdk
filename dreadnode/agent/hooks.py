import typing as t
from dataclasses import dataclass

if t.TYPE_CHECKING:
    from rigging.message import Message
    from rigging.result import AgentResult

    from dreadnode.agent.events import AgentEvent


@dataclass
class HookAction:
    """Base class for all actions that a hook can return to the run loop."""


@dataclass
class ModifyHistory(HookAction):
    """Instructs the run loop to replace the current history with a new one."""

    new_history: list[Message]


@dataclass
class RequestRetry(HookAction):
    """Instructs the run loop to stop the current turn and start a new one with feedback."""

    feedback: str


@dataclass
class TerminateRun(HookAction):
    """Instructs the run loop to stop execution immediately and return a final result."""

    result: AgentResult


@t.runtime_checkable
class Hook(t.Protocol):
    """A protocol for hooks that can be used with agents."""

    def __call__(self, event: "AgentEvent") -> "t.Awaitable[HookAction | None]":
        """Process an agent event and optionally return a modified event."""

import inspect
import typing as t

from dreadnode.agent.reactions import RetryWithFeedback

if t.TYPE_CHECKING:
    from dreadnode.agent.events import (
        AgentEvent,
    )
    from dreadnode.agent.reactions import Reaction


@t.runtime_checkable
class Hook(t.Protocol):
    async def __call__(
        self,
        event: "AgentEvent",
    ) -> "Reaction | None": ...


def retry_with_feedback(
    event_type: "type[AgentEvent] | t.Callable[[AgentEvent], bool]", feedback: str
) -> "Hook":
    """
    Create a hook that provides feedback when the specified event occurs.

    Args:
        event_type: The type of event to listen for, or a callable that returns True if feedback should be provided.
        feedback: The feedback message to provide when the event occurs.

    Returns:
        A hook that provides feedback when the event occurs.
    """

    async def retry_with_feedback(event: "AgentEvent") -> "Reaction | None":
        if isinstance(event_type, type) and not isinstance(event, event_type):
            return None

        if inspect.isfunction(event_type) and not event_type(event):
            return None

        return RetryWithFeedback(feedback=feedback)

    return retry_with_feedback

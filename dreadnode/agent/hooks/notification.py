import typing as t
from abc import ABC, abstractmethod

from loguru import logger

if t.TYPE_CHECKING:
    from dreadnode.agent.events import AgentEvent
    from dreadnode.agent.reactions import Reaction


class NotificationBackend(ABC):
    @abstractmethod
    async def send(self, event: "AgentEvent", message: str) -> None:
        """Send a notification for the given event."""


class LogNotificationBackend(NotificationBackend):
    async def send(self, event: "AgentEvent", message: str) -> None:
        logger.info(f"[{event.agent.name}] {message}")


class TerminalNotificationBackend(NotificationBackend):
    async def send(self, event: "AgentEvent", message: str) -> None:
        import sys

        print(f"[{event.agent.name}] {message}", file=sys.stderr)


class WebhookNotificationBackend(NotificationBackend):
    def __init__(self, url: str, headers: dict[str, str] | None = None):
        self.url = url
        self.headers = headers or {}

    async def send(self, event: "AgentEvent", message: str) -> None:
        import httpx

        payload = {
            "agent": event.agent.name,
            "event": event.__class__.__name__,
            "message": message,
            "timestamp": event.timestamp.isoformat(),
        }

        async with httpx.AsyncClient() as client:
            await client.post(self.url, json=payload, headers=self.headers)


def notify(
    event_type: "type[AgentEvent] | t.Callable[[AgentEvent], bool]",
    message: str | t.Callable[["AgentEvent"], str],
    backend: NotificationBackend | None = None,
) -> t.Callable[["AgentEvent"], t.Awaitable["Reaction | None"]]:
    """
    Create a notification hook that sends notifications when events occur.

    Unlike other hooks, notification hooks don't affect agent execution - they return
    None (no reaction) and run asynchronously to deliver notifications.

    Args:
        event_type: Event type to trigger on, or predicate function
        message: Static message or callable that generates message from event
        backend: Notification backend (defaults to terminal output)

    Returns:
        Hook that sends notifications

    Example:
        ```python
        from dreadnode.agent import Agent
        from dreadnode.agent.events import ToolStart
        from dreadnode.agent.hooks.notification import notify

        agent = Agent(
            name="analyzer",
            hooks=[
                notify(
                    ToolStart,
                    lambda e: f"Starting tool: {e.tool_name}",
                ),
            ],
        )
        ```
    """
    notification_backend = backend or TerminalNotificationBackend()

    async def notification_hook(event: "AgentEvent") -> "Reaction | None":
        should_notify = False

        if isinstance(event_type, type):
            should_notify = isinstance(event, event_type)
        elif callable(event_type):
            should_notify = event_type(event)

        if not should_notify:
            return None

        msg = message(event) if callable(message) else message

        try:
            await notification_backend.send(event, msg)
        except Exception:  # noqa: BLE001
            logger.exception("Notification hook failed")

        return None

    return notification_hook

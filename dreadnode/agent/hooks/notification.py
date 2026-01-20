import typing as t
from abc import ABC, abstractmethod

from loguru import logger

if t.TYPE_CHECKING:
    import httpx

    from dreadnode.agent.events import AgentEvent


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
    def __init__(self, url: str, headers: dict[str, str] | None = None, timeout: float = 5.0):
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "WebhookNotificationBackend":
        import httpx

        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._client:
            await self._client.aclose()

    async def send(self, event: "AgentEvent", message: str) -> None:
        import httpx

        if not self._client:
            self._client = httpx.AsyncClient(timeout=self.timeout)

        payload = self._build_payload(event, message)
        await self._client.post(self.url, json=payload, headers=self.headers)

    def _build_payload(self, event: "AgentEvent", message: str) -> dict[str, str]:
        """Override this to customize webhook payload."""
        return {
            "agent": event.agent.name,
            "event": event.__class__.__name__,
            "message": message,
            "timestamp": event.timestamp.isoformat(),
        }


def notify(
    event_type: "type[AgentEvent] | t.Callable[[AgentEvent], bool]",
    message: str | t.Callable[["AgentEvent"], str] | None = None,
    backend: NotificationBackend | None = None,
) -> t.Callable[["AgentEvent"], t.Awaitable[None]]:
    """
    Create a notification hook that sends notifications when events occur.

    Unlike other hooks, notification hooks don't affect agent execution - they return
    None (no reaction) and run asynchronously to deliver notifications.

    Args:
        event_type: Event type to trigger on, or predicate function
        message: Static message or callable that generates message from event.
                 If None, uses event.format_notification()
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
                notify(ToolStart),  # Uses default formatting
                notify(
                    ToolStart,
                    lambda e: f"Starting tool: {e.tool_name}",
                ),
            ],
        )
        ```
    """
    notification_backend = backend or TerminalNotificationBackend()

    async def notification_hook(event: "AgentEvent") -> None:
        should_notify = False

        if isinstance(event_type, type):
            should_notify = isinstance(event, event_type)
        elif callable(event_type):
            should_notify = event_type(event)

        if not should_notify:
            return

        # Use custom message if provided, otherwise delegate to event
        if message is None:
            msg = event.format_notification()
        else:
            msg = message(event) if callable(message) else message

        try:
            await notification_backend.send(event, msg)
        except Exception:  # noqa: BLE001
            logger.exception("Notification hook failed")

        return

    return notification_hook

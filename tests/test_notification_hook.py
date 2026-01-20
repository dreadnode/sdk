import typing as t
from unittest.mock import AsyncMock, MagicMock

import pytest

from dreadnode.agent.events import AgentEvent, ToolStart
from dreadnode.agent.hooks.notification import (
    LogNotificationBackend,
    NotificationBackend,
    TerminalNotificationBackend,
    WebhookNotificationBackend,
    notify,
)


class MockEvent(AgentEvent):
    pass


@pytest.fixture
def mock_event() -> AgentEvent:
    agent = MagicMock()
    agent.name = "test_agent"
    thread = MagicMock()
    messages: list[t.Any] = []
    events: list[AgentEvent] = []

    return MockEvent(
        session_id=MagicMock(),
        agent=agent,
        thread=thread,
        messages=messages,
        events=events,
    )


async def test_log_notification_backend(mock_event: AgentEvent) -> None:
    from unittest.mock import patch

    backend = LogNotificationBackend()

    with patch("dreadnode.agent.hooks.notification.logger.info") as mock_logger:
        await backend.send(mock_event, "Test notification")

    mock_logger.assert_called_once()
    call_args = mock_logger.call_args[0][0]
    assert "Test notification" in call_args
    assert "test_agent" in call_args


async def test_terminal_notification_backend(mock_event: AgentEvent) -> None:
    from io import StringIO
    from unittest.mock import patch

    backend = TerminalNotificationBackend()

    stderr_capture = StringIO()
    with patch("sys.stderr", stderr_capture):
        await backend.send(mock_event, "Test notification")

    output = stderr_capture.getvalue()
    assert "Test notification" in output
    assert "test_agent" in output


async def test_webhook_notification_backend(mock_event: AgentEvent) -> None:
    mock_client = AsyncMock()
    mock_post = AsyncMock()
    mock_client.__aenter__.return_value.post = mock_post

    backend = WebhookNotificationBackend("https://example.com/webhook")

    from unittest.mock import patch

    import httpx

    with patch.object(httpx, "AsyncClient", return_value=mock_client):
        await backend.send(mock_event, "Test notification")

    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["json"]["message"] == "Test notification"
    assert call_kwargs["json"]["agent"] == "test_agent"


async def test_notify_hook_with_event_type(mock_event: AgentEvent) -> None:
    backend = MagicMock(spec=NotificationBackend)
    backend.send = AsyncMock()

    hook = notify(MockEvent, "Test message", backend=backend)

    reaction = await hook(mock_event)

    assert reaction is None
    backend.send.assert_called_once_with(mock_event, "Test message")


async def test_notify_hook_uses_terminal_by_default(mock_event: AgentEvent) -> None:
    from io import StringIO
    from unittest.mock import patch

    hook = notify(MockEvent, "Default notification")

    stderr_capture = StringIO()
    with patch("sys.stderr", stderr_capture):
        reaction = await hook(mock_event)

    assert reaction is None
    output = stderr_capture.getvalue()
    assert "Default notification" in output


async def test_notify_hook_with_callable_message(mock_event: AgentEvent) -> None:
    backend = MagicMock(spec=NotificationBackend)
    backend.send = AsyncMock()

    hook = notify(MockEvent, lambda e: f"Event from {e.agent.name}", backend=backend)

    reaction = await hook(mock_event)

    assert reaction is None
    backend.send.assert_called_once_with(mock_event, "Event from test_agent")


async def test_notify_hook_with_predicate(mock_event: AgentEvent) -> None:
    backend = MagicMock(spec=NotificationBackend)
    backend.send = AsyncMock()

    hook = notify(lambda e: e.agent.name == "test_agent", "Matched!", backend=backend)

    reaction = await hook(mock_event)

    assert reaction is None
    backend.send.assert_called_once()


async def test_notify_hook_no_match(mock_event: AgentEvent) -> None:
    backend = MagicMock(spec=NotificationBackend)
    backend.send = AsyncMock()

    hook = notify(ToolStart, "Should not send", backend=backend)

    reaction = await hook(mock_event)

    assert reaction is None
    backend.send.assert_not_called()


async def test_notify_hook_handles_backend_failure(mock_event: AgentEvent) -> None:
    from unittest.mock import patch

    backend = MagicMock(spec=NotificationBackend)
    backend.send = AsyncMock(side_effect=Exception("Backend failed"))

    hook = notify(MockEvent, "Test message", backend=backend)

    with patch("dreadnode.agent.hooks.notification.logger.exception") as mock_logger:
        reaction = await hook(mock_event)

    assert reaction is None
    mock_logger.assert_called_once_with("Notification hook failed")

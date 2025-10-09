import asyncio
import random
import time
import typing as t
from dataclasses import dataclass

from loguru import logger

from dreadnode.agent.events import AgentError, AgentEvent, StepStart
from dreadnode.agent.reactions import Reaction, Retry

if t.TYPE_CHECKING:
    from ulid import ULID

    from dreadnode.agent.hooks.base import Hook


@dataclass
class BackoffState:
    tries: int = 0
    start_time: float | None = None
    last_step_seen: int = -1

    def reset(self, step: int = -1) -> None:
        self.tries = 0
        self.start_time = None
        self.last_step_seen = step


def backoff_on_error(
    exception_types: type[Exception] | t.Iterable[type[Exception]],
    *,
    max_tries: int = 8,
    max_time: float = 300.0,
    base_factor: float = 1.0,
    jitter: bool = True,
) -> "Hook":
    """
    Creates a hook that retries with exponential backoff when specific errors occur.

    It listens for `AgentError` events and, if the error matches, waits for an
    exponentially increasing duration before issuing a `Retry` reaction.

    Args:
        exception_types: An exception type or iterable of types to catch.
        max_tries: The maximum number of retries before giving up.
        max_time: The maximum total time in seconds to wait before giving up.
        base_factor: The base duration (in seconds) for the backoff calculation.
        jitter: If True, adds a random jitter to the wait time to prevent synchronized retries.

    Returns:
        An agent hook that implements the backoff logic.
    """
    exceptions = (
        tuple(exception_types) if isinstance(exception_types, t.Iterable) else (exception_types,)
    )

    session_states: dict[ULID, BackoffState] = {}

    async def backoff_hook(event: "AgentEvent") -> "Reaction | None":
        state = session_states.setdefault(event.session_id, BackoffState())

        if isinstance(event, StepStart):
            if event.step > state.last_step_seen:
                state.reset(event.step)
            return None

        if not isinstance(event, AgentError) or not isinstance(event.error, exceptions):
            return None

        if state.start_time is None:
            state.start_time = time.monotonic()

        if state.tries >= max_tries:
            logger.warning(
                f"Backoff aborted for session {event.session_id}: maximum tries ({max_tries}) exceeded."
            )
            return None

        if (time.monotonic() - state.start_time) >= max_time:
            logger.warning(
                f"Backoff aborted for session {event.session_id}: maximum time ({max_time:.2f}s) exceeded."
            )
            return None

        state.tries += 1

        seconds = base_factor * (2 ** (state.tries - 1))
        if jitter:
            seconds += random.uniform(0, base_factor)  # noqa: S311 # nosec

        logger.warning(
            f"Backing off for {seconds:.2f}s (try {state.tries}/{max_tries}) on session {event.session_id} due to error: {event.error}"
        )

        await asyncio.sleep(seconds)
        return Retry()

    return backoff_hook


def backoff_on_ratelimit(
    *,
    max_tries: int = 8,
    max_time: float = 300.0,
    base_factor: float = 1.0,
    jitter: bool = True,
) -> "Hook":
    """
    A convenient default backoff hook for common, ephemeral LLM errors.

    This hook retries on `litellm.exceptions.RateLimitError` and `litellm.exceptions.APIError`
    with an exponential backoff strategy for up to 5 minutes.

    See `backoff_on_error` for more details.

    Args:
        max_tries: The maximum number of retries before giving up.
        max_time: The maximum total time in seconds to wait before giving up.
        base_factor: The base duration (in seconds) for the backoff calculation.
        jitter: If True, adds a random jitter to the wait time to prevent synchronized retries.

    Returns:
        An agent hook that implements the backoff logic.
    """
    import litellm.exceptions

    return backoff_on_error(
        (litellm.exceptions.RateLimitError, litellm.exceptions.APIError),
        max_time=max_time,
        max_tries=max_tries,
        base_factor=base_factor,
        jitter=jitter,
    )

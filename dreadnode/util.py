# Lots of utilities shamelessly copied from the `logfire` package.
# https://github.com/pydantic/logfire

import inspect
import logging
import os
import re
import socket
import sys
import typing as t
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import TracebackType
from urllib.parse import ParseResult, urlparse

from logfire import suppress_instrumentation
from logfire._internal.stack_info import add_non_user_code_prefix, is_user_code
from logfire._internal.stack_info import warn_at_user_stacklevel as _warn_at_user_stacklevel

import dreadnode

warn_at_user_stacklevel = _warn_at_user_stacklevel

SysExcInfo = (
    tuple[type[BaseException], BaseException, TracebackType | None] | tuple[None, None, None]
)
"""
The return type of sys.exc_info(): exc_type, exc_val, exc_tb.
"""

logger = logging.getLogger("dreadnode")

add_non_user_code_prefix(Path(dreadnode.__file__).parent)


def clean_str(string: str, *, max_length: int | None = None) -> str:
    """
    Clean a string by replacing all non-alphanumeric characters (except `/` and `@`) with underscores.
    """
    result = re.sub(r"[^\w/@]+", "_", string.lower()).strip("_")
    if max_length is not None:
        result = result[:max_length]
    return result


def safe_repr(obj: t.Any) -> str:
    """
    Return some kind of non-empty string representation of an object, catching exceptions.
    """

    try:
        result = repr(obj)
    except Exception:  # noqa: BLE001
        result = ""

    if result:
        return result

    try:
        return f"<{type(obj).__name__} object>"
    except Exception:  # noqa: BLE001
        return "<unknown (repr failed)>"


def time_to(future_datetime: datetime) -> str:
    """Get a string describing the time difference between a future datetime and now."""

    now = datetime.now(tz=future_datetime.tzinfo)
    time_difference = future_datetime - now

    days = time_difference.days
    seconds = time_difference.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    result = []
    if days > 0:
        result.append(f"{days}d")
    if hours > 0:
        result.append(f"{hours}hr")
    if minutes > 0:
        result.append(f"{minutes}m")

    return ", ".join(result) if result else "Just now"


def log_internal_error() -> None:
    try:
        current_test = os.environ.get("PYTEST_CURRENT_TEST", "")
        reraise = bool(current_test and "test_internal_exception" not in current_test)
    except Exception:  # noqa: BLE001
        reraise = False

    if reraise:
        raise  # noqa: PLE0704

    with suppress_instrumentation():  # prevent infinite recursion from the logging integration
        logger.exception(
            "Caught an error in Dreadnode. This will not prevent code from running, but you may lose data.",
            exc_info=_internal_error_exc_info(),
        )


def _internal_error_exc_info() -> SysExcInfo:
    """Returns an exc_info tuple with a nicely tweaked traceback."""
    original_exc_info: SysExcInfo = sys.exc_info()
    exc_type, exc_val, original_tb = original_exc_info
    try:
        # First remove redundant frames already in the traceback about where the error was raised.
        tb = original_tb
        if tb and tb.tb_frame and tb.tb_frame.f_code is _HANDLE_INTERNAL_ERRORS_CODE:
            # Skip the 'yield' line in _handle_internal_errors
            tb = tb.tb_next

        if (
            tb
            and tb.tb_frame
            and tb.tb_frame.f_code.co_filename == contextmanager.__code__.co_filename
            and tb.tb_frame.f_code.co_name == "inner"
        ):
            # Skip the 'inner' function frame when handle_internal_errors is used as a decorator.
            # It looks like `return func(*args, **kwds)`
            tb = tb.tb_next

        # Now add useful outer frames that give context, but skipping frames that are just about handling the error.
        frame = inspect.currentframe()
        # Skip this frame right here.
        assert frame  # noqa: S101
        frame = frame.f_back

        if frame and frame.f_code is log_internal_error.__code__:  # pragma: no branch
            # This function is always called from log_internal_error, so skip that frame.
            frame = frame.f_back
            assert frame  # noqa: S101

            if frame.f_code is _HANDLE_INTERNAL_ERRORS_CODE:
                # Skip the line in _handle_internal_errors that calls log_internal_error
                frame = frame.f_back
                # Skip the frame defining the _handle_internal_errors context manager
                assert frame  # noqa: S101
                assert frame.f_code.co_name == "__exit__"  # noqa: S101
                frame = frame.f_back
                assert frame  # noqa: S101
                # Skip the frame calling the context manager, on the `with` line.
                frame = frame.f_back
            else:
                # `log_internal_error()` was called directly, so just skip that frame. No context manager stuff.
                frame = frame.f_back

        # Now add all remaining frames from internal logfire code.
        while frame and not is_user_code(frame.f_code):
            tb = TracebackType(
                tb_next=tb,
                tb_frame=frame,
                tb_lasti=frame.f_lasti,
                tb_lineno=frame.f_lineno,
            )
            frame = frame.f_back

        # Add up to 3 frames from user code.
        for _ in range(3):
            if not frame:  # pragma: no cover
                break
            tb = TracebackType(
                tb_next=tb,
                tb_frame=frame,
                tb_lasti=frame.f_lasti,
                tb_lineno=frame.f_lineno,
            )
            frame = frame.f_back

        assert exc_type  # noqa: S101
        assert exc_val  # noqa: S101
        exc_val = exc_val.with_traceback(tb)
        return exc_type, exc_val, tb  # noqa: TRY300
    except Exception:  # noqa: BLE001
        return original_exc_info


@contextmanager
def handle_internal_errors() -> t.Iterator[None]:
    try:
        yield
    except Exception:  # noqa: BLE001
        log_internal_error()


_HANDLE_INTERNAL_ERRORS_CODE = inspect.unwrap(handle_internal_errors).__code__


def is_docker_service_name(hostname: str) -> bool:
    """Check if this looks like a Docker service name

    Args:
        hostname: The hostname to check.

    Returns:
        bool: True if the hostname looks like a Docker service name, False otherwise.
    """
    return bool(hostname and "." not in hostname and hostname != "localhost")


def resolve_endpoint(endpoint: str | None) -> str | None:
    """Automatically resolve endpoints based on environment

    Args:
        endpoint: The endpoint URL to resolve.

    Returns:
        str: The resolved endpoint URL.

    Raises:
        ValueError: If the endpoint URL is invalid.
    """
    if not endpoint:
        return None
    parsed = urlparse(endpoint)

    # If it's a real domain (has dots), use as-is
    if not parsed.hostname:
        raise ValueError(f"Invalid endpoint URL: {endpoint}")

    if "." in parsed.hostname:
        return endpoint

    # If it's a service name, try to resolve it
    if is_docker_service_name(parsed.hostname):
        return resolve_docker_service(endpoint, parsed)

    return endpoint


def test_connection(endpoint: str) -> bool:
    """
    Simple test to check if the endpoint is reachable.

    Args:
        endpoint: The endpoint URL to test.

    Returns:
        bool: True if the endpoint is reachable, False otherwise.
    """
    try:
        parsed = urlparse(endpoint)
        socket.create_connection((parsed.hostname, parsed.port or 443), timeout=1)
    except Exception:  # noqa: BLE001
        return False

    return True


def resolve_docker_service(original_endpoint: str, parsed: ParseResult) -> str:
    """
    Try different resolution strategies for Docker services

    Args:
        original_endpoint: The original endpoint URL.
        parsed: The parsed URL object.

    Returns:
        str: The resolved endpoint URL.

    Raises:
        RuntimeError: If no valid endpoint is found.
    """
    strategies = [
        original_endpoint,  # Try original first (works if running in same network)
        f"{parsed.scheme}://localhost:{parsed.port}",  # Try localhost
        f"{parsed.scheme}://host.docker.internal:{parsed.port}",  # Docker Desktop
        f"{parsed.scheme}://172.17.0.1:{parsed.port}",  # Docker bridge IP
    ]

    for endpoint in strategies:
        if test_connection(endpoint):
            logger.warning(
                f"Resolved Docker service endpoint '{parsed.hostname}' to '{endpoint}'."  # noqa: G004
            )
            return str(endpoint)

    # If nothing works, return original and let it fail with a helpful error
    raise RuntimeError(f"Failed to connect to the Dreadnode Artifact storage at {endpoint}.")

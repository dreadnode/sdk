import inspect
import os
import sys
import typing as t
from contextlib import contextmanager
from types import TracebackType

from loguru import logger

from dreadnode.core.util import is_user_code

if t.TYPE_CHECKING:
    from dreadnode.core.metric import Metric


T = t.TypeVar("T")

SysExcInfo = (
    tuple[type[BaseException], BaseException, TracebackType | None] | tuple[None, None, None]
)


class DreadnodeUserError(Exception):
    """Raised when the SDK is used incorrectly."""


class ContextRequiredError(DreadnodeUserError):
    """Raised when a method requires an active run/task context."""

    def __init__(self, method: str):
        super().__init__(
            f"{method}() must be called within a run or task context. "
            f"Wrap your code with 'with dreadnode.run(...):'."
        )


class DreadnodeConfigWarning(UserWarning):
    """Warnings related to Dreadnode configuration."""


class DreadnodeUsageWarning(UserWarning):
    """Warnings related to Dreadnode usage."""


class AssertionFailedError(Exception):
    """Raised when a task's output fails one or more assertions."""

    def __init__(self, message: str, failures: "dict[str, list[Metric]]"):
        """
        Args:
            message: The overall exception message.
            failures: A dictionary mapping the name of each failed assertion
                      to the list of Metrics that caused the failure
                      (or an empty list if the metric was not produced).
        """
        super().__init__(message)
        self.failures = failures


@contextmanager
def suppress_instrumentation():
    yield


@contextmanager
def catch_import_error(install_suggestion: str | None = None) -> t.Iterator[None]:
    """
    Context manager to catch ImportError and raise a new ImportError with a custom message.

    Args:
        install_suggestion: The package suggestion to include in the error message.
    """
    try:
        yield
    except ImportError as e:
        message = f"Missing required package `{e.name}`."
        if install_suggestion:
            message += f" Install with: pip install {install_suggestion}"
        raise ImportError(message) from e


def safe_issubclass(cls: t.Any, class_or_tuple: T) -> t.TypeGuard[T]:
    """Safely check if a class is a subclass of another class or tuple."""
    try:
        return isinstance(cls, type) and issubclass(cls, class_or_tuple)  # type: ignore[arg-type]
    except TypeError:
        return False


# Resolution


# Logging
#
# Lots of utilities shamelessly copied from the `logfire` package.
# https://github.com/pydantic/logfire


def log_internal_error() -> None:
    """
    Log an internal error with a detailed traceback.
    """
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
    """
    Returns an exc_info tuple with a nicely tweaked traceback.
    """
    original_exc_info: SysExcInfo = sys.exc_info()
    exc_type, exc_val, original_tb = original_exc_info
    try:
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


def warn_at_user_stacklevel(msg: str, category: type[Warning]) -> None:
    """
    Issue a warning at the user code stack level and log it.

    Args:
        msg: The warning message.
        category: The warning category.
    """
    logger.warning(msg)
    warn_at_user_stacklevel(msg, category)


@contextmanager
def handle_internal_errors() -> t.Iterator[None]:
    try:
        yield
    except DreadnodeUserError:
        raise  # Let user errors propagate
    except Exception:  # noqa: BLE001
        log_internal_error()


_HANDLE_INTERNAL_ERRORS_CODE = inspect.unwrap(handle_internal_errors).__code__

import asyncio
import contextlib
import functools
import inspect
import os
import re
import socket
import sys
import typing as t
from contextlib import aclosing, asynccontextmanager, contextmanager
from datetime import datetime
from pathlib import Path
from types import TracebackType
from urllib.parse import ParseResult, urlparse

import typing_extensions as te
from logfire import suppress_instrumentation
from logfire._internal.stack_info import (
    add_non_user_code_prefix,
)
from logfire._internal.stack_info import (
    get_filepath_attribute as _get_filepath_attribute,
)
from logfire._internal.stack_info import (
    get_user_frame_and_stacklevel as _get_user_frame_and_stacklevel,
)
from logfire._internal.stack_info import (
    is_user_code as _is_user_code,
)
from logfire._internal.stack_info import (
    warn_at_user_stacklevel as _warn_at_user_stacklevel,
)
from loguru import logger

get_user_frame_and_stacklevel = _get_user_frame_and_stacklevel
get_filepath_attribute = _get_filepath_attribute
is_user_code = _is_user_code


import dreadnode  # noqa: E402

SysExcInfo = (
    tuple[type[BaseException], BaseException, TracebackType | None] | tuple[None, None, None]
)
"""
The return type of sys.exc_info(): exc_type, exc_val, exc_tb.
"""

add_non_user_code_prefix(Path(dreadnode.__file__).parent)

T = t.TypeVar("T")
T_in = t.TypeVar("T_in")
T_out = t.TypeVar("T_out")


# Formatting


def shorten_string(
    text: str,
    max_length: int | None = None,
    *,
    max_lines: int | None = None,
    separator: str = "...",
) -> str:
    """
    Shortens text to a maximum number of lines and/or characters by removing
    content from the middle.

    Line shortening is applied first, followed by character shortening.

    Args:
        text: The string to shorten.
        max_lines: The maximum number of lines to allow.
        max_chars: The maximum number of characters to allow.
        separator: The separator to insert in the middle of the shortened text.

    Returns:
        The shortened text
    """
    # 1 - line count first
    if max_lines is not None:
        lines = text.splitlines()
        if len(lines) > max_lines:
            remaining_lines = max_lines - 1  # leave space for the separator
            if remaining_lines <= 0:
                text = separator  # if max_lines is 1, just use the separator
            else:
                half = remaining_lines // 2
                start_lines = lines[:half]
                end_lines = lines[-(remaining_lines - half) :]
                text = "\n".join([*start_lines, separator, *end_lines])

    # 2 - character count
    if max_length is not None and len(text) > max_length:
        remaining_chars = max_length - len(separator)
        if remaining_chars <= 0:
            text = separator
        else:
            half_chars = remaining_chars // 2
            text = text[:half_chars] + separator + text[-half_chars:]

    return text


def truncate_string(text: str, max_length: int = 80, *, suf: str = "...") -> str:
    """
    Return a string at most max_length characters long by removing the end of the string.
    """
    if len(text) <= max_length:
        return text

    remaining = max_length - len(suf)
    if remaining <= 0:
        return suf

    return text[:remaining] + suf


def clean_str(string: str, *, max_length: int | None = None, replace_with: str = "_") -> str:
    """
    Clean a string by replacing all non-alphanumeric characters (except `/`, '.', and `@`)
    with `replace_with` (`_` by default).
    """
    result = re.sub(r"[^a-z0-9/@.]+", replace_with, string.lower()).strip(replace_with)
    if max_length is not None:
        result = result[:max_length]
    return result


def format_dict(data: dict[str, t.Any], max_length: int = 80) -> str:
    """
    Formats a dictionary to a string, prioritizing showing key-value pairs
    and truncating gracefully if the string exceeds a max length.
    """
    parts: list[str] = []
    items = list(data.items())
    max_length = max_length - 2  # Account for the surrounding braces

    for i, (key, value) in enumerate(items):
        part_str = f"{key}={value!r}"
        potential_parts = [*parts, part_str]

        # Check if adding the next full part would exceed the length
        if len(", ".join(potential_parts)) > max_length:
            num_remaining = len(items) - i
            parts.append(f"... (+{num_remaining} more)")
        else:
            parts.append(part_str)

    formatted = ", ".join(parts)
    return f"{{{formatted}}}"


def create_key_from_name(name: str) -> str:
    key = name.strip().lower()

    # 2. Replace one or more spaces or underscores with a single hyphen
    key = re.sub(r"[\s_]+", "-", key)

    # 3. Remove any character that is not a letter, number, or hyphen
    return re.sub(r"[^a-z0-9-]", "", key)


def valid_key(key: str) -> bool:
    """
    Check if the key is valid (only contains lowercase letters, numbers, and hyphens).
    """
    return bool(re.fullmatch(r"[a-z0-9-]+", key))


# Imports


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


# Types


def safe_issubclass(cls: t.Any, class_or_tuple: T) -> t.TypeGuard[T]:
    """Safely check if a class is a subclass of another class or tuple."""
    try:
        return isinstance(cls, type) and issubclass(cls, class_or_tuple)  # type: ignore[arg-type]
    except TypeError:
        return False


# Resolution


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


def get_obj_name(obj: t.Any, *, short: bool = False, clean: bool = False) -> str:
    """
    Return a best effort name for an object.
    """
    name = "unknown"
    if hasattr(obj, "name"):
        name = obj.name
    elif hasattr(obj, "__name__"):
        name = obj.__name__
    elif hasattr(obj.__class__, "__name__"):
        name = obj.__class__.__name__

    if short:
        name = name.split(".")[-1]

    if clean:
        name = clean_str(name)

    return name


def get_callable_name(obj: t.Any, *, short: bool = False) -> str:
    """
    Return a best-effort, comprehensive name for a callable object.

    This function handles a wide variety of callables, including regular
    functions, methods, lambdas, partials, wrapped functions, and callable
    class instances.

    Args:
        obj: The callable object to name.
        short: If True, returns a shorter name suitable for logs or UI,
               typically omitting the module path. The class name is
               retained for methods.

    Returns:
        A string representing the callable's name.
    """
    if not callable(obj):
        return safe_repr(obj)

    if isinstance(obj, functools.partial):
        inner_name = get_callable_name(obj.func, short=short)
        return f"partial({inner_name})"

    unwrapped = obj
    with contextlib.suppress(Exception):
        unwrapped = inspect.unwrap(obj)

    if short:
        name = getattr(unwrapped, "__name__", getattr(unwrapped, "__qualname__", None))
    else:
        name = getattr(unwrapped, "__qualname__", getattr(unwrapped, "__name__", None))

    if name is None:
        if hasattr(obj, "__class__"):
            name = getattr(obj.__class__, "__qualname__", obj.__class__.__name__)
        else:
            return safe_repr(obj)

    if short:
        return str(name).split(".")[-1]  # Return only the last part of the name

    with contextlib.suppress(Exception):
        if module := inspect.getmodule(unwrapped):
            module_name = module.__name__
            if module_name and module_name not in ("builtins", "__main__"):
                return f"{module_name}.{name}"

    return str(name)


# Time


def time_to(future_datetime: datetime) -> str:
    """
    Get a string describing the time difference between a future datetime and now.
    """
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


# Async


async def concurrent(coros: t.Iterable[t.Awaitable[T]], limit: int | None = None) -> list[T]:
    """
    Run multiple coroutines concurrently with a limit on the number of concurrent tasks.

    Args:
        coros: An iterable of coroutines to run concurrently.
        limit: The maximum number of concurrent tasks. If None, no limit is applied.

    Returns:
        A list of results from the coroutines, in the order they were provided.
    """
    coros = list(coros)
    semaphore = asyncio.Semaphore(limit or len(coros))

    async def run_coroutine_with_semaphore(
        coro: t.Awaitable[T],
    ) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(
        *(run_coroutine_with_semaphore(coro) for coro in coros),
    )


# Some weirdness here: https://discuss.python.org/t/overloads-of-async-generators-inconsistent-coroutine-wrapping/56665/2


@t.overload
@contextlib.asynccontextmanager
def concurrent_gen(
    coros: t.Iterable[t.Awaitable[T]],
    limit: int | None = None,
    *,
    return_task: t.Literal[False] = False,
) -> t.AsyncIterator[t.AsyncGenerator[T, None]]: ...


@t.overload
@contextlib.asynccontextmanager
def concurrent_gen(
    coros: t.Iterable[t.Awaitable[T]],
    limit: int | None = None,
    *,
    return_task: t.Literal[True],
) -> t.AsyncIterator[t.AsyncGenerator[asyncio.Task[T], None]]: ...


@contextlib.asynccontextmanager
async def concurrent_gen(
    coros: t.Iterable[t.Awaitable[T]],
    limit: int | None = None,
    *,
    return_task: bool = False,
) -> t.AsyncIterator[t.AsyncGenerator[T | asyncio.Task[T], None]]:
    """
    Run multiple coroutines concurrently with a limit on the number of concurrent tasks.

    Args:
        coros: An iterable of coroutines to run concurrently.
        limit: The maximum number of concurrent tasks. If None, no limit is applied.
        return_task: If True, yields the asyncio.Task object instead of the result.

    Yields:
        An asynchronous generator yielding the results of the coroutines.
        If return_task is True, yields the asyncio.Task objects instead.
    """
    coros = list(coros)
    semaphore = asyncio.Semaphore(limit or len(coros))

    async def run_coroutine_with_semaphore(coro: t.Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    async def generator() -> t.AsyncGenerator[T | asyncio.Task[T], None]:
        pending_tasks = {asyncio.create_task(run_coroutine_with_semaphore(coro)) for coro in coros}

        try:
            while pending_tasks:
                done, pending_tasks = await asyncio.wait(
                    pending_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    yield task if return_task else await task
        finally:
            for task in pending_tasks:
                task.cancel()
            await asyncio.gather(*pending_tasks, return_exceptions=True)

    async with aclosing(generator()) as gen:
        yield gen


async def join_generators(
    *generators: t.AsyncGenerator[T, None],
) -> t.AsyncGenerator[T, None]:
    """
    Join multiple asynchronous generators into a single asynchronous generator.

    If any of the generators raise an exception, the other generators will be
    cancelled immediately and the exception will be raised to the caller.

    Args:
        *generators: The asynchronous generators to join.

    Yields:
        The items yielded by the joined generators.

    Raises:
        Exception: If any of the generators raises an exception.
    """
    FINISHED = object()  # sentinel object to indicate a generator has finished  # noqa: N806
    queue = asyncio.Queue[T | object | Exception](maxsize=1)

    async def _queue_generator(
        generator: t.AsyncGenerator[T, None],
    ) -> None:
        try:
            async with aclosing(generator) as gen:
                async for item in gen:
                    await queue.put(item)
        except Exception as e:  # noqa: BLE001
            await queue.put(e)
        finally:
            await queue.put(FINISHED)

    tasks = [asyncio.create_task(_queue_generator(gen)) for gen in generators]

    finished_count = 0

    try:
        while finished_count < len(generators):
            item = await queue.get()

            if isinstance(item, Exception):
                raise item

            if item is FINISHED:
                finished_count += 1
                continue

            yield t.cast("T", item)

    finally:
        for task in tasks:
            if not task.done():
                task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)


@asynccontextmanager
async def stream_map_and_merge(  # noqa: PLR0915
    source: t.AsyncGenerator[T_in, None],
    processor: t.Callable[[T_in], t.AsyncGenerator[T_out, None]],
    *,
    limit: int | None = None,
    concurrency: int | None = None,
    source_queue_size: int = 1,
    out_queue_size: int = 1,
) -> t.AsyncIterator[t.AsyncGenerator[T_out, None]]:
    """
    The "one-to-many-to-one" abstraction helpful for concurrently processing
    items from a stream using workers which themselves yield items.

    It consumes items from a source stream and creates processing workers
    which concurrently yield items, then merges them into a single interleaved stream.

    Note: As an oddity for supporting better syntax in our most common Study/Search
    use case, the values yielded from the source generator will be sent back into
    the source generator as the value of the next `asend()` call. This is a syntax
    convenience to allow code like `item = await (yield Item(...))`.

    Args:
        source: The source stream of items to process.
        processor: A function that processes each item from the source and
            returns an asynchronous generator of results.
        limit: Maximum number of items to consume from the source before early stopping.
        concurrency: The maximum number of items to process concurrently.
            If None, there will be no limit on workers.
        in_queue_size: The maximum number of items to buffer from the source.
            This limits how far ahead the source can get before backpressure is applied.
        out_queue_size: The maximum number of processed items to buffer for output.
            This limits how far ahead the workers can get before backpressure is applied.

    Yields:
        An asynchronous generator which yields the processed items from the processor streams.
    """

    # In general, this function is quite complex, mainly because we
    # need to properly manage downstream and upstream backpressure,
    # as well as error handling, cancellation, and lifecycle management.
    #
    # In short, it looks a mess, but not for lack of trying

    in_queue = asyncio.Queue[T_in](maxsize=source_queue_size)
    out_queue = asyncio.Queue[T_out](maxsize=out_queue_size)
    error_queue = asyncio.Queue[Exception]()

    feeding_done = asyncio.Event()
    workers_done = asyncio.Event()

    # Workers create the processor stream for a source item
    # and push results to the out_queue

    async def worker(inner_item: T_in) -> None:
        try:
            async with aclosing(processor(inner_item)) as results:
                async for result in results:
                    while True:
                        with contextlib.suppress(asyncio.QueueFull):
                            out_queue.put_nowait(result)
                            break
                        await asyncio.sleep(0)
        except Exception as e:  # noqa: BLE001
            error_queue.put_nowait(e)

    # The feeder pulls from the source generator
    # limited by the in_queue size

    async def feeder() -> None:
        try:
            async with aclosing(source) as stream:
                source_count = 0
                async for item in stream:
                    source_count += 1

                    if limit is not None and source_count > limit:
                        break

                    while True:
                        with contextlib.suppress(asyncio.QueueFull):
                            in_queue.put_nowait(item)
                            break
                        await asyncio.sleep(0)

        except Exception as e:  # noqa: BLE001
            error_queue.put_nowait(e)
        finally:
            feeding_done.set()

    # The worker_pool starts the feed, watches the in_queue,
    # and starts worker tasks for every item yielded from source,
    # respecting the concurrency limit.

    async def worker_pool() -> None:
        workers: set[asyncio.Task[None]] = set()
        feed_task = asyncio.create_task(feeder())

        try:
            while not (feeding_done.is_set() and in_queue.empty()):
                try:
                    item = await asyncio.wait_for(in_queue.get(), timeout=0.01)
                except asyncio.TimeoutError:
                    continue

                # Manage concurrency
                while concurrency and len(workers) >= concurrency:
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait(
                            workers, timeout=0.01, return_when=asyncio.FIRST_COMPLETED
                        )

                # Start new worker
                worker_task = asyncio.create_task(worker(item))
                workers.add(worker_task)
                worker_task.add_done_callback(workers.discard)

            if workers:
                await asyncio.gather(*workers)
        except Exception as e:  # noqa: BLE001
            error_queue.put_nowait(e)
        finally:
            feed_task.cancel()
            for task in workers:
                task.cancel()
            await asyncio.gather(feed_task, *workers, return_exceptions=True)
            workers_done.set()

    async def generator() -> t.AsyncGenerator[T_out, None]:
        worker_pool_task = asyncio.create_task(worker_pool())

        try:
            while not (feeding_done.is_set() and workers_done.is_set() and out_queue.empty()):
                with contextlib.suppress(asyncio.QueueEmpty):
                    error = error_queue.get_nowait()
                    raise error

                with contextlib.suppress(asyncio.QueueEmpty):
                    yield out_queue.get_nowait()

                await asyncio.sleep(0)
        finally:
            worker_pool_task.cancel()
            await asyncio.gather(worker_pool_task, return_exceptions=True)

    async with aclosing(generator()) as gen:
        yield gen


# List utilities


@t.overload
def flatten_list(nested_list: t.Sequence[t.Sequence[t.Sequence[T] | T]]) -> list[T]: ...


@t.overload
def flatten_list(nested_list: t.Sequence[t.Sequence[T] | T]) -> list[T]: ...


def flatten_list(nested_list: t.Sequence[t.Any]) -> list[t.Any]:
    """
    Recursively flatten a nested list into a single list.
    """
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened


def is_homogeneous_list(obj: t.Any, item_type: type[T]) -> te.TypeIs[list[T]]:
    """Type guard to check if an object is a homogeneous list of the specified type."""
    return isinstance(obj, list) and all(isinstance(item, item_type) for item in obj)


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


def warn_at_user_stacklevel(msg: str, category: type[Warning]) -> None:
    """
    Issue a warning at the user code stack level and log it.

    Args:
        msg: The warning message.
        category: The warning category.
    """
    logger.warning(msg)
    _warn_at_user_stacklevel(msg, category)


@contextmanager
def handle_internal_errors() -> t.Iterator[None]:
    """
    Context manager to handle internal errors.
    """
    try:
        yield
    except Exception:  # noqa: BLE001
        log_internal_error()


_HANDLE_INTERNAL_ERRORS_CODE = inspect.unwrap(handle_internal_errors).__code__

# Endpoint and networking


def is_docker_service_name(hostname: str) -> bool:
    """
    Check if this looks like a Docker service name

    Args:
        hostname: The hostname to check.

    Returns:
        bool: True if the hostname looks like a Docker service name, False otherwise.
    """
    return bool(hostname and "." not in hostname and hostname != "localhost")


def resolve_endpoint(endpoint: str | None) -> str | None:
    """
    Automatically resolve endpoints based on environment

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
            logger.warning(f"Resolved Docker service endpoint '{parsed.hostname}' to '{endpoint}'.")
            return str(endpoint)

    # If nothing works, return original and let it fail with a helpful error
    raise RuntimeError(
        f"Failed to connect to the Dreadnode Artifact storage at {original_endpoint}."
    )

import contextlib
import typing as t
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager

import typing_extensions as te
from loguru import logger
from pydantic import ConfigDict, Field

from dreadnode.core.meta import Config, Model
from dreadnode.core.meta.introspect import (
    get_config_model,
    get_inputs_and_params_from_config_model,
)
from dreadnode.core.types.common import AnyDict

if t.TYPE_CHECKING:
    import asyncio

# Type variables for generic executor
EventT = t.TypeVar("EventT", bound="ExecutionEvent")
ResultT = t.TypeVar("ResultT")
InputT = t.TypeVar("InputT")


class ExecutionEvent(Model):
    """Base class for all execution events."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: t.Literal["pending", "running", "finished", "errored", "stalled"] = "running"


class StartEvent(ExecutionEvent):
    """Base class for execution start events."""

    status: t.Literal["pending", "running", "finished", "errored", "stalled"] = "running"
    inputs: AnyDict = Field(default_factory=dict)
    params: AnyDict = Field(default_factory=dict)


class EndEvent(ExecutionEvent, t.Generic[ResultT]):
    """Base class for execution end events."""

    status: t.Literal["pending", "running", "finished", "errored", "stalled"] = "finished"
    result: ResultT | None = None
    error: Exception | str | None = None
    stop_reason: str = "finished"


class StepEvent(ExecutionEvent):
    """Base class for intermediate step events."""

    step: int = 0


class ErrorEvent(ExecutionEvent):
    """Base class for error events."""

    status: t.Literal["pending", "running", "finished", "errored", "stalled"] = "errored"
    error: Exception | str | None = None


class ErrorThresholds(Model):
    """
    Configuration for error thresholds during execution.

    Use with ErrorTracker to implement consistent error handling across executors.
    """

    max_errors: int | None = Config(default=None)
    """Maximum total errors before stopping. None means no limit."""

    max_consecutive_errors: int | None = Config(default=10)
    """Maximum consecutive errors before stopping. None means no limit."""

    def create_tracker(self) -> "ErrorTracker":
        """Create an ErrorTracker instance with these thresholds."""
        return ErrorTracker(self)


class ErrorTracker:
    """
    Tracks errors during execution and determines when thresholds are exceeded.

    Usage:
        thresholds = ErrorThresholds(max_errors=10, max_consecutive_errors=3)
        tracker = thresholds.create_tracker()

        for item in items:
            try:
                process(item)
                tracker.record_success()
            except Exception as e:
                if stop_reason := tracker.record_error():
                    logger.warning(f"Stopping: {stop_reason}")
                    break
    """

    def __init__(self, thresholds: ErrorThresholds) -> None:
        self.thresholds = thresholds
        self.total_errors = 0
        self.consecutive_errors = 0

    def record_success(self) -> None:
        """Record a successful operation, resetting consecutive error count."""
        self.consecutive_errors = 0

    def record_error(self) -> str | None:
        """
        Record an error and check if thresholds are exceeded.

        Returns:
            Stop reason string if a threshold was exceeded, None otherwise.
        """
        self.total_errors += 1
        self.consecutive_errors += 1

        if (
            self.thresholds.max_errors is not None
            and self.total_errors >= self.thresholds.max_errors
        ):
            return "max_errors_reached"

        if (
            self.thresholds.max_consecutive_errors is not None
            and self.consecutive_errors >= self.thresholds.max_consecutive_errors
        ):
            return "max_consecutive_errors_reached"

        return None

    @property
    def should_stop(self) -> bool:
        """Check if any threshold has been exceeded."""
        if (
            self.thresholds.max_errors is not None
            and self.total_errors >= self.thresholds.max_errors
        ):
            return True
        return (
            self.thresholds.max_consecutive_errors is not None
            and self.consecutive_errors >= self.thresholds.max_consecutive_errors
        )


class TraceContext(Model):
    """
    Configuration for tracing/observability during execution.

    Collects inputs and parameters for logging and provides a consistent
    interface for entering trace spans.
    """

    name: str
    tags: list[str] = Field(default_factory=list)
    label: str | None = None
    inputs: AnyDict = Field(default_factory=dict)
    params: AnyDict = Field(default_factory=dict)
    span_type: str = "task"

    @classmethod
    def from_executor(
        cls,
        executor: "Executor[t.Any, t.Any]",
        *,
        extra_inputs: AnyDict | None = None,
        extra_params: AnyDict | None = None,
        exclude: set[str] | None = None,
    ) -> "TraceContext":
        """
        Create a TraceContext from an executor instance.

        Automatically extracts configuration model inputs/params.
        """
        configuration = get_config_model(executor)()
        inputs, params = get_inputs_and_params_from_config_model(
            configuration, exclude=exclude or set()
        )

        if extra_inputs:
            inputs.update(extra_inputs)
        if extra_params:
            params.update(extra_params)

        return cls(
            name=executor.name,
            tags=list(executor.tags),
            label=executor.label,
            inputs=inputs,
            params=params,
        )

    @contextlib.contextmanager
    def span(self) -> t.Iterator[t.Any]:
        """
        Enter a tracing span with the configured context.

        Yields the task context (or None if tracing is not available).
        """
        try:
            from dreadnode import task_and_run

            with task_and_run(
                name=self.name,
                task_type=self.span_type,  # type: ignore[arg-type]
                tags=self.tags,
                label=self.label,
                inputs=self.inputs,
                params=self.params,
            ) as task:
                yield task
        except ImportError:
            yield None


class Executor(Model, ABC, t.Generic[EventT, ResultT]):
    """
    Abstract base class for streaming executors.

    Attributes:
        name: Name of the executor.
        description: Brief description of the executor.
        tags: Tags for categorization and tracing.
        label: Specific label for tracing, otherwise derived from name.

    Provides a consistent pattern for:
    - Streaming execution via async generators
    - Tracing integration
    - Cloning and configuration
    - Running to completion

    Subclasses must implement:
    - _stream(): Core execution logic yielding events
    - _extract_result(): Extract final result from events
    - _get_trace_context(): Provide tracing configuration

    Example:
        class MyExecutor(Executor[MyEvent, MyResult]):
            async def _stream(self) -> AsyncGenerator[MyEvent, None]:
                yield MyStartEvent()
                # ... do work ...
                yield MyEndEvent(result=result)

            def _extract_result(self, event: MyEvent) -> MyResult | None:
                if isinstance(event, MyEndEvent):
                    return event.result
                return None
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    name: str
    description: str = ""
    tags: list[str] = Config(default_factory=list)
    label: str | None = Config(default=None)

    @abstractmethod
    async def _stream(self) -> t.AsyncGenerator[EventT, None]:
        """
        Core execution logic yielding events.

        This is the main execution method that subclasses must implement.
        It should yield events as execution progresses.

        Yields:
            Events describing execution progress.
        """
        yield

    @abstractmethod
    def _extract_result(self, event: EventT) -> ResultT | None:
        """
        Extract the final result from an event.

        Called for each event during run() to determine if execution is complete.

        Args:
            event: The event to check.

        Returns:
            The result if this event represents completion, None otherwise.
        """
        ...  # pragma: no cover

    def _get_trace_context(self) -> TraceContext:
        """
        Get the trace context for this executor.

        Override to customize tracing inputs/params.
        """
        return TraceContext.from_executor(self)

    def _should_trace(self) -> bool:
        """
        Determine if tracing should be enabled.

        Override to conditionally disable tracing.
        """
        return True

    async def _stream_traced(self) -> t.AsyncGenerator[EventT, None]:
        """
        Wrap _stream with tracing context.

        This method handles:
        - Setting up trace spans
        - Logging start/end
        - Capturing metrics
        """
        if not self._should_trace():
            async with contextlib.aclosing(self._stream()) as stream:
                async for event in stream:
                    yield event
            return

        trace_context = self._get_trace_context()

        logger.info(f"Starting {self.__class__.__name__} '{self.name}': tags={self.tags}")

        with trace_context.span():
            async with contextlib.aclosing(self._stream()) as stream:
                async for event in stream:
                    yield event

        logger.info(f"Finished {self.__class__.__name__} '{self.name}'")

    @asynccontextmanager
    async def stream(self) -> t.AsyncIterator[t.AsyncGenerator[EventT, None]]:
        """
        Create an async context manager for the event stream.

        This provides a safe way to access the event stream with proper
        resource cleanup.

        Usage:
            async with executor.stream() as event_stream:
                async for event in event_stream:
                    process(event)

        Yields:
            An async generator producing events.
        """
        async with contextlib.aclosing(self._stream_traced()) as stream:
            yield stream

    async def run(self) -> ResultT:
        """
        Execute to completion and return the final result.

        This is a convenience method that consumes the entire event stream
        and returns the final result.

        Returns:
            The execution result.

        Raises:
            RuntimeError: If execution completes without producing a result.
        """
        async with self.stream() as stream:
            async for event in stream:
                if (result := self._extract_result(event)) is not None:
                    return result

        raise RuntimeError(f"{self.__class__.__name__} '{self.name}' failed to complete")

    def clone(self) -> te.Self:
        """Create a deep copy of this executor."""
        return self.model_copy(deep=True)

    def _apply_updates(
        self,
        updates: dict[str, t.Any],
        list_fields: set[str],
        *,
        append: bool = False,
    ) -> None:
        """
        Apply updates to this instance, handling list fields specially.

        Args:
            updates: Dictionary of field names to new values.
            list_fields: Set of field names that should support append mode.
            append: If True, list fields are appended rather than replaced.
        """
        for field, value in updates.items():
            if value is None:
                continue

            if append and field in list_fields:
                current = getattr(self, field, []) or []
                setattr(self, field, [*current, *value])
            else:
                setattr(self, field, value)


class TrackedExecutor(Executor[EventT, ResultT], ABC):
    """
    Executor with built-in error tracking.

    Extends Executor with error threshold configuration and tracking.
    """

    max_errors: int | None = Config(default=None)
    """Maximum total errors before stopping."""

    max_consecutive_errors: int | None = Config(default=10)
    """Maximum consecutive errors before stopping."""

    def _create_error_tracker(self) -> ErrorTracker:
        """Create an error tracker with this executor's thresholds."""
        thresholds = ErrorThresholds(
            max_errors=self.max_errors,
            max_consecutive_errors=self.max_consecutive_errors,
        )
        return thresholds.create_tracker()


class ConcurrentExecutor(TrackedExecutor[EventT, ResultT], ABC):
    """
    Executor with concurrency control and error tracking.

    Extends TrackedExecutor with semaphore-based concurrency limiting.
    """

    concurrency: int = Config(default=1, ge=1)
    """Maximum number of concurrent operations."""

    def _create_semaphore(self) -> "asyncio.Semaphore":
        """Create a semaphore for concurrency control."""
        import asyncio

        return asyncio.Semaphore(self.concurrency)


# =============================================================================
# Console Adapter Protocol
# =============================================================================


class ConsoleAdapter(t.Protocol[EventT, ResultT]):
    """Protocol for console display adapters."""

    def __init__(self, executor: Executor[EventT, ResultT]) -> None: ...

    async def run(self) -> ResultT: ...

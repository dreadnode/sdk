import asyncio
import inspect
import typing as t
from dataclasses import dataclass

from opentelemetry.trace import Tracer

from .score import Scorer, ScorerCallable # Assuming correct relative import
from .tracing import TaskSpan, current_run_span # Assuming correct relative import

P = t.ParamSpec("P")
R = t.TypeVar("R")


class TaskSpanList(list[TaskSpan[R]]):
    """A specialized list for holding TaskSpan objects.

    Provides convenience methods for sorting and retrieving top-scoring spans
    or their outputs. Inherits from the standard Python list.
    """
    def sorted(self, *, reverse: bool = True) -> "TaskSpanList[R]":
        """Sorts the list of TaskSpans based on their average score.

        Args:
            reverse: If True (default), sorts in descending order (highest score first).
                     If False, sorts in ascending order.

        Returns:
            A new TaskSpanList instance containing the sorted TaskSpans.
        """
        # Assumes TaskSpan has an 'average_score' attribute or property
        return TaskSpanList(sorted(self, key=lambda span: span.average_score, reverse=reverse))

    @t.overload
    def top_n(self, n: int, *, as_outputs: t.Literal[False] = False, reverse: bool = True) -> "TaskSpanList[R]":
        ...

    @t.overload
    def top_n(self, n: int, *, as_outputs: t.Literal[True], reverse: bool = True) -> list[R]:
        ...

    def top_n(self, n: int, *, as_outputs: bool = False, reverse: bool = True) -> "TaskSpanList[R] | list[R]":
        """Retrieves the top N TaskSpans based on score, optionally returning only their outputs.

        Sorts the spans using the `sorted` method and takes the top N elements.

        Args:
            n: The number of top spans or outputs to retrieve.
            as_outputs: If True, returns a list of the output values (type R) from the
                        top N spans. If False (default), returns a TaskSpanList containing
                        the top N TaskSpan objects.
            reverse: If True (default), retrieves the highest scoring spans. If False,
                     retrieves the lowest scoring spans.

        Returns:
            Either a TaskSpanList of the top N spans or a list of their outputs (type R),
            depending on the `as_outputs` argument.
        """
        sorted_spans = self.sorted(reverse=reverse)[:n]
        if as_outputs:
            # Assuming TaskSpan has an 'output' attribute of type R
            return t.cast(list[R], [span.output for span in sorted_spans])
        else:
            return TaskSpanList(sorted_spans)


@dataclass
class Task(t.Generic[P, R]):
    """Represents a traceable and scorable asynchronous operation (task).

    Wraps an asynchronous function, adding capabilities for tracing its execution
    as an OpenTelemetry span, automatically binding arguments, applying scorers
    to its output, and managing associated metadata like attributes and tags.

    Provides methods for running the task individually or multiple times (`map`),
    retrieving top-scoring results, and handling potential exceptions during execution.

    Attributes:
        tracer: An OpenTelemetry Tracer instance used for creating spans.
        name: The name assigned to this task, used for span identification.
        attributes: A dictionary of key-value attributes to attach to the task's span.
        func: The underlying asynchronous function (Callable[P, Awaitable[R]]) that
              this task wraps.
        scorers: A list of Scorer objects that will be applied to the output (R)
                 of the task function.
        tags: A list of string tags associated with the task for categorization or filtering.
    """
    tracer: Tracer

    name: str
    attributes: dict[str, t.Any]
    func: t.Callable[P, t.Awaitable[R]]
    scorers: list[Scorer[R]]
    tags: list[str] # TODO: Should tags be part of attributes?

    def __post_init__(self) -> None:
        """Performs post-initialization setup.

        Sets the `__signature__` attribute for introspection based on the wrapped
        function and updates `__name__` to reflect the wrapped function's name
        if available.
        """
        self.__signature__ = inspect.signature(self.func)
        # Use the function's name for the Task's name if available, otherwise keep provided name
        self.__name__ = getattr(self.func, "__name__", self.name)

    def _bind_args(self, *args: P.args, **kwargs: P.kwargs) -> dict[str, t.Any]:
        """Binds positional and keyword arguments to the wrapped function's signature.

        Applies default values from the function signature.

        Args:
            *args: Positional arguments intended for the wrapped function.
            **kwargs: Keyword arguments intended for the wrapped function.

        Returns:
            A dictionary mapping parameter names to their bound values.
        """
        signature = inspect.signature(self.func)
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return dict(bound_args.arguments)

    def clone(self) -> "Task[P, R]":
        """Creates a shallow copy of the Task instance.

        Copies mutable attributes (attributes, scorers, tags) to avoid unintended
        modification of the original task when modifying the clone via methods like `with_`.

        Returns:
            A new Task instance with copied mutable attributes.
        """
        return Task(
            tracer=self.tracer,
            name=self.name,
            attributes=self.attributes.copy(),
            func=self.func,
            scorers=self.scorers.copy(), # Copy list of scorers
            tags=self.tags.copy(),       # Copy list of tags
        )

    def with_(
        self,
        *,
        scorers: t.Sequence[Scorer[R] | ScorerCallable[R]] | None = None,
        name: str | None = None,
        tags: t.Sequence[str] | None = None,
        append: bool = False,
        **attributes: t.Any,
    ) -> "Task[P, R]":
        """Creates a modified copy of the Task with updated configuration.

        Allows overriding or appending scorers, tags, attributes, and the task name.

        Args:
            scorers: A sequence of Scorer objects or scorer callables to associate
                     with the new task. Replaces existing scorers unless `append` is True.
            name: An optional new name for the task. If None, keeps the original name.
            tags: An optional sequence of string tags. Replaces existing tags unless
                  `append` is True.
            append: If True, new scorers, tags, and attributes are added to the
                    existing ones. If False (default), they replace the existing ones.
            **attributes: Additional key-value attributes to associate with the task.
                          Replaces or updates existing attributes based on `append`.

        Returns:
            A new Task instance with the specified modifications.
        """
        task = self.clone()
        task.name = name or task.name # Override name if provided

        # Process scorers, converting callables if necessary
        new_scorers = [
            scorer if isinstance(scorer, Scorer) else Scorer.from_callable(self.tracer, scorer)
            for scorer in (scorers or [])
        ]
        new_tags = list(tags or []) # Ensure it's a list

        if append:
            task.scorers.extend(new_scorers)
            task.tags.extend(new_tags)
            # Use update for attributes when appending
            task.attributes.update(attributes)
        else:
            task.scorers = new_scorers
            task.tags = new_tags
            # Replace attributes entirely if not appending
            task.attributes = attributes

        return task

    async def run(self, *args: P.args, **kwargs: P.kwargs) -> TaskSpan[R]:
        """Executes the task function once within a traceable span.

        Creates a TaskSpan, executes the wrapped asynchronous function, captures
        its output, applies all configured scorers to the output, and records the
        scores on both the TaskSpan and the current RunSpan.

        Args:
            *args: Positional arguments to pass to the wrapped function.
            **kwargs: Keyword arguments to pass to the wrapped function.

        Returns:
            The TaskSpan object representing this execution, containing the output,
            scores, and trace information.

        Raises:
            RuntimeError: If not called within the context of an active, recording RunSpan.
            Exception: Any exception raised by the wrapped `self.func`.
        """
        run = current_run_span.get()
        if run is None or not run.is_recording(): # Check if run exists and is active
            raise RuntimeError("Tasks must be executed within an active, recording run context.")

        # Bind arguments before starting the span for accurate capture
        bound_args = self._bind_args(*args, **kwargs)

        with TaskSpan[R](
            name=self.name,
            attributes=self.attributes,
            args=bound_args, # Pass bound args
            run_id=run.run_id,
            tracer=self.tracer,
            tags=self.tags, # Include tags in the span
        ) as span:
            # Execute the underlying async function
            output = await self.func(*args, **kwargs)
            span.output = output # Store output on the span

            # Apply scorers
            score_tasks = [scorer(span.output) for scorer in self.scorers]
            computed_scores = await asyncio.gather(*score_tasks) # Run scorers concurrently

            span.scores.extend(computed_scores) # Add scores to the task span
            run.scores.extend(computed_scores) # Add scores to the run span

        return span

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Executes the task and returns only its output.

        Syntactic sugar for calling `run()` and extracting the output from the resulting span.

        Args:
            *args: Positional arguments for the wrapped function.
            **kwargs: Keyword arguments for the wrapped function.

        Returns:
            The output (type R) of the wrapped function.

        Raises:
            RuntimeError: If not called within an active run context (via `run`).
            Exception: Any exception raised by the wrapped `self.func` (via `run`).
        """
        span = await self.run(*args, **kwargs)
        # Type ignore might be needed if type checker struggles with R from TaskSpan[R]
        return span.output # type: ignore

    # --- Mapping and Helper Methods ---

    async def map_run(self, count: int, *args: P.args, **kwargs: P.kwargs) -> TaskSpanList[R]:
        """Runs the task multiple times concurrently and returns all resulting TaskSpans.

        Executes the `run` method `count` times with the same arguments.

        Args:
            count: The number of times to run the task.
            *args: Positional arguments passed to each task execution.
            **kwargs: Keyword arguments passed to each task execution.

        Returns:
            A TaskSpanList containing the TaskSpan object from each execution.
        """
        # Create coroutines for each run
        run_coroutines = [self.run(*args, **kwargs) for _ in range(count)]
        # Execute them concurrently
        spans = await asyncio.gather(*run_coroutines)
        return TaskSpanList(spans)

    async def map(self, count: int, *args: P.args, **kwargs: P.kwargs) -> list[R]:
        """Runs the task multiple times concurrently and returns only the outputs.

        Args:
            count: The number of times to run the task.
            *args: Positional arguments passed to each task execution.
            **kwargs: Keyword arguments passed to each task execution.

        Returns:
            A list containing the output (type R) from each task execution.
        """
        spans = await self.map_run(count, *args, **kwargs)
        # Type ignore might be needed depending on type checker strictness
        return [span.output for span in spans] # type: ignore

    async def top_n(self, count: int, n: int, *args: P.args, **kwargs: P.kwargs) -> list[R]:
        """Runs the task multiple times and returns the outputs of the top N scoring runs.

        Args:
            count: The total number of times to run the task.
            n: The number of top-scoring outputs to return.
            *args: Positional arguments passed to each task execution.
            **kwargs: Keyword arguments passed to each task execution.

        Returns:
            A list containing the outputs (type R) from the `n` highest-scoring executions.
        """
        spans = await self.map_run(count, *args, **kwargs)
        # Use the TaskSpanList helper to get top outputs directly
        return spans.top_n(n, as_outputs=True, reverse=True)

    async def try_run(self, *args: P.args, **kwargs: P.kwargs) -> TaskSpan[R] | None:
        """Executes the task once, catching and logging any exceptions.

        Similar to `run`, but returns None instead of raising an exception if the
        underlying task function fails. Prints a basic error message if an exception occurs.

        Args:
            *args: Positional arguments for the wrapped function.
            **kwargs: Keyword arguments for the wrapped function.

        Returns:
            The TaskSpan object if execution was successful, otherwise None.
        """
        try:
            return await self.run(*args, **kwargs)
        # Catch broad Exception, consider catching more specific types if needed
        except Exception as e:
             # TODO: Improve logging - use a proper logger, include traceback?
            print(f"Task {self.name} failed with exception: {e}")
            # Note: The exception is automatically recorded on the span by the context manager
            # if run() was entered, but here we catch it before it propagates.
            # If run() itself fails (e.g., no active run context), that error *will* raise.
            return None

    async def try_(self, *args: P.args, **kwargs: P.kwargs) -> R | None:
        """Executes the task, catches exceptions, and returns the output or None.

        Syntactic sugar for calling `try_run` and extracting the output.

        Args:
            *args: Positional arguments for the wrapped function.
            **kwargs: Keyword arguments for the wrapped function.

        Returns:
            The output (type R) if execution was successful, otherwise None.
        """
        span = await self.try_run(*args, **kwargs)
        return span.output if span else None

    async def try_map_run(self, count: int, *args: P.args, **kwargs: P.kwargs) -> TaskSpanList[R]:
        """Runs the task multiple times concurrently, catching exceptions for each run.

        Executes `try_run` `count` times. Failed runs are excluded from the result list.

        Args:
            count: The number of times to attempt running the task.
            *args: Positional arguments passed to each task attempt.
            **kwargs: Keyword arguments passed to each task attempt.

        Returns:
            A TaskSpanList containing only the TaskSpan objects from successful executions.
        """
        # Create coroutines for each try_run attempt
        try_run_coroutines = [self.try_run(*args, **kwargs) for _ in range(count)]
        # Execute them concurrently
        spans = await asyncio.gather(*try_run_coroutines)
        # Filter out the None results from failed runs
        return TaskSpanList([span for span in spans if span is not None])

    async def try_top_n(self, count: int, n: int, *args: P.args, **kwargs: P.kwargs) -> list[R]:
        """Runs the task multiple times, catching exceptions, and returns top N outputs.

        Runs the task `count` times using `try_map_run` (ignoring failures) and then
        returns the outputs from the `n` highest-scoring successful runs.

        Args:
            count: The total number of times to attempt running the task.
            n: The number of top-scoring outputs to return from the successful runs.
            *args: Positional arguments passed to each task attempt.
            **kwargs: Keyword arguments passed to each task attempt.

        Returns:
            A list containing the outputs (type R) from the `n` highest-scoring
            successful executions. Returns fewer than `n` items if fewer than `n` runs succeeded.
        """
        successful_spans = await self.try_map_run(count, *args, **kwargs)
        # Get top N outputs from the list of successful spans
        return successful_spans.top_n(n, as_outputs=True, reverse=True)

    async def try_map(self, count: int, *args: P.args, **kwargs: P.kwargs) -> list[R]:
        """Runs the task multiple times concurrently, catching exceptions, and returning outputs.

        Executes `try_map_run` and returns a list of outputs only from the successful runs.

        Args:
            count: The number of times to attempt running the task.
            *args: Positional arguments passed to each task attempt.
            **kwargs: Keyword arguments passed to each task attempt.

        Returns:
            A list containing the outputs (type R) from only the successful executions.
        """
        successful_spans = await self.try_map_run(count, *args, **kwargs)
        # Type ignore might be needed
        return [span.output for span in successful_spans] # type: ignore
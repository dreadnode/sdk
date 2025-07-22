import asyncio
import inspect
import traceback
import typing as t
from dataclasses import dataclass

from logfire._internal.stack_info import warn_at_user_stacklevel
from opentelemetry.trace import Tracer

from dreadnode.metric import Scorer, ScorerCallable
from dreadnode.serialization import seems_useful_to_serialize
from dreadnode.tracing.span import TaskSpan, current_run_span
from dreadnode.types import INHERITED, AnyDict, Inherited

P = t.ParamSpec("P")
R = t.TypeVar("R")


class TaskFailedWarning(UserWarning):
    pass


class TaskSpanList(list[TaskSpan[R]]):
    """
    Lightweight wrapper around a list of TaskSpans to provide some convenience methods.
    """

    def sorted(self, *, reverse: bool = True) -> "TaskSpanList[R]":
        """
        Sorts the spans in this list by their average metric value.

        Args:
            reverse: If True, sorts in descending order. Defaults to True.

        Returns:
            A new TaskSpanList sorted by average metric value.
        """
        return TaskSpanList(
            sorted(
                self,
                key=lambda span: span.get_average_metric_value(),
                reverse=reverse,
            ),
        )

    @t.overload
    def top_n(
        self,
        n: int,
        *,
        as_outputs: t.Literal[False] = False,
        reverse: bool = True,
    ) -> "TaskSpanList[R]": ...

    @t.overload
    def top_n(
        self,
        n: int,
        *,
        as_outputs: t.Literal[True],
        reverse: bool = True,
    ) -> list[R]: ...

    def top_n(
        self,
        n: int,
        *,
        as_outputs: bool = False,
        reverse: bool = True,
    ) -> "TaskSpanList[R] | list[R]":
        """
        Take the top n spans from this list, sorted by their average metric value.

        Args:
            n: The number of spans to take.
            as_outputs: If True, returns a list of outputs instead of spans. Defaults to False.
            reverse: If True, sorts in descending order. Defaults to True.

        Returns:
            A new TaskSpanList or list of outputs sorted by average metric value.
        """
        sorted_ = self.sorted(reverse=reverse)[:n]
        return (
            t.cast("list[R]", [span.output for span in sorted_])
            if as_outputs
            else TaskSpanList(sorted_)
        )


@dataclass
class Task(t.Generic[P, R]):
    """
    Structured task wrapper for a function that can be executed within a run.

    Tasks allow you to associate metadata, inputs, outputs, and metrics for a unit of work.
    """

    tracer: Tracer

    name: str
    "The name of the task. This is used for logging and tracing."
    label: str
    "The label of the task - used to group associated metrics and data together."
    attributes: dict[str, t.Any]
    "A dictionary of attributes to attach to the task span."
    func: t.Callable[P, R]
    "The function to execute as the task."
    scorers: list[Scorer[R]]
    "A list of scorers to evaluate the task's output."
    tags: list[str]
    "A list of tags to attach to the task span."

    log_inputs: t.Sequence[str] | bool | Inherited = INHERITED
    "Log all, or specific, incoming arguments to the function as inputs."
    log_output: bool | Inherited = INHERITED
    "Log the result of the function as an output."
    log_execution_metrics: bool = False
    "Track execution metrics such as success rate and run count."

    def __post_init__(self) -> None:
        self.__signature__ = getattr(
            self.func,
            "__signature__",
            inspect.signature(self.func),
        )
        self.__name__ = getattr(self.func, "__name__", self.name)
        self.__doc__ = getattr(self.func, "__doc__", None)

    def __get__(self, obj: t.Any, objtype: t.Any) -> "Task[P, R]":
        if obj is None:
            return self

        bound_func = self.func.__get__(obj, objtype)

        return Task(
            tracer=self.tracer,
            name=self.name,
            label=self.label,
            attributes=self.attributes,
            func=bound_func,
            scorers=[scorer.clone() for scorer in self.scorers],
            tags=self.tags.copy(),
            log_inputs=self.log_inputs,
            log_output=self.log_output,
        )

    def _bind_args(self, *args: P.args, **kwargs: P.kwargs) -> dict[str, t.Any]:
        signature = inspect.signature(self.func)
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return dict(bound_args.arguments)

    def clone(self) -> "Task[P, R]":
        """
        Clone a task.

        Returns:
            A new Task instance with the same attributes as this one.
        """
        return Task(
            tracer=self.tracer,
            name=self.name,
            label=self.label,
            attributes=self.attributes.copy(),
            func=self.func,
            scorers=[scorer.clone() for scorer in self.scorers],
            tags=self.tags.copy(),
            log_inputs=self.log_inputs,
            log_output=self.log_output,
        )

    def with_(
        self,
        *,
        scorers: t.Sequence[Scorer[R] | ScorerCallable[R]] | None = None,
        name: str | None = None,
        tags: t.Sequence[str] | None = None,
        label: str | None = None,
        log_inputs: t.Sequence[str] | bool | Inherited | None = None,
        log_output: bool | Inherited | None = None,
        log_execution_metrics: bool | None = None,
        append: bool = False,
        attributes: AnyDict | None = None,
    ) -> "Task[P, R]":
        """
        Clone a task and modify its attributes.

        Args:
            scorers: A list of new scorers to set or append to the task.
            name: The new name for the task.
            tags: A list of new tags to set or append to the task.
            label: The new label for the task.
            log_inputs: Log all, or specific, incoming arguments to the function as inputs.
            log_output: Log the result of the function as an output.
            log_execution_metrics: Log execution metrics such as success rate and run count.
            append: If True, appends the new scorers and tags to the existing ones. If False, replaces them.
            attributes: Additional attributes to set or update in the task.

        Returns:
            A new Task instance with the modified attributes.
        """
        task = self.clone()
        task.name = name or task.name
        task.label = label or task.label
        task.log_inputs = log_inputs if log_inputs is not None else task.log_inputs
        task.log_output = log_output if log_output is not None else task.log_output
        task.log_execution_metrics = (
            log_execution_metrics
            if log_execution_metrics is not None
            else task.log_execution_metrics
        )

        new_scorers = [Scorer.from_callable(scorer) for scorer in (scorers or [])]
        new_tags = list(tags or [])

        if append:
            task.scorers.extend(new_scorers)
            task.tags.extend(new_tags)
            task.attributes.update(attributes or {})
        else:
            task.scorers = new_scorers
            task.tags = new_tags
            task.attributes = attributes or {}

        return task

    async def run(self, *args: P.args, **kwargs: P.kwargs) -> TaskSpan[R]:
        """
        Execute the task and return the result as a TaskSpan.

        Args:
            args: The arguments to pass to the task.
            kwargs: The keyword arguments to pass to the task.

        Returns:
            The span associated with task execution.
        """

        run = current_run_span.get()

        log_inputs = (
            (run.autolog if run else False)
            if isinstance(self.log_inputs, Inherited)
            else self.log_inputs
        )
        log_output = (
            (run.autolog if run else False)
            if isinstance(self.log_output, Inherited)
            else self.log_output
        )

        bound_args = self._bind_args(*args, **kwargs)

        inputs_to_log = (
            bound_args
            if log_inputs is True
            else {k: v for k, v in bound_args.items() if k in log_inputs}
            if log_inputs is not False
            else {}
        )

        # If log_inputs is inherited, filter out items that don't seem useful
        # to serialize like `None` or repr fallbacks.
        if isinstance(self.log_inputs, Inherited):
            inputs_to_log = {k: v for k, v in inputs_to_log.items() if seems_useful_to_serialize(v)}

        with TaskSpan[R](
            name=self.name,
            label=self.label,
            attributes=self.attributes,
            tags=self.tags,
            run_id=run.run_id if run else "",
            tracer=self.tracer,
        ) as span:
            if run and self.log_execution_metrics:
                run.log_metric(
                    "count",
                    1,
                    prefix=f"{self.label}.exec",
                    mode="count",
                    attributes={"auto": True},
                )

            input_object_hashes: list[str] = [
                span.log_input(
                    name,
                    value,
                    label=f"{self.label}.input.{name}",
                    attributes={"auto": True},
                )
                for name, value in inputs_to_log.items()
            ]

            try:
                output = t.cast("R | t.Awaitable[R]", self.func(*args, **kwargs))
                if inspect.isawaitable(output):
                    output = await output
            except Exception:
                if run and self.log_execution_metrics:
                    run.log_metric(
                        "success_rate",
                        0,
                        prefix=f"{self.label}.exec",
                        mode="avg",
                        attributes={"auto": True},
                    )
                raise

            if run and self.log_execution_metrics:
                run.log_metric(
                    "success_rate",
                    1,
                    prefix=f"{self.label}.exec",
                    mode="avg",
                    attributes={"auto": True},
                )
            span.output = output

            if (
                run
                and log_output
                and (
                    not isinstance(self.log_inputs, Inherited) or seems_useful_to_serialize(output)
                )
            ):
                output_object_hash = span.log_output(
                    "output",
                    output,
                    label=f"{self.label}.output",
                    attributes={"auto": True},
                )

                # Link the output to the inputs
                for input_object_hash in input_object_hashes:
                    run.link_objects(output_object_hash, input_object_hash)

            for scorer in self.scorers:
                metric = await scorer(output)
                span.log_metric(scorer.name, metric, origin=output)

        # Trigger a run update whenever a task completes
        if run is not None:
            run.push_update()

        return span

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        span = await self.run(*args, **kwargs)
        return span.output

    # NOTE(nick): Not sure I'm in love with these being instance methods here.
    # We could move them to the top level class maybe.

    async def map_run(
        self,
        count: int,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> TaskSpanList[R]:
        """
        Run the task multiple times and return a list of spans.

        Args:
            count: The number of times to run the task.
            args: The arguments to pass to the task.
            kwargs: The keyword arguments to pass to the task.

        Returns:
            A TaskSpanList associated with each task execution.
        """
        spans = await asyncio.gather(*[self.run(*args, **kwargs) for _ in range(count)])
        return TaskSpanList(spans)

    async def map(self, count: int, *args: P.args, **kwargs: P.kwargs) -> list[R]:
        """
        Run the task multiple times and return a list of outputs.

        Args:
            count: The number of times to run the task.
            args: The arguments to pass to the task.
            kwargs: The keyword arguments to pass to the task.

        Returns:
            A list of outputs from each task execution.
        """
        spans = await self.map_run(count, *args, **kwargs)
        return [span.output for span in spans]

    async def top_n(
        self,
        count: int,
        n: int,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> list[R]:
        """
        Run the task multiple times and return the top n outputs.

        Args:
            count: The number of times to run the task.
            n: The number of top outputs to return.
            args: The arguments to pass to the task.
            kwargs: The keyword arguments to pass to the task.

        Returns:
            A list of the top n outputs from the task executions.
        """
        spans = await self.map_run(count, *args, **kwargs)
        return spans.top_n(n, as_outputs=True)

    async def try_run(self, *args: P.args, **kwargs: P.kwargs) -> TaskSpan[R] | None:
        """
        Attempt to run the task and return the result as a TaskSpan.
        If the task fails, a warning is logged and None is returned.

        Args:
            args: The arguments to pass to the task.
            kwargs: The keyword arguments to pass to the task.

        Returns:
            The span associated with task execution, or None if the task failed.
        """
        try:
            return await self.run(*args, **kwargs)
        except Exception:  # noqa: BLE001
            warn_at_user_stacklevel(
                f"Task '{self.name}' ({self.label}) failed:\n{traceback.format_exc()}",
                TaskFailedWarning,
            )
            return None

    async def try_(self, *args: P.args, **kwargs: P.kwargs) -> R | None:
        """
        Attempt to run the task and return the result.
        If the task fails, a warning is logged and None is returned.

        Args:
            args: The arguments to pass to the task.
            kwargs: The keyword arguments to pass to the task.

        Returns:
            The output of the task, or None if the task failed.
        """
        span = await self.try_run(*args, **kwargs)
        return span.output if span else None

    async def try_map_run(
        self,
        count: int,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> TaskSpanList[R]:
        """
        Attempt to run the task multiple times and return a list of spans.
        If any task fails, a warning is logged and None is returned for that task.

        Args:
            count: The number of times to run the task.
            args: The arguments to pass to the task.
            kwargs: The keyword arguments to pass to the task.

        Returns:
            A TaskSpanList associated with each task execution.
        """
        spans = await asyncio.gather(
            *[self.try_run(*args, **kwargs) for _ in range(count)],
        )
        return TaskSpanList([span for span in spans if span])

    async def try_top_n(
        self,
        count: int,
        n: int,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> list[R]:
        """
        Attempt to run the task multiple times and return the top n outputs.
        If any task fails, a warning is logged and None is returned for that task.

        Args:
            count: The number of times to run the task.
            n: The number of top outputs to return.
            args: The arguments to pass to the task.
            kwargs: The keyword arguments to pass to the task.

        Returns:
            A list of the top n outputs from the task executions.
        """
        spans = await self.try_map_run(count, *args, **kwargs)
        return spans.top_n(n, as_outputs=True)

    async def try_map(self, count: int, *args: P.args, **kwargs: P.kwargs) -> list[R]:
        """
        Attempt to run the task multiple times and return a list of outputs.
        If any task fails, a warning is logged and None is returned for that task.

        Args:
            count: The number of times to run the task.
            args: The arguments to pass to the task.
            kwargs: The keyword arguments to pass to the task.

        Returns:
            A list of outputs from each task execution.
        """
        spans = await self.try_map_run(count, *args, **kwargs)
        return [span.output for span in spans if span]


class TaskInputWarning(UserWarning):
    pass


CastT = t.TypeVar("CastT")


class TaskInput:
    """
    A placeholder to dynamically retrieve an input from the active TaskSpan.
    """

    def __init__(self, name: str, *, process: t.Callable[[t.Any], t.Any] | None = None) -> None:
        """
        Args:
            name: The name of the input to retrieve, as logged via `task.log_input(name=...)`.
            process: An optional function to process the input value before returning it.
                This can be used to transform or extract from the raw input value.
        """
        self.name = name
        self.process = process

    def __repr__(self) -> str:
        return f"TaskInput(name='{self.name}')"

    @t.overload
    def resolve(self, *, cast_as: None = None) -> t.Any: ...

    @t.overload
    def resolve(self, *, cast_as: type[CastT]) -> CastT: ...

    def resolve(self, *, cast_as: type[CastT] | None = None) -> t.Any:  # noqa: PLR0911
        """
        Resolve the input from the current TaskSpan.

        Args:
            cast_as: Optionally cast the resolved value to a specific type.

        Returns:
            The value of the input from the current TaskSpan.
        """
        from dreadnode.tracing.span import current_task_span

        if (task := current_task_span.get()) is None:
            warn_at_user_stacklevel(
                "TaskInput.resolve() called outside of an active TaskSpan context. "
                "This will raise an error in future versions.",
                TaskInputWarning,
            )
            return None

        try:
            task_input = task.inputs[self.name].value
        except KeyError:
            warn_at_user_stacklevel(
                f"Input '{self.name}' not found in the active TaskSpan. "
                f"Available inputs are: {list(task.inputs.keys())}",
                TaskInputWarning,
            )
            return None

        try:
            if self.process is not None:
                return self.process(task_input)
        except Exception as e:  # noqa: BLE001
            warn_at_user_stacklevel(
                f"Error processing TaskInput '{self.name}': {e}",
                TaskInputWarning,
            )
            return task_input

        if cast_as is not None:
            try:
                return cast_as(task_input)  # type: ignore [call-arg]
            except Exception as e:  # noqa: BLE001
                warn_at_user_stacklevel(
                    f"Error casting TaskInput '{self.name}' to {cast_as.__name__}: {e}",
                    TaskInputWarning,
                )
                return task_input

        return task_input

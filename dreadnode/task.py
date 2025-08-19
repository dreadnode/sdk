import contextlib
import inspect
import typing as t
from dataclasses import dataclass

from opentelemetry.trace import Tracer

from dreadnode.configurable import (
    CONFIGURABLE_ATTR,
    CONFIGURABLE_FIELDS_ATTR,
    clone_config_attrs,
)
from dreadnode.scorers.base import Scorer, ScorerCallable
from dreadnode.serialization import seems_useful_to_serialize
from dreadnode.tracing.span import TaskSpan, current_run_span
from dreadnode.types import INHERITED, UNSET, AnyDict, Arguments, Inherited, Unset
from dreadnode.util import concurrent_gen

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
    configurable: list[str] | bool = True
    """
    A list of task parameters to expose to the CLI.
    - If True, all keyword parameters are exposed.
    - If None, no parameters are exposed.
    """

    log_inputs: list[str] | bool | Inherited = INHERITED
    "Log all, or specific, incoming arguments to the function as inputs."
    log_output: bool | Inherited = INHERITED
    "Log the result of the function as an output."
    log_execution_metrics: bool = False
    "Track execution metrics such as success rate and run count."

    _prepared_args: t.ClassVar[bool] = False

    def __post_init__(self) -> None:
        self.__signature__ = getattr(
            self.func,
            "__signature__",
            inspect.signature(self.func),
        )
        self.__name__ = getattr(self.func, "__name__", self.name)
        self.__doc__ = getattr(self.func, "__doc__", None)

        # Update our configurable attribute to reflect the task params

        config_fields = ["scorers"]

        kw_only_params = [
            name
            for name, p in self.__signature__.parameters.items()
            if p.kind == inspect.Parameter.KEYWORD_ONLY
        ]

        if self.configurable is True:
            config_fields.extend(kw_only_params)
        elif isinstance(self.configurable, list):
            config_fields.extend(self.configurable)

        setattr(self, CONFIGURABLE_ATTR, True)
        setattr(self, CONFIGURABLE_FIELDS_ATTR, config_fields)

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
            scorers=self.scorers.copy(),
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
        return clone_config_attrs(
            self,
            Task(
                tracer=self.tracer,
                name=self.name,
                label=self.label,
                attributes=self.attributes.copy(),
                func=self.func,
                scorers=self.scorers.copy(),
                tags=self.tags.copy(),
                log_inputs=self.log_inputs,
                log_output=self.log_output,
                log_execution_metrics=self.log_execution_metrics,
                configurable=self.configurable,
            ),
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
        configurable: t.Sequence[str] | None | Unset = UNSET,
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
            configurable: A list of task parameters to expose to the CLI.
                - If None, all keyword parameters are exposed.
                - If [], all parameters are exposed.

        Returns:
            A new Task instance with the modified attributes.
        """
        task = self.clone()
        task.name = name or task.name
        task.label = label or task.label
        task.log_inputs = (
            task.log_inputs
            if log_inputs is None
            else log_inputs
            if isinstance(log_inputs, (bool | Inherited))
            else list(log_inputs)
        )
        task.log_output = task.log_output if log_output is None else log_output
        task.log_execution_metrics = (
            log_execution_metrics
            if log_execution_metrics is not None
            else task.log_execution_metrics
        )
        task.configurable = (
            configurable
            if isinstance(configurable, bool)
            else list(configurable or [])
            if not isinstance(configurable, Unset)
            else task.configurable
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

    async def run_always(self, *args: P.args, **kwargs: P.kwargs) -> TaskSpan[R]:
        """
        Execute the task and return the result as a TaskSpan.

        Note, if the task fails, the span will still be returned with the exception set.

        Args:
            args: The arguments to pass to the task.
            kwargs: The keyword arguments to pass to the task.

        Returns:
            The span associated with task execution.
        """
        from dreadnode import score

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

        task_span = TaskSpan[R](
            name=self.name,
            label=self.label,
            attributes=self.attributes,
            tags=self.tags,
            run_id=run.run_id if run else "",
            tracer=self.tracer,
            arguments=Arguments(args, kwargs),
        )

        with contextlib.suppress(Exception), task_span as span:
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

            await score(output, self.scorers)

            # Trigger a run update whenever a task completes
            if run is not None:
                run.push_update()

        return span

    async def run(self, *args: P.args, **kwargs: P.kwargs) -> TaskSpan[R]:
        """
        Execute the task and return the result as a TaskSpan.
        If the task fails, an exception is raised.

        Args:
            args: The arguments to pass to the task.
            kwargs: The keyword arguments to pass to the task
        """
        span = await self.run_always(*args, **kwargs)
        span.raise_if_failed()
        return span

    async def try_(self, *args: P.args, **kwargs: P.kwargs) -> R | None:
        """
        Attempt to run the task and return the result.
        If the task fails, None is returned.

        Args:
            args: The arguments to pass to the task.
            kwargs: The keyword arguments to pass to the task.

        Returns:
            The output of the task, or None if the task failed.
        """
        span = await self.run_always(*args, **kwargs)
        with contextlib.suppress(Exception):
            return span.output
        return None

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        span = await self.run(*args, **kwargs)
        return span.output

    # Mapping

    def _prepare_map_args(
        self,
        args: list[t.Any] | dict[str, t.Any | list[t.Any]],
    ) -> list[Arguments]:
        positional_args: list[t.Any] = []
        static_kwargs: dict[str, t.Any] = {}
        mapped_kwargs: dict[str, list[t.Any]] = {}
        map_length: int | None = None

        # User gave us a flat list, treat it as positional args.
        if isinstance(args, list):
            positional_args = args
            map_length = len(positional_args)

        # User gave us a dict, separate static and mapped parameters.
        elif isinstance(args, dict):
            for name, value in args.items():
                if not isinstance(value, list):
                    static_kwargs[name] = value
                    continue

                # This is the first list we've seen, it sets the expected length.
                if map_length is None:
                    map_length = len(value)

                if len(value) != map_length:
                    raise ValueError(
                        f"Mismatched lengths for mapped parameters. Expected length {map_length} "
                        f"for parameter '{name}', but got {len(value)}."
                    )

                mapped_kwargs[name] = value

        # Otherwise we don't know how to handle it.
        else:
            raise TypeError(f"Expected 'args' to be a list or dict, but got {type(args).__name__}.")

        # Ensure we are mapping over at least one list.
        if map_length is None:
            raise ValueError("The args for map() must contain at least one list to map over.")

        # Construct the list of keyword argument dictionaries for each call.
        arguments: list[Arguments] = []
        for i in range(map_length):
            kwargs_for_this_run = static_kwargs.copy()
            for name, values_list in mapped_kwargs.items():
                kwargs_for_this_run[name] = values_list[i]
            arguments.append(Arguments((positional_args[i],), kwargs_for_this_run))

        return arguments

    def stream_map(
        self,
        args: list[t.Any] | dict[str, t.Any | list[t.Any]],
        *,
        concurrency: int | None = None,
    ) -> t.AsyncContextManager[t.AsyncGenerator[TaskSpan[R], None]]:
        """
        Runs this task multiple times by mapping over iterable arguments.

        Args:
            args: Either a flat list of the first positional argument, or a dict
                  where each key is a parameter name and the value is either a single value
                  or a list of values to map over.
            concurrency: The maximum number of tasks to run in parallel.
                         If None, runs with unlimited concurrency.

        Returns:
            A TaskSpanList containing the results of each execution.
        """
        arguments = self._prepare_map_args(args)
        tasks = [self.run_always(*args.args, **args.kwargs) for args in arguments]
        return concurrent_gen(tasks, concurrency)

    async def map(
        self,
        args: list[t.Any] | dict[str, t.Any | list[t.Any]],
        *,
        concurrency: int | None = None,
    ) -> list[R]:
        """
        Runs this task multiple times by mapping over iterable arguments.

        Examples:
            ```python

            @dn.task
            async def my_task(input: str, *, suffix: str = "") -> str:
                return f"Processed {input}{suffix}"

            # Map over a list of basic inputs
            await task.map_run(["1", "2", "3"])

            # Map over a dict of parameters
            await task.map_run({
                "input": ["1", "2", "3"],
                "suffix": ["_a", "_b", "_c"]
            })
            ```

        Args:
            args: Either a flat list of the first positional argument, or a dict
                  where each key is a parameter name and the value is either a single value
                  or a list of values to map over.
            concurrency: The maximum number of tasks to run in parallel.
                         If None, runs with unlimited concurrency.

        Returns:
            A TaskSpanList containing the results of each execution.
        """
        async with self.stream_map(args, concurrency=concurrency) as stream:
            return [span.output async for span in stream]

    async def try_map(
        self,
        args: list[t.Any] | dict[str, t.Any | list[t.Any]],
        *,
        concurrency: int | None = None,
    ) -> list[R]:
        """
        Attempt to run this task multiple times by mapping over iterable arguments.
        If any task fails, its result is excluded from the output.

        Args:
            args: Either a flat list of the first positional argument, or a dict
                  where each key is a parameter name and the value is either a single value
                  or a list of values to map over.
            concurrency: The maximum number of tasks to run in parallel.
                         If None, runs with unlimited concurrency.

        Returns:
            A TaskSpanList containing the results of each execution.
        """
        async with self.stream_map(args, concurrency=concurrency) as stream:
            return [span.output async for span in stream if span.exception is None]

    # Many (replicate)

    def stream_many(
        self,
        count: int,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> t.AsyncContextManager[t.AsyncGenerator[TaskSpan[R], None]]:
        """
        Run the task multiple times concurrently and yield each TaskSpan as it completes.

        Args:
            count: The number of times to run the task.
            args: The arguments to pass to the task.
            kwargs: The keyword arguments to pass to the task

        Yields:
            TaskSpan for each task execution, or an Exception if the task fails.
        """
        tasks = [self.run_always(*args, **kwargs) for _ in range(count)]
        return concurrent_gen(tasks)

    async def many(self, count: int, *args: P.args, **kwargs: P.kwargs) -> list[R]:
        """
        Run the task multiple times and return a list of outputs.

        Args:
            count: The number of times to run the task.
            args: The arguments to pass to the task.
            kwargs: The keyword arguments to pass to the task.

        Returns:
            A list of outputs from each task execution.
        """
        async with self.stream_many(count, *args, **kwargs) as stream:
            return [span.output async for span in stream]

    async def try_many(
        self,
        count: int,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> list[R]:
        """
        Attempt to run the task multiple times and return a list of outputs.
        If any task fails, its result is excluded from the output.

        Args:
            count: The number of times to run the task.
            args: The arguments to pass to the task.
            kwargs: The keyword arguments to pass to the task.

        Returns:
            A list of outputs from each task execution.
        """
        async with self.stream_many(count, *args, **kwargs) as stream:
            return [span.output async for span in stream if span.exception is None]

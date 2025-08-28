import contextlib
import inspect
import typing as t
from copy import deepcopy
from pathlib import Path

import typing_extensions as te
from opentelemetry.trace import Tracer

from dreadnode.meta.context import Context
from dreadnode.meta.types import Component, ConfigInfo
from dreadnode.scorers.base import Scorer, ScorerCallable, ScorersLike
from dreadnode.serialization import seems_useful_to_serialize
from dreadnode.tracing.span import TaskSpan, current_run_span
from dreadnode.types import INHERITED, AnyDict, Arguments, Inherited
from dreadnode.util import (
    clean_str,
    concurrent_gen,
    get_callable_name,
    get_filepath_attribute,
)

if t.TYPE_CHECKING:
    from dreadnode.eval.dataset import (
        EvalResult,
        InputDataset,
        InputDatasetProcessor,
        InputT,
        OutputT,
    )
    from dreadnode.eval.eval import Eval

P = t.ParamSpec("P")
R = t.TypeVar("R")

# Some excessive typing here to ensure we can properly
# overload our decorator for sync/async and cases
# where we need the return type of the task to align
# with the scorer inputs


class TaskDecorator(t.Protocol):
    @t.overload
    def __call__(
        self,
        func: t.Callable[P, t.Awaitable[R]],
    ) -> "Task[P, R]": ...

    @t.overload
    def __call__(
        self,
        func: t.Callable[P, R],
    ) -> "Task[P, R]": ...

    def __call__(
        self,
        func: t.Callable[P, t.Awaitable[R]] | t.Callable[P, R],
    ) -> "Task[P, R]": ...


class ScoredTaskDecorator(t.Protocol, t.Generic[R]):
    @t.overload
    def __call__(
        self,
        func: t.Callable[P, t.Awaitable[R]],
    ) -> "Task[P, R]": ...

    @t.overload
    def __call__(
        self,
        func: t.Callable[P, R],
    ) -> "Task[P, R]": ...

    def __call__(
        self,
        func: t.Callable[P, t.Awaitable[R]] | t.Callable[P, R],
    ) -> "Task[P, R]": ...


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


class Task(Component[P, R], t.Generic[P, R]):
    """
    Structured task wrapper for a function that can be executed within a run.

    Tasks allow you to associate metadata, inputs, outputs, and metrics for a unit of work.
    """

    def __init__(
        self,
        func: t.Callable[P, R],
        tracer: Tracer,
        *,
        name: str | None = None,
        label: str | None = None,
        scorers: ScorersLike[R] | None = None,
        log_inputs: t.Sequence[str] | bool | Inherited = INHERITED,
        log_output: bool | Inherited = INHERITED,
        log_execution_metrics: bool = False,
        tags: t.Sequence[str] | None = None,
        attributes: AnyDict | None = None,
        config: dict[str, ConfigInfo] | None = None,
        context: dict[str, Context] | None = None,
    ) -> None:
        unwrapped = inspect.unwrap(func)
        if inspect.isgeneratorfunction(unwrapped) or inspect.isasyncgenfunction(
            unwrapped,
        ):
            raise TypeError("@task cannot be applied to generators")

        func_name = get_callable_name(unwrapped, short=True)
        name = name or func_name
        label = clean_str(label or name)

        attributes = attributes or {}
        attributes["code.function"] = func_name
        with contextlib.suppress(Exception):
            attributes["code.lineno"] = unwrapped.__code__.co_firstlineno
        with contextlib.suppress(Exception):
            attributes.update(
                get_filepath_attribute(
                    inspect.getsourcefile(unwrapped),  # type: ignore [arg-type]
                ),
            )

        super().__init__(func, config=config, context=context)

        self.__dn_attr_config__["scorers"] = ConfigInfo(field_kwargs={"default": scorers})

        self._tracer = tracer

        self.name = name
        "The name of the task. This is used for logging and tracing."
        self.label = label
        "The label of the task - used to group associated metrics and data together."
        self.scorers = Scorer.fit_like(scorers)
        "A list of scorers to evaluate the task's output."
        self.tags = list(tags or [])
        "A list of tags to attach to the task span."
        self.attributes = attributes
        "A dictionary of attributes to attach to the task span."
        self.log_inputs = (
            log_inputs if isinstance(log_inputs, bool | Inherited) else list(log_inputs)
        )
        "Log all, or specific, incoming arguments to the function as inputs."
        self.log_output = log_output
        "Log the result of the function as an output."
        self.log_execution_metrics = log_execution_metrics
        "Track execution metrics such as success rate and run count."

    def __get__(self, obj: t.Any, objtype: t.Any) -> "Task[P, R]":
        if obj is None:
            return self

        bound_func = self.func.__get__(obj, objtype)

        return Task(
            tracer=self._tracer,
            name=self.name,
            label=self.label,
            attributes=self.attributes,
            func=bound_func,
            scorers=self.scorers.copy(),
            tags=self.tags.copy(),
            log_inputs=self.log_inputs,
            log_output=self.log_output,
        )

    def __deepcopy__(self, memo: dict[int, t.Any]) -> "Task[P, R]":
        return Task(
            func=self.func,
            tracer=self._tracer,
            name=self.name,
            label=self.label,
            scorers=self.scorers.copy(),
            log_inputs=self.log_inputs,
            log_output=self.log_output,
            log_execution_metrics=self.log_execution_metrics,
            tags=self.tags.copy(),
            attributes=self.attributes.copy(),
            config=deepcopy(self.__dn_param_config__, memo),
            context=deepcopy(self.__dn_context__, memo),
        )

    def clone(self) -> "Task[P, R]":
        """
        Clone a task.

        Returns:
            A new Task instance with the same attributes as this one.
        """
        return self.__deepcopy__({})

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

        new_scorers = [Scorer(scorer) for scorer in (scorers or [])]
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

    def as_eval(
        self,
        dataset: "InputDataset[InputT] | list[AnyDict] | Path | str",
        *,
        name: str | None = None,
        description: str = "",
        concurrency: int | None = None,
        preprocessor: "InputDatasetProcessor | None" = None,
        scorers: "ScorersLike[R] | None" = None,
        assertions: "ScorersLike[R] | None" = None,
    ) -> "Eval[InputT, R]":
        from dreadnode.eval.eval import Eval

        if isinstance(dataset, str):
            dataset = Path(dataset)

        return Eval[InputT, R](
            dataset=dataset,
            name=name,
            task=self,
            description=description,
            concurrency=concurrency,
            preprocessor=preprocessor,
            scorers=scorers or [],
            assertions=assertions or [],
        )

    async def eval(
        self, dataset: "InputDataset[InputT] | list[AnyDict] | Path | str"
    ) -> "EvalResult[InputT, OutputT]":
        """
        Evaluate the task with the given arguments and return an evaluation result.
        """
        return await self.as_eval(dataset).run()

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
        bound_args_dict = dict(bound_args.arguments)

        inputs_to_log = (
            bound_args_dict
            if log_inputs is True
            else {k: v for k, v in bound_args_dict.items() if k in log_inputs}
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
            tracer=self._tracer,
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
                output = t.cast(
                    "R | t.Awaitable[R]", self.func(*bound_args.args, **bound_args.kwargs)
                )
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

    @te.override
    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:  # type: ignore[override]
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
            arguments.append(
                Arguments((positional_args[i],) if positional_args else (), kwargs_for_this_run)
            )

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

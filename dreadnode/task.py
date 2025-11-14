import contextlib
import inspect
import typing as t
from pathlib import Path

import typing_extensions as te
from opentelemetry.trace import Tracer

from dreadnode.common_types import INHERITED, AnyDict, Arguments, Inherited
from dreadnode.meta import Component, ConfigInfo, Context
from dreadnode.scorers import Scorer, ScorerCallable, ScorersLike
from dreadnode.serialization import seems_useful_to_serialize
from dreadnode.tracing.span import TaskSpan, current_run_span
from dreadnode.util import (
    clean_str,
    concurrent_gen,
    get_callable_name,
    get_filepath_attribute,
)

if t.TYPE_CHECKING:
    from dreadnode.airt.target.custom import CustomTarget
    from dreadnode.eval.eval import (
        Eval,
        InputDatasetProcessor,
    )

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
    """Warning related to task execution failures."""


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
        assert_scores: list[str] | t.Literal[True] | None = None,
        log_inputs: t.Sequence[str] | bool | Inherited = INHERITED,
        log_output: bool | Inherited = INHERITED,
        log_execution_metrics: bool = False,
        tags: t.Sequence[str] | None = None,
        attributes: AnyDict | None = None,
        entrypoint: bool = False,
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

        super().__init__(func, name=name, config=config, context=context, convert_all=entrypoint)

        self._tracer = tracer

        self.name = name
        "The name of the task. This is used for logging and tracing."
        self.label = label
        "The label of the task - used to group associated metrics and data together."
        self.scorers = Scorer.fit_many(scorers)
        "A list of scorers to evaluate the task's output."
        scorer_names = [s.name for s in self.scorers]
        self.assert_scores = scorer_names if assert_scores is True else list(assert_scores or [])
        "A list of score names to ensure have truthy values, otherwise raise an AssertionFailedError."
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
        self.entrypoint = entrypoint
        """
        Indicate this task should be considered an entrypoint. All compatible arguments
        will be treated as configurable and a run will be created automatically when called if
        one is not already active.
        """

        for assertion in self.assert_scores or []:
            if assertion not in scorer_names:
                raise ValueError(
                    f"Unknown '{assertion}' in assert_scores, it must be one of {scorer_names}"
                )

    def __repr__(self) -> str:
        func_name = get_callable_name(self.func, short=True)

        parts: list[str] = [
            f"name='{self.name}'",
            f"func={func_name}",
        ]

        if self.label != clean_str(self.name):
            parts.append(f"label='{self.label}'")
        if self.entrypoint:
            parts.append("entrypoint=True")
        if self.scorers:
            scorers = [scorer.name for scorer in self.scorers]
            parts.append(f"scorers={scorers}")
        if self.assert_scores:
            parts.append(f"assert_scores={self.assert_scores}")
        if self.tags:
            parts.append(f"tags={self.tags}")
        if not isinstance(self.log_inputs, Inherited):
            parts.append(f"log_inputs={self.log_inputs}")
        if not isinstance(self.log_output, Inherited):
            parts.append(f"log_output={self.log_output}")

        return f"{self.__class__.__name__}({', '.join(parts)})"

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
            log_execution_metrics=self.log_execution_metrics,
            assert_scores=self.assert_scores.copy(),
            entrypoint=self.entrypoint,
            config=self.__dn_param_config__,
            context=self.__dn_context__,
        )

    def __deepcopy__(self, memo: dict[int, t.Any]) -> "Task[P, R]":
        return Task(
            func=self.func,
            tracer=self._tracer,
            name=self.name,
            label=self.label,
            scorers=self.scorers.copy(),
            assert_scores=self.assert_scores.copy(),
            log_inputs=self.log_inputs,
            log_output=self.log_output,
            log_execution_metrics=self.log_execution_metrics,
            tags=self.tags.copy(),
            attributes=self.attributes.copy(),
            entrypoint=self.entrypoint,
            config=dict(self.__dn_param_config__),
            context=dict(self.__dn_context__),
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
        assert_scores: t.Sequence[str] | t.Literal[True] | None = None,
        name: str | None = None,
        tags: t.Sequence[str] | None = None,
        label: str | None = None,
        log_inputs: t.Sequence[str] | bool | Inherited | None = None,
        log_output: bool | Inherited | None = None,
        log_execution_metrics: bool | None = None,
        append: bool = False,
        attributes: AnyDict | None = None,
        entrypoint: bool = False,
    ) -> "Task[P, R]":
        """
        Clone a task and modify its attributes.

        Args:
            scorers: A list of new scorers to set or append to the task.
            assert_scores: A list of new assertion names to set or append to the task.
            name: The new name for the task.
            tags: A list of new tags to set or append to the task.
            label: The new label for the task.
            log_inputs: Log all, or specific, incoming arguments to the function as inputs.
            log_output: Log the result of the function as an output.
            log_execution_metrics: Log execution metrics such as success rate and run count.
            append: If True, appends the new scorers and tags to the existing ones. If False, replaces them.
            attributes: Additional attributes to set or update in the task.
            entrypoint: Indicate this task should be considered an entrypoint. All compatible arguments
                will be treated as configurable and a run will be created automatically when called if
                one is not already active.

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
        task.entrypoint = entrypoint

        new_scorers = Scorer.fit_many(scorers or [])
        new_tags = list(tags or [])
        new_assert_scores = (
            [s.name for s in new_scorers] if assert_scores is True else list(assert_scores or [])
        )

        if append:
            task.scorers.extend(new_scorers)
            task.tags.extend(new_tags)
            task.assert_scores.extend(new_assert_scores)
            task.attributes.update(attributes or {})
        else:
            task.scorers = new_scorers
            task.tags = new_tags
            task.assert_scores = new_assert_scores
            task.attributes = attributes or {}

        return task

    def as_eval(
        self,
        *,
        dataset: t.Any | None = None,
        dataset_file: Path | str | None = None,
        name: str | None = None,
        description: str = "",
        tags: list[str] | None = None,
        concurrency: int = 1,
        iterations: int = 1,
        max_errors: int | None = None,
        max_consecutive_errors: int = 10,
        dataset_input_mapping: list[str] | dict[str, str] | None = None,
        parameters: dict[str, list[t.Any]] | None = None,
        preprocessor: "InputDatasetProcessor | None" = None,
        scorers: "ScorersLike[R] | None" = None,
        assert_scores: list[str] | t.Literal[True] | None = None,
    ) -> "Eval[t.Any, R]":
        from dreadnode.eval.eval import Eval

        return Eval[t.Any, R](
            task=t.cast("Task[[t.Any], R]", self),
            dataset=dataset,
            dataset_file=dataset_file,
            name=name,
            description=description,
            tags=tags or ["eval"],
            concurrency=concurrency,
            iterations=iterations,
            max_errors=max_errors,
            max_consecutive_errors=max_consecutive_errors,
            dataset_input_mapping=dataset_input_mapping,
            parameters=parameters,
            preprocessor=preprocessor,
            scorers=scorers or [],
            assert_scores=assert_scores or [],
        )

    def as_target(
        self,
        input_param_name: str | None = None,
    ) -> "CustomTarget[R]":
        """
        Convert this task into a CustomTarget that can be used in AIRT attack patterns.

        Args:
            input_param_name: The name of the parameter in the task's signature where the attack input should be injected.
                              Otherwise the first non-optional parameter will be used, or no injection will occur.

        Returns:
            A CustomTarget wrapping this task.
        """
        from dreadnode.airt.target.custom import CustomTarget

        return CustomTarget(task=self, input_param_name=input_param_name)

    async def run_always(self, *args: P.args, **kwargs: P.kwargs) -> TaskSpan[R]:  # noqa: PLR0912
        """
        Execute the task and return the result as a TaskSpan.

        Note, if the task fails, the span will still be returned with the exception set.

        Args:
            args: The arguments to pass to the task.
            kwargs: The keyword arguments to pass to the task.

        Returns:
            The span associated with task execution.
        """
        from dreadnode import run as run_span
        from dreadnode import score

        current_run = current_run_span.get()
        create_run = current_run is None and self.entrypoint
        run = current_run or (
            run_span(name_prefix=self.name, tags=self.tags, _tracer=self._tracer)
            if self.entrypoint
            else None
        )
        task = TaskSpan[R](
            name=self.name,
            label=self.label,
            attributes=self.attributes,
            tags=self.tags,
            run_id=run.run_id if run else "",
            tracer=self._tracer,
            arguments=Arguments(args, kwargs),
        )

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

        ctx_inputs_to_log = t.cast("AnyDict", kwargs.pop("__dn_ctx_inputs__", {}))

        with run if run and create_run else contextlib.nullcontext():
            with contextlib.suppress(Exception), task:
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
                    inputs_to_log = {
                        k: v for k, v in inputs_to_log.items() if seems_useful_to_serialize(v)
                    }

                if run and self.log_execution_metrics:
                    run.log_metric(
                        "count",
                        1,
                        prefix=f"{self.label}.exec",
                        mode="count",
                        attributes={"auto": True},
                    )

                input_object_hashes: list[str] = []
                for name, value in inputs_to_log.items():
                    input_object_hashes.append(
                        task.log_input(name, value, attributes={"auto": True})
                    )

                    if run is None or not create_run:
                        continue

                    if isinstance(value, int | float | str | bool | None):
                        run.log_param(name, value)
                    else:
                        run.log_input(name, value, attributes={"auto": True})

                for name, value in ctx_inputs_to_log.items():
                    task.log_input(
                        name,
                        value,
                        attributes={"auto": True, "ctx": True},
                    )

                output = t.cast(
                    "R | t.Awaitable[R]", self.func(*bound_args.args, **bound_args.kwargs)
                )
                if inspect.isawaitable(output):
                    output = await output

                task.output = output

                # Log the output

                output_object_hash = None
                if log_output and (
                    not isinstance(self.log_inputs, Inherited) or seems_useful_to_serialize(output)
                ):
                    output_object_hash = task.log_output(
                        "output",
                        output,
                        attributes={"auto": True},
                    )
                    # Link the output to the inputs
                    if run is not None:
                        for input_object_hash in input_object_hashes:
                            run.link_objects(output_object_hash, input_object_hash)
                elif run is not None and create_run:
                    run.log_output("output", output, attributes={"auto": True})

                # Score and check assertions

                await score(output, self.scorers, assert_scores=self.assert_scores)

            if run and self.log_execution_metrics:
                run.log_metric(
                    "success_rate",
                    0 if task.exception else 1,
                    prefix=f"{self.label}.exec",
                    mode="avg",
                    attributes={"auto": True},
                )

            # Trigger a run update whenever a task completes
            if run is not None and not create_run:
                run.push_update()

        return task

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

    # Retry

    async def retry(
        self,
        count: int,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """
        Run the task up to `count` times, returning the output of the first
        successful execution, otherwise raise the most recent exception.

        This is a powerful pattern for non-deterministic tasks where multiple
        attempts may be needed to generate a valid output according to the
        task's `assert_scores`. However, it can also be useful as a retry
        mechanism for transient errors.

        Args:
            count: The maximum number of times to run the task.
            args: The arguments to pass to the task.
            kwargs: The keyword arguments to pass to the task.

        Returns:
            The output of the first successful and valid task execution.
        """
        last_span = None
        for _ in range(count):
            span = await self.run_always(*args, **kwargs)
            last_span = span
            if span.exception is None:
                return span.output

        # If the loop finishes, all attempts failed. Raise the exception
        # from the final attempt for debugging.
        if last_span is not None:
            last_span.raise_if_failed()

        # Just for type checking - should never be called
        raise RuntimeError("Generation failed to produce a valid result.")

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

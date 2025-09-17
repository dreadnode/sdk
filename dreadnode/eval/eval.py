import contextlib
import contextvars
import itertools
import typing as t
from contextlib import asynccontextmanager
from functools import cached_property
from pathlib import Path

import typing_extensions as te
from pydantic import ConfigDict, Field, FilePath, TypeAdapter, computed_field

from dreadnode.common_types import AnyDict, Unset
from dreadnode.discovery import find
from dreadnode.eval.console import EvalConsoleAdapter
from dreadnode.eval.dataset import load_dataset
from dreadnode.eval.events import (
    EvalEnd,
    EvalEvent,
    EvalStart,
    IterationEnd,
    IterationStart,
    SampleComplete,
    ScenarioEnd,
    ScenarioStart,
)
from dreadnode.eval.result import EvalResult, IterationResult, ScenarioResult
from dreadnode.eval.sample import Sample
from dreadnode.meta import Config, DatasetField, Model
from dreadnode.meta.introspect import get_config_model, get_inputs_and_params_from_config_model
from dreadnode.scorers.base import Scorer, ScorersLike
from dreadnode.task import Task
from dreadnode.util import (
    concurrent_gen,
    get_callable_name,
    shorten_string,
)

In = te.TypeVar("In", default=t.Any)
Out = te.TypeVar("Out", default=t.Any)

InputDataset = list[In]
InputDatasetProcessor = t.Callable[[InputDataset], InputDataset]

current_dataset_row = contextvars.ContextVar[t.Mapping[str, t.Any] | None](
    "current_dataset_row", default=None
)


class EvalWarning(UserWarning):
    """Warning raised during evaluation."""


class Eval(Model, t.Generic[In, Out]):
    """
    Prepared evaluation of a task with an associated dataset and configuration.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    task: t.Annotated[Task[[In], Out] | str, Config(expose_as=t.Any)]
    """The task to evaluate. Can be a Task object or a string representing qualified task name."""
    dataset: t.Annotated[InputDataset[In] | list[AnyDict] | FilePath, Config(expose_as=t.Any)]
    """The dataset to use for the evaluation. Can be a list of inputs or a file path to load inputs from."""

    name_: str | None = Field(default=None, alias="name", repr=False, exclude=True)
    """The name of the evaluation."""
    description: str = ""
    """A brief description of the eval's purpose."""
    label: str | None = None
    """Specific label to use for tasks created by this eval."""
    tags: list[str] = Config(default_factory=lambda: ["eval"])
    """A list of tags associated with the evaluation."""
    concurrency: int = Config(default=1)
    """Maximum number of tasks to run in parallel."""
    iterations: int = Config(default=1, ge=1)
    """Number of times to run each scenario."""
    max_errors: int | None = Config(default=None)
    """
    Maximum number of errors (not caused by score assertions)
    to tolerate before stopping the evaluation.
    """
    max_consecutive_errors: int | None = Config(default=10)
    """
    The number of consecutive sample errors (not caused by score assertions)
    before terminating the evaluation run. Set to None to disable.
    """

    dataset_input_mapping: list[str] | dict[str, str] | None = None
    """
    A list of dataset keys to pass as input parameters to the task, or an
    explicit mapping from dataset keys to task parameter names.
    If None, will attempt to map keys that match parameter names.
    """
    parameters: dict[str, list[t.Any]] | None = Config(default=None)
    """
    A dictionary defining a parameter space to run experiments against.
    For each item in the dataset, a scenario will be run for every combination
    of the parameters defined here. Key names should align with
    arguments on the task assigned with a `Config` context.
    """

    preprocessor: InputDatasetProcessor | None = None
    """Optional preprocessor function to transform the dataset before evaluation."""
    scorers: ScorersLike[Out] = Config(default_factory=list)
    """Scorers to evaluate the task's output (appended to existing task scorers)."""
    assert_scores: list[str] | t.Literal[True] = Config(default_factory=list)
    """Scores to ensure are truthy, otherwise the task is marked as failed (appended to existing task assertions)."""

    trace: bool = True
    """Whether to produce trace contexts like runs/tasks for this study."""

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def name(self) -> str:
        return self.name_ or f"Eval {self.task_name}"

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def task_name(self) -> str:
        return self.task.name if isinstance(self.task, Task) else self.task.split(".")[-1]

    def __repr__(self) -> str:
        description = shorten_string(self.description or "", 50)

        parts: list[str] = [
            f"name='{self.name}'",
            f"description='{description}'",
            f"task={self.task!r}",
            f"dataset={self.dataset!r}",
        ]

        if self.parameters:
            parts.append(f"parameter_space={list(self.parameters.keys())}")
        if self.iterations > 1:
            parts.append(f"iterations={self.iterations}")
        if self.scorers:
            scorers = ", ".join(
                get_callable_name(scorer, short=True) for scorer in Scorer.fit_many(self.scorers)
            )
            parts.append(f"scorers=[{scorers}]")
        if self.assert_scores:
            parts.append(f"assertions={self.assert_scores}")
        if self.concurrency > 1:
            parts.append(f"concurrency={self.concurrency}")

        return f"{self.__class__.__name__}({', '.join(parts)})"

    @classmethod
    def _generic_types(cls) -> tuple[type[In], type[Out]]:
        for c in cls.__mro__:
            metadata = getattr(c, "__pydantic_generic_metadata__", {})
            if len(args := (metadata.get("args", ()) or getattr(c, "__args__", ()))) == 2:
                return args  # type: ignore[no-any-return]
        return t.Any, t.Any  # type: ignore[return-value]

    async def _prepare_task_and_dataset(self) -> tuple[Task[[In], Out], list[AnyDict]]:
        task = find(Task, self.task) if isinstance(self.task, str) else self.task

        dataset = self.dataset
        if isinstance(self.dataset, str | Path):
            dataset = load_dataset(self.dataset)

        input_type, _ = self._generic_types()
        dataset = TypeAdapter(list[input_type]).validate_python(dataset)  # type: ignore[valid-type]

        if self.preprocessor:
            dataset = self.preprocessor(dataset)

        return task, dataset  # type: ignore[return-value]

    def _get_param_combinations(self) -> list[dict[str, t.Any]]:
        if self.parameters:
            keys, values = zip(*self.parameters.items(), strict=False)
            return [dict(zip(keys, bundle, strict=False)) for bundle in itertools.product(*values)]
        return [{}]

    def _validate_scorers(self, scorers: list[Scorer[t.Any]], dataset_keys: list[str]) -> None:
        """
        Ensure that every scorer has only one required argument as this stage and that
        any DatasetField configurations align with our available dataset keys
        """

        for scorer in scorers:
            defaults = scorer.defaults
            required_params = [
                name for name, default in defaults.items() if isinstance(default, Unset)
            ]
            if len(required_params) > 1:
                raise ValueError(
                    f"Scorer '{scorer.name}' has more than one required parameter ({', '.join(required_params)}). "
                    "Configure default arguments directly, or use `.configure()` to pre-fill them. "
                    "Consider using a `DatasetField` to take values from your dataset "
                    "(e.g. `.configure(required=DatasetField('field_name'))`)."
                )

            dataset_params = {
                name: value for name, value in defaults.items() if isinstance(value, DatasetField)
            }
            for name, value in dataset_params.items():
                if value.ref_name not in dataset_keys:
                    raise ValueError(
                        f"Scorer '{scorer.name}' is configured to take parameter '{name}' from "
                        f"dataset field '{value.ref_name}', which is not available in the current dataset."
                    )

    @asynccontextmanager
    async def _run_iteration(
        self,
        configured_task: Task[[In], Out],
        dataset: list[AnyDict],
        scenario_params: AnyDict,
        iteration: int,
    ) -> t.AsyncIterator[t.AsyncGenerator[Sample[In, Out], None]]:
        async def _run_sample_with_context(index: int, row: AnyDict) -> Sample[In, Out]:
            token = current_dataset_row.set(row)
            try:
                if self.dataset_input_mapping:
                    if isinstance(self.dataset_input_mapping, list):
                        task_kwargs = {k: row[k] for k in self.dataset_input_mapping}
                    else:
                        task_kwargs = {
                            task_arg: row[ds_key]
                            for ds_key, task_arg in self.dataset_input_mapping.items()
                        }
                else:
                    task_params = set(configured_task.signature.parameters)
                    task_kwargs = {k: v for k, v in row.items() if k in task_params}

                context = {f"dataset_{k}": v for k, v in row.items() if k not in task_params}

                span = await configured_task.run_always(  # type: ignore[call-arg]
                    **{**task_kwargs, "__dn_ctx_inputs__": context}
                )

                first_kwarg = next(iter(task_kwargs.values()), None)
                task_input = task_kwargs if len(task_kwargs) > 1 else first_kwarg

                return Sample.from_task(
                    configured_task,
                    span,
                    task_input,
                    scenario_params=scenario_params,
                    iteration=iteration,
                    index=index,
                    context=context,
                )
            finally:
                current_dataset_row.reset(token)

        coroutines = [_run_sample_with_context(index, row) for index, row in enumerate(dataset)]
        async with concurrent_gen(coroutines, self.concurrency) as sample_stream:
            yield sample_stream

    async def _stream(self) -> t.AsyncGenerator[EvalEvent[In, Out], None]:
        from dreadnode import task_and_run

        base_task, dataset = await self._prepare_task_and_dataset()
        param_combinations = self._get_param_combinations()
        scorers = Scorer.fit_many(self.scorers or [])

        dataset_keys = list(dataset[0].keys()) if dataset else []  # We assume a homogeneous dataset
        self._validate_scorers(scorers, dataset_keys=dataset_keys)

        total_iterations = len(param_combinations) * self.iterations
        total_samples = total_iterations * len(dataset)

        yield EvalStart(
            eval=self,
            dataset_size=len(dataset),
            scenario_count=len(param_combinations),
            total_iterations=total_iterations,
            total_samples=total_samples,
        )

        configuration = get_config_model(self)()
        trace_inputs, trace_params = get_inputs_and_params_from_config_model(configuration)
        trace_params.pop("parameters", None)

        eval_result = EvalResult[In, Out](scenarios=[])
        for scenario_params in param_combinations:
            trace_context = (
                task_and_run(
                    name=self.name,
                    tags=self.tags,
                    inputs=trace_inputs,
                    params={**trace_params, **scenario_params},
                )
                if self.trace
                else contextlib.nullcontext()
            )

            with trace_context as task:
                run_id = task.run_id if task else ""
                yield ScenarioStart(
                    eval=self,
                    run_id=run_id,
                    scenario_params=scenario_params,
                    iteration_count=self.iterations,
                )

                configured_task = base_task.with_(
                    scorers=scorers,
                    assert_scores=self.assert_scores,
                    append=True,
                ).configure(**scenario_params)

                scenario_result = ScenarioResult[In, Out](params=scenario_params)
                consecutive_errors = 0
                total_errors = 0

                for i in range(self.iterations):
                    iteration = i + 1
                    yield IterationStart(
                        eval=self,
                        run_id=run_id,
                        scenario_params=scenario_params,
                        iteration=iteration,
                    )

                    iteration_result = IterationResult[In, Out](iteration=iteration)

                    async with self._run_iteration(
                        configured_task, dataset, scenario_params, iteration
                    ) as sample_stream:
                        async for sample in sample_stream:
                            yield SampleComplete(eval=self, run_id=run_id, sample=sample)
                            iteration_result.samples.append(sample)

                            if not sample.failed:
                                consecutive_errors = 0
                                continue

                            total_errors += 1
                            consecutive_errors += 1
                            max_errors_reached = self.max_errors and total_errors >= self.max_errors
                            max_consecutive_reached = (
                                self.max_consecutive_errors
                                and consecutive_errors >= self.max_consecutive_errors
                            )

                            if not max_errors_reached and not max_consecutive_reached:
                                continue

                            scenario_result.iterations.append(iteration_result)
                            eval_result.scenarios.append(scenario_result)

                            yield EvalEnd(
                                eval=self,
                                result=eval_result,
                                stop_reason="max_errors_reached"
                                if max_errors_reached
                                else "max_consecutive_errors_reached",
                            )
                            return

                    yield IterationEnd(eval=self, run_id=run_id, result=iteration_result)
                    scenario_result.iterations.append(iteration_result)

                yield ScenarioEnd(eval=self, run_id=run_id, result=scenario_result)
                eval_result.scenarios.append(scenario_result)

        yield EvalEnd(eval=self, result=eval_result, stop_reason="finished")

    @asynccontextmanager
    async def stream(self) -> t.AsyncIterator[t.AsyncGenerator[EvalEvent[In, Out], None]]:
        """Create an event stream to monitor the evaluation process."""
        async with contextlib.aclosing(self._stream()) as stream:
            yield stream

    async def run(self) -> EvalResult[In, Out]:
        """Run the configured task evaluation."""
        async with self.stream() as stream:
            async for event in stream:
                if isinstance(event, EvalEnd):
                    return event.result
        raise RuntimeError("Evaluation failed to complete")

    async def console(self) -> EvalResult:
        """Run the evaluation with a live display in the console."""

        adapter = EvalConsoleAdapter(self)
        return await adapter.run()

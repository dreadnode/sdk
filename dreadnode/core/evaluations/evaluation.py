from __future__ import annotations

import contextlib
import contextvars
import inspect
import itertools
import json
import typing as t
from contextlib import asynccontextmanager
from functools import cached_property
from typing import TYPE_CHECKING

import typing_extensions as te
from loguru import logger
from pydantic import (
    ConfigDict,
    Field,
    FilePath,
    TypeAdapter,
    field_validator,
    model_validator,
)

from dreadnode.core.discovery import find
from dreadnode.core.evaluations.events import (
    EvalEnd,
    EvalEvent,
    EvalStart,
    IterationEnd,
    IterationStart,
    SampleComplete,
    ScenarioEnd,
    ScenarioStart,
)
from dreadnode.core.evaluations.result import EvalResult, IterationResult, ScenarioResult
from dreadnode.core.evaluations.sample import Sample
from dreadnode.core.execution import (
    ConcurrentExecutor,
    TraceContext,
)
from dreadnode.core.meta import Config, DatasetField
from dreadnode.core.scorer import Scorer, ScorersLike
from dreadnode.core.types.common import AnyDict, Unset

if TYPE_CHECKING:
    from dreadnode.core.task import Task

from dreadnode.core.util import concurrent_gen, get_callable_name, shorten_string
from dreadnode.datasets import load_dataset

In = te.TypeVar("In", default=t.Any)
Out = te.TypeVar("Out", default=t.Any)

InputDataset = list[In]
InputDatasetProcessor = t.Callable[[InputDataset], InputDataset]
DatasetLike = InputDataset[In] | list[AnyDict]


class DatasetProducer(t.Protocol[In]):
    def __call__(self) -> t.Awaitable[DatasetLike[In]] | DatasetLike[In]: ...


DatasetOrProducer = DatasetLike | DatasetProducer

current_dataset_row = contextvars.ContextVar[t.Mapping[str, t.Any] | None](
    "current_dataset_row", default=None
)


class EvalWarning(UserWarning):
    """Warning raised during evaluation."""


class Evaluation(ConcurrentExecutor[EvalEvent[In, Out], EvalResult[In, Out]], t.Generic[In, Out]):
    """
    Prepared evaluation of a task with an associated dataset and configuration.

    Now extends ConcurrentExecutor for consistent streaming/tracing/error patterns.

    Attributes:
        task: The task to evaluate.
        dataset: The dataset to use for the evaluation.
        dataset_file: File path of a JSONL, CSV, JSON, or YAML dataset.
        name: The name of the evaluation.
        iterations: Number of times to run each scenario.
        dataset_input_mapping: Mapping from dataset keys to task parameter names.
        parameters: Parameter space for experiments.
        preprocessor: Optional preprocessor for the dataset.
        scorers: Scorers to evaluate task output.
        assert_scores: Scores to assert are truthy.
        trace: Whether to produce trace contexts.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    # Override base fields with eval-specific defaults
    name: str = ""  # Will be set in model_post_init if not provided
    tags: list[str] = Config(default_factory=lambda: ["eval"])

    task: Task[..., Out] | str
    dataset: t.Any | None = None
    dataset_file: FilePath | str | None = Config(default=None)
    iterations: int = Config(default=1, ge=1)
    dataset_input_mapping: list[str] | dict[str, str] | None = None
    parameters: dict[str, list[t.Any]] | None = Config(default=None, expose_as=str | None)
    preprocessor: InputDatasetProcessor | None = None
    scorers: ScorersLike[Out] = Field(default_factory=list)
    assert_scores: list[str] | t.Literal[True] = Field(default_factory=list)
    trace: bool = True

    def model_post_init(self, context: t.Any) -> None:
        super().model_post_init(context)
        if not self.name:
            self.name = f"Eval {self.task_name}"

    @model_validator(mode="after")
    def _check_dataset(self) -> te.Self:
        if self.dataset is None and self.dataset_file is None:
            raise ValueError("One of 'dataset' or 'dataset_file' must be provided.")
        return self

    @field_validator("parameters", mode="before")
    @classmethod
    def _deserialize_parameters(cls, value: t.Any) -> t.Any:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception as e:
                raise ValueError(f"Failed to parse parameters: {e}") from e
        return value

    @cached_property
    def task_name(self) -> str:
        # Check if task has a name attribute (Task object) vs string
        if hasattr(self.task, "name"):
            return self.task.name
        return self.task.split(".")[-1]

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
                get_callable_name(s, short=True) for s in Scorer.fit_many(self.scorers)
            )
            parts.append(f"scorers=[{scorers}]")
        if self.assert_scores:
            parts.append(f"assertions={self.assert_scores}")
        if self.concurrency > 1:
            parts.append(f"concurrency={self.concurrency}")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    def _extract_result(self, event: EvalEvent[In, Out]) -> EvalResult[In, Out] | None:
        """Extract result from EvalEnd event."""
        if isinstance(event, EvalEnd):
            return event.result
        return None

    def _should_trace(self) -> bool:
        """Respect the trace configuration."""
        return self.trace

    def _get_trace_context(self) -> TraceContext:
        """Build trace context with eval-specific information."""
        scorers = Scorer.fit_many(self.scorers or [])

        ctx = TraceContext.from_executor(self, exclude={"name", "tags"})
        ctx.span_type = "evaluation"

        ctx.inputs.update(
            {
                # Task being evaluated
                "task_name": self.task_name,
                # Scorers
                "scorers": [s.name for s in scorers],
                "assert_scores": list(self.assert_scores)
                if self.assert_scores is not True
                else ["*"],
                # Dataset mapping
                "dataset_input_mapping": self.dataset_input_mapping,
            }
        )
        ctx.params.update(
            {
                "iterations": self.iterations,
                "concurrency": self.concurrency,
                "max_errors": self.max_errors,
                "max_consecutive_errors": self.max_consecutive_errors,
            }
        )
        return ctx

    async def _stream(self) -> t.AsyncGenerator[EvalEvent[In, Out], None]:
        """Core evaluation execution loop."""
        from dreadnode import get_current_run, task_and_run

        base_task, dataset = await self._prepare_task_and_dataset()
        param_combinations = self._get_param_combinations()
        scorers = Scorer.fit_many(self.scorers or [])

        dataset_keys = list(dataset[0].keys()) if dataset else []
        self._validate_scorers(scorers, dataset_keys=dataset_keys)

        total_iterations = len(param_combinations) * self.iterations
        total_samples = total_iterations * len(dataset)

        inside_run = get_current_run() is not None
        log_to: t.Literal["task-or-run", "both"] = "task-or-run" if inside_run else "both"

        logger.info(
            f"Starting Eval '{self.name}': "
            f"task='{base_task.name}', "
            f"dataset_size={len(dataset)}, "
            f"scenarios={len(param_combinations)}, "
            f"total_samples={total_samples}, "
            f"concurrency={self.concurrency}, "
            f"iterations={self.iterations}"
        )

        yield EvalStart(
            evaluation=self,
            dataset_size=len(dataset),
            scenario_count=len(param_combinations),
            total_iterations=total_iterations,
            total_samples=total_samples,
        )

        eval_result = EvalResult[In, Out](scenarios=[])

        for scenario_params in param_combinations:
            scenario_result = ScenarioResult[In, Out](params=scenario_params)
            error_tracker = self._create_error_tracker()

            trace_context = (
                task_and_run(
                    name=self.name,
                    # Use evaluation name for the span, not the task name
                    # This makes the hierarchy clearer: Evaluation > Agent Task
                    task_type="evaluation",
                    tags=self.tags,
                    inputs=self._get_trace_context().inputs,
                    params={**self._get_trace_context().params, **scenario_params},
                    label=self.label,
                )
                if self.trace
                else contextlib.nullcontext()
            )

            def log_scenario_data(result: ScenarioResult[In, Out] = scenario_result) -> None:
                from dreadnode import log_outputs

                # Log discrete outputs
                log_outputs(
                    stop_reason=eval_result.stop_reason or "finished",
                    dataset_size=len(dataset),
                    total_samples=len(result.samples),
                    passed_count=result.passed_count,
                    failed_count=result.failed_count,
                    error_count=result.error_count,
                    pass_rate=result.pass_rate,
                    mean_scores=result.metrics_aggregated,
                    to=log_to,
                )

            with trace_context as task, contextlib.ExitStack() as stack:
                stack.callback(log_scenario_data, scenario_result)

                run_id = task.run_id if task else ""
                yield ScenarioStart(
                    evaluation=self,
                    run_id=run_id,
                    scenario_params=scenario_params,
                    iteration_count=self.iterations,
                    sample_count=self.iterations * len(dataset),
                )

                configured_task = base_task.with_(
                    scorers=scorers,
                    assert_scores=self.assert_scores,
                    append=True,
                ).configure(**scenario_params)

                for iteration in range(1, self.iterations + 1):
                    logger.debug(f"Starting iteration: {iteration}/{self.iterations}")

                    yield IterationStart(
                        evaluation=self,
                        run_id=run_id,
                        scenario_params=scenario_params,
                        iteration=iteration,
                    )

                    iteration_result = IterationResult[In, Out](iteration=iteration)

                    async with self._run_iteration(
                        configured_task, dataset, scenario_params, iteration
                    ) as sample_stream:
                        async for sample in sample_stream:
                            yield SampleComplete(evaluation=self, run_id=run_id, sample=sample)
                            iteration_result.samples.append(sample)

                            if not sample.failed:
                                error_tracker.record_success()
                                continue

                            if stop_reason := error_tracker.record_error():
                                scenario_result.iterations.append(iteration_result)
                                eval_result.scenarios.append(scenario_result)
                                eval_result.stop_reason = stop_reason

                                logger.warning(
                                    f"Stopping evaluation: reason='{stop_reason}', "
                                    f"consecutive_errors={error_tracker.consecutive_errors}, "
                                    f"total_errors={error_tracker.total_errors}"
                                )
                                yield EvalEnd(evaluation=self, result=eval_result)
                                return

                    yield IterationEnd(evaluation=self, run_id=run_id, result=iteration_result)
                    scenario_result.iterations.append(iteration_result)

                logger.info(
                    f"Finished scenario: pass_rate={scenario_result.pass_rate:.2%}, "
                    f"passed={scenario_result.passed_count}, failed={scenario_result.failed_count}"
                )

                yield ScenarioEnd(evaluation=self, run_id=run_id, result=scenario_result)
                eval_result.scenarios.append(scenario_result)

        eval_result.stop_reason = "finished"
        logger.success(
            f"Finished Eval '{self.name}': reason='{eval_result.stop_reason}', "
            f"passed={eval_result.passed_count}, failed={eval_result.failed_count}"
        )
        yield EvalEnd(evaluation=self, result=eval_result)

    def with_(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        label: str | None = None,
        task: Task[..., Out] | str | None = None,
        dataset: t.Any | None = None,
        iterations: int | None = None,
        concurrency: int | None = None,
        max_errors: int | None = None,
        max_consecutive_errors: int | None = None,
        scorers: ScorersLike[Out] | None = None,
        assert_scores: list[str] | t.Literal[True] | None = None,
        append: bool = False,
    ) -> te.Self:
        """Create a modified clone of the evaluation."""
        new = self.clone()

        # Apply simple updates
        updates = {
            "name_": name,
            "description": description,
            "label": label,
            "task": task,
            "dataset": dataset,
            "iterations": iterations,
            "concurrency": concurrency,
            "max_errors": max_errors,
            "max_consecutive_errors": max_consecutive_errors,
            "assert_scores": assert_scores,
        }
        for field, value in updates.items():
            if value is not None:
                setattr(new, field, value)

        # Apply list fields with append support
        new._apply_updates(
            {"tags": tags, "scorers": Scorer.fit_many(scorers) if scorers else None},
            list_fields={"tags", "scorers"},
            append=append,
        )

        return new

    async def console(self) -> EvalResult[In, Out]:
        """Run the evaluation with a live display in the console."""
        from dreadnode.evaluations.console import EvalConsoleAdapter

        adapter = EvalConsoleAdapter(self)
        return await adapter.run()

    @classmethod
    def _generic_types(cls) -> tuple[type[In], type[Out]]:
        # Look for concrete type arguments in Generic[In, Out], not the executor types
        for c in cls.__mro__:
            # Skip if this is the ConcurrentExecutor parameterization
            if hasattr(c, "__origin__") and c.__origin__ is not t.Generic:
                continue
            metadata = getattr(c, "__pydantic_generic_metadata__", {})
            args = metadata.get("args", ()) or getattr(c, "__args__", ())
            # We want 2 args that are actual types (not TypeVars or complex generics)
            if len(args) == 2:
                # Skip if args are TypeVars or generic types (like EvalEvent[In, Out])
                if all(not isinstance(a, t.TypeVar) and not hasattr(a, "__origin__") for a in args):
                    return args
        return t.Any, t.Any

    async def _prepare_task_and_dataset(self) -> tuple[Task[[In], Out], list[AnyDict]]:
        task = find(Task, self.task) if isinstance(self.task, str) else self.task

        dataset = self.dataset
        if self.dataset_file is not None:
            dataset = load_dataset(self.dataset_file)

        if inspect.isfunction(dataset):
            dataset = dataset()
            if inspect.isawaitable(dataset):
                dataset = await dataset

        input_type, _ = self._generic_types()
        # Only validate if we have a concrete input type (not Any)
        if input_type is not t.Any:
            dataset = TypeAdapter(list[input_type]).validate_python(dataset)
        elif not isinstance(dataset, list):
            dataset = list(dataset)

        if self.preprocessor:
            dataset = self.preprocessor(dataset)

        return task, dataset

    def _get_param_combinations(self) -> list[dict[str, t.Any]]:
        if self.parameters:
            keys, values = zip(*self.parameters.items(), strict=False)
            return [dict(zip(keys, bundle, strict=False)) for bundle in itertools.product(*values)]
        return [{}]

    def _validate_scorers(self, scorers: list[Scorer[t.Any]], dataset_keys: list[str]) -> None:
        for scorer in scorers:
            defaults = scorer.defaults
            required_params = [
                name for name, default in defaults.items() if isinstance(default, Unset)
            ]
            if len(required_params) > 1:
                raise ValueError(
                    f"Scorer '{scorer.name}' has more than one required parameter. "
                    "Configure defaults or use DatasetField."
                )

            dataset_params = {
                name: value for name, value in defaults.items() if isinstance(value, DatasetField)
            }
            for name, value in dataset_params.items():
                if value.ref_name not in dataset_keys:
                    raise ValueError(
                        f"Scorer '{scorer.name}' references dataset field '{value.ref_name}' "
                        "which is not available."
                    )

    @asynccontextmanager
    async def _run_iteration(
        self,
        configured_task: Task[[In], Out],
        dataset: list[AnyDict],
        scenario_params: AnyDict,
        iteration: int,
    ) -> t.AsyncIterator[t.AsyncGenerator[Sample[In, Out], None]]:
        dataset_size = len(dataset)

        async def _run_sample(index: int, row: AnyDict) -> Sample[In, Out]:
            token = current_dataset_row.set(row)
            try:
                task_params = set(configured_task.signature.parameters)
                if self.dataset_input_mapping:
                    if isinstance(self.dataset_input_mapping, list):
                        task_kwargs = {k: row[k] for k in self.dataset_input_mapping}
                    else:
                        task_kwargs = {
                            task_arg: row[ds_key]
                            for ds_key, task_arg in self.dataset_input_mapping.items()
                        }
                else:
                    task_kwargs = {k: v for k, v in row.items() if k in task_params}

                context = {f"dataset_{k}": v for k, v in row.items() if k not in task_kwargs}
                first_kwarg = next(iter(task_kwargs.values()), None)
                task_input = task_kwargs if len(task_kwargs) > 1 else first_kwarg

                # Create task with indexed name for better span identification
                indexed_task = configured_task.with_(
                    name=f"{configured_task.name} [{index + 1}/{dataset_size}]"
                )
                span = await indexed_task.run_always(
                    **{**task_kwargs, "__dn_ctx_inputs__": context}
                )

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

        coroutines = [_run_sample(index, row) for index, row in enumerate(dataset)]
        async with concurrent_gen(coroutines, self.concurrency) as sample_stream:
            yield sample_stream


def evaluation(
    func: t.Callable[..., t.Any] | None = None,
    /,
    *,
    dataset: DatasetLike | None = None,
    dataset_file: str | FilePath | None = None,
    name: str | None = None,
    description: str = "",
    tags: list[str] | None = None,
    concurrency: int = 1,
    iterations: int = 1,
    max_errors: int | None = None,
    max_consecutive_errors: int = 10,
    dataset_input_mapping: list[str] | dict[str, str] | None = None,
    parameters: dict[str, list[t.Any]] | None = None,
    preprocessor: InputDatasetProcessor | None = None,
    scorers: ScorersLike[t.Any] | None = None,
    assert_scores: list[str] | t.Literal[True] | None = None,
) -> Evaluation[t.Any, t.Any] | t.Callable[[t.Callable[..., t.Any]], Evaluation[t.Any, t.Any]]:
    """
    Create an Evaluation from a function. The function becomes the task.

    Can be used as a decorator:
        ```python
        @dreadnode.evaluation(dataset=[{"x": 1}, {"x": 2}])
        async def my_task(x: int) -> int:
            return x * 2

        result = await my_task.run()
        ```

    Args:
        func: The function to convert to a task and evaluate.
        dataset: The dataset to evaluate against.
        dataset_file: Path to a dataset file.
        name: Name of the evaluation.
        description: Description of the evaluation.
        tags: Tags for the evaluation.
        concurrency: Number of concurrent evaluations.
        iterations: Number of iterations per sample.
        max_errors: Maximum errors before stopping.
        max_consecutive_errors: Maximum consecutive errors.
        dataset_input_mapping: Mapping of dataset fields to task inputs.
        parameters: Parameter grid for scenarios.
        preprocessor: Function to preprocess the dataset.
        scorers: Scorers to evaluate outputs.
        assert_scores: Scores that must pass.

    Returns:
        An Evaluation instance or a decorator function.
    """
    from dreadnode.core.task import task as task_decorator

    def make_evaluation(fn: t.Callable[..., t.Any]) -> Evaluation[t.Any, t.Any]:
        """Create an evaluation from a function."""
        # Convert function to a task
        task_instance = task_decorator(fn)

        return Evaluation(
            task=task_instance,
            dataset=dataset,
            dataset_file=dataset_file,
            name=name or "",
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

    # Called as @evaluation on a function directly
    if func is not None:
        return make_evaluation(func)

    # Called as @evaluation(...) - return decorator
    return make_evaluation

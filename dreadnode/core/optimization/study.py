"""
Refactored Study using core execution abstractions.

This shows how the Study class would be simplified by using the core Executor base class.
"""

import asyncio
import contextlib
import contextvars
import typing as t

import typing_extensions as te
from loguru import logger
from pydantic import ConfigDict, Field, FilePath, SkipValidation

from dreadnode.core.evaluations import Evaluation, InputDataset
from dreadnode.core.exceptions import AssertionFailedError
from dreadnode.core.execution import ConcurrentExecutor, TraceContext
from dreadnode.core.meta import Config
from dreadnode.core.optimization.events import (
    NewBestTrialFound,
    StudyEnd,
    StudyEvent,
    StudyStart,
    TrialAdded,
    TrialComplete,
    TrialPruned,
    TrialStart,
)
from dreadnode.core.optimization.result import StudyResult, StudyStopReason
from dreadnode.core.optimization.stopping import StudyStopCondition
from dreadnode.core.optimization.trial import CandidateT, Trial
from dreadnode.core.scorer import Scorer, ScorerLike, ScorersLike
from dreadnode.core.search import OptimizationContext, Search
from dreadnode.core.task import Task
from dreadnode.core.types.common import AnyDict
from dreadnode.core.util import clean_str, stream_map_and_merge

OutputT = te.TypeVar("OutputT", default=t.Any)

Direction = t.Literal["maximize", "minimize"]
ObjectivesLike = t.Sequence[ScorerLike[OutputT] | str] | t.Mapping[str, ScorerLike[OutputT]]

current_trial = contextvars.ContextVar[Trial | None]("current_trial", default=None)


def fit_objectives(objectives: ObjectivesLike[OutputT]) -> t.Sequence[Scorer[OutputT] | str]:
    if isinstance(objectives, t.Mapping):
        return Scorer.fit_many(objectives)
    return [obj if isinstance(obj, str) else Scorer.fit(obj) for obj in objectives]


class Study(
    ConcurrentExecutor[StudyEvent[CandidateT], StudyResult[CandidateT]],
    t.Generic[CandidateT, OutputT],
):
    """
    Optimization study for hyperparameter tuning and experiment tracking.

    Now extends ConcurrentExecutor for consistent streaming/tracing/error patterns.

    Attributes:
        search_strategy: The search strategy to use for suggesting new trials.
        task_factory: Function that accepts a trial candidate and returns a configured Task.
        probe_task_factory: Optional function for probe trials.
        objectives: The objectives to optimize for.
        directions: Optimization direction for each objective.
        dataset: Dataset for evaluating each trial's task.
        probe_concurrency: Concurrency for probes (defaults to main concurrency).
        max_trials: Maximum number of total trials.
        constraints: Constraints to apply to trial candidates.
        stop_conditions: Conditions that will stop the study.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    # Override base with study-specific defaults
    name: str = ""  # Will be set in model_post_init if not provided
    tags: list[str] = Config(default_factory=lambda: ["study"])

    # Study-specific fields
    search_strategy: SkipValidation[Search[CandidateT]]
    task_factory: SkipValidation[t.Callable[[CandidateT], Task[..., OutputT]]]
    probe_task_factory: SkipValidation[t.Callable[[CandidateT], Task[..., OutputT]] | None] = None
    objectives: t.Annotated[ObjectivesLike[OutputT], Config(expose_as=None)]
    directions: list[Direction] = Config(default_factory=lambda: ["maximize"])
    dataset: InputDataset[t.Any] | list[AnyDict] | FilePath | None = Config(
        default=None, expose_as=FilePath | None
    )
    probe_concurrency: int | None = Config(default=None)
    max_trials: int = Config(default=100, ge=1)
    constraints: ScorersLike[CandidateT] | None = Field(default=None)
    stop_conditions: list[StudyStopCondition] = Field(default_factory=list)

    def model_post_init(self, context: t.Any) -> None:
        super().model_post_init(context)

        self.objectives = fit_objectives(self.objectives)
        self.constraints = Scorer.fit_many(self.constraints)
        self.directions = (
            ["maximize"] * len(self.objectives)
            if self.directions == ["maximize"]
            else self.directions
        )

        if len(self.directions) != len(self.objectives):
            raise ValueError(
                f"Number of directions ({len(self.directions)}) must match "
                f"number of objectives ({len(self.objectives)})."
            )

        if not self.name:
            objective_name = clean_str("_and_".join(self.objective_names))
            self.name = f"study - {objective_name}"

    @property
    def objective_names(self) -> list[str]:
        self.objectives = fit_objectives(self.objectives)
        return [o if isinstance(o, str) else o.name for o in self.objectives]

    def _extract_result(self, event: StudyEvent[CandidateT]) -> StudyResult[CandidateT] | None:
        """Extract result from StudyEnd event."""
        if isinstance(event, StudyEnd):
            return event.result
        return None

    def _get_trace_context(self) -> TraceContext:
        """Build trace context with study-specific information."""
        ctx = TraceContext.from_executor(self)
        ctx.span_type = "study"

        ctx.inputs.update(
            {
                # Search configuration
                "search_strategy": type(self.search_strategy).__name__,
                "objectives": self.objective_names,
                "directions": self.directions,
                # Constraints
                "constraints": [c.name for c in Scorer.fit_many(self.constraints)],
                "stop_conditions": [s.name for s in self.stop_conditions],
            }
        )
        ctx.params.update(
            {
                "max_trials": self.max_trials,
                "concurrency": self.concurrency,
                "max_errors": self.max_errors,
                "max_consecutive_errors": self.max_consecutive_errors,
                "objective_count": len(self.objectives),
            }
        )
        return ctx

    async def _stream(self) -> t.AsyncGenerator[StudyEvent[CandidateT], None]:
        """Core study execution loop."""
        stop_reason: StudyStopReason = "unknown"
        stop_explanation: str | None = None
        all_trials: list[Trial[CandidateT]] = []
        all_probes: list[Trial[CandidateT]] = []
        best_trial: Trial[CandidateT] | None = None
        stop_condition_met = False

        optimization_context = OptimizationContext(
            objective_names=self.objective_names,
            directions=self.directions,
        )

        semaphore = asyncio.Semaphore(self.concurrency)
        probe_semaphore = (
            asyncio.Semaphore(self.probe_concurrency) if self.probe_concurrency else semaphore
        )

        logger.info(
            f"Starting study '{self.name}': "
            f"max_trials={self.max_trials}, "
            f"concurrency={self.concurrency}, "
            f"objectives={self.objective_names}, "
            f"directions={self.directions}"
        )

        yield StudyStart(
            study=self, trials=all_trials, probes=all_probes, max_trials=self.max_trials
        )

        async def process_search(
            item: Trial[CandidateT],
        ) -> t.AsyncGenerator[StudyEvent[CandidateT], None]:
            nonlocal all_trials, all_probes

            semaphore_to_use = probe_semaphore if item.is_probe else semaphore

            try:
                if item.is_probe:
                    all_probes.append(item)
                else:
                    item.step = len(all_trials)
                    all_trials.append(item)

                yield TrialAdded(study=self, trials=all_trials, probes=all_probes, trial=item)

                async with semaphore_to_use:
                    async for event in self._process_trial(item):
                        event.trials = list(all_trials)
                        event.probes = list(all_probes)
                        yield event
            finally:
                with contextlib.suppress(asyncio.InvalidStateError):
                    item._future.set_result(item)

        async with stream_map_and_merge(
            source=self.search_strategy(optimization_context),
            processor=process_search,
            limit=self.max_trials,
            concurrency=self.concurrency * 2,
        ) as events:
            async for event in events:
                yield event

                if isinstance(event, (TrialComplete, TrialPruned)):
                    if (
                        not event.trial.is_probe
                        and event.trial.status == "finished"
                        and (best_trial is None or event.trial.score > best_trial.score)
                    ):
                        best_trial = event.trial
                        logger.success(
                            f"New best trial: id={best_trial.id}, score={best_trial.score:.5f}"
                        )
                        yield NewBestTrialFound(
                            study=self, trials=all_trials, probes=all_probes, trial=best_trial
                        )

                    for stop_condition in self.stop_conditions:
                        if stop_condition(all_trials):
                            logger.info(f"Stop condition '{stop_condition.name}' met.")
                            stop_explanation = stop_condition.name
                            stop_condition_met = True
                            break

                if stop_condition_met:
                    break

        stop_reason = (
            "stop_condition_met"
            if stop_condition_met
            else "max_trials_reached"
            if len(all_trials) >= self.max_trials
            else "search_exhausted"
        )

        logger.info(
            f"Study '{self.name}' finished: stop_reason={stop_reason}, "
            f"total_trials={len(all_trials)}, best_score={best_trial.score if best_trial else '-'}"
        )

        yield StudyEnd(
            study=self,
            trials=all_trials,
            probes=all_probes,
            result=StudyResult(
                trials=all_trials,
                stop_reason=stop_reason,
                stop_explanation=stop_explanation,
            ),
        )

    async def _stream_traced(self) -> t.AsyncGenerator[StudyEvent[CandidateT], None]:
        """Override to add study-specific logging."""
        from dreadnode import get_current_run, log_outputs

        log_to: t.Literal["both", "task-or-run"] = (
            "both" if get_current_run() is None else "task-or-run"
        )

        last_event: StudyEvent[CandidateT] | None = None

        def log_study(event: StudyEvent[CandidateT] | None) -> None:
            if not isinstance(event, StudyEnd):
                return

            result = event.result
            outputs: AnyDict = {
                # Completion info
                "stop_reason": result.stop_reason,
                "stop_explanation": result.stop_explanation,
                # Trial counts
                "completed_trials": len(result.trials),
                "finished_trials": len([t for t in result.trials if t.status == "finished"]),
                "failed_trials": len([t for t in result.trials if t.status == "failed"]),
                "pruned_trials": len([t for t in result.trials if t.status == "pruned"]),
            }

            if result.best_trial:
                outputs["best_trial_index"] = result.best_trial.step
                outputs["best_score"] = result.best_trial.score
                outputs["best_candidate"] = result.best_trial.candidate
                outputs["best_scores"] = result.best_trial.scores
                for name in self.objective_names:
                    outputs[f"best_{name}"] = result.best_trial.scores.get(name, -float("inf"))

            log_outputs(to=log_to, **outputs)

        with contextlib.ExitStack() as stack:
            stack.callback(log_study, last_event)

            async for event in super()._stream_traced():
                last_event = event
                self._log_event_metrics(event)
                yield event

    def _log_event_metrics(self, event: StudyEvent[CandidateT]) -> None:
        """Log metrics for specific event types."""
        from dreadnode import log_metric

        if isinstance(event, TrialComplete):
            trial = event.trial
            log_metric(f"{trial.status}_trials", 1, step=trial.step, mode="count")
            if trial.status == "finished":
                log_metric("trial_score", trial.score, step=trial.step)
        elif isinstance(event, NewBestTrialFound):
            log_metric("best_score", event.trial.score, step=event.trial.step)

    def with_(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        search_strategy: Search[CandidateT] | None = None,
        task_factory: t.Callable[[CandidateT], Task[..., OutputT]] | None = None,
        objectives: ObjectivesLike[OutputT] | None = None,
        directions: list[Direction] | None = None,
        dataset: InputDataset[t.Any] | list[AnyDict] | FilePath | None = None,
        concurrency: int | None = None,
        constraints: ScorersLike[CandidateT] | None = None,
        max_trials: int | None = None,
        stop_conditions: list[StudyStopCondition] | None = None,
        append: bool = False,
    ) -> te.Self:
        """Clone the study and modify its attributes."""
        new = self.clone()

        # Apply simple updates
        updates = {
            "name": name,
            "description": description,
            "search_strategy": search_strategy,
            "task_factory": task_factory,
            "dataset": dataset,
            "concurrency": concurrency,
            "max_trials": max_trials,
        }
        for field, value in updates.items():
            if value is not None:
                setattr(new, field, value)

        new_objectives = fit_objectives(objectives) if objectives is not None else []
        new_directions = directions or ["maximize"] * len(new_objectives)

        if append:
            new.tags = [*new.tags, *(tags or [])]
            new.objectives = [*fit_objectives(new.objectives), *new_objectives]
            new.directions = [*new.directions, *new_directions]
            new.stop_conditions = [*new.stop_conditions, *(stop_conditions or [])]
            new.constraints = [*Scorer.fit_many(new.constraints), *Scorer.fit_many(constraints)]
        else:
            new.tags = tags if tags is not None else new.tags
            new.objectives = new_objectives if objectives is not None else new.objectives
            new.directions = new_directions if directions is not None else new.directions
            new.stop_conditions = (
                stop_conditions if stop_conditions is not None else new.stop_conditions
            )
            new.constraints = constraints if constraints is not None else new.constraints

        return new

    async def console(self) -> StudyResult[CandidateT]:
        """Run with live progress dashboard."""
        from dreadnode.core.optimization.console import StudyConsoleAdapter

        adapter = StudyConsoleAdapter(self)
        return await adapter.run()

    async def _process_trial(
        self, trial: Trial[CandidateT]
    ) -> t.AsyncIterator[StudyEvent[CandidateT]]:
        """Process a single trial."""
        from dreadnode import log_inputs, log_metrics, log_outputs
        from dreadnode.core.tracing.spans import trial_span

        task_factory = (
            self.probe_task_factory
            if trial.is_probe and self.probe_task_factory
            else self.task_factory
        )
        task = task_factory(trial.candidate)
        dataset = trial.dataset or self.dataset or [{}]
        probe_or_trial = "probe" if trial.is_probe else "trial"

        trial_token = current_trial.set(trial)

        def log_trial(trial: Trial[CandidateT]) -> None:
            log_outputs(status=trial.status, evaluation_result=trial.evaluation_result)
            if trial.evaluation_result:
                log_metrics(
                    {
                        "total_samples": len(trial.evaluation_result.samples),
                        "pass_rate": trial.evaluation_result.pass_rate,
                        **trial.evaluation_result.metrics_aggregated,
                    },
                    step=trial.step,
                )

        with (
            trial_span(
                trial_id=trial.id,
                step=trial.step,
                candidate=trial.candidate
                if isinstance(trial.candidate, dict)
                else {"value": trial.candidate},
                task_name=task.name,
                is_probe=trial.is_probe,
                tags=[probe_or_trial],
            ) as span,
            contextlib.ExitStack() as stack,
        ):
            stack.callback(log_trial, trial)
            stack.callback(current_trial.reset, trial_token)

            log_inputs(candidate=trial.candidate, dataset=dataset)

            try:
                trial.status = "running"
                yield TrialStart(study=self, trials=[], probes=[], trial=trial)

                # Check constraints
                if not trial.is_probe and self.constraints:
                    await Scorer.evaluate(
                        trial.candidate,
                        Scorer.fit_many(self.constraints),
                        step=trial.step,
                        assert_scores=True,
                    )

                # Evaluate
                scorers: list[Scorer[OutputT]] = [
                    s for s in fit_objectives(self.objectives) if isinstance(s, Scorer)
                ]

                evaluator = Evaluation(
                    task=task,
                    dataset=dataset,
                    scorers=scorers,
                    max_consecutive_errors=self.max_consecutive_errors,
                    max_errors=self.max_errors,
                    trace=False,
                )

                trial.evaluation_result = await evaluator.run()

                if trial.evaluation_result.stop_reason in (
                    "max_errors_reached",
                    "max_consecutive_errors_reached",
                ) or all(s.failed for s in trial.evaluation_result.samples):
                    first_error = next(s.error for s in trial.evaluation_result.samples if s.failed)
                    raise RuntimeError(first_error)

                for i, name in enumerate(self.objective_names):
                    direction = self.directions[i]
                    raw_score = trial.all_scores.get(name, -float("inf"))
                    directional_score = raw_score if direction == "maximize" else -raw_score
                    trial.scores[name] = raw_score
                    trial.directional_scores[name] = directional_score

                trial.score = (
                    sum(trial.directional_scores.values()) / len(trial.directional_scores)
                    if trial.directional_scores
                    else 0.0
                )

                trial.status = "finished"
                span.set_attribute("dreadnode.trial.status", "finished")
                span.set_attribute("dreadnode.trial.scores", trial.scores)

            except AssertionFailedError as e:
                span.add_tags(["pruned"])
                span.set_exception(e)
                span.set_attribute("dreadnode.trial.status", "pruned")
                trial.status = "pruned"
                trial.pruning_reason = f"Constraint not satisfied: {e}"

            except Exception as e:
                span.set_exception(e)
                span.set_attribute("dreadnode.trial.status", "failed")
                trial.status = "failed"
                trial.error = str(e)

        if trial.status == "pruned":
            yield TrialPruned(study=self, trials=[], probes=[], trial=trial)
        else:
            yield TrialComplete(study=self, trials=[], probes=[], trial=trial)

    def add_objective(
        self,
        objective: ScorerLike[OutputT],
        *,
        direction: Direction = "maximize",
        name: str | None = None,
    ) -> te.Self:
        self.objectives = [
            *fit_objectives(self.objectives),
            objective
            if isinstance(objective, str)
            else Scorer[OutputT].fit(objective).with_(name=name),
        ]
        self.directions = [*self.directions, direction]
        return self

    def add_constraint(self, constraint: ScorerLike[CandidateT]) -> te.Self:
        self.constraints = [*Scorer.fit_many(self.constraints), Scorer.fit(constraint)]
        return self

    def add_stop_condition(self, condition: StudyStopCondition) -> te.Self:
        self.stop_conditions.append(condition)
        return self


def study(
    func: t.Callable[[CandidateT], Task[t.Any, OutputT]] | None = None,
    /,
    *,
    name: str | None = None,
    search_strategy: Search[CandidateT] | None = None,
    dataset: InputDataset | None = None,
    dataset_file: str | FilePath | None = None,
    objectives: ObjectivesLike[OutputT] | None = None,
    directions: list[Direction] | None = None,
    constraints: ScorersLike[CandidateT] | None = None,
    max_trials: int = 100,
    concurrency: int = 1,
    stop_conditions: list[StudyStopCondition] | None = None,
    max_errors: int | None = None,
    max_consecutive_errors: int = 10,
) -> (
    Study[CandidateT, OutputT]
    | t.Callable[[t.Callable[[CandidateT], Task[t.Any, OutputT]]], Study[CandidateT, OutputT]]
):
    """
    Create a Study from a task factory function.

    Can be used as a decorator:
        ```python
        from dreadnode.search import grid

        @dreadnode.study(
            search_strategy=grid({"multiplier": [1, 2, 3]}),
            dataset=[{"x": 1}, {"x": 2}],
            objectives=[accuracy_scorer],
        )
        def make_task(params):
            @dreadnode.task
            async def my_task(x: int) -> int:
                return x * params["multiplier"]
            return my_task

        result = await make_task.run()
        ```

    Args:
        func: The task factory function to decorate.
        name: Name of the study (defaults to function name).
        search_strategy: The search strategy (grid, random, optuna).
        dataset: The dataset to evaluate against.
        dataset_file: Path to a dataset file.
        objectives: Scorers to optimize.
        directions: Optimization directions ("maximize" or "minimize").
        constraints: Constraint scorers.
        max_trials: Maximum number of trials.
        concurrency: Number of concurrent trials.
        stop_conditions: Conditions for early stopping.
        max_errors: Maximum errors before stopping.
        max_consecutive_errors: Maximum consecutive errors.

    Returns:
        A Study instance or a decorator function.
    """
    from dreadnode.core.util import get_callable_name

    def make_study(
        fn: t.Callable[[CandidateT], Task[t.Any, OutputT]],
    ) -> Study[CandidateT, OutputT]:
        """Create a study from a task factory function."""
        fn_name = get_callable_name(fn, short=True)

        return Study(
            name=name or fn_name,
            task_factory=fn,
            search_strategy=search_strategy,
            dataset=dataset,
            dataset_file=dataset_file,
            objectives=objectives or [],
            directions=directions or ["maximize"],
            constraints=constraints,
            max_trials=max_trials,
            concurrency=concurrency,
            stop_conditions=stop_conditions or [],
            max_errors=max_errors,
            max_consecutive_errors=max_consecutive_errors,
        )

    # Called as @study on a function directly
    if func is not None:
        return make_study(func)

    # Called as @study(...) - return decorator
    return make_study

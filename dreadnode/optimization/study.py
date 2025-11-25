import asyncio
import contextlib
import contextvars
import typing as t

import typing_extensions as te
from loguru import logger
from pydantic import ConfigDict, Field, FilePath, SkipValidation, computed_field

from dreadnode.common_types import AnyDict
from dreadnode.error import AssertionFailedError
from dreadnode.eval import InputDataset
from dreadnode.eval.eval import Eval
from dreadnode.eval.hooks.base import EvalHook
from dreadnode.meta import Config, Model
from dreadnode.meta.introspect import (
    get_config_model,
    get_inputs_and_params_from_config_model,
)
from dreadnode.optimization.console import StudyConsoleAdapter
from dreadnode.optimization.events import (
    NewBestTrialFound,
    StudyEnd,
    StudyEvent,
    StudyStart,
    TrialAdded,
    TrialComplete,
    TrialPruned,
    TrialStart,
)
from dreadnode.optimization.result import StudyResult, StudyStopReason
from dreadnode.optimization.search import Search
from dreadnode.optimization.search.base import OptimizationContext
from dreadnode.optimization.stop import StudyStopCondition
from dreadnode.optimization.trial import CandidateT, Trial
from dreadnode.scorers import Scorer, ScorerLike, ScorersLike
from dreadnode.task import Task
from dreadnode.util import (
    clean_str,
    stream_map_and_merge,
)

OutputT = te.TypeVar("OutputT", default=t.Any)

Direction = t.Literal["maximize", "minimize"]
"""The direction of optimization for the objective score."""
ObjectivesLike = t.Sequence[ScorerLike[OutputT] | str] | t.Mapping[str, ScorerLike[OutputT]]
"""The objectives to optimize for."""

current_trial = contextvars.ContextVar[Trial | None]("current_trial", default=None)
"""The currently running trial, if any."""


class Study(Model, t.Generic[CandidateT, OutputT]):
    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    name_: str | None = Field(default=None, repr=False, exclude=False, alias="name")
    """The name of the study - otherwise derived from the objective."""
    description: str = ""
    """A brief description of the study's purpose."""
    tags: list[str] = Config(default_factory=lambda: ["study"])
    """A list of tags associated with the study for logging."""
    label: str | None = Config(default=None)
    """Specific label for tracing, otherwise derived from the name."""

    search_strategy: SkipValidation[Search[CandidateT]]
    """The search strategy to use for suggesting new trials."""
    task_factory: SkipValidation[t.Callable[[CandidateT], Task[..., OutputT]]]
    """A function that accepts a trial candidate and returns a configured Task ready for evaluation."""
    probe_task_factory: SkipValidation[t.Callable[[CandidateT], Task[..., OutputT]] | None] = None
    """
    An optional function that accepts a probe candidate and returns a Task.

    Otherwise the main task_factory will be used for both full evaluation Trials and probe Trials.
    """
    objectives: t.Annotated[ObjectivesLike[OutputT], Config(expose_as=None)]
    """
    The objectives to optimize for.

    Can be a list/dict of scorer-like callables or string names of scorers already on the task.
    """
    directions: list[Direction] = Config(default_factory=lambda: ["maximize"])
    """
    The directions of optimization for the objective score.

    The length must match the number of objectives.
    """

    dataset: InputDataset[t.Any] | list[AnyDict] | FilePath | None = Config(
        default=None, expose_as=FilePath | None
    )
    """
    The dataset to use for the evaluation. Can be a list of inputs or a file path to load inputs from.
    If `None`, an empty dataset with a single empty input will be used - in other words the task will
    simply be evaluated once per trial.

    This dataset will be used to evaluate each trial's task during scoring - the mean average score
    for all metrics will be used as the trial's singular objective scores.
    """

    # Config
    concurrency: int = Config(default=1, ge=1)
    """The maximum number of trials to evaluate in parallel."""
    probe_concurrency: int | None = Config(default=None)
    """The maximum number of probes to evaluate in parallel. If not supplied, probes share concurrency with trials."""
    max_evals: int = Config(default=100, ge=1)
    """The maximum number of total evaluations to perform across all trials and probes."""
    max_errors: int | None = Config(default=None)
    """Maximum number of trial evaluation errors to tolerate before stopping the evaluation."""
    max_consecutive_errors: int | None = Config(default=10)
    """
    The number of consecutive trial evaluation errors to tolerate
    before terminating the evaluation run. Set to None to disable.
    """

    constraints: ScorersLike[CandidateT] | None = Field(default=None)
    """A list of Scorer-like constraints to apply to trial candidates. If any constraint scores to a falsy value, the candidate is pruned."""
    hooks: list[EvalHook] = Field(default_factory=list, exclude=True, repr=False)
    """Hooks to run at various points in the study/evaluation lifecycle."""

    stop_conditions: list[StudyStopCondition[CandidateT]] = Field(default_factory=list)
    """A list of conditions that, if any are met, will stop the study."""

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
                f"The number of directions ({len(self.directions)}) must match the "
                f"number of objectives ({len(self.objectives)})."
            )

    @property
    def objective_names(self) -> list[str]:
        self.objectives = fit_objectives(self.objectives)
        return [o if isinstance(o, str) else o.name for o in self.objectives]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def name(self) -> str:
        objective_name = clean_str("_and_".join(self.objective_names))
        return self.name_ or f"study - {objective_name}"

    def clone(self) -> te.Self:
        """
        Clone the study.

        Returns:
            A new Study instance with the same attributes as this one.
        """
        return self.model_copy(deep=True)

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
        stop_conditions: list[StudyStopCondition[CandidateT]] | None = None,
        append: bool = False,
    ) -> te.Self:
        """
        Clone the study and modify its attributes.

        Returns:
            A new Study instance with the modified attributes.
        """
        new = self.clone()

        new.name_ = name or new.name
        new.description = description or new.description
        new.search_strategy = search_strategy or new.search_strategy
        new.task_factory = task_factory or new.task_factory
        new.dataset = dataset if dataset is not None else new.dataset
        new.concurrency = concurrency or new.concurrency
        new.max_evals = max_trials or new.max_evals

        new_objectives = fit_objectives(objectives) if objectives is not None else []
        new_directions = directions or ["maximize"] * len(new_objectives)
        if len(new_directions) != len(new_objectives):
            raise ValueError(
                f"The number of directions ({len(new_directions)}) must match the "
                f"number of objectives ({len(new_objectives)})."
            )

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

    def add_stop_condition(self, condition: StudyStopCondition[CandidateT]) -> te.Self:
        self.stop_conditions.append(condition)
        return self

    async def _process_trial(
        self, trial: Trial[CandidateT]
    ) -> t.AsyncIterator[StudyEvent[CandidateT]]:
        """
        Checks constraints and evaluates a single trial, returning a list of events.
        """
        from dreadnode import log_inputs, log_metrics, log_outputs, task_span

        logger.debug(
            f"Processing trial: id={trial.id}, step={trial.step}, is_probe={trial.is_probe}"
        )

        task_factory = (
            self.probe_task_factory
            if trial.is_probe and self.probe_task_factory
            else self.task_factory
        )
        dataset = trial.dataset or self.dataset or [{}]
        probe_or_trial = "probe" if trial.is_probe else "trial"

        token = current_trial.set(trial)

        def log_trial(trial: Trial[CandidateT]) -> None:
            log_outputs(
                status=trial.status,
                eval_result=trial.eval_result,
            )
            if trial.eval_result:
                log_metrics(
                    {
                        "total_samples": len(trial.eval_result.samples),
                        "pass_rate": trial.eval_result.pass_rate,
                        "passed_count": trial.eval_result.passed_count,
                        "failed_count": trial.eval_result.failed_count,
                        "error_count": trial.eval_result.error_count,
                        **trial.eval_result.metrics_aggregated,
                    },
                    step=trial.step,
                )

        with (
            task_span(
                name=f"{probe_or_trial} - {self.name}",
                tags=[probe_or_trial],
            ) as span,
            contextlib.ExitStack() as stack,
        ):
            stack.callback(log_trial, trial)
            stack.callback(current_trial.reset, token)

            log_inputs(
                candidate=trial.candidate,
                dataset=dataset,
            )

            try:
                trial.status = "running"
                yield TrialStart(study=self, trials=[], probes=[], trial=trial)

                # Check constraints
                await self._check_constraints(trial.candidate, trial)

                # Create task
                task = task_factory(trial.candidate)

                # Get base scorers
                scorers: list[Scorer[OutputT]] = [
                    scorer
                    for scorer in fit_objectives(self.objectives)
                    if isinstance(scorer, Scorer)
                ]

                # Run evaluation (transforms are applied inside Eval now)
                trial.eval_result = await self._run_evaluation(task, dataset, scorers, trial)

                # Extract final scores
                self._extract_trial_scores(trial)

                trial.status = "finished"
                logger.info(
                    f"Completed trial: id={trial.id}, status='{trial.status}', score={trial.score}"
                )
                logger.trace(f"Output: {trial.output}")
                logger.trace(f"Scores: {trial.scores}")
                logger.trace(f"Eval Result: {trial.eval_result}")

            except AssertionFailedError as e:
                span.add_tags(["pruned"])
                span.set_exception(e)
                trial.status = "pruned"
                trial.pruning_reason = f"Constraint not satisfied: {e}"
                logger.warning(f"Pruned trial: id={trial.id}, reason='{trial.pruning_reason}'")

            except Exception as e:  # noqa: BLE001
                span.set_exception(e)
                trial.status = "failed"
                trial.error = str(e)
                logger.warning(f"Failed trial: id={trial.id}, error='{trial.error}'", exc_info=True)

        if trial.status == "pruned":
            yield TrialPruned(study=self, trials=[], probes=[], trial=trial)
        else:
            yield TrialComplete(study=self, trials=[], probes=[], trial=trial)

    async def _check_constraints(self, candidate: CandidateT, trial: Trial[CandidateT]) -> None:
        """Check constraints on the candidate."""
        from dreadnode import score as dn_score

        if trial.is_probe or not self.constraints:
            return

        logger.info(
            f"Checking constraints: trial_id={trial.id}, num_constraints={len(self.constraints)}"
        )

        await dn_score(
            candidate,
            Scorer.fit_many(self.constraints),
            step=trial.step,
            assert_scores=True,
        )

    async def _run_evaluation(
        self,
        task: Task[..., OutputT],
        dataset: t.Any,
        scorers: list[Scorer[OutputT]],
        trial: Trial[CandidateT],
    ) -> t.Any:
        """Run the evaluation with the given task, dataset, and scorers."""
        logger.debug(
            f"Evaluating trial: "
            f"trial_id={trial.id}, "
            f"step={trial.step}, "
            f"dataset_size={len(dataset) if isinstance(dataset, t.Sized) else '<unknown>'}, "
            f"task={task.name}"
        )
        logger.trace(f"Candidate: {trial.candidate!r}")

        if dataset == [{}] or (isinstance(dataset, list) and len(dataset) == 1 and not dataset[0]):
            # Dataset is empty - this is a Study/Attack where the candidate IS the input
            dataset = [{"message": trial.candidate}]
            dataset_input_mapping = ["message"]
        else:
            dataset_input_mapping = None

        evaluator = Eval(
            task=task,
            dataset=dataset,
            dataset_input_mapping=dataset_input_mapping,
            scorers=scorers,
            hooks=self.hooks,
            max_consecutive_errors=self.max_consecutive_errors,
            max_errors=self.max_errors,
            trace=False,
        )

        eval_result = await evaluator.run()

        if eval_result.stop_reason in (
            "max_errors_reached",
            "max_consecutive_errors_reached",
        ) or all(sample.failed for sample in eval_result.samples):
            first_error = next(sample.error for sample in eval_result.samples if sample.failed)
            raise RuntimeError(first_error)

        return eval_result

    def _extract_trial_scores(self, trial: Trial[CandidateT]) -> None:
        """Extract and calculate final scores for the trial."""
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

    async def _stream(self) -> t.AsyncGenerator[StudyEvent[CandidateT], None]:
        """
        Execute the complete optimization study and yield events for each phase.
        """
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

        # For concurrency, we want a few things:
        #
        # 1. Limit the number of trials being evaluated at once as it
        # may be expensive - `concurrency` is a direct reflection of this
        #
        # 2. Prevent us from reading too far ahead in the search strategy,
        # which might be an expensive op as well - and trials might be discarded
        # if we stop early. For this we'll scale concurrency by a factor of 2
        # for the stream_map_and_merge so we can get the pending items immediately
        # and issue a TrialAdded event, but we will set in_queue_size to the concurrency
        # so we don't read too far ahead.

        semaphore = asyncio.Semaphore(self.concurrency)  # we'll use this to
        probe_semaphore = (
            asyncio.Semaphore(self.probe_concurrency) if self.probe_concurrency else semaphore
        )

        logger.info(
            f"Starting study '{self.name}': "
            f"max_evals={self.max_evals}, "
            f"concurrency={self.concurrency}, "
            f"probe_concurrency={self.probe_concurrency or self.concurrency}, "
            f"objectives={self.objective_names}, "
            f"directions={self.directions}, "
            f"constraints={len(self.constraints) if self.constraints else 0}, "
            f"stop_conditions={len(self.stop_conditions)}"
        )

        yield StudyStart(
            study=self, trials=all_trials, probes=all_probes, max_trials=self.max_evals
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
                    item._future.set_result(item)  # noqa: SLF001

        async with stream_map_and_merge(
            source=self.search_strategy(optimization_context),
            processor=process_search,
            limit=self.max_evals,
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
                            f"New best trial: "
                            f"id={best_trial.id}, "
                            f"step={best_trial.step}, "
                            f"score={best_trial.score:.5f}, "
                            f"scores={best_trial.scores}"
                        )
                        yield NewBestTrialFound(
                            study=self, trials=all_trials, probes=all_probes, trial=best_trial
                        )

                    for stop_condition in self.stop_conditions:
                        if stop_condition(all_trials):
                            logger.info(
                                f"Stop condition '{stop_condition.name}' met. Terminating study."
                            )
                            stop_explanation = stop_condition.name
                            stop_condition_met = True
                            break

                if stop_condition_met:
                    break

        stop_reason = (
            "stop_condition_met"
            if stop_condition_met
            else "max_trials_reached"
            if len(all_trials) >= self.max_evals
            else "search_exhausted"
        )

        logger.info(
            f"Study '{self.name}' finished: "
            f"stop_reason={stop_reason}, "
            f"total_trials={len(all_trials)}, "
            f"total_probes={len(all_probes)}, "
            f"best_score={best_trial.score if best_trial else '-'}, "
            f"best_trial_id={best_trial.id if best_trial else '-'}"
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

    def _log_event_metrics(self, event: StudyEvent[CandidateT]) -> None:
        from dreadnode import log_metric

        if isinstance(event, TrialComplete):
            trial = event.trial
            log_metric(f"{trial.status}_trials", 1, step=trial.step, mode="count")
            if trial.status == "finished":
                log_metric("trial_score", trial.score, step=trial.step)

        elif isinstance(event, NewBestTrialFound):
            log_metric("best_score", event.trial.score, step=event.trial.step)

    async def _stream_traced(self) -> t.AsyncGenerator[StudyEvent[CandidateT], None]:
        from dreadnode import get_current_run, log_outputs, task_and_run

        configuration = get_config_model(self)()
        trace_inputs, trace_params = get_inputs_and_params_from_config_model(configuration)
        log_to: t.Literal["both", "task-or-run"] = (
            "both" if get_current_run() is None else "task-or-run"
        )

        last_event: StudyEvent[CandidateT] | None = None

        def log_study(event: StudyEvent[CandidateT] | None) -> None:
            if not isinstance(event, StudyEnd):
                return

            result = event.result
            outputs: AnyDict = {"stop_reason": result.stop_reason}
            if result.best_trial:
                outputs["best_score"] = result.best_trial.score
                for name in self.objective_names:
                    outputs[f"best_{name}"] = result.best_trial.scores.get(name, -float("inf"))
                outputs["best_candidate"] = result.best_trial.candidate
                outputs["best_output"] = result.best_trial.output
            log_outputs(to=log_to, **outputs)

        with (
            task_and_run(
                name=self.name,
                tags=self.tags,
                inputs=trace_inputs,
                params=trace_params,
                label=self.label,
            ),
            contextlib.ExitStack() as stack,
        ):
            stack.callback(log_study, last_event)
            async with contextlib.aclosing(self._stream()) as stream:
                async for event in stream:
                    last_event = event
                    self._log_event_metrics(event)
                    yield event

    @contextlib.asynccontextmanager
    async def stream(
        self,
    ) -> t.AsyncIterator[t.AsyncGenerator[StudyEvent[CandidateT], None]]:
        """
        Create an async context manager for the optimization event stream.

        This provides a safe way to access the optimization event stream with proper
        resource cleanup. The context manager ensures the async generator is properly
        closed to prevent tracing issues.

        Usage:
            async with study.stream() as event_stream:
                async for event in event_stream:
                    # Process optimization events
                    pass

        Yields:
            An async generator that produces StudyEvent objects throughout the optimization.
        """
        async with contextlib.aclosing(self._stream_traced()) as gen:
            yield gen

    async def run(self) -> StudyResult[CandidateT]:
        """
        Execute the optimization study to completion and return final results.

        This is a convenience method that runs the full optimization process and
        returns only the final StudyEnd event containing the complete results.
        Use this when you want the final results without processing intermediate events.

        For real-time monitoring of the optimization process, use the stream() method instead.

        Raises:
            RuntimeError: If the evaluation fails to complete properly.
        """
        async with self.stream() as stream:
            async for event in stream:
                if isinstance(event, StudyEnd):
                    return event.result

        raise RuntimeError("Evaluation failed to complete")

    async def console(self) -> StudyResult[CandidateT]:
        """Runs the optimization study with a live progress dashboard in the console."""

        adapter = StudyConsoleAdapter(self)
        return await adapter.run()


def fit_objectives(objectives: ObjectivesLike[OutputT]) -> t.Sequence[Scorer[OutputT] | str]:
    if isinstance(objectives, t.Mapping):
        return Scorer.fit_many(objectives)

    return [obj if isinstance(obj, str) else Scorer.fit(obj) for obj in objectives]

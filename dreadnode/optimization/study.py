import asyncio
import contextlib
import contextvars
import typing as t

import typing_extensions as te
from pydantic import ConfigDict, Field, FilePath, SkipValidation, computed_field

from dreadnode.common_types import AnyDict
from dreadnode.error import AssertionFailedError
from dreadnode.eval import Eval, InputDataset
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


class Study(Model, t.Generic[CandidateT, OutputT]):
    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    name_: str | None = Field(default=None, repr=False, exclude=False, alias="name")
    """The name of the study - otherwise derived from the objective."""
    description: str = ""
    """A brief description of the study's purpose."""
    tags: list[str] = Config(default_factory=lambda: ["study"])
    """A list of tags associated with the study for logging."""

    search_strategy: SkipValidation[Search[CandidateT]]
    """The search strategy to use for suggesting new trials."""
    task_factory: t.Callable[[CandidateT], Task[..., OutputT]]
    """A function that accepts a candidate and returns a configured Task ready for evaluation."""
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
    constraints: ScorersLike[CandidateT] | None = Config(default=None)
    """A list of Scorer-like constraints to apply to candidates. If any constraint scores to a falsy value, the candidate is pruned."""
    max_trials: int = Config(default=100, ge=1)
    """The maximum number of trials to evaluate."""
    stop_conditions: list[StudyStopCondition[CandidateT]] = Config(default_factory=list)
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
        Clone a task.

        Returns:
            A new Task instance with the same attributes as this one.
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
        new.max_trials = max_trials or new.max_trials

        new_tags = tags or []
        new_objectives = fit_objectives(objectives) if objectives is not None else []
        new_directions = directions or ["maximize"] * len(new_objectives)
        new_stop_conditions = stop_conditions or []
        new_constraints = Scorer.fit_many(constraints) if constraints is not None else []

        if len(new_directions) != len(new_objectives):
            raise ValueError(
                f"The number of directions ({len(new_directions)}) must match the "
                f"number of objectives ({len(new_objectives)})."
            )

        if append:
            new.tags = [*new.tags, *new_tags]
            new.objectives = [*fit_objectives(new.objectives), *new_objectives]
            new.directions = [*new.directions, *new_directions]
            new.stop_conditions = [*new.stop_conditions, *new_stop_conditions]
            new.constraints = [*Scorer.fit_many(new.constraints), *new_constraints]
        else:
            new.tags = new_tags or new.tags
            new.objectives = new_objectives or new.objectives
            new.directions = new_directions or new.directions
            new.stop_conditions = new_stop_conditions or new.stop_conditions
            new.constraints = new_constraints or new.constraints

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

        This worker function is designed to be run concurrently. It mutates the
        input trial object with the results of the evaluation.
        """
        from dreadnode import score as dn_score
        from dreadnode import task_span

        task = self.task_factory(trial.candidate)

        token = current_trial.set(trial)

        try:
            trial.status = "running"
            yield TrialStart(study=self, trials=[], trial=trial)

            with task_span(
                name=f"trial - {task.name}",
                tags=["trial"],
            ):
                # Check constraints

                await dn_score(
                    trial.candidate,
                    Scorer.fit_many(self.constraints),
                    step=trial.step,
                    assert_scores=True,
                )

                # Get the task

                scorers: list[Scorer[OutputT]] = [
                    scorer
                    for scorer in fit_objectives(self.objectives)
                    if isinstance(scorer, Scorer)
                ]

                # TODO(nick): Add max_errors* settings to study so
                # then can be passed down the to eval.
                evaluator = Eval(
                    task=task,
                    dataset=self.dataset or [{}],
                    scorers=scorers,
                    trace=False,
                )

                trial.eval_result = await evaluator.run()

                # If our entire evaluation failed, reflect that in the trial
                # status so it can be handled appropriately upstream.
                #
                # TODO(nick): Certainly some different options here depending
                # on how the study behaves, ideally we would have it reflect
                # this issue in the trial_result?
                if all(sample.failed for sample in trial.eval_result.samples):
                    first_error = next(
                        sample.error for sample in trial.eval_result.samples if sample.failed
                    )
                    raise ValueError(f"All samples failed during evaluation: {first_error}")  # noqa: TRY301

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

        except AssertionFailedError as e:
            trial.status = "pruned"
            trial.pruning_reason = f"Constraint not satisfied: {e}"

        except Exception as e:  # noqa: BLE001
            trial.status = "failed"
            trial.error = str(e)

        finally:
            current_trial.reset(token)

        if trial.status == "pruned":
            yield TrialPruned(study=self, trials=[], trial=trial)
        else:
            yield TrialComplete(study=self, trials=[], trial=trial)

    async def _stream(self) -> t.AsyncGenerator[StudyEvent[CandidateT], None]:
        """
        Execute the complete optimization study and yield events for each phase.
        """
        stop_reason: StudyStopReason = "unknown"
        stop_explanation: str | None = None
        all_trials: list[Trial[CandidateT]] = []
        best_trial: Trial[CandidateT] | None = None
        semaphore = asyncio.Semaphore(self.concurrency)
        stop_condition_met = False
        optimization_context = OptimizationContext(
            objective_names=self.objective_names,
            directions=self.directions,
        )
        yield StudyStart(study=self, trials=all_trials, max_trials=self.max_trials)

        async def process_trial(
            trial: Trial[CandidateT],
        ) -> t.AsyncGenerator[StudyEvent[CandidateT], None]:
            nonlocal semaphore
            nonlocal all_trials

            try:
                trial.step = len(all_trials)
                all_trials.append(trial)

                yield TrialAdded(study=self, trials=all_trials, trial=trial)

                async with semaphore:
                    async for event in self._process_trial(trial):
                        event.trials = all_trials
                        yield event
            finally:
                trial._future.set_result(trial)  # noqa: SLF001

        async with stream_map_and_merge(
            source=self.search_strategy(optimization_context),
            processor=process_trial,
            limit=self.max_trials,
        ) as events:
            async for event in events:
                yield event

                if isinstance(event, (TrialComplete, TrialPruned)):
                    if event.trial.status == "finished" and (
                        best_trial is None or event.trial.score > best_trial.score
                    ):
                        best_trial = event.trial
                        yield NewBestTrialFound(study=self, trials=all_trials, trial=best_trial)

                    for stop_condition in self.stop_conditions:
                        if stop_condition(all_trials):
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

        yield StudyEnd(
            study=self,
            trials=all_trials,
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
        with task_and_run(name=self.name, tags=self.tags, inputs=trace_inputs, params=trace_params):
            try:
                async with contextlib.aclosing(self._stream()) as stream:
                    async for event in stream:
                        last_event = event
                        self._log_event_metrics(event)
                        yield event
            finally:
                if isinstance(last_event, StudyEnd):
                    result = last_event.result
                    outputs: AnyDict = {"stop_reason": result.stop_reason}
                    if result.best_trial:
                        outputs["best_score"] = result.best_trial.score
                        outputs["best_candidate"] = result.best_trial.candidate
                        outputs["best_output"] = result.best_trial.output
                    log_outputs(to=log_to, **outputs)

    @contextlib.asynccontextmanager
    async def stream(
        self,
    ) -> t.AsyncIterator[t.AsyncGenerator[StudyEvent[CandidateT], None]]:
        """
        Create an async context manager for the optimization event stream.

        This provides a safe way to access the optimization event stream with proper
        resource cleanup. The context manager ensures the async generator is properly
        closed even if an exception occurs during iteration.

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

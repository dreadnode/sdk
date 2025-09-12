import asyncio
import contextlib
import contextvars
import typing as t
from copy import deepcopy

import typing_extensions as te
from pydantic import ConfigDict, FilePath, PrivateAttr, computed_field

from dreadnode.common_types import AnyDict
from dreadnode.error import AssertionFailedError
from dreadnode.eval import Eval, InputDataset
from dreadnode.meta import Config, Model
from dreadnode.optimization.console import StudyConsoleAdapter
from dreadnode.optimization.events import (
    NewBestTrialFound,
    StepEnd,
    StepStart,
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
    is_homogeneous_list,
    stream_map_and_merge,
)

OutputT = te.TypeVar("OutputT", default=t.Any)

Direction = t.Literal["maximize", "minimize"]
"""The direction of optimization for the objective score."""
ObjectiveLike = ScorerLike[OutputT] | ScorersLike[OutputT] | str | list[str]
"""
A single or multiple optimization objective(s). Can be any of:

- Single scorer instance or a scorer-like callable
- String name of any scorer already configured on the task.
- List/dict of multiple scorer instances or scorer-like callables (multi-objective).
- List of string names of scorers already on the task (multi-objective).
"""
current_trial = contextvars.ContextVar[Trial | None]("current_trial", default=None)


class Study(Model, t.Generic[CandidateT, OutputT]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name_: str | None = Config(default=None, repr=False, exclude=False, alias="name")
    """The name of the study - otherwise derived from the objective."""
    description: str = Config(default="")
    """A brief description of the study's purpose."""
    tags: list[str] = Config(default_factory=lambda: ["study"])
    """A list of tags associated with the study for logging."""

    search_strategy: t.Annotated[Search[CandidateT], Config(expose_as=None)]
    """The search strategy to use for suggesting new trials."""
    task_factory: t.Callable[[CandidateT], Task[..., OutputT]]
    """A function that accepts a candidate and returns a configured Task ready for evaluation."""
    objective: t.Annotated[ObjectiveLike[OutputT], Config(expose_as=None)]
    """
    The objective(s) to optimize for. Can be any of:

    - Single scorer instance or a scorer-like callable
    - String name of any scorer already configured on the task.
    - List/dict of multiple scorer instances or scorer-like callables (multi-objective).
    - List of string names of scorers already on the task (multi-objective).
    """
    direction: Direction | list[Direction] = Config(default="maximize")
    """
    The direction(s) of optimization for the objective score.

    If multiple directions are specified, the length must match
    the number of objectives.
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
    max_steps: int = Config(default=100, ge=1)
    """The maximum number of optimization steps to run."""
    stop_conditions: list[StudyStopCondition[CandidateT]] = Config(default_factory=list)
    """A list of conditions that, if any are met, will stop the study."""

    # Private fields for parsed state (we flex our init types above)
    _objectives: t.Sequence[Scorer[OutputT] | str] = PrivateAttr(default_factory=list)
    _directions: list[Direction] = PrivateAttr(default_factory=list)
    _constraints: list[Scorer[CandidateT]] = PrivateAttr(default_factory=list)

    def model_post_init(self, context: t.Any) -> None:
        super().model_post_init(context)

        objectives: t.Sequence[Scorer[t.Any] | str]
        if isinstance(self.objective, str):
            objectives = [self.objective]
        elif is_homogeneous_list(self.objective, str):
            objectives = self.objective
        elif isinstance(self.objective, Scorer) or callable(self.objective):
            objectives = [Scorer.fit(self.objective)]
        elif isinstance(self.objective, t.Mapping):
            objectives = Scorer.fit_many(self.objective)
        else:
            objectives = [
                objective if isinstance(objective, str) else Scorer.fit(objective)
                for objective in self.objective
            ]

        self._objectives = objectives

        self._constraints = Scorer.fit_many(self.constraints)
        self._directions = (
            [self.direction] * len(self._objectives)
            if isinstance(self.direction, str)
            else self.direction
        )

        if isinstance(self.direction, list) and len(self.direction) != len(self._objectives):
            raise ValueError(
                f"The number of directions ({len(self.direction)}) must match the "
                f"number of objectives ({len(self._objectives)})."
            )

    @property
    def objective_names(self) -> list[str]:
        return [o if isinstance(o, str) else o.name for o in self._objectives]

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
        return self.model_copy()

    def with_(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        search_strategy: Search[CandidateT] | None = None,
        task_factory: t.Callable[[CandidateT], Task[..., OutputT]] | None = None,
        objective: ObjectiveLike[OutputT] | None = None,
        direction: Direction | list[Direction] | None = None,
        dataset: InputDataset[t.Any] | list[AnyDict] | FilePath | None = None,
        concurrency: int | None = None,
        constraints: ScorersLike[CandidateT] | None = None,
        max_steps: int | None = None,
        stop_conditions: list[StudyStopCondition[CandidateT]] | None = None,
    ) -> te.Self:
        """
        Clone the study and modify its attributes.

        Returns:
            A new Study instance with the modified attributes.
        """
        return self.model_copy(
            update={
                "name_": name or self.name_,
                "description": description or self.description,
                "tags": tags or self.tags,
                "search_strategy": search_strategy or self.search_strategy,
                "task_factory": task_factory or self.task_factory,
                "objective": objective or self.objective,
                "direction": direction or self.direction,
                "dataset": dataset if dataset is not None else self.dataset,
                "concurrency": concurrency or self.concurrency,
                "constraints": constraints if constraints is not None else self.constraints,
                "max_steps": max_steps or self.max_steps,
                "stop_conditions": stop_conditions or self.stop_conditions,
            }
        )

    def add_objective(
        self,
        objective: ScorerLike[OutputT],
        *,
        direction: Direction = "maximize",
        name: str | None = None,
    ) -> te.Self:
        self._objectives = [
            *self._objectives,
            objective
            if isinstance(objective, str)
            else Scorer[OutputT].fit(objective).with_(name=name),
        ]
        self._directions = [*self._directions, direction]
        return self

    def add_constraint(self, constraint: ScorerLike[CandidateT]) -> te.Self:
        self._constraints = [*self._constraints, Scorer.fit(constraint)]
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

        try:
            token = current_trial.set(trial)
            task = self.task_factory(trial.candidate)

            trial.status = "running"
            yield TrialStart(study=self, trials=[], trial=trial)

            with task_span(
                name=f"trial - {task.name}",
                tags=["trial"],
            ):
                # Check constraints

                await dn_score(
                    trial.candidate,
                    self._constraints,
                    step=trial.step,
                    assert_scores=True,
                )

                # Get the task

                scorers: list[Scorer[OutputT]] = [
                    scorer for scorer in self._objectives if isinstance(scorer, Scorer)
                ]

                evaluator = Eval(
                    task=task,
                    dataset=self.dataset or [{}],
                    scorers=scorers,
                    trace=False,
                )

                trial.eval_result = await evaluator.run()

                for i, name in enumerate(self.objective_names):
                    direction = self._directions[i]
                    raw_score = trial.all_scores.get(name, -float("inf"))
                    adjusted_score = raw_score if direction == "maximize" else -raw_score
                    trial.scores[name] = adjusted_score

                trial.score = (
                    sum(trial.scores.values()) / len(trial.scores) if trial.scores else 0.0
                )

                trial.status = "finished"

        except AssertionFailedError:
            trial.status = "pruned"
            trial.pruning_reason = ""

        except Exception as e:  # noqa: BLE001
            trial.status = "failed"
            trial.error = str(e)

        finally:
            current_trial.reset(token)

        if trial.status == "pruned":
            yield TrialPruned(study=self, trials=[], trial=trial)
        else:
            yield TrialComplete(study=self, trials=[], trial=trial)

    def _reset(self) -> None:
        self.search_strategy = deepcopy(self.search_strategy)
        self.search_strategy.reset(
            OptimizationContext(
                objective_names=self.objective_names,
                directions=self._directions,
            )
        )

    async def _stream(self) -> t.AsyncGenerator[StudyEvent[CandidateT], None]:
        """
        Execute the complete optimization study and yield events for each phase.
        """
        self._reset()

        stop_reason: StudyStopReason = "unknown"
        stop_explanation: str | None = None
        all_trials: list[Trial[CandidateT]] = []
        best_trial: Trial[CandidateT] | None = None

        yield StudyStart(study=self, trials=all_trials, max_steps=self.max_steps)

        for step in range(1, self.max_steps + 1):
            yield StepStart(study=self, trials=all_trials, step=step)

            step_trials: list[Trial[CandidateT]] = []
            semaphore = asyncio.Semaphore(self.concurrency)

            async def process_trial(
                trial: Trial[CandidateT],
            ) -> t.AsyncIterator[StudyEvent[CandidateT]]:
                nonlocal semaphore

                yield TrialAdded(study=self, trials=[], trial=trial)

                async with semaphore:  # noqa: B023
                    async for event in self._process_trial(trial):
                        yield event

            async with stream_map_and_merge(
                source=self.search_strategy.suggest(step),
                processor=process_trial,
            ) as events:
                async for event in events:
                    if isinstance(event, TrialStart):
                        all_trials.append(event.trial)
                        step_trials.append(event.trial)

                    event.trials = all_trials

                    if isinstance(event, (TrialComplete, TrialPruned)):  # noqa: SIM102
                        if best_trial is None or event.trial.score > best_trial.score:
                            best_trial = event.trial
                            yield NewBestTrialFound(study=self, trials=all_trials, trial=best_trial)

                    yield event

            if not step_trials:
                stop_reason = "search_exhausted"
                yield StepEnd(study=self, trials=all_trials, step=step)
                break

            await self.search_strategy.observe(step_trials)

            yield StepEnd(study=self, trials=all_trials, step=step)

            stop = False
            for stop_condition in self.stop_conditions:
                if stop_condition(all_trials):
                    stop_reason = "stop_condition_met"
                    stop_explanation = stop_condition.name
                    stop = True
                    break

            if stop:
                break

        yield StudyEnd(
            study=self,
            trials=all_trials,
            result=StudyResult(
                trials=all_trials, stop_reason=stop_reason, stop_explanation=stop_explanation
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
        from dreadnode import log_inputs, log_outputs, log_params, run, task_span
        from dreadnode.tracing.span import current_run_span

        run_using_tasks = current_run_span.get() is not None
        trace_context = (
            task_span(self.name, tags=self.tags)
            if run_using_tasks
            else run(name_prefix=self.name, tags=self.tags)
        )

        # config_model = get_config_model(self)
        # flat_config = {k: v for k, v in flatten_model(config_model()).items() if v is not None}
        flat_config: AnyDict = {}

        with trace_context:
            if run_using_tasks:
                log_inputs(**flat_config)
            else:
                log_params(**flat_config)

            last_event: StudyEvent[CandidateT] | None = None
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
                    log_outputs(**outputs)

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

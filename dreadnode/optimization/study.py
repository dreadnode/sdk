import contextlib
import contextvars
import typing as t

from pydantic import ConfigDict, FilePath, PrivateAttr

from dreadnode.eval import Eval
from dreadnode.eval.result import EvalResult
from dreadnode.eval.sample import InputDataset
from dreadnode.meta import Model
from dreadnode.meta.types import Config
from dreadnode.optimization.console import StudyConsoleAdapter
from dreadnode.optimization.events import (
    CandidatePruned,
    CandidatesSuggested,
    CandidateT,
    NewBestTrialFound,
    StepEnd,
    StepStart,
    StudyEnd,
    StudyEvent,
    StudyStart,
    TrialComplete,
)
from dreadnode.optimization.result import StudyResult
from dreadnode.optimization.search import Search
from dreadnode.optimization.trial import Trial, Trials
from dreadnode.scorers import Scorer, ScorerLike
from dreadnode.task import Task
from dreadnode.tracing.span import current_run_span
from dreadnode.types import AnyDict
from dreadnode.util import concurrent_gen, get_callable_name

Direction = t.Literal["maximize", "minimize"]
"""The direction of optimization for the objective score."""

current_trial = contextvars.ContextVar[Trial | None]("current_trial", default=None)


class Study(Model, t.Generic[CandidateT]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str | None = Config(default=None)
    """The name of the study - otherwise derived from the task."""
    description: str = Config(default="")
    """A brief description of the study's purpose."""
    tags: list[str] = Config(default_factory=lambda: ["study"])
    """A list of tags associated with the study."""

    strategy: t.Annotated[Search[CandidateT], Config()]
    """The search strategy to use for suggesting new trials."""
    task_factory: t.Callable[[CandidateT], Task[..., t.Any]]
    """A function that accepts a candidate and returns a configured Task ready for evaluation."""
    objective: ScorerLike[t.Any] | str = Config(default_factory=list)  # type: ignore[assignment]
    """The objective to optimize. Can be a scorer instance, a scorer-like callable, or a string name of scorer already on the task."""
    dataset: InputDataset[t.Any] | list[AnyDict] | FilePath | None = Config(
        default=None, expose_as=FilePath | None
    )
    """
    The dataset to use for the evaluation. Can be a list of inputs or a file path to load inputs from.
    If `None`, an empty dataset with a single empty input will be used.
    """

    # Config
    direction: Direction = Config(default="maximize")
    """The direction of optimization for the objective score."""
    concurrency: int = Config(default=1, ge=1)
    """The maximum number of trials to evaluate in parallel."""
    constraints: list[Scorer[CandidateT]] | None = Config(default=None)
    """A list of Scorer-like constraints to apply to candidates. If any constraint scores to a falsy value, the candidate is pruned."""
    objective_fn: t.Callable[[EvalResult], float] | None = Config(default=None)
    """An optional function to compute a score from a full EvalResult, overriding the default averaging behavior."""

    # Stopping conditions
    max_steps: int = Config(default=100, ge=1)
    """The maximum number of optimization steps to run."""
    patience: int | None = Config(default=None, ge=1)
    """The number of steps to wait for an improvement before stopping. If None, this is disabled."""
    target_score: float | None = Config(default=None)
    """A target score to achieve. The study will stop if a trial meets or exceeds this score."""

    _steps_since_best: int = PrivateAttr(0)

    async def _check_constraints(self, candidate: CandidateT) -> tuple[bool, str]:
        if not self.constraints:
            return True, ""

        for scorer in self.constraints:
            metric = await scorer.score(candidate)
            if not metric.value:
                return False, f"Failed assertion: {scorer.name} -> {metric}"

        return True, ""

    async def _evaluate_candidate(self, trial: Trial[CandidateT]) -> Trial[CandidateT]:
        task_variant = self.task_factory(trial.candidate)

        scorers: list[ScorerLike[t.Any]] = []
        objective_scorer_name: str

        if isinstance(self.objective, str):
            objective_scorer_name = self.objective
        else:
            scorer = Scorer(self.objective)
            scorers.append(scorer)
            objective_scorer_name = scorer.name

        token = current_trial.set(trial)

        try:
            evaluator = Eval(
                task=task_variant,
                dataset=self.dataset or [{}],
                scorers=scorers,
            )

            trial.eval_result = await evaluator.run()

            score = -float("inf")
            if self.objective_fn is not None:
                score = self.objective_fn(trial.eval_result)
            else:
                score = trial.eval_result.metrics_summary.get(objective_scorer_name, {}).get(
                    "mean", score
                )

            trial.score = score if self.direction == "maximize" else -score
            trial.status = "success"
        except Exception as e:  # noqa: BLE001
            trial.status = "failed"
            trial.error = str(e)
        finally:
            current_trial.reset(token)

        return trial

    async def _stream(self) -> t.AsyncGenerator[StudyEvent[CandidateT], None]:  # noqa: PLR0912, PLR0915
        """
        Execute the complete optimization study and yield events for each phase.
        """
        self._steps_since_best = 0
        self.strategy.reset()

        stop_reason: t.Any = "unknown"
        all_trials: Trials[CandidateT] = []  # type: ignore[assignment]
        best_trial: Trial[CandidateT] | None = None

        yield StudyStart(study=self, trials=all_trials, max_steps=self.max_steps)

        for step in range(1, self.max_steps + 1):
            yield StepStart(study=self, trials=all_trials, step=step)

            new_trials = await self.strategy.suggest(step)
            if not new_trials:
                stop_reason = "no_more_candidates"
                break

            all_trials.extend(new_trials)

            yield CandidatesSuggested(
                study=self, trials=all_trials, candidates=[trial.candidate for trial in new_trials]
            )

            pending_trials: list[Trial[CandidateT]] = []
            pruned_trials: list[Trial[CandidateT]] = []

            for trial in new_trials:
                try:
                    is_valid, reason = await self._check_constraints(trial.candidate)
                    if is_valid:
                        pending_trials.append(trial)
                    else:
                        trial.status = "pruned"
                        trial.pruning_reason = reason
                        pruned_trials.append(trial)
                        yield CandidatePruned(study=self, trials=all_trials, trial=trial)
                except Exception as e:  # noqa: BLE001, PERF203
                    trial.status = "failed"
                    trial.error = str(e)
                    pruned_trials.append(trial)

            if not pending_trials:
                yield StepEnd(study=self, trials=all_trials, step=step)
                continue

            new_best_found_this_step = False
            completed_trials: list[Trial[CandidateT]] = []

            async with concurrent_gen(
                [self._evaluate_candidate(trial) for trial in pending_trials], self.concurrency
            ) as results_stream:
                async for trial in results_stream:
                    completed_trials.append(trial)
                    yield TrialComplete(study=self, trials=all_trials, trial=trial)

                    if trial.status == "success" and (
                        best_trial is None or trial.score > best_trial.score
                    ):
                        best_trial = trial
                        new_best_found_this_step = True
                        yield NewBestTrialFound(study=self, trials=all_trials, trial=best_trial)

            await self.strategy.observe(new_trials)

            yield StepEnd(study=self, trials=all_trials, step=step)

            if new_best_found_this_step:
                self._steps_since_best = 0
            else:
                self._steps_since_best += 1

            # Check if we've met the target score
            if (
                new_best_found_this_step
                and self.target_score is not None
                and best_trial
                and best_trial.score >= self.target_score
            ):
                stop_reason = "target_score"
                break

            # Check if we've run out of patience
            if self.patience is not None and self._steps_since_best >= self.patience:
                stop_reason = "patience"
                break

        # Final event creation is updated to use StudyResult
        yield StudyEnd(
            study=self,
            trials=all_trials,
            result=StudyResult(trials=all_trials, stop_reason=stop_reason),
        )

    def _log_event_metrics(self, event: StudyEvent[t.Any]) -> None:
        from dreadnode import log_metric

        if isinstance(event, TrialComplete):
            trial = event.trial
            if trial.status == "success":
                log_metric("successful_trials", 1, step=trial.step, mode="count")
                log_metric("trial_score", trial.score, step=trial.step)
            elif trial.status == "failed":
                log_metric("failed_trials", 1, step=trial.step, mode="count")
            elif trial.status == "pruned":
                log_metric("pruned_trials", 1, step=trial.step, mode="count")
        elif isinstance(event, NewBestTrialFound):
            log_metric("best_score", event.trial.score, step=event.trial.step)

    async def _stream_traced(self) -> t.AsyncGenerator[StudyEvent[CandidateT], None]:
        from dreadnode import log_inputs, log_outputs, log_params, run, task_span

        objective_name = (
            self.objective
            if isinstance(self.objective, str)
            else get_callable_name(self.objective, short=True)
        )
        name = self.name or f"optimize-{objective_name}"

        run_using_tasks = current_run_span.get() is not None
        trace_context = (
            task_span(name, tags=self.tags)
            if run_using_tasks
            else run(name_prefix=name, tags=self.tags)
        )

        # config_model = get_config_model(self)
        # flat_config = {k: v for k, v in flatten_model(config_model()).items() if v is not None}
        flat_config: dict[str, t.Any] = {}

        with trace_context:
            if run_using_tasks:
                log_inputs(**flat_config)
            else:
                log_params(**flat_config)

            last_event: StudyEvent[t.Any] | None = None
            try:
                async with contextlib.aclosing(self._stream()) as stream:
                    async for event in stream:
                        last_event = event
                        self._log_event_metrics(event)
                        yield event
            finally:
                if isinstance(last_event, StudyEnd):
                    result = last_event.result
                    outputs = {"stop_reason": result.stop_reason}
                    if result.best_trial:
                        outputs["best_score"] = result.best_trial.score  # type: ignore[assignment]
                        outputs["best_candidate"] = result.best_trial.candidate
                        outputs["best_output"] = result.best_trial.output  # type: ignore[assignment]
                    log_outputs(**outputs)  # type: ignore[arg-type]

    @contextlib.asynccontextmanager
    async def stream(self) -> t.AsyncIterator[t.AsyncGenerator[StudyEvent[CandidateT], None]]:
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

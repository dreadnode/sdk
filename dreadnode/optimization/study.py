import contextlib
import typing as t

from pydantic import BaseModel, ConfigDict, Field, FilePath, PrivateAttr

from dreadnode.eval import Eval
from dreadnode.eval.dataset import EvalResult
from dreadnode.optimization.events import (
    CandidatePruned,
    CandidatesSuggested,
    CandidateT,
    NewBestTrialFound,
    StepEnd,
    StepStart,
    StopReason,
    StudyEnd,
    StudyEvent,
    StudyStart,
    TrialComplete,
)
from dreadnode.optimization.trial import Trial
from dreadnode.scorers.base import Scorer, ScorerLike
from dreadnode.task import Task
from dreadnode.types import AnyDict
from dreadnode.util import concurrent_gen

if t.TYPE_CHECKING:
    from dreadnode.optimization.search import Search


class Study(BaseModel, t.Generic[CandidateT]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    strategy: "Search[CandidateT]"
    apply_candidate_fn: t.Callable[[CandidateT], Task[..., t.Any]]
    dataset: list[AnyDict] | FilePath
    objective: ScorerLike[t.Any] | str

    objective_fn: t.Callable[[EvalResult], float] | None = None
    direction: t.Literal["maximize", "minimize"] = "maximize"
    max_steps: int = 100
    concurrency: int = 1
    constraints: list[Scorer[CandidateT]] | None = None
    patience: int | None = None
    target_score: float | None = None
    stop_reason: StopReason = "unknown"
    trials: list[Trial[CandidateT]] = Field(default_factory=list, repr=False)
    best_trial: Trial[CandidateT] | None = None

    _steps_since_best: int = PrivateAttr(0)

    async def _run_assertions(self, candidate: CandidateT) -> tuple[bool, str]:
        """
        Validate a candidate against all configured constraint scorers.

        This method checks if the candidate satisfies all constraints defined for the study.
        Constraints are boolean scorers that must return True for a candidate to be considered valid.

        Args:
            candidate: The candidate to validate against constraints.

        Returns:
            A tuple containing (is_valid, reason) where is_valid indicates if all
            constraints passed and reason provides details for any failed constraint.
        """
        if not self.constraints:
            return True, ""

        for scorer in self.constraints:
            metric = await scorer.score(candidate)
            if not metric.value:
                return False, f"Failed assertion: {scorer.name} -> {metric}"

        return True, ""

    async def _evaluate_candidate(self, trial: Trial[CandidateT]) -> Trial[CandidateT]:
        """
        Execute a complete evaluation of a candidate trial.

        This method performs the core evaluation workflow: applies the candidate to create
        a task variant, runs evaluation with the dataset and scorers, computes the final
        score, and updates the trial with results. Handles both string-based and
        callable objectives.

        Args:
            trial: The trial containing the candidate to evaluate.

        Returns:
            The updated trial with evaluation results, score, and status set.
            Status will be 'success' if evaluation completed or 'failed' if an exception occurred.
        """
        task_variant = self.apply_candidate_fn(trial.candidate)

        scorers: list[ScorerLike[t.Any]] = []
        objective_scorer_name: str

        if isinstance(self.objective, str):
            objective_scorer_name = self.objective
        else:
            scorer = Scorer(self.objective)
            scorers.append(scorer)
            objective_scorer_name = scorer.name

        try:
            evaluator = Eval(
                task=task_variant,
                dataset=self.dataset,
                scorers=scorers,
            )

            trial.eval_result = await evaluator.run()

            score = -float("inf")
            if self.objective_fn is not None:
                score = self.objective_fn(trial.eval_result)
            else:
                sample_scores = [
                    s.get_average_metric_value(objective_scorer_name)
                    for s in trial.eval_result.samples
                ]
                if sample_scores:
                    score = sum(sample_scores) / len(sample_scores)

            trial.score = score if self.direction == "maximize" else -score
            trial.status = "success"
        except Exception as e:  # noqa: BLE001
            trial.status = "failed"
            trial.error = str(e)
        return trial

    def _reset(self) -> None:
        self.trials = []
        self.best_trial = None
        self.stop_reason = "unknown"
        self._steps_since_best = 0

    async def _stream(self) -> t.AsyncGenerator[StudyEvent[CandidateT], None]:  # noqa: PLR0912, PLR0915
        """
        Execute the complete optimization study and yield events for each phase.

        This is the core optimization loop that coordinates the entire study execution.
        It manages the search strategy, candidate evaluation, constraint validation,
        and stopping criteria. The method yields a stream of events that provide
        real-time visibility into the optimization process.

        The optimization follows this workflow for each step:
        1. Request candidates from the search strategy
        2. Validate candidates against constraints (prune invalid ones)
        3. Evaluate remaining candidates concurrently
        4. Update best trial and notify strategy of results
        5. Check stopping criteria (target score, patience, max steps)

        Yields:
            StudyEvent objects representing different phases of the optimization:
            - StudyStart: Signals the beginning of optimization
            - StepStart/StepEnd: Mark the beginning and end of each optimization step
            - CandidatesSuggested: Reports candidates proposed by the search strategy
            - CandidatePruned: Reports candidates that failed constraint validation
            - TrialComplete: Reports completion of candidate evaluation
            - NewBestTrialFound: Reports when a new best score is achieved
            - StudyEnd: Signals completion with final results and stop reason
        """
        self._reset()

        yield StudyStart(
            study=self, initial_candidate=getattr(self.strategy, "initial_candidate", None)
        )

        for step in range(1, self.max_steps + 1):
            yield StepStart(study=self, step=step)

            candidates = await self.strategy.suggest(step)
            if not candidates:
                self.stop_reason = "no_more_candidates"
                break

            yield CandidatesSuggested(study=self, candidates=candidates)

            pending_trials: list[Trial[CandidateT]] = []
            pruned_trials: list[Trial[CandidateT]] = []

            for candidate in candidates:
                try:
                    is_valid, reason = await self._run_assertions(candidate)
                    trial = Trial(candidate=candidate, step=step)
                    if is_valid:
                        pending_trials.append(trial)
                    else:
                        trial.status = "pruned"
                        trial.pruning_reason = reason
                        pruned_trials.append(trial)
                        yield CandidatePruned(study=self, trial=trial)
                except Exception as e:  # noqa: BLE001, PERF203
                    trial = Trial(
                        candidate=candidate,
                        status="failed",
                        error=str(e),
                    )
                    pruned_trials.append(trial)

            if pruned_trials:
                self.trials.extend(pruned_trials)
                self.strategy.observe(pruned_trials)

            if not pending_trials:
                yield StepEnd(study=self, step=step)
                continue

            new_best_found_this_step = False
            completed_trials: list[Trial[CandidateT]] = []

            async with concurrent_gen(
                [self._evaluate_candidate(trial) for trial in pending_trials], self.concurrency
            ) as results_stream:
                async for trial in results_stream:
                    completed_trials.append(trial)
                    yield TrialComplete(study=self, trial=trial)

                    if trial.status == "success" and (
                        self.best_trial is None or trial.score > self.best_trial.score
                    ):
                        self.best_trial = trial
                        new_best_found_this_step = True
                        yield NewBestTrialFound(study=self, trial=self.best_trial)

            self.trials.extend(completed_trials)
            self.strategy.observe(completed_trials)
            yield StepEnd(study=self, step=step)

            if new_best_found_this_step:
                self._steps_since_best = 0
            else:
                self._steps_since_best += 1

            # Check if we've met the target score
            if (
                new_best_found_this_step
                and self.target_score is not None
                and self.best_trial
                and self.best_trial.score >= self.target_score
            ):
                self.stop_reason = "target_score"
                break

            # Check if we've run out of patience
            if self.patience is not None and self._steps_since_best >= self.patience:
                self.stop_reason = "patience"
                break

        yield StudyEnd(
            study=self, steps=step, stop_reason=self.stop_reason, best_trial=self.best_trial
        )

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
        async with contextlib.aclosing(self._stream()) as gen:
            yield gen

    async def run(self) -> StudyEnd[CandidateT]:
        """
        Execute the optimization study to completion and return final results.

        This is a convenience method that runs the full optimization process and
        returns only the final StudyEnd event containing the complete results.
        Use this when you want the final results without processing intermediate events.

        For real-time monitoring of the optimization process, use the stream() method instead.

        Returns:
            StudyEnd event containing the final optimization results including:
            - best_trial: The best trial found during optimization (or None)
            - steps: Total number of optimization steps completed
            - stop_reason: Why the optimization terminated

        Raises:
            RuntimeError: If the evaluation fails to complete properly.
        """
        async with self.stream() as stream:
            async for event in stream:
                if isinstance(event, StudyEnd):
                    return event
            raise RuntimeError("Evaluation failed to complete")

import inspect
import typing as t

from dreadnode.optimization.trial import CandidateT, Trial
from dreadnode.util import get_callable_name


class StudyStopCondition(t.Generic[CandidateT]):
    """
    A condition that determines when a study should stop, based on the
    complete history of trials. Conditions can be combined using & (AND) and | (OR).
    """

    def __init__(self, func: t.Callable[[list[Trial[CandidateT]]], bool], name: str | None = None):
        if name is None:
            unwrapped = inspect.unwrap(func)
            name = get_callable_name(unwrapped, short=True)

        self.func = func
        self.name = name

    def __repr__(self) -> str:
        return f"StudyStopCondition(name='{self.name}')"

    def __call__(self, trials: list[Trial[CandidateT]]) -> bool:
        return self.func(trials)

    def __and__(self, other: "StudyStopCondition[CandidateT]") -> "StudyStopCondition[CandidateT]":
        """Combines this condition with another using AND logic."""
        return StudyStopCondition(
            lambda trials: self(trials) and other(trials), name=f"{self.name}_and_{other.name}"
        )

    def __or__(self, other: "StudyStopCondition[CandidateT]") -> "StudyStopCondition[CandidateT]":
        """Combines this condition with another using OR logic."""
        return StudyStopCondition(
            lambda trials: self(trials) or other(trials), name=f"{self.name}_or_{other.name}"
        )


def score_value(
    metric_name: str | None = None,
    *,
    gt: float | None = None,
    gte: float | None = None,
    lt: float | None = None,
    lte: float | None = None,
) -> StudyStopCondition:
    """
    Terminates if a trial score value meets a given threshold.

    - If `metric_name` is provided, it checks that specific objective score from trial.scores.
    - If `metric_name` is None (the default), it checks the primary, average `trial.score`.

    If you are using multi-objective optimization, it is recommended to specify a `metric_name` to avoid ambiguity.

    Args:
        metric_name: The name of the metric to check.
        gt: Greater than threshold.
        gte: Greater than or equal to threshold.
        lt: Less than threshold.
        lte: Less than or equal to threshold.
    """

    def stop(trials: list[Trial]) -> bool:  # noqa: PLR0911
        finished_trials = [t for t in trials if t.status == "finished"]
        if not finished_trials:
            return False

        finished_trials = [t for t in trials if t.status == "finished"]

        if not finished_trials:
            return False
        for trial in finished_trials:
            value_to_check = trial.scores.get(metric_name) if metric_name else trial.score
            if value_to_check is None:
                continue

            if gt is not None and value_to_check > gt:
                return True
            if gte is not None and value_to_check >= gte:
                return True
            if lt is not None and value_to_check < lt:
                return True
            if lte is not None and value_to_check <= lte:
                return True

        return False

    return StudyStopCondition(stop, name=f"score_value({metric_name or 'score'})")


def score_plateau(
    patience: int, *, min_delta: float = 0, metric_name: str | None = None
) -> StudyStopCondition:
    """
    Stops the study if the best trial score does not improve over time.

    If you are using multi-objective optimization, it is recommended to specify a `metric_name` to avoid ambiguity.

    Args:
        patience: The number of steps to wait before stopping.
        min_delta: The minimum change in score to consider it an improvement.
        metric_name: The name of the metric to check - otherwise an average of all objective metrics are used.
    """

    def stop(
        trials: list[Trial], *, patience: int = patience, min_delta: float = min_delta
    ) -> bool:
        finished_trials = sorted(
            [t for t in trials if t.status == "finished"], key=lambda t: t.step
        )
        if not finished_trials:
            return False

        last_step = finished_trials[-1].step
        if last_step < patience:
            return False

        current_best_score = max(t.get_directional_score(metric_name) for t in finished_trials)

        historical_trials = [t for t in finished_trials if t.step <= (last_step - patience)]
        if not historical_trials:
            return False

        historical_best_score = max(t.get_directional_score(metric_name) for t in historical_trials)
        improvement = current_best_score - historical_best_score

        return improvement < min_delta if min_delta > 0 else improvement > 0

    return StudyStopCondition(stop, name=f"plateau({metric_name or 'score'}, p={patience})")


def pruned_ratio(ratio: float, min_trials: int = 10) -> StudyStopCondition:
    """
    Stops the study if the ratio of pruned trials exceeds a threshold.

    Args:
        ratio: The maximum allowed ratio of pruned trials.
        min_trials: The minimum number of trials before stopping.
    """

    def stop(trials: list[Trial], *, ratio: float = ratio, min_trials: int = min_trials) -> bool:
        if len(trials) < min_trials:
            return False

        pruned_count = sum(1 for t in trials if t.status == "pruned")
        current_ratio = pruned_count / len(trials)
        return current_ratio >= ratio

    return StudyStopCondition(stop, name=f"pruned_ratio({ratio:.0%})")


def failed_ratio(ratio: float, min_trials: int = 10) -> StudyStopCondition:
    """
    Stops the study if the ratio of failed trials exceeds a threshold.

    Args:
        ratio: The maximum allowed ratio of failed trials.
        min_trials: The minimum number of trials before stopping.
    """

    def stop(trials: list[Trial], *, ratio: float = ratio, min_trials: int = min_trials) -> bool:
        if len(trials) < min_trials:
            return False

        failed_count = sum(1 for t in trials if t.status == "failed")
        current_ratio = failed_count / len(trials)
        return current_ratio >= ratio

    return StudyStopCondition(stop, name=f"failed_ratio({ratio:.0%})")

"""
Study-specific stop conditions for optimization.

This module provides stop conditions for controlling when an optimization study should stop.
All conditions operate on a list of Trial objects.
"""

from dreadnode.core.optimization.trial import CandidateT, Trial
from dreadnode.core.stopping import StopCondition

StudyStopCondition = StopCondition[list[Trial[CandidateT]]]
"""Type alias for study stop conditions."""


# =============================================================================
# Score-Based Stop Conditions
# =============================================================================


def score_value(
    metric_name: str | None = None,
    *,
    gt: float | None = None,
    gte: float | None = None,
    lt: float | None = None,
    lte: float | None = None,
    name: str | None = None,
) -> StudyStopCondition:
    """
    Terminates if a trial score value meets a given threshold.

    - If `metric_name` is provided, it checks that specific objective score from trial.scores.
    - If `metric_name` is None (the default), it checks the primary, average `trial.score`.

    If you are using multi-objective optimization, it is recommended to specify
    a `metric_name` to avoid ambiguity.

    Args:
        metric_name: The name of the metric to check. If None, checks the primary score.
        gt: Greater than threshold.
        gte: Greater than or equal to threshold.
        lt: Less than threshold.
        lte: Less than or equal to threshold.
        name: Optional name for the condition.

    Returns:
        A StopCondition that triggers when the score meets the threshold.

    Example:
        ```python
        # Stop when accuracy exceeds 0.95
        stop_on_accuracy = score_value("accuracy", gte=0.95)

        # Stop when loss drops below 0.01
        stop_on_loss = score_value("loss", lt=0.01)
        ```
    """

    def evaluate(trials: list[Trial]) -> bool:
        finished_trials = [t for t in trials if t.status == "finished"]
        if not finished_trials:
            return False

        trial = finished_trials[-1]
        value = trial.scores.get(metric_name) if metric_name else trial.score
        if value is None:
            return False

        if gt is not None and value > gt:
            return True
        if gte is not None and value >= gte:
            return True
        if lt is not None and value < lt:
            return True
        if lte is not None and value <= lte:
            return True

        return False

    return StopCondition(evaluate, name=name or f"score_value({metric_name or 'score'})")


def score_plateau(
    patience: int,
    *,
    min_delta: float = 0,
    metric_name: str | None = None,
    name: str | None = None,
) -> StudyStopCondition:
    """
    Stops the study if the best trial score does not improve over time.

    If you are using multi-objective optimization, it is recommended to specify
    a `metric_name` to avoid ambiguity.

    Args:
        patience: The number of steps to wait before stopping.
        min_delta: The minimum change in score to consider it an improvement.
        metric_name: The name of the metric to check.
        name: Optional name for the condition.

    Returns:
        A StopCondition that triggers when the score plateaus.

    Example:
        ```python
        # Stop if no improvement for 10 trials
        stop_on_plateau = score_plateau(patience=10)

        # Stop if improvement is less than 0.01 for 5 trials
        stop_on_plateau = score_plateau(patience=5, min_delta=0.01)
        ```
    """

    def evaluate(
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

        return improvement < min_delta if min_delta > 0 else improvement <= 0

    return StopCondition(evaluate, name=name or f"plateau({metric_name or 'score'}, p={patience})")


def best_score_not_improved(
    for_trials: int,
    *,
    metric_name: str | None = None,
    name: str | None = None,
) -> StudyStopCondition:
    """
    Stops if the best score has not improved for a number of trials.

    This is a simpler version of score_plateau that just counts trials
    without improvement.

    Args:
        for_trials: Number of trials without improvement before stopping.
        metric_name: The name of the metric to check.
        name: Optional name for the condition.

    Returns:
        A StopCondition that triggers when best score hasn't improved.
    """

    def evaluate(trials: list[Trial]) -> bool:
        finished_trials = [t for t in trials if t.status == "finished"]
        if len(finished_trials) <= for_trials:
            return False

        scores = [t.scores.get(metric_name) if metric_name else t.score for t in finished_trials]
        scores = [s for s in scores if s is not None]

        if not scores:
            return False

        best_score = max(scores)
        best_idx = scores.index(best_score)

        # Check if best score was found more than `for_trials` ago
        return len(scores) - best_idx - 1 >= for_trials

    return StopCondition(
        evaluate, name=name or f"best_not_improved({metric_name or 'score'}, {for_trials})"
    )


# =============================================================================
# Trial Status-Based Stop Conditions
# =============================================================================


def pruned_ratio(
    ratio: float, min_trials: int = 10, *, name: str | None = None
) -> StudyStopCondition:
    """
    Stops the study if the ratio of pruned trials exceeds a threshold.

    Args:
        ratio: The maximum allowed ratio of pruned trials (0.0 to 1.0).
        min_trials: The minimum number of trials before evaluating.
        name: Optional name for the condition.

    Returns:
        A StopCondition that triggers when pruned ratio exceeds the threshold.

    Example:
        ```python
        # Stop if more than 80% of trials are pruned
        stop_on_pruned = pruned_ratio(0.8)
        ```
    """

    def evaluate(
        trials: list[Trial], *, ratio: float = ratio, min_trials: int = min_trials
    ) -> bool:
        if len(trials) < min_trials:
            return False

        pruned_count = sum(1 for t in trials if t.status == "pruned")
        current_ratio = pruned_count / len(trials)
        return current_ratio >= ratio

    return StopCondition(evaluate, name=name or f"pruned_ratio({ratio:.0%})")


def failed_ratio(
    ratio: float, min_trials: int = 10, *, name: str | None = None
) -> StudyStopCondition:
    """
    Stops the study if the ratio of failed trials exceeds a threshold.

    Args:
        ratio: The maximum allowed ratio of failed trials (0.0 to 1.0).
        min_trials: The minimum number of trials before evaluating.
        name: Optional name for the condition.

    Returns:
        A StopCondition that triggers when failed ratio exceeds the threshold.

    Example:
        ```python
        # Stop if more than 50% of trials fail
        stop_on_failed = failed_ratio(0.5)
        ```
    """

    def evaluate(
        trials: list[Trial], *, ratio: float = ratio, min_trials: int = min_trials
    ) -> bool:
        if len(trials) < min_trials:
            return False

        failed_count = sum(1 for t in trials if t.status == "failed")
        current_ratio = failed_count / len(trials)
        return current_ratio >= ratio

    return StopCondition(evaluate, name=name or f"failed_ratio({ratio:.0%})")


def consecutive_failures(count: int, *, name: str | None = None) -> StudyStopCondition:
    """
    Stops if there are consecutive failed trials.

    Args:
        count: The number of consecutive failures before stopping.
        name: Optional name for the condition.

    Returns:
        A StopCondition that triggers after consecutive failures.
    """

    def evaluate(trials: list[Trial]) -> bool:
        if len(trials) < count:
            return False

        recent = trials[-count:]
        return all(t.status == "failed" for t in recent)

    return StopCondition(evaluate, name=name or f"consecutive_failures({count})")


def consecutive_pruned(count: int, *, name: str | None = None) -> StudyStopCondition:
    """
    Stops if there are consecutive pruned trials.

    Args:
        count: The number of consecutive pruned trials before stopping.
        name: Optional name for the condition.

    Returns:
        A StopCondition that triggers after consecutive pruned trials.
    """

    def evaluate(trials: list[Trial]) -> bool:
        if len(trials) < count:
            return False

        recent = trials[-count:]
        return all(t.status == "pruned" for t in recent)

    return StopCondition(evaluate, name=name or f"consecutive_pruned({count})")


# =============================================================================
# Count-Based Stop Conditions
# =============================================================================


def trial_count(max_trials: int, *, name: str | None = None) -> StudyStopCondition:
    """
    Stops after a maximum number of trials.

    Args:
        max_trials: The maximum number of trials to run.
        name: Optional name for the condition.

    Returns:
        A StopCondition that triggers after the specified number of trials.
    """

    def evaluate(trials: list[Trial]) -> bool:
        return len(trials) >= max_trials

    return StopCondition(evaluate, name=name or f"trial_count({max_trials})")


def finished_trial_count(max_finished: int, *, name: str | None = None) -> StudyStopCondition:
    """
    Stops after a maximum number of successfully finished trials.

    Args:
        max_finished: The maximum number of finished trials.
        name: Optional name for the condition.

    Returns:
        A StopCondition that triggers after the specified number of finished trials.
    """

    def evaluate(trials: list[Trial]) -> bool:
        finished = sum(1 for t in trials if t.status == "finished")
        return finished >= max_finished

    return StopCondition(evaluate, name=name or f"finished_trial_count({max_finished})")


# =============================================================================
# Time-Based Stop Conditions
# =============================================================================


def elapsed_time(max_seconds: float, *, name: str | None = None) -> StudyStopCondition:
    """
    Stops if the total study time exceeds a limit.

    Args:
        max_seconds: The maximum number of seconds the study is allowed to run.
        name: Optional name for the condition.

    Returns:
        A StopCondition that triggers when elapsed time exceeds the limit.
    """

    def evaluate(trials: list[Trial]) -> bool:
        if len(trials) < 2:
            return False

        first_trial = trials[0]
        last_trial = trials[-1]

        if first_trial.started_at is None or last_trial.finished_at is None:
            return False

        delta = last_trial.finished_at - first_trial.started_at
        return delta.total_seconds() > max_seconds

    return StopCondition(evaluate, name=name or f"elapsed_time({max_seconds}s)")


def estimated_cost(limit: float, *, name: str | None = None) -> StudyStopCondition:
    """
    Stops if the total estimated cost exceeds a limit.

    Args:
        limit: The maximum cost allowed (USD).
        name: Optional name for the condition.

    Returns:
        A StopCondition that triggers when estimated cost exceeds the limit.
    """

    def evaluate(trials: list[Trial]) -> bool:
        total_cost = sum(t.estimated_cost or 0 for t in trials)
        return total_cost > limit

    return StopCondition(evaluate, name=name or f"estimated_cost(${limit:.2f})")

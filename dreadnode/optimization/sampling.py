import itertools
import random
from collections import defaultdict

from dreadnode.meta import Config, component
from dreadnode.optimization.trial import CandidateT, Trial


@component
def top_k(
    trials: list[Trial[CandidateT]], *, k: int = Config(5, help="Number of top trials to select.")
) -> list[Trial[CandidateT]]:
    """
    Selects the top k trials by score (highest first).
    """
    sorted_trials = sorted(trials, key=lambda t: t.score, reverse=True)
    return sorted_trials[:k]


@component
def random_k(
    trials: list[Trial[CandidateT]],
    *,
    k: int = Config(5, help="Number of random trials to select."),
) -> list[Trial[CandidateT]]:
    """
    Selects k random trials from the pool.
    """
    return random.sample(trials, min(k, len(trials))) if trials else []  # nosec


@component
def epsilon_greedy(
    trials: list[Trial[CandidateT]],
    *,
    k: int = Config(5, help="Number of top trials to select."),
    epsilon: float = Config(0.2, help="Probability of choosing a random trial."),
) -> list[Trial[CandidateT]]:
    """
    Based on the probability `epsilon`, selects either:
    - top k trials by score (highest first),
    - top k-1 trials and one random trial to ensure exploration.
    """
    sorted_trials = sorted(trials, key=lambda t: t.score, reverse=True)

    if random.random() < epsilon and len(sorted_trials) >= k:  # noqa: S311 # nosec
        k_minus_1 = sorted_trials[: k - 1]
        random_choice = random.choice(sorted_trials[k - 1 :])  # noqa: S311 # nosec
        return [*k_minus_1, random_choice]

    return sorted_trials[:k]


@component
def tournament(
    trials: list[Trial[CandidateT]], *, k: int = Config(5), pool_size: int = Config(3)
) -> list[Trial[CandidateT]]:
    """
    Selects at most k winners from the trials using a tournament selection process.

    For each round in `k`, a subset of the trials is selected (`pool_size`), and the
    best trial from this subset is chosen as the winner.
    """
    winners = []
    pool = list(trials)

    for _ in range(k):
        if not pool:
            break

        contestants = random.sample(pool, min(pool_size, len(pool)))  # nosec
        winner = max(contestants, key=lambda t: t.score)
        winners.append(winner)
        pool.remove(winner)

    return winners


@component
def proportional(
    trials: list[Trial[CandidateT]], *, k: int = Config(5, help="Number of trials to select.")
) -> list[Trial[CandidateT]]:
    """
    Selects k trials using fitness proportional selection.

    Also known as "Roulette Wheel Selection" or "Weighted Random Sampling".
    Each trial's chance of being selected is proportional to its score.

    Args:
        trials: The pool of trials to select from.
        k: The number of unique trials to select.

    Returns:
        A list of selected trials.
    """
    if not trials:
        return []

    # 1 - Normalize scores for use as weights

    scores = [t.score for t in trials]
    min_score = min(scores)
    weights = [s - min_score for s in scores] if min_score < 0 else scores.copy()
    total_weight = sum(weights)

    # If all trials have the same score - take the fast route
    if total_weight == 0:
        return random.sample(trials, min(k, len(trials)))  # nosec

    # 2 - Select k winners one by one, without replacement

    winners = []
    pool = list(trials)
    current_weights = list(weights)

    for _ in range(min(k, len(pool))):
        if not pool:
            break

        winner = random.choices(pool, weights=current_weights, k=1)[0]  # noqa: S311 # nosec
        winners.append(winner)

        # Remove the winner from the pool
        idx = pool.index(winner)
        current_weights.pop(idx)
        pool.pop(idx)

        # Stop early if remaining weights are all zero
        if sum(current_weights) == 0:
            break

    return winners


# Utils


def interleave_by_parent(trials: list[Trial[CandidateT]]) -> list[Trial[CandidateT]]:
    """
    Reorders a list of trials to maximize parent diversity (if parent information exists).

    This helps prevent samplers which use `sorted` from
    favoring any particular parent when scores are identical.

    Example: `[P1, P1, P2, P2, P3]` -> [P1, P2, P3, P1, P2]
    """
    if not trials:
        return []

    parent_to_children = defaultdict(list)
    for trial in trials:
        parent_to_children[trial.parent_id].append(trial)

    interleaved_list = []
    for trial_tuple in itertools.zip_longest(*parent_to_children.values()):
        for trial in trial_tuple:
            if trial is not None:
                interleaved_list.append(trial)  # noqa: PERF401

    return interleaved_list

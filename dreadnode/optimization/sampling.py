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


@component
def gepa(  # noqa: PLR0912, PLR0915
    trials: list[Trial[CandidateT]],
    *,
    k: int = Config(5, help="Number of trials to select from the pareto frontier."),
) -> list[Trial[CandidateT]]:
    """
    Selects k trials using GEPA's weighted Pareto optimization algorithm.

    This sampler identifies the Pareto frontier of non-dominated trials and then
    performs weighted random sampling without replacement. A candidate's weight is
    determined by the number of dataset samples for which it is a top performer.

    Note: This method assumes that each trial was evaluated on multiple dataset samples

    See: GEPA - https://arxiv.org/abs/2507.19457

    Args:
        trials: The pool of trials to select from.
        k: The number of unique trials to select from the frontier.

    Returns:
        A list of selected trials from the weighted Pareto frontier.
    """
    if not trials:
        return []

    scored_trials = [t for t in trials if t.score_breakdown]
    if not scored_trials:
        return sorted(trials, key=lambda t: t.score, reverse=True)[:k]

    # 1 - Pre-computation - Find the champions for each task/sample

    champions_by_task: defaultdict[str, set[Trial[CandidateT]]] = defaultdict(set)
    best_score_by_task: defaultdict[str, float] = defaultdict(lambda: -float("inf"))

    all_objectives = {
        f"{objective_name}_{i}"
        for trial in scored_trials
        for objective_name, scores in trial.score_breakdown.items()
        for i in range(len(scores))
    }

    for objective in all_objectives:
        obj_name, task_idx_str = objective.rsplit("_", 1)
        task_idx = int(task_idx_str)

        for trial in scored_trials:
            breakdown = trial.score_breakdown
            if obj_name in breakdown and len(breakdown[obj_name]) > task_idx:
                score = breakdown[obj_name][task_idx]
                if score > best_score_by_task[objective]:
                    best_score_by_task[objective] = score
                    champions_by_task[objective] = {trial}
                elif score == best_score_by_task[objective]:
                    champions_by_task[objective].add(trial)

    per_task_champions = list(champions_by_task.values())

    # 2 - Iteratively remove dominated trials to find the frontier

    candidates = sorted(scored_trials, key=lambda t: t.score)
    dominated_set: set[Trial[CandidateT]] = set()

    while True:
        found_one_to_remove = False
        non_dominated_candidates = {c for c in candidates if c not in dominated_set}

        for candidate_a in candidates:
            if candidate_a in dominated_set:
                continue

            tasks_where_a_is_champion = [
                champions for champions in per_task_champions if candidate_a in champions
            ]

            if not tasks_where_a_is_champion:
                dominated_set.add(candidate_a)
                found_one_to_remove = True
                break

            is_dominated = True
            other_candidates = non_dominated_candidates - {candidate_a}
            for champions in tasks_where_a_is_champion:
                if not any(other in champions for other in other_candidates):
                    is_dominated = False
                    break

            if is_dominated:
                dominated_set.add(candidate_a)
                found_one_to_remove = True
                break

        if not found_one_to_remove:
            break

    pareto_frontier = list(non_dominated_candidates)
    if not pareto_frontier:
        return random.sample(scored_trials, min(k, len(scored_trials)))  # nosec

    # 3 - Perform weighted sampling from the frontier.

    weights = [
        sum(1 for champions in per_task_champions if trial in champions)
        for trial in pareto_frontier
    ]

    population = list(pareto_frontier)
    selected_trials: list[Trial[CandidateT]] = []

    num_to_select = min(k, len(population))

    for _ in range(num_to_select):
        if not population:
            break

        # Select one trial based on current weights
        chosen = random.choices(population, weights=weights, k=1)[0]  # noqa: S311  # nosec
        selected_trials.append(chosen)

        # Remove the chosen trial and its weight for the next iteration
        chosen_idx = population.index(chosen)
        population.pop(chosen_idx)
        weights.pop(chosen_idx)

    return selected_trials


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

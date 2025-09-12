import typing as t
from collections import deque

from dreadnode.meta import Config, component
from dreadnode.optimization.trial import CandidateT, Trial

if t.TYPE_CHECKING:
    from ulid import ULID


@component
def lineage(
    current_trial: Trial[CandidateT], all_trials: list[Trial[CandidateT]], *, depth: int = Config(5)
) -> list[Trial[CandidateT]]:
    """
    Collects related trials by tracing the direct parent lineage, regardless of status.
    """

    def get_parent(trial: Trial[CandidateT]) -> Trial[CandidateT] | None:
        return (
            next((t for t in all_trials if t.id == trial.parent_id), None)
            if trial.parent_id
            else None
        )

    trials: list[Trial[CandidateT]] = []
    parent = get_parent(current_trial)
    while parent:
        trials.insert(0, parent)
        parent = get_parent(parent)

    return trials[:depth]


@component
def finished(_: Trial[CandidateT], all_trials: list[Trial[CandidateT]]) -> list[Trial[CandidateT]]:
    """
    Collects all finished trials, regardless of lineage.
    """
    return [t for t in all_trials if t.status == "finished"]


@component
def local_neighborhood(
    current_trial: Trial[CandidateT],
    all_trials: list[Trial[CandidateT]],
    *,
    depth: int = Config(3, help="The neighborhood depth."),
) -> list[Trial[CandidateT]]:
    """
    Collects a local neighborhood of trials by performing a graph walk from the current trial.

    The maximum distance for any discovered node is `2h-1`.
    """
    if not all_trials:
        return []

    # 1 - Build a bi-directional graph for efficient traversal

    all_trials_map: dict[ULID, Trial] = {t.id: t for t in all_trials}
    children_map: dict[ULID, list[ULID]] = {tid: [] for tid in all_trials_map}
    for trial in all_trials:
        if trial.parent_id:
            children_map.setdefault(trial.parent_id, []).append(trial.id)

    # 2 - Perform a BFS staying within 2h-1

    max_distance = (2 * depth) - 1
    neighborhood_ids: set[ULID] = set()
    # Start at 1 because contextually this node is 1 away from the new child
    queue = deque([(current_trial.id, 1)])  # (trial_id, distance)
    visited: set[ULID] = {current_trial.id}

    while queue:
        tid, distance = queue.popleft()
        neighborhood_ids.add(tid)

        if distance >= max_distance:
            continue

        trial_node = all_trials_map.get(tid)
        if not trial_node:
            continue

        # Up to the parent
        if trial_node.parent_id and trial_node.parent_id not in visited:
            visited.add(trial_node.parent_id)
            queue.append((trial_node.parent_id, distance + 1))

        # Down to all children
        for child_id in children_map.get(tid, []):
            if child_id not in visited:
                visited.add(child_id)
                queue.append((child_id, distance + 1))

    return [all_trials_map[tid] for tid in neighborhood_ids]

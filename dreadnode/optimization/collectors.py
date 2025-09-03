import typing as t
from collections import deque

from dreadnode.meta import Config, component
from dreadnode.optimization.trial import Trial, Trials

if t.TYPE_CHECKING:
    from uuid import UUID

T = t.TypeVar("T")


@component
def lineage(current_trial: Trial[T], all_trials: Trials[T], *, depth: int = Config(5)) -> Trials[T]:
    """
    Collects related trials by tracing the direct parent lineage, regardless of status.
    """

    def get_parent(trial: Trial[T]) -> Trial[T] | None:
        return (
            next((t for t in all_trials if t.id == trial.parent_id), None)
            if trial.parent_id
            else None
        )

    trials: Trials[T] = []
    parent = get_parent(current_trial)
    while parent:
        trials.insert(0, parent)
        parent = get_parent(parent)

    return trials[:depth]


@component
def all_successful(_: Trial[T], all_trials: Trials[T]) -> Trials[T]:
    """
    Collects all successful trials, regardless of lineage.
    """
    return [t for t in all_trials if t.status == "success"]


@component
def local_neighborhood(
    current_trial: Trial[T],
    all_trials: Trials[T],
    *,
    depth: int = Config(3, help="The neighborhood depth."),
) -> Trials[T]:
    """
    Collects a local neighborhood of trials by performing a graph walk from the current trial.

    The maximum distance for any discovered node is `2h-1`.
    """
    if not all_trials:
        return []

    # 1 - Build a bi-directional graph for efficient traversal

    all_trials_map: dict[UUID, Trial] = {t.id: t for t in all_trials}
    children_map: dict[UUID, list[UUID]] = {tid: [] for tid in all_trials_map}
    for trial in [t_ for t_ in all_trials if t_.parent_id]:
        children_map.setdefault(trial.parent_id, []).append(trial.id)

    # 2 - Perform a BFS staying within 2h-1

    max_distance = (2 * depth) - 1
    neighborhood_ids: set[UUID] = set()
    # Start at 1 because contextually this node is 1 away from the new child
    queue = deque([(current_trial.id, 1)])  # (trial_id, distance)
    visited: set[UUID] = {current_trial.id}

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

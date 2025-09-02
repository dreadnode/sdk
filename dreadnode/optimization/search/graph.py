import asyncio

from dreadnode.optimization.search.base import Search
from dreadnode.optimization.trial import CandidateT, Trial, TrialCollector, TrialFilter, Trials
from dreadnode.transforms import Transform
from dreadnode.transforms.base import TransformLike


class LineageCollector(TrialCollector):
    """Collects trials by tracing the direct parent lineage of the current trial."""

    def __call__(self, current_trial: Trial, all_trials: Trials) -> Trials:
        path = [current_trial]
        parent = (
            next((t for t in all_trials if t.id == current_trial.parent_id), None)
            if current_trial.parent_id
            else None
        )
        while parent:
            path.insert(0, parent)
            parent = all_trials.get(parent.parent_id) if parent.parent_id else None
        return path


class GraphSearch(Search[CandidateT]):
    """A generalized, stateful strategy for generative graph-based search."""

    def __init__(
        self,
        transform: TransformLike[Trials[CandidateT], CandidateT],
        initial_candidate: CandidateT,
        *,
        branching_factor: int = 3,
        trial_collector: TrialCollector = LineageContext(),
        leaf_selector: TrialFilter,  # e.g., top-k by score
    ):
        self.transform = Transform.fit(transform)
        self.initial_candidate = initial_candidate
        self.context_collector = context_collector
        self.branching_factor = branching_factor
        self.select_leaves = leaf_selection_strategy

        self._all_trials: dict[UUID, Trial[CandidateT]] = {}
        self.leaves: list[Trial[CandidateT]] = []

    async def suggest(self, step: int) -> list[Trial[CandidateT]]:
        if not self.leaves:
            return [Trial(candidate=self.initial_candidate, step=step)]

        all_new_trials = []
        for leaf in self.leaves:
            context = self.context_collector(leaf, self._all_trials)
            coroutines = [self.transform(context) for _ in range(self.branching_factor)]
            new_candidates = await asyncio.gather(*coroutines)

            # 3. Create the new trial objects with correct parentage.
            for candidate in new_candidates:
                all_new_trials.append(
                    Trial(candidate=candidate, parent_id=leaf.trial_id, step=step)
                )
        return all_new_trials

    def observe(self, trials: list[Trial[CandidateT]]) -> None:
        # Add all new trials to our graph representation.
        for trial in trials:
            self._all_trials[trial.trial_id] = trial

        if not self.leaves:
            self.leaves = trials  # First step
            return

        combined = self.leaves + [t for t in trials if t.status == "success"]
        self.leaves = self.select_leaves(combined)

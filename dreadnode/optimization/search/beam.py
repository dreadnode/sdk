import asyncio
import typing as t
from abc import ABC, abstractmethod

from dreadnode.optimization.trial import CandidateT, Trial
from dreadnode.transforms import Transform


class Search(ABC, t.Generic[CandidateT]):
    """Abstract base class for all optimization search strategies."""

    @abstractmethod
    async def suggest(self) -> list[CandidateT]:
        """Suggests the next batch of candidates."""

    @abstractmethod
    def observe(self, trials: list[Trial[CandidateT]]) -> None:
        """Informs the strategy of the results of recent trials."""


# Define a type for the path of trials leading to a candidate
TrialPath = list[Trial[CandidateT]]


class BeamSearch(Search[CandidateT]):
    """
    A stateful strategy for sequential beam search that tracks trial history.
    """

    def __init__(
        self,
        # The type hint is simplified to `list` to avoid the runtime TypeError.
        transform: Transform[list, CandidateT],
        initial_candidate: CandidateT,
        beam_width: int = 3,
        branching_factor: int = 3,
    ):
        self.transform = Transform(transform)
        self.initial_candidate = initial_candidate
        self.beam_width = beam_width
        self.branching_factor = branching_factor

        # --- FIX: Use internal state to track parentage ---
        # This map permanently stores the parent of a trial.
        self._parent_map: dict[int, Trial[CandidateT]] = {}
        # This map TEMPORARILY links a new candidate to its parent trial
        # for a single step, between `suggest` and `observe`.
        self._pending_parent_map: dict[CandidateT, Trial[CandidateT]] = {}
        # --- END FIX ---

        self.beams: list[Trial[CandidateT]] = []

    def _get_path_for_trial(self, trial: Trial[CandidateT]) -> t.Any:
        """Traces back from a trial to the root to build its historical path."""
        path = [trial]
        parent = self._parent_map.get(id(trial))
        while parent:
            path.insert(0, parent)
            parent = self._parent_map.get(id(parent))
        return path

    async def suggest(self) -> list[CandidateT]:
        # Clear the temporary map at the start of each step.
        self._pending_parent_map.clear()

        if not self.beams:
            return [self.initial_candidate]

        all_new_candidates = []
        for beam in self.beams:
            path = self._get_path_for_trial(beam)

            coroutines = [self.transform(path) for _ in range(self.branching_factor)]
            new_candidates = await asyncio.gather(*coroutines)

            # --- FIX: Store the link in the internal map, not on the candidate ---
            for candidate in new_candidates:
                # If multiple parents generate the same candidate, the last one wins.
                # This is an acceptable trade-off for this minimal-change fix.
                self._pending_parent_map[candidate] = beam
            # --- END FIX ---

            all_new_candidates.extend(new_candidates)

        return all_new_candidates

    def observe(self, trials: list[Trial[CandidateT]]) -> None:
        # --- FIX: Look up parents in the internal map ---
        for trial in trials:
            # Find the parent trial that generated this candidate.
            parent = self._pending_parent_map.get(trial.candidate)
            if parent:
                # Establish the permanent link.
                self._parent_map[id(trial)] = parent
        # --- END FIX ---

        if not self.beams:
            self.beams = trials
            return

        combined = self.beams + [t for t in trials if t.status == "success"]
        sorted_by_score = sorted(combined, key=lambda t: t.score, reverse=True)
        self.beams = sorted_by_score[: self.beam_width]

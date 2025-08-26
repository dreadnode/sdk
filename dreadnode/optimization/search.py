import asyncio
import typing as t
from abc import ABC, abstractmethod

from dreadnode.transforms import Transform, TransformLike

from .trial import CandidateT, Trial


class Search(ABC, t.Generic[CandidateT]):
    """Abstract base class for all optimization search strategies."""

    @abstractmethod
    async def suggest(self, step: int) -> list[CandidateT]:
        """Suggests the next batch of candidates."""

    @abstractmethod
    def observe(self, trials: list[Trial[CandidateT]]) -> None:
        """Informs the strategy of the results of recent trials."""


class BeamSearch(Search[CandidateT]):
    """A stateful strategy for sequential beam search."""

    def __init__(
        self,
        transform: TransformLike[CandidateT, CandidateT],
        initial_candidate: CandidateT,
        beam_width: int = 3,
        branching_factor: int = 3,
    ):
        self.transform = transform if isinstance(transform, Transform) else Transform(transform)
        self.initial_candidate = initial_candidate
        self.beam_width = beam_width
        self.branching_factor = branching_factor
        self.beams: list[Trial[CandidateT]] = []

    async def suggest(self, _: int) -> list[CandidateT]:
        if not self.beams:
            return [self.initial_candidate]

        candidates = []
        for beam in self.beams:
            coroutines = [self.transform(beam.candidate) for _ in range(self.branching_factor)]
            candidates.extend(await asyncio.gather(*coroutines))

        return candidates

    def observe(self, trials: list[Trial[CandidateT]]) -> None:
        if not self.beams:
            self.beams = trials
            return

        combined = self.beams + [t for t in trials if t.status == "success"]
        sorted_by_score = sorted(combined, key=lambda t: t.score, reverse=True)
        self.beams = sorted_by_score[: self.beam_width]

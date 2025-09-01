import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

from dreadnode.optimization.trial import CandidateT, Trial
from dreadnode.types import Primitive


class Search(ABC, t.Generic[CandidateT]):
    """Abstract base class for all optimization search strategies."""

    @abstractmethod
    def reset(self) -> None:
        """Resets the search strategy to its initial state."""

    @abstractmethod
    async def suggest(self) -> list[CandidateT]:
        """Suggests the next batch of candidates."""

    @abstractmethod
    def observe(self, trials: list[Trial[CandidateT]]) -> None:
        """Informs the strategy of the results of recent trials."""


@dataclass
class Distribution:
    """Base class for all search space distributions."""


@dataclass
class Float(Distribution):
    low: float
    high: float
    log: bool = False
    step: float | None = None


@dataclass
class Int(Distribution):
    low: int
    high: int
    log: bool = False
    step: int = 1


@dataclass
class Categorical(Distribution):
    choices: list[Primitive]


SearchSpace = t.Mapping[str, Distribution | list[Primitive]]

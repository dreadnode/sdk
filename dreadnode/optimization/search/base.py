import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

from dreadnode.common_types import Primitive
from dreadnode.optimization.trial import CandidateT, Trial

if t.TYPE_CHECKING:
    from dreadnode.optimization.study import Direction


class Search(ABC, t.Generic[CandidateT]):
    """Abstract base class for all optimization search strategies."""

    def reset(self, context: "OptimizationContext") -> None:
        """Resets the search strategy to a clean state."""

    @abstractmethod
    def suggest(self, step: int) -> t.AsyncIterator[Trial[CandidateT]]:
        """Suggests the next batch of candidates."""

    @abstractmethod
    async def observe(self, trials: list[Trial[CandidateT]]) -> None:
        """Informs the strategy of the results of recent trials."""


@dataclass
class OptimizationContext:
    """Context to prepare search algorithms for objectives."""

    objective_names: list[str]
    directions: "list[Direction]"


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

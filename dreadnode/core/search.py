import typing as t
from dataclasses import dataclass

from dreadnode.core.meta.config import Component
from dreadnode.core.optimization.trial import CandidateT, Trial
from dreadnode.core.types.common import Primitive

if t.TYPE_CHECKING:
    from dreadnode.core.optimization.study import Direction


class Search(
    Component[["OptimizationContext"], t.AsyncGenerator[Trial[CandidateT], None]],
    t.Generic[CandidateT],
):
    pass


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


SearchLike = dict[str, list[t.Any]] | SearchSpace | Search[dict[str, t.Any]]

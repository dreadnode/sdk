import typing as t

from pydantic import Field

from dreadnode.airt.attack.base import Attack, CandidateT
from dreadnode.optimization import Study  # Import Trial for forward reference resolution
from dreadnode.optimization.search import BeamSearch
from dreadnode.scorers.base import Scorer  # Import T TypeVar for forward reference resolution
from dreadnode.transforms import TransformLike


class GenerativeAttack(Attack[CandidateT]):
    """
    A base class for attacks that iteratively generate and test new candidates.

    This class provides the chassis for complex sequential attacks like TAP or PAIR.
    It is configured with a `Mutation` primitive that defines the core generative step.
    """

    transform: TransformLike[CandidateT]
    """The core generative primitive used to create new candidates at each step."""
    initial_candidate: CandidateT
    """The starting point (e.g., an initial prompt or state) for the generative search."""
    max_steps: int = 5
    """The maximum number of generative steps (the 'depth' of the search)."""
    beam_width: int = 1
    """The number of best candidates to keep at each step. (width=1 for PAIR, >1 for TAP)."""
    branching_factor: int = 1
    """The number of new candidates to generate from each beam at each step."""
    constraints: list[Scorer[t.Any]] = Field(default_factory=list)
    """Fast, cheap scorers to prune invalid candidates before full evaluation."""
    direction: t.Literal["maximize", "minimize"] = "maximize"
    """The direction for the optimization objective."""

    def make_study(self) -> Study[CandidateT]:
        search = BeamSearch[CandidateT](
            transform=self.transform,
            initial_candidate=self.initial_candidate,
            beam_width=self.beam_width,
            branching_factor=self.branching_factor,
        )

        return Study[CandidateT](
            strategy=search,
            apply_candidate_fn=self.apply_candidate_fn,
            objective=self.objective,
            dataset=self.dataset,
            max_steps=self.max_steps,
            constraints=self.constraints,
            direction=self.direction,
        )

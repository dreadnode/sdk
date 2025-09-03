import typing as t
from uuid import UUID, uuid4

import typing_extensions as te
from pydantic import BaseModel, ConfigDict, Field, computed_field

from dreadnode.eval.result import EvalResult

CandidateT = te.TypeVar("CandidateT", default=t.Any)
TrialStatus = t.Literal["pending", "success", "failed", "pruned"]


class Trial(BaseModel, t.Generic[CandidateT]):
    """Represents a single, evaluated point in the search space."""

    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    id: UUID = Field(default_factory=uuid4)
    """Unique identifier for the trial."""
    candidate: CandidateT
    """The candidate configuration being assessed."""
    status: TrialStatus = "pending"
    """Current status of the trial."""
    score: float = -float("inf")
    """Fitness score of this candidate. Higher is better."""

    eval_result: EvalResult | None = None
    """Complete evaluation result if the candidate was assessable by the evaluation engine."""
    pruning_reason: str | None = None
    """Reason for pruning this trial, if applicable."""
    error: str | None = None
    """Any error which occurred while processing this trial."""
    step: int = 0
    """The optimization step which produced this trial."""
    parent_id: UUID | None = None
    """The id of the parent trial, used to reconstruct the search graph."""

    def __repr__(self) -> str:
        return f"Trial(id={self.id}, status='{self.status}', score={self.score:.3f})"

    @computed_field
    @property
    def output(self) -> t.Any | None:
        """Get the output of the trial."""
        if self.eval_result and self.eval_result.samples:
            return self.eval_result.samples[0].output
        return None


class Trials(list[Trial[CandidateT]], t.Generic[CandidateT]):
    pass


@te.runtime_checkable
class TrialCollector(t.Protocol, t.Generic[CandidateT]):
    """
    Collect a list of relevant trials based on the current trial.
    """

    def __call__(
        self, current_trial: Trial[CandidateT], all_trials: Trials[CandidateT]
    ) -> Trials[CandidateT]: ...


@te.runtime_checkable
class TrialSampler(t.Protocol, t.Generic[CandidateT]):
    """
    Sample from a list of trials.
    """

    def __call__(self, trials: Trials[CandidateT]) -> Trials[CandidateT]: ...

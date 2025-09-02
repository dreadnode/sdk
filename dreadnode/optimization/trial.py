import typing as t
from uuid import UUID, uuid4

import typing_extensions as te
from pydantic import BaseModel, ConfigDict, Field

from dreadnode.evals.result import EvalResult

CandidateT = te.TypeVar("CandidateT", default=t.Any)
TrialStatus = t.Literal["pending", "success", "failed", "pruned"]


class Trial(BaseModel, t.Generic[CandidateT]):
    """Represents a single, evaluated point in the search space."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: UUID = Field(default_factory=uuid4)
    """Unique identifier."""
    candidate: CandidateT
    """The candidate assessed."""
    status: TrialStatus = "pending"
    """Current status of the trial."""
    score: float = -float("inf")
    """Fitness score of this candidate."""

    eval_result: EvalResult | None = None
    """Complete evaluation result for this candidate."""
    pruning_reason: str | None = None
    """Reason for pruning this trial."""
    error: str | None = None
    """Any error which occurred while processing this trial."""
    step: int = 0
    """The study step which produced this trial."""

    parent_id: UUID | None = None
    """The id of the parent trial for search purposes."""

    def default_trial_formatter(self) -> str:
        """
        A default formatter that converts a trial into a human-readable summary string.
        """
        # Safely access the results from the trial's evaluation
        output_dict = self.eval_result.samples[0].output if self.eval_result else {}
        response_text = output_dict.get("output", "Evaluation failed or is pending.")

        return (
            f"ATTEMPT (Score: {self.score:.2f}):\n"
            f"  - Prompt: {self.candidate}\n"
            f"  - Response: {response_text}"
        )


Trials = list[Trial[CandidateT]]


class TrialCollector(t.Protocol):
    """
    Gather a list of relevant trials based on the current trials.
    """

    def __call__(self, current_trial: Trial, all_trials: Trials) -> Trials: ...


class TrialFilter(t.Protocol):
    """
    Filter down trials based on criteria and/or sorting.
    """

    def __call__(self, trials: Trials) -> Trials: ...

import typing as t

import typing_extensions as te
from pydantic import BaseModel, ConfigDict

from dreadnode.eval.dataset import EvalResult

CandidateT = te.TypeVar("CandidateT", default=t.Any)
TrialStatus = t.Literal["pending", "success", "failed", "pruned"]


class Trial(BaseModel, t.Generic[CandidateT]):
    """Represents a single, evaluated point in the search space."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    candidate: CandidateT
    status: TrialStatus = "pending"
    score: float = -float("inf")

    eval_result: EvalResult | None = None
    pruning_reason: str | None = None
    error: str | None = None
    step: int = 0

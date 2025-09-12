import typing as t
from dataclasses import dataclass, field

import typing_extensions as te

if t.TYPE_CHECKING:
    from dreadnode.optimization.result import StudyResult
    from dreadnode.optimization.study import Study
    from dreadnode.optimization.trial import Trial

# Define over import to avoid cyclic issues
CandidateT = te.TypeVar("CandidateT", default=t.Any)


@dataclass
class StudyEvent(t.Generic[CandidateT]):
    study: "Study[CandidateT]" = field(repr=False)
    trials: list["Trial[CandidateT]"] = field(repr=False)


@dataclass
class StudyStart(StudyEvent[CandidateT]):
    max_steps: int


@dataclass
class StepStart(StudyEvent[CandidateT]):
    step: int


@dataclass
class TrialAdded(StudyEvent[CandidateT]):
    trial: "Trial[CandidateT]"


@dataclass
class TrialStart(StudyEvent[CandidateT]):
    trial: "Trial[CandidateT]"


@dataclass
class TrialPruned(StudyEvent[CandidateT]):
    trial: "Trial[CandidateT]"


@dataclass
class EvaluationStart(StudyEvent[CandidateT]):
    trial: "Trial[CandidateT]"


@dataclass
class TrialComplete(StudyEvent[CandidateT]):
    trial: "Trial[CandidateT]"


@dataclass
class NewBestTrialFound(StudyEvent[CandidateT]):
    trial: "Trial[CandidateT]"


@dataclass
class StepEnd(StudyEvent[CandidateT]):
    step: int


@dataclass
class StudyEnd(StudyEvent[CandidateT]):
    result: "StudyResult[CandidateT]"

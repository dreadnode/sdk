import typing as t
from dataclasses import field  # Some odities with repr=False, otherwise I would use pydantic.Field

from pydantic.dataclasses import dataclass, rebuild_dataclass

if t.TYPE_CHECKING:
    from dreadnode.optimization.study import Study
    from dreadnode.optimization.trial import Trial

CandidateT = t.TypeVar("CandidateT")
StopReason = t.Literal["max_steps", "patience", "target_score", "no_more_candidates", "unknown"]


@dataclass
class StudyEvent(t.Generic[CandidateT]):
    study: "Study[CandidateT]" = field(repr=False)


@dataclass
class StudyStart(StudyEvent[CandidateT]):
    initial_candidate: CandidateT | None


@dataclass
class StepStart(StudyEvent[CandidateT]):
    step: int


@dataclass
class CandidatesSuggested(StudyEvent[CandidateT]):
    candidates: list[t.Any]  # Can be dicts or CandidateT


@dataclass
class CandidatePruned(StudyEvent[CandidateT]):
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
    steps: int
    stop_reason: StopReason
    best_trial: "Trial[CandidateT] | None"


def rebuild_event_models() -> None:
    from dreadnode.optimization.study import Study  # noqa: F401
    from dreadnode.optimization.trial import Trial  # noqa: F401

    rebuild_dataclass(StudyEvent)  # type: ignore[arg-type]
    rebuild_dataclass(StudyStart)  # type: ignore[arg-type]
    rebuild_dataclass(StepStart)  # type: ignore[arg-type]
    rebuild_dataclass(CandidatesSuggested)  # type: ignore[arg-type]
    rebuild_dataclass(CandidatePruned)  # type: ignore[arg-type]
    rebuild_dataclass(EvaluationStart)  # type: ignore[arg-type]
    rebuild_dataclass(TrialComplete)  # type: ignore[arg-type]
    rebuild_dataclass(NewBestTrialFound)  # type: ignore[arg-type]
    rebuild_dataclass(StepEnd)  # type: ignore[arg-type]
    rebuild_dataclass(StudyEnd)  # type: ignore[arg-type]

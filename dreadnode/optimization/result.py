import json
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

from dreadnode.optimization.trial import CandidateT, Trial

if t.TYPE_CHECKING:
    import pandas as pd

StudyStopReason = t.Literal[
    "max_trials_reached",
    "stop_condition_met",
    "search_exhausted",
    "unknown",
]


@dataclass
class StudyResult(t.Generic[CandidateT]):
    """
    The final result of an optimization study, containing all trials, probes, and summary statistics.
    """

    trials: list[Trial[CandidateT]] = field(default_factory=list)
    """A complete list of all trials generated during the study."""
    probes: list[Trial[CandidateT]] = field(default_factory=list)
    """A complete list of all probing trials generated during the study."""
    stop_reason: StudyStopReason = "unknown"
    """The reason the study concluded."""
    stop_explanation: str | None = None
    """A human-readable explanation for why the study stopped (often from stop conditions)."""

    _best_trial: Trial[CandidateT] | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if finished_trials := [t for t in self.trials if t.status == "finished"]:
            self._best_trial = max(finished_trials, key=lambda t: (t.score, t.step))

    def __repr__(self) -> str:
        parts = [
            f"trials={len(self.trials)}",
            f"probes={len(self.probes)}",
            f"stop_reason='{self.stop_reason}'",
        ]
        if self.stop_explanation:
            parts.append(f"stop_explanation='{self.stop_explanation}'")
        if self.best_trial:
            parts.append(f"best_trial={self.best_trial.id}")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    @property
    def best_trial(self) -> Trial[CandidateT] | None:
        """The trial with the highest score among all finished trials. Returns None if no trials succeeded."""
        return self._best_trial

    @property
    def failed_trials(self) -> list[Trial[CandidateT]]:
        """A list of all trials that failed."""
        return [t for t in self.trials if t.status == "failed"]

    @property
    def pruned_trials(self) -> list[Trial[CandidateT]]:
        """A list of all trials that were pruned."""
        return [t for t in self.trials if t.status == "pruned"]

    @property
    def pending_trials(self) -> list[Trial[CandidateT]]:
        """A list of all trials that are still pending."""
        return [t for t in self.trials if t.status == "pending"]

    @property
    def running_trials(self) -> list[Trial[CandidateT]]:
        """A list of all trials that are currently running."""
        return [t for t in self.trials if t.status == "running"]

    def to_dicts(self) -> list[dict[str, t.Any]]:
        """Flattens the results into a list of dictionaries, one for each trial."""
        records = []
        for trial in self.trials:
            base_record = trial.model_dump(exclude={"candidate"}, mode="json")

            candidate = trial.candidate
            if isinstance(candidate, dict):
                # Flatten candidate dict into individual columns
                for key, value in candidate.items():
                    base_record[f"candidate_{key}"] = value
            else:
                base_record["candidate"] = candidate

            records.append(base_record)
        return records

    def to_dataframe(self) -> "pd.DataFrame":
        """Converts the trials into a pandas DataFrame for analysis."""
        import pandas as pd

        return pd.DataFrame(self.to_dicts())

    def to_jsonl(self, path: str | Path) -> None:
        """Saves the trials to a JSON Lines (JSONL) file."""
        records = self.to_dicts()
        with Path(path).open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(record) + "\n" for record in records)

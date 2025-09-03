import json
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

from dreadnode.optimization.trial import CandidateT, Trial

StopReason = t.Literal["max_steps", "patience", "target_score", "no_more_candidates", "unknown"]


@dataclass
class StudyResult(t.Generic[CandidateT]):
    """
    The final result of an optimization study, containing all trials and summary statistics.
    """

    trials: list[Trial[CandidateT]] = field(default_factory=list)
    """A complete list of all trials generated during the study."""
    stop_reason: StopReason = "unknown"
    """The reason the study concluded."""

    _best_trial: Trial[CandidateT] | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if successful_trials := [t for t in self.trials if t.status == "success"]:
            self._best_trial = max(successful_trials, key=lambda t: (t.score, t.step))

    @property
    def best_trial(self) -> Trial[CandidateT] | None:
        """The trial with the highest score among all successful trials. Returns None if no trials succeeded."""
        return self._best_trial

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

    def to_dataframe(self) -> "t.Any":
        """Converts the results into a pandas DataFrame for analysis."""
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for to_dataframe(). Please install with: pip install pandas"
            ) from e

        return pd.DataFrame(self.to_dicts())

    def to_jsonl(self, path: str | Path) -> None:
        """Saves the results to a JSON Lines (JSONL) file."""
        records = self.to_dicts()
        with Path(path).open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    def __repr__(self) -> str:
        best_score_str = f", best_score={self.best_trial.score:.3f}" if self.best_trial else ""
        return f"StudyResult(trials={len(self.trials)}, stop_reason='{self.stop_reason}'{best_score_str})"

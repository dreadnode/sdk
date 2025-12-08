import asyncio
import typing as t
from datetime import datetime

import typing_extensions as te
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field
from ulid import ULID

from dreadnode.eval.result import EvalResult

CandidateT = te.TypeVar("CandidateT", default=t.Any)
TrialStatus = t.Literal["pending", "running", "finished", "failed", "pruned"]


class Trial(BaseModel, t.Generic[CandidateT]):
    """Represents a single, evaluated point in the search space."""

    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    id: ULID = Field(default_factory=ULID)
    """Unique identifier for the trial."""
    candidate: CandidateT
    """The candidate configuration being assessed."""
    status: TrialStatus = "pending"
    """Current status of the trial."""

    score: float = -float("inf")
    """
    The primary, single-value fitness score for this trial.
    This is an average of all objective scores for this trial adjusted
    based on their objective directions (higher is better).
    """
    scores: dict[str, float] = {}
    """A dictionary of all named objective scores for this trial."""
    directional_scores: dict[str, float] = {}
    """
    A dictionary of all named objective scores adjusted
    for their optimization direction (higher is better).

    Typically this is used by search strategies and
    related components to sort trials for sampling and selection.
    """

    eval_result: EvalResult | None = None
    """Complete evaluation result of the trial and associated dataset."""
    pruning_reason: str | None = None
    """Reason for pruning this trial, if applicable."""
    error: str | None = None
    """Any error which occurred while processing this trial."""
    step: int = Field(default=0)
    """The optimization step which produced this trial."""
    parent_id: ULID | None = None
    """The id of the parent trial, as defined by the search strategy."""

    is_probe: bool = False
    """Whether this trial is a probe used for intermediate evaluation."""
    dataset: list[t.Any] | None = None
    """The specific dataset used for probing."""

    _future: "asyncio.Future[Trial[CandidateT]]" = PrivateAttr(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )

    def __repr__(self) -> str:
        parts = [
            f"id={self.id}",
            f"status='{self.status}'",
            f"step={self.step}",
        ]
        if self.is_probe:
            parts.append("probe=True")
        if self.dataset is not None:
            parts.append(f"dataset_size={len(self.dataset)}")
        parts.extend(
            [
                f"score={self.score:.5f}",
                f"scores={{{', '.join(f'{k}={v:.5f}' for k, v in self.scores.items())}}}",
            ]
        )
        return f"{self.__class__.__name__}({', '.join(parts)})"

    def __str__(self) -> str:
        return self.__repr__()

    def __await__(self) -> t.Generator[t.Any, None, "Trial[CandidateT]"]:
        """
        Await the completion of the trial.
        """
        return self._future.__await__()

    def done(self) -> bool:
        """A non-blocking check to see if the trial's evaluation is complete."""
        return self._future.done()

    def as_probe(self, *, dataset: list[t.Any] | None = None) -> "Trial[CandidateT]":
        """
        Ensure this trial is marked as a probe for intermediate evaluation and optional dataset override.

        Args:
            dataset: An optional dataset to use specifically for this trial during probing.
        """
        self.is_probe = True
        self.dataset = dataset
        return self

    def as_trial(self) -> "Trial[CandidateT]":
        """
        Ensure this trial is marked as a full trial, not a probe, triggering a full evaluation.
        """
        self.is_probe = False
        self.dataset = None
        return self

    @staticmethod
    async def wait_for(*trials: "Trial[CandidateT]") -> "list[Trial[CandidateT]]":
        """
        Await the completion of multiple trials.

        Args:
            *trials: The trials to wait for.

        Returns:
            A future that resolves to a list of completed trials.
        """
        return await asyncio.gather(*(trial._future for trial in trials))  # noqa: SLF001

    def get_directional_score(
        self, name: str | None = None, default: float = -float("inf")
    ) -> float:
        """
        Get a specific named objective score - adjusted for optimization direction (higher is better),
        or the overall score if no name is given.

        Args:
            name: The name of the objective.
            default: The value to return if the named score is not found.
        """
        if name is not None:
            return self.scores.get(name, default)
        return self.score

    @computed_field  # type: ignore[prop-decorator]
    @property
    def created_at(self) -> datetime:
        """The creation timestamp of the trial, extracted from its ULID."""
        return self.id.datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def all_scores(self) -> dict[str, float]:
        """
        A dictionary of all named metric mean values from the evaluation result.

        This includes scores not directly related to the objective.
        """
        if not self.eval_result or not self.eval_result.metrics_summary:
            return {}

        return {
            name: summary.get("mean", -float("inf"))
            for name, summary in self.eval_result.metrics_summary.items()
        }

    @property
    def score_breakdown(self) -> dict[str, list[float]]:
        """
        Returns a breakdown of all objective scores across all samples in the evaluation result.

        Returns:
            A dictionary where keys are objective names and values are lists of scores,
            with each score corresponding to a sample from the evaluation dataset.
        """
        if not self.eval_result or not self.eval_result.samples:
            return {}

        return {k: v for k, v in self.eval_result.metrics.items() if k in self.scores}

    @property
    def transformed_input(self) -> CandidateT | None:
        """Returns the transformed input if available from the eval result."""
        if self.eval_result and self.eval_result.samples:
            first_sample = self.eval_result.samples[0]
            if hasattr(first_sample, "transformed_input"):
                return first_sample.transformed_input
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cost(self) -> int:
        """Get the cost of the trial, defined as the number of samples evaluated."""
        if self.eval_result:
            return len(self.eval_result.samples)
        return 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def output(self) -> t.Any | None:
        """Get the output of the trial."""
        if self.eval_result and self.eval_result.samples:
            return self.eval_result.samples[0].output
        return None


@te.runtime_checkable
class TrialCollector(t.Protocol, t.Generic[CandidateT]):
    """
    Collect a list of relevant trials based on the current trial.
    """

    def __call__(
        self,
        current_trial: Trial[CandidateT],
        all_trials: list[Trial[CandidateT]],
        /,
        *args: t.Any,
        **kwargs: t.Any,
    ) -> list[Trial[CandidateT]]: ...


@te.runtime_checkable
class TrialSampler(t.Protocol, t.Generic[CandidateT]):
    """
    Sample from a list of trials.
    """

    def __call__(
        self, trials: list[Trial[CandidateT]], /, *args: t.Any, **kwargs: t.Any
    ) -> list[Trial[CandidateT]]: ...

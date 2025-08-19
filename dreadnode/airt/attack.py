import contextlib
import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field
from rigging import Generator, get_generator

from dreadnode.optimization import Study, StudyEvent, Trial
from dreadnode.scorers import ScorerLike

# Define generic type for candidates
CandidateT = t.TypeVar("CandidateT")


class AttackResult(BaseModel, t.Generic[CandidateT]):
    """The final, clean output of a completed attack."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool
    best_trial: Trial[CandidateT] | None
    study: Study[CandidateT] = Field(repr=False)


class Attack(ABC, BaseModel, t.Generic[CandidateT]):
    """
    The abstract base class for configuring and executing an attack.

    This class acts as a high-level factory for an underlying optimization Study,
    providing a simple and declarative interface for complex attack patterns.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- Core User Configuration ---
    goal: str
    """The initial prompt, objective, or starting point for the attack."""
    target: str | Generator
    """The model or endpoint to attack, as a rigging generator identifier string or object."""
    objective: ScorerLike[str]
    """The scorer that defines the final 'fitness' or 'success' of a candidate."""
    dataset: list[dict] = Field(default_factory=lambda: [{}])
    """The dataset to evaluate each candidate against for robustness."""

    # --- Internal State ---
    _target_generator: Generator = Field(None, repr=False, exclude=True)

    def model_post_init(self, __context: t.Any) -> None:
        """Pydantic hook to initialize the rigging generator after validation."""
        if isinstance(self.target, str):
            self._target_generator = get_generator(self.target)
        else:
            self._target_generator = self.target

    @abstractmethod
    def _configure_study(self) -> Study[CandidateT]:
        """
        [Internal] Each Attack subclass must implement this method.

        Its job is to translate the Attack's high-level configuration into a
        fully-configured Study object with the correct Strategy and glue functions.
        """

    @contextlib.asynccontextmanager
    async def stream(self) -> t.AsyncIterator[t.AsyncGenerator[StudyEvent[CandidateT], None]]:
        study = self._configure_study()
        async with study.stream() as stream:
            yield stream

    async def run(self) -> AttackResult[CandidateT]:
        study = self._configure_study()
        best_trial = await study.run()

        success = False
        if best_trial and best_trial.status == "SUCCESS":
            # Default success criteria: the final score is positive.
            # Could be made more configurable if needed.
            if study.direction == "maximize":
                success = best_trial.score > 0
            else:  # minimize
                success = best_trial.score < 0

        return AttackResult(success=success, best_trial=best_trial, study=study)

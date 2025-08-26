import contextlib
import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from rigging import Generator, get_generator

from dreadnode.optimization import Study, StudyEvent, Trial
from dreadnode.scorers import ScorerLike
from dreadnode.task import Task
from dreadnode.types import AnyDict

# Define generic type for candidates
CandidateT = t.TypeVar("CandidateT")


class AttackResult(BaseModel, t.Generic[CandidateT]):
    """The final, clean output of a completed attack."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    best_trial: Trial[CandidateT] | None
    study: Study[CandidateT] = Field(repr=False)


class Attack(ABC, BaseModel, t.Generic[CandidateT]):
    """
    The abstract base class for configuring and executing an attack.

    This class acts as a high-level factory for an underlying optimization Study,
    providing a simple and declarative interface for complex attack patterns.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    target: str | Generator
    """The model or endpoint to attack, as a rigging generator identifier string or object."""
    objective: ScorerLike[str]
    """The scorer that defines the final 'fitness' or 'success' of a candidate."""
    dataset: list[AnyDict] = Field(default_factory=lambda: [{}])
    """The dataset to evaluate each candidate against for robustness."""

    _target_generator: Generator | None = PrivateAttr(None, init=False)

    def model_post_init(self, _: t.Any) -> None:
        if isinstance(self.target, str):
            self._target_generator = get_generator(self.target)
        else:
            self._target_generator = self.target

    def apply_candidate_fn(self, candidate: CandidateT) -> Task:
        from dreadnode import task

        @task()
        async def run_target_with_candidate() -> str:
            response = await self._target_generator.chat(str(candidate)).run()
            return response.last.content

        return run_target_with_candidate

    @abstractmethod
    def make_study(self) -> Study[CandidateT]:
        """
        [Internal] Each Attack subclass must implement this method.

        Its job is to translate the Attack's high-level configuration into a
        fully-configured Study object with the correct Strategy and glue functions.
        """

    @contextlib.asynccontextmanager
    async def stream(self) -> t.AsyncIterator[t.AsyncGenerator[StudyEvent[CandidateT], None]]:
        study = self.make_study()
        async with study.stream() as stream:
            yield stream

    async def run(self) -> AttackResult[CandidateT]:
        study = self.make_study()
        end = await study.run()
        return AttackResult(best_trial=end.best_trial, study=study)

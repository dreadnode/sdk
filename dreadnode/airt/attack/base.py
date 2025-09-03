import abc
import contextlib
import typing as t

import typing_extensions as te
from pydantic import ConfigDict

from dreadnode.airt.target.base import Target
from dreadnode.meta.types import Config, Model
from dreadnode.optimization import Study
from dreadnode.optimization.events import StudyEvent
from dreadnode.optimization.result import StudyResult
from dreadnode.optimization.search.base import Search
from dreadnode.optimization.study import Direction
from dreadnode.scorers.base import Scorer

In = te.TypeVar("In", default=t.Any)
Out = te.TypeVar("Out", default=t.Any)


class Attack(Model, abc.ABC, t.Generic[In, Out]):
    """
    A declarative configuration for executing an attack.

    This class composes all the necessary components (target, search strategy, objective)
    and internally manages the creation and execution of an optimization Study.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    name: str | None = Config(default=None)
    """The name of the attack - otherwise derived from the target."""
    description: str = Config(default="")
    """A brief description of the attack's purpose."""
    tags: list[str] = Config(default_factory=lambda: ["attack"])
    """A list of tags associated with the attack."""

    target: t.Annotated[Target[In, Out], Config()]
    """The target to attack."""
    search: t.Annotated[Search[In], Config()]
    """The fully configured search strategy to generate attempts."""
    objective: t.Annotated[Scorer[Out], Config()]
    """The single scorer that defines the 'success' of a candidate."""

    direction: Direction = Config(default="maximize")
    """The direction of optimization for the objective score."""
    concurrency: int = Config(default=1, ge=1)
    """The maximum number of trials to evaluate in parallel."""
    constraints: list[Scorer[In]] | None = Config(default=None)
    """A list of Scorer constraints to apply to input attempts. If any constraint scores to a falsy value, the attempt is pruned."""

    # Stopping conditions
    max_steps: int = Config(default=100, ge=1)
    """The maximum number of optimization steps to run."""
    patience: int | None = Config(default=None, ge=1)
    """The number of steps to wait for an improvement before stopping. If None, this is disabled."""
    target_score: float | None = Config(default=None)
    """A target score to achieve. The study will stop if a trial meets or exceeds this score."""

    def as_study(self) -> Study[In]:
        Study.model_rebuild()

        name = self.name or self.target.name

        return Study[In](
            name=name,
            description=self.description,
            tags=self.tags,
            strategy=self.search,
            task_factory=self.target.as_task,
            objective=self.objective,
            direction=self.direction,
            concurrency=self.concurrency,
            constraints=self.constraints,
            max_steps=self.max_steps,
            patience=self.patience,
            target_score=self.target_score,
        )

    @contextlib.asynccontextmanager
    async def stream(self) -> t.AsyncIterator[t.AsyncGenerator[StudyEvent[In], None]]:
        async with self.as_study().stream() as stream:
            yield stream

    async def run(self) -> StudyResult[In]:
        return await self.as_study().run()

    async def console(self) -> StudyResult[In]:
        return await self.as_study().console()

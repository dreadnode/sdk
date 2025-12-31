import typing as t

from pydantic import ConfigDict, Field, SkipValidation

from dreadnode.airt.target import Target
from dreadnode.core.meta import Config
from dreadnode.core.optimization.study import OutputT as Out
from dreadnode.core.optimization.study import Study
from dreadnode.core.optimization.trial import CandidateT as In
from dreadnode.core.task import Task


class Attack(Study[In, Out]):
    """
    A declarative configuration for executing an AIRT attack.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    target: t.Annotated[SkipValidation[Target[In, Out]], Config()]
    """The target to attack."""

    tags: list[str] = Config(default_factory=lambda: ["attack"])
    """A list of tags associated with the attack for logging."""

    # Override the task factory as the target will replace it.
    task_factory: t.Callable[[In], Task[..., Out]] = Field(  # type: ignore[assignment]
        default_factory=lambda: None,
        repr=False,
        init=False,
    )

    def model_post_init(self, context: t.Any) -> None:
        self.task_factory = self.target.task_factory
        super().model_post_init(context)

import typing as t

from pydantic import ConfigDict, Field, SkipValidation

from dreadnode.airt.target.base import Target
from dreadnode.eval.hooks.base import EvalHook
from dreadnode.meta import Config
from dreadnode.optimization.study import OutputT as Out
from dreadnode.optimization.study import Study
from dreadnode.optimization.trial import CandidateT as In
from dreadnode.task import Task


class Attack(Study[In, Out]):
    """
    A declarative configuration for executing an AIRT attack.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    target: t.Annotated[SkipValidation[Target[In, Out]], Config()]
    """The target to attack."""

    tags: list[str] = Config(default_factory=lambda: ["attack"])
    """A list of tags associated with the attack for logging."""
    hooks: list[EvalHook] = Field(default_factory=list, exclude=True, repr=False)
    """Hooks to run at various points in the attack lifecycle."""

    # Override the task factory as the target will replace it.
    task_factory: t.Callable[[In], Task[..., Out]] = Field(  # type: ignore[assignment]
        default_factory=lambda: None,
        repr=False,
        init=False,
    )

    def model_post_init(self, context: t.Any) -> None:
        self.task_factory = self.target.task_factory
        super().model_post_init(context)

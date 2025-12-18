import typing as t

from pydantic import ConfigDict, Field, SkipValidation

from dreadnode.airt.target.base import Target
from dreadnode.eval.hooks.base import EvalHook
from dreadnode.meta import Config
from dreadnode.optimization.study import Study

In = t.TypeVar("In")
Out = t.TypeVar("Out")


class Attack(Study[In, Out]):
    """
    A declarative configuration for executing an AIRT attack.

    Attack automatically derives its task from the target.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    target: t.Annotated[SkipValidation[Target[In, Out]], Config()]
    """The target to attack."""

    tags: list[str] = Config(default_factory=lambda: ["attack"])
    """A list of tags associated with the attack for logging."""

    hooks: list[EvalHook] = Field(default_factory=list, exclude=True, repr=False)
    """Hooks to run at various points in the attack lifecycle."""

    def model_post_init(self, context: t.Any) -> None:
        """Initialize attack by deriving task from target."""
        if self.task is None:
            self.task = self.target.task  # type: ignore[attr-defined]
        super().model_post_init(context)

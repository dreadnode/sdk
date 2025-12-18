import typing as t

from pydantic import ConfigDict, Field

from dreadnode.airt.target.base import Target
from dreadnode.eval.hooks.base import EvalHook
from dreadnode.optimization.study import Study

In = t.TypeVar("In")
Out = t.TypeVar("Out")


class Attack(Study[In, Out]):
    """
    A declarative configuration for executing an AIRT attack.

    Attack automatically derives its task from the target, so users
    only need to specify the target system to attack.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    target: Target[In, Out]
    """The target system to attack."""

    tags: list[str] = Field(default_factory=lambda: ["attack"])
    """Tags associated with the attack for logging."""

    hooks: list[EvalHook] = Field(default_factory=list, exclude=True, repr=False)
    """Hooks to run at various points in the attack lifecycle."""

    def model_post_init(self, context: t.Any) -> None:
        """Initialize attack by deriving task from target."""
        if self.task is None:
            self.task = self.target.task
        super().model_post_init(context)

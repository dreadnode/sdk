import typing as t

import typing_extensions as te
from pydantic import ConfigDict, Field

from dreadnode.airt.target.base import Target
from dreadnode.meta import Config
from dreadnode.optimization import Study
from dreadnode.task import Task

In = te.TypeVar("In", default=t.Any)
Out = te.TypeVar("Out", default=t.Any)


class Attack(Study[In, Out]):
    """
    A declarative configuration for executing an AIRT attack.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    target: t.Annotated[Target[In, Out], Config()]
    """The target to attack."""

    task_factory: t.Callable[[In], Task[..., Out]] = Field(default_factory=lambda: None)

    def model_post_init(self, context: t.Any) -> None:
        self.task_factory = self.target.task_factory
        super().model_post_init(context)

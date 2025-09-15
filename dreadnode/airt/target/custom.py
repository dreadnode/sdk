import typing as t

from pydantic import ConfigDict

from dreadnode.airt.target.base import In, Out, Target
from dreadnode.common_types import Unset
from dreadnode.meta import Config
from dreadnode.task import Task


class CustomTarget(Target[t.Any, Out]):
    """
    Adapts any Task to be used as an attackable target.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    task: t.Annotated[Task[..., Out], Config()]
    """The task to be called with attack input."""
    input_param_name: str | None = None
    """
    The name of the parameter in the task's signature where the attack input should be injected.
    Otherwise the first non-optional parameter will be used, or no injection will occur.
    """

    @property
    def name(self) -> str:
        """Returns the name of the target."""
        return self.task.name

    def model_post_init(self, context: t.Any) -> None:
        super().model_post_init(context)

        if self.input_param_name is None:
            for name, default in self.task.defaults.items():
                if isinstance(default, Unset):
                    self.input_param_name = name
                    break

        if self.input_param_name is None:
            raise ValueError(f"Could not determine input parameter for {self.task!r}")

    def task_factory(self, input: In) -> Task[..., Out]:
        task = self.task
        if self.input_param_name is not None:
            task = self.task.configure(**{self.input_param_name: input})
        return task.with_(tags=["target"], append=True)

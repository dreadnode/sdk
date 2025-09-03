import typing as t
from abc import ABC, abstractmethod

import typing_extensions as te
from pydantic import ConfigDict

from dreadnode.meta import Model
from dreadnode.task import Task
from dreadnode.types import Unset

In = te.TypeVar("In", default=t.Any)
Out = te.TypeVar("Out", default=t.Any)


class BaseTarget(ABC, t.Generic[In, Out]):
    """Abstract base class for any target that can be attacked."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the target."""
        ...

    @abstractmethod
    def as_task(self, input: In) -> Task[..., Out]:
        """Creates a Task that will run the given candidate against the target."""
        ...


class Target(Model, BaseTarget[t.Any, Out]):
    """
    Adapts an existing Task to be used as an attackable target.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    task: Task[..., Out]
    """The task to be called with attack input."""
    input_param_name: str | None = None
    """
    The name of the parameter in the task's signature where the attack input should be injected.
    Otherwise the first non-optional parameter will be used, or no injection will occur.
    """

    @classmethod
    def from_task(cls, task: Task[..., Out]) -> "Target[t.Any, Out]":
        return cls(task=task)

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

    def as_task(self, input: In) -> Task[..., Out]:
        task = self.task
        if self.input_param_name is not None:
            task = self.task.configure(**{self.input_param_name: input})
        return task

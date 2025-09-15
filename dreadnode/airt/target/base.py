import abc
import typing as t

import typing_extensions as te

from dreadnode.meta import Model
from dreadnode.task import Task

In = te.TypeVar("In", default=t.Any)
Out = te.TypeVar("Out", default=t.Any)


class Target(Model, abc.ABC, t.Generic[In, Out]):
    """Abstract base class for any target that can be attacked."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Returns the name of the target."""
        raise NotImplementedError

    @abc.abstractmethod
    def task_factory(self, input: In) -> Task[..., Out]:
        """Creates a Task that will run the given input against the target."""
        raise NotImplementedError

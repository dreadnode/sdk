import typing as t
from abc import ABC, abstractmethod

import typing_extensions as te

from dreadnode.task import Task

In = te.TypeVar("In", default=t.Any)
Out = te.TypeVar("Out", default=t.Any)


class Target(ABC, t.Generic[In, Out]):
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

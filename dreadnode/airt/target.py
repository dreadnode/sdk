from __future__ import annotations

import abc
import typing as t
from functools import cached_property
from typing import TYPE_CHECKING

import typing_extensions as te
from pydantic import ConfigDict

from dreadnode.core.generators.generator import GenerateParams, Generator, get_generator
from dreadnode.core.generators.message import Message
from dreadnode.core.meta import Config, Model
from dreadnode.core.types.common import AnyDict, Unset

if TYPE_CHECKING:
    from dreadnode.core.task import Task

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


class LLMTarget(Target[t.Any, str]):
    """
    Target backed by a generator for LLM inference.

    - Accepts as input any message, conversation, or content-like structure.
    - Returns just the generated text from the LLM.
    """

    model: str | Generator
    """
    The inference model, as a rigging generator identifier string or object.

    See: https://docs.dreadnode.io/open-source/rigging/topics/generators
    """

    params: AnyDict | GenerateParams | None = Config(default=None, expose_as=AnyDict | None)
    """
    Optional generation parameters.

    See: https://docs.dreadnode.io/open-source/rigging/api/generator#generateparams
    """

    @cached_property
    def generator(self) -> Generator:
        return get_generator(self.model) if isinstance(self.model, str) else self.model

    @property
    def name(self) -> str:
        return self.generator.to_identifier(short=True).split("/")[-1]

    def task_factory(self, input: t.Any) -> Task[[], str]:
        from dreadnode import task

        messages = Message.fit_as_list(input) if input else []
        params = (
            self.params
            if isinstance(self.params, GenerateParams)
            else GenerateParams.model_validate(self.params)
            if self.params
            else GenerateParams()
        )

        @task(name=f"target - {self.name}", tags=["target"])
        async def generate(
            messages: list[Message] = messages,
            params: GenerateParams = params,
        ) -> str:
            generated = (await self.generator.generate_messages([messages], [params]))[0]
            if isinstance(generated, BaseException):
                raise generated
            return generated.message.content

        return generate

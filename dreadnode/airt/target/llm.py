import typing as t
from functools import cached_property

import rigging as rg

from dreadnode.airt.target.base import Target
from dreadnode.common_types import AnyDict
from dreadnode.meta import Config
from dreadnode.task import Task


class LLMTarget(Target[t.Any, str]):
    """
    Target backed by a rigging generator for LLM inference.

    - Accepts as input any message, conversation, or content-like structure.
    - Returns just the generated text from the LLM.
    """

    model: str | rg.Generator
    """
    The inference model, as a rigging generator identifier string or object.

    See: https://docs.dreadnode.io/open-source/rigging/topics/generators
    """

    params: AnyDict | rg.GenerateParams | None = Config(default=None, expose_as=AnyDict | None)
    """
    Optional generation parameters.

    See: https://docs.dreadnode.io/open-source/rigging/api/generator#generateparams
    """

    @cached_property
    def generator(self) -> rg.Generator:
        return rg.get_generator(self.model) if isinstance(self.model, str) else self.model

    @property
    def name(self) -> str:
        return self.generator.to_identifier(short=True).split("/")[-1]

    def task_factory(self, input: t.Any) -> Task[[], str]:
        from dreadnode import task

        messages = rg.Message.fit_as_list(input) if input else []
        params = (
            self.params
            if isinstance(self.params, rg.GenerateParams)
            else rg.GenerateParams.model_validate(self.params)
            if self.params
            else rg.GenerateParams()
        )

        @task(name="target - {self.name}", tags=["target"])
        async def generate(
            messages: list[rg.Message] = messages,
            params: rg.GenerateParams = params,
        ) -> str:
            generated = (await self.generator.generate_messages([messages], [params]))[0]
            if isinstance(generated, BaseException):
                raise generated
            return generated.message.content

        return generate

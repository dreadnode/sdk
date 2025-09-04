import typing as t

import rigging as rg
from pydantic import PrivateAttr

from dreadnode.airt.target.base import Target
from dreadnode.meta import Config, Model
from dreadnode.task import Task
from dreadnode.types import AnyDict


class LLMTarget(Model, Target[t.Any, str]):
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

    _generator: rg.Generator | None = PrivateAttr(None, init=False)

    @property
    def name(self) -> str:
        if self._generator is None:
            return "unknown"
        return self._generator.to_identifier(short=True)

    def as_task(self, input: t.Any) -> Task[[], str]:
        from dreadnode import task

        messages = rg.Message.fit_as_list(input) if input else []
        params = (
            self.params
            if isinstance(self.params, rg.GenerateParams)
            else rg.GenerateParams.model_validate(self.params)
            if self.params
            else None
        )

        @task(name=f"generate - {self.name}", label="llm_target_generate", tags=["target"])
        async def generate(
            messages: list[rg.Message] = messages, params: rg.GenerateParams | None = params
        ) -> str:
            if self._generator is None:
                raise ValueError("Generator not initialized")
            generated = (await self._generator.generate_messages([messages], [params]))[0]
            if isinstance(generated, BaseException):
                raise generated
            return generated.message.content

        return generate

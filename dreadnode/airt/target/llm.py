from functools import cached_property

import rigging as rg

from dreadnode import task
from dreadnode.airt.target.base import Target
from dreadnode.common_types import AnyDict
from dreadnode.data_types.message import Message as DnMessage
from dreadnode.meta import Config
from dreadnode.task import Task


class LLMTarget(Target[DnMessage, DnMessage]):
    """
    Target backed by a rigging generator for LLM inference.

    - Accepts dn.Message as input
    - Converts to Rigging format only for LLM API call
    - Returns dn.Message as output (supports multimodal responses)
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

    def task_factory(self, input: DnMessage) -> Task[[], DnMessage]:
        """
        create a task that:
        1. Takes dn.Message as input (auto-logged via to_serializable())
        2. Converts to rg.Message only for LLM API call
        3. Returns dn.Message with full multimodal content (text/images/audio/video)

        Args:
            input: The dn.Message to send to the LLM

        Returns:
            Task that executes the LLM call and returns dn.Message

        Raises:
            TypeError: If input is not a dn.Message
            ValueError: If the message has no content
        """
        if not isinstance(input, DnMessage):
            raise TypeError(f"Expected dn.Message, got {type(input).__name__}")

        if not input.content:
            raise ValueError("Message must have at least one content part")

        dn_message = input
        params = (
            self.params
            if isinstance(self.params, rg.GenerateParams)
            else rg.GenerateParams.model_validate(self.params)
            if self.params
            else rg.GenerateParams()
        )

        @task(name=f"target - {self.name}", tags=["target"])
        async def generate(
            message: DnMessage = dn_message,
            params: rg.GenerateParams = params,
        ) -> DnMessage:
            """Execute LLM generation task."""
            rg_message = message.to_rigging()

            generated = (await self.generator.generate_messages([[rg_message]], [params]))[0]
            if isinstance(generated, BaseException):
                raise generated

            return DnMessage.from_rigging(generated.message)

        return generate

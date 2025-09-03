import typing as t

from pydantic import PrivateAttr
from rigging import Generator, Message

from dreadnode.airt.target.base import BaseTarget
from dreadnode.task import Task


class LLMTarget(BaseTarget[t.Any, str]):
    """Represents a rigging.Generator as an attackable target."""

    model: str | Generator
    """The model or endpoint to attack, as a rigging generator identifier string or object."""

    _generator: Generator | None = PrivateAttr(None, init=False)

    @property
    def name(self) -> str:
        return self.model.split(",")[0]

    def as_task(self, input: t.Any) -> Task[[], str]:
        from dreadnode import task

        @task()
        async def model_generate() -> str:
            chat = await self._generator.chat(t.cast("Message", input)).run()
            return chat.last.content

        return model_generate

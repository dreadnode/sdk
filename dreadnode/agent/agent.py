import typing as t

from pydantic import ConfigDict, Field, PrivateAttr
from rigging import get_generator
from rigging.caching import CacheMode, apply_cache_mode_to_messages
from rigging.chat import Chat
from rigging.generator import GenerateParams, Generator
from rigging.message import inject_system_content
from rigging.tools import ToolMode
from rigging.transform import (
    PostTransform,
    Transform,
    make_tools_to_xml_transform,
    tools_to_json_in_xml_transform,
    tools_to_json_transform,
    tools_to_json_with_tag_transform,
)

from dreadnode.agent.hooks import Hook
from dreadnode.agent.runnable import Runnable
from dreadnode.agent.stop import StopCondition
from dreadnode.agent.types import Message, Tool


class Agent(Runnable):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str | None = None
    """Inference model (rigging generator identifier)."""
    instructions: str | None = None
    """The agent's core instructions."""
    tools: list[Tool[..., t.Any]] = Field(default_factory=list)
    """Tools the agent can use."""
    tool_mode: ToolMode = "auto"
    """The tool calling mode to use (e.g., "xml", "json-with-tag", "json-in-xml", "api") - default is "auto"."""
    caching: CacheMode | None = None
    """How to handle cache_control entries on inference messages."""
    stop_conditions: list[StopCondition] = Field(default_factory=list)
    """The logical condition for successfully stopping a run."""
    max_steps: int = 10
    """The maximum number of steps (generation + tool calls) the agent can take before stopping."""
    hooks: list[Hook] = Field(default_factory=list, exclude=True)
    """Hooks to run at various points in the agent's lifecycle."""

    _generator: Generator | None = PrivateAttr(None, init=False)

    def _get_transforms(self) -> list[Transform]:
        transforms = []

        if self.tools:
            match self.tool_mode:
                case "xml":
                    transforms.append(
                        make_tools_to_xml_transform(self.tools, add_tool_stop_token=True)
                    )
                case "json-in-xml":
                    transforms.append(tools_to_json_in_xml_transform)
                case "json-with-tag":
                    transforms.append(tools_to_json_with_tag_transform)
                case "json":
                    transforms.append(tools_to_json_transform)

        return transforms

    def get_prompt(self) -> str:
        prompt = "You are an agent that can use tools to assist with tasks."
        if self.instructions:
            prompt += f"\n\n<instructions>\n{self.instructions}\n</instructions>"
        return prompt

    async def generate(
        self,
        messages: list[Message],
        *,
        model: str | Generator | None = None,
    ) -> Chat:
        if model is None and self.model is None:
            raise ValueError("No model specified for agent generation.")

        # Build our default generator if we haven't already
        if model is None and self._generator is None:
            self._generator = get_generator(self.model)

        generator = self._generator

        # Override if the user supplied one
        if model is not None:
            if isinstance(model, str):
                self._generator = get_generator(model)
            elif isinstance(model, Generator):
                self._generator = model

        if generator is None:
            raise TypeError("Model must be a string or a Generator instance.")

        params = GenerateParams(
            tools=[tool.api_definition for tool in self.tools],
        )
        messages = inject_system_content(messages, self.get_prompt())

        if self.tool_mode == "auto" and self.tools:
            self.tool_mode = "api" if await generator.supports_function_calling() else "json-in-xml"

        transforms = self._get_transforms()
        post_transforms: list[PostTransform | None] = []
        for transform_callback in transforms:
            messages, params, post_transform = await transform_callback(messages, params)
            post_transforms.append(post_transform)

        try:
            messages = apply_cache_mode_to_messages(self.caching, [messages])[0]

            generated = (await generator.generate_messages([messages], [params]))[0]
            if isinstance(generated, BaseException):
                raise generated  # noqa: TRY301

            chat = Chat(
                messages,
                [generated.message],
                generator=generator,
                params=params,
                stop_reason=generated.stop_reason,
                usage=generated.usage,
                extra=generated.extra,
            )

        except Exception as error:  # noqa: BLE001
            chat = Chat(
                messages,
                [],
                generator=generator,
                params=params,
                failed=True,
                error=error,
            )

        for post_transform in [transform for transform in post_transforms if transform]:
            chat = await post_transform(chat) or chat

        return chat

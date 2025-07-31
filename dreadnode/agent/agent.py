import typing as t
from contextlib import asynccontextmanager

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
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

from dreadnode.agent.configurable import Configurable
from dreadnode.agent.events import AgentStalled, Event
from dreadnode.agent.hooks.base import retry_with_feedback
from dreadnode.agent.reactions import Hook
from dreadnode.agent.result import AgentResult
from dreadnode.agent.stop import StopCondition, StopNever
from dreadnode.agent.thread import Thread
from dreadnode.agent.types import Message, Tool
from dreadnode.util import get_callable_name, shorten_string


class Agent(
    BaseModel,
    Configurable,
    test=["model", "instructions", "max_steps", "tool_mode", "caching"],
):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    """The name of the agent."""
    description: str = ""
    """A brief description of the agent's purpose."""

    model: str | None = None
    """Inference model (rigging generator identifier)."""
    instructions: str | None = None
    """The agent's core instructions."""
    tools: list[Tool[..., t.Any]] = Field(default_factory=list)
    """Tools the agent can use."""
    tool_mode: ToolMode = Field("auto", repr=False)
    """The tool calling mode to use (e.g., "xml", "json-with-tag", "json-in-xml", "api") - default is "auto"."""
    caching: CacheMode | None = Field(None, repr=False)
    """How to handle cache_control entries on inference messages."""
    max_steps: int = 10
    """The maximum number of steps (generation + tool calls) the agent can take before stopping."""

    stop_conditions: list[StopCondition] = Field(default_factory=list)
    """The logical condition for successfully stopping a run."""
    hooks: list[Hook] = Field(default_factory=list, exclude=True, repr=False)
    """Hooks to run at various points in the agent's lifecycle."""

    _generator: Generator | None = PrivateAttr(None, init=False)

    def __repr__(self) -> str:
        description = shorten_string(self.description or "", 50)
        model = (self.model or "").split(",")[0]

        parts: list[str] = [
            f"name='{self.name}'",
            f"description='{description}'",
            f"model='{model}'",
        ]

        if self.instructions:
            instructions = shorten_string(self.instructions or "", 50)
            parts.append(f"instructions='{instructions}'")
        if self.tools:
            tool_names = ", ".join(tool.name for tool in self.tools)
            parts.append(f"tools=[{tool_names}]")
        if self.stop_conditions:
            stop_conditions = ", ".join(repr(cond) for cond in self.stop_conditions)
            parts.append(f"stop_conditions=[{stop_conditions}]")
        if self.hooks:
            hooks = ", ".join(get_callable_name(hook, short=True) for hook in self.hooks)
            parts.append(f"hooks=[{hooks}]")

        return f"{self.__class__.__name__}({', '.join(parts)})"

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

        messages = list(messages)  # Ensure we have a mutable list
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

    def reset(self) -> Thread:
        previous = self.thread
        self.thread = Thread()
        return previous

    @asynccontextmanager
    async def stream(
        self,
        user_input: str,
        *,
        thread: Thread | None = None,
    ) -> t.AsyncIterator[t.AsyncGenerator[Event, None]]:
        thread = thread or self.thread
        async with thread.stream(
            self, user_input, commit="always" if thread == self.thread else "on-success"
        ) as stream:
            yield stream

    async def run(
        self,
        user_input: str,
        *,
        thread: Thread | None = None,
    ) -> AgentResult:
        thread = thread or Thread()
        return await thread.run(
            self, user_input, commit="always" if thread == self.thread else "on-success"
        )


thread = Thread()
agent.run(thread)
agent.run(thread)


class TaskAgent(Agent):
    """
    A specialized agent for running tasks with a focus on completion and reporting.
    It extends the base Agent class to provide task-specific functionality.
    """

    def model_post_init(self, _: t.Any) -> None:
        from dreadnode.agent.tools import finish_task, update_todo

        if not any(tool for tool in self.tools if tool.name == "finish_task"):
            self.tools.append(finish_task)

        if not any(tool for tool in self.tools if tool.name == "update_todo"):
            self.tools.append(update_todo)

        # Force the agent to use finish_task
        self.stop_conditions.append(StopNever())
        self.hooks.insert(
            0,
            retry_with_feedback(
                event_type=AgentStalled,
                feedback="Continue the task if possible or use the 'finish_task' tool to complete it.",
            ),
        )


class MyAgent(TaskAgent):
    @entrypoint
    def run_with_args(self, arg1: str, arg2: int) -> Summary:
        prompt = "..."
        return prompt

    @entrypoint
    async def run_async_with_args(self, arg1: str, arg2: int) -> Summary:
        prompt = "..."
        return prompt


MyAgent.as_kickoff(entrpoints=["run_with_args"])
MyAgent.run_with_args.as_kickoff()

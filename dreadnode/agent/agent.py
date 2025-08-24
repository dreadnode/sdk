import inspect
import typing as t
from contextlib import asynccontextmanager

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator
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

from dreadnode.agent.configurable import configurable
from dreadnode.agent.events import AgentStalled, Event
from dreadnode.agent.hooks.base import retry_with_feedback
from dreadnode.agent.reactions import Hook
from dreadnode.agent.result import AgentResult
from dreadnode.agent.stop import StopCondition, StopNever
from dreadnode.agent.thread import Thread
from dreadnode.agent.tools.base import AnyTool, Tool, Toolset
from dreadnode.agent.types import Message
from dreadnode.util import flatten_list, get_callable_name, shorten_string


@configurable(["model", "instructions", "max_steps", "tool_mode", "caching"])
class Agent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    name: str
    """The name of the agent."""
    description: str = ""
    """A brief description of the agent's purpose."""

    model: str | None = None
    """Inference model (rigging generator identifier)."""
    instructions: str | None = None
    """The agent's core instructions."""
    tools: list[AnyTool | Toolset] = []
    """Tools the agent can use."""
    tool_mode: t.Annotated[ToolMode, Field(repr=False)] = "auto"
    """The tool calling mode to use (e.g., "xml", "json-with-tag", "json-in-xml", "api") - default is "auto"."""
    caching: t.Annotated[CacheMode | None, Field(repr=False)] = None
    """How to handle cache_control entries on inference messages."""
    max_steps: int = 100
    """The maximum number of steps (generation + tool calls) the agent can take before stopping."""

    stop_conditions: list[StopCondition] = []
    """The logical condition for successfully stopping a run."""
    hooks: t.Annotated[list[Hook], Field(exclude=True, repr=False)] = []
    """Hooks to run at various points in the agent's lifecycle."""
    thread: Thread = Field(default_factory=Thread, exclude=True, repr=False)
    """Stateful thread for this agent, for when otherwise not specified during execution."""

    _generator: Generator | None = PrivateAttr(None, init=False)

    @field_validator("tools", mode="before")
    @classmethod
    def validate_tools(cls, value: t.Any) -> t.Any:
        tools: list[AnyTool | Toolset] = []
        for tool in flatten_list(list(value)):
            if isinstance(tool, Toolset):
                tools.append(tool)
                continue

            interior_tools = [
                val
                for _, val in inspect.getmembers(
                    tool,
                    predicate=lambda x: isinstance(x, Tool),
                )
            ]
            if interior_tools:
                tools.extend(interior_tools)
            elif not isinstance(tool, Tool):
                tools.append(Tool.from_callable(tool))
            else:
                tools.append(tool)
        return tools

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
                        make_tools_to_xml_transform(self.all_tools, add_tool_stop_token=True)
                    )
                case "json-in-xml":
                    transforms.append(tools_to_json_in_xml_transform)
                case "json-with-tag":
                    transforms.append(tools_to_json_with_tag_transform)
                case "json":
                    transforms.append(tools_to_json_transform)

        return transforms

    @property
    def all_tools(self) -> list[AnyTool]:
        """Returns a flattened list of all available tools."""
        flat_tools: list[AnyTool] = []
        for item in self.tools:
            if isinstance(item, Toolset):
                flat_tools.extend(item.get_tools())
            elif isinstance(item, Tool):
                flat_tools.append(item)
        return flat_tools

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
            tools=[tool.api_definition for tool in self.all_tools],
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
        thread = thread or self.thread
        return await thread.run(
            self, user_input, commit="always" if thread == self.thread else "on-success"
        )


class TaskAgent(Agent):
    """
    A specialized agent for running tasks with a focus on completion and reporting.
    It extends the base Agent class to provide task-specific functionality.

    - Automatically includes the `finish_task` and `update_todo` tools.
    - Installs a default StopNever condition to trigger stalling behavior when no tools calls are made.
    - Uses the `AgentStalled` event to handle stalled tasks by pushing the model to continue or finish the task.
    """

    def model_post_init(self, _: t.Any) -> None:
        from dreadnode.agent.tools import mark_complete, update_todo

        if not any(tool for tool in self.tools if tool.name == "finish_task"):
            self.tools.append(mark_complete)

        if not any(tool for tool in self.tools if tool.name == "update_todo"):
            self.tools.append(update_todo)

        # Force the agent to use finish_task
        self.stop_conditions.append(StopNever())
        self.hooks.insert(
            0,
            retry_with_feedback(
                event_type=AgentStalled,
                feedback="Continue the task if possible or use the 'mark_complete' tool to complete it.",
            ),
        )

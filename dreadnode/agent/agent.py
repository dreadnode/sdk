import inspect
import json
import re
import typing as t
from contextlib import AsyncExitStack, aclosing, asynccontextmanager
from copy import deepcopy
from textwrap import dedent

import litellm
import rigging as rg
import typing_extensions as te
from loguru import logger
from pydantic import AfterValidator, ConfigDict, Field, PrivateAttr, SkipValidation, field_validator
from rigging.message import inject_system_content
from ulid import ULID

from dreadnode.agent.error import MaxStepsError
from dreadnode.agent.hooks import Hook, retry_with_feedback
from dreadnode.agent.reactions import (
    Continue,
    Fail,
    Finish,
    Reaction,
    Retry,
    RetryWithFeedback,
)
from dreadnode.agent.stop import StopCondition, never
from dreadnode.agent.tools import AnyTool, Tool, Toolset, discover_tools_on_obj
from dreadnode.agent.tools.base import ToolMode
from dreadnode.agent.tools.planning import update_todo
from dreadnode.agent.tools.tasking import finish_task, give_up_on_task
from dreadnode.agent.trajectory import (
    AgentEnd,
    AgentError,
    AgentEvent,
    AgentStalled,
    AgentStart,
    AgentStep,
    AgentStopReason,
    GenerationStep,
    ReactStep,
    ToolEnd,
    ToolError,
    ToolStart,
    ToolStep,
    Trajectory,
)
from dreadnode.meta import Component, Config, Model, component
from dreadnode.meta.introspect import get_config_model, get_inputs_and_params_from_config_model
from dreadnode.scorers import ScorersLike
from dreadnode.util import (
    flatten_list,
    get_callable_name,
    join_generators,
    safe_repr,
    shorten_string,
    warn_at_user_stacklevel,
)

litellm.suppress_debug_info = True

CommitBehavior = t.Literal["always", "on-success"]


class AgentWarning(UserWarning):
    """Warning raised when an agent is used in a way that may not be safe or intended."""


class Agent(Model):
    """
    Agent abstraction for applying tools, event logic, and message state to LLM generation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    agent_id: ULID = Field(default_factory=ULID)
    """The unique identifier for this agent instance."""
    name: str
    """The name of the agent."""
    description: t.Annotated[str, AfterValidator(dedent)] = ""
    """A brief description of the agent's purpose."""
    tags: list[str] = Config(default_factory=lambda: ["agent"])
    """A list of tags associated with the agent."""
    label: str | None = Config(default=None)
    """Specific label for tracing, otherwise derived from the name."""

    model: str | rg.Generator | None = Config(default=None, expose_as=str | None)
    """Inference model (rigging generator or identifier)."""
    instructions: t.Annotated[str | None, AfterValidator(lambda x: dedent(x) if x else x)] = Config(
        default=None
    )
    """The agent's core instructions."""
    max_steps: int = Config(default=10)
    """The maximum number of steps (generation + tool calls)."""
    caching: rg.caching.CacheMode | None = Config(default=None, repr=False)
    """How to handle cache_control entries on inference messages."""

    tools: t.Annotated[list[AnyTool | Toolset], SkipValidation] = Config(default_factory=list)
    """Tools the agent can use."""
    tool_mode: ToolMode = Config(default="auto", repr=False)
    """The tool calling mode to use."""

    hooks: list[Hook] = Field(default_factory=list, exclude=True, repr=False)
    """Hooks to run at various points in the agent's lifecycle."""
    stop_conditions: list[StopCondition] = Field(default_factory=list)
    """The logical condition for successfully stopping a run."""
    trajectory: Trajectory = Field(default_factory=Trajectory, exclude=True, repr=False)
    """Stateful trajectory for this agent, for when otherwise not specified during execution."""
    scorers: ScorersLike[Trajectory] = Field(default_factory=list)
    """Scorers to evaluate the agent output."""
    assert_scores: list[str] | t.Literal[True] = Field(default_factory=list)
    """Scores to ensure are truthy, otherwise the agent task is marked as failed."""

    _generator: rg.Generator | None = PrivateAttr(None, init=False)

    @field_validator("tools", mode="before")
    @classmethod
    def validate_tools(cls, value: t.Any) -> t.Any:
        """
        Validates and normalizes the input for the 'tools' field.

        This validator flattens lists, extracts tools from objects (using `discover_tools_on_obj`),
        and converts bare callables or Components into `Tool` instances.

        Args:
            value: The raw input value for the tools field.

        Returns:
            A list of validated `Tool` or `Toolset` instances.
        """
        tools: list[AnyTool | Toolset] = []
        for tool in flatten_list(list(value)):
            if isinstance(tool, Toolset | Tool):
                tools.append(tool)
            elif interior_tools := discover_tools_on_obj(tool):
                tools.extend(interior_tools)
            else:
                tools.append(
                    Tool.from_callable(tool if isinstance(tool, Component) else component(tool))
                )

        return tools

    def __repr__(self) -> str:
        """
        Returns a string representation of the Agent instance.

        Includes the class name, agent name, abbreviated description, model name,
        and summaries of instructions, tools, stop conditions, and hooks.
        """
        description = shorten_string(self.description or "", 50)

        parts: list[str] = [
            f"name='{self.name}'",
            f"description='{description}'",
            f"model='{self.model_name}'",
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

    @property
    def model_name(self) -> str | None:
        """
        Retrieves the string identifier of the model.

        Returns:
            The model identifier string if configured, otherwise None.
        """
        if self.model is not None:
            return self.generator.to_identifier(short=True)
        return None

    @property
    def generator(self) -> rg.Generator:
        """
        Retrieves or initializes the Rigging Generator instance.

        If the model is defined as a string, a new Generator is created.
        If it is already a Generator, it is returned directly.

        Returns:
            The active `rg.Generator` instance.

        Raises:
            TypeError: If the model field is neither a string nor a Generator.
        """
        if self._generator is not None:
            return self._generator

        if isinstance(self.model, str):
            self._generator = rg.get_generator(self.model)
        elif isinstance(self.model, rg.Generator):
            self._generator = self.model
        else:
            raise TypeError("Model must be a string or a Generator instance.")

        return self._generator

    @property
    def all_tools(self) -> list[AnyTool]:
        """
        Returns a flattened list of all available tools.

        This iterates through the `tools` list and expands any `Toolset` instances
        into their individual constituent tools.

        Returns:
            A list of all individual `Tool` instances available to the agent.
        """
        flat_tools: list[AnyTool] = []
        for item in self.tools:
            if isinstance(item, Toolset):
                flat_tools.extend(item.get_tools())
            elif isinstance(item, Tool):
                flat_tools.append(item)
        return flat_tools

    def clone(self) -> te.Self:
        """
        Creates a deep copy of the current agent instance.

        Returns:
            A new Agent instance with identical attributes.
        """
        return self.model_copy(deep=True)

    def with_(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        label: str | None = None,
        model: str | rg.Generator | None = None,
        instructions: str | None = None,
        max_steps: int | None = None,
        caching: rg.caching.CacheMode | None = None,
        tools: list[AnyTool | Toolset] | None = None,
        tool_mode: ToolMode | None = None,
        hooks: list[Hook] | None = None,
        stop_conditions: list[StopCondition] | None = None,
        scorers: ScorersLike[Trajectory] | None = None,
        assert_scores: list[str] | t.Literal[True] | None = None,
        append: bool = False,
    ) -> te.Self:
        """
        Creates a modified clone of the agent with updated attributes.

        Args:
            name: New name for the agent.
            description: New description.
            tags: Tags to set or append.
            label: Tracing label.
            model: New model identifier or Generator.
            instructions: New system instructions.
            max_steps: New maximum step limit.
            caching: Cache mode strategy.
            tools: Tools to set or append.
            tool_mode: Strategy for tool calling (e.g., 'xml', 'json').
            hooks: Lifecycle hooks to set or append.
            stop_conditions: Stop conditions to set or append.
            scorers: Scorers to set or append.
            assert_scores: Scores validation logic to set or append.
            append: If True, list-based attributes (tags, tools, hooks, etc.)
                will be appended to the existing lists rather than replacing them.

        Returns:
            A new Agent instance with the specified modifications.
        """
        new = self.clone()

        new.name = name or new.name
        new.description = description or new.description
        new.label = label or new.label
        new.model = model or new.model
        new.instructions = instructions or new.instructions
        new.max_steps = max_steps or new.max_steps
        new.caching = caching or new.caching
        new.tool_mode = tool_mode or new.tool_mode

        if append:
            new.tags = [*new.tags, *(tags or [])]
            new.tools = [*new.tools, *(tools or [])]
            new.hooks = [*new.hooks, *(hooks or [])]
            new.stop_conditions = [*new.stop_conditions, *(stop_conditions or [])]
            new.scorers = [*new.scorers, *(scorers or [])]
            if isinstance(assert_scores, bool):
                new.assert_scores = assert_scores
            elif isinstance(new.assert_scores, list):
                new.assert_scores = [*new.assert_scores, *(assert_scores or [])]
            else:
                new.assert_scores = assert_scores or new.assert_scores
        else:
            new.tags = tags if tags is not None else new.tags
            new.tools = tools if tools is not None else new.tools
            new.hooks = hooks if hooks is not None else new.hooks
            new.stop_conditions = (
                stop_conditions if stop_conditions is not None else new.stop_conditions
            )
            new.scorers = scorers if scorers is not None else new.scorers
            new.assert_scores = assert_scores if assert_scores is not None else new.assert_scores

        # Retrigger model_post_init functions to ensure consistency
        new.model_post_init(None)

        return new

    def _get_transforms(self) -> list[rg.Transform]:
        """
        Resolves the message transforms required for the current configuration.

        Determines the appropriate transforms based on the agent's `tool_mode`
        (e.g., converting tools to XML, JSON, or Pythonic formats).

        Returns:
            A list of `rg.Transform` callables.
        """
        transforms = []

        if self.tools:
            match self.tool_mode:
                case "xml":
                    transforms.append(
                        rg.transform.make_tools_to_xml_transform(
                            self.all_tools, add_tool_stop_token=True
                        )
                    )
                case "json-in-xml":
                    transforms.append(rg.transform.tools_to_json_in_xml_transform)
                case "json-with-tag":
                    transforms.append(rg.transform.tools_to_json_with_tag_transform)
                case "json":
                    transforms.append(rg.transform.tools_to_json_transform)
                case "pythonic":
                    transforms.append(rg.transform.tools_to_pythonic_transform)

        return transforms

    async def _generate(
        self,
        messages: list[rg.Message],
    ) -> rg.Chat:
        """
        Executes a single LLM generation step.

        This method handles:
        1. Injecting tool definitions into params.
        2. Applying transforms (pre and post) for tool schemas.
        3. Applying caching strategies.
        4. Calling the underlying generator.
        5. Wrapping the result (or error) in a `rg.Chat` object.

        Args:
            messages: The conversation history to send to the model.

        Returns:
            An `rg.Chat` object containing the generated message, usage stats, or error details.
        """

        messages = list(messages)  # Ensure we have a mutable list
        params = rg.GenerateParams(
            tools=[tool.api_definition for tool in self.all_tools],
        )
        # messages = inject_system_content(messages, self.get_prompt())

        if self.tool_mode == "auto" and self.tools:
            self.tool_mode = (
                "api" if await self.generator.supports_function_calling() else "json-in-xml"
            )

        transforms = self._get_transforms()
        post_transforms: list[rg.PostTransform | None] = []
        for transform_callback in transforms:
            messages, params, post_transform = await transform_callback(messages, params)
            post_transforms.append(post_transform)

        try:
            messages = rg.caching.apply_cache_mode_to_messages(self.caching, [messages])[0]

            logger.trace(f"Generating with model '{self.generator.model}'. Messages: {messages!r}")

            generated = (await self.generator.generate_messages([messages], [params]))[0]
            if isinstance(generated, BaseException):
                raise generated  # noqa: TRY301

            chat = rg.Chat(
                messages,
                [generated.message],
                generator=self.generator,
                params=params,
                stop_reason=generated.stop_reason,
                usage=generated.usage,
                extra=generated.extra,
            )

        except Exception as error:  # noqa: BLE001
            logger.opt(exception=True).error("Error during generation")
            chat = rg.Chat(
                messages,
                [],
                generator=self.generator,
                params=params,
                failed=True,
                error=error,
            )

        for post_transform in [transform for transform in post_transforms if transform]:
            chat = await post_transform(chat) or chat

        return chat

    async def _dispatch(self, event: AgentEvent) -> t.AsyncIterator[AgentEvent]:  # noqa: PLR0912
        """
        The internal event bus for the agent.

        Dispatches an event to all registered hooks, collects reactions,
        and resolves flow control.

        Logic:
        1. Yields the initial event.
        2. Invokes hooks.
        3. Collects `Reaction` objects (Finish, Retry, Continue).
        4. Prioritizes reactions: Finish > Retry > Continue.
        5. If a reaction occurs, recursively dispatches the `ReactStep` event.
        6. Raises the winning reaction to be handled by the execution loop.

        Args:
            step: The event to dispatch.

        Yields:
            AgentEvents as they flow through the system.

        Raises:
            Reaction: If a hook dictates a change in control flow (Retry, Fail, Finish).
        """

        yield event

        logger.debug(
            f"Agent '{self.name}' ({self.trajectory.session_id}) dispatching '{type(event).__name__}': "
            f"hooks={[get_callable_name(h, short=True) for h in self.hooks]}"
        )

        # Run all applicable hooks and collect their reactions
        hook_reactions: dict[str, Reaction | None] = {}
        for hook in self.hooks:
            hook_name = getattr(hook, "__name__", getattr(hook, "__qualname__", safe_repr(hook)))

            reaction: Reaction | None = None
            try:
                reaction = hook(event)  # type: ignore[assignment]
                if inspect.isawaitable(reaction):
                    reaction = t.cast("Reaction", await reaction)
            except Reaction as r:
                reaction = r

            if reaction is None:
                continue

            logger.debug(
                f"Agent '{self.name}' ({self.trajectory.session_id}) hook '{hook_name}' returned reaction: {reaction!r}"
            )

            if not isinstance(reaction, Reaction):
                warn_at_user_stacklevel(
                    f"Hook '{hook_name}' returned {reaction}, but expected a Reaction.",
                    AgentWarning,
                )
                continue

            if isinstance(event, AgentEnd):
                warn_at_user_stacklevel(
                    f"Hook '{hook_name}' returned {reaction} during AgentEnd, but reactions are ignored at this stage.",
                    AgentWarning,
                )
                continue

            if isinstance(event, ReactStep):
                warn_at_user_stacklevel(
                    f"Hook '{hook_name}' returned {reaction} during Reacted, but reactions are ignored at this stage.",
                    AgentWarning,
                )
                continue

            hook_reactions[hook_name] = reaction

        if not hook_reactions:
            return

        # P1 - Termination
        winning_reaction: Reaction | None = next(
            (reaction for reaction in hook_reactions.values() if isinstance(reaction, Finish)),
            None,
        )

        # P2 - Retries
        winning_reaction = winning_reaction or next(
            (
                reaction
                for reaction in hook_reactions.values()
                if isinstance(reaction, Retry | RetryWithFeedback)
            ),
            None,
        )

        # P3 - Continues
        winning_reaction = winning_reaction or next(
            (reaction for reaction in hook_reactions.values() if isinstance(reaction, Continue)),
            None,
        )

        # Take the first reaction otherwise
        winning_reaction = winning_reaction or next(
            reaction for reaction in iter(hook_reactions.values()) if reaction is not None
        )

        # If we still don't have a winning reaction, return
        if winning_reaction is None:
            return

        # Warn for unused reactions
        for hook_name, reaction in hook_reactions.items():
            if reaction is not None and reaction is not winning_reaction:
                warn_at_user_stacklevel(
                    f"Hook '{hook_name}' returned {reaction}, but another hook already reacted. Only the first one will be applied.",
                    AgentWarning,
                )

        winning_hook_name = next(
            (name for name, reaction in hook_reactions.items() if reaction is winning_reaction),
            "unknown",
        )
        reacted_event = ReactStep(
            agent_id=self.agent_id,
            hook_name=winning_hook_name,
            reaction=winning_reaction,
        )

        async for _event in self._dispatch(reacted_event):
            yield _event

        if isinstance(winning_reaction, Continue):
            messages = winning_reaction.messages
            raise Continue(messages=messages) from winning_reaction

        if isinstance(winning_reaction, RetryWithFeedback):
            messages = winning_reaction.messages

            logger.debug(
                f"Agent '{self.name}' ({self.trajectory.session_id}) injecting feedback for retry: '{winning_reaction.feedback}'"
            )
            messages.extend(rg.Message("user", winning_reaction.feedback))
            raise Retry(messages=messages) from winning_reaction

        raise winning_reaction

    async def _process_tool_call(
        self, tool_call: "rg.tools.ToolCall", step_count: int
    ) -> t.AsyncGenerator[AgentEvent, None]:
        """
        Executes a specific tool call requested by the model.

        Manages the full lifecycle of the tool execution:
        - Emits `ToolStart`.
        - Locates the tool instance.
        - Executes the tool.
        - Emits `ToolEnd` (success) or `ToolError` (failure).
        - Emits `ToolStep` containing the resulting message.

        Args:
            tool_call: The tool call object requested by the LLM.

        Yields:
            AgentEvents related to the tool execution.
        """

        # emit ToolStart event
        async for event in self._dispatch(
            ToolStart(
                agent_id=self.agent_id,
                agent_name=self.name,
                status="running",
                tool_call=tool_call,
            )
        ):
            yield event

        message: rg.Message
        stop = False

        # find the tool
        tool = next((t for t in self.all_tools if t.name == tool_call.name), None)

        # tool not found
        if tool is None:
            error_msg = f"Tool '{tool_call.name}' not found."
            logger.warning(error_msg)
            async for event in self._dispatch(
                ToolError(
                    agent_id=self.agent_id,
                    agent_name=self.name,
                    tool_call=tool_call,
                    error=NameError(error_msg),
                )
            ):
                yield event

            message = rg.Message.from_model(
                rg.model.SystemErrorModel(content=f"Tool '{tool_call.name}' not found.")
            )

            # add the error to conversation
            async for event in self._dispatch(
                ToolStep(
                    agent_id=self.agent_id,
                    agent_name=self.name,
                    status="running",
                    step=step_count,
                    messages=[message],
                    error=None,
                    stop=stop,
                    tool_call=tool_call,
                )
            ):
                yield event
            return
        try:
            message, stop = await tool.handle_tool_call(tool_call)

            # emit ToolEnd event
            async for event in self._dispatch(
                ToolEnd(
                    agent_id=self.agent_id,
                    agent_name=self.name,
                    tool_call=tool_call,
                    result=message.content,
                )
            ):
                yield event

            # add output message
            async for event in self._dispatch(
                ToolStep(
                    agent_id=self.agent_id,
                    agent_name=self.name,
                    status="running",
                    step=step_count,
                    messages=[message],
                    error=None,
                    stop=stop,
                    tool_call=tool_call,
                )
            ):
                yield event

        except Exception as e:
            logger.opt(exception=True).error(f"Error executing tool '{tool_call.name}'")

            # there was an error executing the tool
            async for event in self._dispatch(
                ToolError(
                    agent_id=self.agent_id,
                    agent_name=self.name,
                    status="errored",
                    tool_call=tool_call,
                    error=e,
                )
            ):
                yield event
            raise

    async def _stream(  # noqa: PLR0912, PLR0915
        self,
        messages: list[rg.Message],
        *,
        inputs: dict[str, t.Any] | None = None,
        params: dict[str, t.Any] | None = None,
    ) -> t.AsyncGenerator[AgentEvent, None]:
        """
        The core execution loop of the agent.

        Iterates through the following cycle until `max_steps` is reached or a stop condition is met:
        1. Generates a response from the model.
        2. Appends the response to history.
        3. Checks stop conditions.
        4. Checks for stalling (no tool calls).
        5. Processes any tool calls found in the response.

        Args:
            messages: The initial conversation history.
            inputs: Dictionary of input values for tracing.
            params: Dictionary of configuration parameters for tracing.

        Yields:
            AgentEvents describing the progress of the run.
        """

        step_count = 0
        error: Exception | str | None = None

        # Agent start
        async for step in self._dispatch(
            AgentStart(
                agent_id=self.agent_id,
                inputs=inputs or {},
                params=params or {},
            )
        ):
            yield step

        # Core AgentStep loop
        while step_count < self.max_steps:
            step_count += 1

            try:
                # Generation
                step_chat = await self._generate(messages)
                if step_chat.failed and step_chat.error:
                    async for step in self._dispatch(
                        AgentError(
                            agent_id=self.agent_id,
                            status="errored",
                            error=t.cast("Exception", step_chat.error),
                        )
                    ):
                        yield step

                    error = t.cast("Exception", step_chat.error)  # Should be Exception in rigging
                    break

                # Sync extra fields to metadata for storage
                step_chat.generated[-1].metadata.update(step_chat.extra)

                messages.extend(step_chat.generated)

                async for step in self._dispatch(
                    GenerationStep(
                        agent_id=self.agent_id,
                        agent_name=self.name,
                        status="running",
                        generator=self._generator,
                        messages=messages,
                        step=step_count,
                        usage=step_chat.usage,
                    )
                ):
                    yield step

                # Check for stop conditions

                if any(cond(self.trajectory.steps) for cond in self.stop_conditions):
                    logger.info("A stop condition was met. Ending run.")
                    break

                # Check if stalled

                if not messages[-1].tool_calls:
                    if not self.stop_conditions:
                        break

                    logger.warning(
                        f"Agent '{self.name}' ({self.trajectory.session_id}) stalled: No tool calls and no stop conditions met."
                    )

                    async for step in self._dispatch(
                        AgentStalled(agent_id=self.agent_id, agent_name=self.name, status="stalled")
                    ):
                        yield step

                    # If the agent is stalled and nobody handled it, break out
                    break

                # Process tool calls

                stopped_by_tool_call: rg.tools.ToolCall | None = None

                tool_call_generators = [
                    self._process_tool_call(tc, step_count) for tc in messages[-1].tool_calls
                ]

                async for event in join_generators(*tool_call_generators):
                    # for every event yielded by _process_tool_call, we yield it up
                    yield event

                    if isinstance(event, ToolStep):
                        messages.extend(event.messages)
                        if stopped_by_tool_call is None and event.stop:
                            stopped_by_tool_call = event.tool_call

                if stopped_by_tool_call:
                    raise Finish(  # noqa: TRY301
                        f"Tool '{stopped_by_tool_call.name}' handling "
                        f"{stopped_by_tool_call.id} requested to stop the agent."
                    )

                # Check for stop conditions (again)
                if any(cond(self.trajectory.steps) for cond in self.stop_conditions):
                    break

            except Retry as e:
                messages = e.messages or messages
                continue
            except Fail as e:
                error = e.error
                break
            except Finish:
                break
            except Exception as e:  # Catch tool errors re-raised from _process_tool_call
                error = e
                break

        stop_reason: AgentStopReason = "finished"
        if step_count >= self.max_steps:
            error = MaxStepsError(max_steps=self.max_steps)
            stop_reason = "max_steps_reached"
        elif error is not None:
            stop_reason = "error"
        elif self.trajectory.steps and isinstance(self.trajectory.steps[-1], AgentStalled):
            stop_reason = "stalled"

        async for step in self._dispatch(
            AgentEnd(
                agent_id=self.agent_id,
                agent_name=self.name,
                status="errored" if error else "finished",
                stop_reason=stop_reason,
                error=error,
            )
        ):
            yield step

    async def _stream_traced(
        self,
        user_input: str,
    ) -> t.AsyncGenerator[AgentEvent, None]:
        """
        Wraps the `_stream` method with observability and logging context.

        Prepares configuration models, injects system prompts, and sets up
        Dreadnode tracing spans before delegating to the main execution loop.
        Logs summary statistics upon completion.

        Args:
            user_input: The primary input string from the user.

        Yields:
            AgentEvents from the execution loop.
        """
        from dreadnode import log_output, task_and_run

        messages = [*deepcopy(self.trajectory.messages), rg.Message("user", str(user_input))]

        messages = inject_system_content(messages, self.get_prompt())

        configuration = get_config_model(self)()
        trace_inputs, trace_params = get_inputs_and_params_from_config_model(configuration)

        trace_inputs.update(
            {
                "hooks": [get_callable_name(hook, short=True) for hook in self.hooks],
                "stop_conditions": [s.name for s in self.stop_conditions],
                "tools": [t.name for t in self.all_tools],
                "generator_params": self.generator.params.to_dict(),
                "user_input": user_input,
                "instructions": self.instructions,
                "messages": list(messages),
                "tool_schemas": [t.api_definition for t in self.all_tools],
            }
        )
        trace_params.update(
            {
                "name": self.name,
                "model": self.model_name,
                "max_steps": self.max_steps,
                "tool_mode": self.tool_mode,
                "tool_count": len(self.all_tools),
                "instructions_length": len(self.instructions or ""),
                "stop_condition_count": len(self.stop_conditions),
                "message_count": len(messages),
            }
        )
        trace_params.pop("instructions", None)

        for tag in self.tags:
            trace_params[f"tag:{tag}"] = True

        last_step: AgentStep | None = None
        with task_and_run(name=self.name, tags=self.tags, label=self.label):
            try:
                logger.info(
                    f"Starting Agent '{self.name}' ({self.trajectory.session_id}): "
                    f"model='{self.model_name}', "
                    f"max_steps={self.max_steps}, "
                    f"tools={[tool.name for tool in self.all_tools]}"
                )

                # process agent steps
                async with aclosing(
                    self._stream(messages, inputs=trace_inputs, params=trace_params)
                ) as stream:
                    async for step in stream:
                        last_step = step
                        if isinstance(step, AgentStep):
                            self.trajectory.log_step(last_step)

                        yield step
            finally:
                if self.trajectory is not None:
                    log_output("input_tokens", self.trajectory.usage.input_tokens)
                    log_output("output_token", self.trajectory.usage.output_tokens)
                    log_output("total_tokens", self.trajectory.usage.total_tokens)

    def get_prompt(self) -> str:
        """
        Generates the system prompt for the agent.

        Combines a base persona string with the configured `instructions`.

        Returns:
            The complete system prompt string.
        """

        prompt = "You are an agent that can use tools to assist with tasks."
        if self.instructions:
            prompt += f"\n\n<instructions>\n{self.instructions}\n</instructions>"
        return prompt

    def reset(self) -> Trajectory:
        """
        Resets the agent's internal state.

        Creates a fresh `Trajectory` for a new run.

        Returns:
            The `Trajectory` of the previous run.
        """
        previous = self.trajectory
        self.trajectory = Trajectory(agent_id=self.agent_id)
        return previous

    @asynccontextmanager
    async def stream(
        self,
        user_input: str,
    ) -> t.AsyncIterator[t.AsyncGenerator[AgentEvent, None]]:
        """
        Public context manager for streaming the agent's execution.

        Ensures that any tools requiring context management (via `__aenter__`)
        are properly initialized before execution begins.

        Args:
            user_input: The input string for the agent.

        Yields:
            An async generator yielding `AgentEvent` objects.
        """
        async with AsyncExitStack() as stack:
            # Ensure all tools are properly entered if they
            # are context managers before we start using them
            for tool_container in self.tools:
                if hasattr(tool_container, "__aenter__") and hasattr(tool_container, "__aexit__"):
                    await stack.enter_async_context(tool_container)

            async with aclosing(self._stream_traced(user_input)) as stream:
                yield stream

    async def run(
        self,
        user_input: str,
    ) -> Trajectory:
        """
        Executes the agent purely, consuming the stream until completion.

        Args:
            user_input: The input string for the agent.

        Returns:
            The final `Trajectory` object containing the run history and status.

        Raises:
            RuntimeError: If the run finishes without producing an AgentEnd event.
        """
        self.trajectory = Trajectory(agent_id=self.agent_id)

        final_event: AgentEvent | None = None

        async with self.stream(user_input) as stream:
            async for event in stream:
                final_event = event

        if not isinstance(final_event, AgentEnd):
            raise RuntimeError("Agent run finished unexpectedly.")  # noqa: TRY004

        return self.trajectory


class TaskAgent(Agent):
    """
    A specialized agent mixin for running tasks with a focus on completion and reporting.
    It extends the base Agent class to provide task-specific functionality.

    - Automatically includes the `finish_task`, `give_up_on_task`, and `update_todo` tools.
    - Installs a default stop_never condition to trigger stalling behavior when no tools calls are made.
    - Uses the `AgentStalled` event to handle stalled tasks by pushing the model to continue or finish the task.
    """

    def model_post_init(self, context: t.Any) -> None:
        super().model_post_init(context)

        # TODO(nick): Would be better to have a pattern here for
        # add-if-missing for tools, hooks, and stop conditions

        if not any(tool for tool in self.tools if tool.name == "finish_task"):
            self.tools.append(finish_task)

        if not any(tool for tool in self.tools if tool.name == "give_up_on_task"):
            self.tools.append(give_up_on_task)

        if not any(tool for tool in self.tools if tool.name == "update_todo"):
            self.tools.append(update_todo)

        # Force the agent to use finish_task
        if not any(cond for cond in self.stop_conditions if cond.name == "stop_never"):
            self.stop_conditions.append(never())

        if not any(
            hook
            for hook in self.hooks
            if get_callable_name(hook, short=True) == "retry_with_feedback"
        ):
            self.hooks.append(
                retry_with_feedback(
                    event_type=AgentStalled,
                    feedback="No tool calls were observed. Continue the task if possible, use the 'finish_task' tool to complete it, or 'give_up_on_task' if it cannot be completed.",
                )
            )


class RegexRefAgent(Agent):
    """
    An agent mixin that allows for dynamic references of prior text using regex patterns in tool arguments.
    This helps prevent repeating large amounts of prior text in tool calls.

    Instructions are automatically added to the agent's instructions to guide usage of the {find:<pattern>} syntax
    along with a hook that resolves these references during tool calls.
    """

    @staticmethod
    async def resolve_regex_ref(event: AgentStep) -> Reaction | None:
        if not isinstance(event, ToolStart):
            return None

        for m in re.finditer(r"\{find:(.*?)\}", event.tool_call.arguments):
            regex = m.group(1).replace("\\\\", "\\")  # models tend to over-escape
            logger.info(f"Found find reference: {regex}")
            all_message_content = "\n\n".join([m.content for m in event.messages])
            reference_matches = re.findall(regex, all_message_content)
            if reference_matches:
                logger.debug(f"Replacing '{m.group(0)}' with '{reference_matches[-1][:50]}...'.")
                event.tool_call.function.arguments = event.tool_call.arguments.replace(
                    m.group(0), json.dumps(reference_matches[-1]).strip('"')
                )

        return None

    def model_post_init(self, context: t.Any) -> None:
        super().model_post_init(context)

        if not any(
            hook
            for hook in self.hooks
            if get_callable_name(hook, short=True) == "resolve_regex_ref"
        ):
            self.hooks.append(RegexRefAgent.resolve_regex_ref)

        instruction_section = dedent("""
        # Regex Find Instructions
        To efficiently reuse data from the conversation, you can pass {find:<pattern>} anywhere in tool arguments to dynamically
        refer to prior text using a regex pattern. This helps prevent costly repetition of prior text.

        You must escape special characters in the regex.

        Example: If the history contains `$krb5tgs$23$*user...<long_hash>`, use:
        `hashcat(hashes=["{find:\\$krb5tgs\\$.*}"], wordlist="...")`
        and the system will find the full hash for you and insert it into the tool call.
        """)

        if self.instructions is None:
            self.instructions = instruction_section
        elif self.instructions and instruction_section not in self.instructions:
            self.instructions += "\n\n" + instruction_section

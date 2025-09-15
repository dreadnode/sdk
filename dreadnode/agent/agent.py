import inspect
import typing as t
from contextlib import aclosing, asynccontextmanager
from copy import deepcopy

import rigging as rg
from pydantic import ConfigDict, Field, PrivateAttr, field_validator
from rigging.message import inject_system_content  # can't access via rg

from dreadnode.agent.error import MaxStepsError
from dreadnode.agent.events import (
    AgentEnd,
    AgentError,
    AgentEvent,
    AgentEventInStep,
    AgentStalled,
    AgentStart,
    AgentStopReason,
    GenerationEnd,
    Reacted,
    StepStart,
    ToolEnd,
    ToolStart,
    _total_usage_from_events,
)
from dreadnode.agent.hooks import retry_with_feedback
from dreadnode.agent.reactions import (
    Continue,
    Fail,
    Finish,
    Hook,
    Reaction,
    Retry,
    RetryWithFeedback,
)
from dreadnode.agent.result import AgentResult
from dreadnode.agent.stop import StopCondition, never
from dreadnode.agent.thread import Thread
from dreadnode.agent.tools import AnyTool, Tool, Toolset, discover_tools_on_obj
from dreadnode.agent.tools.base import ToolMode
from dreadnode.agent.tools.planning import update_todo
from dreadnode.agent.tools.tasking import finish_task, give_up_on_task
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

CommitBehavior = t.Literal["always", "on-success"]
HookMap = dict[type[AgentEvent], list[Hook]]


class AgentWarning(UserWarning):
    """Warning raised when an agent is used in a way that may not be safe or intended."""


class Agent(Model):
    """
    Agent abstraction for applying tools, event logic, and message state to LLM generation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    name: str
    """The name of the agent."""
    description: str = ""
    """A brief description of the agent's purpose."""
    tags: list[str] = Config(default_factory=lambda: ["agent"])
    """A list of tags associated with the agent."""

    model: str | None = Config(default=None)
    """Inference model (rigging generator identifier)."""
    instructions: str | None = Config(default=None)
    """The agent's core instructions."""
    max_steps: int = Config(default=10)
    """The maximum number of steps (generation + tool calls)."""
    caching: rg.caching.CacheMode | None = Config(default=None, repr=False)
    """How to handle cache_control entries on inference messages."""

    tools: list[AnyTool | Toolset] = Config(default_factory=list)
    """Tools the agent can use."""
    tool_mode: ToolMode = Config(default="auto", repr=False)
    """The tool calling mode to use."""

    hooks: list[Hook] = Config(default_factory=list, exclude=True, repr=False)
    """Hooks to run at various points in the agent's lifecycle."""
    stop_conditions: list[StopCondition] = Config(default_factory=list)
    """The logical condition for successfully stopping a run."""
    thread: Thread = Field(default_factory=Thread, exclude=True, repr=False)
    """Stateful thread for this agent, for when otherwise not specified during execution."""
    scorers: ScorersLike[AgentResult] = Config(default_factory=list)
    """Scorers to evaluate the agent output."""
    assert_scores: list[str] | t.Literal[True] = Config(default_factory=list)
    """Scores to ensure are truthy, otherwise the agent task is marked as failed."""

    _generator: rg.Generator | None = PrivateAttr(None, init=False)

    @field_validator("tools", mode="before")
    @classmethod
    def validate_tools(cls, value: t.Any) -> t.Any:
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

    def _get_transforms(self) -> list[rg.Transform]:
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

        return transforms

    def _get_hooks(self) -> dict[type[AgentEvent], list[Hook]]:
        hooks: dict[type[AgentEvent], list[Hook]] = {}
        for hook in self.hooks:
            sig = inspect.signature(hook)
            if not (params := list(sig.parameters.values())):
                continue
            event_type = params[0].annotation

            if hasattr(event_type, "__origin__") and event_type.__origin__ is t.Union:
                union_args = event_type.__args__
                for arg in union_args:
                    if inspect.isclass(arg) and issubclass(arg, AgentEvent):
                        hooks.setdefault(arg, []).append(hook)
            elif inspect.isclass(event_type) and issubclass(event_type, AgentEvent):
                hooks.setdefault(event_type, []).append(hook)
            else:
                hooks.setdefault(AgentEvent, []).append(hook)

        return hooks

    async def _generate(
        self,
        messages: list[rg.Message],
        *,
        model: str | rg.Generator | None = None,
    ) -> rg.Chat:
        if model is None and self.model is None:
            raise ValueError("No model specified for agent generation.")

        # Build our default generator if we haven't already
        if model is None and self._generator is None:
            self._generator = rg.get_generator(self.model)

        generator = self._generator

        # Override if the user supplied one
        if model is not None:
            if isinstance(model, str):
                self._generator = rg.get_generator(model)
            elif isinstance(model, rg.Generator):
                self._generator = model

        if generator is None:
            raise TypeError("Model must be a string or a Generator instance.")

        messages = list(messages)  # Ensure we have a mutable list
        params = rg.GenerateParams(
            tools=[tool.api_definition for tool in self.all_tools],
        )
        messages = inject_system_content(messages, self.get_prompt())

        if self.tool_mode == "auto" and self.tools:
            self.tool_mode = "api" if await generator.supports_function_calling() else "json-in-xml"

        transforms = self._get_transforms()
        post_transforms: list[rg.PostTransform | None] = []
        for transform_callback in transforms:
            messages, params, post_transform = await transform_callback(messages, params)
            post_transforms.append(post_transform)

        try:
            messages = rg.caching.apply_cache_mode_to_messages(self.caching, [messages])[0]

            generated = (await generator.generate_messages([messages], [params]))[0]
            if isinstance(generated, BaseException):
                raise generated  # noqa: TRY301

            chat = rg.Chat(
                messages,
                [generated.message],
                generator=generator,
                params=params,
                stop_reason=generated.stop_reason,
                usage=generated.usage,
                extra=generated.extra,
            )

        except Exception as error:  # noqa: BLE001
            chat = rg.Chat(
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

    async def _stream(  # noqa: PLR0912, PLR0915
        self,
        thread: "Thread",
        messages: list[rg.Message],
        hooks: HookMap,
        *,
        commit: CommitBehavior,
    ) -> t.AsyncGenerator[AgentEvent, None]:
        events: list[AgentEvent] = []
        stop_conditions = self.stop_conditions

        # Event dispatcher

        async def _dispatch(event: AgentEvent) -> t.AsyncIterator[AgentEvent]:
            nonlocal messages, events

            yield event

            events.append(event)

            # If we have no hooks, just return the event
            applicable_hooks = list(set(hooks.get(type(event), []) + hooks.get(AgentEvent, [])))
            if not applicable_hooks:
                return

            # Run all applicable hooks and collect their reactions
            hook_reactions: dict[str, Reaction | None] = {}
            for hook in applicable_hooks:
                hook_name = getattr(
                    hook, "__name__", getattr(hook, "__qualname__", safe_repr(hook))
                )

                reaction: Reaction | None = None
                try:
                    reaction = hook(event)  # type: ignore[assignment]
                    if inspect.isawaitable(reaction):
                        reaction = t.cast("Reaction", await reaction)
                except Reaction as r:
                    reaction = r

                if reaction is None:
                    continue

                if not isinstance(reaction, Reaction):
                    warn_at_user_stacklevel(
                        f"Hook '{hook_name}' returned {reaction}, but expected a Reaction.",
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
                (
                    reaction
                    for reaction in hook_reactions.values()
                    if isinstance(reaction, Continue)
                ),
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
            reacted_event = Reacted(
                agent=self,
                thread=thread,
                messages=messages,
                events=events,
                hook_name=winning_hook_name,
                reaction=winning_reaction,
            )
            events.append(reacted_event)
            yield reacted_event

            if isinstance(winning_reaction, Continue):
                messages = winning_reaction.messages
                return

            if isinstance(winning_reaction, RetryWithFeedback):
                messages.append(rg.Message("user", winning_reaction.feedback))
                raise Retry(messages=messages) from winning_reaction

            raise winning_reaction

        # Tool calling

        async def _process_tool_call(
            tool_call: "rg.tools.ToolCall",
        ) -> t.AsyncGenerator[AgentEvent, None]:
            async for event in _dispatch(
                ToolStart(
                    agent=self,
                    thread=thread,
                    messages=messages,
                    events=events,
                    tool_call=tool_call,
                )
            ):
                yield event

            message: rg.Message
            stop = False
            tool = next((t for t in self.all_tools if t.name == tool_call.name), None)

            if tool is not None:
                try:
                    message, stop = await tool.handle_tool_call(tool_call)
                except Reaction:
                    raise
                except Exception as e:
                    async for event in _dispatch(
                        AgentError(
                            agent=self,
                            thread=thread,
                            messages=messages,
                            events=events,
                            error=e,
                        )
                    ):
                        yield event
                    raise
            else:
                message = rg.Message.from_model(
                    rg.model.SystemErrorModel(content=f"Tool '{tool_call.name}' not found.")
                )

            async for event in _dispatch(
                ToolEnd(
                    agent=self,
                    thread=thread,
                    messages=messages,
                    events=events,
                    tool_call=tool_call,
                    message=message,
                    stop=stop,
                )
            ):
                yield event

        # Agent start

        async for event in _dispatch(
            AgentStart(
                agent=self,
                thread=thread,
                messages=messages,
                events=events,
            )
        ):
            yield event

        # Core step loop

        step = 1
        error: Exception | str | None = None

        while step <= self.max_steps + 1:
            try:
                async for event in _dispatch(
                    StepStart(
                        agent=self,
                        thread=thread,
                        messages=messages,
                        events=events,
                        step=step,
                    )
                ):
                    yield event

                # Generation

                step_chat = await self._generate(messages=messages)
                if step_chat.failed and step_chat.error:
                    async for event in _dispatch(
                        AgentError(
                            agent=self,
                            thread=thread,
                            messages=messages,
                            events=events,
                            error=t.cast("Exception", step_chat.error),
                        )
                    ):
                        yield event
                    raise step_chat.error

                messages.extend(step_chat.generated)

                async for event in _dispatch(
                    GenerationEnd(
                        agent=self,
                        thread=thread,
                        messages=messages,
                        events=events,
                        message=step_chat.last,
                        usage=step_chat.usage,
                    )
                ):
                    yield event

                # Check for stop conditions

                if any(cond(events) for cond in stop_conditions):
                    break

                # Check if stalled

                if not messages[-1].tool_calls:
                    if not stop_conditions:
                        break

                    async for event in _dispatch(
                        AgentStalled(
                            agent=self,
                            thread=thread,
                            messages=messages,
                            events=events,
                        )
                    ):
                        yield event

                    continue

                # Process tool calls

                stopped_by_tool_call: rg.tools.ToolCall | None = None

                async for event in join_generators(
                    *[_process_tool_call(tool_call) for tool_call in messages[-1].tool_calls]
                ):
                    if isinstance(event, ToolEnd):
                        messages.append(event.message)
                        if stopped_by_tool_call is None and event.stop:
                            stopped_by_tool_call = event.tool_call
                    yield event

                if stopped_by_tool_call:
                    raise Finish(  # noqa: TRY301
                        f"Tool '{stopped_by_tool_call.name}' handling "
                        f"{stopped_by_tool_call.id} requested to stop the agent."
                    )

                # Check for stop conditions (again)

                if any(cond(events) for cond in stop_conditions):
                    break

                step += 1

            except Retry as e:
                messages = e.messages or messages
                continue

            except Fail as e:
                error = e.error
                break

            except Finish:
                break

        stop_reason: AgentStopReason = "finished"
        if step > self.max_steps + 1:
            error = MaxStepsError(max_steps=self.max_steps)
            stop_reason = "max_steps_reached"
        elif error is not None:
            stop_reason = "error"
        elif events and isinstance(events[-1], AgentStalled):
            stop_reason = "stalled"

        # Commit messages back to the thread

        if commit == "always" or (commit == "on-success" and not error):
            thread.messages = messages
            thread.events.extend(events)

        yield AgentEnd(
            agent=self,
            thread=thread,
            messages=messages,
            events=events,
            stop_reason=stop_reason,
            result=AgentResult(
                agent=self,
                messages=messages,
                usage=_total_usage_from_events(events),
                steps=step,
                failed=stop_reason != "finished",
                error=error,
            ),
        )

    def _log_event_metrics(self, event: AgentEvent) -> None:
        from dreadnode import log_metric

        if isinstance(event, AgentEnd):
            log_metric("steps_taken", min(0, event.result.steps - 1))
            log_metric(f"stop_{event.stop_reason}", 1)

        if not isinstance(event, AgentEventInStep):
            return

        if isinstance(event, GenerationEnd) and event.usage:
            log_metric("generations", 1, step=event.step, mode="count")
            log_metric("messages", len(event.messages) + 1, step=event.step)
            log_metric("in_tokens", event.usage.input_tokens)
            log_metric("out_tokens", event.usage.output_tokens)
            log_metric("tokens", event.usage.total_tokens)
        elif isinstance(event, ToolStart):
            log_metric("tool_calls", 1, step=event.step, mode="count")
        elif isinstance(event, AgentError):
            log_metric("errors", 1, step=event.step, mode="count")
        elif isinstance(event, AgentStalled):
            log_metric("stalled", 1, step=event.step, mode="count")
        elif isinstance(event, Reacted):
            log_metric("reactions", 1, step=event.step, mode="count")
            if isinstance(event.reaction, Retry):
                log_metric("retries", 1, step=event.step, mode="count")
                if event.reaction.messages:
                    log_metric("messages", len(event.reaction.messages), step=event.step)
            if isinstance(event.reaction, Continue):
                log_metric("continues", 1, step=event.step, mode="count")
                log_metric("messages", len(event.messages), step=event.step)

    async def _stream_traced(
        self,
        thread: "Thread",
        user_input: str,
        *,
        commit: CommitBehavior = "on-success",
    ) -> t.AsyncGenerator[AgentEvent, None]:
        from dreadnode import log_output, log_outputs, score, task_and_run

        hooks = self._get_hooks()
        messages = [*deepcopy(thread.messages), rg.Message("user", str(user_input))]

        configuration = get_config_model(self)()
        trace_inputs, trace_params = get_inputs_and_params_from_config_model(configuration)

        trace_inputs.update(
            {
                "user_input": user_input,
                "messages": messages,
                "instructions": self.instructions,
                "tools": [t.name for t in self.all_tools],
                "hooks": [get_callable_name(hook, short=True) for hook in self.hooks],
                "stop_conditions": [s.name for s in self.stop_conditions],
            }
        )
        trace_params.update(
            {
                "model": self.model,
                "max_steps": self.max_steps,
                "tool_mode": self.tool_mode,
            }
        )

        last_event: AgentEvent | None = None
        with task_and_run(name=self.name, tags=self.tags, inputs=trace_inputs, params=trace_params):
            try:
                async with aclosing(self._stream(thread, messages, hooks, commit=commit)) as stream:
                    async for event in stream:
                        last_event = event
                        self._log_event_metrics(event)
                        yield event
            finally:
                if last_event is not None:
                    log_outputs(messages=last_event.messages, token_usage=last_event.total_usage)

                if isinstance(last_event, AgentEnd):
                    log_outputs(
                        to="both",
                        steps_taken=min(0, last_event.result.steps - 1),
                        reason=last_event.stop_reason,
                    )
                    if last_event.result.error:
                        log_output("error", last_event.result.error, to="both")
                    await score(
                        last_event.result,
                        self.scorers,
                        assert_scores=self.assert_scores,
                    )

    def get_prompt(self) -> str:
        """
        Generates the prompt for the agent based on its instructions.
        This can be overridden by subclasses to provide custom behavior.
        """
        prompt = "You are an agent that can use tools to assist with tasks."
        if self.instructions:
            prompt += f"\n\n<instructions>\n{self.instructions}\n</instructions>"
        return prompt

    def reset(self) -> Thread:
        """Reset the agent's internal thread and returns the previous thread."""
        previous = self.thread
        self.thread = Thread()
        return previous

    @asynccontextmanager
    async def stream(
        self,
        user_input: str,
        *,
        thread: Thread | None = None,
        commit: CommitBehavior = "always",
    ) -> t.AsyncIterator[t.AsyncGenerator[AgentEvent, None]]:
        thread = thread or self.thread
        async with aclosing(self._stream_traced(thread, user_input, commit=commit)) as stream:
            yield stream

    async def run(
        self,
        user_input: str,
        *,
        thread: Thread | None = None,
        commit: CommitBehavior = "always",
    ) -> AgentResult:
        final_event: AgentEvent | None = None
        async with self.stream(user_input, thread=thread, commit=commit) as stream:
            async for event in stream:
                final_event = event

        if not isinstance(final_event, AgentEnd):
            raise RuntimeError("Agent run finished unexpectedly.")  # noqa: TRY004

        return final_event.result


class TaskAgent(Agent):
    """
    A specialized agent for running tasks with a focus on completion and reporting.
    It extends the base Agent class to provide task-specific functionality.

    - Automatically includes the `finish_task`, `give_up_on_task`, and `update_todo` tools.
    - Installs a default stop_never condition to trigger stalling behavior when no tools calls are made.
    - Uses the `AgentStalled` event to handle stalled tasks by pushing the model to continue or finish the task.
    """

    def model_post_init(self, _: t.Any) -> None:
        if not any(tool for tool in self.tools if tool.name == "finish_task"):
            self.tools.append(finish_task)

        if not any(tool for tool in self.tools if tool.name == "give_up_on_task"):
            self.tools.append(give_up_on_task)

        if not any(tool for tool in self.tools if tool.name == "update_todo"):
            self.tools.append(update_todo)

        # Force the agent to use finish_task
        self.stop_conditions.append(never())
        self.hooks.insert(
            0,
            retry_with_feedback(
                event_type=AgentStalled,
                feedback="Continue the task if possible, use the 'finish_task' tool to complete it, or 'give_up_on_task' if it cannot be completed.",
            ),
        )

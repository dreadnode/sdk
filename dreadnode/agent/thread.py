import asyncio
import inspect
import typing as t
from contextlib import aclosing, asynccontextmanager
from copy import deepcopy

from pydantic import BaseModel, Field
from rigging.generator import Usage
from rigging.message import Message
from rigging.model import SystemErrorModel

from dreadnode.agent.agent import Agent
from dreadnode.agent.error import AgentStop
from dreadnode.agent.events import (
    AgentEvent,
    AgentEventT,
    AgentResult,
    AgentRunEnd,
    AgentRunError,
    AgentStalled,
    AgentStart,
    GenerationEnd,
    StepStart,
)
from dreadnode.agent.hooks import Hook, HookAction

if t.TYPE_CHECKING:
    from rigging.tools import ToolCall

HookMap = dict[type[AgentEvent], list[Hook]]
from dreadnode.util import warn_at_user_stacklevel

class Thread(BaseModel):
    messages: list[Message] = Field(default_factory=list)
    """The log of messages exchanged during the session."""
    events: list[AgentEvent] = Field(default_factory=list)
    """All events that have occurred during the session, including errors."""

    @property
    def total_usage(self) -> Usage:
        """Aggregates the usage from all events in the session."""
        total = Usage(input_tokens=0, output_tokens=0, total_tokens=0)
        for event in self.events:
            if isinstance(event, GenerationEnd) and event.usage:
                total += event.usage
        return total

    @property
    def last_usage(self) -> Usage | None:
        """Returns the usage from the last event, if available."""
        if not self.events:
            return None
        last_event = self.events[-1]
        if isinstance(last_event, GenerationEnd):
            return last_event.usage
        return None

    async def _stream(
        self, agent: Agent, user_input: str, hooks: HookMap
    ) -> t.AsyncIterator[AgentEvent]:
        events: list[AgentEvent] = []

        async def _dispatch(event: AgentEventT) -> HookAction | None:
            applicable_hooks = hooks.get(type(event), [])
            if not applicable_hooks:
                return None

            actions: list[HookAction] = []
            for hook in applicable_hooks:
                if (result := await hook(event)) is not None:
                    actions.append(result)

            history_modified = False
            control_flow_changed = False

            for action in actions:
                if isinstance(action, ModifyHistory):
                    if history_modified:
                        warn_at_user_stacklevel(
                            "Multiple hooks attempted to modify the history. Only the first modification will be applied."
                        )
                        continue

                    run_history = action.new_history
                    history_modified = True
                    yield await self._dispatch(ActionApplied(
                        ..., source_hook_name=action.source_hook_name, action=action
                    ))

                elif isinstance(action, RequestRetry) and not control_flow_changed:
                    control_flow_changed = True
                    # We don't yield here, we raise to change the loop's flow.
                    # The exception itself signals the action was taken.
                    raise ModelRetry(action.feedback)

                elif isinstance(action, TerminateRun) and not control_flow_changed:
                    control_flow_changed = True
                    raise AgentStop(action.result)

                        events.append(event)
                        return event

        # Agent start

        messages = deepcopy(self.messages)

        agent_start = await _dispatch(
            AgentStart(
                agent=agent,
                thread=self,
                messages=messages,
                events=events,
                user_input=user_input,
            )
        )
        yield agent_start
        user_input = agent_start.user_input
        messages = agent_start.messages

        messages.append(Message("user", user_input))

        # Step loop

        try:
            for step in range(1, agent.max_steps + 1):
                step_start = await _dispatch(
                    StepStart(
                        agent=agent,
                        thread=self,
                        messages=messages,
                        events=events,
                        step=step,
                    )
                )
                yield step_start
                messages = step_start.messages

                # Generation

                step_chat = await agent.generate(messages=messages)
                if step_chat.failed and step_chat.error:
                    raise step_chat.error

                generated_message = step_chat.last

                actions = yield from _dispatch(
                    GenerationEnd(
                        agent=agent,
                        thread=self,
                        messages=messages,
                        events=events,
                        usage=step_chat.usage,
                    )
                )
                yield generation_end
                generated_message = generation_end.message

                # Tool calls

                if not generated_message.tool_calls:
                    # Stalling detected
                    correction = "You have not completed the task correctly. The success condition has not been met. You MUST continue working or call `fail_task` if you are stuck."
                    run_history.append(Message("system", correction))
                    event = await self._dispatch(
                        AgentStalled(
                            agent=agent,
                            session=self,
                            correction_prompt=correction,
                            run_history=run_history,
                        )
                    )
                    yield event
                    run_history = event.run_history
                    continue  # Proceed to the next turn

                async def _process_tool_call(tool_call: "ToolCall") -> bool:
                    nonlocal messages

                    if (
                        tool := next((t for t in agent.tools if t.name == tool_call.name), None)
                    ) is None:
                        messages.append(
                            Message.from_model(
                                SystemErrorModel(content=f"Tool '{tool_call.name}' not found.")
                            )
                        )
                        return False

                    message, stop = await tool.handle_tool_call(tool_call)
                    messages.append(message)
                    return stop

                # Process all tool calls in parallel and check for stops

                if max(
                    await asyncio.gather(
                        *[_process_tool_call(tool_call) for tool_call in generated_message.tool_calls],
                    ),
                ):
                    raise AgentStop("Tool call triggered agent stop.")

        except AgentStop as e:


        output = final_message_for_turn
        if agent.output_model:
            output = final_message_for_turn.parse(agent.output_model)

        result = AgentResult(output, run_history, aggregated_usage, turn_number)
        yield await self._dispatch(
            AgentRunEnd(agent=agent, session=self, result=result, run_history=run_history)
        )
        return


        raise MaxStepsReached(agent.max_steps, "Maximum agent steps reached.")

    def _get_hooks(self, agent: Agent) -> dict[type[AgentEvent], list[Hook]]:
        hooks: dict[type[AgentEvent], list[Hook]] = {}
        for hook in agent.hooks:
            sig = inspect.signature(hook)
            if not (params := list(sig.parameters.values())):
                continue
            event_type = params[0].annotation
            if inspect.isclass(event_type) and issubclass(event_type, AgentEvent):
                hooks.setdefault(event_type, []).append(hook)
            else:
                hooks.setdefault(AgentEvent, []).append(hook)
        return hooks

    @asynccontextmanager
    async def stream(self, agent: Agent, user_input: str) -> t.AsyncIterator[AgentEvent]:
        """The user-facing context manager for stepping through a run."""

        hooks = self._get_hooks(agent)

        try:
            async with aclosing(self._stream(agent, user_input)) as stream:
                yield stream
        except Exception as e:
            error_event = AgentRunError(agent=agent, session=self, error=e, run_history=run_history)
            await self._dispatch(error_event)
            raise e

    async def run(self, agent: Agent, user_input: str) -> AgentResult:
        """Executes a full, observable run in this session."""
        final_event: AgentEvent | None = None
        run_history = []
        try:
            async with self.stream(agent, user_input) as stream:
                async for event in stream:
                    final_event = event
                    run_history = event.run_history

            if isinstance(final_event, AgentRunEnd):
                self.history = final_event.result.history
                return final_event.result

            raise RuntimeError("Agent run finished unexpectedly.")
        except Exception as e:
            error_event = AgentRunError(agent=agent, session=self, error=e, run_history=run_history)
            await self._dispatch(error_event)
            raise e

    def fork(self) -> "Thread":
        return Thread(history=deepcopy(self.history), events=deepcopy(self.events))

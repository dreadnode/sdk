import inspect
import typing as t
from contextlib import aclosing, asynccontextmanager
from copy import deepcopy

from pydantic import BaseModel, Field
from rigging.generator import Usage
from rigging.message import Message
from rigging.model import SystemErrorModel

from dreadnode.agent.error import MaxStepsError
from dreadnode.agent.events import (
    AgentEnd,
    AgentError,
    AgentStalled,
    AgentStart,
    Event,
    EventT,
    GenerationEnd,
    Reacted,
    StepStart,
    ToolEnd,
    ToolStart,
)
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
from dreadnode.util import join_generators, safe_repr, warn_at_user_stacklevel

if t.TYPE_CHECKING:
    from rigging.tools import ToolCall

    from dreadnode.agent.agent import Agent

CommitBehavior = t.Literal["always", "on-success"]
HookMap = dict[type[Event], list[Hook]]


class ThreadWarning(UserWarning):
    """A warning that is raised when a thread is used in a way that may not be safe or intended."""


def _total_usage_from_events(events: list[Event]) -> Usage:
    """Calculates the total usage from a list of events."""
    total = Usage(input_tokens=0, output_tokens=0, total_tokens=0)
    for event in events:
        if isinstance(event, GenerationEnd) and event.usage:
            total += event.usage
    return total


class Thread(BaseModel):
    messages: list[Message] = Field(default_factory=list)
    """The log of messages exchanged during the session."""
    events: list[Event] = Field(default_factory=list)
    """All events that have occurred during the session, including errors."""

    def __repr__(self) -> str:
        if not self.messages and not self.events:
            return "Thread()"
        return f"Thread(messages={len(self.messages)}, events={len(self.events)}, last_event={self.events[-1] if self.events else 'None'})"

    @property
    def total_usage(self) -> Usage:
        """Aggregates the usage from all events in the session."""
        return _total_usage_from_events(self.events)

    @property
    def last_usage(self) -> Usage | None:
        """Returns the usage from the last generation event, if available."""
        if not self.events:
            return None
        last_event = self.events[-1]
        if isinstance(last_event, GenerationEnd):
            return last_event.usage
        return None

    async def _stream(  # noqa: PLR0912, PLR0915
        self, agent: "Agent", message: Message, hooks: HookMap, *, commit: CommitBehavior
    ) -> t.AsyncGenerator[Event, None]:
        events: list[Event] = []
        messages = [*deepcopy(self.messages), message]
        stop_conditions = agent.stop_conditions

        # Event dispatcher

        async def _dispatch(event: EventT) -> t.AsyncIterator[Event]:
            nonlocal messages, events

            yield event

            events.append(event)

            # If we have no hooks, just return the event
            applicable_hooks = list(set(hooks.get(type(event), []) + hooks.get(Event, [])))
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
                        ThreadWarning,
                    )
                    continue

                hook_reactions[hook_name] = reaction

            # P1 - Termination
            winning_reaction: Reaction | None = next(
                (reaction for reaction in hook_reactions.values() if isinstance(reaction, Finish)),
                None,
            )

            # P2 - Retries
            winning_reaction = winning_reaction or next(
                (reaction for reaction in hook_reactions.values() if isinstance(reaction, Retry)),
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

            if winning_reaction is None:
                return

            # Warn for unused reactions
            for hook_name, reaction in hook_reactions.items():
                if reaction is not None and reaction is not winning_reaction:
                    warn_at_user_stacklevel(
                        f"Hook '{hook_name}' returned {reaction}, but another hook already reacted. Only the first one will be applied.",
                        ThreadWarning,
                    )

            winning_hook_name = next(
                (name for name, reaction in hook_reactions.items() if reaction is winning_reaction),
                "unknown",
            )
            reacted_event = Reacted(
                agent=agent,
                thread=self,
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
                messages.append(Message("user", winning_reaction.feedback))
                raise Retry(messages=messages) from winning_reaction

            raise winning_reaction

        # Tool calling

        async def _process_tool_call(
            tool_call: "ToolCall",
        ) -> t.AsyncGenerator[Event, None]:
            async for event in _dispatch(
                ToolStart(
                    agent=agent,
                    thread=self,
                    messages=messages,
                    events=events,
                    tool_call=tool_call,
                )
            ):
                yield event

            message: Message
            stop = False
            tool = next((t for t in agent.tools if t.name == tool_call.name), None)

            if tool is not None:
                try:
                    message, stop = await tool.handle_tool_call(tool_call)
                except Exception as e:
                    async for event in _dispatch(
                        AgentError(
                            agent=agent,
                            thread=self,
                            messages=messages,
                            events=events,
                            error=e,
                        )
                    ):
                        yield event
                    raise
            else:
                message = Message.from_model(
                    SystemErrorModel(content=f"Tool '{tool_call.name}' not found.")
                )

            async for event in _dispatch(
                ToolEnd(
                    agent=agent,
                    thread=self,
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
                agent=agent,
                thread=self,
                messages=messages,
                events=events,
            )
        ):
            yield event

        # Core step loop

        step = 1
        error: Exception | str | None = None

        while step <= agent.max_steps + 1:
            try:
                # Start a new step

                async for event in _dispatch(
                    StepStart(
                        agent=agent,
                        thread=self,
                        messages=messages,
                        events=events,
                        step=step,
                    )
                ):
                    yield event

                # Generation

                step_chat = await agent.generate(messages=messages)
                if step_chat.failed and step_chat.error:
                    async for event in _dispatch(
                        AgentError(
                            agent=agent,
                            thread=self,
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
                        agent=agent,
                        thread=self,
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
                            agent=agent,
                            thread=self,
                            messages=messages,
                            events=events,
                        )
                    ):
                        yield event

                    messages.append(Message("user", "continue"))
                    continue

                # Process tool calls

                stopped_by_tool_call: ToolCall | None = None

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

        if step > agent.max_steps + 1:
            error = MaxStepsError(max_steps=agent.max_steps)

        if commit == "always" or (commit == "on-success" and not error):
            self.messages = messages
            self.events.extend(events)

        yield AgentEnd(
            agent=agent,
            thread=self,
            messages=messages,
            events=events,
            result=AgentResult(
                agent=agent,
                messages=messages,
                usage=_total_usage_from_events(events),
                steps=step - 1,
                failed=bool(error),
                error=error,
            ),
        )

    def _get_hooks(self, agent: "Agent") -> dict[type[Event], list[Hook]]:
        hooks: dict[type[Event], list[Hook]] = {}
        for hook in agent.hooks:
            sig = inspect.signature(hook)
            if not (params := list(sig.parameters.values())):
                continue
            event_type = params[0].annotation

            if hasattr(event_type, "__origin__") and event_type.__origin__ is t.Union:
                union_args = event_type.__args__
                for arg in union_args:
                    if inspect.isclass(arg) and issubclass(arg, Event):
                        hooks.setdefault(arg, []).append(hook)
            elif inspect.isclass(event_type) and issubclass(event_type, Event):
                hooks.setdefault(event_type, []).append(hook)
            else:
                hooks.setdefault(Event, []).append(hook)

        return hooks

    @asynccontextmanager
    async def stream(
        self, agent: "Agent", user_input: str, *, commit: CommitBehavior = "on-success"
    ) -> t.AsyncIterator[t.AsyncGenerator[Event, None]]:
        """The user-facing context manager for stepping through a run."""

        hooks = self._get_hooks(agent)
        message = Message("user", str(user_input))

        async with aclosing(self._stream(agent, message, hooks, commit=commit)) as stream:
            yield stream

    async def run(
        self, agent: "Agent", user_input: str, *, commit: CommitBehavior = "on-success"
    ) -> AgentResult:
        """Executes a full, observable run in this session."""
        final_event: Event | None = None
        async with self.stream(agent, user_input, commit=commit) as stream:
            async for event in stream:
                final_event = event

            if not isinstance(final_event, AgentEnd):
                raise TypeError("Agent run finished unexpectedly.")

            return final_event.result

    def fork(self) -> "Thread":
        return Thread(messages=deepcopy(self.messages), events=deepcopy(self.events))

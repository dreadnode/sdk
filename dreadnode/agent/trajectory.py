import json
import typing as t
from datetime import datetime, timezone

import rigging as rg
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rigging import Generator
from rigging.generator import Usage
from ulid import ULID

from dreadnode.agent.format import format_message
from dreadnode.agent.reactions import (
    Continue,
    Fail,
    Finish,
    Reaction,
    RetryWithFeedback,
)
from dreadnode.util import format_dict, shorten_string

if t.TYPE_CHECKING:
    from dreadnode.common_types import AnyDict


AgentEventT = t.TypeVar("AgentEventT", bound="AgentEvent")
AgentStepT = t.TypeVar("AgentStepT", bound="AgentStep")
AgentStopReason = t.Literal["finished", "max_steps_reached", "error", "stalled"]
AgentStatus = t.Literal["running", "stalled", "errored", "finished"]


class AgentEvent(BaseModel):
    """
    A log event in the agent's lifecycle.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), kw_only=True)
    """The timestamp of when the event occurred (UTC)."""
    agent_id: ULID = Field(default_factory=ULID)
    """The name of the agent that generated this event."""
    agent_name: str | None = None
    """The name of the agent that generated this event."""
    status: AgentStatus | None = Field(default=None)
    """The status of the agent at the time of this event."""


class AgentStep(AgentEvent):
    """
    A discrete unit of work that advances the agent's state.

    A Step is an Event that contains messages that will be part of the
    ongoing chat history.

    Additionally, tracks step count, token usage, etc.
    """

    generator: Generator | None = None
    """The model or generator used by the agent during this step."""
    step: int = 0
    """The step number in the agent's execution when this event occurred."""
    messages: list[rg.Message] = Field(default_factory=list)
    """The messages generated or processed during this step."""
    usage: Usage = Usage(input_tokens=0, output_tokens=0, total_tokens=0)
    """The token usage associated with this step, if applicable."""
    error: Exception | None = None
    """An optional error that occurred during this step's execution."""
    stop: bool | None = None
    """Indicates if this step signals a stop condition for the agent."""

    @property
    def estimated_cost(self) -> float | None:
        """Estimates the cost of the agent run based on total token usage and model pricing."""
        import litellm

        if self.generator is None:
            return None

        model = self.generator.model
        while model not in litellm.model_cost:
            if "/" not in model:
                return None
            model = "/".join(model.split("/")[1:])

        model_info: AnyDict = litellm.model_cost[model]
        input_token_cost = float(model_info.get("input_cost_per_token", 0))
        output_token_cost = float(model_info.get("output_cost_per_token", 0))

        return (
            input_token_cost * self.usage.input_tokens
            + output_token_cost * self.usage.output_tokens
        )

    def __repr__(self) -> str:
        message_content = shorten_string(str(self.messages[0].content), 50)
        tool_call_count = len(self.messages[0].tool_calls) if self.messages[0].tool_calls else 0
        message = f"Message(role={self.messages[0].role}, content='{message_content}', tool_calls={tool_call_count})"
        return f"StepEnd(message={message})"

    def format_as_panel(self, *, truncate: bool = False) -> Panel:
        cost = round(self.estimated_cost, 6) if self.estimated_cost else ""
        usage = str(self.usage) or ""
        return Panel(
            format_message(self.messages[0], truncate=truncate),
            title="Step End",
            title_align="left",
            subtitle=f"[dim]{usage} [{cost} USD][/dim]",
            subtitle_align="right",
            padding=(1, 1),
        )

    def log_metrics(self, *, detailed: bool = False) -> None:
        """Logs metrics for tool usage if this step"""


class AgentStart(AgentEvent):
    """Event: The agent's execution process has started."""

    inputs: dict[str, t.Any] = Field(default_factory=dict)
    """The inputs provided to the agent at the start of execution."""
    params: dict[str, t.Any] = Field(default_factory=dict)
    """The parameters used to configure the agent at the start of execution."""


class AgentEnd(AgentEvent):
    """Event: The agent's execution process has finished."""

    stop_reason: AgentStopReason
    """The reason why the agent stopped, if applicable."""
    error: Exception | str | None = None
    """The error that caused the agent to stop, if applicable."""


class AgentStalled(AgentEvent):
    """Event: The agent is stalled and there are no tool calls, or stop condition)."""

    def format_as_panel(self, *, truncate: bool = False) -> Panel:  # noqa: ARG002
        return Panel(
            Text(
                "Agent has no tool calls to make and has not met a stop condition.",
                style="dim white",
            ),
            title="Agent Stalled",
            title_align="left",
            border_style="bright_black",
        )


class AgentError(AgentEvent):
    """Event: An error occurred, functionally halting the agent."""

    error: BaseException

    def format_as_panel(self, *, truncate: bool = False) -> Panel:  # noqa: ARG002
        return Panel(
            repr(self),
            title="Agent Error",
            title_align="left",
            border_style="red",
        )


class ToolStep(AgentStep):
    """A step representing the completion of a tool call by the agent."""

    tool_call: rg.tools.ToolCall
    """The tool call that was completed."""

    def __repr__(self) -> str:
        message_content = shorten_string(str(self.messages[0].content), 50)
        message = f"Message(role={self.messages[0].role}, content='{message_content}')"
        return f"ToolEnd(tool_call={self.tool_call}, message={message}, stop={self.stop})"

    def log_metrics(self, *, detailed: bool = False) -> None:
        from dreadnode import log_metric

        _start_times: dict[str, datetime] = {}

        tool_name = self.tool_call.name
        _start_times[self.tool_call.id] = self.timestamp
        start_time = _start_times.pop(self.tool_call.name, self.timestamp)
        duration_seconds = (self.timestamp - start_time).total_seconds()
        errored = "error" in self.error if self.error else False

        log_metric(f"tool/count.{tool_name}", 1, step=self.step, mode="count")
        log_metric("tool/total_time", duration_seconds, step=self.step, mode="sum")
        log_metric("tool/success_rate", 0 if errored else 1, step=self.step, mode="avg")

        if errored:
            log_metric("tool/failed_count", 1, step=self.step, mode="count")

        if detailed:
            log_metric(
                f"tool/time.{tool_name}",
                duration_seconds,
                step=self.step,
                mode="sum",
            )
            log_metric(
                f"tool/avg_time.{tool_name}",
                duration_seconds,
                step=self.step,
                mode="avg",
            )
            log_metric(
                f"tool/success_rate.{tool_name}",
                0 if errored else 1,
                step=self.step,
                mode="avg",
            )

            if errored:
                log_metric(
                    f"tool/failed_count.{tool_name}",
                    1,
                    step=self.step,
                    mode="count",
                )

    def format_as_panel(self, *, truncate: bool = False) -> Panel:
        panel = format_message(self.messages[0], truncate=truncate)
        subtitle = f"[dim]{self.tool_call.id}[/dim]"
        if self.stop:
            subtitle += " [bold red](Requesting Stop)[/bold red]"
        return Panel(
            panel.renderable,
            title=f"Tool End: {self.tool_call.name}",
            title_align="left",
            border_style="orange3",
            subtitle=subtitle,
            subtitle_align="right",
            padding=(1, 1),
        )


class ToolStart(AgentEvent):
    """Event: A tool call is about to be executed."""

    tool_call: rg.tools.ToolCall

    def __repr__(self) -> str:
        return f"ToolStart(tool_call={self.tool_call})"

    def format_as_panel(self, *, truncate: bool = False) -> Panel:
        content: RenderableType
        try:
            args: AnyDict = json.loads(self.tool_call.function.arguments)
            if not args:
                content = Text("No arguments.", style="dim")
            elif truncate:
                content = Text(format_dict(args), style="default")
            else:
                content = Table.grid(padding=(0, 1))
                content.add_column("key", style="dim", no_wrap=True)
                content.add_column("value")
                for k, v in args.items():
                    content.add_row(f"{k}:", repr(v))
        except (json.JSONDecodeError, TypeError):
            # Fallback for non-JSON or unparsable arguments
            content = Text(self.tool_call.function.arguments, style="default")

        return Panel(
            content,
            title=f"Tool Start: {self.tool_call.name}",
            title_align="left",
            border_style="dark_orange3",
            subtitle=f"[dim]{self.tool_call.id}[/dim]",
            subtitle_align="right",
            padding=(1, 1),
        )


class ToolEnd(AgentEvent):
    """Event: A tool call has completed."""

    tool_call: rg.tools.ToolCall
    """The tool call that was completed."""
    result: str | None = None
    """The result returned by the tool, if applicable."""

    def __repr__(self) -> str:
        result_content = shorten_string(str(self.result), 50)
        return f"ToolEnd(tool_call={self.tool_call}, result='{result_content}')"

    def format_as_panel(self, *, truncate: bool = False) -> Panel:
        result_text = self.result or "No result."
        if truncate:
            result_text = shorten_string(result_text, 100)

        return Panel(
            Text(result_text, style="default"),
            title=f"Tool End: {self.tool_call.name}",
            title_align="left",
            border_style="orange3",
            subtitle=f"[dim]{self.tool_call.id}[/dim]",
            subtitle_align="right",
            padding=(1, 1),
        )


class ToolError(AgentEvent):
    """Event: An error occurred during a tool call."""

    tool_call: rg.tools.ToolCall
    """The tool call that caused the error."""
    error: BaseException
    """The error that occurred during the tool call."""

    def format_as_panel(self, *, truncate: bool = False) -> Panel:  # noqa: ARG002
        return Panel(
            repr(self.error),
            title=f"Tool Error: {self.tool_call.name}",
            title_align="left",
            border_style="red",
        )


class GenerationStep(AgentStep):
    """
    A step representing a call to the generator.
    """

    generator: Generator | None = None
    """The model or generator used by the agent during this step."""

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield Rule(f"Step {self.step}: Generation", style="dim cyan", characters="·")

    def log_metrics(self, *, detailed=False):
        from dreadnode import log_metric

        log_metric("generation/total_count", 1, step=self.step, mode="count")
        log_metric(
            "generation/total_tokens", self.usage.total_tokens, step=self.step, mode="direct"
        )
        log_metric(
            "generation/input_tokens", self.usage.input_tokens, step=self.step, mode="direct"
        )
        log_metric(
            "generation/output_tokens", self.usage.output_tokens, step=self.step, mode="direct"
        )

        if detailed:
            gen_name = self.generator.model if self.generator else "unknown"
            log_metric(f"generation/count.{gen_name}", 1, step=self.step, mode="count")


class GenerationStart(AgentEvent):
    """Event: The agent is starting a generation step."""

    generator: Generator | None = None
    """The model or generator used by the agent during this step."""


class GenerationEnd(AgentStep):
    """Event: The agent has completed a generation step."""

    generator: Generator | None = None
    """The model or generator used by the agent during this step."""


class GenerationError(AgentEvent):
    """Event: An error occurred during a generation step."""

    generator: Generator | None = None
    """The model or generator used by the agent during this step."""
    error: BaseException
    """The error that occurred during the generation step."""


class ReactStep(AgentStep):
    """A step representing a reaction from a hook."""

    hook_name: str | None = None
    """The name of the hook that generated this event, if applicable."""
    reaction: Reaction | None = None
    """The reaction taken by a hook, if applicable."""

    def format_as_panel(self, *, truncate: bool = False) -> Panel:  # noqa: ARG002
        reaction_name = self.reaction.__class__.__name__
        details = ""

        if isinstance(self.reaction, RetryWithFeedback):
            details = f" ▸ Feedback: [italic]{self.reaction.feedback}[/italic]"
        elif isinstance(self.reaction, Finish) and self.reaction.reason:
            details = f" ▸ Reason: [italic]{self.reaction.reason}[/italic]"
        elif isinstance(self.reaction, Fail) and self.reaction.error:
            details = f" ▸ Error: [italic]{self.reaction.error}[/italic]"
        elif isinstance(self.reaction, Continue):
            details = (
                f" ▸ Modifying messages ({len(self.messages)} -> {len(self.reaction.messages)})"
            )

        return Panel(
            Text.from_markup(details, style="default"),
            title=f"Hook '{self.hook_name}' reacted: {reaction_name}",
            title_align="left",
            border_style="blue_violet",
        )

    def log_metrics(self, *, detailed=False):
        from dreadnode import log_metric

        log_metric("hook/total_count", 1, step=self.step, mode="count")

        if detailed:
            log_metric(f"hook/{self.hook_name}/count", 1, step=self.step, mode="count")
            log_metric(
                f"reaction/{self.hook_name}/{self.reaction.__class__.__name__}/count",
                1,
                step=self.step,
                mode="count",
            )


class Trajectory(BaseModel):
    """
    The Trajectory creates ordered sequence of all events and steps for a single agent run.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: ULID = Field(default_factory=ULID)
    """The unique identifier for this agent session."""
    agent_id: ULID | None = None
    """The unique identifier for the agent associated with this trajectory."""
    events: list[AgentEvent] = Field(default_factory=list)
    """The ordered list of events and steps in this trajectory."""

    @property
    def steps(self) -> list[AgentStep]:
        """Returns only the AgentStep instances from the event history."""
        return [event for event in self.events if isinstance(event, AgentStep)]

    @property
    def messages(self) -> list[rg.Message]:
        """Returns the reconstructed history from all steps."""
        msgs = []
        for step in self.steps:
            # Since each step only contains the delta, we can safely extend
            msgs.extend(step.messages)
        return msgs

    @property
    def usage(self) -> Usage:
        """Calculates the total usage from all steps in the trajectory."""
        total = Usage(input_tokens=0, output_tokens=0, total_tokens=0)
        for step in self.steps:
            total += step.usage
        return total

    def add_event(self, event: AgentEvent) -> None:
        """Adds a new event or step to the trajectory."""
        self.events.append(event)

    # No changes needed for get_events_by_type, but it's more intuitive now
    def get_events_by_type(self, event_type: type[AgentEventT]) -> list[AgentEventT]:
        return [event for event in self.events if isinstance(event, event_type)]

    def get_summary(self) -> str:
        agent_end = self.get_events_by_type(AgentEnd)[-1]

        return (
            f"Agent '{self.agent_id}' finished: "
            f"reason='{agent_end.stop_reason}', "
            f"steps={len(self.steps)}, "
            f"total_tokens={self.usage.total_tokens}, "
            f"in_tokens={self.usage.input_tokens}, "
            f"out_tokens={self.usage.output_tokens}"
        )

    # In Trajectory:

    def _format_single_step(self, step: AgentStep) -> list[dict[str, t.Any]]:
        if not step.messages:
            return []

        rows = []

        base_payload = {
            "session_id": str(self.session_id),
            "step": step.step,
            "timestamp": step.timestamp.isoformat(),
            "tokens": step.usage.total_tokens if step.usage else 0,
            "type": None,
            "role": None,
            "content": None,
            "tool_calls": None,
            "tool_call_id": None,
            "name": None,
        }

        # --- SPECIAL LOGIC: STEP 1 CONTEXT BACKFILL ---
        # If this is the first step, we must log the System and User messages
        # that happened *before* the agent started running.
        if step.step == 1:
            context_messages = step.messages[:-1]

            for msg in context_messages:
                row = base_payload.copy()
                row.update(
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "type": "context",
                    }
                )
                rows.append(row)

        # --- CURRENT STEP LOGIC ---
        last_msg = step.messages[-1]
        current_row = base_payload.copy()

        if isinstance(step, GenerationStep):
            current_row.update(
                {
                    "type": "generation",
                    "role": "assistant",
                    "content": last_msg.content,
                    "tool_calls": [t.model_dump() for t in last_msg.tool_calls]
                    if last_msg.tool_calls
                    else None,
                }
            )
            rows.append(current_row)

        elif isinstance(step, ToolStep):
            current_row.update(
                {
                    "type": "tool_call",
                    "role": "tool",
                    "content": last_msg.content,
                    "tool": getattr(last_msg, "name", step.tool_call.name),
                }
            )
            rows.append(current_row)

        return rows

    def log_step(self, step: AgentStep) -> None:
        from dreadnode import log_outputs, task_span

        self.events.append(step)

        payload = self._format_single_step(step)

        if payload:
            with task_span(f"step_{step.step}"):
                [log_outputs(**row) for row in payload]

    def trajectory_to_rows(self) -> list[dict[str, t.Any]]:
        rows = []

        session_meta = {
            "session_id": str(self.session_id),
            "agent_id": self.agent_id,
        }

        global_sequence_id = 0

        for step in self.steps:
            if not hasattr(step, "messages") or not step.messages:
                continue

            usage_meta = {}
            if isinstance(step, AgentStep) and step.usage:
                usage_meta = {
                    "input_tokens": step.usage.input_tokens,
                    "output_tokens": step.usage.output_tokens,
                    "total_tokens": step.usage.total_tokens,
                }

            for msg in step.messages:
                row = {
                    **session_meta,
                    "sequence_id": global_sequence_id,
                    "step_index": step.step if hasattr(step, "step") else 0,
                    "role": msg.role,
                    "content": msg.content,
                    "has_tool_calls": bool(msg.tool_calls),
                    "tool_calls_json": msg.tool_calls if msg.tool_calls else None,
                    **usage_meta,
                }
                rows.append(row)
                global_sequence_id += 1

        return rows

    def trajectory_to_turns(self) -> list[dict[str, t.Any]]:
        rows = []

        final_messages = self.messages
        system_prompt = next((m.content for m in final_messages if m.role == "system"), None)

        for event in self.steps:
            if not isinstance(event, AgentStep) or not event.messages:
                continue

            if event.messages[-1].role != "assistant":
                continue

            history = event.messages
            assistant_msg = history[-1]

            user_content = None
            if len(history) >= 2:
                prev_msg = history[-2]
                user_content = prev_msg.content

                if prev_msg.role == "tool":
                    user_content = f"[Tool Result from {prev_msg.tool_call_id}]\n{user_content}"

            rows.append(
                {
                    "session_id": str(self.session_id),
                    "step": event.step,
                    "system": system_prompt,
                    "user": user_content,
                    "assistant": assistant_msg.content,
                    "tool_calls": [t.model_dump() for t in assistant_msg.tool_calls]
                    if assistant_msg.tool_calls
                    else None,
                    "usage_tokens": event.usage.total_tokens if event.usage else 0,
                }
            )

        return rows

    def trajectory_to_nemo(self, tools: list[dict[str, t.Any]] | None = None) -> dict[str, t.Any]:
        """
        Converts the trajectory into a single SFT training example for NeMo-RL.
        """

        formatted_messages = []

        for msg in self.messages:
            entry = {
                "role": msg.role,
                "content": msg.content,
            }

            if msg.tool_calls:
                entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]

            if msg.role == "tool":
                entry["tool_call_id"] = getattr(msg, "tool_call_id", None)
                entry["name"] = getattr(msg, "name", None)

            formatted_messages.append(entry)

        return {
            "session_id": str(self.session_id),
            "agent_id": str(self.agent_id) if self.agent_id else None,
            "messages": formatted_messages,
            "tools": tools,  # Optional: Include schema so the model learns tool definitions
        }

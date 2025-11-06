import json
import typing as t
from dataclasses import dataclass, field
from datetime import datetime, timezone

import rigging as rg
from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
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
    from dreadnode.agent.agent import Agent
    from dreadnode.agent.reactions import Reaction
    from dreadnode.agent.result import AgentResult, AgentStopReason
    from dreadnode.agent.thread import Thread
    from dreadnode.common_types import AnyDict


AgentEventT = t.TypeVar("AgentEventT", bound="AgentEvent")


@dataclass
class AgentEvent:
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc), kw_only=True, repr=False
    )
    """The timestamp of when the event occurred (UTC)."""

    session_id: ULID = field(repr=False)
    """The unique identifier for the agent run session."""
    agent: "Agent" = field(repr=False)
    """The agent associated with this event."""
    thread: "Thread" = field(repr=False)
    """The thread associated with this event."""
    messages: "list[rg.Message]" = field(repr=False)
    """Current messages for this run session."""
    events: "list[AgentEvent]" = field(repr=False)
    """Current events for this run session."""

    @property
    def total_usage(self) -> rg.generator.Usage:
        """Aggregates the usage from all events in the run session."""
        return _total_usage_from_events(self.events)

    @property
    def last_usage(self) -> rg.generator.Usage | None:
        """Returns the usage from the last generation event, if available."""
        if not self.events:
            return None
        last_event = self.events[-1]
        if isinstance(last_event, GenerationEnd):
            return last_event.usage
        return None

    @property
    def estimated_cost(self) -> float | None:
        """Estimates the cost of the agent run based on total token usage and model pricing."""
        import litellm

        if self.agent._generator is None:  # noqa: SLF001
            return None

        model = self.agent._generator.model  # noqa: SLF001
        while model not in litellm.model_cost:
            if "/" not in model:
                return None
            model = "/".join(model.split("/")[1:])

        model_info: AnyDict = litellm.model_cost[model]
        usage = self.total_usage
        input_token_cost = float(model_info.get("input_cost_per_token", 0))
        output_token_cost = float(model_info.get("output_cost_per_token", 0))

        return input_token_cost * usage.input_tokens + output_token_cost * usage.output_tokens

    def get_latest_event_by_type(self, event_type: type[AgentEventT]) -> AgentEventT | None:
        """
        Returns the latest event of the specified type from the thread's events.

        Args:
            event_type: The type of event to search for.
        """
        for event in reversed(self.events):
            if isinstance(event, event_type):
                return event
        return None

    def get_events_by_type(self, event_type: type[AgentEventT]) -> list[AgentEventT]:
        """
        Returns all events of the specified type from the thread's events.

        Args:
            event_type: The type of event to search for.
        """
        return [event for event in self.events if isinstance(event, event_type)]

    def format_as_panel(self, *, truncate: bool = False) -> Panel:  # noqa: ARG002
        """Renders the event as a rich Panel. Can be customized by higher-level systems."""
        return Panel(
            Text(repr(self)),
            title=f"[dim]{self.__class__.__name__}[/dim]",
            border_style="dim",
        )

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield self.format_as_panel()


@dataclass
class AgentStart(AgentEvent):
    def format_as_panel(self, *, truncate: bool = False) -> Panel:
        return Panel(
            format_message(self.messages[0], truncate=truncate),
            title=f"Agent Start: {self.agent.name}",
            title_align="left",
            padding=(1, 1),
        )


@dataclass
class AgentEventInStep(AgentEvent):
    @property
    def step(self) -> int:
        """Returns the current step number."""
        last_step_start = self.get_latest_event_by_type(StepStart)
        return last_step_start.step if last_step_start else 0


@dataclass
class StepStart(AgentEvent):
    step: int

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield Rule(f"Step {self.step}", style="dim cyan", characters="·")


@dataclass
class GenerationEnd(AgentEventInStep):
    message: rg.Message
    usage: "rg.generator.Usage | None"

    def __repr__(self) -> str:
        message_content = shorten_string(str(self.message.content), 50)
        tool_call_count = len(self.message.tool_calls) if self.message.tool_calls else 0
        message = f"Message(role={self.message.role}, content='{message_content}', tool_calls={tool_call_count})"
        return f"GenerationEnd(message={message})"

    def format_as_panel(self, *, truncate: bool = False) -> Panel:
        cost = round(self.estimated_cost, 6) if self.estimated_cost else ""
        usage = str(self.usage) or ""
        return Panel(
            format_message(self.message, truncate=truncate),
            title="Generation End",
            title_align="left",
            subtitle=f"[dim]{usage} [{cost} USD][/dim]",
            subtitle_align="right",
            padding=(1, 1),
        )


@dataclass
class AgentStalled(AgentEventInStep):
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


@dataclass
class AgentError(AgentEventInStep):
    error: BaseException

    def format_as_panel(self, *, truncate: bool = False) -> Panel:  # noqa: ARG002
        return Panel(
            repr(self),
            title="Agent Error",
            title_align="left",
            border_style="red",
        )


@dataclass
class ToolStart(AgentEventInStep):
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


@dataclass
class ToolEnd(AgentEventInStep):
    tool_call: rg.tools.ToolCall
    message: rg.Message
    stop: bool

    def __repr__(self) -> str:
        message_content = shorten_string(str(self.message.content), 50)
        message = f"Message(role={self.message.role}, content='{message_content}')"
        return f"ToolEnd(tool_call={self.tool_call}, message={message}, stop={self.stop})"

    def format_as_panel(self, *, truncate: bool = False) -> Panel:
        panel = format_message(self.message, truncate=truncate)
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


@dataclass
class Reacted(AgentEventInStep):
    hook_name: str
    reaction: "Reaction"

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


@dataclass
class AgentEnd(AgentEvent):
    stop_reason: "AgentStopReason"
    result: "AgentResult"

    def format_as_panel(self, *, truncate: bool = False) -> Panel:  # noqa: ARG002
        res = self.result
        status = "[bold red]Failed[/bold red]" if res.failed else "[bold green]Success[/bold green]"

        table = Table.grid(padding=(0, 2))
        table.add_column(style="dim", justify="right")
        table.add_column()
        table.add_row("Status:", status)
        table.add_row("Stop Reason:", str(self.stop_reason))
        table.add_row("Steps Taken:", str(res.steps))
        table.add_row("Messages:", str(len(res.messages)))
        table.add_row("In Tokens:", str(res.usage.input_tokens))
        table.add_row("Out Tokens:", str(res.usage.output_tokens))
        table.add_row("Total Tokens:", str(res.usage.total_tokens))
        table.add_row(
            "Estimated Cost:",
            f"{round(self.estimated_cost, 6) if self.estimated_cost else '-'} USD",
        )
        return Panel(
            table,
            title="Agent End",
            title_align="left",
            padding=(1, 1),
        )


def _total_usage_from_events(events: list[AgentEvent]) -> rg.generator.Usage:
    """Calculates the total usage from a list of events."""
    total = rg.generator.Usage(input_tokens=0, output_tokens=0, total_tokens=0)
    for event in events:
        if isinstance(event, GenerationEnd) and event.usage:
            total += event.usage
    return total

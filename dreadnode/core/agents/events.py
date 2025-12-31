import json
import typing as t
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from ulid import ULID

from dreadnode.core.agents.format import format_message
from dreadnode.core.agents.reactions import (
    Continue,
    Fail,
    Finish,
    Reaction,
    RetryWithFeedback,
)
from dreadnode.core.generators.generator import Generator, Usage
from dreadnode.core.generators.message import Message
from dreadnode.core.tools import ToolCall
from dreadnode.core.util import format_dict, shorten_string

if t.TYPE_CHECKING:
    from dreadnode.core.types.common import AnyDict

AgentEventT = t.TypeVar("AgentEventT", bound="AgentEvent")
AgentStepT = t.TypeVar("AgentStepT", bound="AgentStep")
AgentStopReason = t.Literal["finished", "max_steps_reached", "error", "stalled"]
AgentStatus = t.Literal["running", "stalled", "errored", "finished"]


class AgentEvent(BaseModel):
    """
    A log event in the agent's lifecycle.

    Attributes:
        timestamp: The timestamp of when the event occurred (UTC).
        agent_id: The name of the agent that generated this event.
        agent_name: The name of the agent that generated this event.
        status: The status of the agent at the time of this event.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), kw_only=True)
    agent_id: ULID = Field(default_factory=ULID)
    agent_name: str | None = None
    status: AgentStatus | None = Field(default=None)


class AgentStep(AgentEvent):
    """
    A discrete unit of work that advances the agent's state.

    A Step is an Event that contains messages that will be part of the
    ongoing chat history.

    Additionally, tracks step count, token usage, etc.

    Attributes:
        generator: The model or generator used by the agent during this step.
        step: The step number in the agent's execution when this event occurred.
        messages: The messages generated or processed during this step.
        usage: The token usage associated with this step, if applicable.
        error: An optional error that occurred during this step's execution.
        stop: Indicates if this step signals a stop condition for the agent.
        estimated_cost: Estimates the cost of the agent run based on total token usage and model pricing.

    """

    generator: Generator | None = None
    step: int = 0
    messages: list[Message] = Field(default_factory=list)
    usage: Usage = Usage(input_tokens=0, output_tokens=0, total_tokens=0)
    error: Exception | None = None
    stop: bool | None = None

    @property
    def estimated_cost(self) -> float | None:
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
    """Event: The agent's execution process has started.

    Attributes:
        inputs: The inputs provided to the agent at the start of execution.
        params: The parameters used to configure the agent at the start of execution.
    """

    inputs: dict[str, t.Any] = Field(default_factory=dict)
    params: dict[str, t.Any] = Field(default_factory=dict)


class AgentEnd(AgentEvent):
    """Event: The agent's execution process has finished.

    Attributes:
        stop_reason: The reason why the agent stopped, if applicable.
        error: The error that caused the agent to stop, if applicable.
    """

    stop_reason: AgentStopReason
    error: Exception | str | None = None


class AgentStalled(AgentEvent):
    """Event: The agent is stalled and there are no tool calls, or stop condition).

    Attributes:
        None
    """

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
    """Event: An error occurred, functionally halting the agent.

    Attributes:
        error: The error that occurred during the agent's execution.
    """

    error: BaseException

    def format_as_panel(self, *, truncate: bool = False) -> Panel:  # noqa: ARG002
        return Panel(
            repr(self),
            title="Agent Error",
            title_align="left",
            border_style="red",
        )


class ToolStep(AgentStep):
    """A step representing the completion of a tool call by the agent.

    Attributes:
       tool_call: The tool call that was completed."""

    tool_call: ToolCall

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
    """Event: A tool call is about to be executed.

    Attributes:
        tool_call: The tool call that is being started.
    """

    tool_call: ToolCall

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
    """Event: A tool call has completed.

    Attributes:
        tool_call: The tool call that was completed.
        result: The result returned by the tool, if applicable.
    """

    tool_call: ToolCall
    result: str | None = None

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
    """Event: An error occurred during a tool call.

    Attributes:
        tool_call: The tool call that caused the error.
        error: The error that occurred during the tool call.
    """

    tool_call: ToolCall
    error: BaseException

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

    Attributes:
        generator: The model or generator used by the agent during this step.
    """

    generator: Generator | None = None

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
    """Event: The agent is starting a generation step.

    Attributes:
        generator: The model or generator used by the agent during this step.
    """

    generator: Generator | None = None


class GenerationEnd(AgentStep):
    """Event: The agent has completed a generation step.

    Attributes:
        generator: The model or generator used by the agent during this step.
    """

    generator: Generator | None = None


class GenerationError(AgentEvent):
    """Event: An error occurred during a generation step

    Attributes:
        generator: The model or generator used by the agent during this step.
        error: The error that occurred during the generation step.
    """

    generator: Generator | None = None
    error: BaseException


class ReactStep(AgentStep):
    """A step representing a reaction from a hook.

    Attributes:
        hook_name: The name of the hook that generated this event, if applicable.
        reaction: The reaction taken by a hook, if applicable.

    """

    hook_name: str | None = None
    reaction: Reaction | None = None

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

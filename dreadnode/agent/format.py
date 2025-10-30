import typing as t

import rigging as rg
from rich import box
from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from dreadnode.util import get_callable_name, shorten_string

if t.TYPE_CHECKING:
    from dreadnode.agent.agent import Agent


def format_agents(agents: "list[Agent]") -> RenderableType:
    """
    Takes a list of Agent objects and formats them into a concise rich Table.
    """
    table = Table(box=box.ROUNDED)
    table.add_column("Name", style="orange_red1", no_wrap=True)
    table.add_column("Description", min_width=20)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Tools", style="cyan")

    for agent in agents:
        tool_names = ", ".join(tool.name for tool in agent.tools) if agent.tools else "-"
        table.add_row(
            agent.name,
            agent.description or "-",
            agent.model_name or "-",
            tool_names,
        )

    return table


def format_agent(agent: "Agent") -> RenderableType:
    """
    Takes a single Agent and formats its full details into a rich Panel.
    """
    details = Table(
        box=box.MINIMAL,
        show_header=False,
        style="orange_red1",
    )
    details.add_column("Property", style="bold dim", justify="right", no_wrap=True)
    details.add_column("Value", style="white")

    details.add_row(Text("Description", justify="right"), agent.description or "-")
    details.add_row(Text("Model", justify="right"), agent.model_name or "-")
    details.add_row(
        Text("Instructions", justify="right"),
        f'"{shorten_string(agent.instructions, 100)}"' if agent.instructions else "-",
    )

    if agent.tools:
        tool_names = ", ".join(f"[cyan]{tool.name}[/]" for tool in agent.tools)
        details.add_row(Text("Tools", justify="right"), tool_names)

    if agent.hooks:
        hook_names = ", ".join(
            f"[cyan]{get_callable_name(hook, short=True)}[/]" for hook in agent.hooks
        )
        details.add_row(Text("Hooks", justify="right"), hook_names)

    if agent.stop_conditions:
        stop_names = ", ".join(f"[yellow]{cond.name}[/]" for cond in agent.stop_conditions)
        details.add_row(Text("Stops", justify="right"), stop_names)

    return Panel(
        details,
        title=f"[bold]{agent.name}[/]",
        title_align="left",
        border_style="orange_red1",
    )


def format_message(message: rg.Message, *, truncate: bool = False, markdown: bool = False) -> Panel:
    """Formats a single message into a rich renderable."""
    color = (
        "magenta"
        if message.role == "system"
        else "blue"
        if message.role == "user"
        else "magenta"
        if message.role == "tool"
        else "cyan"
    )

    items: list[RenderableType] = []
    for part in message.content_parts:
        if isinstance(part, rg.ContentText):
            text = (
                shorten_string(part.text, max_length=500, max_lines=15, separator="\n[...]\n")
                if truncate
                else part.text
            )
            items.append(Markdown(text) if markdown else Text(text))

        else:
            items.append(Text.from_markup(f" |- [magenta]{part}[/]"))

    if message.tool_calls:
        if message.content_parts:
            items.append(Rule(style="dim"))

        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = (
                shorten_string(tool_call.function.arguments, 100)
                if truncate
                else tool_call.function.arguments
            )
            items.append(
                Text.from_markup(f" |- [magenta]{function_name}[/]([dim]{function_args}[/])")
            )

    return Panel(
        Group(*items),
        title=f"[bold {color}]{message.role}[/]",
        title_align="left",
        border_style=color,
    )


def format_messages(
    messages: t.Sequence[rg.Message], *, truncate: bool = False, markdown: bool = False
) -> RenderableType:
    """Formats a list of messages into a rich renderable."""
    panels = [format_message(m, truncate=truncate, markdown=markdown) for m in messages]
    return Group(*panels)

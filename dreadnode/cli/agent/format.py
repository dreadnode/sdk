from rich import box
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from dreadnode.agent.agent import Agent
from dreadnode.util import get_callable_name, shorten_string


def format_agents_table(agents: list[Agent]) -> RenderableType:
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
            # Show only the primary model for brevity in the table
            (agent.model or "-").split(",")[0],
            tool_names,
        )

    return table


def format_agent(agent: Agent) -> RenderableType:
    """
    Takes a single Agent and formats its full details into a rich Panel.
    This is used for the --verbose view.
    """
    details = Table(
        box=box.MINIMAL,
        show_header=False,
        style="orange_red1",
    )
    details.add_column("Property", style="bold dim", justify="right", no_wrap=True)
    details.add_column("Value", style="white")

    # Add rows, borrowing logic directly from the Agent.__repr__
    details.add_row(Text("Description", justify="right"), agent.description or "-")
    details.add_row(Text("Model", justify="right"), agent.model or "-")
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
        stop_names = ", ".join(f"[yellow]{cond!r}[/]" for cond in agent.stop_conditions)
        details.add_row(Text("Stops", justify="right"), stop_names)

    return Panel(
        details,
        title=f"[bold]{agent.name}[/]",
        title_align="left",
        border_style="orange_red1",
    )

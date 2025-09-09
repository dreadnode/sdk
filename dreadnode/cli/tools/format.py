import typing as t

from rich import box
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if t.TYPE_CHECKING:
    from dreadnode.agent.tools import Toolset


def format_tools_table(tools: "list[Toolset]") -> RenderableType:
    """
    Takes a list of Toolset objects and formats them into a concise rich Table.
    """

    table = Table(box=box.ROUNDED)
    table.add_column("Name", style="orange_red1", no_wrap=True)
    table.add_column("Description", min_width=20)

    table.add_column("Variant", style="cyan", no_wrap=True)

    table.add_column("Methods", style="cyan")

    for toolset in tools:
        tool_names = ", ".join(tool.name for tool in toolset.get_tools()) if toolset else "-"

        table.add_row(
            toolset.name,
            toolset.__doc__.strip().split("\n")[0] if toolset.__doc__ else "-",
            toolset.variant or "-",
            tool_names,
        )

    return table


def format_tool(toolset: "Toolset") -> RenderableType:
    """
    Takes a single Toolset and formats its full details into a rich Panel.
    """

    details = Table(
        box=box.MINIMAL,
        show_header=False,
        style="orange_red1",
    )

    details.add_column("Property", style="bold dim", justify="right", no_wrap=True)
    details.add_column("Value", style="white")
    details.add_row(
        Text("Description", justify="right"), toolset.__doc__.strip() if toolset.__doc__ else "-"
    )
    details.add_row(Text("Variant", justify="right"), toolset.variant or "-")

    if toolset.get_tools():
        tool_names = ", ".join(f"[cyan]{tool.name}[/]" for tool in toolset.get_tools())

        details.add_row(Text("Methods", justify="right"), tool_names)

    return Panel(
        details,
        title=f"[bold]{toolset.name}[/]",
        title_align="left",
        border_style="orange_red1",
    )

import typing as t

from rich import box
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if t.TYPE_CHECKING:
    from dreadnode.task import Task


def format_tasks(tasks: "list[Task[..., t.Any]]") -> RenderableType:
    """
    Takes a list of Task objects and formats them into a concise rich Table.
    """
    table = Table(box=box.ROUNDED)
    table.add_column("Name", style="orange_red1", no_wrap=True)
    table.add_column("Description", min_width=20)
    table.add_column("Scorers", style="cyan")

    for task in tasks:
        scorer_names = ", ".join(scorer.name for scorer in task.scorers) if task.scorers else "-"
        table.add_row(
            task.name,
            task.func.__doc__ or "-",
            scorer_names,
        )

    return table


def format_task(task: "Task[..., t.Any]") -> RenderableType:
    """
    Takes a single Task and formats its full details into a rich Panel.
    """
    details = Table(
        box=box.MINIMAL,
        show_header=False,
        style="orange_red1",
    )
    details.add_column("Property", style="bold dim", justify="right", no_wrap=True)
    details.add_column("Value", style="white")

    details.add_row(Text("Description", justify="right"), task.func.__doc__ or "-")

    if task.scorers:
        scorer_names = ", ".join(f"[cyan]{scorer.name}[/]" for scorer in task.scorers)
        details.add_row(Text("Scorers", justify="right"), scorer_names)

    if task.tags:
        tag_names = ", ".join(f"[yellow]{tag}[/]" for tag in task.tags)
        details.add_row(Text("Tags", justify="right"), tag_names)

    return Panel(
        details,
        title=f"[bold]{task.name}[/]",
        title_align="left",
        border_style="orange_red1",
    )

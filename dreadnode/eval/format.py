import typing as t
from pathlib import Path

from rich import box
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from dreadnode.scorers.base import Scorer

if t.TYPE_CHECKING:
    from dreadnode.eval import Eval


def format_evals(evals: "list[Eval]") -> RenderableType:
    """
    Takes a list of Eval objects and formats them into a concise rich Table.
    """
    table = Table(box=box.ROUNDED)
    table.add_column("Name", style="orange_red1", no_wrap=True)
    table.add_column("Description", min_width=20)
    table.add_column("Task", style="cyan", no_wrap=True)
    table.add_column("Dataset", style="cyan")
    table.add_column("Scorers", style="cyan")

    for evaluation in evals:
        scorer_names = (
            ", ".join(scorer.name for scorer in Scorer.fit_many(evaluation.scorers))
            if evaluation.scorers
            else "-"
        )
        table.add_row(
            evaluation.name,
            evaluation.description or "-",
            evaluation.task_name,
            format_dataset(evaluation.dataset, verbose=False),
            scorer_names,
        )

    return table


def format_eval(evaluation: "Eval") -> RenderableType:
    """
    Takes a single Eval and formats its full details into a rich Panel.
    """
    details = Table(
        box=box.MINIMAL,
        show_header=False,
        style="orange_red1",
    )
    details.add_column("Property", style="bold dim", justify="right", no_wrap=True)
    details.add_column("Value", style="white")

    details.add_row(Text("Description", justify="right"), evaluation.description or "-")
    details.add_row(Text("Task", justify="right"), str(evaluation.task))
    details.add_row(
        Text("Dataset", justify="right"), format_dataset(evaluation.dataset, verbose=True)
    )

    if evaluation.parameters:
        param_keys = ", ".join(f"[cyan]{key}[/]" for key in evaluation.parameters)
        details.add_row(Text("Parameters", justify="right"), param_keys)

    if evaluation.scorers:
        scorer_names = ", ".join(
            f"[cyan]{scorer.name}[/]" for scorer in Scorer.fit_many(evaluation.scorers)
        )
        details.add_row(Text("Scorers", justify="right"), scorer_names)

    if evaluation.assert_scores:
        assertions = (
            ", ".join(f"[yellow]{assertion}[/]" for assertion in evaluation.assert_scores)
            if isinstance(evaluation.assert_scores, list)
            else "[yellow]All[/]"
        )
        details.add_row(Text("Assertions", justify="right"), assertions)

    return Panel(
        details,
        title=f"[bold]{evaluation.name}[/]",
        title_align="left",
        border_style="orange_red1",
    )


def format_dataset(dataset: t.Any, *, verbose: bool = False) -> RenderableType:
    """Formats a dataset into a rich renderable, handling large lists gracefully."""
    if isinstance(dataset, (str, Path)):
        return Text(str(dataset), style="green")

    if isinstance(dataset, list):
        count = len(dataset)
        if not count:
            return Text("Empty list", style="dim")

        if not verbose:
            return Text(f"List ({count} items)", style="cyan")

        details = Table(box=None, show_header=False)
        details.add_column(style="bold dim", justify="right")
        details.add_column(style="white")
        details.add_row("Total Items", str(count))

        first_item = dataset[0]
        if isinstance(first_item, dict):
            keys = ", ".join(f"[cyan]{key}[/]" for key in first_item)
            details.add_row("Item Keys", keys)

        return Panel(
            details,
            title="[bold]In-Memory Dataset[/]",
            border_style="green",
            title_align="left",
        )

    return Text(str(dataset))

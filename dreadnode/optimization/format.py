import typing as t

from rich import box
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from dreadnode.eval.format import format_dataset
from dreadnode.scorers.base import Scorer
from dreadnode.util import get_callable_name

if t.TYPE_CHECKING:
    from dreadnode.optimization import Study


def format_studies(studies: "list[Study]") -> RenderableType:
    """
    Takes a list of Study objects and formats them into a concise rich Table.
    """
    table = Table(box=box.ROUNDED)
    table.add_column("Name", style="orange_red1", no_wrap=True)
    table.add_column("Description", min_width=20)
    table.add_column("Objectives", style="cyan")
    table.add_column("Search Strategy", style="cyan")

    for study in studies:
        objective_names = ", ".join(study.objective_names)
        table.add_row(
            study.name,
            study.description or "-",
            objective_names,
            get_callable_name(study.search_strategy, short=True),
        )

    return table


def format_study(study: "Study") -> RenderableType:
    """
    Takes a single Study and formats its full details into a rich Panel.
    """
    details = Table(
        box=box.MINIMAL,
        show_header=False,
        style="orange_red1",
    )
    details.add_column("Property", style="bold dim", justify="right", no_wrap=True)
    details.add_column("Value", style="white")

    details.add_row(Text("Description", justify="right"), study.description or "-")
    details.add_row(Text("Task Factory", justify="right"), get_callable_name(study.task_factory))
    details.add_row(
        Text("Search Strategy", justify="right"), get_callable_name(study.search_strategy)
    )

    if study.dataset is not None:
        details.add_row(
            Text("Dataset", justify="right"), format_dataset(study.dataset, verbose=True)
        )

    if study.objectives:
        objective_names = ", ".join(f"[cyan]{name}[/]" for name in study.objective_names)
        details.add_row(Text("Objectives", justify="right"), objective_names)
        directions = ", ".join(f"[yellow]{direction}[/]" for direction in study.directions)
        details.add_row(Text("Directions", justify="right"), directions)

    if study.constraints:
        constraint_names = ", ".join(
            f"[cyan]{c.name}[/]" for c in Scorer.fit_many(study.constraints)
        )
        details.add_row(Text("Constraints", justify="right"), constraint_names)

    if study.stop_conditions:
        stop_names = ", ".join(f"[yellow]{cond.name}[/]" for cond in study.stop_conditions)
        details.add_row(Text("Stops", justify="right"), stop_names)

    return Panel(
        details,
        title=f"[bold]{study.name}[/]",
        title_align="left",
        border_style="orange_red1",
    )

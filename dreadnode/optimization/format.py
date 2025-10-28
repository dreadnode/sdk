import typing as t

from rich import box
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from dreadnode.eval.format import _format_dataset
from dreadnode.scorers.base import Scorer
from dreadnode.util import get_callable_name, shorten_string

if t.TYPE_CHECKING:
    from dreadnode.optimization import Study
    from dreadnode.optimization.result import StudyResult
    from dreadnode.optimization.trial import Trial


def format_studies(studies: "t.Sequence[Study]") -> RenderableType:
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
            study.search_strategy.name,
        )

    return table


def format_study(study: "Study") -> RenderableType:
    """
    Format a single Study object into a detailed rich Panel.
    """
    from dreadnode.airt import Attack

    details = Table(
        box=box.MINIMAL,
        show_header=False,
        style="orange_red1",
    )
    details.add_column("Property", style="bold dim", justify="right", no_wrap=True)
    details.add_column("Value", style="white")

    details.add_row(Text("Description", justify="right"), study.description or "-")

    # TODO(nick): Move attack formatting out of here
    if isinstance(study, Attack):
        details.add_row(Text("Target", justify="right"), repr(study.target))
    else:
        details.add_row(
            Text("Task Factory", justify="right"), get_callable_name(study.task_factory)
        )

    details.add_row(Text("Search Strategy", justify="right"), study.search_strategy.name)

    if study.dataset is not None:
        details.add_row(
            Text("Dataset", justify="right"), _format_dataset(study.dataset, verbose=True)
        )

    if study.objectives:
        objectives = " | ".join(
            f"[cyan]{name} :arrow_upper_right:[/]"
            if direction == "maximize"
            else f"[magenta]{name} :arrow_lower_right:[/]"
            for name, direction in zip(study.objective_names, study.directions, strict=True)
        )
        details.add_row(Text("Objectives", justify="right"), objectives)

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


def format_study_result(result: "StudyResult") -> RenderableType:
    """
    Format a StudyResult into a rich Table.
    """
    table = Table.grid(padding=(0, 2))
    table.add_column("Metric", style="dim")
    table.add_column("Value")
    table.add_row("Stop Reason:", f"[bold]{result.stop_reason}[/bold]")
    table.add_row("Explanation:", result.stop_explanation or "-")
    if (num_failed_trials := len(result.failed_trials)) > 0:
        table.add_row("Failed Trials:", f"[red]{num_failed_trials}[/red]")
    if (num_pruned_trials := len(result.pruned_trials)) > 0:
        table.add_row("Pruned Trials:", f"[yellow]{num_pruned_trials}[/yellow]")
    if (num_pending_trials := len(result.pending_trials)) > 0:
        table.add_row("Pending Trials:", f"[dim]{num_pending_trials}[/dim]")
    table.add_row("Total Trials:", str(len(result.trials)))

    return table


def format_trial(trial: "Trial[t.Any]") -> RenderableType:
    """
    Format a Trial object into a rich Table.
    """
    table = Table.grid(padding=(0, 2))
    table.add_column("Name")
    table.add_column("Score", justify="right", min_width=10)
    for name, value in trial.scores.items():
        table.add_row(
            name,
            f"[bold magenta]{value:.6f}[/bold magenta]",
        )

    for name, value in trial.all_scores.items():
        if name not in trial.scores:
            table.add_row(f"[dim]{name}[/dim]", f"[dim]{value:.6f}[/dim]")

    # Main content grid
    candidate_str = shorten_string(str(trial.candidate), max_length=500, separator="\n\n[...]\n\n")
    output_str = (
        shorten_string(str(trial.output), max_length=500, separator="\n\n[...]\n\n")
        if trial.output
        else ""
    )

    grid = Table.grid(expand=True)
    grid.add_column()
    grid.add_row(Panel(table, title="Scores", title_align="left"))
    grid.add_row(
        Panel(
            Text(candidate_str, style="dim"),
            title="Candidate",
            title_align="left",
        )
    )
    if trial.output:
        grid.add_row(Panel(Text(output_str, style="dim"), title="Output", title_align="left"))

    return grid

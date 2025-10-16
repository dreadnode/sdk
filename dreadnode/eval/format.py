import typing as t
from functools import reduce
from operator import mul
from pathlib import Path

from rich import box
from rich.console import Group, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from dreadnode.scorers.base import Scorer

if t.TYPE_CHECKING:
    from dreadnode.eval import Eval
    from dreadnode.eval.result import EvalResult, ScenarioResult


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
    table.add_column("Iterations", style="magenta")
    table.add_column("Parameters", style="magenta")

    for evaluation in evals:
        scorer_names = (
            ", ".join(scorer.name for scorer in Scorer.fit_many(evaluation.scorers))
            if evaluation.scorers
            else "-"
        )

        param_info = "-"
        if evaluation.parameters:
            num_keys = len(evaluation.parameters)
            # Calculate the total number of combinations
            total_combinations = reduce(mul, (len(v) for v in evaluation.parameters.values()), 1)
            param_info = f"{total_combinations} combinations ({num_keys} keys)"

        table.add_row(
            evaluation.name,
            evaluation.description or "-",
            evaluation.task_name,
            _format_dataset(evaluation.dataset, evaluation.dataset_file, verbose=False),
            scorer_names,
            str(evaluation.iterations),
            param_info,
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
        Text("Dataset", justify="right"),
        _format_dataset(evaluation.dataset, evaluation.dataset_file, verbose=True),
    )

    if evaluation.label:
        details.add_row(Text("Label", justify="right"), evaluation.label)

    if evaluation.tags:
        details.add_row(Text("Tags", justify="right"), ", ".join(evaluation.tags))

    details.add_row(Text("Iterations", justify="right"), str(evaluation.iterations))
    details.add_row(Text("Concurrency", justify="right"), str(evaluation.concurrency))

    if evaluation.max_errors is not None:
        details.add_row(Text("Max Errors", justify="right"), str(evaluation.max_errors))

    if evaluation.max_consecutive_errors is not None:
        details.add_row(
            Text("Max Consecutive Errors", justify="right"),
            str(evaluation.max_consecutive_errors),
        )

    if evaluation.preprocessor:
        details.add_row(
            Text("Preprocessor", justify="right"),
            f"[cyan]{evaluation.preprocessor.__name__}[/]",
        )

    if evaluation.parameters:
        details.add_row(
            Text("Parameters", justify="right"), _format_parameters(evaluation.parameters)
        )

    if evaluation.dataset_input_mapping:
        details.add_row(
            Text("Input Mapping", justify="right"),
            str(evaluation.dataset_input_mapping),
        )

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


def format_eval_result(result: "EvalResult") -> RenderableType:
    """
    Formats an EvalResult into a rich, detailed Panel containing an overall
    summary and a breakdown for each scenario.
    """
    summary_table = Table(show_header=False, box=box.MINIMAL)
    summary_table.add_column(style="dim")
    summary_table.add_column(style="white")

    stop_reason_style = "green" if result.stop_reason == "finished" else "yellow"
    failed_style = "yellow" if result.failed_count > 0 else "dim"
    error_style = "red" if result.error_count > 0 else "dim"

    summary_table.add_row("Stop Reason", f"[{stop_reason_style}]{result.stop_reason}[/]")
    summary_table.add_row("Scenarios", str(len(result.scenarios)))
    summary_table.add_row("Pass Rate", f"[bold]{result.pass_rate:.2%}[/]")
    summary_table.add_row("Samples", str(len(result.samples)))
    summary_table.add_row("Passed", f"[green]{result.passed_count}[/]")
    summary_table.add_row("Failed", f"[{failed_style}]{result.failed_count}[/]")
    summary_table.add_row("Errors", f"[{error_style}]{result.error_count}[/]")

    content: list[RenderableType] = [summary_table]
    for i, scenario in enumerate(result.scenarios):
        content.append(Padding(Rule(title=f"Scenario {i + 1}", style="cyan"), 1))
        content.append(_format_scenario_result(scenario))

    return Panel(
        Group(*content),
        title="Evaluation Result",
        border_style="magenta",
        title_align="left",
    )


def _format_scenario_result(result: "ScenarioResult") -> RenderableType:
    """
    Formats a ScenarioResult into a rich Panel, including parameters,
    stats, and breakdowns for assertions and metrics.
    """

    param_table = Table(show_header=False, box=box.MINIMAL)
    param_table.add_column("Parameter", style="cyan")
    param_table.add_column("Value", overflow="fold")
    for key, value in result.params.items():
        param_table.add_row(key, str(value))

    summary_table = Table(show_header=False, box=box.MINIMAL)
    summary_table.add_column(style="dim")
    summary_table.add_column()
    failed_style = "yellow" if result.failed_count > 0 else "dim"
    error_style = "red" if result.error_count > 0 else "dim"
    summary_table.add_row("Pass Rate", f"{result.pass_rate:.2%}")
    summary_table.add_row("Passed", f"[green]{result.passed_count}[/]")
    summary_table.add_row("Failed", f"[{failed_style}]{result.failed_count}[/]")
    summary_table.add_row("Errors", f"[{error_style}]{result.error_count}[/]")

    summary = Panel(
        summary_table, title="[bold dim]Summary[/]", border_style="dim", title_align="left"
    )
    params = Panel(
        param_table, title="[bold dim]Parameters[/]", border_style="dim", title_align="left"
    )
    assertions = Panel(
        _format_assertions(result.assertions_summary),
        title="[bold dim]Assertions[/]",
        border_style="dim",
        title_align="left",
    )
    metrics = Panel(
        _format_metrics_summary(result.metrics_summary),
        title="[bold dim]Metrics[/]",
        border_style="dim",
        title_align="left",
    )

    grid = Table.grid("left", "right", expand=True)
    grid.add_column()
    grid.add_row(summary, params)
    grid.add_row(assertions, metrics)

    return grid


def _format_assertions(assertions: dict[str, dict[str, float | int]]) -> RenderableType:
    """
    Formats the assertions summary into a rich Table.
    """
    table = Table(box=box.MINIMAL)
    table.add_column("Assertion", style="cyan", overflow="fold")
    table.add_column("Passed", justify="right")
    table.add_column("Failed", justify="right")
    table.add_column("Pass Rate", justify="right")

    for name, stats in assertions.items():
        failed_count = stats["failed_count"]
        passed_count = stats["passed_count"]
        total = passed_count + failed_count
        pass_rate = (passed_count / total) if total > 0 else 0.0
        failed_style = "yellow" if failed_count > 0 else "dim"

        table.add_row(
            name,
            f"[green]{passed_count}[/]",
            f"[{failed_style}]{failed_count}[/]",
            f"{pass_rate:.2%}",
        )

    return table


def _format_metrics_summary(metrics_summary: dict[str, dict[str, float]]) -> RenderableType:
    """
    Formats the metrics summary into a rich Table.
    """
    table = Table(box=box.MINIMAL)
    table.add_column("Metric", style="cyan", overflow="fold")
    table.add_column("Mean", justify="right")
    table.add_column("Std Dev", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Count", justify="right")

    for name, stats in metrics_summary.items():
        table.add_row(
            name,
            f"{stats['mean']:.5f}",
            f"{stats['stdev']:.5f}",
            f"{stats['min']:.5f}",
            f"{stats['max']:.5f}",
            str(stats["count"]),
        )

    return table


def _format_dataset(  # noqa: PLR0911
    dataset: t.Any, dataset_file: t.Any = None, *, verbose: bool = False
) -> RenderableType:
    """
    Formats a dataset into a rich renderable, handling large lists gracefully.
    """
    if dataset_file:
        return Text(str(dataset_file), style="green")

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

        if count > 0 and isinstance(dataset[0], dict):
            keys = ", ".join(f"[cyan]{key}[/]" for key in dataset[0])
            details.add_row("Item Keys", keys)

        return Panel(
            details,
            title="[bold]In-Memory Dataset[/]",
            border_style="green",
            title_align="left",
        )

    if callable(dataset):
        return Text(f"Callable: {dataset.__name__}", style="cyan")

    return Text(str(dataset))


def _format_parameters(
    parameters: dict[str, list[t.Any]], *, max_display: int = 5
) -> RenderableType:
    """
    Formats the parameters of an Eval into a rich Table.
    """
    if not parameters:
        return Text("-", style="dim")

    param_table = Table(show_header=False, expand=True)
    param_table.add_column("Parameter", style="cyan", no_wrap=True)
    param_table.add_column("Values")

    for key, values in parameters.items():
        num_values = len(values)

        display_values = [f"[bright_white]{v!s}[/]" for v in values[:max_display]]
        value_str = ", ".join(display_values)

        if num_values > max_display:
            value_str += f", ... ([yellow]{num_values} total[/])"

        param_table.add_row(key, value_str)

    return param_table

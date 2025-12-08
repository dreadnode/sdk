import statistics
import typing as t
from collections import deque
from copy import copy

from rich import box
from rich.columns import Columns
from rich.console import Console, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from dreadnode.optimization.events import (
    NewBestTrialFound,
    StudyEnd,
    StudyEvent,
    TrialAdded,
    TrialComplete,
    TrialPruned,
    TrialStart,
)
from dreadnode.optimization.format import format_study_result, format_trial
from dreadnode.optimization.result import StudyResult

if t.TYPE_CHECKING:
    from dreadnode.optimization.study import Study
    from dreadnode.optimization.trial import Trial

from dreadnode.logging_ import console as logger_console


class StudyConsoleAdapter:
    """Consumes a Study's event stream and renders a live progress dashboard."""

    def __init__(
        self, study: "Study[t.Any]", *, console: Console | None = None, max_log_entries: int = 50
    ):
        self.study = study
        self.console = console or logger_console

        self._best_trial: Trial | None = None
        self._result: StudyResult | None = None
        self._trials: deque[Trial] = deque(maxlen=max_log_entries)
        self._probes: deque[Trial] = deque(maxlen=max_log_entries)

        self._trials_running = 0
        self._trials_pending = 0
        self._trials_completed = 0
        self._probes_running = 0
        self._probes_pending = 0
        self._probes_completed = 0
        self._trials_since_best = 0
        self._completed_evals = 0
        self._total_cost = 0
        self._last_error: str | None = None

        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=60),
            TextColumn("{task.completed}/{task.total} Evals"),
            SpinnerColumn("dots"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True,
        )
        self._progress_task_id: TaskID = self._progress.add_task(
            "[bold]Overall Progress", total=self.study.max_evals
        )

    def _get_score_progress_indicator(self, trial_score: float) -> tuple[str, str]:
        """Returns (indicator_text, style) based on standard deviations from mean."""
        completed_scores = [
            t.score for t in self._trials if t.status == "finished" and t.score is not -float("inf")
        ]

        if len(completed_scores) < 2:
            return "", "dim"

        try:
            mean_score = statistics.mean(completed_scores)
            std_dev = statistics.stdev(completed_scores)
        except statistics.StatisticsError:
            return "", "dim"

        if std_dev == 0:  # All scores are identical
            return "", "dim"

        z_score = (trial_score - mean_score) / std_dev

        # Color coding based on standard deviations
        if z_score >= 2.0:
            style = "green4"
        elif z_score >= 1.0:
            style = "dark_green"
        elif z_score < -1.0:
            style = "dark_red"
        elif z_score < -2.0:
            style = "deep_pink4"
        else:
            style = "dim"

        return f"{z_score:+.1f}", style

    def _build_header(self) -> RenderableType:
        grid = Table.grid(expand=True)
        grid.add_column("Best Score", justify="left", ratio=1)
        grid.add_column("Status", justify="right", ratio=2)

        best_score_text = "[dim]-[/dim]"
        if self._best_trial:
            best_score_text = f"[bold magenta]{self._best_trial.score:.5f}[/bold magenta]"

        status_text = Text.from_markup(
            f"Trials: [bold cyan]{self._trials_running}[/] / [bold]{self._trials_pending}[/] / [bold green]{self._trials_completed}[/] | "
            f"Probes: [bold cyan]{self._probes_running}[/] / [bold]{self._probes_pending}[/] / [bold green]{self._probes_completed}[/] | "
            f"Since Best: [bold magenta]{self._trials_since_best}[/] | "
            f"Total Cost: [bold]{self._total_cost}[/]"
        )

        grid.add_row(
            Text.from_markup(f"[b]Best Score:[/b] {best_score_text}"),
            status_text,
        )
        return grid

    def _build_best_trial_panel(self) -> RenderableType:
        if not self._best_trial:
            return Panel(
                Text("No successful trials yet.", style="dim", justify="center"),
                title="[bold magenta]Current Best[/bold magenta]",
                border_style="dim",
            )

        trial = self._best_trial
        transformed_input = trial.transformed_input

        # If we have transforms, display both versions
        if transformed_input is not None and transformed_input != trial.candidate:
            # Original candidate panel
            original_panel = Panel(
                format_trial(trial),
                title="[dim]Original[/dim]",
                border_style="dim",
            )

            # Transformed candidate panel
            display_trial = copy(trial)
            display_trial.candidate = transformed_input
            transformed_panel = Panel(
                format_trial(display_trial),
                title="[bold]Transformed (Sent to Target)[/bold]",
                border_style="green",
            )

            return Panel(
                Columns([original_panel, transformed_panel]),
                title="[bold magenta]Current Best[/bold magenta]",
                border_style="magenta",
            )

        # No transforms or transforms didn't change input
        return Panel(
            format_trial(trial),
            title="[bold magenta]Current Best[/bold magenta]",
            border_style="magenta",
        )

    def _build_trials_panel(self) -> RenderableType:
        table = Table(expand=True, box=box.ROUNDED)
        table.add_column("ID", width=8)
        table.add_column("Status", width=8)
        table.add_column("Cost", justify="right", style="dim")
        table.add_column("Score/σ", justify="right")  # noqa: RUF001

        for trial in self._trials:
            color = {
                "finished": "default",
                "failed": "red",
                "pruned": "yellow",
                "running": "cyan",
            }.get(trial.status, "dim")
            status_text = f"[{color}]{trial.status}[/{color}]"

            if trial.status == "finished":
                indicator, indicator_style = self._get_score_progress_indicator(trial.score)
                score_str = f"[bold]{trial.score:.6f}[/bold] [{indicator_style}]{indicator}[/{indicator_style}]"
            else:
                score_str = "..."

            trial_id = Text(
                str(trial.id)[16:],
                style="bold magenta"
                if self._best_trial and trial.id == self._best_trial.id
                else "dim",
            )

            table.add_row(trial_id, status_text, str(trial.cost), score_str)

        return Panel(
            table if self._trials else Text("No trials yet.", style="dim", justify="center"),
            title="[bold]Trials[/bold]",
            border_style="dim",
        )

    def _build_probes_panel(self) -> RenderableType:
        table = Table(expand=True, box=box.ROUNDED)
        table.add_column("ID", style="dim", width=8)
        table.add_column("Status", width=8)
        table.add_column("Cost", justify="right", style="dim")
        table.add_column("Score", justify="right")

        for probe in self._probes:
            color = {"finished": "default", "failed": "red", "running": "cyan"}.get(
                probe.status, "dim"
            )
            status_text = f"[{color}]{probe.status}[/{color}]"
            score_str = f"{probe.score:.5f}" if probe.status == "finished" else "..."
            table.add_row(str(probe.id)[16:], status_text, str(probe.cost), score_str)

        return Panel(table, title="[bold]Probes[/bold]", border_style="dim")

    def _build_dashboard(self) -> RenderableType:
        layout = Layout()

        layout.split_column(
            Layout(Padding(self._build_header(), 1), size=3),
            Layout(Rule(style="cyan"), size=1),
            Layout(name="body"),
            Layout(Padding(self._progress, 1), size=3),
        )

        if self._last_error:
            layout.add_split(
                Layout(
                    Panel(
                        self._last_error,
                        title="Last Error",
                        title_align="left",
                        border_style="red",
                    ),
                    size=3,
                )
            )

        layout["body"].split_row(
            Layout(self._build_best_trial_panel(), name="left", ratio=3),
            Layout(name="right", ratio=2),
        )

        if self._probes:
            layout["body"]["right"].split_column(
                self._build_trials_panel(),
                self._build_probes_panel(),
            )
        else:
            layout["body"]["right"].update(self._build_trials_panel())

        return Layout(
            Panel(
                layout,
                title=Text(self.study.name, justify="center", style="bold cyan"),
                border_style="cyan",
            )
        )

    def _handle_event(self, event: StudyEvent[t.Any]) -> None:  # noqa: PLR0912
        if self._best_trial:
            self._trials_since_best = self._trials_completed - self._best_trial.step

        if isinstance(event, TrialAdded):
            if event.trial.is_probe:
                self._probes_pending += 1
                self._probes.appendleft(event.trial)
            else:
                self._trials_pending += 1
                self._trials.appendleft(event.trial)
        elif isinstance(event, TrialStart):
            if event.trial.is_probe:
                self._probes_pending -= 1
                self._probes_running += 1
            else:
                self._trials_pending -= 1
                self._trials_running += 1
        elif isinstance(event, TrialComplete | TrialPruned):
            if event.trial.is_probe:
                self._probes_running -= 1
                self._probes_completed += 1
            else:
                self._trials_running -= 1
                self._trials_completed += 1
            self._completed_evals += 1
            self._total_cost += event.trial.cost
        elif isinstance(event, NewBestTrialFound):
            self._best_trial = event.trial
        elif isinstance(event, StudyEnd):
            self._result = event.result

        self._progress.update(self._progress_task_id, completed=self._completed_evals)

        if isinstance(event, TrialComplete) and event.trial.status == "failed":
            self._last_error = event.trial.error

    def _render_final_summary(self, result: StudyResult) -> None:
        """Renders a final, static summary of the study results."""
        self.console.print(
            Rule(f"[bold] {self.study.name}: Optimization Complete [/bold]", style="cyan")
        )
        self.console.print(
            Panel(format_study_result(result), border_style="dim", title="Study Summary")
        )

        if result.best_trial:
            best_trial = result.best_trial

            display_trial = best_trial
            if best_trial.transformed_input is not None:
                from copy import copy

                display_trial = copy(best_trial)
                display_trial.candidate = best_trial.transformed_input

            self.console.print(
                Panel(
                    format_trial(display_trial),
                    title="[bold magenta]Best Trial[/bold magenta]",
                    border_style="magenta",
                )
            )
        else:
            self.console.print(Panel("[yellow]No successful trials were completed.[/yellow]"))

    async def run(self) -> StudyResult:
        with Live(self._build_dashboard(), console=self.console) as live:
            async with self.study.stream() as stream:
                async for event in stream:
                    self._handle_event(event)
                    live.update(self._build_dashboard())

        if self._result:
            self._render_final_summary(self._result)
            return self._result

        raise RuntimeError("Optimization did not produce a final result.")

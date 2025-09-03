import typing as t
from collections import deque

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from dreadnode.optimization.events import (
    CandidatePruned,
    CandidatesSuggested,
    NewBestTrialFound,
    StepEnd,
    StudyEnd,
    StudyEvent,
    StudyStart,
    TrialComplete,
)
from dreadnode.optimization.result import StudyResult
from dreadnode.util import get_callable_name

if t.TYPE_CHECKING:
    from dreadnode.optimization.study import Study
    from dreadnode.optimization.trial import Trial

StudyT = t.TypeVar("StudyT", bound="Study")


class StudyConsoleAdapter(t.Generic[StudyT]):
    """Consumes a Study's event stream and renders a live progress dashboard."""

    def __init__(
        self,
        study: StudyT,
        *,
        console: Console | None = None,
        max_log_entries: int = 10,
    ):
        self.study = study
        self.console = console or Console()
        self.max_log_entries = max_log_entries
        self.final_result: StudyResult | None = None

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            expand=True,
        )
        self._trial_log: deque[Trial] = deque(maxlen=max_log_entries)
        self._best_trial_summary = Text("No successful trials yet.", style="dim")
        self._summary_stats = {
            "Successful": 0,
            "Failed": 0,
            "Pruned": 0,
            "Pending": 0,
        }
        self._steps_task_id: TaskID | None = None
        self._patience_task_id: TaskID | None = None
        self._trials_task_id: TaskID | None = None

    def _build_summary_table(self) -> Table:
        table = Table.grid(expand=True)
        table.add_column("Statistic", style="dim")
        table.add_column("Value")
        for stat, value in self._summary_stats.items():
            color = {
                "Successful": "green",
                "Failed": "red",
                "Pruned": "yellow",
                "Pending": "dim",
            }.get(stat)
            table.add_row(f"{stat}:", f"[{color}]{value}[/{color}]" if color else str(value))
        return table

    def _build_trial_log_table(self) -> Table:
        table = Table(title="Trial Log", expand=True)
        table.add_column("Step", justify="right", style="dim")
        table.add_column("Status")
        table.add_column("Score", justify="right")
        table.add_column("Candidate")

        # Iterating over reversed() to show oldest first, newest last
        for trial in reversed(self._trial_log):
            color = {
                "success": "green",
                "failed": "red",
                "pruned": "yellow",
                "pending": "dim",
            }.get(trial.status)
            score_str = f"{trial.score:.3f}" if trial.score is not None else "N/A"
            table.add_row(
                str(trial.step),
                f"[{color}]{trial.status.capitalize()}[/{color}]",
                score_str,
                str(trial.candidate),
            )
        return table

    def _build_dashboard(self) -> Panel:
        summary_panel = Panel(
            self._build_summary_table(),
            title="[bold]Summary[/bold]",
            border_style="dim",
            padding=(1, 2),
        )
        trials_panel = Panel(
            self._build_trial_log_table(),
            title="[bold]Live Trials[/bold]",
            border_style="dim",
            padding=(1, 2),
        )
        best_trial_panel = Panel(
            self._best_trial_summary,
            title="[bold green]Current Best[/bold green]",
            border_style="green",
            padding=(1, 2),
        )

        middle_row = Layout(name="middle_row", size=8)
        middle_row.split_row(
            Layout(best_trial_panel, name="best_trial", ratio=2),
            Layout(summary_panel, name="summary", ratio=1),
        )

        layout = Layout()
        layout.split_column(
            Layout(self._progress, name="progress", size=5),
            middle_row,
            Layout(trials_panel, name="trials_log", ratio=1, minimum_size=5),
        )

        name = self.study.name or get_callable_name(self.study.objective, short=True)

        return Panel(
            layout,
            title=Text(f"Optimizing '{name}'", justify="center", style="bold cyan"),
            border_style="cyan",
        )

    def _handle_event(self, event: StudyEvent) -> None:  # noqa: PLR0912
        if isinstance(event, StudyStart):
            self._steps_task_id = self._progress.add_task(
                "[bold]Overall Steps", total=self.study.max_steps
            )
            if self.study.patience:
                self._patience_task_id = self._progress.add_task(
                    "[dim]Steps Since Best", total=self.study.patience
                )
            self._trials_task_id = self._progress.add_task("Step Trials", total=0, visible=False)

        elif isinstance(event, CandidatesSuggested):
            self._summary_stats["Pending"] += len(event.candidates)
            if self._trials_task_id is not None:
                self._progress.update(
                    self._trials_task_id,
                    total=len(event.candidates),
                    completed=0,
                    visible=True,
                )
            for trial in event.trials[-len(event.candidates) :]:
                self._trial_log.appendleft(trial)

        elif isinstance(event, TrialComplete):
            self._summary_stats["Pending"] -= 1
            if event.trial.status == "success":
                self._summary_stats["Successful"] += 1
            elif event.trial.status == "failed":
                self._summary_stats["Failed"] += 1
            if self._trials_task_id is not None:
                self._progress.update(self._trials_task_id, advance=1)

        elif isinstance(event, CandidatePruned):
            self._summary_stats["Pending"] -= 1
            self._summary_stats["Pruned"] += 1
            if self._trials_task_id is not None:
                self._progress.update(self._trials_task_id, advance=1)

        elif isinstance(event, NewBestTrialFound):
            table = Table.grid(expand=True)
            table.add_column("Label", style="dim")
            table.add_column("Value")

            table.add_row("Score:", f"[bold green]{event.trial.score:.4f}[/bold green]")
            table.add_row("Candidate:", str(event.trial.candidate))
            table.add_row("Output:", str(event.trial.output))

            self._best_trial_summary = table
            if self._patience_task_id is not None:
                self._progress.reset(self._patience_task_id)

        elif isinstance(event, StepEnd):
            if self._steps_task_id is not None:
                self._progress.update(self._steps_task_id, advance=1)
            if self._patience_task_id is not None:
                if self.study._steps_since_best > 0:  # noqa: SLF001
                    self._progress.update(self._patience_task_id, advance=1)
                else:
                    self._progress.reset(self._patience_task_id)
            if self._trials_task_id is not None:
                self._progress.update(self._trials_task_id, visible=False)

        elif isinstance(event, StudyEnd):
            self._progress.stop()
            self.final_result = event.result

    def _render_final_summary(self, result: StudyResult) -> None:
        """Renders a final, static summary of the study results."""
        summary_table = Table.grid(expand=True)
        summary_table.add_column("Metric", style="dim")
        summary_table.add_column("Value")

        summary_table.add_row("Stop Reason:", f"[yellow]{result.stop_reason}[/yellow]")

        if result.best_trial:
            summary_table.add_row("-" * 20, "-" * 50)  # Separator
            summary_table.add_row(
                "Best Score:", f"[bold green]{result.best_trial.score:.4f}[/bold green]"
            )
            summary_table.add_row("Best Candidate:", str(result.best_trial.candidate))
            summary_table.add_row("Best Output:", str(result.best_trial.output))
        else:
            summary_table.add_row("Best Trial:", "No successful trials were completed.")

        panel = Panel(
            summary_table,
            title="[bold cyan]Optimization Complete[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
        self.console.print(panel)

    async def run(self) -> StudyResult:
        with Live(
            self._build_dashboard(),
            console=self.console,
            screen=True,
            transient=True,
        ) as live:
            async with self.study.stream() as stream:
                async for event in stream:
                    self._handle_event(event)
                    live.update(self._build_dashboard())

        if self.final_result:
            self._render_final_summary(self.final_result)
            return self.final_result

        raise RuntimeError("Optimization did not produce a final result.")

import typing as t
from collections import deque
from dataclasses import dataclass

from rich import box
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
    StepEnd,
    StepStart,
    StudyEnd,
    StudyEvent,
    StudyStart,
    TrialAdded,
    TrialComplete,
    TrialPruned,
    TrialStart,
)
from dreadnode.optimization.result import StudyResult

if t.TYPE_CHECKING:
    from dreadnode.optimization.study import Study
    from dreadnode.optimization.trial import Trial


@dataclass
class DashboardState:
    max_steps: int = 0
    steps_completed: int = 0
    steps_since_best: int = 0
    best_trial: "Trial | None" = None
    current_step_trials: int = 0
    is_step_running: bool = False


class StudyConsoleAdapter:
    """Consumes a Study's event stream and renders a live progress dashboard."""

    def __init__(
        self, study: "Study[t.Any]", *, console: Console | None = None, max_log_entries: int = 15
    ):
        self.study = study
        self.console = console or Console()
        self.final_result: StudyResult | None = None

        # The single source of truth for dynamic data
        self.state = DashboardState(max_steps=self.study.max_steps)
        self._trials: deque[Trial] = deque(maxlen=max_log_entries)

        # A single, unified progress bar object
        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            SpinnerColumn("dots"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True,
        )
        self._steps_task_id: TaskID = self._progress.add_task(
            "[bold]Overall Steps", total=self.study.max_steps
        )

    def _build_header(self) -> RenderableType:
        grid = Table.grid(expand=True)
        grid.add_column("Best Score", justify="left", ratio=1)
        grid.add_column("Patience", justify="center", ratio=1)
        grid.add_column("Status", justify="right", ratio=1)

        # Best Score
        best_score_text = Text("---", style="dim")
        if self.state.best_trial:
            best_score_text = Text(f"{self.state.best_trial.score:.4f}", style="bold magenta")

        # Patience
        patience_text = Text(
            f"Waiting for improvement... ({self.state.steps_since_best} steps)", style="dim"
        )
        if self.state.steps_since_best == 0 and self.state.best_trial:
            patience_text = Text("New best found", style="magenta")

        # Status
        status_text = Text("Initializing ...", style="dim")
        if self.state.is_step_running:
            status_text = Text.from_markup(
                f"Running step [bold]{self.state.steps_completed + 1}[/bold] ([bold]{self.state.current_step_trials}[/bold] trials)",
                style="cyan",
            )
        elif self.state.steps_completed > 0:
            status_text = Text(
                f"Step {self.state.steps_completed} complete. Waiting ...", style="dim"
            )

        grid.add_row(
            Text.from_markup(f"[b]Best Score:[/b] {best_score_text}"),
            patience_text,
            status_text,
        )
        return grid

    def _build_best_trial_panel(self) -> RenderableType:
        if not self.state.best_trial:
            return Panel(
                Text("No successful trials yet.", style="dim", justify="center"),
                title="[bold magenta]Current Best[/bold magenta]",
                border_style="dim",
            )

        trial = self.state.best_trial

        scores_table = Table.grid(padding=(0, 2))
        scores_table.add_column("Name")
        scores_table.add_column("Score", justify="right", min_width=8)
        for name in self.study.objective_names:
            scores_table.add_row(
                name, f"[bold magenta]{trial.scores.get(name, -float('inf')):.3f}[/bold magenta]"
            )

        for name, value in trial.all_scores.items():
            if name not in self.study.objective_names:
                scores_table.add_row(f"[dim]{name}[/dim]", f"[dim]{value:.3f}[/dim]")

        # Main content grid
        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_row(Panel(scores_table, title="Scores", title_align="left"))
        grid.add_row(
            Panel(
                Text(str(trial.candidate), style="dim"),
                title="Candidate",
                title_align="left",
            )
        )
        if trial.output:
            grid.add_row(
                Panel(Text(str(trial.output), style="dim"), title="Output", title_align="left")
            )

        return Panel(
            grid, title="[bold magenta]Current Best[/bold magenta]", border_style="magenta"
        )

    def _build_trials_panel(self) -> RenderableType:
        table = Table(expand=True, box=box.ROUNDED)
        table.add_column("ID", style="dim", width=8)
        table.add_column("Step", justify="right", style="dim", width=4)
        table.add_column("Status")
        table.add_column("Score", justify="right")

        for trial in self._trials:
            color = {
                "finished": "default",
                "failed": "red",
                "pruned": "yellow",
                "running": "cyan",
            }.get(trial.status, "dim")
            status_text = f"[{color}]{trial.status}[/{color}]"
            score_str = f"{trial.score:.3f}" if trial.status == "finished" else "..."
            table.add_row(str(trial.id)[16:], str(trial.step), status_text, score_str)

        return Panel(
            table if self._trials else Text("No trials yet.", style="dim", justify="center"),
            title="[bold]Trials[/bold]",
            border_style="dim",
        )

    def _build_dashboard(self) -> RenderableType:
        layout = Layout()

        layout.split_column(
            Layout(Padding(self._build_header(), 1), size=3),
            Layout(Rule(style="cyan"), size=1),
            Layout(name="body"),
            Layout(Padding(self._progress, 1), size=3),
        )

        layout["body"].split_row(
            Layout(self._build_best_trial_panel(), ratio=2),
            Layout(self._build_trials_panel(), ratio=1),
        )

        return Layout(
            Panel(
                layout,
                title=Text(self.study.name, justify="center", style="bold cyan"),
                border_style="cyan",
            )
        )

    def _handle_event(self, event: StudyEvent[t.Any]) -> None:
        if isinstance(event, StudyStart):
            self.state = DashboardState(max_steps=self.study.max_steps)
        elif isinstance(event, StepStart):
            self.state.is_step_running = True
            self.state.current_step_trials = 0
        elif isinstance(event, TrialAdded):
            self._trials.appendleft(event.trial)
            self.state.current_step_trials += 1
        elif isinstance(event, TrialStart | TrialComplete | TrialPruned):
            for i, t in enumerate(self._trials):
                if t.id == event.trial.id:
                    self._trials[i] = event.trial
                    break
            else:
                self._trials.appendleft(event.trial)
        elif isinstance(event, NewBestTrialFound):
            self.state.best_trial = event.trial
            self.state.steps_since_best = 0
        elif isinstance(event, StepEnd):
            self.state.is_step_running = False
            self.state.steps_completed += 1
            if self.state.best_trial:
                self.state.steps_since_best += 1
            self._progress.update(self._steps_task_id, completed=self.state.steps_completed)
        elif isinstance(event, StudyEnd):
            self.final_result = event.result

    def _render_final_summary(self, result: StudyResult) -> None:
        """Renders a final, static summary of the study results."""
        self.console.print(
            Rule(f"[bold] {self.study.name}: Optimization Complete [/bold]", style="cyan")
        )
        summary_table = Table.grid(padding=(0, 2))
        summary_table.add_column("Metric", style="dim")
        summary_table.add_column("Value")
        summary_table.add_row("Stop Reason:", f"[bold]{result.stop_reason}[/bold]")
        summary_table.add_row("Explanation:", result.stop_explanation or "-")
        summary_table.add_row("Steps Taken:", str(result.steps_taken))
        summary_table.add_row("Total Trials:", str(len(result.trials)))

        panel = Panel(summary_table, border_style="dim", title="Study Summary")
        self.console.print(panel)

        if result.best_trial:
            self.console.print(self._build_best_trial_panel())
        else:
            self.console.print(Panel("[yellow]No successful trials were completed.[/yellow]"))

    async def run(self) -> StudyResult:
        with Live(console=self.console, get_renderable=self._build_dashboard, screen=True):
            async with self.study.stream() as stream:
                async for event in stream:
                    self._handle_event(event)

        if self.final_result:
            self._render_final_summary(self.final_result)
            return self.final_result

        raise RuntimeError("Optimization did not produce a final result.")

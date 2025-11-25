import typing as t
from collections import deque
from datetime import datetime

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from dreadnode.eval.events import (
    EvalEnd,
    EvalEvent,
    EvalStart,
    SampleComplete,
    ScenarioEnd,
    ScenarioStart,
)
from dreadnode.eval.format import format_eval_result
from dreadnode.eval.result import EvalResult
from dreadnode.util import format_dict

if t.TYPE_CHECKING:
    from dreadnode.eval import Eval


class EvalConsoleAdapter:
    """
    Consumes an Eval's event stream and renders a live progress dashboard.
    """

    def __init__(
        self,
        eval: "Eval",
        *,
        console: Console | None = None,
        max_events_to_show: int = 10,
    ):
        self.eval = eval
        self.console = console or Console()
        self.final_result: EvalResult | None = None
        self.max_events_to_show = max_events_to_show
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            "•",
            TimeRemainingColumn(),
            expand=True,
        )
        self._event_log: deque[str] = deque(maxlen=max_events_to_show)
        self._total_task_id: TaskID | None = None
        self._scenario_task_id: TaskID | None = None
        self._iteration_task_id: TaskID | None = None
        self._total_samples_processed = 0
        self._passed_count = 0
        self._failure_count = 0
        self._assert_count = 0

    def _build_summary_table(self) -> Table:
        success_rate = (
            f"{self._passed_count / self._total_samples_processed:.1%}"
            if self._total_samples_processed > 0
            else "N/A"
        )
        table = Table.grid(expand=True)
        table.add_column("Statistic", style="dim")
        table.add_column("Value")
        table.add_row("Success Rate:", success_rate)
        table.add_row("Total Samples:", str(self._total_samples_processed))
        table.add_row("  Passed:", f"[green]{self._passed_count}[/green]")
        table.add_row("  Failed:", f"[yellow]{self._failure_count}[/yellow]")
        table.add_row("  Errors:", f"[red]{self._assert_count}[/red]")
        return table

    def _build_dashboard(self) -> Panel:
        events_panel = Panel(
            "\n".join(self._event_log),
            title="[dim]Events[/dim]",
            border_style="dim",
        )
        stats_panel = Panel(
            self._build_summary_table(),
            title="[dim]Summary[/dim]",
            border_style="dim",
        )

        layout = Layout()

        # Split into top (progress) and bottom sections
        layout.split_column(
            Layout(Padding(self._progress, (1, 0, 0, 0)), name="progress", size=4),
            Layout(name="bottom"),
        )

        # Split the bottom section: 2/3 events, 1/3 stats
        layout["bottom"].split_row(
            Layout(events_panel, ratio=2),
            Layout(stats_panel, ratio=1),
        )

        eval_name = (
            self.eval.name or self.eval.task
            if isinstance(self.eval.task, str)
            else self.eval.task.name
        )

        return Panel(
            layout,
            title=Text(
                f"Evaluating '{eval_name}'",
                justify="center",
                style="bold",
            ),
            border_style="cyan",
            height=self.max_events_to_show + 10,
        )

    def _log_event(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")  # noqa: DTZ005
        self._event_log.append(f"[dim]{timestamp}[/dim] {message}")

    def _handle_event(self, event: EvalEvent) -> None:  # noqa: PLR0912
        """Mutates the adapter's state based on an incoming event."""
        if isinstance(event, EvalStart):
            self._log_event("Evaluation started.")
            self._total_task_id = self._progress.add_task(
                "[bold]Total Progress", total=event.total_samples
            )
            self._scenario_task_id = self._progress.add_task(
                "Scenarios", total=event.scenario_count, visible=False
            )
        elif isinstance(event, ScenarioStart):
            params_str = format_dict(event.scenario_params)
            self._log_event(f"Running scenario: [bold cyan]{params_str}[/bold cyan]")
            self._iteration_task_id = self._progress.add_task(
                f"  Scenario ({params_str})", total=event.sample_count
            )
        elif isinstance(event, SampleComplete):
            self._total_samples_processed += 1
            if event.sample.failed:
                if event.sample.error:
                    self._failure_count += 1
                    error_short = str(event.sample.error).split("\n")[0]
                    self._log_event(f"[red]ERROR[/red] Sample failed: {error_short[:80]}")
                else:
                    self._assert_count += 1
            else:
                self._passed_count += 1
            if self._total_task_id is not None:
                self._progress.update(self._total_task_id, advance=1)
            if self._iteration_task_id is not None:
                self._progress.update(self._iteration_task_id, advance=1)
        elif isinstance(event, ScenarioEnd):
            params_str = format_dict(event.result.params)
            self._log_event(f"Scenario complete: [bold cyan]{params_str}[/bold cyan]")
            if self._iteration_task_id is not None:
                self._progress.remove_task(self._iteration_task_id)
            if self._scenario_task_id is not None:
                self._progress.update(self._scenario_task_id, advance=1)
        elif isinstance(event, EvalEnd):
            self._progress.stop()
            self._log_event(f"[bold]Evaluation complete: {event.result.stop_reason}[/bold]")
            self.final_result = event.result

    async def run(self) -> EvalResult:
        """Runs the evaluation and renders the console interface."""
        with Live(self._build_dashboard(), console=self.console) as live:
            async with self.eval.stream() as stream:
                async for event in stream:
                    self._handle_event(event)
                    live.update(self._build_dashboard(), refresh=True)

        if self.final_result:
            self.console.print(format_eval_result(self.final_result))
            return self.final_result

        raise RuntimeError("Evaluation did not produce a final result.")

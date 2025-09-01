import typing as t
from collections import deque
from datetime import datetime

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from dreadnode.eval.eval import In, Out
from dreadnode.eval.events import (
    EvalEnd,
    EvalEvent,
    EvalStart,
    SampleComplete,
    ScenarioEnd,
    ScenarioStart,
)
from dreadnode.eval.result import EvalResult
from dreadnode.util import format_dict

if t.TYPE_CHECKING:
    from dreadnode.eval import Eval

# Type variable for the generic Eval object
EvalT = t.TypeVar("EvalT", bound="Eval")


class EvalConsoleAdapter(t.Generic[In, Out]):
    """
    Consumes an Eval's event stream and renders a live progress dashboard.
    """

    def __init__(
        self,
        eval_instance: EvalT,
        *,
        console: Console | None = None,
        max_events_to_show: int = 10,
    ):
        self.eval = eval_instance
        self.console = console or Console()
        self.final_result: EvalResult | None = None
        self.max_events_to_show = max_events_to_show
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            "•",
            TimeRemainingColumn(),
        )
        self._event_log = deque(maxlen=max_events_to_show)
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
        table = Table.grid(expand=True, padding=(0, 2))
        table.add_column("Statistic", style="dim", justify="right")
        table.add_column("Value")
        table.add_row("Success Rate:", success_rate)
        table.add_row("Total Samples:", str(self._total_samples_processed))
        table.add_row("  Passed:", f"[green]{self._passed_count}[/green]")
        table.add_row("  Failed:", f"[red]{self._failure_count}[/red]")
        table.add_row("  Errors:", f"[yellow]{self._assert_count}[/yellow]")
        return table

    def _build_dashboard(self) -> Panel:
        table = Table.grid(expand=True)
        table.add_row(self._progress)
        table.add_row(self._build_summary_table())
        (
            table.add_row(
                Panel(
                    "\n".join(self._event_log),
                    title="[dim]Events[/dim]",
                    border_style="dim",
                    height=self.max_events_to_show + 2,
                )
            ),
        )
        return Panel(
            table,
            title=Text(
                f"Evaluating '{self.eval.name or self.eval.task.name}'",
                justify="center",
                style="bold",
            ),
            border_style="cyan",
        )

    def _log_event(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")  # noqa: DTZ005
        self._event_log.append(f"[dim]{timestamp}[/dim] {message}")

    def _handle_event(self, event: EvalEvent):
        """Mutates the adapter's state based on an incoming event."""
        if isinstance(event, EvalStart):
            self._log_event("Evaluation started.")
            total_samples = event.total_iterations * len(self.eval.dataset)
            self._total_task_id = self._progress.add_task(
                "[bold]Total Progress", total=total_samples
            )
            self._scenario_task_id = self._progress.add_task(
                "Scenarios", total=event.scenario_count, visible=False
            )
        elif isinstance(event, ScenarioStart):
            params_str = format_dict(event.scenario_params)
            self._log_event(f"Running scenario: [bold cyan]{params_str}[/bold cyan]")
            total_samples_in_scenario = event.iteration_count * len(self.eval.dataset)
            self._iteration_task_id = self._progress.add_task(
                f"  Scenario ({params_str})", total=total_samples_in_scenario
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
            self._progress.update(self._total_task_id, advance=1)
            if self._iteration_task_id is not None:
                self._progress.update(self._iteration_task_id, advance=1)
        elif isinstance(event, ScenarioEnd):
            self._log_event("Scenario complete.")
            self._progress.remove_task(self._iteration_task_id)
            self._progress.update(self._scenario_task_id, advance=1)
        elif isinstance(event, EvalEnd):
            self._progress.stop()
            self._log_event("[bold green]Evaluation complete.[/bold green]")
            self.final_result = event.result

    async def run(self) -> EvalResult:
        """Runs the evaluation and renders the console interface."""
        with Live(self._build_dashboard(), console=self.console):
            async with self.eval.stream() as stream:
                async for event in stream:
                    self._handle_event(event)

        if self.final_result:
            self.console.print(self.final_result)
            return self.final_result

        raise RuntimeError("Evaluation did not produce a final result.")


async def console(eval_instance: EvalT) -> EvalResult:
    """Convenience wrapper to run an eval with a console adapter."""
    adapter = EvalConsoleAdapter(eval_instance)
    return await adapter.run()

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from dreadnode.agent.events import AgentEvent, ToolEnd, ToolStart


class AgentConsoleRenderer:
    """
    Renders an agent's event stream to the console in real-time.

    This class manages stateful UI elements, primarily a "status board"
    for displaying concurrent tool calls using rich.Live.
    """

    def __init__(self, console: Console):
        self.console = console
        self._live_status: Live | None = None
        self._status_table: Table | None = None
        self._active_tool_ids: set[str] = set()

    def render(self, event: AgentEvent) -> None:
        """
        Renders a single event to the console.

        This method acts as a dispatcher. It handles stateful events like
        ToolStart and ToolEnd itself, and delegates static events to be
        printed, relying on their __rich_console__ implementation.
        """
        if isinstance(event, ToolStart):
            self._handle_tool_start(event)
        elif isinstance(event, ToolEnd):
            self._handle_tool_end(event)
        else:
            # For all other events, print their rich representation directly.
            self.console.print(event)

    def _handle_tool_start(self, event: ToolStart) -> None:
        """Adds a tool to the live status board."""
        self._active_tool_ids.add(event.tool_call.id)

        # If this is the first active tool, create and start the Live display.
        if self._live_status is None:
            self._status_table = Table.grid(padding=(0, 2), expand=True)
            self._status_table.add_column()
            self._live_status = Live(
                self._status_table, console=self.console, transient=True, auto_refresh=True
            )
            self._live_status.start()
        if self._status_table is not None:
            # Add a new row for the running tool with a spinner.
            self._status_table.add_row(
                Text(f"Running [bold]{event.tool_call.name}[/bold]...", style="yellow")
            )

    def _handle_tool_end(self, event: ToolEnd) -> None:
        """Prints the tool's result and cleans up the status board."""
        # First, print the static result panel. This ensures it's in the
        # console history even after the live display is gone.
        self.console.print(event)

        # Remove the tool from the active set.
        self._active_tool_ids.discard(event.tool_call.id)

        # If all tools are finished, stop the Live display.
        if not self._active_tool_ids and self._live_status:
            self._live_status.stop()
            self._live_status = None
            self._status_table = None

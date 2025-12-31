"""
Logging utilities using loguru with rich formatting.

Usage:
    from log import logger, console, configure_logging

    configure_logging("debug")  # optional, defaults to "info"

    logger.info("Hello")
    logger.success("Done!")
"""

import os
import pathlib
import typing as t

from loguru import logger
from rich.console import Console
from rich.prompt import Confirm
from rich.theme import Theme

LogLevel = t.Literal["trace", "debug", "info", "success", "warning", "error", "critical"]

console = Console(
    theme=Theme(
        {
            "logging.level.success": "green",
            "logging.level.trace": "dim blue",
        }
    )
)

# In vscode jupyter, disable rich's jupyter detection to avoid issues with styling
if "VSCODE_PID" in os.environ:
    console.is_jupyter = False


def _rich_sink(message: str) -> None:
    """Sink that writes to rich console."""
    console.print(message, end="")


def configure_logging(
    level: LogLevel = "info",
    log_file: pathlib.Path | None = None,
    log_file_level: LogLevel = "debug",
) -> None:
    """
    Configure loguru with rich console output.

    Args:
        level: Console log level.
        log_file: Optional file path for logging.
        log_file_level: Log level for file output.
    """
    logger.remove()
    logger.enable("dreadnode")

    # Rich-formatted console output
    logger.add(
        _rich_sink,
        format="<level>{level.icon}</level> {message}",
        level=level.upper(),
        colorize=True,
    )

    if log_file is not None:
        logger.add(log_file, level=log_file_level.upper())
        logger.info(f"Logging to {log_file}")


def confirm(action: str) -> bool:
    """Prompt user for confirmation."""
    return Confirm.ask(
        f"[bold magenta]↔[/] {action}",
        default=False,
        case_sensitive=False,
        console=console,
    )


# Configure with defaults on import
configure_logging()

"""
We use loguru for logging. This module provides a function to configure logging handlers.

To just enable dreadnode logs to flow, call `logger.enable("dreadnode")` after importing the module.
"""

import os
import pathlib
import typing as t
from textwrap import dedent

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler
from rich.prompt import Confirm
from rich.theme import Theme

LogLevel = t.Literal["trace", "debug", "info", "success", "warning", "error", "critical"]
"""Valid logging levels."""

console = Console(
    theme=Theme(  # rich doesn't include default colors for these
        {
            "logging.level.success": "green",
            "logging.level.trace": "dim blue",
        }
    )
)

# In vscode jupyter, disable rich's jupyter detection to avoid issues with styling
if "VSCODE_PID" in os.environ:
    console.is_jupyter = False


def configure_logging(
    log_level: LogLevel = "info",
    log_file: pathlib.Path | None = None,
    log_file_level: LogLevel = "debug",
) -> None:
    """
    Configures common loguru handlers.

    Args:
        log_level: The desired log level.
        log_file: The path to the log file. If None, logging
            will only be done to the console.
        log_file_level: The log level for the log file.
    """
    logger.enable("dreadnode")

    logger.remove()
    logger.add(
        RichHandler(console=console, log_time_format="%X", rich_tracebacks=True),
        format=lambda _: "{message}",
        level=log_level.upper(),
    )

    if log_file is not None:
        logger.add(log_file, level=log_file_level.upper())
        logger.info(f"Logging to {log_file}")


def print_success(message: str) -> None:
    console.print(f"[bold green]:heavy_check_mark:[/] {dedent(message)}")


def print_error(message: str) -> None:
    console.print(f"[bold red]:heavy_multiplication_x:[/] {dedent(message)}")


def print_warning(message: str) -> None:
    console.print(f"[bold yellow]:warning:[/] {dedent(message)}")


def print_info(message: str) -> None:
    console.print(dedent(message))


def confirm(action: str) -> bool:
    return Confirm.ask(
        f"[bold magenta]:left-right_arrow:[/] {action}",
        default=False,
        case_sensitive=False,
        console=console,
    )

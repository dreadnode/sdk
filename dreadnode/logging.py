"""
We use loguru for logging. This module provides a function to configure logging handlers.

To just enable dreadnode logs to flow, call `logger.enable("dreadnode")` after importing the module.
"""

import pathlib
import sys
import typing as t

from loguru import logger

if t.TYPE_CHECKING:
    from loguru import Record as LogRecord

g_configured: bool = False

LogLevelList = ["trace", "debug", "info", "success", "warning", "error", "critical"]
LogLevelLiteral = t.Literal["trace", "debug", "info", "success", "warning", "error", "critical"]
"""Valid logging levels."""


def log_formatter(record: "LogRecord") -> str:
    return "".join(
        (
            "<green>{time:HH:mm:ss.SSS}</green> | ",
            "<dim>{extra[prefix]}</dim> " if record["extra"].get("prefix") else "",
            "<level>{message}</level>\n",
        )
    )


def configure_logging(
    log_level: LogLevelLiteral = "info",
    log_file: pathlib.Path | None = None,
    log_file_level: LogLevelLiteral = "debug",
) -> None:
    """
    Configures common loguru handlers.

    Args:
        log_level: The desired log level.
        log_file: The path to the log file. If None, logging
            will only be done to the console.
        log_file_level: The log level for the log file.
    """
    global g_configured  # noqa: PLW0603

    if g_configured:
        return

    logger.enable("dreadnode")

    logger.level("TRACE", color="<magenta>", icon="[T]")
    logger.level("DEBUG", color="<blue>", icon="[_]")
    logger.level("INFO", color="<cyan>", icon="[=]")
    logger.level("SUCCESS", color="<green>", icon="[+]")
    logger.level("WARNING", color="<yellow>", icon="[-]")
    logger.level("ERROR", color="<red>", icon="[!]")
    logger.level("CRITICAL", color="<RED>", icon="[x]")

    logger.remove()
    logger.add(sys.stderr, format=log_formatter, level=log_level.upper())

    if log_file is not None:
        logger.add(log_file, format=log_formatter, level=log_file_level.upper())
        logger.info(f"Logging to {log_file}")

    g_configured = True

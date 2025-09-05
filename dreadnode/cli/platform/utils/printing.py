import sys

import rich


def print_success(message: str, prefix: str | None = None) -> None:
    """Print success message in green"""
    prefix = prefix or "✓"
    rich.print(f"[bold green]{prefix}[/] [green]{message}[/]")


def print_error(message: str, prefix: str | None = None) -> None:
    """Print error message in red"""
    prefix = prefix or "✗"
    rich.print(f"[bold red]{prefix}[/] [red]{message}[/]", file=sys.stderr)


def print_warning(message: str, prefix: str | None = None) -> None:
    """Print warning message in yellow"""
    prefix = prefix or "⚠"
    rich.print(f"[bold yellow]{prefix}[/] [yellow]{message}[/]")


def print_info(message: str, prefix: str | None = None) -> None:
    """Print info message in blue"""
    prefix = prefix or "i"
    rich.print(f"[bold blue]{prefix}[/] [blue]{message}[/]")


def print_debug(message: str, prefix: str | None = None) -> None:
    """Print debug message in dim gray"""
    prefix = prefix or "🐛"
    rich.print(f"[dim]{prefix}[/] [dim]{message}[/]")


def print_heading(message: str) -> None:
    """Print section heading"""
    rich.print(f"\n[bold underline]{message}[/]\n")


def print_muted(message: str) -> None:
    """Print muted text"""
    rich.print(f"[dim]{message}[/]")

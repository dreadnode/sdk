"""Datasets CLI for managing and viewing datasets."""

import typing as t

import cyclopts
import rich
from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from dreadnode.core.log import console, logger

# Note: datasets don't use the discoverable pattern since they are package-based, not file-based

dataset_cli = cyclopts.App("dataset", help="Manage and view datasets.")


@dataset_cli.command(name=["list", "ls", "show"])
def list_datasets() -> None:
    """
    List all installed datasets.
    """
    from dreadnode.datasets.dataset import Dataset

    names = Dataset.list()
    if not names:
        logger.info("No datasets installed.")
        logger.info("Use `dreadnode pull <dataset>` to install a dataset.")
        return

    table = Table(box=box.ROUNDED)
    table.add_column("Name", style="orange_red1", no_wrap=True)
    table.add_column("Version", style="cyan")
    table.add_column("Description", min_width=20)
    table.add_column("Format", style="magenta")
    table.add_column("Rows", style="green", justify="right")

    for name in names:
        try:
            ds = Dataset(name)
            table.add_row(
                ds.name,
                ds.version,
                ds.description or "-",
                ds.format or "-",
                str(ds.row_count) if ds.row_count else "-",
            )
        except Exception as e:
            table.add_row(name, "-", f"[red]Error: {e}[/]", "-", "-")

    console.print(table)


@dataset_cli.command()
def info(
    name: t.Annotated[str, cyclopts.Parameter(help="Dataset name to show info for")],
) -> None:
    """
    Show detailed information about a dataset.
    """
    from dreadnode.datasets.dataset import Dataset

    try:
        ds = Dataset(name)
    except KeyError:
        logger.error(f"Dataset '{name}' not found.")
        logger.info("Use `dreadnode dataset list` to see installed datasets.")
        return

    details = Table(
        box=box.MINIMAL,
        show_header=False,
        style="orange_red1",
    )
    details.add_column("Property", style="bold dim", justify="right", no_wrap=True)
    details.add_column("Value", style="white")

    details.add_row(Text("Name", justify="right"), ds.name)
    details.add_row(Text("Version", justify="right"), ds.version)
    details.add_row(Text("Description", justify="right"), ds.description or "-")
    details.add_row(Text("Format", justify="right"), ds.format or "-")

    if ds.row_count:
        details.add_row(Text("Rows", justify="right"), str(ds.row_count))

    if ds.schema:
        schema_str = ", ".join(f"[cyan]{k}[/]: {v}" for k, v in ds.schema.items())
        details.add_row(Text("Schema", justify="right"), schema_str)

    if ds.files:
        files_str = ", ".join(f"[green]{f}[/]" for f in ds.files)
        details.add_row(Text("Files", justify="right"), files_str)

    # Verification status
    try:
        verified = ds.verify()
        status = "[green]Verified[/]" if verified else "[red]Failed[/]"
        details.add_row(Text("Integrity", justify="right"), status)
    except Exception:
        details.add_row(Text("Integrity", justify="right"), "[dim]Unknown[/]")

    console.print(Panel(
        details,
        title=f"[bold]{ds.name}[/]",
        title_align="left",
        border_style="orange_red1",
    ))


@dataset_cli.command()
def preview(
    name: t.Annotated[str, cyclopts.Parameter(help="Dataset name to preview")],
    *,
    split: t.Annotated[str | None, cyclopts.Parameter(help="Split to load (if applicable)")] = None,
    rows: t.Annotated[int, cyclopts.Parameter(help="Number of rows to show")] = 10,
) -> None:
    """
    Preview the contents of a dataset.
    """
    from dreadnode.datasets.dataset import Dataset

    try:
        ds = Dataset(name)
    except KeyError:
        logger.error(f"Dataset '{name}' not found.")
        return

    try:
        table = ds.load(split=split)
        df = table.slice(0, rows).to_pandas()
        rich.print(df.to_string())
        rich.print(f"\n[dim]Showing {min(rows, len(table))} of {len(table)} rows[/]")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")

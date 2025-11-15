import cyclopts
from rich import box
from rich.table import Table

from dreadnode.cli.api import create_api_client
from dreadnode.logging_ import console

cli = cyclopts.App("organizations", help="View and manage organizations.", help_flags=[])


@cli.command(name=["list", "ls", "show"])
def show() -> None:
    # get the client and call the list organizations endpoint
    client = create_api_client()
    organizations = client.list_organizations()

    table = Table(box=box.ROUNDED)
    table.add_column("Name", style="orange_red1")
    table.add_column("Key", style="green")
    table.add_column("ID")

    for org in organizations:
        table.add_row(
            org.name,
            org.key,
            str(org.id),
        )

    console.print(table)

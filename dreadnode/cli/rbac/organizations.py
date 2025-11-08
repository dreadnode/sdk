import cyclopts

from dreadnode.cli.api import create_api_client
from dreadnode.logging_ import print_info

cli = cyclopts.App("organizations", help="View and manage organizations.", help_flags=[])


@cli.command(name=["list", "ls", "show"])
def show() -> None:
    # get the client and call the list organizations endpoint
    client = create_api_client()
    organizations = client.list_organizations()
    for org in organizations:
        print_info(f"- {org.name} (ID: {org.id})")

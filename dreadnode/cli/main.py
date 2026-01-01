import contextlib
import importlib.metadata
import platform
import sys
import typing as t
import webbrowser

import cyclopts
import rich
from click import confirm
from rich import box
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from dreadnode.agents.cli import agent_cli
from dreadnode.airt.cli import airt_cli
from dreadnode.core.api import Token
from dreadnode.core.api.client import ApiClient, create_api_client
from dreadnode.core.api.models import Organization, Workspace, WorkspaceFilter
from dreadnode.core.api.session import ServerConfig, UserConfig
from dreadnode.core.log import console, logger
from dreadnode.core.util import create_key_from_name, time_to
from dreadnode.datasets.cli import dataset_cli
from dreadnode.evaluations.cli import evaluation_cli
from dreadnode.platform.cli import platform_cli
from dreadnode.core.settings import DEBUG, PLATFORM_BASE_URL
from dreadnode.studies.cli import study_cli

cli = cyclopts.App(
    help="Interact with Dreadnode platforms",
    version_flags=[],
    help_on_error=True,
    console=console,
)

cli["--help"].group = "Meta"

cli.command(agent_cli)
cli.command(airt_cli)
cli.command(dataset_cli)
cli.command(evaluation_cli)
cli.command(platform_cli)
cli.command(study_cli)

# user_cli is defined below after the helper commands
# It gets registered at the end of this module


@cli.meta.default
def meta(
    *tokens: t.Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
) -> None:
    try:
        console.print()
        cli(tokens)
    except Exception as e:
        if DEBUG:
            raise

        logger.exception("Unhandled exception")

        rich.print()
        rich.print(
            Panel(str(e), title=e.__class__.__name__, title_align="left", border_style="red")
        )
        sys.exit(1)


@cli.command(group="Auth")
def login(
    *,
    server: t.Annotated[str | None, cyclopts.Parameter(name=["--server", "-s"])] = None,
    profile: t.Annotated[str | None, cyclopts.Parameter(name=["--profile", "-p"])] = None,
) -> None:
    """
    Authenticate to a Dreadnode platform server and save the profile.

    Args:
        server: The server URL to authenticate against.
        profile: The profile name to save the server configuration under.
    """
    if not server:
        server = PLATFORM_BASE_URL
        with contextlib.suppress(Exception):
            existing_config = UserConfig.read().get_server_config(profile)
            server = existing_config.url

    client = ApiClient(base_url=server)

    logger.info("Requesting device code ...")

    codes = client.get_device_codes()

    verification_url = client.url_for_user_code(codes.user_code)
    verification_url_base = verification_url.split("?")[0]

    logger.info(
        f"""
        Attempting to automatically open the authorization page in your default browser.
        If the browser does not open or you wish to use a different device, open the following URL:

        [bold]{verification_url_base}[/]

        Then enter the code: [bold]{codes.user_code}[/]
        """
    )

    webbrowser.open(verification_url)

    tokens = client.poll_for_token(codes.device_code)

    client = ApiClient(
        server,
        cookies={
            "refresh_token": tokens.refresh_token,
            "access_token": tokens.access_token,
        },
    )
    user = client.get_user()

    user_config = UserConfig.read()
    user_config.set_server_config(
        ServerConfig(
            url=server,
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            email=user.email_address,
            username=user.username,
            api_key=user.api_key.key,
        ),
        profile,
    )
    user_config.active = profile
    user_config.write()

    logger.success(f"Authenticated as {user.email_address} ({user.username})")


@cli.command(group="Auth")
def refresh() -> None:
    """Refresh the active server profile with the latest user data."""

    user_config = UserConfig.read()
    server_config = user_config.get_server_config()

    client = create_api_client()
    user = client.get_user()

    server_config.email = user.email_address
    server_config.username = user.username
    server_config.api_key = user.api_key.key

    user_config.set_server_config(server_config).write()

    logger.success(
        f"Refreshed '[bold]{user_config.active}[/bold]' ([magenta]{user.email_address}[/] / [cyan]{user.username}[/])"
    )


@cli.command(help="Show versions and exit.", group="Meta")
def version() -> None:
    ver = importlib.metadata.version("dreadnode")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    os_name = platform.system()
    arch = platform.machine()
    logger.info(f"Platform:   {os_name} ({arch})")
    logger.info(f"Python:     {python_version}")
    logger.info(f"Dreadnode:  {ver}")


@cli.command()
def workspaces(
    organization: str | None = None,
) -> None:
    client = create_api_client()
    matched_organization: Organization
    if organization:
        matched_organization = client.get_organization(organization)
        if not matched_organization:
            logger.info(f"Organization '{organization}' not found.")
            return
    else:
        user_organizations = client.list_organizations()
        if len(user_organizations) == 0:
            logger.info("No organizations found.")
            return
        if len(user_organizations) > 1:
            logger.error(
                "Multiple organizations found. Please specify an organization to list workspaces from."
            )
            return
        matched_organization = user_organizations[0]

    workspace_filter = WorkspaceFilter(org_id=matched_organization.id)
    workspaces = client.list_workspaces(filters=workspace_filter)

    table = Table(box=box.ROUNDED)
    table.add_column("Name", style="orange_red1")
    table.add_column("Key", style="green")
    table.add_column("ID")
    table.add_column("dn.configure() Command", style="cyan")

    _print_workspace_table(workspaces, matched_organization)


@cli.command()
def create_workspace(
    name: str,
    key: str | None = None,
    description: str | None = None,
    organization: str | None = None,
) -> None:
    # get the client and call the create workspace endpoint
    client = create_api_client()
    if not key:
        key = create_key_from_name(name)

    if organization:
        matched_organization = client.get_organization(organization)
        if not matched_organization:
            logger.info(f"Organization '{organization}' not found.")
            return
    else:
        user_organizations = client.list_organizations()
        if len(user_organizations) == 0:
            logger.info("No organizations found. Please specify an organization.")
            return
        if len(user_organizations) > 1:
            logger.error(
                "Multiple organizations found. Please specify an organization to create the workspace in."
            )
            return
        matched_organization = user_organizations[0]
        logger.info(
            f"Workspace '{name}' ([cyan]{key}[/cyan]) will be created in organization '{matched_organization.name}'"
        )
        # verify with the user
        if not confirm("Do you want to continue?"):
            logger.info("Workspace creation cancelled.")
            return

    workspace: Workspace = client.create_workspace(
        name=name,
        key=key,
        organization_id=matched_organization.id,
        description=description,
    )
    _print_workspace_table([workspace], matched_organization)


@cli.command()
def delete_workspace(
    workspace: str,
    organization: str | None = None,
) -> None:
    client = create_api_client()
    if organization:
        matched_organization = client.get_organization(organization)
        if not matched_organization:
            logger.info(f"Organization '{organization}' not found.")
            return
    else:
        user_organizations = client.list_organizations()
        if len(user_organizations) == 0:
            logger.info("No organizations found. Please specify an organization.")
            return
        if len(user_organizations) > 1:
            logger.error(
                "Multiple organizations found. Please specify an organization to delete the workspace from."
            )
            return
        matched_organization = user_organizations[0]

    matched_workspace = client.get_workspace(workspace, org_id=matched_organization.id)
    if not matched_workspace:
        logger.info(f"Workspace '{workspace}' not found.")
        return

    # verify with the user
    if not confirm(
        f"Do you want to delete workspace '{matched_workspace.name}' from organization '{matched_organization.name}'? This will remove all associated data and access for all users."
    ):
        logger.info("Workspace deletion cancelled.")
        return

    client.delete_workspace(matched_workspace.id)
    logger.info(f"Workspace '{matched_workspace.name}' deleted.")


@cli.command()
def orgs() -> None:
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


user_cli = cyclopts.App(name="user", help="Manage user settings and profiles.", help_flags=[])


@user_cli.command(name=["show", "list", "ls"])
def show() -> None:
    """List all configured server profiles."""

    config = UserConfig.read()
    if not config.servers:
        logger.error("No server profiles are configured")
        return

    table = Table(box=box.ROUNDED)
    table.add_column("Profile", style="orange_red1")
    table.add_column("URL", style="cyan")
    table.add_column("Email")
    table.add_column("Username")
    table.add_column("Valid Until")

    for profile, server in config.servers.items():
        active = profile == config.active
        refresh_token = Token(server.refresh_token)

        table.add_row(
            profile + ("*" if active else ""),
            server.url,
            server.email,
            server.username,
            "[red]expired[/]"
            if refresh_token.is_expired()
            else f"{refresh_token.expires_at.astimezone().strftime('%c')} ({time_to(refresh_token.expires_at)})",
            style="bold" if active else None,
        )

    console.print(table)


@user_cli.command()
def switch(
    profile: t.Annotated[str | None, cyclopts.Parameter(help="Profile to switch to")] = None,
) -> None:
    """Set the active server profile"""
    config = UserConfig.read()

    if not config.servers:
        logger.error("No server profiles are configured")
        return

    # If no profile provided, prompt user to choose
    if profile is None:
        profiles = list(config.servers.keys())
        logger.info("Available profiles:")
        for i, p in enumerate(profiles, 1):
            active_marker = " (current)" if p == config.active else ""
            logger.info(f"  {i}. [bold orange_red1]{p}[/]{active_marker}")

        choice = Prompt.ask(
            "\nSelect a profile",
            choices=[str(i) for i in range(1, len(profiles) + 1)] + profiles,
            show_choices=False,
            console=console,
        )

        profile = profiles[int(choice) - 1] if choice.isdigit() else choice

    if profile not in config.servers:
        logger.error(f"Profile [bold]{profile}[/] does not exist")
        return

    config.active = profile
    config.write()

    logger.success(
        f"Switched to [bold orange_red1]{profile}[/]\n"
        f"|- email:    [bold]{config.servers[profile].email}[/]\n"
        f"|- username: {config.servers[profile].username}\n"
        f"|- url:      {config.servers[profile].url}\n"
    )


@user_cli.command()
def forget(
    profile: t.Annotated[str, cyclopts.Parameter(help="Profile of the server to remove")],
) -> None:
    """Remove a server profile from the configuration."""
    config = UserConfig.read()
    if profile not in config.servers:
        logger.error(f"Profile [bold]{profile}[/] does not exist")
        return

    del config.servers[profile]
    config.write()

    logger.success(f"Forgot about [bold]{profile}[/]")


def _print_workspace_table(workspaces: list[Workspace], organization: Organization) -> None:
    table = Table(box=box.ROUNDED)
    table.add_column("Name", style="orange_red1")
    table.add_column("Key", style="green")
    table.add_column("ID")
    table.add_column("dn.configure() Command", style="cyan")

    for ws in workspaces:
        table.add_row(
            ws.name,
            ws.key,
            str(ws.id),
            f'dn.configure(organization="{organization.key}", workspace="{ws.key}")',
        )

    console.print(table)


# Register user_cli after it's fully defined
cli.command(user_cli)

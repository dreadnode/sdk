import typing as t

import cyclopts
import rich
from rich import box
from rich.prompt import Prompt
from rich.table import Table

from dreadnode.cli.api import Token
from dreadnode.user_config import UserConfig
from dreadnode.util import time_to

cli = cyclopts.App(name="profile", help="Manage server profiles")


@cli.command(name=["show", "list", "ls"])
def show() -> None:
    """List all configured server profiles."""

    config = UserConfig.read()
    if not config.servers:
        rich.print(":exclamation: No server profiles are configured")
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

    rich.print(table)


@cli.command()
def switch(
    profile: t.Annotated[str | None, cyclopts.Parameter(help="Profile to switch to")] = None,
) -> None:
    """Set the active server profile"""
    config = UserConfig.read()

    if not config.servers:
        rich.print(":exclamation: No server profiles are configured")
        return

    # If no profile provided, prompt user to choose
    if profile is None:
        profiles = list(config.servers.keys())
        rich.print("\nAvailable profiles:")
        for i, p in enumerate(profiles, 1):
            active_marker = " (current)" if p == config.active else ""
            rich.print(f"  {i}. [bold orange_red1]{p}[/]{active_marker}")

        choice = Prompt.ask(
            "\nSelect a profile",
            choices=[str(i) for i in range(1, len(profiles) + 1)] + profiles,
            show_choices=False,
        )

        profile = profiles[int(choice) - 1] if choice.isdigit() else choice

    if profile not in config.servers:
        rich.print(f":exclamation: Profile [bold]{profile}[/] does not exist")
        return

    config.active = profile
    config.write()

    rich.print(f":laptop_computer: Switched to [bold orange_red1]{profile}[/]")
    rich.print(f"|- email:    [bold]{config.servers[profile].email}[/]")
    rich.print(f"|- username: {config.servers[profile].username}")
    rich.print(f"|- url:      {config.servers[profile].url}")
    rich.print()


@cli.command()
def forget(
    profile: t.Annotated[str, cyclopts.Parameter(help="Profile of the server to remove")],
) -> None:
    """Remove a server profile from the configuration."""
    config = UserConfig.read()
    if profile not in config.servers:
        rich.print(f":exclamation: Profile [bold]{profile}[/] does not exist")
        return

    del config.servers[profile]
    config.write()

    rich.print(f":axe: Forgot about [bold]{profile}[/]")

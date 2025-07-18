import typing as t

import cyclopts
import rich
from rich import box
from rich.table import Table

from dreadnode.cli.api import Token
from dreadnode.cli.config import UserConfig
from dreadnode.util import time_to

cli = cyclopts.App(name="profile", help="Manage server profiles")


@cli.command(name=["show", "list"], help="List all server profiles")
def show() -> None:
    config = UserConfig.read()
    if not config.servers:
        rich.print(":exclamation: No server profiles are configured")
        return

    table = Table(box=box.ROUNDED)
    table.add_column("Profile", style="magenta")
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


@cli.command(help="Set the active server profile")
def switch(profile: t.Annotated[str, cyclopts.Parameter(help="Profile to switch to")]) -> None:
    config = UserConfig.read()
    if profile not in config.servers:
        rich.print(f":exclamation: Profile [bold]{profile}[/] does not exist")
        return

    config.active = profile
    config.write()

    rich.print(f":laptop_computer: Switched to [bold magenta]{profile}[/]")
    rich.print(f"|- email:    [bold]{config.servers[profile].email}[/]")
    rich.print(f"|- username: {config.servers[profile].username}")
    rich.print(f"|- url:      {config.servers[profile].url}")
    rich.print()


@cli.command(help="Remove a server profile")
def forget(
    profile: t.Annotated[str, cyclopts.Parameter(help="Profile of the server to remove")],
) -> None:
    config = UserConfig.read()
    if profile not in config.servers:
        rich.print(f":exclamation: Profile [bold]{profile}[/] does not exist")
        return

    del config.servers[profile]
    config.write()

    rich.print(f":axe: Forgot about [bold]{profile}[/]")

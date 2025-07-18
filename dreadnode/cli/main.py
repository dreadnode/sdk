import contextlib
import typing as t
import webbrowser

import cyclopts
import rich

from dreadnode.api.client import ApiClient
from dreadnode.cli.api import create_api_client
from dreadnode.cli.config import ServerConfig, UserConfig
from dreadnode.cli.profile import cli as profile_cli
from dreadnode.constants import PLATFORM_BASE_URL

cli = cyclopts.App(help="Interact with Dreadnode platforms", version_flags=[], help_on_error=True)

cli["--help"].group = "Meta"

cli.command(profile_cli)


@cli.meta.default
def meta(
    *tokens: t.Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
) -> None:
    rich.print()
    cli(tokens)


@cli.command(help="Authenticate to a platform server.", group="Auth")
def login(
    *,
    server: t.Annotated[
        str | None, cyclopts.Parameter(name=["--server", "-s"], help="URL of the server")
    ] = None,
    profile: t.Annotated[
        str | None,
        cyclopts.Parameter(name=["--profile", "-p"], help="Profile alias to assign / update"),
    ] = None,
) -> None:
    if not server:
        server = PLATFORM_BASE_URL
        with contextlib.suppress(Exception):
            existing_config = UserConfig.read().get_server_config(profile)
            server = existing_config.url

    # create client with no auth data
    client = ApiClient(base_url=server)

    rich.print(":laptop_computer: Requesting device code ...")

    # request user and device codes
    codes = client.get_device_codes()

    # present verification URL to user
    verification_url = client.url_for_user_code(codes.user_code)
    verification_url_base = verification_url.split("?")[0]

    rich.print()
    rich.print(
        f"""\
Attempting to automatically open the authorization page in your default browser.
If the browser does not open or you wish to use a different device, open the following URL:

:link: [bold]{verification_url_base}[/]

Then enter the code: [bold]{codes.user_code}[/]
"""
    )

    webbrowser.open(verification_url)

    # poll for the access token after user verification
    tokens = client.poll_for_token(codes.device_code)

    client = ApiClient(
        server, cookies={"refresh_token": tokens.refresh_token, "access_token": tokens.access_token}
    )
    user = client.get_user()

    UserConfig.read().set_server_config(
        ServerConfig(
            url=server,
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            email=user.email_address,
            username=user.username,
            api_key=user.api_key.key,
        ),
        profile,
    ).write()

    rich.print(f":white_check_mark: Authenticated as {user.email_address} ({user.username})")


@cli.command(help="Refresh data for the active server profile.", group="Auth")
def refresh() -> None:
    user_config = UserConfig.read()
    server_config = user_config.get_server_config()

    client = create_api_client()
    user = client.get_user()

    server_config.email = user.email_address
    server_config.username = user.username
    server_config.api_key = user.api_key.key

    user_config.set_server_config(server_config).write()

    rich.print(
        f":white_check_mark: Refreshed '[bold]{user_config.active}[/bold]' ([magenta]{user.email_address}[/] / [cyan]{user.username}[/])"
    )


@cli.command(help="Show versions and exit.", group="Meta")
def version() -> None:
    import importlib.metadata
    import platform
    import sys

    version = importlib.metadata.version("dreadnode")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    os_name = platform.system()
    arch = platform.machine()
    rich.print(f"Platform:   {os_name} ({arch})")
    rich.print(f"Python:     {python_version}")
    rich.print(f"Dreadnode:  {version}")

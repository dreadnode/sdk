import contextlib
import importlib.metadata
import pathlib
import platform
import shutil
import sys
import typing as t
import webbrowser

import cyclopts
import rich
from rich.panel import Panel
from rich.prompt import Prompt

from dreadnode.api.client import ApiClient
from dreadnode.cli.agent import cli as agent_cli
from dreadnode.cli.api import create_api_client
from dreadnode.cli.eval import cli as eval_cli
from dreadnode.cli.github import (
    GithubRepo,
    download_and_unzip_archive,
    validate_server_for_clone,
)
from dreadnode.cli.platform import cli as platform_cli
from dreadnode.cli.profile import cli as profile_cli
from dreadnode.cli.study import cli as study_cli
from dreadnode.constants import DEBUG, PLATFORM_BASE_URL
from dreadnode.user_config import ServerConfig, UserConfig

cli = cyclopts.App(help="Interact with Dreadnode platforms", version_flags=[], help_on_error=True)

cli["--help"].group = "Meta"

cli.command(agent_cli)
cli.command(eval_cli)
cli.command(study_cli)
cli.command(platform_cli)
cli.command(profile_cli)


@cli.meta.default
def meta(
    *tokens: t.Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
) -> None:
    try:
        rich.print()
        cli(tokens)
    except Exception as e:
        if DEBUG:
            raise

        rich.print()
        rich.print(Panel(str(e), title="Error", title_align="left", border_style="red"))
        sys.exit(1)


@cli.command(group="Auth")
def login(
    *,
    server: t.Annotated[
        str | None,
        cyclopts.Parameter(name=["--server", "-s"], help="URL of the server"),
    ] = None,
    profile: t.Annotated[
        str | None,
        cyclopts.Parameter(name=["--profile", "-p"], help="Profile alias to assign / update"),
    ] = None,
) -> None:
    """Authenticate to a Dreadnode platform server and save the profile."""
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
        server,
        cookies={
            "refresh_token": tokens.refresh_token,
            "access_token": tokens.access_token,
        },
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

    rich.print(
        f":white_check_mark: Refreshed '[bold]{user_config.active}[/bold]' ([magenta]{user.email_address}[/] / [cyan]{user.username}[/])"
    )


@cli.command()
def clone(
    repo: t.Annotated[str, cyclopts.Parameter(help="Repository name or URL")],
    target: t.Annotated[
        pathlib.Path | None,
        cyclopts.Parameter(help="The target directory"),
    ] = None,
) -> None:
    """Clone a GitHub repository to a local directory"""

    github_repo = GithubRepo(repo)

    # Check if the target directory exists
    target = target or pathlib.Path(github_repo.repo)
    if target.exists():
        if (
            Prompt.ask(f":axe: Overwrite {target.absolute()}?", choices=["y", "n"], default="n")
            == "n"
        ):
            return
        rich.print()
        shutil.rmtree(target)

    # Check if the repo is accessible
    if github_repo.exists:
        temp_dir = download_and_unzip_archive(github_repo.zip_url)

    # This could be a private repo that the user can access
    # by getting an access token from our API
    elif github_repo.namespace == "dreadnode":
        # Validate server configuration for private repository access
        user_config = UserConfig.read()
        profile_to_use = validate_server_for_clone(user_config, None)

        if profile_to_use is None:
            return  # User cancelled

        github_access_token = create_api_client(profile=profile_to_use).get_github_access_token(
            [github_repo.repo]
        )
        rich.print(":key: Accessed private repository")
        temp_dir = download_and_unzip_archive(
            github_repo.api_zip_url,
            headers={"Authorization": f"Bearer {github_access_token.token}"},
        )

    else:
        raise RuntimeError(f"Repository '{github_repo}' not found or inaccessible")

    # We assume the repo download results in a single
    # child folder which is the real target
    sub_dirs = list(temp_dir.iterdir())
    if len(sub_dirs) == 1 and sub_dirs[0].is_dir():
        temp_dir = sub_dirs[0]

    shutil.move(temp_dir, target)

    rich.print()
    rich.print(f":tada: Cloned [b]{repo}[/] to [b]{target.absolute()}[/]")


@cli.command(help="Show versions and exit.", group="Meta")
def version() -> None:
    version = importlib.metadata.version("dreadnode")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    os_name = platform.system()
    arch = platform.machine()
    rich.print(f"Platform:   {os_name} ({arch})")
    rich.print(f"Python:     {python_version}")
    rich.print(f"Dreadnode:  {version}")

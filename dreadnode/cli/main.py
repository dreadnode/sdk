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
from loguru import logger
from rich.panel import Panel

from dreadnode.api.client import ApiClient
from dreadnode.cli.agent import cli as agent_cli
from dreadnode.cli.api import create_api_client
from dreadnode.cli.attack import cli as attack_cli
from dreadnode.cli.docker import DockerImage, docker_login, docker_pull, docker_tag, get_registry
from dreadnode.cli.eval import cli as eval_cli
from dreadnode.cli.github import (
    GithubRepo,
    download_and_unzip_archive,
    validate_server_for_clone,
)
from dreadnode.cli.platform import cli as platform_cli
from dreadnode.cli.profile import cli as profile_cli
from dreadnode.cli.rbac.organizations import cli as rbac_organizations_cli
from dreadnode.cli.rbac.workspaces import cli as rbac_workspaces_cli
from dreadnode.cli.study import cli as study_cli
from dreadnode.cli.task import cli as task_cli
from dreadnode.constants import DEBUG, PLATFORM_BASE_URL
from dreadnode.logging_ import confirm, console, print_info, print_success
from dreadnode.logging_ import console as logging_console
from dreadnode.user_config import ServerConfig, UserConfig

cli = cyclopts.App(
    help="Interact with Dreadnode platforms",
    version_flags=[],
    help_on_error=True,
    console=logging_console,
)

cli["--help"].group = "Meta"

cli.command(agent_cli)
cli.command(task_cli)
cli.command(eval_cli)
cli.command(study_cli)
cli.command(attack_cli)
cli.command(rbac_organizations_cli)
cli.command(rbac_workspaces_cli)
cli.command(platform_cli)
cli.command(profile_cli)


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

    # create client with no auth data
    client = ApiClient(base_url=server)

    print_info("Requesting device code ...")

    # request user and device codes
    codes = client.get_device_codes()

    # present verification URL to user
    verification_url = client.url_for_user_code(codes.user_code)
    verification_url_base = verification_url.split("?")[0]

    print_info(
        f"""
        Attempting to automatically open the authorization page in your default browser.
        If the browser does not open or you wish to use a different device, open the following URL:

        [bold]{verification_url_base}[/]

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

    print_success(f"Authenticated as {user.email_address} ({user.username})")


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

    print_success(
        f"Refreshed '[bold]{user_config.active}[/bold]' ([magenta]{user.email_address}[/] / [cyan]{user.username}[/])"
    )


@cli.command()
def pull(image: str) -> None:
    """
    Pull a capability image from the dreadnode registry.

    Args:
        image: The name of the image to pull (e.g. dreadnode/agent:latest).
    """
    user_config = UserConfig.read()
    if not user_config.active_profile_name:
        raise RuntimeError("No server profile is set, use [bold]dreadnode login[/] to authenticate")

    server_config = user_config.get_server_config()

    docker_image = DockerImage(image)
    tag_as: str | None = None
    if docker_image.repository.startswith("dreadnode/") and not docker_image.registry:
        docker_image = docker_image.with_(
            registry=get_registry(user_config.get_server_config()),
        )
        tag_as = image

    if docker_image.registry and docker_image.registry != "docker.io":
        print_info(f"Authenticating to [bold]{docker_image.registry}[/] ...")
        docker_login(docker_image.registry, server_config.username, server_config.api_key)

    print_info(f"Pulling image [bold]{docker_image}[/] ...")
    docker_pull(docker_image)

    if tag_as:
        docker_tag(docker_image, tag_as)


@cli.command()
def clone(
    repo: str,
    target: pathlib.Path | None = None,
) -> None:
    """
    Clone a GitHub repository to a local directory

    Args:
        repo: Repository name or URL.
        target: The target directory.
    """
    github_repo = GithubRepo(repo)

    # Check if the target directory exists
    target = target or pathlib.Path(github_repo.repo)
    if target.exists():
        if not confirm(f"{target.absolute()} exists, overwrite?"):
            return
        console.print()
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
        print_info("Accessed private repository")
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

    print_success(f"Cloned [b]{repo}[/] to [b]{target.absolute()}[/]")


@cli.command(help="Show versions and exit.", group="Meta")
def version() -> None:
    version = importlib.metadata.version("dreadnode")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    os_name = platform.system()
    arch = platform.machine()
    print_info(f"Platform:   {os_name} ({arch})")
    print_info(f"Python:     {python_version}")
    print_info(f"Dreadnode:  {version}")

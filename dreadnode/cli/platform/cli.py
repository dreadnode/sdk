import typing as t

import cyclopts

from dreadnode.cli.platform.configure import configure_platform, list_configurations
from dreadnode.cli.platform.download import download_platform
from dreadnode.cli.platform.login import log_into_registries
from dreadnode.cli.platform.start import start_platform
from dreadnode.cli.platform.status import platform_status
from dreadnode.cli.platform.stop import stop_platform
from dreadnode.cli.platform.upgrade import upgrade_platform
from dreadnode.cli.platform.utils.printing import print_info
from dreadnode.cli.platform.utils.versions import get_current_version

cli = cyclopts.App("platform", help="Run and manage the platform.", help_flags=[])


@cli.command()
def start(
    tag: t.Annotated[
        str | None, cyclopts.Parameter(help="Optional image tag to use when starting the platform.")
    ] = None,
    **env_overrides: t.Annotated[
        str,
        cyclopts.Parameter(
            help="Environment variable overrides. Use --key value format. "
            "Examples: --proxy-host myproxy.local"
        ),
    ],
) -> None:
    """Start the platform. Optionally, provide a tagged version to start.

    Args:
        tag: Optional image tag to use when starting the platform.
        **env_overrides: Key-value pairs to override environment variables in the
            platform's .env file. e.g `--proxy-host myproxy.local`
    """
    start_platform(tag=tag, **env_overrides)


@cli.command(name=["stop", "down"])
def stop() -> None:
    """Stop the running platform."""
    stop_platform()


@cli.command()
def download(
    tag: t.Annotated[
        str | None, cyclopts.Parameter(help="Optional image tag to use when starting the platform.")
    ] = None,
) -> None:
    """Download platform files for a specific tag.

    Args:
        tag: Optional image tag to download.
    """
    download_platform(tag=tag)


@cli.command()
def upgrade() -> None:
    """Upgrade the platform to the latest version."""
    upgrade_platform()


@cli.command()
def refresh_registry_auth() -> None:
    """Refresh container registry credentials for platform access.

    Used for out of band Docker management.
    """
    log_into_registries()


@cli.command()
def configure(
    *args: t.Annotated[
        str,
        cyclopts.Parameter(
            help="Key-value pairs to set. Must be provided in pairs (key value key value ...). ",
        ),
    ],
    tag: t.Annotated[
        str | None, cyclopts.Parameter(help="Optional image tag to use when starting the platform.")
    ] = None,
    list: t.Annotated[
        bool,
        cyclopts.Parameter(
            ["--list", "-l"], help="List current configuration without making changes."
        ),
    ] = False,
    unset: t.Annotated[
        bool,
        cyclopts.Parameter(["--unset", "-u"], help="Remove the specified configuration."),
    ] = False,
) -> None:
    """Configure the platform for a specific service.
    Configurations will take effect the next time the platform is started and are persisted.

    Usage: platform configure KEY VALUE [KEY2 VALUE2 ...]
    Examples:
        platform configure proxy-host myproxy.local
        platform configure proxy-host myproxy.local api-port 8080

    Args:
        *args: Key-value pairs to set. Must be provided in pairs (key value key value ...).
        tag: Optional image tag to use when starting the platform.
    """
    if list:
        if args:
            raise ValueError("The --list option does not take any positional arguments.")
        list_configurations()
        return
    # Parse positional arguments into key-value pairs
    if not unset and len(args) % 2 != 0:
        raise ValueError(
            "Arguments must be provided in key-value pairs like: KEY VALUE [KEY2 VALUE2 ...]"
        )

    # Convert positional args to dict
    env_overrides = {}
    for i in range(0, len(args), 2):
        key = args[i]
        value = args[i + 1] if not unset else None
        env_overrides[key] = value

    configure_platform(tag=tag, **env_overrides)


@cli.command()
def version(
    verbose: t.Annotated[  # noqa: FBT002
        bool,
        cyclopts.Parameter(
            ["--verbose", "-v"], help="Display detailed information for the version."
        ),
    ] = False,
) -> None:
    """Show the current platform version."""
    version = get_current_version()
    if version:
        if verbose:
            print_info(version.details)
        else:
            print_info(f"Current platform version: {version!s}")

    else:
        print_info("No current platform version is set.")


@cli.command()
def status(
    tag: t.Annotated[
        str | None, cyclopts.Parameter(help="Optional image tag to use when checking status.")
    ] = None,
) -> None:
    """Get the status of the platform with the specified or current version.

    Args:
        tag: Optional image tag to use. If not provided, uses the current
            version or downloads the latest available version.
    """
    platform_status(tag=tag)

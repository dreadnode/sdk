import typing as t

import cyclopts

from dreadnode.core.integrations.docker import (
    DockerError,
    get_env_var_from_container,
)
from dreadnode.core.log import confirm, logger
from dreadnode.core.settings import PLATFORM_SERVICES
from dreadnode.platform.compose import (
    build_compose_override_file,
    compose_down,
    compose_login,
    compose_logs,
    compose_up,
    platform_is_running,
)
from dreadnode.platform.download import download_platform
from dreadnode.platform.env_mgmt import (
    build_env_file,
    read_env_file,
    remove_overrides_env,
    write_overrides_env,
)
from dreadnode.platform.tag import tag_to_semver
from dreadnode.platform.version import VersionConfig

platform_cli = cyclopts.App("platform", help="Run and manage the platform.", help_flags=[])


@platform_cli.command()
def start(tag: str | None = None, **env_overrides: str) -> None:
    """
    Start the platform.

    Args:
        tag: Image tag to use when starting the platform.
        env_overrides: Key-value pairs to override environment variables in the
            platform's .env file. e.g `--proxy-host myproxy.local`
    """
    version_config = VersionConfig.read()
    version = version_config.get_current_version(tag=tag) or download_platform(tag)
    version_config.set_current_version(version)

    if platform_is_running(version):
        logger.info(f"Platform {version.tag} is already running.")
        logger.info("Use `dreadnode platform stop` to stop it first.")
        return

    compose_login(version)

    if env_overrides:
        write_overrides_env(version.arg_overrides_env_file, **env_overrides)

    logger.info(f"Starting platform [cyan]{version.tag}[/] ...")
    try:
        compose_up(version)
        logger.success("Platform started.")
        origin = get_env_var_from_container("dreadnode-ui", "ORIGIN")
        if origin:
            logger.info("You can access the app at the following URLs:")
            logger.info(f" - {origin}")
        else:
            logger.info(" - Unable to determine the app URL.")
            logger.info("Please check the container logs for more information.")
    except DockerError as e:
        compose_logs(version, tail=10)
        logger.error(str(e))


@platform_cli.command(name=["stop", "down"])
def stop(*, remove_volumes: t.Annotated[bool, cyclopts.Parameter(negative=False)] = False) -> None:
    """
    Stop the running platform.

    Args:
        remove_volumes: Also remove Docker volumes associated with the platform.
    """
    version = VersionConfig.read().get_current_version()
    if not version:
        logger.error("No current version found. Nothing to stop.")
        return

    remove_overrides_env(version.arg_overrides_env_file)
    compose_down(version, remove_volumes=remove_volumes)
    logger.success("Platform stopped.")


@platform_cli.command()
def logs(tail: int = 100) -> None:
    """
    View the platform logs.

    Args:
        tail: Number of lines to show from the end of the logs for each service.
    """
    version = VersionConfig.read().get_current_version()
    if not version:
        logger.error("No current version found. Nothing to show logs for.")
        return

    compose_logs(version, tail=tail)


@platform_cli.command()
def download(tag: str | None = None) -> None:
    """
    Download platform files for a specific tag.

    Args:
        tag: Specific version tag to download.
    """
    download_platform(tag=tag)


@platform_cli.command()
def upgrade() -> None:
    """
    Upgrade the platform to the latest available version.

    Downloads the latest version, compares it with the current version,
    and performs the upgrade if a newer version is available. Optionally
    merges configuration files from the current version to the new version.
    Stops the current platform and starts the upgraded version.
    """
    version_config = VersionConfig.read()
    current_version = version_config.get_current_version()
    if not current_version:
        start()
        return

    latest_version = download_platform()

    current_semver = tag_to_semver(current_version.tag)
    remote_semver = tag_to_semver(latest_version.tag)

    if current_semver >= remote_semver:
        logger.info(f"You are using the latest ({current_semver}) version of the platform.")
        return

    if not confirm(
        f"Upgrade from [cyan]{current_version.tag}[/] -> [magenta]{latest_version.tag}[/]?"
    ):
        return

    version_config.set_current_version(latest_version)

    # copy the configuration overrides from the current version to the new version
    if (
        current_version.configure_overrides_compose_file.exists()
        and current_version.configure_overrides_env_file.exists()
    ):
        latest_version.configure_overrides_compose_file.write_text(
            current_version.configure_overrides_compose_file.read_text()
        )
        latest_version.configure_overrides_env_file.write_text(
            current_version.configure_overrides_env_file.read_text()
        )

    logger.info("Stopping current platform ...")
    compose_down(current_version)
    compose_up(latest_version)
    logger.success(f"Platform upgraded to version [magenta]{latest_version.tag}[/].")


@platform_cli.command()
def refresh_registry_auth() -> None:
    """
    Refresh container registry credentials for platform access.

    Used for out of band Docker management.
    """
    current_version = VersionConfig.read().get_current_version()
    if not current_version:
        logger.info("There are no registries configured. Run `dreadnode platform start` to start.")
        return

    compose_login(current_version, force=True)


@platform_cli.command()
def configure(
    *args: str,
    tag: str | None = None,
    list: t.Annotated[
        bool,
        cyclopts.Parameter(["--list", "-l"], negative=False),
    ] = False,
    unset: t.Annotated[
        bool,
        cyclopts.Parameter(["--unset", "-u"], negative=False),
    ] = False,
) -> None:
    """
    Configure the platform for a specific service.

    Configurations will take effect the next time the platform is started and are persisted.

    Usage: platform configure KEY VALUE [KEY2 VALUE2 ...]
    Examples:
        platform configure proxy-host myproxy.local
        platform configure proxy-host myproxy.local api-port 8080

    Args:
        args: Key-value pairs to set. Must be provided in pairs (key value key value ...).
        tag: Optional image tag to use when starting the platform.
        list: List current configuration without making changes.
        unset: Remove the specified configuration.
    """
    current_version = VersionConfig.read().get_current_version(tag=tag)
    if not current_version:
        logger.info("No current platform version is set. Please start or download the platform.")
        return

    if list:
        overrides_env_file = current_version.configure_overrides_env_file
        if not overrides_env_file.exists():
            logger.info("No configuration overrides found.")
            return

        logger.info(f"Configuration overrides from {overrides_env_file}:")
        env_vars = read_env_file(overrides_env_file)
        for key, value in env_vars.items():
            logger.info(f" - {key}={value}")
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
        env_overrides[key] = args[i + 1] if not unset else None

    if not env_overrides:
        logger.warning("No configuration changes specified.")
        return

    logger.info("Setting environment overrides ...")
    build_compose_override_file(PLATFORM_SERVICES, current_version)
    build_env_file(current_version.configure_overrides_env_file, **env_overrides)
    logger.info(
        f"Configuration written to {current_version.local_path}.\n\n"
        "These will take effect the next time the platform is started. "
        "You can modify or remove them at any time."
    )


@platform_cli.command()
def version(
    verbose: t.Annotated[  # noqa: FBT002
        bool, cyclopts.Parameter(["--verbose", "-v"])
    ] = False,
) -> None:
    """
    Show the current platform version.

    Args:
        verbose: Display detailed information about the version.
    """
    version_config = VersionConfig.read()
    version = version_config.get_current_version()
    if version is None:
        logger.info("No current platform version is set.")
        return

    logger.info(f"Current version: [cyan]{version}[/]")
    if verbose:
        logger.info(version.details)


@platform_cli.command()
def status() -> None:
    """
    Get the current status of the platform.
    """
    version_config = VersionConfig.read()
    version = version_config.get_current_version()
    if version is None:
        logger.error("No current platform version is set. Please start or download the platform.")
        return

    if platform_is_running(version):
        logger.success(f"Platform {version.tag} is running.")
    else:
        logger.error(f"Platform {version.tag} is not fully running.")

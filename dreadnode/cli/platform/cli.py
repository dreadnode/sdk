import cyclopts

from dreadnode.cli.platform.configure import configure_platform
from dreadnode.cli.platform.download import download_platform
from dreadnode.cli.platform.login import log_into_registries
from dreadnode.cli.platform.start import start_platform
from dreadnode.cli.platform.stop import stop_platform
from dreadnode.cli.platform.upgrade import upgrade_platform

cli = cyclopts.App("platform", help="Run and manage the platform.", help_flags=[])


@cli.command()
def start(tag: str | None = None) -> None:
    """Start the platform. Optionally, provide a tagged version to start.

    Args:
        tag: Optional image tag to use when starting the platform.
    """
    start_platform(tag=tag)


@cli.command(name=["stop", "down"])
def stop() -> None:
    """Stop the running platform."""
    stop_platform()


@cli.command()
def download(tag: str) -> None:
    """Download platform files for a specific tag.

    Args:
        tag: Image tag to download.
    """
    download_platform(tag)


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
def configure(service: str = "api") -> None:
    """Configure the platform for a specific service.

    Args:
        service: The name of the service to configure.
    """
    configure_platform(service=service)

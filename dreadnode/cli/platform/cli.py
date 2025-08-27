import cyclopts

from dreadnode.cli.platform.check_for_updates import check_for_updates as check_for_updates_
from dreadnode.cli.platform.configure import configure_platform
from dreadnode.cli.platform.docker.download import download as download_platform
from dreadnode.cli.platform.docker.login import docker_login
from dreadnode.cli.platform.docker.start import start as start_platform
from dreadnode.cli.platform.docker.start import stop as stop_platform
from dreadnode.cli.platform.init import init as init_platform
from dreadnode.cli.platform.init import initialized as platform_initilized

cli = cyclopts.App("platform", help="Run and manage the platform.", help_flags=[])


@cli.command()
def init(tag: str = "latest", arch: str | None = None) -> None:
    """
    Initialize the platform.
    """
    init_platform(tag=tag, arch=arch)


@cli.command()
def download(tag: str = "latest", arch: str | None = None) -> None:
    """
    Download the platform files.
    """
    docker_login()

    if not platform_initilized() or tag != "latest" or arch:
        init_platform(tag=tag, arch=arch)

    download_platform()


@cli.command()
def configure() -> None:
    """
    Configure the platform.
    """
    configure_platform()


@cli.command()
def start() -> None:
    """
    Start the platform services.
    """
    start_platform()


@cli.command()
def stop() -> None:
    """
    Stop the platform services.
    """
    stop_platform()


@cli.command()
def check_for_updates() -> None:
    """
    Check for platform updates.
    """
    check_for_updates_()


@cli.command()
def update() -> None:
    """
    Update the platform.
    """
    stop_platform()
    download_platform()
    start_platform()

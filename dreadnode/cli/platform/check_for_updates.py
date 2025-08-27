import rich

from dreadnode.cli.api import create_api_client
from dreadnode.cli.platform.constants import SERVICES
from dreadnode.cli.platform.utils import get_local_arch, get_local_cache_dir, get_local_version


def check_for_updates() -> None:
    import importlib.metadata  # noqa: PLC0415

    local_cache_dir = get_local_cache_dir()
    rich.print(f"Checking local cache directory: {local_cache_dir}")

    if not local_cache_dir.exists():
        rich.print(
            "Local cache directory does not exist. Please run \n[dim]$[/dim] [bold green]dreadnode platform init[/bold green]"
        )
        return

    arch = get_local_arch()
    api_client = create_api_client()
    registry_image_details = api_client.get_platform_releases(
        arch=arch,
        tag="latest",
        services=SERVICES,
        cli_version=importlib.metadata.version("dreadnode"),
    )

    local_image_details = get_local_version()

    for image_detail in local_image_details.images:
        for remote_image_detail in registry_image_details.images:
            if image_detail.service == remote_image_detail.service:
                if image_detail.version != remote_image_detail.version:
                    rich.print(
                        f"[yellow]Update available for {image_detail.service}: "
                        f"{image_detail.version} -> {remote_image_detail.version}[/yellow]"
                    )
                else:
                    rich.print(
                        f"[green]{image_detail.service} is up to date: {image_detail.version}[/green]"
                    )
    rich.print(
        "[blue]You can update with:[/blue]\n[dim]$[/dim] [bold green]dreadnode platform update[/bold green]"
    )

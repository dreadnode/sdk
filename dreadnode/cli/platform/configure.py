import rich

from dreadnode.cli.platform.utils import get_local_cache_dir


def configure_platform() -> None:
    rich.print(f"Configure the API by modifying {get_local_cache_dir()}/.api.env")
    rich.print(f"Configure the UI by modifying {get_local_cache_dir()}/.ui.env")
    rich.print("See https://docs.dreadnode.io/platform/manage for more details.")

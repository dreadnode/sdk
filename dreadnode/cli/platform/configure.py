from dreadnode.cli.platform.utils.env_mgmt import open_env_file
from dreadnode.cli.platform.utils.printing import print_info
from dreadnode.cli.platform.utils.versions import get_current_version, get_local_cache_dir


def configure_platform(service: str = "api", tag: str | None = None) -> None:
    """Configure the platform for a specific service.

    Args:
        service: The name of the service to configure.
    """
    if not tag:
        current_version = get_current_version()
        tag = current_version.tag if current_version else "latest"

    print_info(f"Configuring {service} service...")
    env_file = get_local_cache_dir() / tag / f".{service}.env"
    open_env_file(env_file)
    print_info(
        f"Configuration for {service} service loaded. It will take effect the next time the service is started."
    )

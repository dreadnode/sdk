from dreadnode.cli.platform.docker_ import docker_ps
from dreadnode.cli.platform.schemas import LocalVersionSchema
from dreadnode.cli.platform.utils.printing import print_error, print_success
from dreadnode.cli.platform.utils.versions import get_current_version, get_local_version


def platform_is_running(selected_version: LocalVersionSchema) -> bool:
    """Check if the platform with the specified or current version is running.

    Args:
        tag: Optional image tag to use. If not provided, uses the current
            version or downloads the latest available version.
    """

    container_details = docker_ps(selected_version)
    if not container_details:
        return False
    return all(
        container.is_running
        for container in container_details
        if container.name
        in {
            "dreadnode-postgres",
            "dreadnode-clickhouse",
            "dreadnode-traefik",
            "dreadnode-ui",
            "dreadnode-api",
        }
    )


def platform_status(tag: str | None = None) -> bool:
    """Get the status of the platform with the specified or current version.

    Args:
        tag: Optional image tag to use. If not provided, uses the current
            version or downloads the latest available version.
    """
    if tag:
        selected_version = get_local_version(tag)
    elif current_version := get_current_version():
        selected_version = current_version
    else:
        print_error("No current platform version is set. Please start or download the platform.")
        return False
    required_containers_running = platform_is_running(selected_version)
    if required_containers_running:
        print_success(f"Platform {selected_version.tag} is running.")
    else:
        print_error(f"Platform {selected_version.tag} is not fully running.")
    return required_containers_running

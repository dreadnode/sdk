from dreadnode.cli.platform.docker_ import docker_stop
from dreadnode.cli.platform.utils.env_mgmt import remove_overrides_env
from dreadnode.cli.platform.utils.printing import print_error, print_success
from dreadnode.cli.platform.utils.versions import (
    get_current_version,
)


def stop_platform() -> None:
    """Stop the currently running platform.

    Uses the current version's compose file to stop all platform containers
    via docker compose down.
    """
    current_version = get_current_version()
    if not current_version:
        print_error("No current version found. Nothing to stop.")
        return
    remove_overrides_env(current_version.arg_overrides_env_file)
    docker_stop(current_version)
    print_success("Platform stopped successfully.")

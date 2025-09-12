from dreadnode.cli.platform.constants import SERVICES
from dreadnode.cli.platform.docker_ import build_docker_compose_override_file
from dreadnode.cli.platform.utils.env_mgmt import build_env_file, read_env_file
from dreadnode.cli.platform.utils.printing import print_info
from dreadnode.cli.platform.utils.versions import get_current_version, get_local_version


def list_configurations() -> None:
    """List the current platform configuration overrides, if any."""
    current_version = get_current_version()
    if not current_version:
        print_info("No current platform version is set. Please start or download the platform.")
        return

    overrides_env_file = current_version.configure_overrides_env_file
    if not overrides_env_file.exists():
        print_info("No configuration overrides found.")
        return

    print_info(f"Configuration overrides from {overrides_env_file}:")
    env_vars = read_env_file(overrides_env_file)
    for key, value in env_vars.items():
        print_info(f" - {key}={value}")


def configure_platform(tag: str | None = None, **env_overrides: str | None) -> None:
    """Configure the platform for a specific service.

    Args:
        service: The name of the service to configure.
    """
    selected_version = get_local_version(tag) if tag else get_current_version()
    # No need to mark current version on configure

    if not selected_version:
        print_info("No current platform version is set. Please start or download the platform.")
        return

    if env_overrides:
        print_info("Setting environment overrides...")
        build_docker_compose_override_file(SERVICES, selected_version)
        build_env_file(selected_version.configure_overrides_env_file, **env_overrides)
        print_info(
            f"Configuration written to {selected_version.local_path}. "
            "These will take effect the next time the platform is started."
            " You can modify or remove them at any time."
        )

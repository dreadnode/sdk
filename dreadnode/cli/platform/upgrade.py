from dreadnode.cli.platform.docker_ import docker_stop
from dreadnode.cli.platform.download import download_platform
from dreadnode.cli.platform.start import start_platform
from dreadnode.cli.platform.utils.printing import print_error, print_info
from dreadnode.cli.platform.utils.versions import (
    confirm_with_context,
    create_local_latest_tag,
    get_current_version,
    get_semver_from_tag,
    mark_current_version,
    newer_remote_version,
)


def upgrade_platform() -> None:
    """Upgrade the platform to the latest available version.

    Downloads the latest version, compares it with the current version,
    and performs the upgrade if a newer version is available. Optionally
    merges configuration files from the current version to the new version.
    Stops the current platform and starts the upgraded version.
    """
    latest_tag = create_local_latest_tag()

    latest_version = download_platform(latest_tag)
    current_local_version = get_current_version()

    if not current_local_version:
        print_error(
            "No current platform version found. Run `dreadnode platform start` to start the latest version."
        )
        return

    current_semver = get_semver_from_tag(current_local_version.tag)
    remote_semver = get_semver_from_tag(latest_version.tag)

    if not newer_remote_version(current_semver, remote_semver):
        print_info(f"You are using the latest ({current_semver}) version of the platform.")
        return

    if not confirm_with_context(
        f"Are you sure you want to upgrade from {current_local_version.tag} to {latest_version.tag}?"
    ):
        print_error("Aborting upgrade.")
        return

    # copy the configuration overrides from the current version to the new version
    if (
        current_local_version.configure_overrides_compose_file.exists()
        and current_local_version.configure_overrides_env_file.exists()
    ):
        latest_version.configure_overrides_compose_file.write_text(
            current_local_version.configure_overrides_compose_file.read_text()
        )
        latest_version.configure_overrides_env_file.write_text(
            current_local_version.configure_overrides_env_file.read_text()
        )

    print_info(f"Stopping current platform version {current_local_version.tag}...")
    docker_stop(current_local_version)
    print_info(f"Current platform version {current_local_version.tag} stopped.")

    mark_current_version(latest_version)
    start_platform()

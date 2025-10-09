from dreadnode.cli.platform.docker_ import (
    DockerError,
    docker_login,
    docker_requirements_met,
    docker_run,
    docker_stop,
    get_available_local_images,
    get_env_var_from_container,
    get_required_images,
)
from dreadnode.cli.platform.download import download_platform
from dreadnode.cli.platform.status import platform_is_running
from dreadnode.cli.platform.utils.env_mgmt import write_overrides_env
from dreadnode.cli.platform.utils.printing import print_error, print_info, print_success
from dreadnode.cli.platform.utils.versions import (
    create_local_latest_tag,
    get_current_version,
    mark_current_version,
)


def start_platform(tag: str | None = None, **env_overrides: str) -> None:
    """Start the platform with the specified or current version.

    Args:
        tag: Optional image tag to use. If not provided, uses the current
            version or downloads the latest available version.
    """
    if not docker_requirements_met():
        print_error("Docker and Docker Compose must be installed to start the platform.")
        return

    if tag:
        selected_version = download_platform(tag)
        mark_current_version(selected_version)
    elif current_version := get_current_version():
        selected_version = current_version
        # no need to mark
    else:
        latest_tag = create_local_latest_tag()
        selected_version = download_platform(latest_tag)
        mark_current_version(selected_version)

    is_running = platform_is_running(selected_version)
    if is_running:
        print_info(f"Platform {selected_version.tag} is already running.")
        print_info("Use `dreadnode platform stop` to stop it first.")
        return

    # check to see if all required images are available locally
    required_images = get_required_images(selected_version)
    available_images = get_available_local_images()
    missing_images = [img for img in required_images if img not in available_images]
    if missing_images:
        registries_attempted = set()
        for image in selected_version.images:
            if image.registry not in registries_attempted:
                docker_login(image.registry)
                registries_attempted.add(image.registry)

    if env_overrides:
        write_overrides_env(selected_version.arg_overrides_env_file, **env_overrides)

    print_info(f"Starting platform: {selected_version.tag}")
    try:
        docker_run(selected_version)
        print_success(f"Platform {selected_version.tag} started successfully.")
        origin = get_env_var_from_container("dreadnode-ui", "ORIGIN")
        if origin:
            print_info("You can access the app at the following URLs:")
            print_info(f" - {origin}")
        else:
            print_info(" - Unable to determine the app URL.")
            print_info("Please check the container logs for more information.")
    except DockerError as e:
        print_error(f"Failed to start platform {selected_version.tag}: {e}")
        print_info("Stopping any partially started containers...")
        docker_stop(selected_version)
        print_info("You can check the logs for more details.")

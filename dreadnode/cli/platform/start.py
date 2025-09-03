from dreadnode.cli.platform.docker_ import docker_login, docker_run, get_origin
from dreadnode.cli.platform.download import download_platform
from dreadnode.cli.platform.utils.printing import print_info, print_success
from dreadnode.cli.platform.utils.versions import (
    create_local_latest_tag,
    get_current_version,
    mark_current_version,
)


def start_platform(tag: str | None = None) -> None:
    """Start the platform with the specified or current version.

    Args:
        tag: Optional image tag to use. If not provided, uses the current
            version or downloads the latest available version.
    """
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

    registries_attempted = set()
    for image in selected_version.images:
        if image.registry not in registries_attempted:
            docker_login(image.registry)
            registries_attempted.add(image.registry)
    print_info(f"Starting platform: {selected_version.tag}")
    docker_run(selected_version.compose_file)
    print_success(f"Platform {selected_version.tag} started successfully.")
    origin = get_origin("dreadnode-ui")
    if origin:
        print_info("You can access the app at the following URLs:")
        print_info(f" - {origin}")
    else:
        print_info(" - Unable to determine the app URL.")
        print_info("Please check the container logs for more information.")

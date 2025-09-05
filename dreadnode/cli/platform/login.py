from dreadnode.cli.platform.docker_ import docker_login
from dreadnode.cli.platform.utils.printing import print_info
from dreadnode.cli.platform.utils.versions import get_current_version


def log_into_registries() -> None:
    """Log into all Docker registries for the current platform version.

    Iterates through all images in the current version and logs into their
    respective registries. If no current version is set, displays an error message.
    """
    current_version = get_current_version()
    if not current_version:
        print_info("There are no registries configured. Run `dreadnode platform start` to start.")
        return
    registries_attempted = set()
    for image in current_version.images:
        if image.registry not in registries_attempted:
            docker_login(image.registry)
            registries_attempted.add(image.registry)

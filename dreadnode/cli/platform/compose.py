import typing as t

import yaml
from yaml import safe_dump

from dreadnode.cli.api import create_api_client
from dreadnode.cli.docker import (
    DockerImage,
    docker_compose_ps,
    docker_login,
    docker_run,
    get_available_local_images,
)
from dreadnode.cli.platform.constants import PlatformService
from dreadnode.cli.platform.env_mgmt import read_env_file
from dreadnode.cli.platform.version import LocalVersion
from dreadnode.logging_ import print_info


def build_compose_override_file(
    services: list[PlatformService],
    version: LocalVersion,
) -> None:
    # build a yaml docker compose override file
    # that only includes the service being configured
    # and has an `env_file` attribute for the service
    override = {
        "services": {
            f"{service}": {"env_file": [version.configure_overrides_env_file.as_posix()]}
            for service in services
        },
    }

    with version.configure_overrides_compose_file.open("w") as f:
        safe_dump(override, f, sort_keys=False)


def get_required_images(version: LocalVersion) -> list[DockerImage]:
    """
    Get the list of required Docker images for the specified platform version.

    Args:
        version: The selected version of the platform.

    Returns:
        list[str]: List of required Docker image names.
    """
    result = docker_run(
        ["compose", *get_compose_args(version), "config", "--images"],
        timeout=120,
        capture_output=True,
    )

    if result.returncode != 0:
        return []

    return [DockerImage(line) for line in result.stdout.splitlines() if line.strip()]


def get_required_services(version: LocalVersion) -> list[str]:
    """Get the list of required services from the docker-compose file.

    Returns:
        list[str]: List of required service names.
    """
    contents: dict[str, object] = yaml.safe_load(version.compose_file.read_text())
    services = t.cast("dict[str, object]", contents.get("services", {}) or {})
    return [
        str(cfg.get("container_name"))
        for name, cfg in services.items()
        if isinstance(cfg, dict) and cfg.get("x-required")
    ]


def get_profiles_to_enable(version: LocalVersion) -> list[str]:
    """
    Get the list of profiles to enable based on environment variables.

    If any of the `x-profile-disabled-vars` are set in the environment,
    the profile will be disabled.

    E.g.

        services:
        myservice:
            image: myimage:latest
            profiles:
            - myprofile
            x-profile-override-vars:
            - MY_SERVICE_HOST

    If MY_SERVICE_HOST is set in the environment, the `myprofile` profile
    will NOT be excluded from the docker compose --profile <profile> cmd.
    """

    contents: dict[str, object] = yaml.safe_load(version.compose_file.read_text())
    services = t.cast("dict[str, object]", contents.get("services", {}) or {})
    profiles_to_enable: set[str] = set()
    for service in services.values():
        if not isinstance(service, dict):
            continue

        profiles = service.get("profiles", [])
        if not profiles or not isinstance(profiles, list):
            continue

        x_override_vars = service.get("x-profile-override-vars", [])
        if not x_override_vars or not isinstance(x_override_vars, list):
            profiles_to_enable.update(profiles)
            continue

        configuration_file = version.configure_overrides_env_file
        overrides_file = version.arg_overrides_env_file

        env_vars: dict[str, str] = {}
        if configuration_file.exists():
            env_vars.update(read_env_file(configuration_file))
        if overrides_file.exists():
            env_vars.update(read_env_file(overrides_file))

        # check if any of the override vars are set in the env
        if any(var in env_vars for var in x_override_vars):
            continue  # skip enabling this profile

        profiles_to_enable.update(profiles)

    return list(profiles_to_enable)


def get_compose_args(
    version: LocalVersion, *, project_name: str = "dreadnode-platform"
) -> list[str]:
    command = ["-p", project_name]
    compose_files = [version.compose_file]
    env_files = [version.api_env_file, version.ui_env_file]

    if (
        version.configure_overrides_compose_file.exists()
        and version.configure_overrides_env_file.exists()
    ):
        compose_files.append(version.configure_overrides_compose_file)
        env_files.append(version.configure_overrides_env_file)

    for compose_file in compose_files:
        command.extend(["-f", compose_file.as_posix()])

    for profile in get_profiles_to_enable(version):
        command.extend(["--profile", profile])

    if version.arg_overrides_env_file.exists():
        env_files.append(version.arg_overrides_env_file)

    for env_file in env_files:
        command.extend(["--env-file", env_file.as_posix()])

    return command


def platform_is_running(version: LocalVersion) -> bool:
    """
    Check if the platform with the specified or current version is running.

    Args:
        version: LocalVersionSchema of the platform to check.
    """
    containers = docker_compose_ps(get_compose_args(version))
    if not containers:
        return False

    for service in get_required_services(version):
        if service not in [c.name for c in containers if c.state == "running"]:
            return False

    return True


def compose_up(version: LocalVersion) -> None:
    docker_run(
        ["compose", *get_compose_args(version), "up", "-d"],
    )


def compose_down(version: LocalVersion, *, remove_volumes: bool = False) -> None:
    args = ["compose", *get_compose_args(version), "down"]
    if remove_volumes:
        args.append("--volumes")
    docker_run(args)


def compose_logs(version: LocalVersion, *, tail: int = 100) -> None:
    docker_run(["compose", *get_compose_args(version), "logs", "--tail", str(tail)])


def compose_login(version: LocalVersion, *, force: bool = False) -> None:
    # check to see if all required images are available locally
    required_images = get_required_images(version)
    available_images = get_available_local_images()
    missing_images = [img for img in required_images if img not in available_images]
    if not missing_images and not force:
        return

    client = create_api_client()
    registry_credentials = client.get_platform_registry_credentials()

    registries_attempted = set()
    for image in version.images:
        if image.registry not in registries_attempted:
            print_info(f"Logging in to Docker registry: {image.registry} ...")
            docker_login(
                image.registry, registry_credentials.username, registry_credentials.password
            )
            registries_attempted.add(image.registry)

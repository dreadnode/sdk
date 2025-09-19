import json
import subprocess
import typing as t
from dataclasses import dataclass
from enum import Enum

import yaml
from pydantic import BaseModel, Field
from yaml import safe_dump

from dreadnode.cli.api import create_api_client
from dreadnode.cli.platform.constants import DEFAULT_DOCKER_PROJECT_NAME, PlatformService
from dreadnode.cli.platform.schemas import LocalVersionSchema
from dreadnode.cli.platform.utils.env_mgmt import read_env_file
from dreadnode.cli.platform.utils.printing import print_error, print_info, print_success

DockerContainerState = t.Literal[
    "running", "exited", "paused", "restarting", "removing", "created", "dead"
]


# create a DockerError exception that I can catch
class DockerError(Exception):
    pass


class CaptureOutput(str, Enum):
    TRUE = "true"
    FALSE = "false"


@dataclass
class DockerImage:
    repository: str
    tag: str | None = None
    digest: str | None = None

    @classmethod
    def from_string(cls, image_string: str) -> "DockerImage":
        """
        Parse a Docker image string into repository, tag, and SHA components.

        Examples:
        - postgres:16 -> repository="postgres", tag="16", sha=None
        - minio/minio:latest -> repository="minio/minio", tag="latest", sha=None
        - image@sha256:abc123 -> repository="image", tag=None, sha="sha256:abc123"
        """
        # Check if there's a SHA digest (contains @)
        if "@" in image_string:
            repo_part, sha = image_string.split("@", 1)
            # Check if there's also a tag before the @
            if ":" in repo_part:
                repository, tag = repo_part.rsplit(":", 1)
                return cls(repository=repository, tag=tag, digest=sha)
            return cls(repository=repo_part, tag=None, digest=sha)

        # Check if there's a tag (contains :)
        if ":" in image_string:
            # Use rsplit to handle cases like registry.com:5000/image:tag
            repository, tag = image_string.rsplit(":", 1)
            return cls(repository=repository, tag=tag, digest=None)

        # Just repository name
        return cls(repository=image_string, tag=None, digest=None)

    def __str__(self) -> str:
        """Reconstruct the original image string format."""
        result = self.repository
        if self.tag:
            result += f":{self.tag}"
        if self.digest:
            result += f"@{self.digest}"
        return result

    def __eq__(self, other: object) -> bool:
        """Check if two DockerImage instances are equal.

        If they both have digests, compare digests.
        If they both have tags, compare tags.

        """
        if not isinstance(other, DockerImage):
            return False
        if self.repository != other.repository:
            return False
        if self.digest and other.digest:
            return self.digest == other.digest
        if self.tag and other.tag:
            return self.tag == other.tag
        return False

    def __ne__(self, other: object) -> bool:
        """Check if two DockerImage instances are not equal."""
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """Generate a hash for the DockerImage instance."""
        if self.tag:
            return hash((self.repository, self.tag))
        if self.digest:
            return hash((self.repository, self.digest))
        return hash((self.repository,))


class DockerPSResult(BaseModel):
    name: str = Field(..., alias="Name")
    exit_code: int = Field(..., alias="ExitCode")
    state: DockerContainerState = Field(..., alias="State")
    status: str = Field(..., alias="Status")

    @property
    def is_running(self) -> bool:
        return self.state == "running"


def _build_docker_compose_base_command(
    selected_version: LocalVersionSchema,
) -> list[str]:
    cmds = []
    compose_files = [selected_version.compose_file]
    env_files = [
        selected_version.api_env_file,
        selected_version.ui_env_file,
    ]

    if (
        selected_version.configure_overrides_compose_file.exists()
        and selected_version.configure_overrides_env_file.exists()
    ):
        compose_files.append(selected_version.configure_overrides_compose_file)
        env_files.append(selected_version.configure_overrides_env_file)

    for compose_file in compose_files:
        cmds.extend(["-f", compose_file.as_posix()])

    for profile in _get_profiles_to_enable(selected_version):
        cmds.extend(["--profile", profile])

    if selected_version.arg_overrides_env_file.exists():
        env_files.append(selected_version.arg_overrides_env_file)

    for env_file in env_files:
        cmds.extend(["--env-file", env_file.as_posix()])
    return cmds


def _check_docker_installed() -> bool:
    """Check if Docker is installed on the system."""
    try:
        cmd = ["docker", "--version"]
        subprocess.run(  # noqa: S603
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    except subprocess.CalledProcessError:
        print_error("Docker is not installed. Please install Docker and try again.")
        return False

    return True


def _check_docker_compose_installed() -> bool:
    """Check if Docker Compose is installed on the system."""
    try:
        cmd = ["docker", "compose", "--version"]
        subprocess.run(  # noqa: S603
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        print_error("Docker Compose is not installed. Please install Docker Compose and try again.")
        return False
    return True


def get_required_service_names(selected_version: LocalVersionSchema) -> list[str]:
    """Get the list of require service names from the docker-compose file."""
    contents: dict[str, t.Any] = yaml.safe_load(selected_version.compose_file.read_text())
    services = contents.get("services", {}) or {}
    return [name for name, cfg in services.items() if isinstance(cfg, dict) and "x-required" in cfg]


def _get_profiles_to_enable(selected_version: LocalVersionSchema) -> list[str]:
    """Get the list of profiles to enable based on environment variables.

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

    Args:
        selected_version: The selected version of the platform.

    Returns:
        List of profile names to enable.
    """

    contents: dict[str, t.Any] = yaml.safe_load(selected_version.compose_file.read_text())
    services = contents.get("services", {}) or {}
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

        configuration_file = selected_version.configure_overrides_env_file
        overrides_file = selected_version.arg_overrides_env_file
        env_vars = {}
        if configuration_file.exists():
            env_vars.update(read_env_file(configuration_file))
        if overrides_file.exists():
            env_vars.update(read_env_file(overrides_file))
        # check if any of the override vars are set in the env
        if any(var in env_vars for var in x_override_vars):
            continue  # skip enabling this profile
        profiles_to_enable.update(profiles)

    return list(profiles_to_enable)


def _run_docker_compose_command(
    args: list[str],
    timeout: int = 300,
    stdin_input: str | None = None,
    capture_output: CaptureOutput | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute a docker compose command with common error handling and configuration.

    Args:
        args: Additional arguments for the docker compose command.
        compose_file: Path to docker-compose file.
        timeout: Command timeout in seconds.
        command_name: Name of the command for error messages.
        stdin_input: Input to pass to stdin (for commands like docker login).

    Returns:
        CompletedProcess object with command results.

    Raises:
        subprocess.CalledProcessError: If command fails.
        subprocess.TimeoutExpired: If command times out.
        FileNotFoundError: If docker/docker-compose not found.
    """
    cmd = ["docker", "compose"]

    cmd.extend(["-p", DEFAULT_DOCKER_PROJECT_NAME])

    # Add the specific command arguments
    cmd.extend(args)

    cmd_str = " ".join(cmd)

    try:
        # Remove capture_output=True to allow real-time streaming
        # stdout and stderr will go directly to the terminal
        result = subprocess.run(  # noqa: S603
            cmd,
            check=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            input=stdin_input,
            capture_output=bool(capture_output == CaptureOutput.TRUE),
        )

    except subprocess.CalledProcessError as e:
        print_error(f"{cmd_str} failed with exit code {e.returncode}")
        raise DockerError(f"Docker command failed: {e}") from e

    except subprocess.TimeoutExpired as e:
        print_error(f"{cmd_str} timed out after {timeout} seconds")
        raise DockerError(f"Docker command timed out after {timeout} seconds") from e

    except FileNotFoundError as e:
        print_error("Docker or docker compose not found. Please ensure Docker is installed.")
        raise DockerError(f"Docker compose file not found: {e}") from e

    return result


def build_docker_compose_override_file(
    services: list[PlatformService],
    selected_version: LocalVersionSchema,
) -> None:
    # build a yaml docker compose override file
    # that only includes the service being configured
    # and has an `env_file` attribute for the service
    override = {
        "services": {
            f"{service}": {"env_file": [selected_version.configure_overrides_env_file.as_posix()]}
            for service in services
        },
    }

    with selected_version.configure_overrides_compose_file.open("w") as f:
        safe_dump(override, f, sort_keys=False)


def get_available_local_images() -> list[DockerImage]:
    """Get the list of available Docker images on the local system.

    Returns:
        list[str]: List of available Docker image names.
    """
    cmd = ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}@{{.Digest}}"]
    cp = subprocess.run(  # noqa: S603
        cmd,
        check=True,
        text=True,
        capture_output=True,
    )
    images: list[DockerImage] = []
    for line in cp.stdout.splitlines():
        if line.strip():
            img = DockerImage.from_string(line.strip())
            images.append(img)
    return images


def get_env_var_from_container(container_name: str, var_name: str) -> str | None:
    """
    Get the specified environment variable from the container and return
    its value.

    Args:
        container_name: Name of the container to inspect.
        var_name: Name of the environment variable to retrieve.

    Returns:
        str | None: Value of the environment variable, or None if not found.
    """
    try:
        cmd = [
            "docker",
            "inspect",
            "-f",
            "{{range .Config.Env}}{{println .}}{{end}}",
            container_name,
        ]
        cp = subprocess.run(  # noqa: S603
            cmd,
            check=True,
            text=True,
            capture_output=True,
        )

        for line in cp.stdout.splitlines():
            if line.startswith(f"{var_name.upper()}="):
                return line.split("=", 1)[1]

    except subprocess.CalledProcessError:
        return None

    return None


def get_required_images(selected_version: LocalVersionSchema) -> list[DockerImage]:
    """Get the list of required Docker images for the specified platform version.

    Args:
        selected_version: The selected version of the platform.

    Returns:
        list[str]: List of required Docker image names.
    """
    base = _build_docker_compose_base_command(selected_version)
    args = [*base, "config", "--images"]
    result = _run_docker_compose_command(
        args,
        timeout=120,
        capture_output=CaptureOutput.TRUE,
    )

    if result.returncode != 0:
        return []

    required_images: list[DockerImage] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        # Validate each line is a valid Docker image string
        DockerImage.from_string(line.strip())
        required_images.append(DockerImage.from_string(line.strip()))

    return required_images


def docker_requirements_met() -> bool:
    """Check if Docker and Docker Compose are installed."""
    return _check_docker_installed() and _check_docker_compose_installed()


def docker_login(registry: str) -> None:
    """Log into a Docker registry using API credentials.

    Args:
        registry: Registry hostname to log into.

    Raises:
        subprocess.CalledProcessError: If docker login command fails.
    """

    print_info(f"Logging in to Docker registry: {registry} ...")
    client = create_api_client()
    container_registry_creds = client.get_container_registry_credentials()

    cmd = ["docker", "login", container_registry_creds.registry]
    cmd.extend(["--username", container_registry_creds.username])
    cmd.extend(["--password-stdin"])

    try:
        subprocess.run(  # noqa: S603
            cmd,
            input=container_registry_creds.password,
            text=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print_success("Logged in to container registry ...")
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to log in to container registry: {e}")
        raise


def docker_ps(
    selected_version: LocalVersionSchema,
    timeout: int = 120,
) -> list[DockerPSResult]:
    """Get container status for the compose project as JSON.

    This mirrors:
        docker compose -f <...> -f <...> --env-file <...> --env-file <...> ps --format json [SERVICE...]

    Args:
        selected_version: Version object providing compose/env files.
        services: Optional list of PlatformService to filter (translated to 'platform-<service>').
        timeout: Command timeout in seconds.

    Returns:
        A list of dicts parsed from `docker compose ps --format json`.

    Raises:
        ValueError: If the returned output is not valid JSON.
        subprocess.CalledProcessError / TimeoutExpired / FileNotFoundError: On execution errors.
    """
    base = _build_docker_compose_base_command(selected_version)
    args = [*base, "ps", "--format", "json"]

    result = _run_docker_compose_command(
        args,
        timeout=timeout,
        capture_output=CaptureOutput.TRUE,
    )

    try:
        # docker compose ps --format json returns a JSON array
        if not result.stdout:
            return []
        stdout = str(result.stdout)
        stdout_lines = stdout.splitlines()
        container_info_models: list[DockerPSResult] = []
        for line in stdout_lines:
            if not line.strip():
                continue
            j = json.loads(line)
            dpr = DockerPSResult(**j)
            container_info_models.append(dpr)
    except json.JSONDecodeError as e:
        print_error(f"Failed read status from the Dreadnode Platform': {e}")
        raise ValueError("Unexpected non-JSON output from 'docker compose ps'") from e

    return container_info_models


def docker_run(
    selected_version: LocalVersionSchema,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    """Run docker containers for the platform.

    Args:
        compose_file: Path to docker-compose file.
        timeout: Command timeout in seconds.

    Returns:
        CompletedProcess object with command results.

    Raises:
        subprocess.CalledProcessError: If command fails.
        subprocess.TimeoutExpired: If command times out.
    """
    cmds = _build_docker_compose_base_command(selected_version)

    # Apply the compose and env override files in priority order
    # 1. base compose file and env files
    # 2. configure overrides compose and env files (if any)
    # 3. arg overrides env file (if any)

    cmds += ["up", "-d"]
    return _run_docker_compose_command(cmds, timeout=timeout)


def docker_stop(
    selected_version: LocalVersionSchema,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    """Stop docker containers for the platform.

    Args:
        selected_version: The selected version of the platform.
        timeout: Command timeout in seconds.

    Returns:
        CompletedProcess object with command results.

    Raises:
        subprocess.CalledProcessError: If command fails.
        subprocess.TimeoutExpired: If command times out.
    """
    cmds = _build_docker_compose_base_command(selected_version)
    cmds.append("down")
    return _run_docker_compose_command(cmds, timeout=timeout)

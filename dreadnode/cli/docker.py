import re
import subprocess  # nosec
import typing as t

from pydantic import AliasChoices, BaseModel, Field

from dreadnode.common_types import UNSET, Unset
from dreadnode.constants import (
    DEFAULT_DOCKER_REGISTRY_IMAGE_TAG,
    DEFAULT_DOCKER_REGISTRY_LOCAL_PORT,
    DEFAULT_DOCKER_REGISTRY_SUBDOMAIN,
    DEFAULT_PLATFORM_BASE_DOMAIN,
)
from dreadnode.logging_ import print_info
from dreadnode.user_config import ServerConfig

DockerContainerState = t.Literal[
    "running", "exited", "paused", "restarting", "removing", "created", "dead"
]


class DockerError(Exception):
    pass


class DockerImage(str):  # noqa: SLOT000
    """
    A string subclass that normalizes and parses various Docker image string formats.

    Supported formats:
    - ubuntu
    - ubuntu:22.04
    - library/ubuntu:22.04
    - docker.io/library/ubuntu:22.04
    - myregistry:5000/my/image:latest
    - myregistry:5000/my/image@sha256:f6e42a...
    - dreadnode/image (correctly parsed as a Docker Hub image)
    """

    registry: str | None
    repository: str
    tag: str | None
    digest: str | None

    def __new__(cls, value: str, *_: t.Any, **__: t.Any) -> "DockerImage":
        value = value.strip()
        if not value:
            raise ValueError("Invalid Docker image format: input cannot be empty")

        # 1. Separate digest from the rest
        digest: str | None = None
        if "@" in value:
            value, digest = value.split("@", 1)

        # 2. Separate tag from the repository path
        tag: str | None = None
        repo_path = value
        # A tag is present if there's a colon that is NOT part of a port number in a registry hostname
        if ":" in value:
            possible_repo, possible_tag = value.rsplit(":", 1)
            # If the part before the colon contains a slash, or no slash at all, it's a tag.
            # This correctly handles "ubuntu:22.04" and "gcr.io/my/image:tag" but not "localhost:5000/image".
            if "/" in possible_repo or "/" not in value:
                repo_path, tag = possible_repo, possible_tag

        if not repo_path:
            raise ValueError("Invalid Docker image format: missing repository name")

        # 3. Determine the registry and the final repository name
        registry: str | None = None
        repository = repo_path

        if "/" not in repo_path:
            # Case 1: An official image like "ubuntu".
            repository = f"library/{repo_path}"
        else:
            # Case 2: A namespaced path. It could be "dreadnode/image"
            # or "gcr.io/google-containers/busybox".
            first_part = repo_path.split("/", 1)[0]
            if "." in first_part or ":" in first_part:
                # If the first part has a "." or ":", it's a registry hostname.
                registry = first_part
                repository = repo_path.split("/", 1)[1]

        # 4. Default to 'latest' tag if no tag or digest is provided
        if not tag and not digest:
            tag = "latest"

        # 5. Construct the full, normalized string for the object's value
        full_image_str = repository
        if registry:
            full_image_str = f"{registry}/{repository}"

        if tag:
            full_image_str += f":{tag}"
        if digest:
            full_image_str += f"@{digest}"

        obj = super().__new__(cls, full_image_str)
        obj.registry = registry
        obj.repository = repository
        obj.tag = tag
        obj.digest = digest

        return obj

    def __repr__(self) -> str:
        parts = [f"repository='{self.repository}'"]
        if self.registry:
            parts.append(f"registry='{self.registry}'")
        if self.tag:
            parts.append(f"tag='{self.tag}'")
        if self.digest:
            parts.append(f"digest='{self.digest}'")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    def with_(
        self,
        *,
        repository: str | Unset = UNSET,
        registry: str | None | Unset = UNSET,
        tag: str | None | Unset = UNSET,
        digest: str | None | Unset = UNSET,
    ) -> "DockerImage":
        """
        Create a new DockerImage instance with updated elements.
        """
        new_registry = registry if not isinstance(registry, Unset) else self.registry
        new_repository = repository if not isinstance(repository, Unset) else self.repository
        new_tag = tag if not isinstance(tag, Unset) else self.tag
        new_digest = digest if not isinstance(digest, Unset) else self.digest

        new_image = new_repository
        if new_registry:
            new_image = f"{new_registry}/{new_repository}"

        if new_tag:
            new_image += f":{new_tag}"
        if new_digest:
            new_image += f"@{new_digest}"

        return DockerImage(new_image)


class DockerContainer(BaseModel):
    id: str = Field(..., alias="ID")
    name: str = Field(..., validation_alias=AliasChoices("Name", "Names"))
    exit_code: int = Field(-1, alias="ExitCode")
    state: DockerContainerState = Field(..., alias="State")
    status: str = Field(..., alias="Status")
    raw_ports: str = Field(..., alias="Ports")
    image: str = Field(..., alias="Image")
    command: str = Field(..., alias="Command")

    @property
    def is_running(self) -> bool:
        return self.state == "running"

    @property
    def ports(self) -> list[tuple[int, int]]:
        """
        Parse the raw_ports string into a list of tuples mapping host ports to container ports.
        """
        ports = []
        for mapping in self.raw_ports.split(","):
            host_part, container_part = mapping.split("->")
            host_port = int(host_part.split(":")[-1])
            container_port = int(container_part.split("/")[0])
            ports.append((host_port, container_port))
        return ports


def docker_run(
    args: list[str],
    *,
    timeout: int = 300,
    stdin_input: str | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """
    Execute a docker command with common error handling and configuration.

    Args:
        args: Additional arguments for the docker command.
        timeout: Command timeout in seconds.
        stdin_input: Optional input string to pass to the command's stdin.
        capture_output: Whether to capture the command's output.

    Returns:
        CompletedProcess object with command results.

    Raises:
        subprocess.CalledProcessError: If command fails.
        subprocess.TimeoutExpired: If command times out.
        FileNotFoundError: If docker/docker-compose not found.
    """
    try:
        result = subprocess.run(  # noqa: S603 # nosec
            ["docker", *args],  # noqa: S607
            check=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            input=stdin_input,
            capture_output=capture_output,
        )
        if not capture_output:
            print_info("")  # Some padding after command output

    except subprocess.CalledProcessError as e:
        command_str = " ".join(e.cmd)
        error = f"Docker command failed: {command_str}"
        if e.stderr:
            error += f"\n\n{e.stderr}"
        raise DockerError(error) from e

    except subprocess.TimeoutExpired as e:
        command_str = " ".join(e.cmd)
        raise DockerError(f"Docker command timed out after {timeout} seconds: {command_str}") from e

    except FileNotFoundError as e:
        raise DockerError(
            "`docker` not found, please ensure it is installed and in your PATH."
        ) from e

    return result


def get_available_local_images() -> list[DockerImage]:
    """
    Get the list of available Docker images on the local system.

    Returns:
        List of available Docker image names.
    """
    result = docker_run(
        ["images", "--format", "{{.Repository}}:{{.Tag}}@{{.Digest}}"],
        capture_output=True,
        timeout=30,
    )
    return [DockerImage(line) for line in result.stdout.splitlines() if line.strip()]


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
    result = docker_run(
        ["inspect", "-f", "{{range .Config.Env}}{{println .}}{{end}}", container_name],
        capture_output=True,
        timeout=30,
    )
    for line in result.stdout.splitlines():
        if line.startswith(f"{var_name.upper()}="):
            return line.split("=", 1)[1]
    return None


def docker_login(registry: str, username: str, password: str) -> None:
    """
    Log into a Docker registry.

    Args:
        registry: Registry hostname to log into.
        username: Username for the registry.
        password: Password for the registry.
    """
    docker_run(
        ["login", registry, "--username", username, "--password-stdin"],
        stdin_input=password,
        capture_output=True,
        timeout=60,
    )


def docker_ps() -> list[DockerContainer]:
    """
    List and parse running containers using `docker ps`.

    Returns:
        A list of DockerPSResult objects.
    """
    result = docker_run(
        ["ps", "--format", "json"],
        capture_output=True,
    )
    return [
        DockerContainer.model_validate_json(line)
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def docker_compose_ps(args: list[str] | None = None) -> list[DockerContainer]:
    """
    List and parse running containers using `docker compose ps`.

    This mirrors:
        docker compose [*args] ps --format json

    Args:
        args: Additional docker compose arguments.

    Returns:
        A list of DockerPSResult objects.
    """
    result = docker_run(
        ["compose", *(args or []), "ps", "--format", "json"],
        capture_output=True,
    )
    return [
        DockerContainer.model_validate_json(line)
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def docker_tag(image: str | DockerImage, new_tag: str) -> None:
    """
    Tag a Docker image with a new tag.

    Args:
        image: The name of the image to tag.
        new_tag: The new tag to apply to the image.
    """
    docker_run(
        ["tag", str(image), new_tag],
        capture_output=True,
        timeout=60,
    )


def get_local_registry_port() -> int:
    for container in docker_ps():
        if DEFAULT_DOCKER_REGISTRY_IMAGE_TAG in container.image and container.ports:
            # return the first mapped port
            return container.ports[0][0]

    # fallback to the default port
    return DEFAULT_DOCKER_REGISTRY_LOCAL_PORT


def get_registry(config: ServerConfig) -> str:
    # localhost is a special case
    if "localhost" in config.url or "127.0.0.1" in config.url:
        return f"localhost:{get_local_registry_port()}"

    prefix = ""
    if "staging-" in config.url:
        prefix = "staging-"
    elif "dev-" in config.url:
        prefix = "dev-"

    return f"{prefix}{DEFAULT_DOCKER_REGISTRY_SUBDOMAIN}.{DEFAULT_PLATFORM_BASE_DOMAIN}"


def docker_pull(image: str | DockerImage) -> None:
    """
    Pull a Docker image.

    Args:
        image: The name of the image to pull.
    """
    docker_run(["pull", image])


def clean_username(name: str) -> str:
    """
    Sanitizes an agent or user name to be used in a Docker repository URI.
    """
    # convert to lowercase
    name = name.lower()
    # replace non-alphanumeric characters with hyphens
    name = re.sub(r"[^\w\s-]", "", name)
    # replace one or more whitespace characters with a single hyphen
    name = re.sub(r"[-\s]+", "-", name)
    # remove leading or trailing hyphens
    return name.strip("-")

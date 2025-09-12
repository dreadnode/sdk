import json
import subprocess
import time
from pathlib import Path

from yaml import safe_dump

from dreadnode.cli.api import create_api_client
from dreadnode.cli.platform.constants import SERVICES, PlatformService
from dreadnode.cli.platform.schemas import LocalVersionSchema
from dreadnode.cli.platform.utils.printing import print_error, print_info, print_success


def _run_docker_compose_command(
    args: list[str],
    # compose_file: Path,
    timeout: int = 300,
    stdin_input: str | None = None,
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

    # Add compose file
    # cmd.extend(["-f", compose_file.as_posix()])

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
        )

    except subprocess.CalledProcessError as e:
        print_error(f"{cmd_str} failed with exit code {e.returncode}")
        raise

    except subprocess.TimeoutExpired:
        print_error(f"{cmd_str} timed out after {timeout} seconds")
        raise

    except FileNotFoundError:
        print_error("Docker or docker compose not found. Please ensure Docker is installed.")
        raise

    return result


def get_origin(ui_container: str) -> str | None:
    """
    Get the ORIGIN environment variable from the UI container and return
    a friendly message for the user.

    Args:
        ui_container: Name of the UI container (default: dreadnode-ui).

    Returns:
        str | None: Message with the origin URL, or None if not found.
    """
    try:
        cmd = [
            "docker",
            "inspect",
            "-f",
            "{{range .Config.Env}}{{println .}}{{end}}",
            ui_container,
        ]
        cp = subprocess.run(  # noqa: S603
            cmd,
            check=True,
            text=True,
            capture_output=True,
        )

        for line in cp.stdout.splitlines():
            if line.startswith("ORIGIN="):
                return line.split("=", 1)[1]

    except subprocess.CalledProcessError:
        return None

    return None


def _check_docker_creds_exist(registry: str) -> bool:
    """Check if Docker credentials exist for the specified registry.

    Args:
        registry: Registry hostname to check credentials for.

    Returns:
        bool: True if credentials exist, False otherwise.
    """
    config_path = Path.home() / ".docker" / "config.json"

    if not config_path.exists():
        return False

    try:
        with config_path.open() as f:
            config = json.load(f)

        auths = config.get("auths", {})
    except (json.JSONDecodeError, KeyError):
        return False
    return registry in auths


def _are_docker_creds_fresh(registry: str, max_age_hours: int = 1) -> bool:
    """Check if Docker credentials are fresh (recently updated).

    Args:
        registry: Registry hostname to check credentials for.
        max_age_hours: Maximum age in hours for credentials to be considered fresh.

    Returns:
        bool: True if credentials are fresh, False otherwise.
    """
    config_path = Path.home() / ".docker" / "config.json"

    if not config_path.exists():
        return False

    # Check file modification time
    mtime = config_path.stat().st_mtime
    age_hours = (time.time() - mtime) / 3600

    return age_hours < max_age_hours and _check_docker_creds_exist(registry)


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


def _build_docker_compose_base_command(
    selected_version: LocalVersionSchema,
) -> list[str]:
    cmds = []
    compose_files = [selected_version.compose_file]
    env_files = [selected_version.api_env_file, selected_version.ui_env_file]

    if (
        selected_version.configure_overrides_compose_file.exists()
        and selected_version.configure_overrides_env_file.exists()
    ):
        compose_files.append(selected_version.configure_overrides_compose_file)
        env_files.append(selected_version.configure_overrides_env_file)

    for compose_file in compose_files:
        cmds.extend(["-f", compose_file.as_posix()])

    if selected_version.arg_overrides_env_file.exists():
        env_files.append(selected_version.arg_overrides_env_file)

    for env_file in env_files:
        cmds.extend(["--env-file", env_file.as_posix()])
    return cmds


def build_docker_compose_override_file(
    services: list[PlatformService],
    selected_version: LocalVersionSchema,
) -> None:
    # build a yaml docker compose override file
    # that only includes the service being configured
    # and has an `env_file` attribute for the service
    override = {
        "version": "3.8",
        "services": {
            f"platform-{service}": {
                "env_file": [selected_version.configure_overrides_env_file.as_posix()]
            }
            for service in services
        },
    }

    with selected_version.configure_overrides_compose_file.open("w") as f:
        safe_dump(override, f, sort_keys=False)


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
    if _are_docker_creds_fresh(registry):
        print_info(f"Docker credentials for {registry} are fresh. Skipping login.")
        return

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
    return _run_docker_compose_command(cmds, timeout, "Docker compose up")


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
    return _run_docker_compose_command(cmds, timeout, "Docker compose down")

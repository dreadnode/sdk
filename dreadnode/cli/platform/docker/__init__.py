import subprocess
import sys

import rich

from dreadnode.cli.platform.utils import get_compose_file_path


def run_docker_compose_command(
    args: list[str],
    compose_file: str | None = None,
    project_name: str | None = None,
    timeout: int = 300,
    command_name: str = "docker compose",
    stdin_input: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """
    Execute a docker compose command with common error handling and configuration.

    Args:
        args: Additional arguments for the docker compose command
        compose_file: Path to docker-compose file (optional)
        project_name: Docker compose project name (optional)
        timeout: Command timeout in seconds
        command_name: Name of the command for error messages
        stdin_input: Input to pass to stdin (for commands like docker login)

    Returns:
        CompletedProcess object with command results

    Raises:
        subprocess.CalledProcessError: If command fails
        subprocess.TimeoutExpired: If command times out
        FileNotFoundError: If docker/docker-compose not found
    """
    cmd = ["docker", "compose"]

    # Add compose file
    compose_file = compose_file or get_compose_file_path()
    cmd.extend(["-f", compose_file])

    # Add project name if specified
    if project_name:
        cmd.extend(["-p", project_name])

    # Add the specific command arguments
    cmd.extend(args)

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
        rich.print(f"{command_name} failed with exit code {e.returncode}", file=sys.stderr)
        raise

    except subprocess.TimeoutExpired:
        rich.print(f"{command_name} timed out after {timeout} seconds", file=sys.stderr)
        raise

    except FileNotFoundError:
        rich.print(
            "Docker or docker compose not found. Please ensure Docker is installed.",
            file=sys.stderr,
        )
        raise

    return result

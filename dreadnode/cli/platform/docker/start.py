import rich

from dreadnode.cli.platform.docker import run_docker_compose_command


def _start_infra(
    compose_file: str | None = None, project_name: str | None = None, timeout: int = 300
) -> None:
    """Start infrastructure services."""
    run_docker_compose_command(
        ["up", "-d"], compose_file, project_name, timeout, "Docker compose up (infra)"
    )


def _create_storage(
    compose_file: str | None = None, project_name: str | None = None, timeout: int = 300
) -> None:
    """Create S3 buckets."""
    run_docker_compose_command(
        ["--profile", "create-s3-buckets", "up", "-d"],
        compose_file,
        project_name,
        timeout,
        "Docker compose up (storage)",
    )


def _start_services(
    compose_file: str | None = None, project_name: str | None = None, timeout: int = 300
) -> None:
    """Start application services."""
    run_docker_compose_command(
        ["--profile", "run", "up", "-d"],
        compose_file,
        project_name,
        timeout,
        "Docker compose up (services)",
    )


def start(
    compose_file: str | None = None, project_name: str | None = None, timeout: int = 300
) -> None:
    """Start all platform services."""
    rich.print("Starting platform services...")
    _start_infra(compose_file, project_name, timeout)
    _create_storage(compose_file, project_name, timeout)
    _start_services(compose_file, project_name, timeout)


def stop(
    compose_file: str | None = None, project_name: str | None = None, timeout: int = 300
) -> None:
    """Stop platform services."""
    rich.print("Stopping platform services...")
    run_docker_compose_command(["stop"], compose_file, project_name, timeout, "Docker compose stop")

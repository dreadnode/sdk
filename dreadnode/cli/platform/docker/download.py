import subprocess

from dreadnode.cli.platform.docker import run_docker_compose_command

# def download_platform(
#     registry: str, username: str, password: str, image_name: str, tag: str
# ) -> Image:
#     try:
#         import docker  # type: ignore[import-untyped,unused-ignore]
#     except ImportError as e:
#         raise ImportError(
#             "Running a local platform requires `docker`. Install with: pip install dreadnode\\[platform]"
#         ) from e

#     # Initialize Docker client
#     client = docker.from_env()

#     # # Method 1: Login first, then pull
#     # client.login(
#     #     username=username,
#     #     password=password,
#     #     registry=registry,
#     # )

#     # # Pull the private image
#     # image = client.images.pull(f"{registry}/{image}:{tag}")

#     # Method 2: Pull with auth parameter
#     return client.images.pull(
#         f"{registry}/{image_name}:{tag}",
#         auth_config={"username": username, "password": password},
#     )


# def parse_compose_file(compose_file_path: str) -> dict[str, Any]:
#     """Parse Docker Compose file with proper error handling."""
#     try:
#         with Path(compose_file_path).open("r", encoding="utf-8") as f:
#             compose_config = yaml.safe_load(f)

#         if not compose_config:
#             raise ValueError("Empty or invalid compose file")

#         # Validate basic structure
#         if not isinstance(compose_config, dict):
#             raise TypeError("Compose file must contain a YAML mapping")

#     except yaml.YAMLError as e:
#         raise ValueError(f"Invalid YAML syntax: {e}") from e
#     except FileNotFoundError as e:
#         raise FileNotFoundError(f"Compose file not found: {compose_file_path}") from e

#     return compose_config


# def pull_images_from_compose(compose_file_path: str) -> None:
#     """Pull all images defined in a Docker Compose file."""

#     # Initialize Docker client
#     try:
#         import docker  # type: ignore[import-untyped,unused-ignore]
#     except ImportError as e:
#         raise ImportError(
#             "Running a local platform requires `docker`. Install with: pip install dreadnode\\[platform]"
#         ) from e

#     # Initialize Docker client
#     client = docker.from_env()

#     compose_config = parse_compose_file(compose_file_path)

#     # Handle different compose file versions
#     services = compose_config.get("services", {})

#     if not services:
#         logger.error("No services found in compose file")
#         return

#     for service_name, service_config in services.items():
#         if not isinstance(service_config, dict):
#             logger.warning(f"⚠ Skipping invalid service config for '{service_name}'")
#             continue

#         image = service_config.get("image")
#         if image:
#             try:
#                 logger.info(f"Pulling {image}...")
#                 client.images.pull(image)
#                 logger.success(f"✓ Pulled {image}")
#             except DockerApiError as e:
#                 logger.error(f"✗ Failed to pull {image}: {e}")
#         else:
#             # Handle services with 'build' context instead of 'image'
#             build_config = service_config.get("build")
#             if build_config:
#                 logger.warning(f"⚠ Service '{service_name}' uses build context, skipping pull")
#             else:
#                 logger.warning(f"⚠ Service '{service_name}' has no image or build config")


def download(
    compose_file: str | None = None, project_name: str | None = None, timeout: int = 300
) -> subprocess.CompletedProcess[str]:
    """
    Pull docker images for the platform.

    Args:
        compose_file: Path to docker-compose file (optional)
        project_name: Docker compose project name (optional)
        timeout: Command timeout in seconds

    Returns:
        CompletedProcess object with command results

    Raises:
        subprocess.CalledProcessError: If command fails
        subprocess.TimeoutExpired: If command times out
    """
    return run_docker_compose_command(
        ["--profile", "run", "pull"], compose_file, project_name, timeout, "Docker compose pull"
    )

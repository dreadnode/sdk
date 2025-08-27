import json
from pathlib import Path

import rich
from rich.prompt import Confirm

from dreadnode.api.models import PlatformImage, RegistryImageDetails
from dreadnode.cli.api import create_api_client
from dreadnode.cli.platform.constants import (
    API_ENV_TEMPLATE,
    API_SERVICE,
    DOCKER_COMPOSE_TEMPLATE,
    SERVICES,
    UI_ENV_TEMPLATE,
    UI_SERVICE,
)
from dreadnode.cli.platform.utils import (
    get_compose_file_path,
    get_local_arch,
    get_local_cache_dir,
    render_with_string_replace,
)


def _write_version_manifest(
    local_cache_dir: Path, resolution_response: RegistryImageDetails
) -> None:
    rich.print(f"Writing version file for {resolution_response.version} ...")
    version_file = local_cache_dir / ".version"
    version_file.write_text(json.dumps(resolution_response.model_dump()))
    rich.print(f"Version file written to {version_file}")


def _create_docker_compose_file(images: list[PlatformImage]) -> None:
    rich.print("Updating Compose template ...")
    for image in images:
        if image.service == API_SERVICE:
            api_image_digest = image.full_uri
        elif image.service == UI_SERVICE:
            ui_image_digest = image.full_uri
        else:
            raise ValueError(f"Unknown image service: {image.service}")
    render_with_string_replace(
        api_image_digest=api_image_digest,
        ui_image_digest=ui_image_digest,
        template_path=DOCKER_COMPOSE_TEMPLATE,
        output_path=get_compose_file_path(),
    )
    rich.print(f"Compose file written to {get_compose_file_path()}")


def _create_env_files(local_cache_dir: Path) -> None:
    rich.print("Updating environment files ...")

    for env_file in [API_ENV_TEMPLATE, UI_ENV_TEMPLATE]:
        dest = local_cache_dir / env_file.name
        dest.write_text(env_file.read_text())
        rich.print(f"Environment file written to {dest}")

    # concatenate environment variables
    api_env = local_cache_dir / API_ENV_TEMPLATE.name
    ui_env = local_cache_dir / UI_ENV_TEMPLATE.name
    dest = local_cache_dir / ".env"
    dest.write_text(f"{api_env.read_text()}\n{ui_env.read_text()}")
    rich.print(f"Combined environment file written to {dest}")


def _confirm_with_context(action: str, details: str | None = None) -> bool:
    """Confirmation with additional context in a panel."""
    return Confirm.ask(
        f"[bold red]Are you sure you want to {action}? {details}[/bold red]", default=False
    )


def init(tag: str, arch: str | None = None) -> None:
    if initialized() and not _confirm_with_context(
        "re-initialize the platform", "This will overwrite existing files."
    ):
        return

    import importlib.metadata  # noqa: PLC0415

    local_cache_dir = get_local_cache_dir()
    rich.print(f"Using local cache directory: {local_cache_dir}")

    if not local_cache_dir.exists():
        local_cache_dir.mkdir(parents=True, exist_ok=True)
        rich.print(f"Local cache directory created at {local_cache_dir}")
    else:
        rich.print("Local cache directory already exists.")

    if not arch:
        arch = get_local_arch()
    api_client = create_api_client()
    registry_image_details = api_client.get_platform_releases(
        arch=arch,
        tag=tag,
        services=SERVICES,
        cli_version=importlib.metadata.version("dreadnode"),
    )

    _write_version_manifest(local_cache_dir, registry_image_details)
    _create_docker_compose_file(registry_image_details.images)
    _create_env_files(local_cache_dir)

    rich.print("Initialization complete.")


def initialized() -> bool:
    rich.print("Checking initialization ...")
    local_cache_dir = get_local_cache_dir()
    if not local_cache_dir.exists():
        rich.print("Local cache directory does not exist.")
        return False

    if not (local_cache_dir / "docker-compose.yaml").exists():
        rich.print("Docker Compose file is missing.")
        return False

    if not (local_cache_dir / ".env").exists():
        rich.print("Environment file is missing.")
        return False

    rich.print("All required files are present.")
    return True

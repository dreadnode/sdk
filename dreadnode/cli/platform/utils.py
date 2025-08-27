import json
import platform
import typing as t
from pathlib import Path

from dreadnode.api.models import RegistryImageDetails

archs = t.Literal["amd64", "arm64"]


def get_local_arch() -> archs:
    arch = platform.machine()

    # Check for specific architectures
    if arch in ["x86_64", "AMD64"]:
        return "amd64"
    if arch in ["arm64", "aarch64", "ARM64"]:
        return "arm64"
    raise ValueError(f"Unsupported architecture: {arch}")


def get_local_cache_dir() -> Path:
    return Path.home() / ".dreadnode" / "platform"


def get_local_version() -> RegistryImageDetails | None:
    local_cache_dir = get_local_cache_dir()
    version_file = local_cache_dir / ".version"
    if version_file.exists():
        return RegistryImageDetails(**json.loads(version_file.read_text()))
    return None


def get_compose_file_path() -> Path:
    return get_local_cache_dir() / "docker-compose.yaml"


def render_with_string_replace(
    api_image_digest: str,
    ui_image_digest: str,
    template_path: str,
    output_path: str,
) -> str:
    """
    Simple string replacement - lightest option.
    Works for basic {{ variable }} patterns.
    """

    with Path(template_path).open() as file:
        content = file.read()

    rendered = content.replace("{{ api_image_digest }}", api_image_digest).replace(
        "{{ ui_image_digest }}", ui_image_digest
    )

    if output_path:
        with Path(output_path).open("w") as file:
            file.write(rendered)

    return rendered

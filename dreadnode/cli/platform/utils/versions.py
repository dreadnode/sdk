import importlib.metadata
import json
import platform
from pathlib import Path

from packaging.version import Version
from rich.prompt import Confirm

from dreadnode.cli.platform.constants import (
    SUPPORTED_ARCHITECTURES,
    VERSIONS_MANIFEST,
    SupportedArchitecture,
)
from dreadnode.cli.platform.schemas import LocalVersionSchema, LocalVersionsSchema
from dreadnode.constants import DEFAULT_LOCAL_STORAGE_DIR


def _get_local_arch() -> SupportedArchitecture:
    """Get the local machine architecture in supported format.

    Returns:
        SupportedArchitecture: The architecture as either "amd64" or "arm64".

    Raises:
        ValueError: If the local architecture is not supported.
    """
    arch = platform.machine()

    if arch in ["x86_64", "AMD64"]:
        return "amd64"
    if arch in ["arm64", "aarch64", "ARM64"]:
        return "arm64"
    raise ValueError(f"Unsupported architecture: {arch}")


def get_local_cache_dir() -> Path:
    """Get the local cache directory path for dreadnode platform files.

    Returns:
        Path: Path to the local cache directory (~/<DEFAULT_LOCAL_STORAGE_DIR>/platform).
    """
    return DEFAULT_LOCAL_STORAGE_DIR / "platform"


def get_cli_version() -> str:
    """Get the version of the dreadnode CLI package.

    Returns:
        str | None: The version string of the dreadnode package, or None if not found.
    """
    return importlib.metadata.version("dreadnode")


def confirm_with_context(action: str) -> bool:
    """Prompt the user for confirmation with a formatted action message.

    Args:
        action: The action description to display in the confirmation prompt.

    Returns:
        bool: True if the user confirms, False otherwise. Defaults to False.
    """
    return Confirm.ask(f"[bold blue]{action}[/bold blue]", default=False)


def get_available_local_versions() -> LocalVersionsSchema:
    """Get all available local platform versions from the manifest file.

    Creates the manifest file with an empty schema if it doesn't exist.

    Returns:
        LocalVersionsSchema: Schema containing all available local platform versions.
    """
    try:
        local_cache_dir = get_local_cache_dir()
        manifest_path = local_cache_dir / VERSIONS_MANIFEST
        with manifest_path.open(encoding="utf-8") as f:
            versions_manifest_data = json.load(f)
            return LocalVersionsSchema(**versions_manifest_data)
    except FileNotFoundError:
        # create the file
        local_cache_dir = get_local_cache_dir()
        manifest_path = local_cache_dir / VERSIONS_MANIFEST
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        blank_schema = LocalVersionsSchema(versions=[])
        with manifest_path.open(encoding="utf-8", mode="w") as f:
            json.dump(blank_schema.model_dump(), f)
        return blank_schema


def get_current_version() -> LocalVersionSchema | None:
    """Get the currently active local platform version.

    Returns:
        LocalVersionSchema | None: The current version schema if one is marked as current,
            None otherwise.
    """
    available_local_versions = get_available_local_versions()
    if not available_local_versions.versions:
        return None
    for version in available_local_versions.versions:
        if version.current:
            return version
    return None


def get_local_version(tag: str) -> LocalVersionSchema:
    """Get a specific local platform version by its tag.

    Args:
        tag: The tag of the version to retrieve.

    Returns:
        LocalVersionSchema: The version schema matching the provided tag.

    Raises:
        ValueError: If no version with the specified tag is found.
    """
    available_local_versions = get_available_local_versions()
    for version in available_local_versions.versions:
        if version.tag == tag:
            return version
    raise ValueError(f"No local version found with tag: {tag}")


def mark_current_version(current_version: LocalVersionSchema) -> None:
    """Mark a specific version as the current active version.

    Updates the versions manifest to mark the specified version as current
    and all others as not current.

    Args:
        current_version: The version to mark as current.
    """
    available_local_versions = get_available_local_versions()
    for available_version in available_local_versions.versions:
        if available_version.tag == current_version.tag:
            available_version.current = True
        else:
            available_version.current = False

    local_cache_dir = get_local_cache_dir()
    manifest_path = local_cache_dir / VERSIONS_MANIFEST
    with manifest_path.open(encoding="utf-8", mode="w") as f:
        json.dump(available_local_versions.model_dump(), f, indent=2)


def create_local_latest_tag() -> str:
    """Create a latest tag string for the local architecture.

    Returns:
        str: A tag in the format "latest-{arch}" where arch is the local architecture.
    """
    arch = _get_local_arch()
    return f"latest-{arch}"


def get_semver_from_tag(tag: str) -> str:
    """Extract semantic version from a tag by removing architecture suffix.

    Args:
        tag: The tag string that may contain an architecture suffix.

    Returns:
        str: The tag with any supported architecture suffix removed.
    """
    for arch in SUPPORTED_ARCHITECTURES:
        if arch in tag:
            return tag.replace(f"-{arch}", "")
    return tag


def newer_remote_version(local_version: str, remote_version: str) -> bool:
    """Check if the remote version is newer than the local version.

    Args:
        local_version: The local version string in semantic version format.
        remote_version: The remote version string in semantic version format.

    Returns:
        bool: True if the remote version is newer than the local version, False otherwise.
    """
    # compare the semvers of two versions to see if the remote is "newer"
    return Version(remote_version) > Version(local_version)

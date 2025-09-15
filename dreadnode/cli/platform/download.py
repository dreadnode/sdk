import io
import json
import zipfile

from dreadnode.api.models import RegistryImageDetails
from dreadnode.cli.api import create_api_client
from dreadnode.cli.platform.constants import SERVICES, VERSIONS_MANIFEST
from dreadnode.cli.platform.schemas import LocalVersionSchema
from dreadnode.cli.platform.utils.env_mgmt import (
    create_default_env_files,
)
from dreadnode.cli.platform.utils.printing import (
    print_error,
    print_info,
    print_success,
    print_warning,
)
from dreadnode.cli.platform.utils.versions import (
    confirm_with_context,
    create_local_latest_tag,
    get_available_local_versions,
    get_cli_version,
    get_local_cache_dir,
)


def _resolve_latest(tag: str) -> str:
    """Resolve 'latest' tag to actual version tag from API.

    Args:
        tag: Version tag that contains 'latest'.

    Returns:
        str: Resolved actual version tag.
    """
    api_client = create_api_client()
    release_info = api_client.get_platform_releases(
        tag, services=[str(service) for service in SERVICES], cli_version=get_cli_version()
    )
    return release_info.tag


def _create_local_version_file_structure(
    tag: str, release_info: RegistryImageDetails
) -> LocalVersionSchema:
    """Create local file structure and update manifest for a new version.

    Args:
        tag: Version tag to create structure for.
        release_info: Registry image details from API.

    Returns:
        LocalVersionSchema: Created local version schema.
    """
    available_local_versions = get_available_local_versions()

    # Create a new local version schema
    local_cache_dir = get_local_cache_dir()
    new_version = LocalVersionSchema(
        **release_info.model_dump(),
        local_path=local_cache_dir / tag,
        current=False,
    )

    # Add the new version to the available local versions
    available_local_versions.versions.append(new_version)

    # sort the manifest by semver, newest first
    available_local_versions.versions.sort(key=lambda v: v.tag, reverse=True)

    # update the manifest file
    manifest_path = local_cache_dir / VERSIONS_MANIFEST
    with manifest_path.open(encoding="utf-8", mode="w") as f:
        json.dump(available_local_versions.model_dump(), f, indent=2)

    print_success(f"Updated versions manifest at {manifest_path} with {new_version.tag}")

    if new_version.local_path.exists():
        print_warning(f"Version {tag} already exists locally.")
        if not confirm_with_context("overwrite it?"):
            print_error("Aborting download.")
            return new_version

    # create the directory
    new_version.local_path.mkdir(parents=True, exist_ok=True)

    return new_version


def _download_version_files(tag: str) -> LocalVersionSchema:
    """Download platform version files from API and extract locally.

    Args:
        tag: Version tag to download.

    Returns:
        LocalVersionSchema: Downloaded local version schema.
    """
    api_client = create_api_client()
    release_info = api_client.get_platform_releases(
        tag, services=[str(service) for service in SERVICES], cli_version=get_cli_version()
    )
    zip_content = api_client.get_platform_templates(tag)

    new_local_version = _create_local_version_file_structure(release_info.tag, release_info)

    with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:
        zip_file.extractall(new_local_version.local_path)
        print_success(f"Downloaded version {tag} to {new_local_version.local_path}")

    create_default_env_files(new_local_version)
    return new_local_version


def download_platform(tag: str | None = None) -> LocalVersionSchema:
    """Download platform version if not already available locally.

    Args:
        tag: Version tag to download (supports 'latest').

    Returns:
        LocalVersionSchema: Local version schema for the downloaded/existing version.
    """
    if not tag or tag == "latest":
        # all remote images are tagged with architecture
        tag = create_local_latest_tag()

    if "latest" in tag:
        tag = _resolve_latest(tag)

    # get what's available
    available_local_versions = get_available_local_versions()

    # if there are versions available
    if available_local_versions.versions:
        for available_local_version in available_local_versions.versions:
            if tag == available_local_version.tag:
                print_success(
                    f"Version {tag} is already downloaded at {available_local_version.local_path}"
                )
                return available_local_version

    print_info(f"Version {tag} is not available locally. Attempting to download it now ...")
    return _download_version_files(tag)

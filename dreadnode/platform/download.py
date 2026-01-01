import io
import zipfile

from dreadnode.core.api import create_api_client
from dreadnode.core.integrations.docker import docker_run
from dreadnode.core.log import confirm, logger
from dreadnode.platform.compose import compose_login, get_compose_args
from dreadnode.platform.env_mgmt import (
    create_default_env_files,
)
from dreadnode.platform.tag import add_tag_arch_suffix
from dreadnode.platform.version import (
    LocalVersion,
    VersionConfig,
)
from dreadnode.core.settings import PLATFORM_SERVICES, PLATFORM_STORAGE_DIR


def download_platform(tag: str | None = None) -> LocalVersion:
    """
    Download platform version if not already available locally.

    Args:
        tag: Version tag to download (supports 'latest').

    Returns:
        LocalVersionSchema: Local version schema for the downloaded/existing version.
    """
    version_config = VersionConfig.read()
    api_client = create_api_client()

    # 1 - Resolve the tag

    tag = tag or "latest"
    tag = add_tag_arch_suffix(tag)

    if "latest" in tag:
        tag = api_client.get_platform_releases(
            tag, services=[str(service) for service in PLATFORM_SERVICES]
        ).tag

    if version_config.versions:
        for available_local_version in version_config.versions:
            if tag == available_local_version.tag:
                logger.success(f"[cyan]{tag}[/] is already downloaded.")
                return available_local_version

    logger.info(f"Downloading [cyan]{tag}[/] ...")

    api_client = create_api_client()
    release_info = api_client.get_platform_releases(
        tag, services=[str(service) for service in PLATFORM_SERVICES]
    )

    new_version = LocalVersion(
        **release_info.model_dump(),
        local_path=PLATFORM_STORAGE_DIR / tag,
        current=False,
    )

    if new_version.local_path.exists() and not confirm(
        f"{new_version.local_path} exists, overwrite?"
    ):
        return new_version

    version_config.add_version(new_version)

    zip_content = api_client.get_platform_templates(tag)
    new_version.local_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:
        zip_file.extractall(new_version.local_path)
        logger.success(f"Downloaded [cyan]{tag}[/] to {new_version.local_path}")

    create_default_env_files(new_version)

    logger.info(f"Pulling Docker images for [cyan]{tag}[/] ...")
    compose_login(new_version, force=True)
    docker_run(
        ["compose", *get_compose_args(new_version), "pull"],
    )

    return new_version

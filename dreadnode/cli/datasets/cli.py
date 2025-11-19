from pathlib import Path

import cyclopts

cli = cyclopts.App("dataset", help="Run and manage datasets.")


@cli.command(name="list")
def list() -> None:
    """
    List available datasets on the Dreadnode platform.
    """
    print("Listing datasets available on the Dreadnode platform.")


@cli.command(name="push")
def push(dataset: Path) -> None:
    """
    Push a dataset to the Dreadnode platform.
    """
    print(f"Pushing dataset from {dataset} to Dreadnode platform.")


@cli.command(name="pull")
def pull(dataset_id: str, destination: Path | None = None) -> None:
    """
    Pull a dataset from the Dreadnode platform.
    """

    # def log_artifact(
    #     self,
    #     local_uri: str | Path,
    # ) -> None:
    # """
    #     Logs a local file or directory as an artifact to the object store.
    #     Preserves directory structure and uses content hashing for deduplication.

    #     Args:
    #         local_uri: Path to the local file or directory

    #     Returns:
    #         DirectoryNode representing the artifact's tree structure

    #     Raises:
    #         FileNotFoundError: If the path doesn't exist
    #     """
    #     artifact_tree = self._artifact_tree_builder.process_artifact(local_uri)
    #     self._artifact_merger.add_tree(artifact_tree)
    #     self._artifacts = self._artifact_merger.get_merged_trees()


import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from dreadnode.api.models import UserDataCredentials
from dreadnode.constants import DATASETS_CACHE, METADATA_FILE
from dreadnode.storage.base import BaseStorage
from dreadnode.storage.datasets.metadata import DatasetMetadata


class DatasetStorage(BaseStorage):
    """
    High-level client for dataset operations.

    This is the main interface users interact with.
    """

    def __init__(
        self,
        credential_fetcher: Callable[[], UserDataCredentials] | None = None,
        cache_dir: Path | None = None,
    ):
        """
        Initialize dataset client.

        Args:
            credential_fetcher: Function to get S3 credentials
            cache_dir: Custom cache directory
        """
        self._credential_fetcher = credential_fetcher
        self.cache_dir = cache_dir or DATASETS_CACHE

    def list_cached_datasets(self) -> list[DatasetMetadata]:
        """
        List all datasets in cache.

        Returns:
            List of metadata for cached datasets
        """
        datasets = []

        for org_dir in self.cache_dir.iterdir():
            if not org_dir.is_dir():
                continue

            for dataset_dir in org_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue

                for version_dir in dataset_dir.iterdir():
                    if not version_dir.is_dir():
                        continue

                    metadata_path = version_dir / METADATA_FILE
                    if metadata_path.exists():
                        try:
                            metadata = DatasetMetadata.load(metadata_path)
                            datasets.append(metadata)
                        except Exception as e:
                            logger.warning(f"Failed to load metadata from {metadata_path}: {e}")

        return datasets

    def delete_dataset(
        self,
        uri: str,
        version: str | None = None,
        *,
        cache_only: bool = True,
    ) -> bool:
        """
        Remove dataset from cache and optionally remote.

        Args:
            uri: Dataset URI
            version: Specific version (removes all if None)
            cache_only: If True, only remove from cache

        Returns:
            True if removed successfully
        """
        parsed_uri, parsed_version = self.parse_uri(uri)
        version = version or parsed_version

        logger.info(f"Removing dataset {parsed_uri}@{version or 'all'}")

        # Remove from cache
        if version:
            cache_path = self.get_dataset_path(uri, version)
            if cache_path.exists():
                logger.info(f"Removing dataset from cache: {cache_path}")
                shutil.rmtree(cache_path)
                return True
            return False

        if not cache_only and self._credential_fetcher:
            # Remove from remote (implementation depends on backend)
            logger.warning("Remote deletion not implemented")

        return True

    def list_datasets(
        self,
        *,
        remote: bool = False,
        cache_only: bool = True,
    ) -> list[any]:
        """
        List available datasets.

        Args:
            remote: If True, list from remote storage
            cache_only: If True, only list cached datasets

        Returns:
            List of dataset metadata
        """
        if cache_only or not remote:
            return self.list_cached_datasets()

        # Remote listing would require API implementation
        logger.warning("Remote dataset listing not implemented")
        return []

    def search_datasets(
        self,
        name_pattern: str | None = None,
        tags: list[str] | None = None,
        version: str | None = None,
        *,
        remote: bool = False,
    ) -> list[any]:
        """
        Search for datasets matching criteria.

        Args:
            name_pattern: Pattern to match in name
            tags: Required tags
            version: Specific version
            remote: Search remote storage

        Returns:
            List of matching dataset metadata
        """
        if remote and not self._credential_fetcher:
            logger.warning("Remote search requires credential fetcher")
            remote = False

        all_datasets = self.list_cached_datasets()

        results = [ds for ds in all_datasets if ds.matches_filter(name_pattern, tags, version)]

        if remote:
            # Remote search would require API implementation
            logger.warning("Remote dataset search not implemented")

        return results

    def get_cache_info(self) -> dict[str, Any]:
        """
        Get information about the cache.

        Returns:
            Dictionary with cache statistics

        Examples:
            >>> client = DatasetStorage()
            >>> info = client.get_cache_info()
            >>> print(f"Cache size: {info['size_gb']:.2f} GB")
        """
        size_bytes = self.get_cache_size()
        datasets = self.list_cached_datasets()

        return {
            "cache_dir": str(self.cache_dir),
            "size_bytes": size_bytes,
            "size_mb": size_bytes / (1024 * 1024),
            "size_gb": size_bytes / (1024 * 1024 * 1024),
            "dataset_count": len(datasets),
            "datasets": [
                {
                    "name": ds.name,
                    "version": ds.version,
                    "uri": ds.uri,
                }
                for ds in datasets
            ],
        }

    def get_cache_size(self) -> int:
        """
        Get total size of cache in bytes.

        Returns:
            Cache size in bytes
        """
        total_size = 0
        for path in self.cache_dir.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size
        return total_size

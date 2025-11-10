"""
Dataset storage implementation for fsspec-compatible file systems.
Provides efficient uploading of files and directories
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from dreadnode.dataset import Dataset
from dreadnode.storage.base import BaseStorage

if TYPE_CHECKING:
    from dreadnode.api.models import UserDataCredentials


CHUNK_SIZE = 8 * 1024 * 1024  # 8MB


class DatasetStorage(BaseStorage):
    """
    Storage for datasets with efficient handling of large files and directories.
    """

    def __init__(self, credential_fetcher: Callable[[], "UserDataCredentials"]):
        """
        Initialize artifact storage with credential manager.

        Args:
            credential_fetcher: function that returns new UserDataCredentials for S3.
        """
        super().__init__(credential_fetcher=credential_fetcher)
        self._prefix: str = "datasets"

    def save(self, dataset: Dataset, path: Path, *, to_hub: bool = False) -> Dataset | None:
        """
        Saves a dataset to the local cache.

        Args:
            dataset: The Dataset instance to save.
            path: The full URL to the dataset's directory.

        """

        fs, dataset_path = self.get_filesystem(path)

        # check local cache
        check_cache_path = self._local_cache_path / dataset_path.stem
        if not fs.exists(check_cache_path):
            dataset.ds.write_to_dataset(str(check_cache_path), format="parquet", filesystem=fs)

        # check remote paths

        return None

    def load(self, path: str, lazy: bool = False) -> Dataset | None:
        """
        Loads a dataset from the given directory URL.

        Args:
            path: The full URL to the dataset's directory.
            lazy: If True, loads a `pyarrow.dataset.Dataset` pointer without
                  reading data into memory. If False (default), loads the data
                  into an in-memory `pyarrow.Table`.

        Returns:
            A new Dataset instance.
        """

        fs, dataset_path = self.get_filesystem(path)

        # check local cache first
        check_cache_path = self._local_cache_path / dataset_path.stem
        if fs.exists(check_cache_path):
            print(f"Loading dataset from cache at {check_cache_path}")
            # if there is a metadata file, load from that
            with fs.open(check_cache_path.joinpath(".metadata.json"), "r") as f:
                metadata_json = json.load(f)
                return Dataset.from_json(metadata_json)

        # check remote paths
        if fs.exists(dataset_path):
            print(f"Loading dataset from path at {dataset_path}")
            return Dataset.from_path(path)

        print(f"Dataset not found at path: {path}")
        return None

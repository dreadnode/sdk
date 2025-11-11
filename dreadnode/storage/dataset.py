"""
Dataset storage implementation for fsspec-compatible file systems.
Provides efficient uploading of files and directories
"""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from fsspec.core import strip_protocol
from fsspec.utils import get_protocol

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

    def save(self, dataset: Dataset, *, cache: bool) -> None:
        """
        Saves a dataset to the local cache.

        Args:
            dataset: The Dataset instance to save.
            cache: If True, saves to local cache. If False, saves to remote storage.

        """

        if cache:
            # save to the local cache
            cache_path = self._local_cache_path
            fs, fs_path = self.get_filesystem(cache_path)
            dataset.save(cache_path, _fs=fs)

        else:
            # save to remote storage
            remote_path = f"dn://{dataset.name}"
            fs, fs_path = self.get_filesystem(remote_path)
            dataset.save(fs_path, _fs=fs)

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
        protocol = get_protocol(path)

        if protocol in ("dn", "dreadnode"):
            # check the cache first
            cache_path = Path(self._local_cache_path).joinpath(strip_protocol(path))
            if not cache_path.exists():
                # if not in cache, load from remote and cache locally
                print("Loading dataset from remote storage...")
                fs, fs_path = self.get_filesystem(path)
            else:
                # load from cache
                print("Loading dataset from local cache...")
                fs, fs_path = self.get_filesystem(cache_path)

            loaded_ds = Dataset.from_path(fs_path, _fs=fs)

        else:
            # load directly from the path
            if not Path(path).exists():
                raise FileNotFoundError(f"Dataset path does not exist: {path}")

            fs, fs_path = self.get_filesystem(path)
            print("Loading dataset from local path...")
            loaded_ds = Dataset.from_path(fs_path, _fs=fs)

        return loaded_ds

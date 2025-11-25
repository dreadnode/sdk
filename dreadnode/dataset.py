from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem

from dreadnode.constants import METADATA_FILE
from dreadnode.storage.datasets.core import (
    FilesystemManager,
)
from dreadnode.storage.datasets.manifest import Manifest
from dreadnode.storage.datasets.metadata import DatasetMetadata, VersionInfo


class Dataset:
    """
    A versioned dataset with complete metadata tracking.
    """

    def __init__(
        self,
        ds: ds.Dataset | pa.Table,
        _metadata: DatasetMetadata | None = None,
    ) -> None:
        self.ds = ds
        self._metadata = _metadata
        self._manifest: Manifest | None = None

    @property
    def metadata(self) -> DatasetMetadata:
        if not self._metadata:
            self._metadata = DatasetMetadata()
        return self._metadata

    @staticmethod
    def _normalize_uri(name: str) -> str:
        return name.replace(" ", "_").lower()

    def create_manifest(self) -> None:
        self._manifest = Manifest.create_manifest(
            (self.metadata.uri, self.metadata.version),
            version=self.metadata.version,
            exclude_patterns=[METADATA_FILE],
        )

    def update_metadata(self, **kwargs) -> None:
        """
        Update the dataset metadata before saving.
        """
        self.metadata.set_organization(kwargs.get("organization", self.metadata.organization))
        self.metadata.set_version(kwargs.get("version", self.metadata.version))
        self.metadata.set_ds_schema(self.ds)
        self.metadata.set_ds_files(self.ds)
        self.metadata.set_size_and_row_count(self.ds)
        self.metadata.set_updated_at()


def _create_manifest(path: str, dataset: Dataset) -> Manifest:
    manifest = Manifest.create_manifest(
        path=path,
        version=dataset.metadata.version,
        exclude_patterns=[METADATA_FILE],
    )
    return manifest


def _create_default_metadata() -> DatasetMetadata:
    metadata = DatasetMetadata()
    metadata.set_version(VersionInfo(major=0, minor=0, patch=1))
    metadata.set_created_at()
    metadata.set_updated_at()

    return metadata


def _create_dataset(
    uri: Path,
    format: str,
    *,
    filesystem: AbstractFileSystem,
    materialize: bool = False,
) -> Dataset:
    """
    Create a Dataset from a local path.
    """

    metadata_path = uri.joinpath(METADATA_FILE)
    if filesystem.exists(metadata_path):
        print(f"[*] Loading dataset metadata from: {metadata_path}")

        metadata = DatasetMetadata.load(metadata_path, filesystem=filesystem)
        uri = uri.joinpath("data")
    else:
        metadata = _create_default_metadata()

    dataset = ds.dataset(str(uri), format=format, filesystem=filesystem)
    if materialize:
        dataset = dataset.to_table()

    return Dataset(ds=dataset, _metadata=metadata)


def save_dataset(
    dataset: Dataset,
    *,
    version: str | None = None,
    repo: str = "datasets",
    to_cache: bool = False,
    fsm: FilesystemManager,
    **kwargs: Any,
) -> None:
    """
    Save dataset to remote storage, or local cache.
    """

    if version is not None:
        VersionInfo.model_validate(version)

    if to_cache:
        path = fsm._cache_root / repo / "main" / dataset.metadata.name
        print("[*] Saving dataset to local cache")

    else:
        print("[*] Saving dataset to remote storage")
        path = "dn://" + f"main/{dataset.metadata.name}"

    fs, fs_path = fsm.get_filesystem(str(path))

    fs.mkdir(fs_path, exist_ok=True)

    save_path = fsm.get_dataset_dir(
        path=fs_path,
        version=version,
        repo=repo,
        to_cache=to_cache,
        filesystem=fs,
    )

    print("[+] Saved dataset")

    with fs.open(save_path.joinpath("metadata.json"), "w") as f:
        f.write(dataset.metadata.model_dump_json(indent=2))

    ds.write_dataset(
        data=dataset.ds,
        base_dir=str(save_path.joinpath("data")),
        format="parquet",
        filesystem=fs,
        existing_data_behavior="overwrite_or_ignore",
        **kwargs,
    )

    with fs.open(save_path.joinpath("manifest.json"), "wb") as f:
        f.write(b"")


def load_dataset(
    uri: str,
    version: str | None = None,
    repo: str = "datasets",
    format: str = "parquet",
    *,
    materialize: bool = True,
    fsm: FilesystemManager,
) -> Dataset:
    """
    Load dataset from local path, cache, or remote storage.

    Resolution order:
    1. Explicit remote URI (dn://org/name).
    2. Cached dataset (org/name)
    3. Local filesystem path (../data/my_data)
    4. Remote fallback (org/name)
    """

    fs, fs_path = fsm.get_filesystem(uri)

    if version is not None:
        VersionInfo.model_validate(version)

    # load from explicit remote (dn://org/name)
    if not isinstance(fs, LocalFileSystem):
        remote_uri = fsm.get_dataset_dir(
            path=fs_path,
            version=version,
            repo=repo,
            to_cache=False,
            filesystem=fs,
        )
        print(f"[+] Loading dataset from remote: {remote_uri}")
        return _create_dataset(
            uri=remote_uri,
            format=format,
            filesystem=fs,
            materialize=materialize,
        )

    # then check the cache (org/name)
    try:
        cache_path = fsm.get_dataset_dir(
            path=uri,
            version=version,
            repo=repo,
            to_cache=True,
            filesystem=fs,
        )
        if cache_path.exists():
            print(f"[+] Loading dataset from cache: {cache_path}")
            return _create_dataset(
                uri=cache_path,
                format=format,
                filesystem=fs,
                materialize=materialize,
            )
    except (FileNotFoundError, ValueError):
        local_path = Path(uri).expanduser().resolve()
        if fs.exists(local_path):
            print(f"[+] Loading dataset from local: {local_path}")
            return _create_dataset(
                uri=local_path, format=format, filesystem=fs, materialize=materialize
            )

    # remote fallback if neither path not spec'd with protocol(org/name -> remote)
    try:
        print(f"[*] Dataset not found locally. Attempting remote load for: {uri}")
        remote_uri = fsm.get_dataset_dir(
            path=fs_path, version=version, repo=repo, to_cache=False, filesystem=fs
        )
        return _create_dataset(
            uri=remote_uri,
            format=format,
            filesystem=fs,
            materialize=materialize,
        )
    except Exception as e:
        print(f"[!] Failed to load dataset from remote: {e}")

    raise FileNotFoundError(f"[!] Dataset not found locally or remotely: {uri}")

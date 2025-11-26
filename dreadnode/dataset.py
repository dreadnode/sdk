# from pathlib import Path
# from typing import Any

# import pyarrow as pa
# import pyarrow.dataset as ds
# from fsspec import AbstractFileSystem
# from fsspec.implementations.local import LocalFileSystem

# from dreadnode.constants import METADATA_FILE
# from dreadnode.storage.datasets.core import (
#     FilesystemManager,
# )
# from dreadnode.storage.datasets.manifest import Manifest
# from dreadnode.storage.datasets.metadata import DatasetMetadata, VersionInfo


# class Dataset:
#     """
#     A versioned dataset with complete metadata tracking.
#     """

#     def __init__(
#         self,
#         ds: ds.Dataset | pa.Table,
#         _metadata: DatasetMetadata | None = None,
#     ) -> None:
#         self.ds = ds
#         self._metadata = _metadata
#         self._manifest: Manifest | None = None

#     @property
#     def metadata(self) -> DatasetMetadata:
#         if not self._metadata:
#             self._metadata = DatasetMetadata()
#         return self._metadata

#     @staticmethod
#     def _normalize_uri(name: str) -> str:
#         return name.replace(" ", "_").lower()

#     def create_manifest(self) -> None:
#         self._manifest = Manifest.create_manifest(
#             (self.metadata.uri, self.metadata.version),
#             version=self.metadata.version,
#             exclude_patterns=[METADATA_FILE],
#         )

#     def update_metadata(self, **kwargs) -> None:
#         """
#         Update the dataset metadata before saving.
#         """
#         self.metadata.set_organization(kwargs.get("organization", self.metadata.organization))
#         self.metadata.set_version(kwargs.get("version", self.metadata.version))
#         self.metadata.set_ds_schema(self.ds)
#         self.metadata.set_ds_files(self.ds)
#         self.metadata.set_size_and_row_count(self.ds)
#         self.metadata.set_updated_at()


# def _create_dataset(
#     uri: Path,
#     format: str,
#     *,
#     filesystem: AbstractFileSystem,
#     materialize: bool = False,
# ) -> Dataset:
#     """
#     Create a Dataset from a local path.
#     """

#     metadata_path = uri.joinpath(METADATA_FILE)
#     if filesystem.exists(metadata_path):
#         print(f"[*] Loading dataset metadata from: {metadata_path}")

#         metadata = DatasetMetadata.load(metadata_path, filesystem=filesystem)
#         uri = uri.joinpath("data")
#     else:
#         metadata = _create_default_metadata()

#     dataset = ds.dataset(str(uri), format=format, filesystem=filesystem)
#     if materialize:
#         dataset = dataset.to_table()

#     return Dataset(ds=dataset, _metadata=metadata)


# def save_dataset(
#     dataset: Dataset,
#     *,
#     repo: str = "datasets",
#     to_cache: bool = False,
#     version: str | None = None,
#     fsm: FilesystemManager,
#     **kwargs: Any,
# ) -> None:
#     """
#     Save dataset to remote storage, or local cache.
#     """

#     if to_cache:
#         path = fsm._cache_root / repo / "main" / dataset.metadata.name
#         print("[*] Saving dataset to local cache")

#     else:
#         print("[*] Saving dataset to remote storage")
#         path = "dn://" + f"main/{dataset.metadata.name}"

#     fs, fs_path = fsm.get_filesystem(str(path))

#     # fs.mkdir(fs_path, exist_ok=True)

#     save_path = fsm.get_dataset_dir(
#         path=fs_path,
#         version=None,
#         repo=repo,
#         to_cache=to_cache,
#         filesystem=fs,
#     )

#     print("[+] Saved dataset")

#     with fs.open(save_path.joinpath("metadata.json"), "w") as f:
#         f.write(dataset.metadata.model_dump_json(indent=2))

#     ds.write_dataset(
#         data=dataset.ds,
#         base_dir=str(save_path.joinpath("data")),
#         format="parquet",
#         filesystem=fs,
#         existing_data_behavior="overwrite_or_ignore",
#         **kwargs,
#     )

#     manifest = _create_manifest(str(save_path), dataset, fs)

#     with fs.open(save_path.joinpath("manifest.json"), "w") as f:
#         f.write(manifest.model_dump_json(indent=2))


# def load_dataset(
#     uri: str,
#     version: str | None = None,
#     repo: str = "datasets",
#     format: str = "parquet",
#     *,
#     materialize: bool = True,
#     fsm: FilesystemManager,
# ) -> Dataset:
#     """
#     Load dataset from local path, cache, or remote storage.

#     Resolution order:
#     1. Explicit remote URI (dn://org/name).
#     2. Cached dataset (org/name)
#     3. Local filesystem path (../data/my_data)
#     4. Remote fallback (org/name)
#     """

#     fs, fs_path = fsm.get_filesystem(uri)

#     if version is not None:
#         VersionInfo.model_validate(version)

#     # load from explicit remote (dn://org/name)
#     if not isinstance(fs, LocalFileSystem):
#         remote_uri = fsm.get_dataset_dir(
#             path=fs_path,
#             version=version,
#             repo=repo,
#             to_cache=False,
#             filesystem=fs,
#         )
#         print(f"[+] Loading dataset from remote: {remote_uri}")
#         return _create_dataset(
#             uri=remote_uri,
#             format=format,
#             filesystem=fs,
#             materialize=materialize,
#         )

#     # then check the cache (org/name)
#     try:
#         cache_path = fsm.get_dataset_dir(
#             path=uri,
#             version=version,
#             repo=repo,
#             to_cache=True,
#             filesystem=fs,
#         )
#         if cache_path.exists():
#             print(f"[+] Loading dataset from cache: {cache_path}")
#             return _create_dataset(
#                 uri=cache_path,
#                 format=format,
#                 filesystem=fs,
#                 materialize=materialize,
#             )
#     except (FileNotFoundError, ValueError):
#         local_path = Path(uri).expanduser().resolve()
#         if fs.exists(local_path):
#             print(f"[+] Loading dataset from local: {local_path}")
#             return _create_dataset(
#                 uri=local_path, format=format, filesystem=fs, materialize=materialize
#             )

#     # remote fallback if neither path not spec'd with protocol(org/name -> remote)
#     try:
#         print(f"[*] Dataset not found locally. Attempting remote load for: {uri}")
#         remote_uri = fsm.get_dataset_dir(
#             path=fs_path, version=version, repo=repo, to_cache=False, filesystem=fs
#         )
#         return _create_dataset(
#             uri=remote_uri,
#             format=format,
#             filesystem=fs,
#             materialize=materialize,
#         )
#     except Exception as e:
#         print(f"[!] Failed to load dataset from remote: {e}")

#     raise FileNotFoundError(f"[!] Dataset not found locally or remotely: {uri}")


import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
from pyarrow.fs import FileSystem, FileType

# We don't need fsspec or s3fs anymore for the core logic
# from fsspec import AbstractFileSystem
from dreadnode.constants import METADATA_FILE

# (Assuming FilesystemManager is still used for logic, but not for creating the FS object)
from dreadnode.storage.datasets.core import FilesystemManager
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


def _resolve_uri(uri: str) -> tuple[FileSystem, str]:
    """
    Robustly resolves a URI into a Native PyArrow FileSystem and a clean path.

    Input: "s3://my-bucket/data" -> (S3FileSystem, "my-bucket/data")
    Input: "/tmp/data"           -> (LocalFileSystem, "/tmp/data")
    """
    try:
        # This is the "Magic" native method
        fs, path = FileSystem.from_uri(uri)
    except ValueError:
        # Fallback: if it's a simple local path without file://
        fs = FileSystem.from_uri("file:///")[0]
        path = str(Path(uri).expanduser().resolve())

    return fs, path


def _create_manifest(path: str, dataset: Dataset, fs: FileSystem) -> Manifest:
    manifest = Manifest.create_manifest(
        path=path,
        version="0.0.1",
        exclude_patterns=[METADATA_FILE],
        fs=fs,
    )
    return manifest


def _create_default_metadata() -> DatasetMetadata:
    metadata = DatasetMetadata()
    metadata.set_version(VersionInfo(major=0, minor=0, patch=1))
    metadata.set_created_at()
    metadata.set_updated_at()

    return metadata


def _ensure_dir(fs: FileSystem, path: str):
    """
    Native PyArrow equivalent of mkdir -p
    """
    fs.create_dir(path, recursive=True)


def _read_json_native(fs: FileSystem, path: str) -> dict:
    """
    Helper to read JSON from native binary streams.
    """
    with fs.open_input_stream(path) as f:
        return json.load(f)


def _write_json_native(fs: FileSystem, path: str, data_model: Any):
    """
    Helper to write Pydantic models to native binary streams.
    """
    json_bytes = data_model.model_dump_json(indent=2).encode("utf-8")
    with fs.open_output_stream(path) as f:
        f.write(json_bytes)


def _create_dataset(
    uri: str,
    format: str,
    *,
    filesystem: FileSystem,
    materialize: bool = False,
) -> Dataset:
    metadata_path = f"{uri}/{METADATA_FILE}"

    file_info = filesystem.get_file_info(metadata_path)

    if file_info.type != FileType.NotFound:
        print(f"[*] Loading dataset metadata from: {metadata_path}")

        data = _read_json_native(filesystem, metadata_path)
        metadata = DatasetMetadata(**data)

        data_uri = f"{uri}/data"
    else:
        metadata = _create_default_metadata()
        data_uri = uri

    # Load the dataset
    dataset = ds.dataset(data_uri, format=format, filesystem=filesystem)

    if materialize:
        dataset = dataset.to_table()

    return Dataset(ds=dataset, _metadata=metadata)


def save_dataset(
    dataset: Dataset,
    *,
    repo: str = "datasets",
    to_cache: bool = False,
    fsm: FilesystemManager,
    **kwargs: Any,
) -> None:
    fsm._needs_refresh()
    # 1. Determine the destination URI string
    if to_cache:
        # Keep path generation, but convert to string immediately
        path_str = str(fsm._cache_root / repo / "main" / dataset.metadata.name)
        print("[*] Saving dataset to local cache")
    else:
        print("[*] Saving dataset to remote storage")
        path_str = f"s3://{fsm.organization}/{dataset.metadata.name}"

    # 2. Get Native Filesystem
    fs, base_path = _resolve_uri(path_str)
    print(f"[*] Saving dataset to: {base_path}")

    # 3. Create generic directory (works on S3 and Local)
    _ensure_dir(fs, base_path)

    # 4. Write Metadata
    print("[+] Saving metadata...")
    _write_json_native(fs, f"{base_path}/metadata.json", dataset.metadata)

    # 5. Write Data (Parquet)
    ds.write_dataset(
        data=dataset.ds,
        base_dir=f"{base_path}/data",
        format="parquet",
        filesystem=fs,
        existing_data_behavior="overwrite_or_ignore",
        **kwargs,
    )

    # 6. Create and Write Manifest
    manifest = Manifest.create_manifest(
        path=base_path,
        version=dataset.metadata.version.to_string(),
        exclude_patterns=[METADATA_FILE],
        fs=fs,
    )

    _write_json_native(fs, f"{base_path}/manifest.json", manifest)

    print("[+] Saved dataset successfully")


def load_dataset(
    uri: str,
    version: str | None = None,
    repo: str = "datasets",
    format: str = "parquet",
    *,
    materialize: bool = True,
    fsm: FilesystemManager,
) -> Dataset:
    if version is not None:
        VersionInfo.model_validate(version)

    fsm._needs_refresh()

    # 1. Resolve Input URI
    fs, fs_path = _resolve_uri(uri)

    # 2. Logic to determine paths (Remote vs Cache vs Local)
    info = fs.get_file_info(fs_path)

    if info.type != FileType.NotFound:
        print(f"[+] Loading dataset from: {uri}")
        return _create_dataset(
            uri=fs_path,
            format=format,
            filesystem=fs,
            materialize=materialize,
        )

    # 3. Fallback logic (Cache)
    cache_path_str = str(fsm._cache_root / repo / "main" / Path(uri).name)
    local_fs, local_path = _resolve_uri(cache_path_str)

    if local_fs.get_file_info(local_path).type != FileType.NotFound:
        print(f"[+] Loading dataset from cache: {local_path}")
        return _create_dataset(
            uri=local_path, format=format, filesystem=local_fs, materialize=materialize
        )

    # 4. Remote Fallback (dn:// -> s3://)
    remote_uri = f"s3://{repo}/main/{uri}"
    remote_fs, remote_path = _resolve_uri(remote_uri)

    print(f"[*] Attempting remote load for: {remote_uri}")
    try:
        # Verify it exists before trying to load
        if remote_fs.get_file_info(remote_path).type == FileType.NotFound:
            raise FileNotFoundError(f"Remote path {remote_uri} does not exist")

        return _create_dataset(
            uri=remote_path,
            format=format,
            filesystem=remote_fs,
            materialize=materialize,
        )
    except Exception as e:
        print(f"[!] Failed to load dataset from remote: {e}")

    raise FileNotFoundError(f"[!] Dataset not found: {uri}")


# Helper for Manifest if you need to walk files
# Native PyArrow implementation of a file walker
def _native_walk(fs: FileSystem, path: str):
    selector = pa.fs.FileSelector(path, recursive=True)
    return fs.get_file_info(selector)

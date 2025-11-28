import contextlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
from fsspec.core import strip_protocol
from fsspec.utils import get_protocol
from pyarrow.fs import FileSystem, FileType

from dreadnode.constants import METADATA_FILE
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
        self._metadata: DatasetMetadata = _metadata
        self._manifest: Manifest | None = None

    @property
    def metadata(self) -> DatasetMetadata:
        if not self._metadata:
            self._metadata = DatasetMetadata()
        return self._metadata


def _resolve_uri(uri: str) -> tuple[FileSystem, str]:
    """
    Resolves a URI into a Native PyArrow FileSystem and a clean path.

    Input: "s3://my-bucket/data" -> (S3FileSystem, "my-bucket/data")
    Input: "/tmp/data"           -> (LocalFileSystem, "/tmp/data")
    """
    try:
        fs, path = FileSystem.from_uri(uri)
    except ValueError:
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
    Creates directory if local. Skips if S3.

    Why skip S3?
    1. S3 has no real directories (only keys).
    2. pyarrow.fs.S3FileSystem.create_dir() executes a 'HeadBucket' call.
    3. Restricted IAM policies (write-only) will fail HeadBucket with ACCESS_DENIED.
    """
    if fs.type_name == "s3":
        return
    with contextlib.suppress(OSError):
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

    if to_cache:
        path_str = str(fsm._cache_root / repo / fsm.organization / dataset.metadata.name)
        print("[*] Saving dataset to local cache")
    else:
        print("[*] Saving dataset to remote storage")
        path_str = fsm.get_remote_uri(repo=repo, name=dataset.metadata.name)

    fs, base_path = fsm.get_fs_and_path(path_str)
    print(f"[*] Saving dataset to: {base_path}")

    _ensure_dir(fs, base_path)

    # 5. Write Data (Parquet)
    ds.write_dataset(
        data=dataset.ds,
        base_dir=f"{base_path}/data",
        format="parquet",
        filesystem=fs,
        existing_data_behavior="overwrite_or_ignore",
        create_dir=to_cache,
        **kwargs,
    )

    # 4. Write Metadata
    print("[+] Saving metadata...")
    _write_json_native(fs, f"{base_path}/metadata.json", dataset.metadata)

    # 6. Create and Write Manifest
    manifest = Manifest.create_manifest(
        path=base_path,
        version=dataset.metadata.version.to_string(),
        exclude_patterns=[METADATA_FILE],
        fs=fs,
    )

    print("[+] Saving manifest...")
    _write_json_native(fs, f"{base_path}/manifest.json", manifest)

    print("[+] Saved dataset successfully")


def load_dataset(
    uri: str,
    version: str = "latest",
    repo: str = "datasets",
    format: str = "parquet",
    *,
    materialize: bool = True,
    fsm: FilesystemManager,
) -> Dataset:
    if version is not None:
        VersionInfo.model_validate(version)

    protocol = get_protocol(uri)
    print(f"[*] Resolving dataset URI with protocol: {protocol}")

    if protocol in ("file", "local", ""):
        # check the cache first
        cache_path = Path(fsm._cache_root).joinpath(repo, uri)
        if not cache_path.exists() and Path(uri).exists():
            print("[+] Loading dataset from local path...")
            fs, fs_path = fsm.get_fs_and_path(uri)
            return _create_dataset(
                uri=fs_path,
                format=format,
                filesystem=fs,
                materialize=materialize,
            )

        print("[*] Loading from cache...")
        fs, fs_path = fsm.get_fs_and_path(str(cache_path))
        return _create_dataset(
            uri=fs_path,
            format=format,
            filesystem=fs,
            materialize=materialize,
        )

    print("[*] Loading from remote storage...")
    try:
        fsm._needs_refresh()

        remote_uri = fsm.get_remote_uri(repo=repo, name=strip_protocol(uri))

        print(f"[*] Resolved remote URI: {remote_uri}")

        fs, fs_path = fsm.get_fs_and_path(remote_uri)

        print(f"[*] Checking existence of remote path: {fs_path}")
        # Verify it exists before trying to load
        if fs.get_file_info(fs_path).type == FileType.NotFound:
            raise FileNotFoundError(f"Remote path {fs_path} does not exist")

        return _create_dataset(
            uri=fs_path,
            format=format,
            filesystem=fs,
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

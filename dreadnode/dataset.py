from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
from fsspec.core import strip_protocol
from fsspec.utils import get_protocol
from pyarrow.fs import FileSystem

from dreadnode.constants import MANIFEST_FILE, METADATA_FILE
from dreadnode.storage.datasets.manager import DatasetManager
from dreadnode.storage.datasets.manifest import DatasetManifest, create_manifest
from dreadnode.storage.datasets.metadata import DatasetMetadata, VersionInfo


class Dataset:
    """
    A versioned dataset.

    Attributes:
        ds: The PyArrow Dataset or Table.
        metadata: The DatasetMetadata associated with this dataset.
        manifest: The DatasetManifest associated with this dataset.
    """

    def __init__(
        self,
        ds: ds.Dataset,
        metadata: DatasetMetadata,
        manifest: DatasetManifest | None = None,
        *,
        materialize: bool = True,
    ) -> None:
        self.ds = ds
        self.metadata = metadata
        self.manifest = manifest

        if materialize:
            self.ds = ds.to_table()

        if not metadata:
            print("[*] No metadata provided, check your dataset!")

    def update_metadata(self, metadata: DatasetMetadata) -> None:
        self.metadata = metadata

    def update_manifest(self, manifest: DatasetManifest) -> None:
        self.manifest = manifest

    def set_ds_schema(self) -> None:
        """
        Set the dataset schema in metadata.
        Handles both on-disk Dataset and in-memory Table.
        """
        schema = self.ds.schema

        if schema.metadata:
            pass

        ds_schema = {}
        for field in schema:
            ds_schema[field.name] = str(field.type)
        self.ds_schema = ds_schema

    def set_ds_files(self) -> None:
        """
        Set the list of files. Relativizes paths.
        """
        if isinstance(self.ds, pa.Table):
            self.files = []
            return

        if hasattr(self.ds, "files"):
            clean_files = []
            for f in self.ds.files:
                f_str = str(f).replace("\\", "/")
                filename = f_str.split("/")[-1]
                clean_files.append(f"data/{filename}")

            self.files = clean_files

    def set_size_and_row_count(self) -> None:
        """
        Safely extracts size and row count.
        """
        if isinstance(self.ds, pa.Table):
            self.size_bytes = self.ds.nbytes
            self.row_count = self.ds.num_rows
            return

        self.size_bytes = 0

        try:
            self.row_count = self.ds.count_rows()
        except Exception:
            self.row_count = 0

    def save_metadata(self, path: str, fs: FileSystem) -> None:
        """
        Helper to write Pydantic models to native binary streams.
        """
        if self.metadata is None:
            raise ValueError("Dataset metadata is not set")

        # update metadata fields before saving
        self.metadata.save(path, fs)

    def save_dataset(
        self,
        path: str,
        fs: FileSystem,
        *,
        to_cache: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Helper to write the dataset to storage.
        """

        ds.write_dataset(
            data=self.ds,
            base_dir=path,
            format="parquet",
            filesystem=fs,
            existing_data_behavior="overwrite_or_ignore",
            create_dir=to_cache,
            **kwargs,
        )


def save_dataset(
    dataset: Dataset,
    *,
    to_cache: bool = False,
    fsm: DatasetManager,
    **kwargs: Any,
) -> None:
    if to_cache:
        path_str = fsm.get_cache_save_uri(metadata=dataset.metadata)
        print("[*] Saving dataset to local cache")
    else:
        path_str = fsm.get_remote_save_uri(metadata=dataset.metadata)
        print("[*] Saving dataset to remote storage")

    fs, base_path = fsm.get_fs_and_path(path_str)

    try:
        fsm.ensure_dir(fs, base_path)

        dataset.save_dataset(path=f"{base_path}/data", fs=fs, to_cache=to_cache, **kwargs)

        dataset.save_metadata(path=f"{base_path}/{METADATA_FILE}", fs=fs)

        manifest = create_manifest(
            path=base_path,
            version=dataset.metadata.version,
            previous_manifest=dataset.manifest if dataset.manifest else None,
            fs=fs,
        )
        manifest.save(f"{base_path}/{MANIFEST_FILE}", fs=fs)
    except Exception as e:
        # if remote save failed, notify API
        if not to_cache:
            fsm.remote_save_complete(success=False, dataset_id=dataset.metadata.id)
        print(f"[!] Failed to save dataset: {e}")
        raise

    # if remote save succeeded, notify API
    if not to_cache:
        fsm.remote_save_complete(success=True, dataset_id=dataset.metadata.id)

    print("[+] Saved dataset successfully")


def load_dataset(
    uri: str,
    version: str | None = None,
    format: str = "parquet",
    *,
    materialize: bool = True,
    fsm: DatasetManager,
    **kwargs: dict,
) -> Dataset:
    """
    Loads a dataset from the given URI.
    Checks local cache first, then remote storage.

    Args:
        uri: The URI of the dataset.
        version: The version to load. Defaults to "latest".
        repo: The repository name. Defaults to "datasets".
        format: The dataset format. Defaults to "parquet".
        materialize: Whether to materialize the dataset into memory. Defaults to True.
        fsm: The DatasetManager instance.
        kwargs: Additional arguments to pass to pyarrow.dataset.load_dataset.
    """
    if version is not None:
        VersionInfo.from_string(version)

    protocol = get_protocol(uri)

    # if local path
    if protocol in ("file", "local", ""):
        # check cache first
        if not fsm.check_cache(uri, version):
            print("[+] Dataset not found in cache. Loading dataset from local path...")

            # load directly from local path
            fs, fs_path = fsm.get_fs_and_path(uri)

            if not Path(fs_path).exists():
                raise FileNotFoundError(f"[!] Dataset not found at local path: {uri}")

            # load dataset
            dataset = ds.dataset(fs_path, format=format, filesystem=fs, **kwargs)

            # metadata and manifest files are not expected in local path
            metadata = DatasetMetadata(organization=fsm.organization, format=format)

            return Dataset(ds=dataset, metadata=metadata, materialize=materialize)

        # if in cache, load from cache
        print("[+] Loading dataset from cache...")

        # get the filesystem and path
        fs, fs_path = fsm.get_fs_and_path(uri)

        # resolve versioned cache path
        cache_path = fsm.get_cache_load_uri(fs_path, version, fs)

        # metadata and manifest files are expected in cache
        if fsm.metadata_exists(path=cache_path):
            metadata = DatasetMetadata.load(path=f"{cache_path}/{METADATA_FILE}", fs=fs)

        # the manifest will be compared on save
        if fsm.manifest_exists(path=cache_path):
            manifest = DatasetManifest.load(path=f"{cache_path}/{MANIFEST_FILE}", fs=fs)

        # load dataset
        dataset = ds.dataset(f"{cache_path}/data", format=format, filesystem=fs, **kwargs)

        return Dataset(ds=dataset, materialize=materialize, metadata=metadata, manifest=manifest)

    # if not local path, and not in cache, load from remote
    print("[+] Loading from remote storage...")
    try:
        # get remote URI
        remote_uri = fsm.get_remote_load_uri(uri=strip_protocol(uri), version=version)

        # get the filesystem and path
        fs, fs_path = fsm.get_fs_and_path(remote_uri)

        # metadata and manifest files are expected in remote storage
        info = fs.get_file_info(f"{fs_path}/{METADATA_FILE}")
        if info.is_file:
            metadata = DatasetMetadata.load(path=f"{fs_path}/{METADATA_FILE}", fs=fs)

        # the manifest will be compared on save
        info = fs.get_file_info(f"{fs_path}/{MANIFEST_FILE}")
        if info.is_file:
            manifest = DatasetManifest.load(path=f"{fs_path}/{MANIFEST_FILE}", fs=fs)

            # validate manifest
            is_valid = manifest.validate(fs_path, fs)
            if not is_valid:
                # invalid manifest, sync from remote
                print("[!] Remote dataset manifest validation failed.")

        # load dataset
        dataset = ds.dataset(f"{fs_path}/data", format=format, filesystem=fs, **kwargs)

        return Dataset(ds=dataset, metadata=metadata, manifest=manifest, materialize=materialize)
    except Exception as e:
        print(f"[!] Failed to load dataset from remote: {e}")

    raise FileNotFoundError(f"[!] Dataset not found: {uri}")

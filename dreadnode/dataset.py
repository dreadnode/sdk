from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
from fsspec.core import strip_protocol
from fsspec.utils import get_protocol
from pyarrow.fs import FileSystem

from dreadnode.constants import MANIFEST_FILE, METADATA_FILE
from dreadnode.logging_ import print_info
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
            print_info("[*] No metadata provided, check your dataset!")

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
        create_dir: bool = False,
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
            create_dir=create_dir,
            **kwargs,
        )


def _persist_dataset(
    dataset: Dataset,
    path_str: str,
    *,
    create_dir: bool = False,
    fsm: DatasetManager,
    **kwargs: Any,
) -> None:
    """Persists a dataset to the given path.

    Args:
        dataset: The Dataset to persist.
        path_str: The path to persist the dataset to.
        create_dir: Whether to create the directory if it doesn't exist. Defaults to False.
        fsm: The DatasetManager instance.
        kwargs: Additional arguments to pass to pyarrow.dataset.write_dataset.
    """
    fs, base_path = fsm.get_fs_and_path(path_str)

    fsm.ensure_dir(fs, base_path)
    data_path = f"{base_path}/data"
    fsm.ensure_dir(fs, data_path)

    dataset.save_dataset(path=data_path, fs=fs, create_dir=create_dir, **kwargs)

    dataset.save_metadata(path=f"{base_path}/{METADATA_FILE}", fs=fs)

    manifest = create_manifest(
        path=base_path,
        version=dataset.metadata.version,
        previous_manifest=dataset.manifest if dataset.manifest else None,
        fs=fs,
    )
    manifest.save(f"{base_path}/{MANIFEST_FILE}", fs=fs)

    print_info("[+] Saved dataset successfully")


def save_dataset_to_disk(
    dataset: Dataset,
    *,
    fsm: DatasetManager,
    **kwargs: Any,
) -> None:
    """Saves a dataset to local disk cache.

    Args:
        dataset: The Dataset to save.
        fsm: The DatasetManager instance.
        kwargs: Additional arguments to pass to pyarrow.dataset.write_dataset.

    Returns:
        None

    """
    path_str = fsm.get_cache_save_uri(metadata=dataset.metadata)
    print_info("[*] Saving dataset to local cache")

    _persist_dataset(
        dataset=dataset,
        path_str=path_str,
        create_dir=True,
        fsm=fsm,
        **kwargs,
    )


def push_dataset(
    dataset: Dataset,
    *,
    to_cache: bool = True,
    fsm: DatasetManager,
    **kwargs: Any,
) -> None:
    """Pushes a dataset to remote storage.

    Args:
        dataset: The Dataset to push.
        to_cache: Whether to save to local cache first. Defaults to True.
        fsm: The DatasetManager instance.
        kwargs: Additional arguments to pass to pyarrow.dataset.write_dataset.

    Returns:
        None
    """
    if to_cache:
        save_dataset_to_disk(
            dataset=dataset,
            fsm=fsm,
            **kwargs,
        )

    dataset_id, path_str = fsm.get_remote_save_uri(metadata=dataset.metadata)
    dataset.metadata.id = dataset_id
    print_info("[*] Saving dataset to remote storage")
    try:
        _persist_dataset(
            dataset=dataset,
            path_str=path_str,
            fsm=fsm,
            **kwargs,
        )
        fsm.remote_save_complete(complete=True, dataset_id=dataset.metadata.id)
    except Exception:
        # if remote save failed, remove the record from API
        fsm.delete_remote_dataset_record(dataset_id_or_key=dataset.metadata.id)
        raise


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
            print_info("[+] Dataset not found in cache. Loading dataset from local path...")

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
        print_info("[+] Loading dataset from cache...")

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
    print_info("[+] Loading from remote storage...")
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
                print_info("[!] Remote dataset manifest validation failed.")

        # load dataset
        dataset = ds.dataset(f"{fs_path}/data", format=format, filesystem=fs, **kwargs)

        return Dataset(ds=dataset, metadata=metadata, manifest=manifest, materialize=materialize)
    except Exception as e:
        print_info(f"[!] Failed to load dataset from remote: {e}")

    raise FileNotFoundError(f"[!] Dataset not found: {uri}")

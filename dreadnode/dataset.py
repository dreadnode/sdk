import hashlib
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
from fsspec.core import strip_protocol
from fsspec.utils import get_protocol
from pyarrow.fs import FileSystem

from dreadnode.constants import MANIFEST_FILE, METADATA_FILE
from dreadnode.logging_ import print_info, print_success, print_warning
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
        self.row_count: int = 0

        if materialize:
            self.ds = ds.to_table()

        if not metadata:
            print_info("[*] No metadata provided, check your dataset!")

    def compute_dataset_fingerprint(self) -> str:
        """
        Computes a deterministic hash of the dataset content without writing to disk.
        Handles both pyarrow.Table and pyarrow.dataset.Dataset.

        Returns:
            str: The SHA256 hash of the dataset content.
        """
        hasher = hashlib.sha256()

        # 1. Hash Metadata (Schema & Row Count)
        # This catches structural changes instantly
        schema_str = self.ds.schema.to_string(show_field_metadata=True)
        hasher.update(schema_str.encode("utf-8"))

        if self.row_count is not None:
            hasher.update(str(self.row_count).encode("utf-8"))

        # 2. Get the Batch Iterator
        # pyarrow.Table uses to_batches(), pyarrow.dataset.Dataset uses scanner()
        if isinstance(self.ds, pa.Table):
            # max_chunksize ensures we don't try to hash 10GB in one go if it's a single chunk
            batch_iter = self.ds.to_batches(max_chunksize=64 * 1024)
        else:
            batch_iter = self.ds.scanner(batch_size=64 * 1024).to_batches()

        # 3. Hash Content (Streaming)
        for batch in batch_iter:
            for column in batch.columns:
                # Hash the validity bitmap (null positions) and the value buffers
                for buffer in column.buffers():
                    if buffer:
                        hasher.update(buffer)

        return hasher.hexdigest()

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
        except pa.ArrowException:
            self.row_count = 0

    def save_metadata(self, path: str, fs: FileSystem) -> None:
        """
        Helper to write Pydantic models to native binary streams.
        """
        if self.metadata is None:
            raise ValueError("Dataset metadata is not set")

        self.metadata.fingerprint = self.get_content_fingerprint()

        # update metadata fields before saving
        self.metadata.save(path, fs)

    def save_readme(self, path: str, fs: FileSystem) -> None:
        """
        Helper to write README content to native binary streams.
        """
        if self.metadata is None or self.metadata.readme is None:
            print_info("[*] No README content to save.")
        else:
            with fs.open_output_stream(path) as f:
                f.write(self.metadata.readme.encode("utf-8"))

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

    def get_content_fingerprint(self) -> str:
        """
        Generates a unique hash of the current in-memory content (schema + data).
        """
        # Reuse the optimized logic we discussed previously
        return self.compute_dataset_fingerprint()


def _ensure_version_bump(
    dataset: Dataset, dataset_manager: DatasetManager, latest_path: str | None = None
) -> bool:
    """Ensures that the dataset version is bumped if content has changed.

    Args:
        dataset: The Dataset to check.
        dataset_manager: The DatasetManager instance.
        latest_path: The path to the latest saved dataset version.

    Returns:
        bool: True if the dataset should be saved (version bumped or new), False otherwise.
    """

    if not latest_path:
        print_info("[*] No previous version found. Saving new dataset.")
        return True

    latest_version = dataset_manager.get_version_from_path(latest_path)

    fs, clean_path = dataset_manager.get_fs_and_path(latest_path)

    # 1. New dataset (no history) -> Always save
    manifest_exists = dataset_manager.manifest_exists(clean_path, fs)
    if dataset.manifest is None and not manifest_exists:
        return True

    print_info("[*] Checking for changes against previous version...")

    # Load previous metadata to get the old fingerprint

    latest_metadata = DatasetMetadata.load(path=f"{clean_path}/{METADATA_FILE}", fs=fs)
    latest_fingerprint = latest_metadata.fingerprint

    if dataset.metadata.version > latest_version:
        print_info("[*] Dataset version is already higher than previous. Skipping version check.")
        return True

    # 2. Compute New Fingerprint
    dataset.metadata.fingerprint = dataset.get_content_fingerprint()

    # 3. Compare
    if latest_fingerprint and dataset.metadata.fingerprint == latest_fingerprint:
        print_info("[!] No content changes detected (Fingerprint match).")
        print_info("[!] Save skipped.")
        return False

    # 4. Fallback for Legacy Datasets (No old fingerprint)
    # If the old dataset exists but has no fingerprint, we technically
    # should read IT to compute its fingerprint, or just assume "Modified" to be safe.
    # For enormous datasets, it is safer/faster to assume "Modified" than to read the old one.
    if latest_fingerprint is None:
        print_info("[*] No previous fingerprint found. forcing update to establish baseline.")
        return True

    # 5. Data Changed: Bump Version
    dataset.metadata.update_version(latest_version)
    print_warning(f"[+] Changes detected. Auto-bumping version to {dataset.metadata.version}")
    return True


def _persist_dataset(
    dataset: Dataset,
    path_str: str,
    *,
    create_dir: bool = False,
    dataset_manager: DatasetManager,
    **kwargs: Any,
) -> None:
    """Persists a dataset to the given path.

    Args:
        dataset: The Dataset to persist.
        path_str: The path to persist the dataset to.
        create_dir: Whether to create the directory if it doesn't exist. Defaults to False.
        dataset_manager: The DatasetManager instance.
        kwargs: Additional arguments to pass to pyarrow.dataset.write_dataset.
    """
    fs, base_path = dataset_manager.get_fs_and_path(path_str)

    dataset_manager.ensure_dir(fs, base_path)
    data_path = f"{base_path}/data"
    dataset_manager.ensure_dir(fs, data_path)

    dataset.save_dataset(path=data_path, fs=fs, create_dir=create_dir, **kwargs)

    dataset.save_metadata(path=f"{base_path}/{METADATA_FILE}", fs=fs)

    dataset.save_readme(path=f"{base_path}/README.md", fs=fs)

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
    dataset_manager: DatasetManager,
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

    if dataset.metadata.auto_version:
        latest_path = dataset_manager.get_latest_cache_save_uri(metadata=dataset.metadata)
        should_save = _ensure_version_bump(dataset, dataset_manager, latest_path)
        if not should_save:
            return
    print_info("[*] Saving dataset to local cache")

    _persist_dataset(
        dataset=dataset,
        path_str=dataset_manager.get_cache_save_uri(metadata=dataset.metadata),
        create_dir=True,
        dataset_manager=dataset_manager,
        **kwargs,
    )


def push_dataset(
    dataset: Dataset,
    *,
    to_cache: bool = True,
    dataset_manager: DatasetManager,
    **kwargs: Any,
) -> None:
    """Pushes a dataset to remote storage.

    Args:
        dataset: The Dataset to push.
        to_cache: Whether to save to local cache first. Defaults to True.
        dataset_manager: The DatasetManager instance.
        kwargs: Additional arguments to pass to pyarrow.dataset.write_dataset.

    Returns:
        None
    """
    if dataset.metadata.auto_version:
        latest_path = dataset_manager.get_remote_load_uri(
            uri=f"{dataset.metadata.organization}/{dataset.metadata.name}", version="latest"
        )
        should_save = _ensure_version_bump(dataset, dataset_manager, latest_path)
        if not should_save:
            return

    dataset_id, path_str = dataset_manager.get_remote_save_uri(metadata=dataset.metadata)
    dataset.metadata.id = dataset_id

    if to_cache:
        save_dataset_to_disk(
            dataset=dataset,
            dataset_manager=dataset_manager,
            **kwargs,
        )

    print_info("[*] Saving dataset to remote storage")
    try:
        _persist_dataset(
            dataset=dataset,
            path_str=path_str,
            dataset_manager=dataset_manager,
            **kwargs,
        )
        print_info("[*] Dataset saved to remote storage successfully. Updating remote record...")
        dataset_manager.remote_save_complete(complete=True, dataset_id=dataset.metadata.id)
        print_success("[+] Dataset push complete.")
    except Exception:
        # if remote save failed, remove the record from API
        print_info("[!] Remote dataset save failed. Cleaning up remote record...")
        dataset_manager.delete_remote_dataset_record(dataset_id_or_key=dataset.metadata.id)
        print_info("[*] Remote record cleaned up.")
        raise


def load_dataset(
    uri: str,
    version: str | None = None,
    format: str = "parquet",
    *,
    materialize: bool = True,
    dataset_manager: DatasetManager,
    **kwargs: dict[str, Any],
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
        if not dataset_manager.check_cache(uri, version):
            print_info("[+] Dataset not found in cache. Loading dataset from local path...")

            # load directly from local path
            fs, fs_path = dataset_manager.get_fs_and_path(uri)

            if not Path(fs_path).exists():
                raise FileNotFoundError(f"[!] Dataset not found at local path: {uri}")

            # load dataset
            dataset = ds.dataset(fs_path, format=format, filesystem=fs, **kwargs)

            # metadata and manifest files are not expected in local path
            metadata = DatasetMetadata(organization=dataset_manager.organization, format=format)

            return Dataset(ds=dataset, metadata=metadata, materialize=materialize)

        # if in cache, load from cache
        print_info("[+] Loading dataset from cache...")

        # get the filesystem and path
        fs, fs_path = dataset_manager.get_fs_and_path(uri)

        # resolve versioned cache path
        cache_path = dataset_manager.get_cache_load_uri(fs_path, version, fs)

        # metadata and manifest files are expected in cache
        if dataset_manager.metadata_exists(path=cache_path):
            metadata = DatasetMetadata.load(path=f"{cache_path}/{METADATA_FILE}", fs=fs)

        # the manifest will be compared on save
        if dataset_manager.manifest_exists(path=cache_path):
            manifest = DatasetManifest.load(path=f"{cache_path}/{MANIFEST_FILE}", fs=fs)

        # load dataset
        dataset = ds.dataset(f"{cache_path}/data", format=format, filesystem=fs, **kwargs)

        return Dataset(ds=dataset, materialize=materialize, metadata=metadata, manifest=manifest)

    # if not local path, and not in cache, load from remote
    print_info("[+] Loading from remote storage...")
    try:
        # get remote URI
        remote_uri = dataset_manager.get_remote_load_uri(strip_protocol(uri), version)

        # get the filesystem and path
        fs, fs_path = dataset_manager.get_fs_and_path(remote_uri)

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
        raise FileNotFoundError(f"[!] Dataset not found: {uri}") from e

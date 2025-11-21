import uuid
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from fsspec import AbstractFileSystem
from fsspec.utils import get_protocol
from pydantic import BaseModel, ConfigDict, Field

from dreadnode.constants import DATASETS_CACHE, MANIFEST_FILE, METADATA_FILE
from dreadnode.logging_ import console as logging_console

# Import the high-performance filesystem tools
from dreadnode.storage.datasets.core import (
    SyncStrategy,
    get_filesystem,
    sync_to_local,
    sync_to_remote,
)
from dreadnode.storage.datasets.loader import DatasetLoader, DatasetWriter
from dreadnode.storage.datasets.manifest import FileManifest, ManifestBuilder
from dreadnode.storage.datasets.metadata import DatasetMetadata, DatasetSchema


class Dataset(BaseModel):
    """
    A versioned dataset with complete metadata tracking.
    """

    ds: ds.Dataset | pa.Table
    name: str | None = None
    description: str | None = None
    version: str = "0.0.1"
    license: str | None = None
    tags: list[str] = Field(default_factory=list)
    format: str = "parquet"

    # Private attributes
    _metadata: DatasetMetadata | None = None
    _uri: str | None = None
    _manifest: FileManifest | None = None
    # _history: VersionHistory | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, _context: Any) -> None:
        if self._metadata:
            self.name = self._metadata.name
            self.description = self._metadata.description
            self.version = self._metadata.version
            self.license = self._metadata.license
            self.tags = self._metadata.tags
            self.format = self._metadata.format
            self._uri = self._metadata.uri
            return

        if not self.name:
            self.name = f"dataset-{uuid.uuid4().hex[:8]}"

        if not self._uri:
            self._uri = self._normalize_uri(self.name)

    @staticmethod
    def _normalize_uri(name: str) -> str:
        return name.replace(" ", "_").replace("/", "-").lower()

    @property
    def uri(self) -> str:
        if not self._uri:
            self._uri = self._normalize_uri(self.name or "dataset")
            print(self._uri)
        return self._uri

    @property
    def schema(self) -> pa.Schema:
        if isinstance(self.ds, ds.Dataset):
            return self.ds.schema
        return self.ds.schema

    @property
    def ds_schema(self) -> DatasetSchema:
        return DatasetSchema.from_arrow_schema(self.schema)

    @property
    def metadata(self) -> DatasetMetadata:
        if not self._metadata:
            self._metadata = self.create_metadata()
        return self._metadata

    @property
    def manifest(self) -> FileManifest | None:
        if not self._manifest:
            logging_console.print("Building dataset manifest...")
            self._manifest = ManifestBuilder.build_from_directory(
                get_dataset_path(self.uri, self.version),
                version=self.version,
                exclude_patterns=[METADATA_FILE],
            )
        return self._manifest

    @property
    def files(self) -> list[str]:
        if isinstance(self.ds, ds.Dataset):
            return self.ds.files
        return []

    def create_metadata(
        self,
    ) -> DatasetMetadata:  # TODO: Some fields not being calculated, bytes. But are written to disk. Some race condition.
        metadata = DatasetMetadata(
            name=self.name or "dataset",
            description=self.description,
            version=self.version,
            license=self.license,
            tags=self.tags,
            uri=self.uri,
            ds_schema=self.ds_schema,
            format=self.format,
            files=self.files or [],
        )
        metadata.update_from_data(self.ds, self.files)
        return metadata

    def save(
        self,
        name: str | None = None,
        version: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        license: str | None = None,
        *,
        to_remote: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Save the dataset to cache and optionally sync to remote.
        """
        self.name = name or self.name
        self.version = version or self.version or "0.0.1"
        self.description = description or self.description
        self.tags = tags or self.tags
        self.license = license or self.license

        dataset_path = get_dataset_path(self.uri, self.version)

        dataset_path.mkdir(parents=True, exist_ok=True)

        DatasetWriter.write(
            self.ds,
            str(dataset_path / "data"),
            format=self.format,
            filesystem=None,  # TODO: Save to remote
            **kwargs,
        )

        if not self._metadata:
            self._metadata = self.create_metadata()
        self._metadata.save(dataset_path / METADATA_FILE)

        if not self._manifest:
            self._manifest = ManifestBuilder.build_from_directory(
                dataset_path, version=self.version
            )
        self._manifest.save(dataset_path / MANIFEST_FILE)

        if to_remote:
            remote_url = f"dn://{self.uri}/{self.version}"
            logging_console.print(f"Syncing dataset to remote: {remote_url}")

            stats = sync_to_remote(
                local_dir=dataset_path,
                remote_url=remote_url,
                strategy=SyncStrategy.HASH,
                delete=False,
            )
            logging_console.print(f"Dataset saved and synced: {stats}")

    def to_table(self) -> pa.Table:
        if isinstance(self.ds, pa.Table):
            return self.ds
        return self.ds.to_table()

    def head(self, n: int = 5) -> pa.Table:
        table = self.to_table()
        return table.slice(0, min(n, len(table)))

    def filter(self, expression: pc.Expression) -> "Dataset":
        if isinstance(self.ds, pa.Table):
            filtered = self.ds.filter(expression)
        else:
            filtered = self.ds.to_table().filter(expression)
        return self._reconstruct(filtered)

    def select(self, columns: list[str]) -> "Dataset":
        table = self.to_table()
        selected = table.select(columns)
        return self._reconstruct(selected)

    def _reconstruct(self, new_ds: pa.Table) -> "Dataset":
        return Dataset(
            ds=new_ds,
            name=self.name,
            description=self.description,
            version=self.version,
            license=self.license,
            tags=self.tags,
            format=self.format,
        )

    @classmethod
    def from_path(
        cls,
        path: str,
        *,
        lazy: bool = False,
        format: str | None = None,
        fs: AbstractFileSystem | None = None,
        load_metadata: bool = True,
        **kwargs: Any,
    ) -> "Dataset":
        if fs is None:
            fs, path = get_filesystem(path)

        # Try to load metadata
        metadata = None
        if load_metadata:
            try:
                meta_path = f"{path}/{METADATA_FILE}"
                if fs.exists(meta_path):
                    metadata = DatasetMetadata.load(meta_path, fs=fs)
                    format = format or metadata.format
            except Exception as e:
                logging_console.print(f"Could not load metadata: {e}")

        # Determine data path (assuming standard structure)
        # Check if 'data' subdir exists
        data_path = f"{path}/data"
        if not fs.exists(data_path):
            data_path = path

        ds_obj = DatasetLoader.load(
            path=data_path,
            format=format,
            filesystem=fs,
            lazy=lazy,
            **kwargs,
        )

        if metadata:
            return cls(ds=ds_obj, _metadata=metadata)

        return cls(
            ds=ds_obj,
            name=Path(path).name,
            format=format or "parquet",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any], metadata: dict[str, Any] | None = None) -> "Dataset":
        table = pa.Table.from_pydict(data)
        meta_obj = DatasetMetadata(**metadata) if metadata else None
        return cls(ds=table, _metadata=meta_obj)

    @classmethod
    def from_pandas(cls, df: Any, metadata: dict[str, Any] | None = None) -> "Dataset":
        table = pa.Table.from_pandas(df)
        meta_obj = DatasetMetadata(**metadata) if metadata else None
        return cls(ds=table, _metadata=meta_obj)

    def to_pandas(self) -> Any:
        return self.to_table().to_pandas()

    def __len__(self) -> int:
        if isinstance(self.ds, pa.Table):
            return len(self.ds)
        return self.ds.count_rows()


def parse_uri(uri: str) -> tuple[str, str | None]:
    """Parse dataset URI into org/name and version."""
    # Remove protocol (dn://, s3://)
    if "://" in uri:
        uri = uri.split("://", 1)[1]

    if "@" in uri:
        path, version = uri.rsplit("@", 1)
        return path, version

    # Also support path/version syntax which dn protocol maps to
    parts = uri.split("/")
    if len(parts) > 2:
        return f"{parts[0]}/{parts[1]}", parts[2]

    return uri


def get_dataset_path(
    uri: str,
    version: str | None = None,
    *,
    is_remote: bool = False,
) -> Path:
    """Get local cache path for a dataset."""
    print(uri)
    dataset_path = Path(DATASETS_CACHE) / uri
    print("get_dataset_path", dataset_path)

    if not version:
        # If version not specified, find latest
        versions = [d.name for d in dataset_path.iterdir() if d.is_dir()]
        if not versions:
            raise FileNotFoundError(f"No cached versions found for dataset: {uri}")
        versions.sort(reverse=True)  # Assuming semantic versioning
        version = versions[0]

    return dataset_path / version


def is_cached(uri: str, version: str | None = None) -> bool:
    """Check if dataset exists in local cache."""
    path = get_dataset_path(uri, version)
    return (path).exists()


def load_dataset(
    uri: str,
    version: str | None = None,
    *,
    lazy: bool = False,
    from_remote: bool = False,
    sync: bool = True,
    **kwargs: Any,
) -> Dataset:
    """
    Load dataset from local path, cache, or remote storage.

    Resolution Order:
    1. If `uri` is an explicit local path that exists -> Load Local.
    2. If `from_remote=True` -> Stream directly from Remote.
    3. If `sync=True` -> Sync from Remote (DN) to Cache -> Load Cache.
    4. Check Cache -> Load Cache.
    """
    protocol = get_protocol(uri)

    if protocol == "file":
        # Check if it's a local path
        local_path = Path(uri).expanduser().resolve()
        if local_path.exists():
            return Dataset.from_path(
                path=str(local_path),
                lazy=lazy,
                **kwargs,
            )

    # If we are here, it's either:
    # a) An explicit remote URL (s3://, dn://)
    # b) A managed dataset key (org/dataset) that doesn't exist locally

    # If it looks like "org/dataset", treat it as "dn://org/dataset" for syncing
    if protocol == "file":
        # It's a managed key, not an existing path
        parsed_uri, parsed_version = parse_uri(uri)
    else:
        # It's already a remote protocol (s3://, dn://)
        parsed_uri, parsed_version = parse_uri(uri)

    dataset_path = get_dataset_path(parsed_uri, parsed_version)

    if from_remote:
        return Dataset.from_path(dataset_path, lazy=lazy, version=version, **kwargs)

    should_sync = sync and (protocol in ("dn", "dreadnode") or protocol == "file")

    if should_sync:
        sync_url = f"dn://{parsed_uri}/{version}" if protocol == "file" else dataset_path

        if (
            protocol in ("dn", "dreadnode")
            and "@" not in sync_url
            and "/" not in sync_url.replace("://", "")
        ):
            sync_url = f"{sync_url}/{version}"

        try:
            sync_to_local(
                remote_url=sync_url,
                local_dir=dataset_path,
                strategy=SyncStrategy.SIZE_AND_TIMESTAMP,
                delete=False,
            )
        except Exception as e:
            logging_console.print(f"Sync failed ({e}). Attempting to load existing cache...")

    # load from Cache
    if dataset_path.exists():
        return Dataset.from_path(
            path=str(dataset_path),
            lazy=lazy,
            **kwargs,
        )

    # Failure
    raise FileNotFoundError(
        f"Dataset not found.\n"
        f" - Checked Local Path: {uri}\n"
        f" - Checked Cache: {dataset_path}\n"
        f" - Remote Sync Attempted: {should_sync}"
    )

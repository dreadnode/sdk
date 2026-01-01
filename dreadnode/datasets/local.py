"""Local dataset storage without package installation."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from dreadnode.core.packaging.manifest import DatasetManifest
from dreadnode.core.storage.storage import Storage, hash_file
from dreadnode.core.settings import DEFAULT_CACHE_DIR

if TYPE_CHECKING:
    import datasets
    import pyarrow as pa


def load_file(path: Path, fmt: str | None = None) -> pa.Table:
    """Load file as PyArrow Table.

    Supports parquet, csv, arrow, feather, json, jsonl formats.

    Args:
        path: Path to the file.
        fmt: Format override. If None, inferred from extension.

    Returns:
        PyArrow Table with the data.
    """

    fmt = fmt or path.suffix.lstrip(".")

    if fmt == "parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path)
    if fmt == "csv":
        from pyarrow import csv

        return csv.read_csv(path)
    if fmt in ("arrow", "feather"):
        from pyarrow import feather

        return feather.read_table(path)
    if fmt in ("json", "jsonl"):
        import pyarrow.json as pj

        return pj.read_json(path)

    raise ValueError(f"Unknown format: {fmt}")


def write_table(
    table: pa.Table,
    path: Path,
    fmt: Literal["parquet", "arrow", "feather"] = "parquet",
) -> None:
    """Write PyArrow Table to file.

    Args:
        table: PyArrow Table to write.
        path: Destination path.
        fmt: Output format.
    """
    if fmt == "parquet":
        import pyarrow.parquet as pq

        pq.write_table(table, path)
    elif fmt in ("arrow", "feather"):
        from pyarrow import feather

        feather.write_feather(table, path)
    else:
        raise ValueError(f"Unsupported write format: {fmt}")


class LocalDataset:
    """Dataset stored in CAS, usable without package installation.

    This class provides a way to work with datasets stored in the
    Content-Addressable Storage without requiring them to be installed
    as Python packages with entry points.

    Example:
        >>> from dreadnode.datasets import LocalDataset
        >>> from dreadnode.core.storage import Storage
        >>>
        >>> storage = Storage()
        >>>
        >>> # Create from HuggingFace dataset
        >>> from datasets import load_dataset
        >>> hf_ds = load_dataset("squad", split="train[:100]")
        >>> local_ds = LocalDataset.from_hf(hf_ds, "my-squad", storage)
        >>>
        >>> # Use with HuggingFace features
        >>> ds = local_ds.to_hf()
        >>> ds = ds.map(lambda x: {"lower": x["question"].lower()})
        >>>
        >>> # Load existing dataset
        >>> local_ds = LocalDataset("my-squad", storage)
    """

    def __init__(
        self,
        name: str,
        storage: Storage,
        version: str | None = None,
    ):
        """Load a local dataset by name.

        Args:
            name: Dataset name.
            storage: Storage instance for CAS access.
            version: Specific version to load. If None, loads latest.
        """
        self.name = name
        self.storage = storage

        if version is None:
            version = storage.latest_version("datasets", name)
            if version is None:
                raise FileNotFoundError(f"Dataset not found: {name}")

        self.version = version
        self._manifest: DatasetManifest | None = None

    @property
    def manifest(self) -> DatasetManifest:
        """Load and cache the manifest."""
        if self._manifest is None:
            content = self.storage.get_manifest("datasets", self.name, self.version)
            self._manifest = DatasetManifest.model_validate_json(content)
        return self._manifest

    @property
    def format(self) -> str:
        """Data format (parquet, csv, arrow, etc.)."""
        return self.manifest.format

    @property
    def schema(self) -> dict[str, str]:
        """Column schema."""
        return self.manifest.data_schema

    @property
    def row_count(self) -> int | None:
        """Number of rows."""
        return self.manifest.row_count

    @property
    def files(self) -> list[str]:
        """List of artifact file paths."""
        return list(self.manifest.artifacts.keys())

    @property
    def splits(self) -> list[str] | None:
        """Available splits, if any."""
        if self.manifest.splits:
            return list(self.manifest.splits.keys())
        return None

    def _resolve(self, path: str) -> Path:
        """Resolve artifact path to local file."""
        if path not in self.manifest.artifacts:
            raise FileNotFoundError(f"Artifact not in manifest: {path}")

        oid = self.manifest.artifacts[path]
        local_path = self.storage.blob_path(oid)

        if not local_path.exists():
            # Try to download from remote
            self.storage.download_blob(oid)

        return local_path

    def load(self, split: str | None = None) -> pa.Table:
        """Load dataset as PyArrow Table.

        Args:
            split: Optional split name to load (e.g., "train", "test").
                   If None, loads the first/only file.

        Returns:
            PyArrow Table with the data.
        """
        if split and self.manifest.splits:
            if split not in self.manifest.splits:
                raise ValueError(
                    f"Unknown split: {split}. Available: {list(self.manifest.splits.keys())}"
                )
            path = self.manifest.splits[split]
        elif split:
            # Try to match split in filename
            matches = [f for f in self.files if split in f]
            if not matches:
                raise ValueError(f"No file matching split '{split}'")
            path = matches[0]
        else:
            path = self.files[0]

        local_path = self._resolve(path)
        return load_file(local_path, self.format)

    def to_hf(self, split: str | None = None) -> datasets.Dataset:
        """Load and convert to HuggingFace Dataset.

        Args:
            split: Optional split to load.

        Returns:
            HuggingFace Dataset with full functionality.
        """
        from dreadnode.datasets.hf import arrow_to_hf

        table = self.load(split)
        return arrow_to_hf(table)

    def to_pandas(self, split: str | None = None) -> Any:
        """Load as pandas DataFrame.

        Args:
            split: Optional split to load.

        Returns:
            pandas DataFrame.
        """
        return self.load(split).to_pandas()

    @classmethod
    def from_hf(
        cls,
        hf_dataset: datasets.Dataset | datasets.DatasetDict,
        name: str,
        storage: Storage,
        format: Literal["parquet", "arrow", "feather"] = "parquet",
        version: str = "0.1.0",
    ) -> LocalDataset:
        """Store HuggingFace Dataset in CAS and return LocalDataset.

        Args:
            hf_dataset: HuggingFace Dataset or DatasetDict to store.
            name: Name for the dataset.
            storage: Storage instance for CAS access.
            format: Output format (parquet, arrow, feather).
            version: Version string.

        Returns:
            LocalDataset instance for the stored data.

        Example:
            >>> from datasets import load_dataset
            >>> hf_ds = load_dataset("squad", split="train[:100]")
            >>> local_ds = LocalDataset.from_hf(hf_ds, "my-squad", storage)
        """
        from dreadnode.datasets.hf import infer_schema, require_datasets

        require_datasets()
        import datasets as hf_datasets

        artifacts: dict[str, str] = {}
        splits: dict[str, str] | None = None
        total_rows = 0
        data_schema: dict[str, str] = {}

        if isinstance(hf_dataset, hf_datasets.DatasetDict):
            # Handle multiple splits
            splits = {}
            for split_name, split_ds in hf_dataset.items():
                filename = f"{split_name}.{format}"
                oid = cls._store_dataset(split_ds, filename, storage, format)
                artifacts[filename] = oid
                splits[split_name] = filename
                total_rows += len(split_ds)

                if not data_schema:
                    data_schema = infer_schema(split_ds)
        else:
            # Single dataset
            filename = f"data.{format}"
            oid = cls._store_dataset(hf_dataset, filename, storage, format)
            artifacts[filename] = oid
            total_rows = len(hf_dataset)
            data_schema = infer_schema(hf_dataset)

        # Create manifest
        manifest = DatasetManifest(
            format=format,
            data_schema=data_schema,
            row_count=total_rows,
            artifacts=artifacts,
            splits=splits,
        )

        # Store manifest
        manifest_json = manifest.model_dump_json(indent=2)
        storage.store_manifest("datasets", name, version, manifest_json)

        return cls(name, storage, version)

    @classmethod
    def _store_dataset(
        cls,
        hf_dataset: datasets.Dataset,
        filename: str,
        storage: Storage,
        format: Literal["parquet", "arrow", "feather"],
    ) -> str:
        """Store a single HF Dataset in CAS.

        Returns:
            The oid of the stored blob.
        """
        from dreadnode.datasets.hf import hf_to_arrow

        table = hf_to_arrow(hf_dataset)

        # Write to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / filename
            write_table(table, tmp_path, format)

            # Hash and store
            file_hash = hash_file(tmp_path)
            oid = f"sha256:{file_hash}"
            storage.store_blob(oid, tmp_path)

        return oid

    def publish(self, version: str | None = None) -> None:
        """Create a DN package for signing and distribution.

        This converts the local dataset into a proper Python package
        with entry points that can be installed and discovered.

        Args:
            version: Version for the package. If None, uses current version.

        Raises:
            NotImplementedError: Package creation not yet implemented.
        """
        raise NotImplementedError(
            "Package publishing is not yet implemented. Use the CLI to create and publish packages."
        )

    def __repr__(self) -> str:
        return f"LocalDataset(name={self.name!r}, version={self.version!r})"


def load_dataset(
    path: str,
    *,
    name: str | None = None,
    storage: Storage | None = None,
    split: str | None = None,
    format: Literal["parquet", "arrow", "feather"] = "parquet",
    version: str = "0.1.0",
    **kwargs: Any,
) -> LocalDataset:
    """Load a dataset from HuggingFace Hub and store in local CAS.

    This is a convenience function that wraps HuggingFace's load_dataset
    and automatically stores the result in Content-Addressable Storage.

    Args:
        path: HuggingFace dataset path (e.g., "squad", "imdb").
        name: Name to store the dataset as. Defaults to the path.
        storage: Storage instance. If None, creates default storage.
        split: Dataset split to load (e.g., "train", "test", "train[:100]").
        format: Storage format (parquet, arrow, feather).
        version: Version string for the stored dataset.
        **kwargs: Additional arguments passed to HuggingFace's load_dataset.

    Returns:
        LocalDataset instance with the loaded data.

    Example:
        >>> from dreadnode.datasets import load_dataset
        >>>
        >>> # Load and store a HuggingFace dataset
        >>> ds = load_dataset("squad", split="train[:100]")
        >>> ds = ds.to_hf().map(lambda x: {"lower": x["question"].lower()})
        >>>
        >>> # Load with custom name and storage
        >>> ds = load_dataset("imdb", name="my-imdb", split="train")
    """
    from dreadnode.datasets.hf import require_datasets

    require_datasets()
    import datasets as hf_datasets

    # Create default storage if not provided
    if storage is None:
        storage = Storage(cache=DEFAULT_CACHE_DIR)

    # Use path as name if not specified
    if name is None:
        # Clean the path for use as a name (e.g., "squad" -> "squad")
        name = path.replace("/", "-").replace(":", "-")

    # Load from HuggingFace
    hf_dataset = hf_datasets.load_dataset(path, split=split, **kwargs)

    # Store in CAS and return LocalDataset
    return LocalDataset.from_hf(
        hf_dataset,
        name=name,
        storage=storage,
        format=format,
        version=version,
    )

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from dreadnode.core.packaging.loader import BaseLoader
from dreadnode.core.packaging.manifest import DatasetManifest

if TYPE_CHECKING:
    import pyarrow as pa


class Dataset(BaseLoader):
    """Loader for dataset components."""

    entry_point_group = "dreadnode.datasets"
    manifest_class = DatasetManifest

    @property
    def format(self) -> str:
        return self.manifest.format

    @property
    def schema(self) -> dict[str, str]:
        return self.manifest.data_schema

    @property
    def row_count(self) -> int | None:
        return self.manifest.row_count

    def load(self, split: str | None = None) -> pa.Table:
        """Load dataset as PyArrow Table."""
        if split:
            matches = [f for f in self.files if split in f]
            if not matches:
                raise ValueError(f"No file matching split '{split}'")
            path = self.resolve(matches[0])
        else:
            path = self.resolve(self.files[0])

        return self._load_file(path)

    def _load_file(self, path: Path) -> pa.Table:
        fmt = self.format or path.suffix.lstrip(".")

        if fmt == "parquet":
            import pyarrow.parquet as pq

            return pq.read_table(path)
        if fmt == "csv":
            from pyarrow import csv

            return csv.read_csv(path)
        if fmt in ("arrow", "feather"):
            from pyarrow import feather

            return feather.read_table(path)
        raise ValueError(f"Unknown format: {fmt}")

    def to_pandas(self, split: str | None = None) -> Any:
        """Load as pandas DataFrame."""
        return self.load(split).to_pandas()


def load_dataset(component: str | Path) -> Dataset:
    """Load a dataset component.

    Args:
        component: Component identifier or path.

    Returns:
        Loaded Dataset.
    """
    loader = Dataset.load_component(component)
    if not isinstance(loader, Dataset):
        raise TypeError(f"Component is not a Dataset: {type(loader)}")
    return loader

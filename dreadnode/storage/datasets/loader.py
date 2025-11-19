"""
Dataset loaders for different file formats.
"""

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
from fsspec import AbstractFileSystem
from pyarrow import csv

from dreadnode.constants import SUPPORTED_FORMATS
from dreadnode.logging_ import console as logging_console
from dreadnode.storage.datasets.core import get_filesystem


class DatasetLoader:
    """Handles loading datasets from various formats."""

    @staticmethod
    def detect_format(path: str, filesystem: AbstractFileSystem | None = None) -> str:
        """Detect format from file extension or directory content."""
        # If no filesystem provided, resolve it to handle remote URLs
        if filesystem is None:
            try:
                filesystem, path = get_filesystem(path)
            except Exception:
                pass  # Fallback to local logic if valid URL check fails

        # Check file extension first
        suffix = Path(path).suffix.lower().lstrip(".")
        if suffix in SUPPORTED_FORMATS:
            return SUPPORTED_FORMATS[suffix]

        # If directory, check contents using filesystem
        try:
            if filesystem and filesystem.isdir(path):
                files = filesystem.ls(path, detail=False)
                for f in files:
                    f_suffix = Path(f).suffix.lower()
                    if ".parquet" in f_suffix:
                        return "parquet"
                    if ".csv" in f_suffix:
                        return "csv"
                    if ".json" in f_suffix:
                        return "json"
                    if ".arrow" in f_suffix:
                        return "arrow"
        except Exception:
            pass

        return "parquet"  # Default

    @staticmethod
    def load_parquet(
        path: str,
        filesystem: AbstractFileSystem | None = None,
        *,
        lazy: bool = False,
    ) -> pa.Table | ds.Dataset:
        """Load parquet dataset."""
        # PyArrow handles fsspec filesystems natively
        dataset = ds.dataset(path, format="parquet", filesystem=filesystem)
        if lazy:
            return dataset
        return dataset.to_table()

    @staticmethod
    def load_csv(
        path: str,
        filesystem: AbstractFileSystem | None = None,
        *,
        lazy: bool = False,
        **read_options: Any,
    ) -> pa.Table | ds.Dataset:
        """Load CSV dataset."""
        if lazy:
            return ds.dataset(path, format="csv", filesystem=filesystem)

        dataset = ds.dataset(path, format="csv", filesystem=filesystem)
        return dataset.to_table()

    @staticmethod
    def load_json(
        path: str,
        filesystem: AbstractFileSystem | None = None,
        *,
        lazy: bool = False,
        **read_options: Any,
    ) -> pa.Table | ds.Dataset:
        """Load JSON/JSONL dataset."""
        if lazy:
            return ds.dataset(path, format="json", filesystem=filesystem)

        dataset = ds.dataset(path, format="json", filesystem=filesystem)
        return dataset.to_table()

    @staticmethod
    def load_arrow(
        path: str,
        filesystem: AbstractFileSystem | None = None,
        *,
        lazy: bool = False,
    ) -> pa.Table | ds.Dataset:
        """Load Arrow IPC dataset."""
        if lazy:
            return ds.dataset(path, format="ipc", filesystem=filesystem)

        # For Arrow IPC, ds.dataset is robust
        dataset = ds.dataset(path, format="ipc", filesystem=filesystem)
        return dataset.to_table()

    @classmethod
    def load(
        cls,
        path: str,
        format: str | None = None,
        filesystem: AbstractFileSystem | None = None,
        *,
        lazy: bool = False,
        **kwargs: Any,
    ) -> pa.Table | ds.Dataset:
        """Load dataset with automatic FS resolution."""

        # Resolve filesystem and clean path if not provided
        if filesystem is None:
            filesystem, path = get_filesystem(path)

        if format is None:
            format = cls.detect_format(path, filesystem)

        logging_console.print(f"Loading dataset from {path} with format {format}")

        loaders = {
            "parquet": cls.load_parquet,
            "csv": cls.load_csv,
            "json": cls.load_json,
            "arrow": cls.load_arrow,
        }

        loader = loaders.get(format)
        if not loader:
            raise ValueError(f"Unsupported format: {format}")

        return loader(path, filesystem=filesystem, lazy=lazy, **kwargs)


class DatasetWriter:
    """Handles writing datasets to various formats."""

    @staticmethod
    def get_written_files(
        base_path: str,
        filesystem: AbstractFileSystem | None = None,
        pattern: str = "*.parquet",
    ) -> list[str]:
        """Get list of files written by dataset operations."""
        if filesystem is None:
            # Fallback for local if no FS passed (though get_filesystem usually handles this)
            filesystem, base_path = get_filesystem(base_path)

        try:
            # Normalize base path
            base_path = base_path.rstrip("/")

            all_files = filesystem.find(base_path, detail=False)

            import fnmatch

            matched_files = [f for f in all_files if fnmatch.fnmatch(Path(f).name, pattern)]

            # Return relative paths
            results = []
            for f in matched_files:
                if f.startswith(base_path):
                    # +1 for the slash
                    rel = f[len(base_path) :].lstrip("/")
                    if rel:
                        results.append(rel)
                else:
                    results.append(Path(f).name)
            return results

        except Exception as e:
            logging_console.print(f"Failed to glob with filesystem: {e}")
            return []

    @staticmethod
    def write_parquet(
        data: pa.Table | ds.Dataset,
        path: str,
        filesystem: AbstractFileSystem | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Write dataset as parquet."""
        ds.write_dataset(
            data=data,
            base_dir=path,
            format="parquet",
            filesystem=filesystem,
            existing_data_behavior="overwrite_or_ignore",
            **kwargs,
        )

        return DatasetWriter.get_written_files(path, filesystem, "*.parquet")

    @staticmethod
    def write_csv(
        data: pa.Table,
        path: str,
        filesystem: AbstractFileSystem | None = None,
        max_rows_per_file: int | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Write CSV with efficient streaming."""
        if filesystem is None:
            filesystem, path = get_filesystem(path)

        filesystem.makedirs(path, exist_ok=True)
        written_files = []

        def write_chunk(chunk: pa.Table, filename: str):
            full_path = f"{path}/{filename}"
            # Open remote file stream directly
            with filesystem.open(full_path, "wb") as f:
                csv.write_csv(chunk, f, **kwargs)
            written_files.append(filename)

        if max_rows_per_file is None or len(data) <= max_rows_per_file:
            write_chunk(data, "data.csv")
        else:
            num_files = (len(data) + max_rows_per_file - 1) // max_rows_per_file
            for i in range(num_files):
                start = i * max_rows_per_file
                end = min((i + 1) * max_rows_per_file, len(data))
                write_chunk(data.slice(start, end - start), f"part-{i:05d}.csv")

        return written_files

    @staticmethod
    def write_json(
        data: pa.Table,
        path: str,
        filesystem: AbstractFileSystem | None = None,
        max_rows_per_file: int | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Write JSONL with efficient streaming."""
        if filesystem is None:
            filesystem, path = get_filesystem(path)

        filesystem.makedirs(path, exist_ok=True)
        written_files = []

        def write_chunk(chunk: pa.Table, filename: str):
            full_path = f"{path}/{filename}"
            with filesystem.open(full_path, "w") as f:
                _write_jsonl(chunk, f)
            written_files.append(filename)

        if max_rows_per_file is None or len(data) <= max_rows_per_file:
            write_chunk(data, "data.jsonl")
        else:
            num_files = (len(data) + max_rows_per_file - 1) // max_rows_per_file
            for i in range(num_files):
                start = i * max_rows_per_file
                end = min((i + 1) * max_rows_per_file, len(data))
                write_chunk(data.slice(start, end - start), f"part-{i:05d}.jsonl")

        return written_files

    @classmethod
    def write(
        cls,
        data: pa.Table | ds.Dataset,
        path: str,
        format: str = "parquet",
        filesystem: AbstractFileSystem | None = None,
        max_rows_per_file: int | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Generic write entry point."""
        if filesystem is None:
            filesystem, path = get_filesystem(path)

        if isinstance(data, ds.Dataset):
            data = data.to_table()

        writers = {
            "parquet": cls.write_parquet,
            "csv": cls.write_csv,
            "json": cls.write_json,
        }

        writer = writers.get(format)
        if not writer:
            raise ValueError(f"Unsupported format: {format}")

        return writer(
            data,
            path,
            filesystem=filesystem,
            max_rows_per_file=max_rows_per_file,
            **kwargs,
        )


def _write_jsonl(table: pa.Table, file_handle: Any) -> None:
    import json

    # Iterate over batches to reduce memory usage vs to_pylist() on whole table
    for batch in table.to_batches():
        for row in batch.to_pylist():
            file_handle.write(json.dumps(row) + "\n")

# formats.py
"""Format handlers for different dataset types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pyarrow as pa
import pyarrow.json as pajson
import pyarrow.parquet as pq
from pyarrow import csv

from .types import CompressionType, DatasetFormat


class FormatHandler(ABC):
    """Base class for format handlers."""

    @abstractmethod
    def read(self, path: Path) -> pa.Table:
        """Read data from path into PyArrow Table."""
        ...

    @abstractmethod
    def write(
        self,
        table: pa.Table,
        path: Path,
        compression: CompressionType,
    ) -> None:
        """Write PyArrow Table to path."""
        ...

    @abstractmethod
    def get_extension(self) -> str:
        """Get file extension for this format."""
        ...


class ParquetHandler(FormatHandler):
    """Handler for Parquet format."""

    def read(self, path: Path) -> pa.Table:
        """Read Parquet file."""
        return pq.read_table(path, use_threads=True, memory_map=True)

    def write(
        self,
        table: pa.Table,
        path: Path,
        compression: CompressionType,
    ) -> None:
        """Write Parquet file with optimal settings."""
        pq.write_table(
            table,
            path,
            compression=compression.value if compression != CompressionType.NONE else None,
            use_dictionary=True,
            write_statistics=True,
            version="2.6",
            data_page_size=1024 * 1024,  # 1MB pages
            row_group_size=1024 * 1024,  # 1M rows per group
            use_compliant_nested_type=True,
        )

    def get_extension(self) -> str:
        """Get file extension."""
        return ".parquet"


class CSVHandler(FormatHandler):
    """Handler for CSV format."""

    def read(self, path: Path) -> pa.Table:
        """Read CSV file."""
        return csv.read_csv(path)

    def write(
        self,
        table: pa.Table,
        path: Path,
        compression: CompressionType,
    ) -> None:
        """Write CSV file."""
        # PyArrow CSV writer doesn't support compression directly
        # Convert to pandas for more control
        df = table.to_pandas()

        compression_map = {
            CompressionType.NONE: None,
            CompressionType.GZIP: "gzip",
            CompressionType.BROTLI: "bz2",
            CompressionType.ZSTD: "zstd",
        }

        df.to_csv(
            path,
            index=False,
            compression=compression_map.get(compression),
        )

    def get_extension(self) -> str:
        """Get file extension."""
        return ".csv"


class JSONHandler(FormatHandler):
    """Handler for JSON format."""

    def read(self, path: Path) -> pa.Table:
        """Read JSON file."""
        with open(path) as f:
            return pajson.read_json(f)

    def write(
        self,
        table: pa.Table,
        path: Path,
        compression: CompressionType,
    ) -> None:
        """Write JSON file."""
        df = table.to_pandas()

        compression_map = {
            CompressionType.NONE: None,
            CompressionType.GZIP: "gzip",
            CompressionType.BROTLI: "bz2",
        }

        df.to_json(
            path,
            orient="records",
            lines=False,
            compression=compression_map.get(compression),
        )

    def get_extension(self) -> str:
        """Get file extension."""
        return ".json"


class JSONLHandler(FormatHandler):
    """Handler for JSONL (newline-delimited JSON) format."""

    def read(self, path: Path) -> pa.Table:
        """Read JSONL file."""
        with open(path) as f:
            return pajson.read_json(f)

    def write(
        self,
        table: pa.Table,
        path: Path,
        compression: CompressionType,
    ) -> None:
        """Write JSONL file."""
        df = table.to_pandas()

        compression_map = {
            CompressionType.NONE: None,
            CompressionType.GZIP: "gzip",
            CompressionType.BROTLI: "bz2",
        }

        df.to_json(
            path,
            orient="records",
            lines=True,
            compression=compression_map.get(compression),
        )

    def get_extension(self) -> str:
        """Get file extension."""
        return ".jsonl"


class ArrowIPCHandler(FormatHandler):
    """Handler for Arrow IPC (Feather v2) format."""

    def read(self, path: Path) -> pa.Table:
        """Read Arrow IPC file."""
        with pa.memory_map(str(path), "r") as source:
            return pa.ipc.open_file(source).read_all()

    def write(
        self,
        table: pa.Table,
        path: Path,
        compression: CompressionType,
    ) -> None:
        """Write Arrow IPC file."""
        compression_map = {
            CompressionType.NONE: None,
            CompressionType.ZSTD: "zstd",
            CompressionType.LZ4: "lz4",
        }

        with (
            pa.OSFile(str(path), "wb") as sink,
            pa.ipc.new_file(
                sink,
                table.schema,
                options=pa.ipc.IpcWriteOptions(compression=compression_map.get(compression)),
            ) as writer,
        ):
            writer.write_table(table)

    def get_extension(self) -> str:
        """Get file extension."""
        return ".arrow"


class FormatRegistry:
    """Registry for format handlers."""

    _handlers: dict[DatasetFormat, FormatHandler] = {
        DatasetFormat.PARQUET: ParquetHandler(),
        DatasetFormat.CSV: CSVHandler(),
        DatasetFormat.JSON: JSONHandler(),
        DatasetFormat.JSONL: JSONLHandler(),
        DatasetFormat.ARROW: ArrowIPCHandler(),
    }

    @classmethod
    def get_handler(cls, format: DatasetFormat) -> FormatHandler:
        """Get handler for format."""
        return cls._handlers[format]

    @classmethod
    def detect_format(cls, path: Path) -> DatasetFormat:
        """Detect format from file extension."""
        suffix = path.suffix.lower()
        format_map = {
            ".parquet": DatasetFormat.PARQUET,
            ".pq": DatasetFormat.PARQUET,
            ".csv": DatasetFormat.CSV,
            ".json": DatasetFormat.JSON,
            ".jsonl": DatasetFormat.JSONL,
            ".ndjson": DatasetFormat.JSONL,
            ".arrow": DatasetFormat.ARROW,
            ".feather": DatasetFormat.ARROW,
        }

        if suffix in format_map:
            return format_map[suffix]

        raise ValueError(f"Cannot detect format from extension: {suffix}")

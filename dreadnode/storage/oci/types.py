# types.py
"""Type definitions for the dataset storage system."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict

from pydantic import BaseModel, Field, field_validator


class DatasetFormat(str, Enum):
    """Supported dataset formats."""

    PARQUET = "parquet"
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    TEXT = "text"
    ARROW = "arrow"  # Apache Arrow IPC format


class CompressionType(str, Enum):
    """Supported compression types."""

    NONE = "none"
    SNAPPY = "snappy"
    GZIP = "gzip"
    BROTLI = "brotli"
    ZSTD = "zstd"
    LZ4 = "lz4"


class StorageStrategy(str, Enum):
    """Storage strategy for datasets."""

    FILE = "file"  # Store as individual files
    VOLUME = "volume"  # Pack into a volume artifact


class DatasetMetadata(BaseModel):
    """Metadata for a dataset."""

    name: str
    org: str
    version: str
    format: DatasetFormat
    compression: CompressionType
    storage_strategy: StorageStrategy

    # Statistics
    size_bytes: int
    row_count: int | None = None
    column_count: int | None = None
    checksum: str  # xxhash64 of content

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Provenance
    source_path: str | None = None
    parent_version: str | None = None
    description: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)

    # Schema information (for structured formats)
    schema_json: str | None = None

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Ensure version follows semver format."""
        parts = v.split(".")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError(f"Version must be in semver format (x.y.z), got: {v}")
        return v


class PushResult(BaseModel):
    """Result of a push operation."""

    success: bool
    target: str
    digest: str | None = None
    size_bytes: int
    metadata: DatasetMetadata
    error: str | None = None


class PullResult(BaseModel):
    """Result of a pull operation."""

    success: bool
    source: str
    local_path: Path
    metadata: DatasetMetadata
    cached: bool  # Whether it was already cached
    error: str | None = None


class CacheManifest(TypedDict):
    """Cache manifest structure."""

    version: str
    datasets: dict[str, dict[str, Any]]  # org/dataset/version -> metadata
    last_updated: str  # ISO timestamp

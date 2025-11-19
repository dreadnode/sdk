"""
Dataset metadata models and management.
"""

import contextlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fsspec import AbstractFileSystem
from pydantic import BaseModel, Field, field_validator

from dreadnode.storage.datasets.core import get_filesystem


class DatasetSchema(BaseModel):
    fields: list[dict[str, str]] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"fields": self.fields}

    @classmethod
    def from_arrow_schema(cls, arrow_schema: Any) -> "DatasetSchema":
        fields = []
        for name, dtype in zip(arrow_schema.names, arrow_schema.types):
            fields.append({"name": name, "type": str(dtype)})
        return cls(fields=fields)


class DatasetMetadata(BaseModel):
    name: str
    description: str | None = None
    version: str = "0.0.1"
    license: str | None = None
    tags: list[str] = Field(default_factory=list)
    uri: str
    ds_schema: DatasetSchema | None = None
    files: list[str] = Field(default_factory=list)
    format: str = "parquet"
    size_bytes: int | None = None
    row_count: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    checksum: str | None = None
    author: str | None = None
    source: str | None = None
    custom_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tags", mode="before")
    @classmethod
    def validate_tags(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        if self.ds_schema:
            data["ds_schema"] = self.ds_schema.to_dict()
        return data

    def save(self, path: str | Path, fs: AbstractFileSystem | None = None) -> None:
        import json

        if fs is None:
            fs, path = get_filesystem(str(path))

        with fs.open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path, fs: AbstractFileSystem | None = None) -> "DatasetMetadata":
        if fs is None:
            fs, path = get_filesystem(str(path))
        with fs.open(path, "r") as f:
            data = json.load(f)
        if data.get("ds_schema"):
            data["ds_schema"] = DatasetSchema(**data["ds_schema"])
        return cls(**data)

    def update_from_data(self, ds: Any, files: list[str]) -> None:
        self.files = files
        self.updated_at = datetime.now(timezone.utc)
        if hasattr(ds, "schema"):
            self.ds_schema = DatasetSchema.from_arrow_schema(ds.schema)
        if hasattr(ds, "num_rows"):
            self.row_count = ds.num_rows
        elif hasattr(ds, "__len__"):
            with contextlib.suppress(TypeError, AttributeError):
                self.row_count = len(ds)

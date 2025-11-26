"""
Dataset metadata models and management (Native PyArrow Version).
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import coolname
import pyarrow as pa
import pyarrow.dataset as ds
from pyarrow.fs import FileSystem
from pydantic import BaseModel, Field, field_validator


class VersionInfo(BaseModel):
    major: int
    minor: int
    patch: int

    @field_validator("major", "minor", "patch", mode="before")
    @classmethod
    def validate_non_negative(cls, v: Any) -> int:
        iv = int(v)
        if iv < 0:
            raise ValueError("Version numbers must be non-negative integers")
        return iv

    def to_string(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def increment_major(self) -> None:
        self.major += 1
        self.minor = 0
        self.patch = 0

    def increment_minor(self) -> None:
        self.minor += 1
        self.patch = 0

    def increment_patch(self) -> None:
        self.patch += 1

    @staticmethod
    def from_string(version_str: str) -> "VersionInfo":
        parts = version_str.split(".")
        if len(parts) != 3:
            raise ValueError("Version string must be in the format 'major.minor.patch'")
        return VersionInfo(
            major=int(parts[0]),
            minor=int(parts[1]),
            patch=int(parts[2]),
        )


class DatasetMetadata(BaseModel):
    name: str | None = Field(default_factory=lambda: coolname.generate_slug(2))
    version: VersionInfo = VersionInfo(major=0, minor=0, patch=1)
    license: str | None = None
    organization: str | None = None
    tags: list[str] = Field(default_factory=list)
    uri: str | None = None
    readme: str | None = None

    # Dataset metadata
    format: str | None = None
    ds_schema: dict | None = None
    files: list[str] = Field(default_factory=list)
    size_bytes: int | None = None
    row_count: int | None = None

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Custom
    custom_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tags", mode="before")
    @classmethod
    def validate_tags(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)

    @classmethod
    def load(cls, path: str, fs: FileSystem) -> "DatasetMetadata":
        """
        Load metadata using Native PyArrow streams.
        """
        # Ensure we are reading the specific file
        if not path.endswith(".json"):
            path = f"{path}/metadata.json"

        with fs.open_input_stream(path) as f:
            data = json.load(f)

        return cls.model_validate(data)

    def save(self, *, path: str, fs: FileSystem) -> None:
        """
        Save metadata using Native PyArrow streams.
        """
        if not path.endswith(".json"):
            path = f"{path}/metadata.json"

        json_bytes = self.model_dump_json(indent=2).encode("utf-8")

        with fs.open_output_stream(path) as f:
            f.write(json_bytes)

    def set_uri(self, uri: str) -> None:
        self.uri = uri

    def set_ds_schema(self, ds_obj: ds.Dataset | pa.Table) -> None:
        """
        Set the dataset schema in metadata.
        Handles both on-disk Dataset and in-memory Table.
        """
        schema = ds_obj.schema

        if schema.metadata:
            pass

        ds_schema = {}
        for field in schema:
            ds_schema[field.name] = str(field.type)
        self.ds_schema = ds_schema

    def set_ds_files(self, ds_obj: ds.Dataset | pa.Table) -> None:
        """
        Set the list of files. IMPORTANT: Relativizes paths.
        """
        # If it's a Table (in-memory), it has no files.
        if isinstance(ds_obj, pa.Table):
            self.files = []
            return

        # If it's a Dataset (on-disk)
        if hasattr(ds_obj, "files"):
            # PyArrow returns absolute URIs (s3://bucket/path/to/data/file.parquet)
            # We want to store only "data/file.parquet" to make the metadata portable.
            clean_files = []
            for f in ds_obj.files:
                # 1. Normalize slashes
                f_str = str(f).replace("\\", "/")

                # 2. Extract just the filename (assuming flat structure inside /data)
                # Or if you have partitions, you might need smarter logic.
                # For now, we assume standard "data/" folder structure.
                filename = f_str.split("/")[-1]
                clean_files.append(f"data/{filename}")

            self.files = clean_files

    def set_size_and_row_count(self, ds_obj: ds.Dataset | pa.Table) -> None:
        """
        Safely extracts size and row count.
        """
        # 1. Handle in-memory Table (Fast and Easy)
        if isinstance(ds_obj, pa.Table):
            self.size_bytes = ds_obj.nbytes
            self.row_count = ds_obj.num_rows
            return

        # 2. Handle on-disk Dataset
        # Note: PyArrow Datasets do NOT have a simple size_bytes attribute
        # We would have to sum file sizes, but we leave that to the Manifest class.
        self.size_bytes = 0

        # Counting rows on S3 can be slow (it requires reading footers)
        # We try to do it via scanner if not already computed
        try:
            self.row_count = ds_obj.count_rows()
        except Exception:
            self.row_count = 0

    def set_name(self, name: str) -> None:
        self.name = name

    def set_format(self, format: str) -> None:
        self.format = format

    def set_version(self, version: VersionInfo | str) -> None:
        if isinstance(version, str):
            self.version = VersionInfo.from_string(version_str=version)
        else:
            self.version = version

    def update_version(self, part: str) -> None:
        if part == "major":
            self.version.increment_major()
        elif part == "minor":
            self.version.increment_minor()
        elif part == "patch":
            self.version.increment_patch()
        else:
            raise ValueError("part must be 'major', 'minor', or 'patch'")

    def set_license(self, license_content: str | Path) -> None:
        """
        Accepts raw string content OR a local Path object to read from.
        """
        if isinstance(license_content, Path):
            self.license = license_content.read_text(encoding="utf-8")
        else:
            self.license = license_content

    def set_organization(self, organization: str) -> None:
        self.organization = organization

    def add_tags(self, tags: list[str] | str) -> None:
        if isinstance(tags, str):
            tags = [tags]
        self.tags.extend(tags)
        self.tags = list(set(self.tags))  # Ensure uniqueness

    def set_readme(self, readme_content: str | Path) -> None:
        """
        Accepts raw string content OR a local Path object to read from.
        """
        if isinstance(readme_content, Path):
            self.readme = readme_content.read_text(encoding="utf-8")
        else:
            self.readme = readme_content

    def set_created_at(self) -> None:
        self.created_at = datetime.now(timezone.utc)

    def set_updated_at(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

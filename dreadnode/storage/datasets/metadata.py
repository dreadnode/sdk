"""
Dataset metadata models and management.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import coolname
from fsspec import AbstractFileSystem
from pyarrow import dataset
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
    name: str | None = Field(default=coolname.generate_slug(2))
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

    # Timestamps and checksum
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    checksum: str | None = None

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
    def load(cls, path: str, filesystem: AbstractFileSystem) -> "DatasetMetadata":
        with filesystem.open(path, "r") as f:
            content = f.read()

        return cls.model_validate_json(content)

    def save(self, *, path: str, fs: AbstractFileSystem) -> None:
        with fs.open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    def set_uri(self, uri: str) -> None:
        self.uri = uri

    def set_ds_schema(self, ds: dataset.Dataset) -> None:
        """Set the dataset schema in metadata."""
        if not ds.schema.metadata:
            ds_schema = {}
            for field in ds.schema:
                ds_schema[field.name] = str(field.type)
            self.ds_schema = ds_schema
        else:
            self.ds_schema = ds.schema.metadata

    def update_files(self) -> None:
        """
        Update the list of files in metadata for being managed.
        Will remove existing path and prepend 'data/' to each file.
        """

        files = []
        for file in self.files:
            file_path = Path("data") / Path(file).name
            files.append(str(file_path))
        self.files = files

    def set_ds_files(self, ds: dataset.Dataset) -> None:
        """Set the list of files in metadata."""
        if hasattr(ds, "files"):
            self.files = ds.files

    def set_size_and_row_count(self, ds: dataset.Dataset) -> None:
        if hasattr(ds, "size_bytes"):
            self.size_bytes = ds.size_bytes
        if hasattr(ds, "num_rows"):
            self.row_count = ds.num_rows

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

    def set_license(self, license: str | Path) -> None:
        self.license = license.read_text() if isinstance(license, Path) else license

    def set_organization(self, organization: str) -> None:
        self.organization = organization

    def add_tags(self, tags: list[str] | str) -> None:
        if isinstance(tags, str):
            tags = [tags]
        self.tags.extend(tags)
        self.tags = list(set(self.tags))  # Ensure uniqueness

    def set_readme(self, readme: str | Path) -> None:
        self.readme = readme.read_text() if isinstance(readme, Path) else readme

    def set_created_at(self) -> None:
        """Set metadata fields based on the dataset object."""
        self.created_at = datetime.now(timezone.utc)

    def set_updated_at(self) -> None:
        """Set metadata fields based on the dataset object."""
        self.updated_at = datetime.now(timezone.utc)

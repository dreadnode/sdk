import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import coolname
from packaging.version import Version
from pyarrow.fs import FileSystem
from pydantic import BaseModel, Field, field_validator

from dreadnode.common_types import VersionStrategy


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
        version = Version(version_str)
        parts = [version.major, version.minor, version.micro]
        if len(parts) != 3:
            raise ValueError("Version string must be in the format 'major.minor.patch'")
        return VersionInfo(
            major=int(parts[0]),
            minor=int(parts[1]),
            patch=int(parts[2]),
        )

    def __gt__(self, other: "VersionInfo") -> bool:
        if self.major != other.major:
            return self.major > other.major
        if self.minor != other.minor:
            return self.minor > other.minor
        return self.patch > other.patch


class DatasetMetadata(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    organization: str | None = None
    uri: str | None = None
    name: str | None = Field(default=coolname.generate_slug(2))
    version: str = Field(default=VersionInfo(major=0, minor=1, patch=0).to_string())
    license: str = Field(default="This dataset is not licensed.")
    tags: list[str] = Field(default_factory=list)
    readme: str = Field(default="# Dataset README\n\n")
    format: str | None = None
    ds_schema: dict | None = None
    files: list[str] = Field(default_factory=list)
    size_bytes: int | None = None
    row_count: int | None = None
    custom_metadata: dict[str, Any] = Field(default_factory=dict)
    auto_version: bool = Field(default=True)
    auto_version_strategy: VersionStrategy | None = Field(default="patch")
    fingerprint: str | None = None

    created_at: str = Field(default=datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default=datetime.now(timezone.utc).isoformat())

    class Config:
        arbitrary_types_allowed = True

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
        Helper to read JSON from native binary streams.
        """
        with fs.open_input_stream(path) as f:
            data = json.load(f)

        return cls.model_validate(data)

    def save(self, path: str, fs: FileSystem) -> None:
        """
        Helper to write Pydantic models to native binary streams.
        """
        json_bytes = self.model_dump_json(indent=2).encode("utf-8")
        with fs.open_output_stream(path) as f:
            f.write(json_bytes)

    def set_name(self, name: str) -> None:
        self.name = name

    def set_format(self, format: str) -> None:
        self.format = format

    def set_version(self, version: VersionInfo) -> None:
        self.version = version.to_string()

    def update_version(self, version: str) -> None:
        version_info = VersionInfo.from_string(version)
        if self.auto_version_strategy == "major":
            version_info.increment_major()
        elif self.auto_version_strategy == "minor":
            version_info.increment_minor()
        elif self.auto_version_strategy == "patch":
            version_info.increment_patch()
        else:
            raise ValueError("part must be 'major', 'minor', or 'patch'")
        self.version = version_info.to_string()

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
        self.created_at = datetime.now(timezone.utc).isoformat()

    def set_updated_at(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def set_uri(self, uri: str) -> None:
        self.uri = uri

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import coolname
from packaging.version import Version
from pyarrow.fs import FileSystem
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    PlainSerializer,
    WithJsonSchema,
    field_validator,
    model_validator,
)

from dreadnode.common_types import VersionStrategy
from dreadnode.util import parse_version

DatasetVersion = Annotated[
    Version,
    BeforeValidator(parse_version),
    PlainSerializer(lambda v: str(v), return_type=str),
    WithJsonSchema({"type": "string"}),
]


class DatasetMetadata(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    organization: str | None = None
    name: str = Field(default=coolname.generate_slug(2))
    uri: str | None = None
    version: DatasetVersion = Field(default=Version("0.1.0"))
    license: str = Field(default="This dataset is not licensed.")
    tags: list[str] = Field(default_factory=list)
    readme: str = Field(default="# Dataset README\n\n")
    format: str | None = None
    ds_schema: dict[str, Any] | None = None
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

    @model_validator(mode="before")
    @classmethod
    def validate_name_and_organization(cls, data: Any) -> Any:
        # if org is not set, name should be of form org/name
        if (
            data.get("organization") is None
            and data.get("name") is not None
            and "/" in data.get("name")
        ):
            org, name = data["name"].split("/", 1)
            data["organization"] = org
            data["name"] = name
        elif (
            data.get("organization") is None
            and data.get("name") is not None
            and "/" not in data.get("name")
        ):
            raise ValueError("organization must be set if name does not contain '/'")
        elif (
            data.get("organization") is not None
            and data.get("name") is not None
            and "/" in data.get("name")
        ):
            raise ValueError("name must not contain '/' if organization is set")
        else:
            pass  # both are set, all good

        return data

    @classmethod
    def load(cls, path: str, fs: FileSystem) -> "DatasetMetadata":
        """
        Helper to read JSON from native binary streams.
        """
        with fs.open_input_stream(path) as f:
            data = json.load(f)

        return cls(**data)

    @staticmethod
    def bump_patch_version(version: DatasetVersion) -> DatasetVersion:
        return Version(f"{version.major}.{version.minor}.{version.micro + 1}")

    @property
    def ref(self) -> str:
        return f"{self.organization}/{self.name}"

    @property
    def versioned_ref(self) -> str:
        return f"{self.organization}/{self.name}/{self.version}"

    def save(self, path: str, fs: FileSystem) -> None:
        """
        Helper to write Pydantic models to native binary streams.
        """
        json_bytes = self.model_dump_json(indent=2).encode("utf-8")
        with fs.open_output_stream(path) as f:
            f.write(json_bytes)

    def set_license(self, license_content: str | Path) -> None:
        """
        Accepts raw string content OR a local Path object to read from.
        """
        if isinstance(license_content, Path):
            self.license = license_content.read_text(encoding="utf-8")
        else:
            self.license = license_content

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

from pydantic import BaseModel, Field


class BaseManifest(BaseModel):
    """Base manifest with artifact tracking."""

    artifacts: dict[str, str] = Field(default_factory=dict)  # path -> oid
    artifacts_hash: str | None = None


class DatasetManifest(BaseManifest):
    """Dataset-specific manifest."""

    format: str = "parquet"
    data_schema: dict[str, str] = Field(default_factory=dict)
    row_count: int | None = None
    splits: dict[str, str] | None = None  # split_name -> artifact_path


class ModelManifest(BaseManifest):
    """Model-specific manifest."""

    framework: str = "safetensors"
    task: str | None = None
    architecture: str | None = None


class ToolsetManifest(BaseManifest):
    """Toolset manifest. Tools are discovered via @tool decorator."""


class AgentManifest(BaseManifest):
    """Agent-specific manifest."""

    entrypoint: str = "main:run"
    toolsets: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)


class EnvironmentManifest(BaseManifest):
    """Environment-specific manifest."""


# Mapping from singular type name to manifest class
MANIFEST_TYPES = {
    "dataset": DatasetManifest,
    "model": ModelManifest,
    "toolset": ToolsetManifest,
    "agent": AgentManifest,
    "environment": EnvironmentManifest,
}

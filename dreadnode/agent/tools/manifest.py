import platform
import sys
from typing import Any, Literal

from packaging import version
from packaging.markers import Marker
from packaging.specifiers import SpecifierSet
from pydantic import BaseModel, ConfigDict, Field, field_validator


class EntryPoint(BaseModel):
    module: str
    qualname: str | None = None
    factory: str | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)


class Capabilities(BaseModel):
    consumes: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    variants: list[str] = Field(default_factory=list)


class PythonReq(BaseModel):
    min: str | None = None
    packages: list[str] = Field(default_factory=list)


class SystemReq(BaseModel):
    apt: list[str] = Field(default_factory=list)
    brew: list[str] = Field(default_factory=list)
    choco: list[str] = Field(default_factory=list)


class BinaryReq(BaseModel):
    name: str
    optional: bool = False


class Requirements(BaseModel):
    python: PythonReq = Field(default_factory=PythonReq)
    system: SystemReq = Field(default_factory=SystemReq)
    binaries: list[BinaryReq] = Field(default_factory=list)


class InstallSpec(BaseModel):
    strategy: Literal["inproc", "uv-venv", "subprocess", "ray-actor"] = "inproc"
    venv_name: str | None = None
    preinstall: list[str] = Field(default_factory=list)
    postinstall: list[str] = Field(default_factory=list)


class Compatibility(BaseModel):
    requires_core: str | None = None
    platforms: list[str] = Field(default_factory=list)
    markers: str | None = None  # PEP 508 string

    def satisfied(self, *, core_version: str) -> bool:
        if self.requires_core:
            if version.parse(core_version) not in SpecifierSet(self.requires_core):
                return False
        if self.platforms:
            cur = sys.platform
            plat = platform.system().lower()
            if cur.startswith("linux"):
                cur_norm = "linux"
            elif cur.startswith("win"):
                cur_norm = "windows"
            elif cur.startswith("darwin"):
                cur_norm = "darwin"
            else:
                cur_norm = plat or cur
            if cur_norm not in {p.lower() for p in self.platforms}:
                return False
        if self.markers:
            try:
                if not Marker(self.markers).evaluate():
                    return False
            except Exception:
                return False
        return True


class Permissions(BaseModel):
    network: bool = True
    filesystem: list[str] = Field(default_factory=list)
    subprocess: list[str] = Field(default_factory=list)


class ConfigSchema(BaseModel):
    # Minimal JSON‑Schema subset
    properties: dict[str, dict[str, Any]] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)

    def defaults(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, spec in self.properties.items():
            if "default" in spec:
                out[k] = spec["default"]
        return out


class ToolManifest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    manifest_version: Literal[1] = 1
    id: str
    name: str
    version: str
    description: str | None = None
    author: str | None = None
    license: str | None = None

    entrypoint: EntryPoint
    capabilities: Capabilities = Field(default_factory=Capabilities)
    config_schema: ConfigSchema = Field(default_factory=ConfigSchema)
    requirements: Requirements = Field(default_factory=Requirements)
    install: InstallSpec = Field(default_factory=InstallSpec)
    compatibility: Compatibility = Field(default_factory=Compatibility)
    permissions: Permissions = Field(default_factory=Permissions)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _id_nonempty(cls, v: str) -> str:
        if not v or ":" in v or " " in v:
            raise ValueError("id must be a simple, colon/space‑free string")
        return v

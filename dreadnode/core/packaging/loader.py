"""Base loader for dreadnode components."""

from __future__ import annotations

import hashlib
import json
from functools import cached_property
from importlib.metadata import entry_points, metadata
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from dreadnode.core.storage.storage import Storage

if TYPE_CHECKING:
    from dreadnode.core.packaging.manifest import BaseManifest


class BaseLoader:
    """Base class for loading dreadnode components."""

    entry_point_group: ClassVar[str]
    manifest_class: ClassVar[type[BaseManifest]]

    def __init__(self, name: str):
        """Load a component by entry point name."""
        self._name = name
        self._ep = self._find_entry_point(name)
        self._module = self._ep.load()
        self._pkg_meta = metadata(self._module.__name__)

    def _find_entry_point(self, name: str):
        for ep in entry_points(group=self.entry_point_group):
            if ep.name == name:
                return ep
        raise KeyError(f"Component not found: {name} in {self.entry_point_group}")

    @classmethod
    def list(cls) -> list[str]:
        """List all available component names."""
        return [ep.name for ep in entry_points(group=cls.entry_point_group)]

    @classmethod
    def discover(cls) -> dict[str, BaseLoader]:
        """Discover all components of this type."""
        return {name: cls(name) for name in cls.list()}

    @property
    def name(self) -> str:
        return self._pkg_meta["Name"]

    @property
    def version(self) -> str:
        return self._pkg_meta["Version"]

    @property
    def description(self) -> str | None:
        return self._pkg_meta.get("Summary")

    # Manifest

    @cached_property
    def manifest(self) -> BaseManifest:
        """Load manifest.json from the package."""
        pkg_files = files(self._module.__name__)
        content = pkg_files.joinpath("manifest.json").read_text()
        return self.manifest_class.model_validate_json(content)

    @property
    def files(self) -> list[str]:
        """List of artifact files."""
        return list(self.manifest.artifacts.keys())

    @property
    def artifacts_hash(self) -> str | None:
        return self.manifest.artifacts_hash

    def verify(self) -> bool:
        """Verify manifest hash."""
        if not self.artifacts_hash:
            return True

        artifacts_json = json.dumps(self.manifest.artifacts, sort_keys=True)
        expected = f"sha256:{hashlib.sha256(artifacts_json.encode()).hexdigest()}"
        return self.artifacts_hash == expected

    # Artifact resolution

    def resolve(self, path: str) -> Path:
        """Resolve an artifact path to a local file."""
        if path not in self.manifest.artifacts:
            raise FileNotFoundError(f"Artifact not in manifest: {path}")

        oid = self.manifest.artifacts[path]
        return self._fetch_blob(oid)

    def _fetch_blob(self, oid: str) -> Path:
        """Fetch a blob from CAS."""
        # Get org from package name (e.g., "acme-sentiment-data" -> "acme")
        org = self.name.split("-")[0]
        storage = Storage(org)

        if storage.blob_exists(oid):
            return storage.get_blob(oid)

        # TODO: download from remote if not local
        raise FileNotFoundError(f"Blob not found locally: {oid}")

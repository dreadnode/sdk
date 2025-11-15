# cache.py
"""Local cache management for datasets."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .types import CacheManifest, DatasetMetadata


class CacheManager:
    """Manages local dataset cache."""

    def __init__(self, cache_root: Path = Path.home() / ".dreadnode" / "datasets") -> None:
        """
        Initialize cache manager.

        Args:
            cache_root: Root directory for cache
        """
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = cache_root / "manifest.json"
        self._manifest: CacheManifest | None = None

    @property
    def manifest(self) -> CacheManifest:
        """Load or return cached manifest."""
        if self._manifest is None:
            self._manifest = self._load_manifest()
        return self._manifest

    def _load_manifest(self) -> CacheManifest:
        """Load manifest from disk."""
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)  # type: ignore[return-value]
        return {"version": "1.0.0", "datasets": {}}

    def _save_manifest(self) -> None:
        """Save manifest to disk."""
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2, default=str)

    def get_dataset_path(self, org: str, name: str, version: str) -> Path:
        """
        Get path for a dataset in cache.

        Args:
            org: Organization name
            name: Dataset name
            version: Dataset version

        Returns:
            Path to dataset directory
        """
        return self.cache_root / org / name / version

    def is_cached(self, org: str, name: str, version: str) -> bool:
        """Check if dataset exists in cache."""
        key = f"{org}/{name}/{version}"
        return key in self.manifest["datasets"]

    def get_metadata(self, org: str, name: str, version: str) -> DatasetMetadata | None:
        """Get metadata for cached dataset."""
        key = f"{org}/{name}/{version}"
        if key in self.manifest["datasets"]:
            return DatasetMetadata(**self.manifest["datasets"][key])
        return None

    def add_dataset(
        self,
        metadata: DatasetMetadata,
        source_path: Path,
    ) -> Path:
        """
        Add dataset to cache.

        Args:
            metadata: Dataset metadata
            source_path: Path to source data

        Returns:
            Path where dataset was cached
        """
        dest_path = self.get_dataset_path(metadata.org, metadata.name, metadata.version)
        dest_path.mkdir(parents=True, exist_ok=True)

        # Copy data to cache
        if source_path.is_file():
            shutil.copy2(source_path, dest_path / source_path.name)
        else:
            # Copy directory contents
            for item in source_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, dest_path / item.name)
                else:
                    shutil.copytree(item, dest_path / item.name, dirs_exist_ok=True)

        # Save metadata
        metadata_path = dest_path / "metadata.json"
        with open(metadata_path, "w") as f:
            f.write(metadata.model_dump_json(indent=2))

        # Update manifest
        key = f"{metadata.org}/{metadata.name}/{metadata.version}"
        self.manifest["datasets"][key] = metadata.model_dump(mode="json")
        self._save_manifest()

        return dest_path

    def remove_dataset(self, org: str, name: str, version: str) -> None:
        """Remove dataset from cache."""
        key = f"{org}/{name}/{version}"
        if key in self.manifest["datasets"]:
            del self.manifest["datasets"][key]
            self._save_manifest()

        path = self.get_dataset_path(org, name, version)
        if path.exists():
            shutil.rmtree(path)

    def list_datasets(self, org: str | None = None) -> list[DatasetMetadata]:
        """List all cached datasets, optionally filtered by org."""
        datasets = []
        for key, meta_dict in self.manifest["datasets"].items():
            metadata = DatasetMetadata(**meta_dict)
            if org is None or metadata.org == org:
                datasets.append(metadata)
        return datasets

    def clear_cache(self, org: str | None = None) -> None:
        """Clear cache, optionally for specific org."""
        if org is None:
            shutil.rmtree(self.cache_root)
            self.cache_root.mkdir(parents=True, exist_ok=True)
            self._manifest = {"version": "1.0.0", "datasets": {}}
            self._save_manifest()
        else:
            to_remove = [k for k in self.manifest["datasets"] if k.startswith(f"{org}/")]
            for key in to_remove:
                org_name, name, version = key.split("/")
                self.remove_dataset(org_name, name, version)

# storage.py
"""OCI storage operations for datasets."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import oras.client
import oras.provider

from .types import (
    DatasetMetadata,
    PushResult,
)


class OCIStorageClient:
    """Client for OCI storage operations."""

    def __init__(
        self,
        registry: str = "localhost:5000",
        insecure: bool = False,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        """
        Initialize OCI storage client.

        Args:
            registry: Registry hostname
            insecure: Whether to use insecure HTTP
            username: Optional username for authentication
            password: Optional password for authentication
        """
        self.registry = registry
        self.client = oras.client.OrasClient(insecure=insecure)

        if username and password:
            self.client.login(
                username=username,
                password=password,
                insecure=insecure,
            )

    def push_dataset(
        self,
        file_path: Path,
        metadata: DatasetMetadata,
        target: str,
    ) -> PushResult:
        """
        Push dataset to registry.

        Args:
            file_path: Path to dataset file
            metadata: Dataset metadata
            target: Target location (e.g., 'registry/org/dataset:version')

        Returns:
            PushResult with operation details
        """
        try:
            # Prepare annotations
            annotations = {
                "org.opencontainers.image.created": metadata.created_at.isoformat(),
                "org.opencontainers.image.title": metadata.name,
                "io.dreadnode.dataset.format": metadata.format.value,
                "io.dreadnode.dataset.compression": metadata.compression.value,
                "io.dreadnode.dataset.strategy": metadata.storage_strategy.value,
                "io.dreadnode.dataset.size": str(metadata.size_bytes),
                "io.dreadnode.dataset.checksum": metadata.checksum,
            }

            if metadata.row_count is not None:
                annotations["io.dreadnode.dataset.rows"] = str(metadata.row_count)
            if metadata.column_count is not None:
                annotations["io.dreadnode.dataset.columns"] = str(metadata.column_count)
            if metadata.schema_json:
                annotations["io.dreadnode.dataset.schema"] = metadata.schema_json
            if metadata.description:
                annotations["org.opencontainers.image.description"] = metadata.description

            # Add custom tags
            for key, value in metadata.tags.items():
                annotations[f"io.dreadnode.dataset.tag.{key}"] = value

            # Create config with metadata
            with tempfile.TemporaryDirectory() as tmpdir:
                config_path = Path(tmpdir) / "config.json"
                with open(config_path, "w") as f:
                    json.dump(
                        {
                            "created": metadata.created_at.isoformat(),
                            "metadata": metadata.model_dump(mode="json"),
                        },
                        f,
                        indent=2,
                        default=str,
                    )

                # Push to registry
                response = self.client.push(
                    files=[str(file_path), str(config_path)],
                    target=target,
                )

                # Extract digest from response if available
                digest = None
                if hasattr(response, "headers") and "Docker-Content-Digest" in response.headers:
                    digest = response.headers["Docker-Content-Digest"]

                return PushResult(
                    success=True,
                    target=target,
                    digest=digest,
                    size_bytes=metadata.size_bytes,
                    metadata=metadata,
                )

        except Exception as e:
            return PushResult(
                success=False,
                target=target,
                digest=None,
                size_bytes=0,
                metadata=metadata,
                error=str(e),
            )

    def pull_dataset(
        self,
        target: str,
        output_dir: Path,
    ) -> tuple[Path, dict[str, Any]]:
        """
        Pull dataset from registry.

        Args:
            target: Source location
            output_dir: Directory to save dataset

        Returns:
            Tuple of (data_file_path, metadata_dict)
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Pull from registry
        files = self.client.pull(target=target, outdir=str(output_dir))

        if not files:
            raise ValueError(f"No files retrieved from {target}")

        # Find data file and config
        data_file = None
        config_file = None

        for file_path in files:
            path = Path(file_path)
            if path.name == "config.json":
                config_file = path
            else:
                data_file = path

        if not data_file:
            raise ValueError(f"No data file found in {target}")

        # Load metadata from config if available
        metadata_dict = {}
        if config_file and config_file.exists():
            with open(config_file) as f:
                config_data = json.load(f)
                metadata_dict = config_data.get("metadata", {})

        return data_file, metadata_dict

    def get_manifest(self, target: str) -> dict[str, Any]:
        """Get manifest for target."""
        try:
            return self.client.remote.get_manifest(target)  # type: ignore[return-value]
        except Exception as e:
            raise RuntimeError(f"Failed to get manifest for {target}: {e}") from e

    def tag_exists(self, target: str) -> bool:
        """Check if a tag exists in the registry."""
        try:
            self.get_manifest(target)
            return True
        except Exception:
            return False

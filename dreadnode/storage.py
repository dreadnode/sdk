# New storage.py module
import io
import os
from abc import ABC, abstractmethod

import fsspec

from .artifact import Artifact, ArtifactMetadata


class ArtifactStorage(ABC):
    """Base class for artifact storage."""

    @abstractmethod
    def save_artifact(self, run_id: str, artifact: Artifact) -> str:
        """Save an artifact and return its URI."""
        pass

    @abstractmethod
    def get_artifact(self, run_id: str, artifact_path: str) -> io.BytesIO:
        """Get an artifact's content."""
        pass

    @abstractmethod
    def list_artifacts(self, run_id: str, path: str | None = None) -> list[ArtifactMetadata]:
        """List artifacts for a run."""
        pass

    def get_artifact_uri(self, run_id: str, artifact_path: str) -> str:
        """Get a URI for referencing an artifact."""
        return f"artifact://{run_id}/{artifact_path}"


class FsspecArtifactStorage(ArtifactStorage):
    """Artifact storage using fsspec."""

    def __init__(self, base_uri: str):
        """Initialize with a URI that fsspec can handle."""
        self.base_uri = base_uri
        # Make sure base URI ends with a slash
        if not self.base_uri.endswith("/"):
            self.base_uri += "/"

        # Create content store for deduplication
        self.content_store_uri = f"{self.base_uri}content_store/"
        fs, _, _ = fsspec.get_fs_token_paths(self.content_store_uri)
        fs.makedirs(self.content_store_uri, exist_ok=True)

    def _get_run_path(self, run_id: str, artifact_path: str | None = None) -> str:
        """Get the full path for an artifact in a run."""
        if artifact_path:
            return f"{self.base_uri}{run_id}/artifacts/{artifact_path}"
        else:
            return f"{self.base_uri}{run_id}/artifacts/"

    def _get_content_path(self, content_hash: str) -> str:
        """Get the path for a content hash in the content store."""
        return f"{self.content_store_uri}{content_hash}"

    def save_artifact(self, run_id: str, artifact: Artifact) -> str:
        """Save an artifact using content addressing for deduplication."""
        # Set run_id if not already set
        if artifact.run_id is None:
            artifact.run_id = run_id

        # Get content hash path
        content_path = self._get_content_path(artifact.metadata.content_hash)

        # Check if content already exists
        fs, _, _ = fsspec.get_fs_token_paths(content_path)

        # Save content if it doesn't exist
        if not fs.exists(content_path):
            if artifact.local_path:
                with open(artifact.local_path, "rb") as f_src, fs.open(content_path, "wb") as f_dest:
                    f_dest.write(f_src.read())
            elif artifact.content:
                with fs.open(content_path, "wb") as f:
                    f.write(artifact.content)
            else:
                raise ValueError("Artifact has no content to save")

        # Now create the artifact reference
        artifact_path = self._get_run_path(run_id, artifact.metadata.path)

        # Ensure parent directories exist
        parent_dir = os.path.dirname(artifact_path)
        fs.makedirs(parent_dir, exist_ok=True)

        # Create the artifact reference (a JSON file with metadata)
        metadata = {
            "content_hash": artifact.metadata.content_hash,
            "size": artifact.metadata.size,
            "content_type": artifact.metadata.content_type,
            "description": artifact.metadata.description,
            "created_at": artifact.metadata.created_at.isoformat(),
            "custom_metadata": artifact.metadata.custom_metadata,
        }

        # Write metadata
        with fs.open(f"{artifact_path}.meta", "w") as f:
            import json

            json.dump(metadata, f)

        return artifact.metadata.path

    def get_artifact(self, run_id: str, artifact_path: str) -> io.BytesIO:
        """Get an artifact's content."""
        # Get metadata to find content hash
        meta_path = f"{self._get_run_path(run_id, artifact_path)}.meta"
        fs, _, _ = fsspec.get_fs_token_paths(meta_path)

        if not fs.exists(meta_path):
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        # Read metadata to get content hash
        with fs.open(meta_path, "r") as f:
            import json

            metadata = json.load(f)

        # Get content from content store
        content_path = self._get_content_path(metadata["content_hash"])

        # Read content
        with fs.open(content_path, "rb") as f:
            return io.BytesIO(f.read())

    def list_artifacts(self, run_id: str, path: str | None = None) -> list[ArtifactMetadata]:
        """List artifacts for a run."""
        base_path = self._get_run_path(run_id, path)
        fs, _, _ = fsspec.get_fs_token_paths(base_path)

        if not fs.exists(base_path):
            return []

        artifacts = []

        # List all .meta files in the directory
        for item in fs.glob(f"{base_path}/**/*.meta"):
            # Read metadata
            with fs.open(item, "r") as f:
                import json

                metadata = json.load(f)

            # Get artifact path (strip .meta from the end)
            rel_path = os.path.relpath(item[:-5], self._get_run_path(run_id, ""))

            # Create artifact metadata
            artifact_meta = ArtifactMetadata(
                path=rel_path,
                content_hash=metadata["content_hash"],
                size=metadata["size"],
                content_type=metadata["content_type"],
                description=metadata.get("description", ""),
                created_at=datetime.fromisoformat(metadata["created_at"]),
                custom_metadata=metadata.get("custom_metadata", {}),
            )

            artifacts.append(artifact_meta)

        return artifacts

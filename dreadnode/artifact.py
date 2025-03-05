from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

# --- Artifact Core ---


@dataclass
class ArtifactInfo:
    """Metadata about an artifact."""

    path: str  # Relative path within the run
    size: int = 0
    is_dir: bool = False
    file_format: str | None = None  # Optional format if known


@dataclass
class Artifact:
    """Represents an artifact associated with a run."""

    path: str  # Relative path within the run
    description: str = ""

    # Source of the artifact (only one should be set)
    local_path: Path | None = None  # Path to local file/dir to upload
    content: bytes | None = None  # In-memory content

    # Flag for eager saving
    eager: bool = False

    @property
    def is_dir(self) -> bool:
        """Check if this artifact represents a directory."""
        return self.local_path is not None and self.local_path.is_dir()


# --- Storage Abstraction ---


class ArtifactRepository(ABC):
    """Interface for artifact storage systems."""

    @abstractmethod
    def log_artifact(self, run_id: str, artifact: Artifact) -> None:
        """Save an artifact to the repository."""
        pass

    @abstractmethod
    def log_artifacts(self, run_id: str, local_dir: Path, artifact_path: str | None = None) -> None:
        """Save a directory of artifacts."""
        pass

    @abstractmethod
    def list_artifacts(self, run_id: str, path: str | None = None) -> list[ArtifactInfo]:
        """List artifacts for a run."""
        pass

    @abstractmethod
    def download_artifact(self, run_id: str, path: str) -> Path:
        """Download an artifact to a local path and return the path."""
        pass

    @abstractmethod
    def get_artifact_uri(self, run_id: str, path: str | None = None) -> str:
        """Get the URI for an artifact."""
        pass


class LocalArtifactRepository(ArtifactRepository):
    """Store artifacts in the local filesystem."""

    def __init__(self, base_path: Path | str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_run_artifact_dir(self, run_id: str) -> Path:
        """Get the artifact directory for a run."""
        run_dir = self.base_path / run_id / "artifacts"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def log_artifact(self, run_id: str, artifact: Artifact) -> None:
        """Save an artifact to the repository."""
        artifact_dir = self._get_run_artifact_dir(run_id)

        # Handle artifact path with possible subdirectories
        if artifact.path:
            dest_dir = artifact_dir / os.path.dirname(artifact.path)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = artifact_dir / artifact.path
        else:
            dest_path = artifact_dir / os.path.basename(str(artifact.local_path or "artifact"))

        # Save the artifact
        if artifact.local_path:
            if artifact.local_path.is_dir():
                self.log_artifacts(run_id, artifact.local_path, artifact.path)
            else:
                with open(artifact.local_path, "rb") as src, open(dest_path, "wb") as dst:
                    dst.write(src.read())
        elif artifact.content:
            with open(dest_path, "wb") as f:
                f.write(artifact.content)
        else:
            raise ValueError("Artifact has no content to save")

    def log_artifacts(self, run_id: str, local_dir: Path, artifact_path: str | None = None) -> None:
        """Save a directory of artifacts."""
        artifact_dir = self._get_run_artifact_dir(run_id)

        if artifact_path:
            dest_dir = artifact_dir / artifact_path
        else:
            dest_dir = artifact_dir

        dest_dir.mkdir(parents=True, exist_ok=True)

        for root, _, files in os.walk(local_dir):
            for file in files:
                src_path = Path(root) / file
                rel_path = src_path.relative_to(local_dir)
                dst_path = dest_dir / rel_path

                dst_path.parent.mkdir(parents=True, exist_ok=True)
                with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
                    dst.write(src.read())

    def list_artifacts(self, run_id: str, path: str | None = None) -> list[ArtifactInfo]:
        """List artifacts for a run."""
        artifact_dir = self._get_run_artifact_dir(run_id)

        if path:
            list_dir = artifact_dir / path
        else:
            list_dir = artifact_dir

        if not list_dir.exists():
            return []

        artifact_infos = []
        for item in list_dir.iterdir():
            rel_path = item.relative_to(artifact_dir)
            info = ArtifactInfo(
                path=str(rel_path),
                size=item.stat().st_size if item.is_file() else 0,
                is_dir=item.is_dir(),
            )
            artifact_infos.append(info)

        return artifact_infos

    def download_artifact(self, run_id: str, path: str) -> Path:
        """Download an artifact (local copy for consistency with remote repos)."""
        artifact_path = self._get_run_artifact_dir(run_id) / path
        return artifact_path

    def get_artifact_uri(self, run_id: str, path: str | None = None) -> str:
        """Get the URI for an artifact."""
        if path:
            return f"file://{self._get_run_artifact_dir(run_id) / path}"
        else:
            return f"file://{self._get_run_artifact_dir(run_id)}"

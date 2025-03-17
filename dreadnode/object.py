import contextlib
import hashlib
import typing as t
from dataclasses import dataclass

from pydantic import TypeAdapter


@dataclass
class ObjectRef:
    name: str
    kind: str
    hash: str


@dataclass
class ObjectUri:
    type: t.Literal["uri"]
    uri: str
    size: int


@dataclass
class ObjectVal:
    type: t.Literal["val"]
    value: t.Any


Object = ObjectUri | ObjectVal


def _hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()[:16]  # noqa: S324


def universal_hash(obj: t.Any) -> str:
    # Type adapter for most common objects
    with contextlib.suppress(Exception):
        json_str = TypeAdapter(t.Any).dump_json(obj)
        return _hash(json_str)

    # Numpy
    with contextlib.suppress(Exception):
        import numpy as np

        if isinstance(obj, np.ndarray):
            return _hash(obj.tobytes())

    # Pandas
    with contextlib.suppress(Exception):
        import pandas as pd  # type: ignore [import-untyped]

        if isinstance(obj, pd.DataFrame | pd.Series):
            return _hash(pd.util.hash_pandas_object(obj))

    # Hashable objects
    with contextlib.suppress(Exception):
        return _hash(hash(obj).to_bytes(16, "big"))

    # Try repr() - commenting for now as it's probably not stable
    # with contextlib.suppress(Exception):
    #     return g_hash_func(repr(obj).encode()).hexdigest()

    # Fallback to id()
    return _hash(id(obj).to_bytes(16, "big"))


# Storage


# class ObjectStorage(abc.ABC):
#     async def setup(self) -> None:
#         """Setup the artifact storage manager."""

#     @staticmethod
#     @abstractmethod
#     async def _save_artifact(
#         self,
#         run_state: RunState,
#         artifact_id: uuid.UUID,
#         data: ArtifactDataTypes,
#         original_type: ArtifactOriginalTypes,
#     ) -> ArtifactSaved:
#         """
#         Asynchronously saves an artifact.
#         Args:
#             run_state (RunState): The current state of the run.
#             artifact_id (uuid.UUID): The unique identifier of the artifact.
#             data (ArtifactDataTypes): The data of the artifact to be saved.
#             original_type (ArtifactOriginalTypes): The original type of the artifact.
#         Returns:
#             ArtifactSaved: The result of the artifact save operation.
#         """

#     @abstractmethod
#     async def _get_artifact(self, uri: str) -> ArtifactDataTypes:
#         """
#         Asynchronously retrieves an artifact from the given path.
#         Args:
#             path (str | Path): The path to the artifact.
#         Returns:
#             ArtifactDataTypes: The data type of the retrieved artifact.
#         """

#     @abstractmethod
#     async def _delete_artifact(self, uri: str | Path) -> bool:
#         """
#         Asynchronously deletes an artifact at the given path.
#         Args:
#             path (str | Path): The path to the artifact to be deleted.
#         Returns:
#             bool: True if the artifact was successfully deleted, False otherwise.
#         """


# class LocalArtifactRepository(ArtifactRepository):
#     """Store artifacts in the local filesystem."""

#     def __init__(self, base_path: Path | str):
#         self.base_path = Path(base_path)
#         self.base_path.mkdir(parents=True, exist_ok=True)

#     def _get_run_artifact_dir(self, run_id: str) -> Path:
#         """Get the artifact directory for a run."""
#         run_dir = self.base_path / run_id / "artifacts"
#         run_dir.mkdir(parents=True, exist_ok=True)
#         return run_dir

#     def log_artifact(self, run_id: str, artifact: Artifact) -> None:
#         """Save an artifact to the repository."""
#         artifact_dir = self._get_run_artifact_dir(run_id)

#         # Handle artifact path with possible subdirectories
#         if artifact.path:
#             dest_dir = artifact_dir / os.path.dirname(artifact.path)
#             dest_dir.mkdir(parents=True, exist_ok=True)
#             dest_path = artifact_dir / artifact.path
#         else:
#             dest_path = artifact_dir / os.path.basename(str(artifact.local_path or "artifact"))

#         # Save the artifact
#         if artifact.local_path:
#             if artifact.local_path.is_dir():
#                 self.log_artifacts(run_id, artifact.local_path, artifact.path)
#             else:
#                 with open(artifact.local_path, "rb") as src, open(dest_path, "wb") as dst:
#                     dst.write(src.read())
#         elif artifact.content:
#             with open(dest_path, "wb") as f:
#                 f.write(artifact.content)
#         else:
#             raise ValueError("Artifact has no content to save")

#     def log_artifacts(self, run_id: str, local_dir: Path, artifact_path: str | None = None) -> None:
#         """Save a directory of artifacts."""
#         artifact_dir = self._get_run_artifact_dir(run_id)

#         if artifact_path:
#             dest_dir = artifact_dir / artifact_path
#         else:
#             dest_dir = artifact_dir

#         dest_dir.mkdir(parents=True, exist_ok=True)

#         for root, _, files in os.walk(local_dir):
#             for file in files:
#                 src_path = Path(root) / file
#                 rel_path = src_path.relative_to(local_dir)
#                 dst_path = dest_dir / rel_path

#                 dst_path.parent.mkdir(parents=True, exist_ok=True)
#                 with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
#                     dst.write(src.read())

#     def list_artifacts(self, run_id: str, path: str | None = None) -> list[ArtifactInfo]:
#         """List artifacts for a run."""
#         artifact_dir = self._get_run_artifact_dir(run_id)

#         if path:
#             list_dir = artifact_dir / path
#         else:
#             list_dir = artifact_dir

#         if not list_dir.exists():
#             return []

#         artifact_infos = []
#         for item in list_dir.iterdir():
#             rel_path = item.relative_to(artifact_dir)
#             info = ArtifactInfo(
#                 path=str(rel_path),
#                 size=item.stat().st_size if item.is_file() else 0,
#                 is_dir=item.is_dir(),
#             )
#             artifact_infos.append(info)

#         return artifact_infos

#     def download_artifact(self, run_id: str, path: str) -> Path:
#         """Download an artifact (local copy for consistency with remote repos)."""
#         artifact_path = self._get_run_artifact_dir(run_id) / path
#         return artifact_path

#     def get_artifact_uri(self, run_id: str, path: str | None = None) -> str:
#         """Get the URI for an artifact."""
#         if path:
#             return f"file://{self._get_run_artifact_dir(run_id) / path}"
#         else:
#             return f"file://{self._get_run_artifact_dir(run_id)}"

"""
File manifest and change tracking for datasets.
"""

import hashlib
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from fsspec import AbstractFileSystem
from pydantic import BaseModel, Field
from tqdm import tqdm

from dreadnode.constants import CHUNK_SIZE, MANIFEST_FILE
from dreadnode.logging_ import console as logging_console


class FileStatus(str, Enum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    UNCHANGED = "unchanged"


class FileEntry(BaseModel):
    path: str
    size_bytes: int
    hash: str
    modified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FileEntry):
            return False
        return self.hash == other.hash

    def __hash__(self) -> int:
        return hash(self.hash)


class FileChange(BaseModel):
    path: str
    status: FileStatus
    old_entry: FileEntry | None = None
    new_entry: FileEntry | None = None


class FileManifest(BaseModel):
    version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    files: dict[str, FileEntry] = Field(default_factory=dict)
    total_size_bytes: int = 0
    file_count: int = 0
    checksum: str | None = None
    parent_version: str | None = None

    def model_post_init(self, _context: Any) -> None:
        self.file_count = len(self.files)
        self.total_size_bytes = sum(f.size_bytes for f in self.files.values())
        if not self.checksum:
            self.checksum = self.compute_checksum()

    def add_file(self, path: str, file_entry: FileEntry) -> None:
        self.files[path] = file_entry
        self._recalc_stats()

    def remove_file(self, path: str) -> None:
        if path in self.files:
            del self.files[path]
            self._recalc_stats()

    def _recalc_stats(self):
        self.file_count = len(self.files)
        self.total_size_bytes = sum(f.size_bytes for f in self.files.values())
        self.checksum = self.compute_checksum()

    def compute_checksum(self) -> str:
        hasher = hashlib.sha256()
        for path in sorted(self.files.keys()):
            file_entry = self.files[path]
            hasher.update(path.encode("utf-8"))
            hasher.update(file_entry.hash.encode("utf-8"))
        return hasher.hexdigest()

    def diff(self, other: "FileManifest") -> list[FileChange]:
        changes: list[FileChange] = []
        all_paths = set(self.files.keys()) | set(other.files.keys())
        for path in sorted(all_paths):
            old_entry = self.files.get(path)
            new_entry = other.files.get(path)
            if old_entry is None and new_entry is not None:
                changes.append(FileChange(path=path, status=FileStatus.ADDED, new_entry=new_entry))
            elif old_entry is not None and new_entry is None:
                changes.append(
                    FileChange(path=path, status=FileStatus.DELETED, old_entry=old_entry)
                )
            elif old_entry != new_entry:
                changes.append(
                    FileChange(
                        path=path,
                        status=FileStatus.MODIFIED,
                        old_entry=old_entry,
                        new_entry=new_entry,
                    )
                )
        return changes

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "files": {p: e.model_dump(mode="json") for p, e in self.files.items()},
            "total_size_bytes": self.total_size_bytes,
            "file_count": self.file_count,
            "checksum": self.checksum,
            "parent_version": self.parent_version,
        }

    def save(self, path: str | Path, fs: AbstractFileSystem | None = None) -> None:
        import json

        path = Path(path).joinpath(MANIFEST_FILE)

        with fs.open(path, "w") as f:
            json.dump(self.model_dump_json(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path, fs: AbstractFileSystem | None = None) -> "FileManifest":
        import json

        path = Path(path).joinpath(MANIFEST_FILE)

        with fs.open(path, "r") as f:
            data = json.load(f)

        return cls(**data)


class VersionHistory(BaseModel):
    uri: str
    versions: dict[str, FileManifest] = Field(default_factory=dict)
    current_version: str | None = None

    def add_version(self, manifest: FileManifest) -> None:
        self.versions[manifest.version] = manifest
        self.current_version = manifest.version

    def get_latest(self) -> FileManifest | None:
        if self.current_version:
            return self.versions.get(self.current_version)
        if not self.versions:
            return None
        return max(self.versions.values(), key=lambda m: m.created_at)


class Manifest(BaseModel):
    @staticmethod
    def compute_file_hash(
        file_path: str,
        fs: AbstractFileSystem,
        algorithm: str = "sha256",
        chunk_size: int = CHUNK_SIZE,
    ) -> str:
        """Compute hash using the provided filesystem (local or remote)."""
        hasher = hashlib.new(algorithm)
        try:
            # Use block_size for optimizations on streaming reads
            with fs.open(file_path, "rb", block_size=chunk_size) as f:
                while chunk := f.read(chunk_size):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logging_console.print(f"Failed to hash {file_path}: {e}")
            return ""

    @staticmethod
    def create_file_entry(
        file_path: str,
        fs: AbstractFileSystem,
        base_path: str,
        algorithm: str = "sha256",
    ) -> FileEntry:
        # Get file info
        info = fs.info(file_path)

        # Calculate relative path
        # Normalize paths to forward slashes
        file_path_str = str(file_path).replace("\\", "/")
        base_path_str = str(base_path).replace("\\", "/").rstrip("/")

        if file_path_str.startswith(base_path_str):
            rel_path = file_path_str[len(base_path_str) :].lstrip("/")
        else:
            rel_path = Path(file_path_str).name

        file_hash = Manifest.compute_file_hash(file_path, fs, algorithm)

        return FileEntry(
            path=rel_path,
            size_bytes=info.get("size", 0),
            hash=file_hash,
        )

    @staticmethod
    def create_manifest(
        path: str | Path,
        version: str,
        parent_version: str | None = None,
        exclude_patterns: list[str] | None = None,
        algorithm: str = "sha256",
        fs: AbstractFileSystem | None = None,
    ) -> FileManifest:
        # Use recursive listing
        files = fs.find(path, detail=False, withdirs=False)

        manifest = FileManifest(version=version, parent_version=parent_version)
        exclude_patterns = exclude_patterns or []
        import fnmatch

        for file_path in tqdm(files, desc="Hashing files"):
            # Check exclusion
            rel_name = Path(file_path).name
            if any(fnmatch.fnmatch(rel_name, pat) for pat in exclude_patterns):
                continue

            try:
                entry = Manifest.create_file_entry(file_path, fs, path, algorithm)
                manifest.add_file(entry.path, entry)
            except Exception as e:
                logging_console.print(f"Skipping {file_path}: {e}")

        return manifest

    def save(self, path: str | Path, fs: AbstractFileSystem | None = None) -> None:
        import json

        path = Path(path).joinpath(MANIFEST_FILE)

        with fs.open(path, "w") as f:
            json.dump(self.model_dump_json(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path, fs: AbstractFileSystem | None = None) -> "VersionHistory":
        import json

        path = Path(path).joinpath(MANIFEST_FILE)

        with fs.open(path, "r") as f:
            data = json.load(f)
        return cls.model_validate(data)

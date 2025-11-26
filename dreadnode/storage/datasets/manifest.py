import fnmatch
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyarrow.fs import FileSelector, FileSystem, FileType
from pydantic import BaseModel, Field
from tqdm import tqdm

from dreadnode.constants import CHUNK_SIZE, MANIFEST_FILE
from dreadnode.logging_ import console as logging_console


class FileEntry(BaseModel):
    path: str
    size_bytes: int
    hash: str
    modified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FileEntry):
            return False
        return self.hash == other.hash


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

    def add_file(self, path: str, entry: FileEntry):
        self.files[path] = entry

    def compute_checksum(self) -> str:
        """Computes a deterministic hash of the manifest content itself."""
        hasher = hashlib.sha256()
        # Sort keys to ensure deterministic ordering
        for path in sorted(self.files.keys()):
            entry = self.files[path]
            clean_path = path.replace("\\", "/")
            hasher.update(clean_path.encode("utf-8"))
            hasher.update(entry.hash.encode("utf-8"))
        return hasher.hexdigest()

    def save(self, path: str, fs: FileSystem) -> None:
        full_path = f"{path}/{MANIFEST_FILE}"
        # Serialize to bytes
        json_bytes = self.model_dump_json(indent=2).encode("utf-8")
        with fs.open_output_stream(full_path) as f:
            f.write(json_bytes)

    @classmethod
    def load(cls, path: str, fs: FileSystem) -> "FileManifest":
        full_path = f"{path}/{MANIFEST_FILE}"
        try:
            with fs.open_input_stream(full_path) as f:
                data = json.load(f)
            return cls(**data)
        except Exception as e:
            raise FileNotFoundError(f"Could not load manifest at {full_path}: {e}")

    def diff(self, other: "FileManifest") -> dict[str, set[str]]:
        """
        Returns what changed from 'other' (old) to 'self' (new).
        """
        added = set(self.files.keys()) - set(other.files.keys())
        removed = set(other.files.keys()) - set(self.files.keys())
        modified = set()

        common = set(self.files.keys()) & set(other.files.keys())
        for path in common:
            if self.files[path].hash != other.files[path].hash:
                modified.add(path)

        return {"added": added, "removed": removed, "modified": modified}

    def validate(self, base_path: str, fs: FileSystem) -> bool:
        """
        Verifies that files on disk match the manifest.
        Returns True if valid, raises exception or returns False if invalid.
        """
        logging_console.print(f"[*] Validating manifest version {self.version}...")

        valid = True
        for rel_path, entry in tqdm(self.files.items(), desc="Verifying integrity"):
            full_path = f"{base_path}/{rel_path}"

            # 1. Check Existence
            info = fs.get_file_info(full_path)
            if info.type == FileType.NotFound:
                logging_console.print(f"[!] Missing file: {rel_path}")
                valid = False
                continue

            # 2. Check Size (Fast fail)
            if info.size != entry.size_bytes:
                logging_console.print(f"[!] Size mismatch: {rel_path}")
                valid = False
                continue

            # 3. Check Hash (Slow, but necessary for full validation)
            # Note: You might make this optional or flag-based
            current_hash = Manifest.compute_file_hash(full_path, fs)
            if current_hash != entry.hash:
                logging_console.print(f"[!] Hash mismatch (Corruption): {rel_path}")
                valid = False

        return valid


class Manifest:
    @staticmethod
    def compute_file_hash(
        file_path: str,
        fs: FileSystem,
        algorithm: str = "sha256",
        chunk_size: int = CHUNK_SIZE,
    ) -> str:
        hasher = hashlib.new(algorithm)
        try:
            with fs.open_input_stream(file_path) as f:
                while chunk := f.read(chunk_size):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logging_console.print(f"Failed to hash {file_path}: {e}")
            return ""

    @staticmethod
    def create_manifest(
        path: str,
        version: str,
        parent_version: str | None = None,
        previous_manifest: FileManifest | None = None,
        exclude_patterns: list[str] | None = None,
        algorithm: str = "sha256",
        fs: FileSystem | None = None,
    ) -> FileManifest:
        if fs is None:
            raise ValueError("FileSystem must be provided")

        manifest = FileManifest(version=version, parent_version=parent_version)
        exclude_patterns = exclude_patterns or []

        selector = FileSelector(path, recursive=True)
        entries = fs.get_file_info(selector)

        # Normalize base path
        base_path_str = path.replace("\\", "/")
        if not base_path_str.endswith("/"):
            base_path_str += "/"

        # Create a lookup for the old manifest to speed up hashing
        old_files_map = previous_manifest.files if previous_manifest else {}

        skipped_count = 0
        hashed_count = 0

        for info in tqdm(entries, desc="Indexing dataset"):
            if info.type != FileType.File:
                continue

            full_path = info.path.replace("\\", "/")
            if full_path.startswith(base_path_str):
                rel_path = full_path[len(base_path_str) :]
            else:
                rel_path = Path(full_path).name

            if any(fnmatch.fnmatch(rel_path, pat) for pat in exclude_patterns):
                continue

            current_mtime = datetime.fromtimestamp(info.mtime_ns / 1e9, tz=timezone.utc)

            cached_entry = old_files_map.get(rel_path)

            is_unchanged = (
                cached_entry is not None
                and cached_entry.size_bytes == info.size
                and abs((cached_entry.modified_at - current_mtime).total_seconds()) < 1.0
            )

            if is_unchanged and cached_entry:
                entry = cached_entry
                entry.modified_at = current_mtime
                skipped_count += 1
            else:
                # COMPUTE HASH
                file_hash = Manifest.compute_file_hash(full_path, fs, algorithm)
                entry = FileEntry(
                    path=rel_path, size_bytes=info.size, hash=file_hash, modified_at=current_mtime
                )
                hashed_count += 1

            manifest.add_file(entry.path, entry)

        if previous_manifest:
            logging_console.print(
                f"[*] Incremental build: Re-hashed {hashed_count} files, Skipped {skipped_count} files."
            )

        return manifest

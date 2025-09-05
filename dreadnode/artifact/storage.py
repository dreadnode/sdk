"""
Artifact storage implementation for fsspec-compatible file systems.
Provides efficient uploading of files and directories with deduplication.
"""

import hashlib
from pathlib import Path

from loguru import logger

from dreadnode.artifact.credential_manager import CredentialManager

CHUNK_SIZE = 8 * 1024 * 1024  # 8MB


class ArtifactStorage:
    """
    Storage for artifacts with efficient handling of large files and directories.

    Supports:
    - Content-based deduplication using SHA1 hashing
    - Batch uploads for directories handled by fsspec
    """

    def __init__(self, credential_manager: CredentialManager):
        """
        Initialize artifact storage with credential manager.

        Args:
            credential_manager: Optional credential manager for S3 operations
        """
        self._credential_manager: CredentialManager = credential_manager

    def store_file(self, file_path: Path, target_key: str) -> str:
        """
        Store a file in the storage system, using multipart upload for large files.

        Args:
            file_path: Path to the local file
            target_key: Key/path where the file should be stored

        Returns:
            Full URI with protocol to the stored file
        """

        def store_operation() -> str:
            filesystem = self._credential_manager.get_filesystem()

            if not filesystem.exists(target_key):
                filesystem.put(str(file_path), target_key)
                logger.info("Artifact successfully stored at %s", target_key)
            else:
                logger.info("Artifact already exists at %s, skipping upload.", target_key)

            return str(filesystem.unstrip_protocol(target_key))

        return self._credential_manager.execute_with_retry(store_operation)

    def batch_upload_files(self, source_paths: list[str], target_paths: list[str]) -> list[str]:
        """
        Upload multiple files in a single batch operation.

        Args:
            source_paths: List of local file paths
            target_paths: List of target keys/paths

        Returns:
            List of URIs for the uploaded files
        """
        if not source_paths:
            return []

        def batch_upload_operation() -> list[str]:
            filesystem = self._credential_manager.get_filesystem()

            srcs = []
            dsts = []

            for src, dst in zip(source_paths, target_paths, strict=False):
                if not filesystem.exists(dst):
                    srcs.append(src)
                    dsts.append(dst)

            if srcs:
                filesystem.put(srcs, dsts)
                logger.info("Batch upload completed for %d files", len(srcs))
            else:
                logger.info("All files already exist, skipping upload")

            return [str(filesystem.unstrip_protocol(target)) for target in target_paths]

        return self._credential_manager.execute_with_retry(batch_upload_operation)

    def compute_file_hash(self, file_path: Path, stream_threshold_mb: int = 10) -> str:
        """
        Compute SHA1 hash of a file, using streaming only for larger files.

        Args:
            file_path: Path to the file
            stream_threshold_mb: Size threshold in MB for streaming vs. loading whole file

        Returns:
            First 16 chars of SHA1 hash
        """

        file_size = file_path.stat().st_size
        stream_threshold = stream_threshold_mb * 1024 * 1024

        sha1 = hashlib.sha1()  # noqa: S324 # nosec

        if file_size < stream_threshold:
            with file_path.open("rb") as f:
                data = f.read()
                sha1.update(data)
        else:
            with file_path.open("rb") as f:
                for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
                    sha1.update(chunk)

        return sha1.hexdigest()[:16]

    def compute_file_hashes(self, file_paths: list[Path]) -> dict[str, str]:
        """
        Compute SHA1 hashes for multiple files.

        Args:
            file_paths: List of file paths to hash

        Returns:
            Dictionary mapping file paths to their hash values
        """
        result = {}
        for file_path in file_paths:
            file_path_str = file_path.resolve().as_posix()
            result[file_path_str] = self.compute_file_hash(file_path)
        return result

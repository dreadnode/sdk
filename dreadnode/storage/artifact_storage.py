"""
Artifact storage implementation for fsspec-compatible file systems.
Provides efficient uploading of files and directories with deduplication.
"""

import hashlib
import logging
import mimetypes
from pathlib import Path

import fsspec

logger = logging.getLogger(__name__)

# Constants for multipart uploads
MULTIPART_THRESHOLD = 8 * 1024 * 1024  # 8MB
PART_SIZE = 8 * 1024 * 1024  # 8MB
MAX_PARTS = 10000  # S3 limit is 10,000 parts


class ArtifactStorage:
    """
    Storage for artifacts with efficient handling of large files and directories.

    Supports:
    - Content-based deduplication using SHA1 hashing
    - Multipart uploads for large files
    - Concurrent uploads for directories
    - File type detection
    """

    def __init__(self, file_system: fsspec.AbstractFileSystem):
        """
        Initialize artifact storage with a file system and prefix path.

        Args:
            file_system: FSSpec-compatible file system
        """
        self.file_system = file_system

    def store_file(self, file_path: Path, target_key: str) -> str:
        """
        Store a file in the storage system, using multipart upload for large files.

        Args:
            file_path: Path to the local file
            target_key: Key/path where the file should be stored

        Returns:
            Full URI with protocol to the stored file
        """
        file_size = file_path.stat().st_size

        if file_size < MULTIPART_THRESHOLD:
            return self._store_small_file(file_path, target_key)
        return self._store_large_file(file_path, target_key, file_size)

    def _store_small_file(self, file_path: Path, target_key: str) -> str:
        """
        Store a small file directly in the file system, using deduplication if supported.

        This method checks if the target file already exists in the file system. If not,
        it reads the file from the local path and writes it to the target location.
        The method ensures that the returned URI includes the appropriate protocol.

        Logging is used to track the file storage process, and exceptions are raised
        for common issues such as file not found, permission errors, or other I/O errors.

        Args:
            file_path: Path to the local file to be stored.
            target_key: Key/path where the file should be stored in the file system.

        Returns:
            str: Full URI with protocol to the stored file.

        Raises:
            FileNotFoundError: If the local file does not exist.
            PermissionError: If there are insufficient permissions to read/write the file.
            Exception: For any other unexpected errors during the file storage process.
        """
        try:
            if not self.file_system.exists(target_key):
                logger.info("Storing small file: %s to %s", file_path, target_key)
                with file_path.open("rb") as f_in:
                    data = f_in.read()
                    with self.file_system.open(target_key, "wb") as f_out:
                        f_out.write(data)
                logger.info("File successfully stored at %s", target_key)
            else:
                logger.info("File already exists at %s, skipping upload.", target_key)
        except FileNotFoundError:
            logger.exception("File not found: %s.", file_path)
            raise
        except Exception:
            logger.exception(
                "An error occurred while storing the file: %s.",
                file_path,
            )
            raise

        return str(self.file_system.unstrip_protocol(target_key))

    def _store_large_file(self, file_path: Path, target_key: str, file_size: int) -> str:
        """
        Store a large file in the file system using multipart upload if supported.

        This method checks if the target file already exists in the file system. If not,
        it determines whether the file system supports multipart uploads. If supported,
        the file is uploaded in parts using the `_multipart_upload` method. Otherwise,
        the `_chunked_upload` method is used to upload the file in sequential chunks.

        Logging is used to track the upload process, and exceptions are raised for
        common issues such as file system errors or unexpected failures.

        Args:
            file_path: Path to the local file to be uploaded.
            target_key: Key/path where the file should be stored in the file system.
            file_size: Size of the file in bytes.

        Returns:
            str: Full URI with protocol to the stored file.

        Raises:
            Exception: For any unexpected errors during the file storage process.
        """
        try:
            if not self.file_system.exists(target_key):
                logger.info("Storing large file: %s to %s", file_path, target_key)

                # Check if multipart upload is supported
                if self._supports_multipart_upload():
                    logger.info("Multipart upload supported. Uploading in parts.")
                    self._multipart_upload(file_path, target_key, file_size)
                else:
                    logger.info("Multipart upload not supported. Uploading in chunks.")
                    self._chunked_upload(file_path, target_key)

                logger.info("Large file successfully stored at %s", target_key)
            else:
                logger.info("File already exists at %s, skipping upload.", target_key)
        except Exception:
            logger.exception(
                "An error occurred while storing the large file: %s.",
                file_path,
            )
            raise

        return str(self.file_system.unstrip_protocol(target_key))

    def _supports_multipart_upload(self) -> bool:
        """
        Check if the file system supports multipart uploads.

        This method determines whether the underlying file system has the capability
        to perform multipart uploads.

        The method checks for the presence of two attributes in the file system:
        - `_s3`: Indicates that the file system is backed by an S3-compatible storage service.
        - `initiate_multipart_upload`: Confirms that the file system supports initiating
        multipart uploads.

        Returns:
            bool: True if the file system supports multipart uploads, False otherwise.
        """
        return hasattr(self.file_system, "_s3") and hasattr(
            self.file_system,
            "initiate_multipart_upload",
        )

    def _multipart_upload(self, file_path: Path, target_key: str, file_size: int) -> None:
        """
        Upload a file using the S3 multipart upload API.

        This method splits the file into parts and uploads each part using the
        multipart upload API. If the upload fails, it falls back to a chunked
        upload method.

        Args:
            file_path (Path): Path to the local file to be uploaded.
            target_key (str): Key/path where the file should be stored in the file system.
            file_size (int): Size of the file in bytes.

        Raises:
            IOError: If there is an issue reading the file.
            OSError: If there is an issue with the file system.
            fsspec.exceptions.FSTimeoutError: If the upload times out.
        """
        try:
            # Calculate optimal part size and number of parts
            num_parts = (file_size + PART_SIZE - 1) // PART_SIZE
            part_size = PART_SIZE

            # Ensure we don't exceed S3's part limit
            if num_parts > MAX_PARTS:
                part_size = (file_size + MAX_PARTS - 1) // MAX_PARTS
                part_size = max(part_size, 5 * 1024 * 1024)  # S3 minimum 5MB
                num_parts = (file_size + part_size - 1) // part_size

            # Initiate multipart upload
            logger.info("Initiating multipart upload for %s to %s", file_path, target_key)
            mpu = self.file_system.initiate_multipart_upload(target_key)

            # Upload parts
            parts = []
            with file_path.open("rb") as f:
                for part_number in range(1, num_parts + 1):
                    data = f.read(part_size)
                    if not data:
                        break
                    logger.debug(
                        "Uploading part %d of %d for %s",
                        part_number,
                        num_parts,
                        target_key,
                    )
                    etag = self.file_system.upload_part(mpu, part_number, data)
                    parts.append({"PartNumber": part_number, "ETag": etag})

            # Complete the multipart upload
            logger.info("Completing multipart upload for %s to %s", file_path, target_key)
            self.file_system.complete_multipart_upload(mpu, parts)

        except (OSError, fsspec.exceptions.FSTimeoutError) as e:
            logger.warning(
                "Multipart upload failed for %s to %s. Falling back to chunked upload. Error: %s",
                file_path,
                target_key,
                e,
            )
            self._chunked_upload(file_path, target_key)
        except Exception as e:
            logger.exception(
                "Unexpected error during multipart upload for %s to %s. Error: %s",
                file_path,
                target_key,
                e,
            )
            raise

    def _chunked_upload(self, file_path: Path, target_key: str) -> None:
        """
        Upload a file in chunks for file systems without multipart upload support.

        This method reads the file in fixed-size chunks and writes each chunk
        sequentially to the target location in the file system. It ensures efficient
        memory usage by processing one chunk at a time, making it suitable for large files.

        Args:
            file_path (Path): Path to the local file to be uploaded.
            target_key (str): Key/path where the file should be stored in the file system.
        """
        with self.file_system.open(target_key, "wb") as f_out, file_path.open("rb") as f_in:
            for chunk in iter(lambda: f_in.read(PART_SIZE), b""):
                f_out.write(chunk)

    def compute_file_hash(self, file_path: Path) -> str:
        """
        Compute SHA1 hash of a file, streaming for large files.

        Args:
            file_path: Path to the file

        Returns:
            First 16 chars of SHA1 hash
        """
        file_size = file_path.stat().st_size

        if file_size < MULTIPART_THRESHOLD:
            # Small file - read entirely
            with Path.open(file_path, "rb") as f:
                data = f.read()
                return hashlib.sha1(data).hexdigest()[:16]  # noqa: S324
        else:
            # Large file - stream in chunks
            sha1 = hashlib.sha1()  # noqa: S324
            with Path.open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(PART_SIZE), b""):
                    sha1.update(chunk)
            return sha1.hexdigest()[:16]

    def get_mime_type(self, file_path: Path) -> str | None:
        """Get MIME type for a file based on its extension."""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type

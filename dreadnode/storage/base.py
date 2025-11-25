"""
Base storage implementation for fsspec-compatible filesystems.
"""

import hashlib
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from botocore.exceptions import ClientError  # type: ignore[import-untyped]
from fsspec import AbstractFileSystem
from fsspec.callbacks import TqdmCallback
from fsspec.core import url_to_fs
from fsspec.utils import get_protocol
from loguru import logger

from dreadnode.constants import CHUNK_SIZE, FS_CREDENTIAL_REFRESH_BUFFER

if TYPE_CHECKING:
    from dreadnode.api.models import UserDataCredentials

T = TypeVar("T")


class BaseStorage:
    """
    Base storage for managing datasets with local and remote fsspec backends.
    """

    def __init__(
        self,
        credential_fetcher: Callable[[], "UserDataCredentials"] | None = None,
    ):
        """
        Initialize storage manager.

        Args:
            credential_fetcher: Function that returns UserDataCredentials for S3.
        """
        self._credential_fetcher = credential_fetcher
        self._s3_credentials: UserDataCredentials | None = None
        self._s3_credentials_expiry: datetime | None = None

        self._prefix: str = "datasets"

    def _get_s3_storage_options(self) -> dict[str, any]:
        """
        Get storage_options for S3, refreshing credentials if needed.
        """
        if not self._credential_fetcher:
            raise ValueError("No credential fetcher configured for S3 access")

        now = datetime.now(timezone.utc)
        needs_refresh = (
            not self._s3_credentials
            or not self._s3_credentials_expiry
            or (self._s3_credentials_expiry - now).total_seconds() < FS_CREDENTIAL_REFRESH_BUFFER
        )

        if needs_refresh:
            try:
                self._s3_credentials = self._credential_fetcher()
                self._s3_credentials_expiry = self._s3_credentials.expiration
                logger.debug("Refreshed S3 credentials")
            except Exception:
                logger.exception("Failed to refresh S3 storage credentials")
                raise

        return {
            "key": self._s3_credentials.access_key_id,
            "secret": self._s3_credentials.secret_access_key,
            "token": self._s3_credentials.session_token,
            "client_kwargs": {
                "endpoint_url": self._s3_credentials.endpoint,
                "region_name": self._s3_credentials.region,
            },
            "skip_instance_cache": True,
            "listings_expiry_time": 0,
        }

    def get_filesystem(self, path: str) -> tuple[AbstractFileSystem, str]:
        """
        Get the correct filesystem and clean path for a URL.

        Args:
            path: Full URL (e.g., "dn://org/dataset", "/local/path")

        Returns:
            Tuple of (filesystem, clean_path)
        """
        protocol = get_protocol(path)
        storage_options: dict[str, any] = {}
        is_remote = False

        if protocol in ("dn", "dreadnode"):
            storage_options = self._get_s3_storage_options()
            is_remote = True
        elif protocol == "file" or protocol == "":
            storage_options = {"auto_mkdir": True}

        fs, fs_path = url_to_fs(path, **storage_options)

        if is_remote and self._s3_credentials:
            # Construct full S3 path
            fs_path = (
                f"{self._s3_credentials.bucket}/"
                f"{self._s3_credentials.prefix}/"
                f"{self._prefix}/"
                f"{fs_path}"
            )

        return fs, fs_path

    def execute_with_retry(
        self,
        operation: Callable[[], T],
        max_retries: int = 3,
    ) -> T:
        """
        Execute operation with automatic credential refresh on auth errors.

        Args:
            operation: Operation to execute
            max_retries: Maximum retry attempts

        Returns:
            Operation result
        """
        for attempt in range(max_retries):
            try:
                return operation()
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                is_auth_error = error_code in [
                    "ExpiredToken",
                    "InvalidAccessKeyId",
                    "SignatureDoesNotMatch",
                ]

                if is_auth_error and attempt < max_retries - 1:
                    logger.info(
                        f"S3 credential error on attempt {attempt + 1}/{max_retries}, "
                        "forcing refresh..."
                    )
                    # Force expiry
                    self._s3_credentials_expiry = None
                    time.sleep(attempt + 1)
                    continue
                raise

        raise RuntimeError(f"Operation failed after {max_retries} attempts")

    def store_file(self, local_path: Path, target_url: str) -> str:
        """
        Store a file to target URL.

        Args:
            local_path: Local file path
            target_url: Destination URL

        Returns:
            Full URL to stored file
        """

        def store_operation() -> str:
            fs, target_path = self.get_filesystem(target_url)

            if not fs.exists(target_path):
                logger.info(f"Uploading {local_path} to {target_url}")
                fs.put(str(local_path), target_path, callback=TqdmCallback())
            else:
                logger.info(f"File already exists at {target_url}, skipping")

            return fs.unstrip_protocol(target_path)

        return self.execute_with_retry(store_operation)

    def batch_upload_files(
        self,
        source_paths: list[str],
        target_urls: list[str],
    ) -> list[str]:
        """
        Upload multiple files in batch.

        Args:
            source_paths: List of local file paths
            target_urls: List of destination URLs

        Returns:
            List of full URLs to uploaded files
        """
        if not source_paths:
            return []

        if len(source_paths) != len(target_urls):
            raise ValueError("source_paths and target_urls must have same length")

        # Group by protocol
        grouped_uploads = defaultdict(lambda: {"sources": [], "targets": []})
        for src, url in zip(source_paths, target_urls, strict=True):
            protocol = get_protocol(url)
            grouped_uploads[protocol]["sources"].append(src)
            grouped_uploads[protocol]["targets"].append(url)

        def batch_upload_operation() -> list[str]:
            for protocol, uploads in grouped_uploads.items():
                if not uploads["sources"]:
                    continue

                fs, _ = self.get_filesystem(uploads["targets"][0])

                srcs_to_upload = []
                dsts_to_upload = []

                for src, dst in zip(
                    uploads["sources"],
                    uploads["targets"],
                    strict=True,
                ):
                    _, dst_path = self.get_filesystem(dst)
                    if not fs.exists(dst_path):
                        srcs_to_upload.append(src)
                        dsts_to_upload.append(dst_path)

                if srcs_to_upload:
                    logger.info(f"Batch uploading {len(srcs_to_upload)} files")
                    fs.put(srcs_to_upload, dsts_to_upload, callback=TqdmCallback())
                else:
                    logger.info(f"All {len(uploads['sources'])} files already exist")

            return target_urls

        return self.execute_with_retry(batch_upload_operation)

    def compute_file_hash(
        self,
        file_path: Path,
        stream_threshold_mb: int = 10,
    ) -> str:
        """
        Compute SHA1 hash of file.

        Args:
            file_path: Path to file
            stream_threshold_mb: Size threshold for streaming

        Returns:
            First 16 chars of SHA1 hash
        """
        file_size = file_path.stat().st_size
        stream_threshold = stream_threshold_mb * 1024 * 1024
        sha1 = hashlib.sha1()  # nosec

        if file_size < stream_threshold:
            with file_path.open("rb") as f:
                sha1.update(f.read())
        else:
            with file_path.open("rb") as f:
                for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
                    sha1.update(chunk)

        return sha1.hexdigest()[:16]

    def compute_file_hashes(self, file_paths: list[Path]) -> dict[str, str]:
        """
        Compute hashes for multiple files.

        Args:
            file_paths: List of file paths

        Returns:
            Dictionary mapping file path to hash
        """
        result = {}
        for file_path in file_paths:
            file_path_str = file_path.resolve().as_posix()
            result[file_path_str] = self.compute_file_hash(file_path)
        return result

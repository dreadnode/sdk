"""
Artifact storage implementation for fsspec-compatible file systems.
Provides efficient uploading of files and directories with deduplication.
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

from dreadnode.constants import DATASETS_CACHE, FS_CREDENTIAL_REFRESH_BUFFER

if TYPE_CHECKING:
    from dreadnode.api.models import UserDataCredentials


T = TypeVar("T")


CHUNK_SIZE = 8 * 1024 * 1024  # 8MB


class BaseStorage:
    """
    Storage for artifacts that dynamically handles local and remote fsspec backends.
    """

    def __init__(self, credential_fetcher: Callable[[], "UserDataCredentials"]):
        """
        Initialize storage manager.

        Args:
            credential_fetcher: Function that returns new UserDataCredentials for S3.
        """
        self._credential_fetcher = credential_fetcher
        self._s3_credentials: UserDataCredentials | None = None
        self._s3_credentials_expiry: datetime | None = None
        self._local_cache_path: Path | None = Path(DATASETS_CACHE)
        self._remote_storage_path: Path | None = None

    def _get_s3_storage_options(self) -> dict:
        """
        Get the storage_options dictionary for S3, refreshing credentials if needed.
        """
        now = datetime.now(timezone.utc)
        needs_refresh = (
            not self._s3_credentials_expiry
            or (self._s3_credentials_expiry - now).total_seconds() < FS_CREDENTIAL_REFRESH_BUFFER
        )

        if needs_refresh:
            logger.info("Refreshing S3 storage credentials.")
            try:
                self._s3_credentials = self._credential_fetcher()
                self._s3_credentials_expiry = self._s3_credentials.expiration
                self._remote_storage_path = Path(
                    f"{self._s3_credentials.bucket}/{self._s3_credentials.prefix}"
                )
                logger.info(f"S3 credentials refreshed, valid until {self._s3_credentials_expiry}")
            except Exception:
                logger.exception("Failed to refresh S3 storage credentials.")
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
        }

    def get_filesystem(self, path: str) -> tuple[AbstractFileSystem, Path]:
        """
        Dynamically get the correct fsspec filesystem instance based on the URL protocol.

        Args:
            urlpath: The full URL to the resource (e.g., "s3://bucket/key" or "/local/path").

        Returns:
            A tuple containing the filesystem instance and the path within that filesystem.
        """
        protocol = get_protocol(path)
        storage_options = {}  # would prefer to pass these in and get fs based on config

        if protocol in ("dn", "dreadnode"):
            storage_options = self._get_s3_storage_options()
            fs, path = url_to_fs(path, **storage_options)
            path = self._remote_storage_path / "datasets"
        elif protocol == "file":
            storage_options = {"auto_mkdir": True}
            fs, path = url_to_fs(path, **storage_options)

        return fs, Path(path)

    def execute_with_retry(self, operation: Callable[[], T], max_retries: int = 3) -> T:
        """
        Execute an operation with automatic credential refresh on S3 auth errors.
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
                        "S3 credential error on attempt %d/%d, forcing refresh...",
                        attempt + 1,
                        max_retries,
                    )
                    # Force expiry to trigger a refresh on the next call to _get_s3_storage_options
                    self._s3_credentials_expiry = None
                    time.sleep(attempt + 1)
                    continue
                raise

        raise RuntimeError(f"Operation failed after {max_retries} attempts.")

    def store_file(self, local_path: Path, target_url: str) -> str:
        """
        Store a file, automatically detecting if the target is local or remote.

        Args:
            local_path: Path to the local file.
            target_url: The destination URL (e.g., "s3://bucket/key" or "/local/path/file").

        Returns:
            The full URL to the stored file.
        """

        def store_operation() -> str:
            fs, target_path = self.get_filesystem(target_url)

            if not fs.exists(target_path):
                fs.put(str(local_path), target_path, callback=TqdmCallback())
            else:
                logger.info("Artifact already exists at %s, skipping upload.", target_url)

            return fs.unstrip_protocol(target_path)

        # The retry logic is now primarily for S3, but is safe for local files.
        return self.execute_with_retry(store_operation)

    def batch_upload_files(self, source_paths: list[str], target_urls: list[str]) -> list[str]:
        """
        Upload multiple files in a single batch, handling mixed local/remote targets.

        Args:
            source_paths: List of local file paths to upload.
            target_urls: List of destination URLs (can be mixed protocols).

        Returns:
            List of full URLs for the uploaded files.
        """
        if not source_paths:
            return []

        if len(source_paths) != len(target_urls):
            raise ValueError("source_paths and target_urls must have the same number of elements.")

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
                for src, dst in zip(uploads["sources"], uploads["targets"], strict=True):
                    if not fs.exists(dst):
                        srcs_to_upload.append(src)
                        dsts_to_upload.append(dst)

                if srcs_to_upload:
                    fs.put(srcs_to_upload, dsts_to_upload, callback=TqdmCallback())
                else:
                    logger.info(
                        "All %d files for %s already exist, skipping upload.",
                        len(uploads["sources"]),
                        protocol,
                    )

            return target_urls

        return self.execute_with_retry(batch_upload_operation)

    def compute_file_hash(self, file_path: Path, stream_threshold_mb: int = 10) -> str:
        file_size = file_path.stat().st_size
        stream_threshold = stream_threshold_mb * 1024 * 1024
        sha1 = hashlib.sha1()
        if file_size < stream_threshold:
            with file_path.open("rb") as f:
                sha1.update(f.read())
        else:
            with file_path.open("rb") as f:
                for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
                    sha1.update(chunk)
        return sha1.hexdigest()[:16]

    def compute_file_hashes(self, file_paths: list[Path]) -> dict[str, str]:
        result = {}
        for file_path in file_paths:
            file_path_str = file_path.resolve().as_posix()
            result[file_path_str] = self.compute_file_hash(file_path)
        return result

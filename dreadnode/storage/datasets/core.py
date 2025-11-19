"""
High-performance filesystem operations with bidirectional sync.
"""

import fnmatch
import hashlib
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, TypeVar, cast

from botocore.exceptions import ClientError
from fsspec import AbstractFileSystem
from fsspec.callbacks import TqdmCallback
from fsspec.core import url_to_fs
from fsspec.utils import get_protocol

from dreadnode.logging_ import console as logging_console

# Mock constant if not provided in environment
try:
    from dreadnode.constants import FS_CREDENTIAL_REFRESH_BUFFER
except ImportError:
    FS_CREDENTIAL_REFRESH_BUFFER = 300  # Default 5 minutes

if TYPE_CHECKING:
    from dreadnode.api.models import UserDataCredentials

T = TypeVar("T")

# Configuration
CHUNK_SIZE = 64 * 1024 * 1024  # 64MB chunks for streaming
MAX_WORKERS = 10  # Parallel uploads/downloads
MULTIPART_THRESHOLD = 100 * 1024 * 1024  # 100MB
MULTIPART_CHUNKSIZE = 64 * 1024 * 1024  # 64MB parts


class FileMetadata(TypedDict):
    """Type definition for file metadata."""

    name: str
    size: int
    mtime: float
    type: str
    full_path: str


class SyncStats(TypedDict):
    """Type definition for sync statistics."""

    uploaded: int
    downloaded: int
    deleted: int
    skipped: int
    total_source: int
    total_dest: int
    operations: list[tuple[str, str, Any | None, Any | None] | tuple[str, str]]


class SyncDirection(Enum):
    UPLOAD = "upload"  # Local -> Remote
    DOWNLOAD = "download"  # Remote -> Local
    BIDIRECTIONAL = "bidirectional"  # Both directions (newest wins)


class SyncStrategy(Enum):
    SIZE = "size"
    TIMESTAMP = "timestamp"
    HASH = "hash"
    SIZE_AND_TIMESTAMP = "size_and_timestamp"


class FilesystemManager:
    """
    High-performance filesystem manager with bidirectional sync.
    """

    _instance: "FilesystemManager | None" = None
    _credential_fetcher: Callable[[], "UserDataCredentials"] | None = None
    _s3_credentials: "UserDataCredentials | None" = None
    _s3_credentials_expiry: datetime | None = None
    _max_workers: int = MAX_WORKERS
    _chunk_size: int = CHUNK_SIZE

    def __new__(cls) -> "FilesystemManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def configure(
        cls,
        credential_fetcher: Callable[[], "UserDataCredentials"] | None = None,
        max_workers: int = MAX_WORKERS,
        chunk_size: int = CHUNK_SIZE,
    ) -> None:
        """Configure the filesystem manager."""
        instance = cls()
        instance._credential_fetcher = credential_fetcher
        instance._max_workers = max_workers
        instance._chunk_size = chunk_size

    def _needs_credential_refresh(self) -> bool:
        """Check if credentials need refreshing."""
        if not self._s3_credentials or not self._s3_credentials_expiry:
            return True

        now = datetime.now(timezone.utc)
        if self._s3_credentials_expiry.tzinfo is None:
            expiry = self._s3_credentials_expiry.replace(tzinfo=timezone.utc)
        else:
            expiry = self._s3_credentials_expiry

        time_until_expiry = (expiry - now).total_seconds()
        return float(time_until_expiry) < float(FS_CREDENTIAL_REFRESH_BUFFER)

    def _refresh_credentials(self) -> None:
        """Refresh S3 credentials."""
        if not self._credential_fetcher:
            raise ValueError("No credential fetcher configured for remote access")

        try:
            self._s3_credentials = self._credential_fetcher()
            self._s3_credentials_expiry = self._s3_credentials.expiration
            logging_console.print("Refreshed S3 credentials")
        except Exception:
            logging_console.print("Failed to refresh S3 credentials")
            raise

    def _get_storage_options(self, protocol: str) -> dict[str, Any]:
        """Get storage options for the given protocol."""
        if protocol in ("dn", "dreadnode", "s3"):
            if self._needs_credential_refresh():
                self._refresh_credentials()

            if not self._s3_credentials:
                raise ValueError("S3 credentials not available")

            return {
                "key": self._s3_credentials.access_key_id,
                "secret": self._s3_credentials.secret_access_key,
                "token": self._s3_credentials.session_token,
                "client_kwargs": {
                    "endpoint_url": self._s3_credentials.endpoint,
                    "region_name": self._s3_credentials.region,
                },
                "skip_instance_cache": True,
                "use_listings_cache": False,
                "config_kwargs": {
                    "max_pool_connections": self._max_workers,
                    "retries": {"max_attempts": 5, "mode": "adaptive"},
                },
            }
        if protocol in ("", "file"):
            return {"auto_mkdir": True}
        return {}

    def get_filesystem(self, url: str) -> tuple[AbstractFileSystem, str]:
        """Get filesystem and clean path for a URL."""
        protocol = get_protocol(url)
        storage_options = self._get_storage_options(protocol)

        fs, fs_path = url_to_fs(url, **storage_options)

        if protocol in ("dn", "dreadnode") and self._s3_credentials:
            clean_fs_path = fs_path.lstrip("/")
            fs_path = f"{self._s3_credentials.bucket}/{self._s3_credentials.prefix}/datasets/{clean_fs_path}"

        return fs, fs_path

    def is_remote(self, url: str) -> bool:
        """Check if URL points to remote storage."""
        protocol = get_protocol(url)
        return protocol not in ("", "file")

    def force_refresh(self) -> None:
        """Force credential refresh on next filesystem access."""
        self._s3_credentials_expiry = None

    def stream_upload(
        self,
        local_path: Path,
        remote_url: str,
        chunk_size: int | None = None,
        callback: Any | None = None,
    ) -> str:
        """Stream upload large file without loading into memory."""
        chunk_size = chunk_size or self._chunk_size
        fs, remote_path = self.get_filesystem(remote_url)

        local_path = local_path.resolve()
        file_size = local_path.stat().st_size

        print(f"Streaming upload: {local_path} ({file_size / 1e9:.2f} GB) -> {remote_url}")

        if self.is_remote(remote_url):
            fs.put_file(
                str(local_path),
                remote_path,
                callback=callback or TqdmCallback(),
            )
        else:
            Path(remote_path).parent.mkdir(parents=True, exist_ok=True)

            if callback:
                if hasattr(callback, "set_size"):
                    callback.set_size(file_size)

                with Path.open(local_path, "rb") as src, Path.open(remote_path, "wb") as dst:
                    while True:
                        chunk = src.read(chunk_size)
                        if not chunk:
                            break
                        dst.write(chunk)
                        if hasattr(callback, "relative_update"):
                            callback.relative_update(len(chunk))
            else:
                shutil.copy2(local_path, remote_path)

        return fs.unstrip_protocol(remote_path)

    def stream_download(
        self,
        remote_url: str,
        local_path: Path,
        chunk_size: int | None = None,
        callback: Any | None = None,
    ) -> Path:
        """
        Stream download large file with optimized block sizes.

        Ensures strict streaming (low memory) and optimized read-ahead buffers
        for remote filesystems like S3.
        """
        chunk_size = chunk_size or self._chunk_size
        fs, remote_path = self.get_filesystem(remote_url)

        local_path.parent.mkdir(parents=True, exist_ok=True)
        logging_console.print(f"Streaming download: {remote_url} -> {local_path}")

        #  pass block_size to fs.open().
        open_kwargs = {}
        if self.is_remote(remote_url):
            open_kwargs["block_size"] = chunk_size
            # Use readahead cache for sequential downloads
            open_kwargs["cache_type"] = "readahead"

        # Manual stream loop to enforce memory usage and accurate progress tracking
        # try-except block is outside the tight loop for performance
        try:
            with fs.open(remote_path, "rb", **open_kwargs) as src:
                if callback and hasattr(callback, "set_size"):
                    try:
                        size = fs.size(remote_path)
                        callback.set_size(size)
                    except Exception:
                        pass

                with Path.open(local_path, "wb") as dst:
                    while True:
                        chunk = src.read(chunk_size)
                        if not chunk:
                            break
                        dst.write(chunk)

                        if callback and hasattr(callback, "relative_update"):
                            callback.relative_update(len(chunk))

        except Exception as e:
            logging_console.print(f"Failed to stream download {remote_url}: {e}")
            # Clean up partial download
            if local_path.exists():
                local_path.unlink()
            raise

        return local_path

    def batch_upload(
        self,
        file_pairs: list[tuple[Path, str]],
        max_workers: int | None = None,
    ) -> list[str]:
        """Upload multiple files in parallel."""
        max_workers = max_workers or self._max_workers

        if not file_pairs:
            return []

        logging_console.print(f"Batch uploading {len(file_pairs)} files with {max_workers} workers")

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Disable callbacks for batch operations to prevent console spam/locking
            futures = {
                executor.submit(
                    self.stream_upload,
                    local_path,
                    remote_url,
                    chunk_size=None,
                    callback=None,
                ): (local_path, remote_url)
                for local_path, remote_url in file_pairs
            }

            for future in as_completed(futures):
                local_path, _ = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logging_console.print(f"Uploaded: {local_path.name}")
                except Exception as e:
                    logging_console.print(f"Failed to upload {local_path}: {e}")
                    raise

        return results

    def batch_download(
        self,
        file_pairs: list[tuple[str, Path]],
        max_workers: int | None = None,
    ) -> list[Path]:
        """Download multiple files in parallel."""
        max_workers = max_workers or self._max_workers

        if not file_pairs:
            return []

        logging_console.print(
            f"Batch downloading {len(file_pairs)} files with {max_workers} workers"
        )

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.stream_download,
                    remote_url,
                    local_path,
                    chunk_size=None,
                    callback=None,
                ): (remote_url, local_path)
                for remote_url, local_path in file_pairs
            }

            for future in as_completed(futures):
                remote_url, local_path = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logging_console.print(f"Downloaded: {local_path.name}")
                except Exception as e:
                    logging_console.print(f"Failed to download {remote_url}: {e}")
                    raise

        return results

    def _list_files_recursive(self, fs: AbstractFileSystem, path: str) -> list[FileMetadata]:
        """List all files recursively in a directory."""
        try:
            files = fs.find(path, detail=True, withdirs=False)
            if isinstance(files, dict):
                result: list[FileMetadata] = []
                for k, v in files.items():
                    if v.get("type") == "file":
                        full_path_str = str(k).replace("\\", "/")
                        base_path_str = str(path).replace("\\", "/")

                        if base_path_str in full_path_str:
                            rel_path = full_path_str.replace(base_path_str, "").lstrip("/")
                        else:
                            rel_path = full_path_str

                        result.append(
                            {
                                "name": rel_path,
                                "size": v.get("size", 0),
                                "mtime": float(v.get("mtime", 0.0)),
                                "type": v.get("type", "file"),
                                "full_path": k,
                            }
                        )
                return result
            return []
        except Exception as e:
            logging_console.print(f"Failed to list files in {path}: {e}")
            return []

    def _compute_file_hash(self, fs: AbstractFileSystem, path: str, chunk_size: int = 8192) -> str:
        """Compute MD5 hash of a file."""
        hasher = hashlib.md5()  # Same as S3 ETag for single-part uploads
        try:
            with fs.open(path, "rb") as f:
                while chunk := f.read(chunk_size):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logging_console.print(f"Failed to hash {path}: {e}")
            return ""

    def _should_sync_file(
        self,
        source_file: FileMetadata,
        dest_file: FileMetadata | None,
        strategy: SyncStrategy,
        source_fs: AbstractFileSystem | None = None,
        dest_fs: AbstractFileSystem | None = None,
    ) -> bool:
        """Determine if a file should be synced based on strategy."""
        if dest_file is None:
            return True

        if strategy == SyncStrategy.SIZE:
            return source_file["size"] != dest_file["size"]

        if strategy == SyncStrategy.TIMESTAMP:
            return source_file["mtime"] > dest_file["mtime"]

        if strategy == SyncStrategy.SIZE_AND_TIMESTAMP:
            size_diff = source_file["size"] != dest_file["size"]
            time_diff = source_file["mtime"] > dest_file["mtime"]
            return size_diff or time_diff

        if strategy == SyncStrategy.HASH:
            if source_fs is None or dest_fs is None:
                logging_console.print("Hash strategy requires filesystems, falling back to size")
                return source_file["size"] != dest_file["size"]

            source_hash = self._compute_file_hash(source_fs, source_file["full_path"])
            dest_hash = self._compute_file_hash(dest_fs, dest_file["full_path"])
            return source_hash != dest_hash

        return False

    def sync_directory(
        self,
        source: str,
        destination: str,
        direction: SyncDirection = SyncDirection.UPLOAD,
        strategy: SyncStrategy = SyncStrategy.SIZE_AND_TIMESTAMP,
        *,
        delete: bool = False,
        dry_run: bool = False,
        max_workers: int | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> SyncStats:
        """Bidirectional directory sync with multiple strategies."""
        max_workers = max_workers or self._max_workers
        exclude_patterns = exclude_patterns or []

        source_fs, source_path = self.get_filesystem(source)
        dest_fs, dest_path = self.get_filesystem(destination)

        print(
            f"Syncing directory: {source} -> {destination} "
            f"(direction={direction.value}, strategy={strategy.value})"
        )

        source_files = self._list_files_recursive(source_fs, source_path)
        dest_files = self._list_files_recursive(dest_fs, dest_path)

        if exclude_patterns:
            source_files = [
                f
                for f in source_files
                if not any(fnmatch.fnmatch(f["name"], pat) for pat in exclude_patterns)
            ]
            dest_files = [
                f
                for f in dest_files
                if not any(fnmatch.fnmatch(f["name"], pat) for pat in exclude_patterns)
            ]

        source_dict = {f["name"]: f for f in source_files}
        dest_dict = {f["name"]: f for f in dest_files}

        to_transfer: list[tuple[str, str, Any, Any]] = []
        to_delete: list[tuple[str, str]] = []

        if direction in (SyncDirection.UPLOAD, SyncDirection.BIDIRECTIONAL):
            for name, source_file in source_dict.items():
                dest_file = dest_dict.get(name)
                should_sync = False

                if direction == SyncDirection.BIDIRECTIONAL and dest_file:
                    if source_file["mtime"] > dest_file["mtime"]:
                        should_sync = self._should_sync_file(
                            source_file, dest_file, strategy, source_fs, dest_fs
                        )
                elif self._should_sync_file(source_file, dest_file, strategy, source_fs, dest_fs):
                    should_sync = True

                if should_sync:
                    to_transfer.append(("upload", name, source_file, dest_file))

        if direction in (SyncDirection.DOWNLOAD, SyncDirection.BIDIRECTIONAL):
            for name, dest_file in dest_dict.items():
                source_file = source_dict.get(name)
                should_sync = False

                if direction == SyncDirection.BIDIRECTIONAL and source_file:
                    if dest_file["mtime"] > source_file["mtime"]:
                        should_sync = self._should_sync_file(
                            dest_file, source_file, strategy, dest_fs, source_fs
                        )
                elif source_file is None:
                    should_sync = True
                    source_file = None
                elif self._should_sync_file(dest_file, source_file, strategy, dest_fs, source_fs):
                    should_sync = True

                if should_sync:
                    to_transfer.append(("download", name, dest_file, source_file))

        if delete:
            if direction == SyncDirection.UPLOAD:
                to_delete = [("delete_dest", name) for name in dest_dict if name not in source_dict]
            elif direction == SyncDirection.DOWNLOAD:
                to_delete = [
                    ("delete_source", name) for name in source_dict if name not in dest_dict
                ]

        stats: SyncStats = {
            "uploaded": 0,
            "downloaded": 0,
            "deleted": 0,
            "skipped": len(source_dict) - len([t for t in to_transfer if t[0] == "upload"]),
            "total_source": len(source_files),
            "total_dest": len(dest_files),
            "operations": cast(
                "list[tuple[str, str, Any | None, Any | None] | tuple[str, str]]",
                to_transfer + to_delete,
            ),
        }

        if dry_run:
            logging_console.print("Dry run - no changes made")
            return stats

        uploads = [t for t in to_transfer if t[0] == "upload"]
        if uploads:
            upload_pairs: list[tuple[Path, str]] = []
            is_source_remote = self.is_remote(source)
            is_dest_remote = self.is_remote(destination)

            for _, name, _, _ in uploads:
                src_p = Path(source_path) / name if not is_source_remote else None
                dst_p = f"{destination.rstrip('/')}/{name}"
                if src_p:
                    upload_pairs.append((src_p, dst_p))

            if is_source_remote:
                for _, name, source_file, _ in uploads:
                    source_file_path = source_file["full_path"]
                    dest_file_path = f"{dest_path}/{name}"
                    if not is_dest_remote:
                        Path(dest_file_path).parent.mkdir(parents=True, exist_ok=True)
                    try:
                        # using block_size for remote-to-remote copy
                        with (
                            source_fs.open(
                                source_file_path, "rb", block_size=self._chunk_size
                            ) as src,
                            dest_fs.open(dest_file_path, "wb") as dst,
                        ):
                            while chunk := src.read(self._chunk_size):
                                dst.write(chunk)
                        stats["uploaded"] += 1
                    except Exception as e:
                        print(f"Failed to copy {name}: {e}")
            else:
                self.batch_upload(upload_pairs, max_workers=max_workers)
                stats["uploaded"] = len(uploads)

        downloads = [t for t in to_transfer if t[0] == "download"]
        if downloads:
            download_pairs: list[tuple[str, Path]] = []
            is_dest_remote = self.is_remote(destination)

            for _, name, _, _ in downloads:
                src_url = f"{source.rstrip('/')}/{name}"
                if not is_dest_remote:
                    dst_p = Path(destination) / name
                    download_pairs.append((src_url, dst_p))

            if not is_dest_remote:
                self.batch_download(download_pairs, max_workers=max_workers)
                stats["downloaded"] = len(downloads)

        for op, name in to_delete:
            try:
                if op == "delete_dest":
                    dest_fs.rm(f"{dest_path}/{name}")
                elif op == "delete_source":
                    source_fs.rm(f"{source_path}/{name}")
                stats["deleted"] += 1
            except Exception as e:
                print(f"Failed to delete {name}: {e}")

        print(
            f"Sync complete: uploaded={stats['uploaded']}, "
            f"downloaded={stats['downloaded']}, deleted={stats['deleted']}"
        )

        return stats


_fs_manager = FilesystemManager()


def configure_filesystem(
    credential_fetcher: Callable[[], "UserDataCredentials"] | None = None,
    max_workers: int = MAX_WORKERS,
    chunk_size: int = CHUNK_SIZE,
) -> None:
    FilesystemManager.configure(credential_fetcher, max_workers, chunk_size)


def get_filesystem(url: str) -> tuple[AbstractFileSystem, str]:
    return _fs_manager.get_filesystem(url)


def is_remote(url: str) -> bool:
    return _fs_manager.is_remote(url)


def sync_to_local(
    remote_url: str,
    local_dir: Path,
    strategy: SyncStrategy = SyncStrategy.SIZE_AND_TIMESTAMP,
    *,
    delete: bool = False,
    dry_run: bool = False,
    max_workers: int = MAX_WORKERS,
    exclude_patterns: list[str] | None = None,
) -> SyncStats:
    return _fs_manager.sync_directory(
        source=remote_url,
        destination=str(local_dir),
        direction=SyncDirection.DOWNLOAD,
        strategy=strategy,
        delete=delete,
        dry_run=dry_run,
        max_workers=max_workers,
        exclude_patterns=exclude_patterns,
    )


def sync_to_remote(
    local_dir: Path,
    remote_url: str,
    strategy: SyncStrategy = SyncStrategy.SIZE_AND_TIMESTAMP,
    *,
    delete: bool = False,
    dry_run: bool = False,
    max_workers: int = MAX_WORKERS,
    exclude_patterns: list[str] | None = None,
) -> SyncStats:
    return _fs_manager.sync_directory(
        source=str(local_dir),
        destination=remote_url,
        direction=SyncDirection.UPLOAD,
        strategy=strategy,
        delete=delete,
        dry_run=dry_run,
        max_workers=max_workers,
        exclude_patterns=exclude_patterns,
    )


def sync_bidirectional(
    local_dir: Path,
    remote_url: str,
    strategy: SyncStrategy = SyncStrategy.TIMESTAMP,
    *,
    dry_run: bool = False,
    max_workers: int = MAX_WORKERS,
    exclude_patterns: list[str] | None = None,
) -> SyncStats:
    return _fs_manager.sync_directory(
        source=str(local_dir),
        destination=remote_url,
        direction=SyncDirection.BIDIRECTIONAL,
        strategy=strategy,
        delete=False,
        dry_run=dry_run,
        max_workers=max_workers,
        exclude_patterns=exclude_patterns,
    )


def upload_file(
    local_path: Path,
    remote_url: str,
    *,
    show_progress: bool = True,
) -> str:
    callback = TqdmCallback(tqdm_kwargs={"desc": "Uploading"}) if show_progress else None
    return _fs_manager.stream_upload(local_path, remote_url, callback=callback)


def download_file(
    remote_url: str,
    local_path: Path,
    *,
    show_progress: bool = True,
) -> Path:
    callback = TqdmCallback(tqdm_kwargs={"desc": "Downloading"}) if show_progress else None
    return _fs_manager.stream_download(remote_url, local_path, callback=callback)


def upload_directory(
    local_dir: Path,
    remote_url: str,
    max_workers: int = MAX_WORKERS,
) -> list[str]:
    files = list(local_dir.rglob("*"))
    file_pairs = [
        (f, f"{remote_url}/{f.relative_to(local_dir).as_posix()}") for f in files if f.is_file()
    ]
    return _fs_manager.batch_upload(file_pairs, max_workers=max_workers)


def download_directory(
    remote_url: str,
    local_dir: Path,
    max_workers: int = MAX_WORKERS,
) -> list[Path]:
    fs, remote_path = _fs_manager.get_filesystem(remote_url)
    files = _fs_manager._list_files_recursive(fs, remote_path)

    download_pairs = [(f"{remote_url}/{f['name']}", local_dir / f["name"]) for f in files]

    return _fs_manager.batch_download(download_pairs, max_workers=max_workers)


@contextmanager
def filesystem_context(url: str):
    max_retries = 3 if is_remote(url) else 1
    last_exception = None

    for attempt in range(max_retries):
        try:
            fs, path = get_filesystem(url)
            yield fs, path
            return
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            is_auth_error = error_code in [
                "ExpiredToken",
                "InvalidAccessKeyId",
                "SignatureDoesNotMatch",
            ]
            if is_auth_error and attempt < max_retries - 1:
                logging_console.print(
                    f"Auth error on attempt {attempt + 1}/{max_retries}, refreshing..."
                )
                _fs_manager.force_refresh()
                last_exception = e
                continue
            raise
        except Exception as e:
            last_exception = e
            raise
    if last_exception:
        raise last_exception


def with_filesystem_retry(func: Callable[..., T]) -> Callable[..., T]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        max_retries = 3
        last_exception = None
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                is_auth_error = error_code in [
                    "ExpiredToken",
                    "InvalidAccessKeyId",
                    "SignatureDoesNotMatch",
                ]
                if is_auth_error and attempt < max_retries - 1:
                    logging_console.print(
                        f"Auth error on attempt {attempt + 1}/{max_retries}, refreshing..."
                    )
                    _fs_manager.force_refresh()
                    last_exception = e
                    continue
                raise
            except Exception as e:
                last_exception = e
                raise
        if last_exception:
            raise last_exception
        raise RuntimeError("Retry loop failed unexpectedly")

    return wrapper

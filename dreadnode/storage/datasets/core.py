import hashlib
from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, TypeVar

from fsspec import AbstractFileSystem
from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem
from fsspec.utils import get_protocol

from dreadnode.constants import (
    CHUNK_SIZE,
    DEFAULT_LOCAL_STORAGE_DIR,
    DEFAULT_REMOTE_STORAGE_PREFIX,
    FS_CREDENTIAL_REFRESH_BUFFER,
    MAX_WORKERS,
)
from dreadnode.logging_ import console as logging_console

if TYPE_CHECKING:
    from dreadnode.api.models import UserDataCredentials

T = TypeVar("T")


class Repository(str, Enum):
    """Type definition for storage object."""

    DATASETS = "datasets"
    MODELS = "models"
    ARTIFACTS = "artifacts"
    AGENTS = "agents"
    TOOLSETS = "toolsets"
    ENVIRONMENTS = "environments"


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
    Filesystem manager with bidirectional sync.
    """

    organization: str | None = None

    _instance: "FilesystemManager | None" = None
    _credential_fetcher: Callable[[], "UserDataCredentials"] | None = None
    _credentials_expiry: datetime | None = None

    _max_workers: int = MAX_WORKERS
    _chunk_size: int = CHUNK_SIZE
    _remote_prefix: Path = Path(DEFAULT_REMOTE_STORAGE_PREFIX)
    _cache_root: Path = Path(DEFAULT_LOCAL_STORAGE_DIR)

    def __new__(cls) -> "FilesystemManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def configure(
        cls,
        credential_fetcher: Callable[[], "UserDataCredentials"] | None = None,
        organization: str | None = None,
    ) -> None:
        """Configure the filesystem manager."""
        instance = cls()
        instance._credential_fetcher = credential_fetcher
        instance.organization = organization

        return instance

    def _configure_local_fs(self, *, auto_mkdir: bool = True, **kwargs) -> LocalFileSystem:
        """Configure local filesystem."""
        return {"auto_mkdir": auto_mkdir, **kwargs}

    def _configure_remote_fs(self) -> AbstractFileSystem:
        """Configure remote filesystem."""
        if self._needs_credential_refresh() or self._credentials_expiry is None:
            storage_options = self._refresh_credentials()
        return storage_options

    def _needs_credential_refresh(self) -> bool:
        """Check if credentials need refreshing."""
        if not self._credentials_expiry:
            return True

        now = datetime.now(timezone.utc)
        if self._credentials_expiry.tzinfo is None:
            expiry = self._credentials_expiry.replace(tzinfo=timezone.utc)
        else:
            expiry = self._credentials_expiry

        time_until_expiry = (expiry - now).total_seconds()
        return float(time_until_expiry) < float(FS_CREDENTIAL_REFRESH_BUFFER)

    def _refresh_credentials(self) -> dict[str, Any]:
        """Refresh credentials."""
        if not self._credential_fetcher:
            raise ValueError("No credential fetcher configured for remote access")

        try:
            credentials = self._credential_fetcher()
            storage_options = {
                "key": credentials.access_key_id,
                "secret": credentials.secret_access_key,
                "token": credentials.session_token,
                "client_kwargs": {
                    "endpoint_url": credentials.endpoint,
                    "region_name": credentials.region,
                },
                "skip_instance_cache": True,
                "use_listings_cache": False,
                "config_kwargs": {
                    "max_pool_connections": self._max_workers,
                    "retries": {"max_attempts": 5, "mode": "adaptive"},
                },
            }
        except Exception:
            logging_console.print("Failed to refresh credentials")
            raise

        return storage_options

    def get_dataset_dir(
        self,
        path: str,
        version: str | None,
        repo: Repository,
        *,
        to_cache: bool,
        filesystem: AbstractFileSystem,
    ) -> Path:
        """
        Resolves the dataset directory path.
        """
        if to_cache:
            dataset_dir = self._cache_root.joinpath(repo, path)
            if dataset_dir.exists():
                return self._resolve_version(dataset_dir, version=version, fs=filesystem)
            raise FileNotFoundError(f"Dataset not found in cache at {dataset_dir}")

        dataset_dir = self._remote_prefix.joinpath(repo, path)
        return self._resolve_version(dataset_dir, version=version, fs=filesystem)

    def get_protocol(self, url: str) -> str:
        """Get protocol for a URL."""
        return get_protocol(url)

    def get_filesystem(self, url: str) -> tuple[AbstractFileSystem, str]:
        """Get filesystem and clean path for a URL."""
        if self.is_remote(url):
            fs, uri = url_to_fs(url, **self._configure_remote_fs())
        else:
            fs, uri = url_to_fs(url, **self._configure_local_fs())

        return fs, uri

    def is_remote(self, uri: str) -> bool:
        """Check if URL points to remote storage."""
        protocol = get_protocol(uri)
        return protocol in ("dn", "dreadnode", "s3", "gs", "azure", "hdfs", "http", "https")

    def force_refresh(self) -> None:
        """Force credential refresh on next filesystem access."""
        self._credentials_expiry = None

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
        """
        Compute MD5 hash of a file. Uses MD5 for E-tag compatibility.
        """
        hasher = hashlib.md5()  # nosec
        try:
            with fs.open(path, "rb") as f:
                while chunk := f.read(chunk_size):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logging_console.print(f"Failed to hash {path}: {e}")
            return ""

    def _resolve_version(self, path: Path, version: str | None, fs: AbstractFileSystem) -> Path:
        """
        Finds the correct version. Path should be 'org/name'.
        """

        if not fs.exists(path):
            target_version = version or "0.0.1"
            print(f"[*] New dataset, initializing version {target_version} at {path}")
            return Path(path) / target_version

        try:
            entries = fs.ls(path, detail=False)

            # no entries found, return path with default version
            if not entries:
                print(f"[!] No versions found at {path}, defaulting to version '0.0.1'")
                return path.joinpath(version if version else "0.0.1")

            # if version is None, get the latest version
            if not version:
                print(f"[*] No version specified, resolving latest version for {path}")
                return Path(sorted(entries, reverse=True)[0])

            if fs.exists(path.joinpath(version)):
                print(f"[*] Resolved version {version} for {path}")
                return Path(path).joinpath(version)
            print(f"[*] Version {version} not found in {path}, creating new version.")
            return Path(path).joinpath(version)

        except Exception as e:
            raise FileNotFoundError(f"Failed to resolve version for {path}: {e}") from e

from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow.fs as pafs  # The Native FS
from pyarrow.fs import FileSystem, FileType

from dreadnode.constants import (
    DEFAULT_LOCAL_STORAGE_DIR,
    DEFAULT_REMOTE_STORAGE_PREFIX,
    FS_CREDENTIAL_REFRESH_BUFFER,
)
from dreadnode.logging_ import console as logging_console

if TYPE_CHECKING:
    from dreadnode.api.models import UserDataCredentials


class Repository(str, Enum):
    DATASETS = "datasets"
    MODELS = "models"
    ARTIFACTS = "artifacts"
    AGENTS = "agents"
    TOOLSETS = "toolsets"
    ENVIRONMENTS = "environments"


class FilesystemManager:
    """
    Filesystem manager that generates Native PyArrow filesystems
    injected with your custom credentials.
    """

    organization: str | None = None
    _instance: "FilesystemManager | None" = None

    # Auth State
    _credential_fetcher: Callable[[], "UserDataCredentials"] | None = None
    _credentials_expiry: datetime | None = None
    _cached_s3_fs: pafs.S3FileSystem | None = None

    _remote_prefix: str = DEFAULT_REMOTE_STORAGE_PREFIX
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
    ) -> "FilesystemManager":
        instance = cls()
        instance._credential_fetcher = credential_fetcher
        instance.organization = organization
        return instance

    def _get_s3_config(self) -> dict[str, Any]:
        """
        Translates your UserDataCredentials into PyArrow S3 arguments.
        """
        if not self._credential_fetcher:
            raise ValueError("No credential fetcher configured")

        creds = self._credential_fetcher()

        # Mapping UserDataCredentials -> PyArrow S3FileSystem kwargs
        return {
            "access_key": creds.access_key_id,
            "secret_key": creds.secret_access_key,
            "session_token": creds.session_token,
            "region": creds.region,
            "endpoint_override": creds.endpoint,  # Crucial for Minio/Custom S3
            "scheme": "http" if "localhost" in creds.endpoint else "https",
        }

    def _needs_refresh(self) -> bool:
        if not self._credentials_expiry:
            return True
        now = datetime.now(timezone.utc)
        expiry = (
            self._credentials_expiry.replace(tzinfo=timezone.utc)
            if self._credentials_expiry.tzinfo is None
            else self._credentials_expiry
        )
        return (expiry - now).total_seconds() < FS_CREDENTIAL_REFRESH_BUFFER

    def get_fs_and_path(self, uri: str) -> tuple[FileSystem, str]:
        """
        The Master Resolver.
        Returns: (NativeFileSystem, clean_path_string)
        """

        # 1. Handle Local Files
        if "://" not in uri or uri.startswith("file://"):
            # Native Local FS
            fs = pafs.LocalFileSystem()
            # Clean local path
            path = uri.replace("file://", "")
            return fs, path

        # 2. Handle Remote (S3/Minio)
        try:
            protocol, path_body = uri.split("://", 1)
        except ValueError:
            # Fallback if weird formatting
            return pafs.LocalFileSystem(), uri

        # Reuse the cached S3 connection unless expired
        if self._cached_s3_fs is None or self._needs_refresh():
            try:
                config = self._get_s3_config()
                self._cached_s3_fs = pafs.S3FileSystem(**config)
                self._credentials_expiry = self._credential_fetcher().expiration

            except Exception as e:
                logging_console.print(f"Auth failed: {e}")
                raise

        return self._cached_s3_fs, path_body

    def resolve_version(self, fs: FileSystem, path: str, version: str | None) -> str:
        """
        PyArrow Native implementation of version resolution.
        """

        # 1. Check if the base path exists
        info = fs.get_file_info(path)
        if info.type == FileType.NotFound:
            target = version or "0.0.1"
            print(f"[*] New dataset, initializing {target} at {path}")
            return f"{path}/{target}"

        # 2. If version explicitly requested
        if version:
            ver_path = f"{path}/{version}"
            info = fs.get_file_info(ver_path)
            if info.type == FileType.NotFound:
                print(f"[*] Creating new version {version}")
            else:
                print(f"[*] Resolved existing version {version}")
            return ver_path

        # 3. Find latest version (List directories)
        selector = pafs.FileSelector(path, recursive=False)
        entries = fs.get_file_info(selector)

        versions = [
            e.base_name
            for e in entries
            if e.type == FileType.Directory or (e.type == FileType.File and "/" in e.path)
        ]

        if not versions:
            print("[!] No versions found, defaulting to 0.0.1")
            return f"{path}/0.0.1"

        latest = sorted(versions, reverse=True)[0]
        print(f"[*] Resolved latest version {latest}")
        return f"{path}/{latest}"

import contextlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import pyarrow.fs as pafs  # The Native FS
from pyarrow.fs import FileSystem, FileType

from dreadnode.api import ApiClient
from dreadnode.api.models import (
    CreateDatasetRequest,
    DatasetDownloadRequest,
    DatasetUploadCompleteRequest,
)
from dreadnode.constants import (
    DEFAULT_LOCAL_STORAGE_DIR,
    DEFAULT_REMOTE_STORAGE_PREFIX,
    FS_CREDENTIAL_REFRESH_BUFFER,
    MANIFEST_FILE,
    METADATA_FILE,
)
from dreadnode.logging_ import console as logging_console
from dreadnode.logging_ import print_info
from dreadnode.storage.datasets.metadata import DatasetMetadata, VersionInfo
from dreadnode.util import resolve_endpoint


class DatasetManager:
    """
    DatasetManager manages dataset storage and retrieval for remote and local files.

    It is responsible for:
    - Caching datasets locally
    - Interfacing with remote storage (e.g., S3)
    - Making sure paths are resolved correctly and the correct filesystem is used
    - Handling authentication for remote storage access
    - Communicating with the Dreadnode API for dataset uploads/downloads
    """

    organization: str | None = None
    organization_id: UUID | None = None
    _instance: "DatasetManager | None" = None
    _api: ApiClient | None = None

    # Auth State
    _credentials_expiry: datetime | None = None
    _cached_s3_fs: pafs.S3FileSystem | None = None

    remote_prefix: str = DEFAULT_REMOTE_STORAGE_PREFIX
    cache_root: Path = Path(DEFAULT_LOCAL_STORAGE_DIR)

    def __new__(cls) -> "DatasetManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def get_version_from_path(path: str) -> str | None:
        """
        Extracts version string from a given path.
        Assumes version is the last part of the path.
        Example: /path/to/dataset/1.0.0 -> "1.0.0"
        """

        resolved_path = Path(path).resolve()
        version_candidate = resolved_path.name

        try:
            _ = VersionInfo.from_string(version_candidate)

        except ValueError:
            return None

        return version_candidate

    @classmethod
    def configure(
        cls,
        api: ApiClient | None = None,
        organization: str | None = None,
        organization_id: UUID | None = None,
    ) -> "DatasetManager":
        instance = cls()
        instance._api = api
        instance.organization = organization
        instance.organization_id = organization_id
        return instance

    def metadata_exists(self, clean_path: str, fs: FileSystem) -> bool:
        """
        Checks if metadata file exists in the local cache.
        """

        target_path = f"{clean_path}/{METADATA_FILE}"

        info = fs.get_file_info(target_path)

        return info.type != FileType.NotFound

    def manifest_exists(self, clean_path: str, fs: FileSystem) -> bool:
        """
        Checks if manifest file exists in the local cache.
        """
        # if "://" not in path or path.startswith("file://"):
        #     target_path = Path(f"{path}/{MANIFEST_FILE}").resolve()

        #     return target_path.exists()

        # fs, clean_path = self.get_fs_and_path(path)
        target_path = f"{clean_path}/{MANIFEST_FILE}"

        info = fs.get_file_info(target_path)

        return info.type != FileType.NotFound

    def check_cache(self, uri: str, version: str | None = None) -> bool:
        cache_base = self.cache_root / "datasets"
        cache_base = cache_base.resolve()

        clean_uri = uri.lstrip("/")

        if version:
            clean_uri = f"{clean_uri}/{version}"

        target_path = (cache_base / clean_uri).resolve()

        if not str(target_path).startswith(str(cache_base)):
            return False

        return target_path.exists()

    def get_cache_save_uri(self, metadata: DatasetMetadata, *, with_version: bool = True) -> str:
        """
        Constructs the full local cache path.
        Example: /home/user/.dreadnode/datasets/main/my-dataset/1.0.0
        """
        dataset_uri = Path(f"{self.cache_root}/datasets/{metadata.organization}/{metadata.name}")
        if with_version:
            dataset_uri = dataset_uri / metadata.version

        dataset_uri = dataset_uri.resolve()

        return str(dataset_uri)

    def get_latest_cache_save_uri(self, metadata: DatasetMetadata) -> str | None:
        """
        Constructs the full local cache path to the latest version.
        Example: /home/user/.dreadnode/datasets/main/my-dataset/latest-version
        """

        dataset_uri = Path(
            f"{self.cache_root}/datasets/{metadata.organization}/{metadata.name}"
        ).resolve()

        latest = self.resolve_latest_local_version(metadata)

        if latest is None:
            return None

        return str(dataset_uri / latest)

    def get_cache_load_uri(
        self, uri: str, version: str | None = None, fs: FileSystem = None
    ) -> str:
        """
        Constructs the full local cache path.
        Example: /home/user/.dreadnode/datasets/main/my-dataset
        """

        if version:
            uri = f"{uri}/{version}"
            dataset_uri = Path(f"{self.cache_root}/datasets/{uri}").resolve()

            if dataset_uri.exists():
                return str(dataset_uri)

        dataset_uri = Path(f"{self.cache_root}/datasets/{uri}").resolve()

        latest = self.resolve_latest_local_version(str(dataset_uri), fs)

        return str(dataset_uri / latest)

    def get_remote_save_uri(self, metadata: DatasetMetadata) -> tuple[UUID, str]:
        """
        Constructs the full remote storage URI.
        Example: dreadnode://datasets/main/my-dataset
        """

        if not self._api:
            raise ValueError("No client configured")

        upload_request = CreateDatasetRequest(
            org_key=metadata.organization,
            key=metadata.name,
            version=metadata.version,
            tags=metadata.tags,
        )

        response = self._api.create_dataset(request=upload_request)
        dataset_id = response.dataset_id
        user_data_access_response = response.user_data_access_response
        self._cached_s3_fs = pafs.S3FileSystem(
            access_key=user_data_access_response.access_key_id,
            secret_key=user_data_access_response.secret_access_key,
            session_token=user_data_access_response.session_token,
            endpoint_override=resolve_endpoint(user_data_access_response.endpoint),
            region=user_data_access_response.region,
            check_directory_existence_before_creation=True,
        )
        self._credentials_expiry = user_data_access_response.expiration
        return dataset_id, user_data_access_response.uri

    def remote_save_complete(self, dataset_id: str, *, complete: bool) -> None:
        """
        Notifies the API that the remote upload is complete.
        """

        if not self._api:
            raise ValueError("No client configured")

        request = DatasetUploadCompleteRequest(dataset_id=dataset_id, complete=complete)

        self._api.upload_complete(request=request)

    def get_remote_load_uri(self, uri: str, version: str | None = None) -> str | None:
        """
        Requests the download URI for a dataset from the API.
        """

        request = DatasetDownloadRequest(
            dataset_uri=uri,
            version=version or "latest",
        )
        try:
            response = self._api.download_dataset(request)

            print_info(f"[*] Download URI: {response.uri}")
        except RuntimeError as e:
            if "404: Dataset not found" in str(e):
                logging_console.print(f"[*] Dataset not found: [green]{uri}[/green]")
                return None
            raise

        self._cached_s3_fs = pafs.S3FileSystem(
            access_key=response.access_key_id,
            secret_key=response.secret_access_key,
            session_token=response.session_token,
            endpoint_override=resolve_endpoint(response.endpoint),
            region=response.region,
            check_directory_existence_before_creation=True,
        )
        self._credentials_expiry = response.expiration
        return response.uri

    def get_s3_config(self, dataset_id: UUID, version: str | None = None) -> dict[str, Any]:
        """
        Translates your UserDataCredentials into PyArrow S3 arguments.
        """
        if not self._api:
            raise ValueError("No client configured")

        creds = self._api.get_dataset_access_credentials(
            dataset_id=dataset_id, version=version or "latest"
        )
        self._credentials_expiry = creds.expiration
        resolved_endpoint = resolve_endpoint(creds.endpoint)

        # Mapping UserDataCredentials -> PyArrow S3FileSystem kwargs
        return {
            "access_key": creds.access_key_id,
            "secret_key": creds.secret_access_key,
            "session_token": creds.session_token,
            "endpoint_override": resolved_endpoint,
            "region": creds.region,
            "check_directory_existence_before_creation": True,
        }

    def needs_refresh(self) -> bool:
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

        if "://" not in uri or uri.startswith("file://"):
            fs = pafs.LocalFileSystem()
            path = uri.replace("file://", "")
            return fs, path

        try:
            _, path_body = uri.split("://", 1)
            path_body = path_body.rstrip("/")
        except ValueError:
            return pafs.LocalFileSystem(), uri

        if self._cached_s3_fs is None or self.needs_refresh():
            try:
                # Try to extract dataset ID from URI which expect is of the form dn://<user bucket>/<org_id>/datasets/<dataset_id>/<version>
                dataset_id = UUID(path_body.split("/")[3])
                config = self.get_s3_config(dataset_id=dataset_id)
                self._cached_s3_fs = pafs.S3FileSystem(**config)
            except ValueError:
                logging_console.print(f"[red]Invalid dataset ID in URI: [green]{uri}[/green][/red]")
                raise
            except Exception as e:
                logging_console.print(f"Auth failed: {e}")
                raise

        return self._cached_s3_fs, path_body

    def resolve_latest_local_version(self, metadata: DatasetMetadata) -> str | None:
        """
        PyArrow Native implementation of version resolution.
        """
        uri = self.get_cache_save_uri(metadata, with_version=False)

        selector = pafs.FileSelector(uri, recursive=False)
        fs = pafs.LocalFileSystem()
        entries = fs.get_file_info(selector)

        versions = [
            e.base_name
            for e in entries
            if e.type == FileType.Directory or (e.type == FileType.File and "/" in e.path)
        ]
        if not versions:
            return None
        latest = sorted(versions, reverse=True)[0]
        # ensure it's a valid version
        try:
            _ = VersionInfo.from_string(latest)
        except ValueError as e:
            raise ValueError(f"No valid versions found in {uri}") from e
        return latest

    def ensure_dir(self, fs: FileSystem, path: str) -> None:
        """
        Creates directory if local. Skips if S3.
        """
        if fs.type_name == "s3":
            return
        with contextlib.suppress(OSError):
            fs.create_dir(path, recursive=True)

    def delete_remote_dataset_record(self, dataset_id_or_key: UUID | str) -> None:
        """
        Deletes a remote dataset via the API.
        """

        if not self._api:
            raise ValueError("No client configured")

        self._api.delete_dataset(dataset_id_or_key=dataset_id_or_key)

from dataclasses import dataclass
from typing import Literal

from fsspec import AbstractFileSystem

StorageProvider = Literal["s3", "r2", "minio", "local"]


@dataclass
class S3Credentials:
    """AWS S3 / S3-compatible (R2, MinIO) credentials."""

    access_key_id: str
    secret_access_key: str
    session_token: str | None = None
    endpoint_url: str | None = None
    region: str | None = None

    def to_storage_options(self) -> dict:
        opts = {
            "key": self.access_key_id,
            "secret": self.secret_access_key,
        }
        if self.session_token:
            opts["token"] = self.session_token
        if self.endpoint_url or self.region:
            opts["client_kwargs"] = {}
            if self.endpoint_url:
                opts["client_kwargs"]["endpoint_url"] = self.endpoint_url
            if self.region:
                opts["client_kwargs"]["region_name"] = self.region
        return opts


@dataclass
class MinioCredentials:
    """MinIO credentials."""

    access_key_id: str
    secret_access_key: str
    session_token: str | None = None
    endpoint_url: str | None = None
    region: str | None = None

    def to_storage_options(self) -> dict:
        opts = {
            "key": self.access_key_id,
            "secret": self.secret_access_key,
        }
        if self.session_token:
            opts["token"] = self.session_token
        if self.endpoint_url or self.region:
            opts["client_kwargs"] = {}
            if self.endpoint_url:
                opts["client_kwargs"]["endpoint_url"] = self.endpoint_url
            if self.region:
                opts["client_kwargs"]["region_name"] = self.region
        return opts


def from_provider(
    provider: StorageProvider,
    credentials: dict | None = None,
) -> AbstractFileSystem:
    """Create filesystem from provider and credentials."""
    if provider == "local":
        from fsspec.implementations.local import LocalFileSystem

        return LocalFileSystem()

    if credentials is None:
        raise ValueError(f"Credentials required for provider: {provider}")

    if provider in ("s3", "r2"):
        from s3fs import S3FileSystem

        creds = S3Credentials(**credentials)
        return S3FileSystem(**creds.to_storage_options())

    if provider == "minio":
        from s3fs import S3FileSystem

        creds = MinioCredentials(**credentials)
        return S3FileSystem(**creds.to_storage_options())

    raise ValueError(f"Unsupported storage provider: {provider}")

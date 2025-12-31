import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import fsspec
from fsspec import AbstractFileSystem
from ulid import ULID

from dreadnode.core.api.models import UserDataCredentials
from dreadnode.core.api.session import Session
from dreadnode.core.settings import FS_CREDENTIAL_REFRESH_BUFFER
from dreadnode.core.storage.providers import StorageProvider, from_provider

PackageType = Literal["datasets", "models", "toolsets", "agents", "environments"]


class Storage:
    """Storage manager for local and remote storage.

    Directory structure:
        ~/.dreadnode/
          <org_key>/
            packages/
              datasets/
              agents/
              models/
              toolsets/
              environments/
            cas/
              sha256/
                ab/cd/...
            workspaces/
              <workspace_key>/
                <project_key>/
                  <run_id>/
                    spans.jsonl
                    metrics.jsonl
    """

    def __init__(
        self,
        session: Session | None = None,
        cache: Path | None = None,
        registry: str | None = None,
        provider: StorageProvider | None = None,
    ):
        """Create storage for an organization.

        Args:
            session: Session with org/workspace/project context.
            cache: Root cache directory. Defaults to ~/.dreadnode.
            registry: PyPI registry URL for pull operations.
            provider: Storage provider for remote operations.
        """
        self.session = session
        self.cache = cache
        self.registry = registry
        self.provider = provider

        self._remote_fs: AbstractFileSystem | None = None
        self._local_fs: AbstractFileSystem | None = None
        self._credentials: UserDataCredentials | None = None

        self.expiration: datetime | None = None

    def _get_filesystem(self) -> AbstractFileSystem:
        """Get filesystem, refreshing credentials if needed.

        Falls back to local filesystem if no remote storage is configured.
        """
        if self._remote_fs is not None and not self.expired():
            return self._remote_fs

        # Explicit local provider
        if self.provider == "local":
            self._remote_fs = from_provider("local")
            return self._remote_fs

        # No session means local-only mode
        if self.session is None:
            self._remote_fs = from_provider("local")
            return self._remote_fs

        # Try to get remote credentials
        try:
            self._credentials = self._refresh_credentials()

            # Determine provider if not explicitly set
            provider = self.provider
            if provider is None:
                # MinIO if endpoint is set (local dev), otherwise S3
                provider = "minio" if self._credentials.endpoint else "s3"

            # Convert credentials to dict
            creds_dict = {
                "access_key_id": self._credentials.access_key_id,
                "secret_access_key": self._credentials.secret_access_key,
                "session_token": self._credentials.session_token,
                "endpoint_url": self._credentials.endpoint,
                "region": self._credentials.region,
            }

            self._remote_fs = from_provider(provider, creds_dict)
            self.expiration = self._credentials.expiration

        except Exception:
            # Fall back to local filesystem
            self._remote_fs = from_provider("local")

        return self._remote_fs

    def _refresh_credentials(self) -> UserDataCredentials:
        if self.session is None:
            raise RuntimeError("No session configured for remote storage")
        return self.session.api.get_storage_access(workspace_id=self.session.workspace.id)

    @property
    def remote_prefix(self) -> str:
        """Get the remote storage prefix from credentials."""
        if self._credentials is None:
            # Force credential refresh to get prefix
            self._get_filesystem()
        if self._credentials is None:
            raise RuntimeError("No credentials available for remote storage")
        return self._credentials.prefix

    # Paths

    @property
    def org_path(self) -> Path:
        """Path to org directory using org key."""
        if self.session is None:
            return self.cache / "_local"
        return self.cache / self.session.organization.key

    @property
    def packages_path(self) -> Path:
        """Path to packages directory."""
        return self.org_path / "packages"

    @property
    def cas_path(self) -> Path:
        """Path to CAS directory (org-scoped)."""
        return self.org_path / "cas"

    @property
    def workspaces_path(self) -> Path:
        """Path to workspaces directory."""
        return self.org_path / "workspaces"

    @property
    def workspace_path(self) -> Path:
        """Path to workspace directory using workspace key."""
        if self.session is None:
            return self.workspaces_path / "_local"
        return self.workspaces_path / self.session.workspace.key

    @property
    def artifacts_path(self) -> Path:
        """Path to workspace-level artifact CAS."""
        return self.workspace_path / "artifacts"

    def artifact_blob_path(self, oid: str) -> Path:
        """Path to artifact blob in workspace CAS."""
        algo, hash_val = oid.split(":", 1)
        return self.artifacts_path / algo / hash_val[:2] / hash_val[2:4] / hash_val

    def remote_artifact_path(self, oid: str) -> str:
        """Remote path for artifact blob."""
        algo, hash_val = oid.split(":", 1)
        bucket = self._credentials.bucket if self._credentials else "user-data"
        return f"{bucket}/{self.remote_prefix}/artifacts/{algo}/{hash_val[:2]}/{hash_val[2:4]}/{hash_val}"

    @property
    def project_path(self) -> Path:
        """Path to project directory using project key."""
        if self.session is None:
            return self.workspace_path / "_local"
        return self.workspace_path / self.session.project.key

    def run_path(self, run_id: str | ULID) -> Path:
        """Path to run directory for trace data."""
        return self.project_path / str(run_id)

    def trace_path(self, run_id: str | ULID, filename: str = "spans.jsonl") -> Path:
        """Path to trace file within a run directory.

        Args:
            run_id: The run identifier.
            filename: Full filename with extension (e.g., 'spans.jsonl', 'spans.parquet').

        Returns:
            Full path to the trace file.
        """
        run_dir = self.run_path(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir / filename

    def expired(self) -> bool:
        if self.expiration is None:
            return False
        now = datetime.now(timezone.utc)
        expiry = self.expiration
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        return (expiry - now).total_seconds() < FS_CREDENTIAL_REFRESH_BUFFER

    def blob_path(self, oid: str) -> Path:
        """Path to blob in CAS."""
        algo, hash_val = oid.split(":", 1)
        return self.cas_path / algo / hash_val[:2] / hash_val[2:4] / hash_val

    def remote_blob_path(self, oid: str) -> str:
        """Remote path for blob (includes bucket for s3fs)."""
        algo, hash_val = oid.split(":", 1)
        bucket = self._credentials.bucket if self._credentials else "user-data"
        return f"{bucket}/{self.remote_prefix}/cas/{algo}/{hash_val[:2]}/{hash_val[2:4]}/{hash_val}"

    def package_path(
        self,
        package_type: PackageType,
        name: str,
        version: str | None = None,
    ) -> Path:
        """Path to package directory.

        Returns: ~/.dreadnode/<org_key>/packages/<package_type>/<name>/[version/]
        """
        base = self.packages_path / package_type / name
        if version:
            return base / version
        return base

    def package_source_path(
        self,
        package_type: PackageType,
        name: str,
    ) -> Path:
        """Path for package source code during development.

        Returns: ~/.dreadnode/<org_key>/packages/<package_type>/<name>/
        """
        path = self.packages_path / package_type / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def manifest_path(
        self,
        package_type: PackageType,
        name: str,
        version: str,
    ) -> Path:
        """Path to manifest.json."""
        return self.package_path(package_type, name, version) / "manifest.json"

    # Blob operations

    def blob_exists(self, oid: str) -> bool:
        """Check if blob exists in local CAS."""
        return self.blob_path(oid).exists()

    def remote_blob_exists(self, oid: str) -> bool:
        """Check if blob exists in remote storage."""
        fs = self._get_filesystem()
        return fs.exists(self.remote_blob_path(oid))

    def store_blob(self, oid: str, source: Path) -> Path:
        """Store blob in local CAS."""
        dest = self.blob_path(oid)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
        return dest

    def get_blob(self, oid: str) -> Path:
        """Get blob from local CAS."""
        path = self.blob_path(oid)
        if not path.exists():
            raise FileNotFoundError(f"Blob not found: {oid}")
        return path

    # Artifact operations

    def store_artifact(self, source: Path, *, upload: bool = True) -> str:
        """Store artifact in workspace CAS and optionally upload to remote.

        Args:
            source: Path to the file to store.
            upload: Whether to upload to remote storage immediately.

        Returns:
            The oid (sha256:<hash>) of the stored artifact.
        """
        # Compute hash
        file_hash = hash_file(source)
        oid = f"sha256:{file_hash}"

        # Store locally
        dest = self.artifact_blob_path(oid)
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(source.read_bytes())

        # Upload to remote
        if upload:
            self.upload_artifact(oid)

        return oid

    def upload_artifact(self, oid: str) -> None:
        """Upload artifact from workspace CAS to remote storage."""
        source = self.artifact_blob_path(oid)
        if not source.exists():
            raise FileNotFoundError(f"Artifact not found locally: {oid}")

        fs = self._get_filesystem()
        if self._is_local_filesystem():
            return  # No remote upload needed for local filesystem

        remote_path = self.remote_artifact_path(oid)
        if not fs.exists(remote_path):
            fs.put_file(str(source), remote_path)

    def get_artifact(self, oid: str) -> Path:
        """Get artifact from workspace CAS, downloading if needed."""
        local_path = self.artifact_blob_path(oid)
        if local_path.exists():
            return local_path

        # Try to download from remote
        fs = self._get_filesystem()
        if self._is_local_filesystem():
            raise FileNotFoundError(f"Artifact not found: {oid}")

        remote_path = self.remote_artifact_path(oid)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        fs.get_file(remote_path, str(local_path))

        # Verify hash
        algo, expected = oid.split(":", 1)
        actual = hash_file(local_path)
        if actual != expected:
            local_path.unlink()
            raise ValueError(f"Artifact hash mismatch: expected {expected}, got {actual}")

        return local_path

    # Batch operations

    def hash_files(self, paths: list[Path], algo: str = "sha256") -> dict[Path, str]:
        """Compute hashes for multiple files.

        Args:
            paths: Files to hash.
            algo: Hash algorithm.

        Returns:
            Mapping of path to hash.
        """
        return {p: hash_file(p, algo) for p in paths}

    def upload_blobs(
        self,
        files: dict[Path, str],
        *,
        skip_existing: bool = True,
    ) -> tuple[int, int]:
        """Upload multiple blobs to remote storage.

        Args:
            files: Mapping of local path to oid.
            skip_existing: Skip blobs that already exist remotely.

        Returns:
            Tuple of (uploaded_count, skipped_count).
        """
        fs = self._get_filesystem()
        is_local = self._is_local_filesystem()

        sources: list[str] = []
        targets: list[str] = []
        skipped = 0

        for local_path, oid in files.items():
            # Use local blob_path for local filesystem, remote path for S3/MinIO
            if is_local:
                target_path = str(self.blob_path(oid))
            else:
                target_path = self.remote_blob_path(oid)

            if skip_existing and fs.exists(target_path):
                skipped += 1
                continue

            sources.append(str(local_path))
            targets.append(target_path)

        if sources:
            # For local filesystem, ensure parent directories exist
            if is_local:
                for target in targets:
                    Path(target).parent.mkdir(parents=True, exist_ok=True)
            fs.put(sources, targets)

        return len(sources), skipped

    def _is_local_filesystem(self) -> bool:
        """Check if the current filesystem is local."""
        from fsspec.implementations.local import LocalFileSystem

        fs = self._get_filesystem()
        return isinstance(fs, LocalFileSystem)

    def download_blobs(
        self,
        oids: list[str],
        *,
        skip_existing: bool = True,
    ) -> tuple[int, int]:
        """Download multiple blobs from remote storage.

        Args:
            oids: Object IDs to download.
            skip_existing: Skip blobs that already exist locally.

        Returns:
            Tuple of (downloaded_count, skipped_count).
        """
        fs = self._get_filesystem()

        sources: list[str] = []
        targets: list[str] = []
        skipped = 0

        for oid in oids:
            local_path = self.blob_path(oid)

            if skip_existing and local_path.exists():
                skipped += 1
                continue

            local_path.parent.mkdir(parents=True, exist_ok=True)
            sources.append(self.remote_blob_path(oid))
            targets.append(str(local_path))

        if sources:
            fs.get(sources, targets)

            # Verify hashes
            for oid, target in zip(oids, targets, strict=True):
                if target in targets:  # Only verify downloaded files
                    algo, expected = oid.split(":", 1)
                    actual = hash_file(Path(target), algo)
                    if actual != expected:
                        Path(target).unlink()
                        raise ValueError(f"Hash mismatch for {oid}")

        return len(sources), skipped

    # Manifest operations

    def store_manifest(
        self,
        package_type: PackageType,
        name: str,
        version: str,
        content: str,
    ) -> Path:
        """Store manifest.json."""
        path = self.manifest_path(package_type, name, version)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def get_manifest(
        self,
        package_type: PackageType,
        name: str,
        version: str,
    ) -> str:
        """Get manifest.json content."""
        return self.manifest_path(package_type, name, version).read_text()

    def manifest_exists(
        self,
        package_type: PackageType,
        name: str,
        version: str,
    ) -> bool:
        """Check if manifest exists."""
        return self.manifest_path(package_type, name, version).exists()

    # Version operations

    def list_versions(
        self,
        package_type: PackageType,
        name: str,
    ) -> list[str]:
        """List available versions."""
        base = self.package_path(package_type, name)
        if not base.exists():
            return []
        return sorted([v.name for v in base.iterdir() if v.is_dir()], reverse=True)

    def latest_version(
        self,
        package_type: PackageType,
        name: str,
    ) -> str | None:
        """Get latest version."""
        versions = self.list_versions(package_type, name)
        return versions[0] if versions else None

    # Single blob remote operations (for backwards compat)

    def download_blob(self, oid: str) -> Path:
        """Download blob from remote to local CAS."""
        dest = self.blob_path(oid)
        if dest.exists():
            return dest

        fs = self._get_filesystem()
        remote_path = self.remote_blob_path(oid)

        dest.parent.mkdir(parents=True, exist_ok=True)
        fs.get_file(remote_path, str(dest))

        # Verify
        algo, expected = oid.split(":", 1)
        actual = hash_file(dest, algo)
        if actual != expected:
            dest.unlink()
            raise ValueError(f"Hash mismatch: expected {expected}, got {actual}")

        return dest

    def upload_blob(self, oid: str) -> None:
        """Upload blob from local CAS to remote."""
        source = self.blob_path(oid)
        if not source.exists():
            raise FileNotFoundError(f"Blob not found: {oid}")

        fs = self._get_filesystem()
        remote_path = self.remote_blob_path(oid)

        fs.put_file(str(source), remote_path)

    # API operations

    async def upload_package(self, package_path: Path, repo_id: ULID) -> None:
        """Upload wheel to registry."""
        if self.session is None:
            raise RuntimeError("API client not configured")
        self.session.api.upload_package(repo_id=repo_id, package_path=package_path)

    def upload_complete(self, repo_id: ULID) -> None:
        """Notify API that upload is complete."""
        if self.session is None:
            raise RuntimeError("Session not configured")
        self.session.api.upload_complete(repo_id=repo_id)

    # URI resolution

    def resolve(self, uri: str, **storage_options) -> tuple[AbstractFileSystem, str]:
        """Resolve URI to filesystem and path."""
        fs, path = fsspec.url_to_fs(uri, **storage_options)
        return fs, path


def hash_file(path: Path, algo: str = "sha256") -> str:
    """Compute hash of file."""
    hasher = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

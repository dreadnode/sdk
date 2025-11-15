"""
Enhanced OCI Dataset Storage System
Author: Dreadnode AI Infrastructure Team
License: MIT

Features:
- Smart local caching with automatic sync
- PyArrow-optimized data operations
- Incremental dataset updates with change tracking
- Support for partitioned datasets
- Compression and deduplication
- Volume-based packaging for containerized workflows
"""

import hashlib
import json
import logging
import shutil
import sqlite3
import tarfile
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import oras.client
import oras.provider
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    """Comprehensive dataset metadata."""

    org: str
    name: str
    version: str
    format: str  # parquet, csv, json, jsonl, text
    created_at: str
    updated_at: str
    size_bytes: int
    row_count: int | None
    column_count: int | None
    schema: dict | None
    compression: str
    checksum: str
    parent_version: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    partitioned: bool = False
    partition_columns: list[str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetMetadata":
        return cls(**data)


class DatasetCache:
    """
    Manages local dataset cache with SQLite for metadata tracking.
    """

    def __init__(self, cache_root: str = "~/.dreadnode"):
        self.cache_root = Path(cache_root).expanduser()
        self.datasets_dir = self.cache_root / "datasets"
        self.db_path = self.cache_root / "cache.db"
        self.config_path = self.cache_root / "config.json"

        # Create directories
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(exist_ok=True)

        # Initialize database
        self._init_db()

        # Load config
        self.config = self._load_config()

    def _init_db(self):
        """Initialize SQLite database for metadata."""
        with self._db_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    org TEXT,
                    name TEXT,
                    version TEXT,
                    format TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    size_bytes INTEGER,
                    row_count INTEGER,
                    checksum TEXT,
                    metadata_json TEXT,
                    PRIMARY KEY (org, name, version)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS sync_state (
                    org TEXT,
                    name TEXT,
                    version TEXT,
                    last_pulled TEXT,
                    last_pushed TEXT,
                    remote_digest TEXT,
                    PRIMARY KEY (org, name, version)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_org_name 
                ON datasets(org, name)
            """)

    @contextmanager
    def _db_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _load_config(self) -> dict:
        """Load cache configuration."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return json.load(f)

        # Default config
        config = {
            "cache_size_limit_gb": 100,
            "auto_cleanup": True,
            "compression_default": "zstd",
            "registry_default": "localhost:5000",
        }

        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

        return config

    def get_dataset_path(self, org: str, name: str, version: str) -> Path:
        """Get local path for dataset."""
        return self.datasets_dir / org / name / version

    def get_metadata(self, org: str, name: str, version: str) -> DatasetMetadata | None:
        """Retrieve dataset metadata from cache."""
        with self._db_connection() as conn:
            cursor = conn.execute(
                "SELECT metadata_json FROM datasets WHERE org=? AND name=? AND version=?",
                (org, name, version),
            )
            row = cursor.fetchone()

            if row:
                return DatasetMetadata.from_dict(json.loads(row["metadata_json"]))

        return None

    def save_metadata(self, metadata: DatasetMetadata):
        """Save dataset metadata to cache."""
        with self._db_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO datasets
                (org, name, version, format, created_at, updated_at, 
                 size_bytes, row_count, checksum, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metadata.org,
                    metadata.name,
                    metadata.version,
                    metadata.format,
                    metadata.created_at,
                    metadata.updated_at,
                    metadata.size_bytes,
                    metadata.row_count,
                    metadata.checksum,
                    json.dumps(metadata.to_dict()),
                ),
            )

    def update_sync_state(
        self,
        org: str,
        name: str,
        version: str,
        action: str,  # 'pull' or 'push'
        digest: str | None = None,
    ):
        """Update sync state after push/pull."""
        with self._db_connection() as conn:
            if action == "pull":
                conn.execute(
                    """
                    INSERT OR REPLACE INTO sync_state
                    (org, name, version, last_pulled, remote_digest)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (org, name, version, datetime.utcnow().isoformat(), digest),
                )
            elif action == "push":
                conn.execute(
                    """
                    INSERT OR REPLACE INTO sync_state
                    (org, name, version, last_pushed, remote_digest)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (org, name, version, datetime.utcnow().isoformat(), digest),
                )

    def list_versions(self, org: str, name: str) -> list[str]:
        """List all cached versions of a dataset."""
        with self._db_connection() as conn:
            cursor = conn.execute(
                "SELECT version FROM datasets WHERE org=? AND name=? ORDER BY created_at DESC",
                (org, name),
            )
            return [row["version"] for row in cursor.fetchall()]

    def compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


class DatasetOCIStorage:
    """
    Enhanced OCI storage for datasets with caching and versioning.
    """

    def __init__(
        self,
        registry: str = "localhost:5000",
        insecure: bool = False,
        username: str | None = None,
        password: str | None = None,
        cache_root: str = "~/.dreadnode",
    ):
        self.registry = registry
        self.cache = DatasetCache(cache_root)
        self.client = oras.client.OrasClient(insecure=insecure)

        # Authenticate if credentials provided
        if username and password:
            self.client.login(username=username, password=password, insecure=insecure)
            logger.info(f"Authenticated to registry: {registry}")

    def _compute_dataset_checksum(self, dataset_path: Path) -> str:
        """Compute checksum for entire dataset directory."""
        checksums = []

        for file_path in sorted(dataset_path.rglob("*")):
            if file_path.is_file():
                checksums.append(self.cache.compute_checksum(file_path))

        combined = "".join(checksums).encode("utf-8")
        return hashlib.sha256(combined).hexdigest()

    def _get_arrow_schema(self, file_path: Path, format: str) -> dict | None:
        """Extract Arrow schema from data file."""
        try:
            if format == "parquet":
                schema = pq.read_schema(file_path)
                return {
                    "fields": [{"name": field.name, "type": str(field.type)} for field in schema]
                }
            if format == "csv":
                # Read first few rows to infer schema
                df = pd.read_csv(file_path, nrows=100)
                table = pa.Table.from_pandas(df)
                return {
                    "fields": [
                        {"name": field.name, "type": str(field.type)} for field in table.schema
                    ]
                }
        except Exception as e:
            logger.warning(f"Could not extract schema: {e}")

        return None

    def _count_rows_parquet(self, file_path: Path) -> int:
        """Count rows in parquet file efficiently."""
        try:
            parquet_file = pq.ParquetFile(file_path)
            return parquet_file.metadata.num_rows
        except Exception as e:
            logger.warning(f"Could not count rows: {e}")
            return 0

    # ========================
    # USE CASE 1: Import and Push Dataset from External Location
    # ========================

    def import_dataset(
        self,
        source_path: str,
        org: str,
        dataset_name: str,
        version: str,
        format: str = "parquet",
        compression: str = "zstd",
        description: str | None = None,
        tags: list[str] | None = None,
        push_after_import: bool = True,
        convert_to_parquet: bool = False,
        partition_columns: list[str] | None = None,
    ) -> DatasetMetadata:
        """
        Import dataset from external location to cache, then optionally push to registry.

        Use Case 1: Dataset lives on disk outside of ~/.dreadnode/datasets

        Args:
            source_path: Path to source dataset (file or directory)
            org: Organization name
            dataset_name: Dataset name
            version: Version string (e.g., 'v1.0.0', '2024-01-15')
            format: Data format ('parquet', 'csv', 'json', 'jsonl', 'text')
            compression: Compression codec ('zstd', 'snappy', 'gzip', 'none')
            description: Optional dataset description
            tags: Optional tags for categorization
            push_after_import: Whether to push to registry after import
            convert_to_parquet: Convert CSV/JSON to Parquet for efficiency
            partition_columns: Columns to partition by (parquet only)

        Returns:
            DatasetMetadata object

        Example:
            >>> storage = DatasetOCIStorage()
            >>> metadata = storage.import_dataset(
            ...     source_path='/data/my_dataset.csv',
            ...     org='dreadnode',
            ...     dataset_name='user_events',
            ...     version='v1.0.0',
            ...     format='csv',
            ...     convert_to_parquet=True,
            ...     push_after_import=True
            ... )
        """
        source_path = Path(source_path).expanduser().resolve()

        if not source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {source_path}")

        logger.info(f"Importing dataset from {source_path}")

        # Create cache directory
        cache_path = self.cache.get_dataset_path(org, dataset_name, version)
        cache_path.mkdir(parents=True, exist_ok=True)
        data_path = cache_path / "data"
        data_path.mkdir(exist_ok=True)

        # Handle different source types
        row_count = None
        column_count = None
        schema_dict = None
        size_bytes = 0

        if source_path.is_file():
            # Single file
            if convert_to_parquet and format in ("csv", "json", "jsonl"):
                logger.info(f"Converting {format} to parquet with {compression} compression")
                output_file = data_path / "data.parquet"

                # Read source data
                if format == "csv":
                    table = pa.csv.read_csv(source_path)
                elif format == "json":
                    df = pd.read_json(source_path)
                    table = pa.Table.from_pandas(df)
                elif format == "jsonl":
                    df = pd.read_json(source_path, lines=True)
                    table = pa.Table.from_pandas(df)

                # Write parquet
                if partition_columns:
                    ds.write_dataset(
                        table,
                        data_path,
                        format="parquet",
                        partitioning=ds.partitioning(
                            pa.schema([table.schema.field(col) for col in partition_columns]),
                            flavor="hive",
                        ),
                        compression=compression,
                    )
                    format = "parquet"
                    row_count = table.num_rows
                    column_count = table.num_columns
                else:
                    pq.write_table(table, output_file, compression=compression, version="2.6")
                    format = "parquet"
                    row_count = table.num_rows
                    column_count = table.num_columns

                schema_dict = self._get_arrow_schema(output_file, "parquet")
                size_bytes = output_file.stat().st_size
            else:
                # Copy file as-is
                dest_file = data_path / source_path.name
                shutil.copy2(source_path, dest_file)
                size_bytes = dest_file.stat().st_size

                if format == "parquet":
                    row_count = self._count_rows_parquet(dest_file)
                    parquet_file = pq.ParquetFile(dest_file)
                    column_count = len(parquet_file.schema)
                    schema_dict = self._get_arrow_schema(dest_file, format)

        elif source_path.is_dir():
            # Directory - copy entire structure
            for item in source_path.rglob("*"):
                if item.is_file():
                    rel_path = item.relative_to(source_path)
                    dest_file = data_path / rel_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest_file)
                    size_bytes += dest_file.stat().st_size

            # Try to read as Arrow dataset
            if format == "parquet":
                try:
                    dataset = ds.dataset(data_path, format="parquet")
                    row_count = dataset.count_rows()
                    column_count = len(dataset.schema)
                    schema_dict = {
                        "fields": [
                            {"name": field.name, "type": str(field.type)}
                            for field in dataset.schema
                        ]
                    }
                except Exception as e:
                    logger.warning(f"Could not read as Arrow dataset: {e}")

        # Compute checksum
        checksum = self._compute_dataset_checksum(data_path)

        # Create metadata
        metadata = DatasetMetadata(
            org=org,
            name=dataset_name,
            version=version,
            format=format,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            size_bytes=size_bytes,
            row_count=row_count,
            column_count=column_count,
            schema=schema_dict,
            compression=compression,
            checksum=checksum,
            description=description,
            tags=tags or [],
            partitioned=partition_columns is not None,
            partition_columns=partition_columns,
        )

        # Save metadata
        metadata_file = cache_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        self.cache.save_metadata(metadata)

        logger.info(f"Dataset imported to cache: {cache_path}")
        logger.info(f"Size: {size_bytes / 1024 / 1024:.2f} MB, Rows: {row_count}")

        # Push to registry if requested
        if push_after_import:
            self.push_dataset(org, dataset_name, version)

        return metadata

    def push_dataset(
        self, org: str, dataset_name: str, version: str, annotations: dict[str, str] | None = None
    ) -> dict:
        """
        Push dataset from cache to OCI registry.

        Args:
            org: Organization name
            dataset_name: Dataset name
            version: Version string
            annotations: Optional OCI annotations

        Returns:
            Dictionary with push results
        """
        cache_path = self.cache.get_dataset_path(org, dataset_name, version)

        if not cache_path.exists():
            raise FileNotFoundError(f"Dataset not found in cache: {cache_path}")

        # Load metadata
        metadata = self.cache.get_metadata(org, dataset_name, version)
        if not metadata:
            raise ValueError(f"Metadata not found for {org}/{dataset_name}:{version}")

        logger.info(f"Pushing dataset {org}/{dataset_name}:{version} to {self.registry}")

        # Create tarball of dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            tarball_path = Path(tmpdir) / f"{dataset_name}-{version}.tar.gz"

            with tarfile.open(tarball_path, "w:gz") as tar:
                tar.add(cache_path, arcname=".")

            tarball_size = tarball_path.stat().st_size
            logger.info(f"Created tarball: {tarball_size / 1024 / 1024:.2f} MB")

            # Prepare OCI annotations
            oci_annotations = {
                "org.opencontainers.image.created": datetime.utcnow().isoformat(),
                "org.opencontainers.image.title": f"{dataset_name}:{version}",
                "io.dreadnode.dataset.org": org,
                "io.dreadnode.dataset.name": dataset_name,
                "io.dreadnode.dataset.version": version,
                "io.dreadnode.dataset.format": metadata.format,
                "io.dreadnode.dataset.compression": metadata.compression,
                "io.dreadnode.dataset.size": str(metadata.size_bytes),
                "io.dreadnode.dataset.checksum": metadata.checksum,
            }

            if metadata.row_count:
                oci_annotations["io.dreadnode.dataset.rows"] = str(metadata.row_count)
            if metadata.column_count:
                oci_annotations["io.dreadnode.dataset.columns"] = str(metadata.column_count)
            if metadata.description:
                oci_annotations["io.dreadnode.dataset.description"] = metadata.description

            if annotations:
                oci_annotations.update(annotations)

            # Push to registry
            target = f"{self.registry}/{org}/{dataset_name}:{version}"

            try:
                with tqdm(total=tarball_size, unit="B", unit_scale=True, desc="Pushing") as pbar:
                    response = self.client.push(
                        files=[str(tarball_path)],
                        target=target,
                    )
                    pbar.update(tarball_size)

                # Update sync state
                self.cache.update_sync_state(org, dataset_name, version, "push")

                logger.info(f"Successfully pushed {target}")

                return {
                    "status": "success",
                    "target": target,
                    "response": response,
                    "size_bytes": tarball_size,
                    "annotations": oci_annotations,
                }

            except Exception as e:
                logger.error(f"Push failed: {e}")
                return {"status": "error", "error": str(e), "target": target}

    # ========================
    # USE CASE 1a: Pack as Volume
    # ========================

    def push_dataset_as_volume(
        self, org: str, dataset_name: str, version: str, volume_name: str | None = None
    ) -> dict:
        """
        Pack dataset as OCI volume artifact for containerized workflows.

        Use Case 1a: Pack the OCI artifact dataset into a volume

        This creates a volume that can be mounted in containers using
        Docker/Podman/Kubernetes volume drivers.

        Args:
            org: Organization name
            dataset_name: Dataset name
            version: Version string
            volume_name: Optional custom volume name

        Returns:
            Dictionary with push results

        Example:
            >>> storage = DatasetOCIStorage()
            >>> result = storage.push_dataset_as_volume(
            ...     org='dreadnode',
            ...     dataset_name='user_events',
            ...     version='v1.0.0'
            ... )
        """
        cache_path = self.cache.get_dataset_path(org, dataset_name, version)

        if not cache_path.exists():
            raise FileNotFoundError(f"Dataset not found in cache: {cache_path}")

        volume_name = volume_name or f"{dataset_name}-{version}"

        logger.info(f"Packing dataset as volume: {volume_name}")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create volume tarball
            volume_path = Path(tmpdir) / f"{volume_name}.tar.gz"

            with tarfile.open(volume_path, "w:gz") as tar:
                # Add data directory as /data in volume
                data_path = cache_path / "data"
                tar.add(data_path, arcname="data")

                # Add metadata as /metadata.json
                metadata_path = cache_path / "metadata.json"
                tar.add(metadata_path, arcname="metadata.json")

            volume_size = volume_path.stat().st_size
            logger.info(f"Volume size: {volume_size / 1024 / 1024:.2f} MB")

            # Push as volume artifact
            target = f"{self.registry}/{org}/volumes/{volume_name}:{version}"

            try:
                response = self.client.push(
                    files=[str(volume_path)],
                    target=target,
                )

                logger.info(f"Successfully pushed volume: {target}")

                return {
                    "status": "success",
                    "target": target,
                    "volume_name": volume_name,
                    "size_bytes": volume_size,
                    "mount_instructions": self._get_volume_mount_instructions(target, volume_name),
                }

            except Exception as e:
                logger.error(f"Volume push failed: {e}")
                return {"status": "error", "error": str(e), "target": target}

    def _get_volume_mount_instructions(self, target: str, volume_name: str) -> dict[str, str]:
        """Generate instructions for mounting volume in containers."""
        return {
            "docker": f"""
# Pull volume
oras pull {target}

# Extract to volume
docker volume create {volume_name}
docker run --rm -v {volume_name}:/data -v $(pwd):/source alpine sh -c "tar -xzf /source/{volume_name}.tar.gz -C /data"

# Use in container
docker run -v {volume_name}:/data myimage
            """.strip(),
            "kubernetes": f"""
# Create PV from volume
apiVersion: v1
kind: PersistentVolume
metadata:
  name: {volume_name}
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadOnlyMany
  hostPath:
    path: /mnt/datasets/{volume_name}

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {volume_name}-claim
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 10Gi

# Mount in pod
volumes:
  - name: dataset
    persistentVolumeClaim:
      claimName: {volume_name}-claim
            """.strip(),
        }

    def pull_volume_and_load(
        self, org: str, volume_name: str, version: str, extract_to: str | None = None
    ) -> Path:
        """
        Pull volume artifact and extract to local directory.

        Use Case 1a (continued): Load dataset from volume

        Args:
            org: Organization name
            volume_name: Volume name
            version: Version string
            extract_to: Optional extraction directory

        Returns:
            Path to extracted data

        Example:
            >>> storage = DatasetOCIStorage()
            >>> data_path = storage.pull_volume_and_load(
            ...     org='dreadnode',
            ...     volume_name='user_events-v1.0.0',
            ...     version='v1.0.0'
            ... )
            >>> # Load data with PyArrow
            >>> import pyarrow.parquet as pq
            >>> table = pq.read_table(data_path / 'data')
        """
        target = f"{self.registry}/{org}/volumes/{volume_name}:{version}"

        logger.info(f"Pulling volume: {target}")

        # Determine extraction directory
        if extract_to:
            extract_path = Path(extract_to).expanduser()
        else:
            extract_path = self.cache.datasets_dir / org / "volumes" / volume_name / version

        extract_path.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Pull volume tarball
            files = self.client.pull(target=target, outdir=tmpdir)

            if not files:
                raise ValueError(f"No files retrieved from {target}")

            volume_file = files[0]

            # Extract tarball
            logger.info(f"Extracting volume to {extract_path}")
            with tarfile.open(volume_file, "r:gz") as tar:
                tar.extractall(extract_path)

            logger.info("Volume extracted successfully")

            return extract_path

    # ========================
    # USE CASE 2: Pull Dataset from Registry
    # ========================

    def pull_dataset(self, org: str, dataset_name: str, version: str, force: bool = False) -> Path:
        """
        Pull dataset from registry to local cache.

        Use Case 2: Dataset already in repo needs to be pulled and loaded

        Args:
            org: Organization name
            dataset_name: Dataset name
            version: Version string
            force: Force re-download even if cached

        Returns:
            Path to cached dataset

        Example:
            >>> storage = DatasetOCIStorage()
            >>> cache_path = storage.pull_dataset(
            ...     org='dreadnode',
            ...     dataset_name='user_events',
            ...     version='v1.0.0'
            ... )
            >>> # Load with PyArrow
            >>> import pyarrow.parquet as pq
            >>> table = pq.read_table(cache_path / 'data' / 'data.parquet')
        """
        cache_path = self.cache.get_dataset_path(org, dataset_name, version)

        # Check if already cached
        if cache_path.exists() and not force:
            logger.info(f"Dataset already cached at {cache_path}")
            return cache_path

        target = f"{self.registry}/{org}/{dataset_name}:{version}"

        logger.info(f"Pulling dataset from {target}")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Pull tarball
            files = self.client.pull(target=target, outdir=tmpdir)

            if not files:
                raise ValueError(f"No files retrieved from {target}")

            tarball_file = files[0]

            # Extract to cache
            cache_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Extracting to cache: {cache_path}")
            with tarfile.open(tarball_file, "r:gz") as tar:
                tar.extractall(cache_path)

            # Load metadata
            metadata_file = cache_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata_dict = json.load(f)
                    metadata = DatasetMetadata.from_dict(metadata_dict)
                    self.cache.save_metadata(metadata)

            # Update sync state
            self.cache.update_sync_state(org, dataset_name, version, "pull")

            logger.info(f"Dataset pulled successfully to {cache_path}")

            return cache_path

    def load_dataset(
        self,
        org: str,
        dataset_name: str,
        version: str,
        columns: list[str] | None = None,
        filter_expr: pc.Expression | None = None,
        use_memory_map: bool = True,
    ) -> pa.Table | ds.Dataset:
        """
        Load dataset from cache into memory as PyArrow Table.

        This provides zero-copy memory-mapped access for large datasets.

        Args:
            org: Organization name
            dataset_name: Dataset name
            version: Version string
            columns: Optional list of columns to load
            filter_expr: Optional PyArrow filter expression
            use_memory_map: Use memory mapping for large files

        Returns:
            PyArrow Table or Dataset

        Example:
            >>> storage = DatasetOCIStorage()
            >>>
            >>> # Ensure dataset is pulled
            >>> storage.pull_dataset('dreadnode', 'user_events', 'v1.0.0')
            >>>
            >>> # Load entire dataset
            >>> table = storage.load_dataset('dreadnode', 'user_events', 'v1.0.0')
            >>>
            >>> # Load specific columns with filter
            >>> import pyarrow.compute as pc
            >>> table = storage.load_dataset(
            ...     'dreadnode', 'user_events', 'v1.0.0',
            ...     columns=['user_id', 'event_type', 'timestamp'],
            ...     filter_expr=(pc.field('event_type') == 'click')
            ... )
        """
        cache_path = self.cache.get_dataset_path(org, dataset_name, version)

        if not cache_path.exists():
            logger.info("Dataset not in cache, pulling...")
            self.pull_dataset(org, dataset_name, version)

        metadata = self.cache.get_metadata(org, dataset_name, version)
        if not metadata:
            raise ValueError(f"Metadata not found for {org}/{dataset_name}:{version}")

        data_path = cache_path / "data"

        if metadata.format == "parquet":
            # Check if partitioned dataset
            if metadata.partitioned or any(data_path.rglob("_metadata")):
                # Read as partitioned dataset
                dataset = ds.dataset(data_path, format="parquet", partitioning="hive")

                if columns or filter_expr:
                    table = dataset.to_table(columns=columns, filter=filter_expr)
                    return table

                return dataset
            # Single parquet file
            parquet_file = next(data_path.glob("*.parquet"))
            return pq.read_table(
                parquet_file, columns=columns, filters=filter_expr, memory_map=use_memory_map
            )

        if metadata.format == "csv":
            csv_file = next(data_path.glob("*.csv"))
            return pa.csv.read_csv(csv_file, read_options=pa.csv.ReadOptions(column_names=columns))

        if metadata.format in ("json", "jsonl"):
            json_file = next(data_path.glob("*.json*"))
            df = pd.read_json(json_file, lines=(metadata.format == "jsonl"))
            if columns:
                df = df[columns]
            return pa.Table.from_pandas(df)

        raise ValueError(f"Unsupported format for loading: {metadata.format}")

    # ========================
    # USE CASE 3: Update Existing Dataset
    # ========================

    def update_dataset(
        self,
        org: str,
        dataset_name: str,
        current_version: str,
        new_version: str,
        update_data: pa.Table | pd.DataFrame | Path | str,
        update_mode: str = "replace",  # 'replace', 'append', 'upsert'
        upsert_keys: list[str] | None = None,
        push_after_update: bool = True,
    ) -> DatasetMetadata:
        """
        Update existing dataset and create new version with change tracking.

        Use Case 3: Dataset in repo has been updated, save and track changes

        Args:
            org: Organization name
            dataset_name: Dataset name
            current_version: Current version to update from
            new_version: New version string
            update_data: New data (Table, DataFrame, or path to file)
            update_mode: How to update ('replace', 'append', 'upsert')
            upsert_keys: Key columns for upsert mode
            push_after_update: Whether to push after updating

        Returns:
            New DatasetMetadata

        Example:
            >>> storage = DatasetOCIStorage()
            >>>
            >>> # Load current data
            >>> current_table = storage.load_dataset('dreadnode', 'user_events', 'v1.0.0')
            >>>
            >>> # Create new data
            >>> new_rows = pa.table({
            ...     'user_id': [1001, 1002],
            ...     'event_type': ['click', 'view'],
            ...     'timestamp': [datetime.now(), datetime.now()]
            ... })
            >>>
            >>> # Append new rows and create v1.1.0
            >>> metadata = storage.update_dataset(
            ...     org='dreadnode',
            ...     dataset_name='user_events',
            ...     current_version='v1.0.0',
            ...     new_version='v1.1.0',
            ...     update_data=new_rows,
            ...     update_mode='append'
            ... )
        """
        # Ensure current version is in cache
        current_cache_path = self.cache.get_dataset_path(org, dataset_name, current_version)
        if not current_cache_path.exists():
            logger.info("Pulling current version...")
            self.pull_dataset(org, dataset_name, current_version)

        current_metadata = self.cache.get_metadata(org, dataset_name, current_version)
        if not current_metadata:
            raise ValueError(f"Current version not found: {current_version}")

        logger.info(f"Updating {org}/{dataset_name} from {current_version} to {new_version}")

        # Create new version cache path
        new_cache_path = self.cache.get_dataset_path(org, dataset_name, new_version)
        new_cache_path.mkdir(parents=True, exist_ok=True)
        new_data_path = new_cache_path / "data"
        new_data_path.mkdir(exist_ok=True)

        # Load current data
        if current_metadata.format == "parquet":
            current_data_path = current_cache_path / "data"

            # Read current dataset
            if current_metadata.partitioned:
                current_dataset = ds.dataset(current_data_path, format="parquet")
                current_table = current_dataset.to_table()
            else:
                current_file = next(current_data_path.glob("*.parquet"))
                current_table = pq.read_table(current_file)

            # Process update data
            if isinstance(update_data, pd.DataFrame):
                update_table = pa.Table.from_pandas(update_data)
            elif isinstance(update_data, pa.Table):
                update_table = update_data
            elif isinstance(update_data, (Path, str)):
                update_path = Path(update_data)
                if update_path.suffix == ".parquet":
                    update_table = pq.read_table(update_path)
                elif update_path.suffix == ".csv":
                    update_table = pa.csv.read_csv(update_path)
                else:
                    raise ValueError(f"Unsupported file type: {update_path.suffix}")
            else:
                raise ValueError(f"Unsupported update_data type: {type(update_data)}")

            # Apply update mode
            if update_mode == "replace":
                result_table = update_table
                logger.info("Replacing entire dataset")

            elif update_mode == "append":
                result_table = pa.concat_tables([current_table, update_table])
                logger.info(f"Appended {len(update_table)} rows")

            elif update_mode == "upsert":
                if not upsert_keys:
                    raise ValueError("upsert_keys required for upsert mode")

                logger.info(f"Upserting on keys: {upsert_keys}")

                # Convert to pandas for upsert logic
                current_df = current_table.to_pandas()
                update_df = update_table.to_pandas()

                # Remove rows with matching keys
                mask = ~current_df[upsert_keys].apply(tuple, axis=1).isin(
                    update_df[upsert_keys].apply(tuple, axis=1)
                )
                filtered_df = current_df[mask]

                # Append updated rows
                result_df = pd.concat([filtered_df, update_df], ignore_index=True)
                result_table = pa.Table.from_pandas(result_df)

                logger.info(f"Upserted {len(update_table)} rows")

            else:
                raise ValueError(f"Unknown update_mode: {update_mode}")

            # Write new dataset
            output_file = new_data_path / "data.parquet"
            pq.write_table(
                result_table, output_file, compression=current_metadata.compression, version="2.6"
            )

            # Compute new metadata
            row_count = len(result_table)
            column_count = len(result_table.schema)
            size_bytes = output_file.stat().st_size
            schema_dict = {
                "fields": [
                    {"name": field.name, "type": str(field.type)} for field in result_table.schema
                ]
            }

        else:
            raise NotImplementedError(
                f"Update not yet implemented for format: {current_metadata.format}"
            )

        # Compute checksum
        checksum = self._compute_dataset_checksum(new_data_path)

        # Create new metadata
        new_metadata = DatasetMetadata(
            org=org,
            name=dataset_name,
            version=new_version,
            format=current_metadata.format,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            size_bytes=size_bytes,
            row_count=row_count,
            column_count=column_count,
            schema=schema_dict,
            compression=current_metadata.compression,
            checksum=checksum,
            parent_version=current_version,  # Track lineage
            description=current_metadata.description,
            tags=current_metadata.tags,
            partitioned=current_metadata.partitioned,
            partition_columns=current_metadata.partition_columns,
        )

        # Save metadata
        metadata_file = new_cache_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(new_metadata.to_dict(), f, indent=2)

        self.cache.save_metadata(new_metadata)

        # Create version lineage file
        lineage_file = new_cache_path / "lineage.json"
        lineage = {
            "version": new_version,
            "parent_version": current_version,
            "update_mode": update_mode,
            "updated_at": datetime.utcnow().isoformat(),
            "changes": {
                "rows_added": row_count
                - (current_table.num_rows if "current_table" in locals() else 0),
                "size_change_bytes": size_bytes - current_metadata.size_bytes,
            },
        }
        with open(lineage_file, "w") as f:
            json.dump(lineage, f, indent=2)

        logger.info(f"Dataset updated: {new_version}")
        logger.info(f"New size: {size_bytes / 1024 / 1024:.2f} MB, Rows: {row_count}")

        # Push if requested
        if push_after_update:
            self.push_dataset(org, dataset_name, new_version)

        return new_metadata

    def get_version_history(self, org: str, dataset_name: str) -> list[dict]:
        """
        Get version history with lineage information.

        Returns:
            List of version info dictionaries
        """
        versions = self.cache.list_versions(org, dataset_name)
        history = []

        for version in versions:
            metadata = self.cache.get_metadata(org, dataset_name, version)
            if metadata:
                version_info = {
                    "version": version,
                    "created_at": metadata.created_at,
                    "size_bytes": metadata.size_bytes,
                    "row_count": metadata.row_count,
                    "parent_version": metadata.parent_version,
                }

                # Load lineage if available
                cache_path = self.cache.get_dataset_path(org, dataset_name, version)
                lineage_file = cache_path / "lineage.json"
                if lineage_file.exists():
                    with open(lineage_file) as f:
                        version_info["lineage"] = json.load(f)

                history.append(version_info)

        return history

    # ========================
    # Utility Methods
    # ========================

    def diff_versions(self, org: str, dataset_name: str, version1: str, version2: str) -> dict:
        """
        Compare two versions of a dataset.

        Returns:
            Dictionary with diff information
        """
        # Ensure both versions are cached
        for version in [version1, version2]:
            cache_path = self.cache.get_dataset_path(org, dataset_name, version)
            if not cache_path.exists():
                self.pull_dataset(org, dataset_name, version)

        meta1 = self.cache.get_metadata(org, dataset_name, version1)
        meta2 = self.cache.get_metadata(org, dataset_name, version2)

        if not meta1 or not meta2:
            raise ValueError("Could not load metadata for one or both versions")

        return {
            "version1": version1,
            "version2": version2,
            "size_change_bytes": meta2.size_bytes - meta1.size_bytes,
            "row_change": (meta2.row_count or 0) - (meta1.row_count or 0),
            "schema_changed": meta1.schema != meta2.schema,
            "checksum_changed": meta1.checksum != meta2.checksum,
        }

    def cleanup_old_versions(self, org: str, dataset_name: str, keep_latest: int = 3):
        """
        Clean up old versions from cache, keeping only the latest N versions.

        Args:
            org: Organization name
            dataset_name: Dataset name
            keep_latest: Number of latest versions to keep
        """
        versions = self.cache.list_versions(org, dataset_name)

        if len(versions) <= keep_latest:
            logger.info(f"Only {len(versions)} versions, no cleanup needed")
            return

        versions_to_delete = versions[keep_latest:]

        for version in versions_to_delete:
            cache_path = self.cache.get_dataset_path(org, dataset_name, version)
            if cache_path.exists():
                shutil.rmtree(cache_path)
                logger.info(f"Deleted old version: {version}")

            # Remove from database
            with self.cache._db_connection() as conn:
                conn.execute(
                    "DELETE FROM datasets WHERE org=? AND name=? AND version=?",
                    (org, dataset_name, version),
                )
                conn.execute(
                    "DELETE FROM sync_state WHERE org=? AND name=? AND version=?",
                    (org, dataset_name, version),
                )


# ========================
# High-Level Helper Functions
# ========================


def quick_push(
    source_path: str,
    registry_target: str,
    format: str = "parquet",
    registry: str = "localhost:5000",
    **kwargs,
) -> dict:
    """
    Quick helper to push a dataset with minimal configuration.

    Args:
        source_path: Path to source data
        registry_target: Target in format 'org/dataset:version'
        format: Data format
        registry: Registry URL
        **kwargs: Additional arguments for import_dataset

    Example:
        >>> quick_push(
        ...     '/data/events.csv',
        ...     'dreadnode/user-events:v1.0.0',
        ...     format='csv',
        ...     convert_to_parquet=True
        ... )
    """
    parts = registry_target.split("/")
    org = parts[0]
    name_version = parts[1].split(":")
    name = name_version[0]
    version = name_version[1] if len(name_version) > 1 else "latest"

    storage = DatasetOCIStorage(registry=registry)
    metadata = storage.import_dataset(
        source_path=source_path,
        org=org,
        dataset_name=name,
        version=version,
        format=format,
        push_after_import=True,
        **kwargs,
    )

    return {"status": "success", "metadata": metadata.to_dict()}


def quick_pull(
    registry_target: str, registry: str = "localhost:5000", output_format: str = "table"
) -> pa.Table | pd.DataFrame | Path:
    """
    Quick helper to pull and load a dataset.

    Args:
        registry_target: Target in format 'org/dataset:version'
        registry: Registry URL
        output_format: 'table', 'dataframe', or 'path'

    Example:
        >>> df = quick_pull('dreadnode/user-events:v1.0.0', output_format='dataframe')
    """
    parts = registry_target.split("/")
    org = parts[0]
    name_version = parts[1].split(":")
    name = name_version[0]
    version = name_version[1] if len(name_version) > 1 else "latest"

    storage = DatasetOCIStorage(registry=registry)

    if output_format == "path":
        return storage.pull_dataset(org, name, version)

    table = storage.load_dataset(org, name, version)

    if output_format == "dataframe":
        return table.to_pandas()

    return table

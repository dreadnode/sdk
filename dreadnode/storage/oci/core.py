# core.py
"""Main dataset manager implementation."""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
from rich.console import Console

from .cache import CacheManager
from .formats import FormatRegistry
from .storage import OCIStorageClient
from .types import (
    CompressionType,
    DatasetFormat,
    DatasetMetadata,
    PullResult,
    PushResult,
    StorageStrategy,
)
from .utils import compute_checksum, sanitize_name, serialize_schema
from .volume import VolumeStorage

logger = logging.getLogger(__name__)
console = Console()


class DatasetManager:
    """
    Main dataset manager for OCI artifact storage.

    Provides high-level API for dataset operations with intelligent
    caching, version control, and format handling.
    """

    def __init__(
        self,
        registry: str = "localhost:5000",
        insecure: bool = False,
        username: str | None = None,
        password: str | None = None,
        cache_root: Path | None = None,
    ) -> None:
        """
        Initialize dataset manager.

        Args:
            registry: OCI registry hostname
            insecure: Whether to use insecure HTTP
            username: Registry username
            password: Registry password
            cache_root: Root directory for cache
        """
        self.registry = registry
        self.storage = OCIStorageClient(
            registry=registry,
            insecure=insecure,
            username=username,
            password=password,
        )
        self.cache = CacheManager(cache_root or Path.home() / ".dreadnode" / "datasets")
        self.volume_storage = VolumeStorage(self.storage.client)
        self.format_registry = FormatRegistry()

    def save_and_push(
        self,
        data_path: Path,
        org: str,
        dataset_name: str,
        version: str,
        *,
        format: DatasetFormat | None = None,
        compression: CompressionType = CompressionType.SNAPPY,
        storage_strategy: StorageStrategy = StorageStrategy.FILE,
        description: str | None = None,
        tags: dict[str, str] | None = None,
        parent_version: str | None = None,
    ) -> PushResult:
        """
        Save dataset that exists outside the cache. Save it to the local cache and then push to registry.

        Args:
            data_path: Path to dataset (file or directory)
            org: Organization name
            dataset_name: Dataset name
            version: Version (semver format)
            format: Dataset format (auto-detected if None)
            compression: Compression type
            storage_strategy: FILE or VOLUME storage
            description: Dataset description
            tags: Custom tags
            parent_version: Parent version for tracking lineage

        Returns:
            PushResult with operation details

        Example:
            >>> manager = DatasetManager("localhost:5000", insecure=True)
            >>> result = manager.save_and_push(
            ...     Path("/data/my_dataset.parquet"),
            ...     org="acme",
            ...     dataset_name="op-logs",
            ...     version="1.0.0",
            ...     description="acme corporation operational logs",
            ... )
            >>> print(f"Pushed to {result.target}")
        """
        console.print(f"[bold blue]Processing dataset:[/] {dataset_name}")

        # Detect format if not specified
        if format is None:
            format = self.format_registry.detect_format(data_path)

        # Read data to get statistics
        console.print("[dim]Reading dataset...")
        handler = self.format_registry.get_handler(format)
        table = handler.read(data_path)

        # Compute checksum
        console.print("[dim]Computing checksum...")
        checksum = compute_checksum(data_path)

        # Create metadata
        metadata = DatasetMetadata(
            name=sanitize_name(dataset_name),
            org=sanitize_name(org),
            version=version,
            format=format,
            compression=compression,
            storage_strategy=storage_strategy,
            size_bytes=data_path.stat().st_size,
            row_count=table.num_rows,
            column_count=table.num_columns,
            checksum=checksum,
            source_path=str(data_path),
            parent_version=parent_version,
            description=description,
            tags=tags or {},
            schema_json=serialize_schema(table.schema),
        )

        # Add to cache
        console.print("[dim]Caching dataset...")
        cached_path = self.cache.add_dataset(metadata, data_path)

        # Prepare target
        target = f"{self.registry}/{metadata.org}/{metadata.name}:{metadata.version}"

        # Push based on strategy
        if storage_strategy == StorageStrategy.FILE:
            console.print(f"[dim]Pushing to {target}...")
            result = self.storage.push_dataset(data_path, metadata, target)
        else:  # VOLUME
            console.print(f"[dim]Packing and pushing volume to {target}...")
            result = self.volume_storage.push_volume(data_path, metadata, target)

        if result.success:
            console.print(f"[bold green]✓[/] Successfully pushed to {target}")
            if result.digest:
                console.print(f"[dim]Digest: {result.digest}")
        else:
            console.print(f"[bold red]✗[/] Failed to push: {result.error}")

        return result

    def pull_and_load(
        self,
        org: str,
        dataset_name: str,
        version: str,
        *,
        force_refresh: bool = False,
    ) -> PullResult:
        """
        Pull dataset from registry and load to cache.

        Use Case 2: Dataset already in repo, pull to cache and load.

        Args:
            org: Organization name
            dataset_name: Dataset name
            version: Dataset version
            force_refresh: Force pull even if cached

        Returns:
            PullResult with dataset and metadata

        Example:
            >>> manager = DatasetManager("localhost:5000", insecure=True)
            >>> result = manager.pull_and_load("acme", "sales-data", "1.0.0")
            >>> table = result.table  # PyArrow Table
            >>> print(f"Loaded {result.metadata.row_count} rows")
        """
        org = sanitize_name(org)
        dataset_name = sanitize_name(dataset_name)
        target = f"{self.registry}/{org}/{dataset_name}:{version}"

        # Check if already cached
        if not force_refresh and self.cache.is_cached(org, dataset_name, version):
            console.print(f"[bold green]Using cached dataset:[/] {dataset_name}")

            cached_path = self.cache.get_dataset_path(org, dataset_name, version)
            metadata = self.cache.get_metadata(org, dataset_name, version)

            if metadata is None:
                return PullResult(
                    success=False,
                    source=target,
                    local_path=cached_path,
                    metadata=DatasetMetadata(
                        name=dataset_name,
                        org=org,
                        version=version,
                        format=DatasetFormat.PARQUET,
                        compression=CompressionType.NONE,
                        storage_strategy=StorageStrategy.FILE,
                        size_bytes=0,
                        checksum="",
                    ),
                    cached=True,
                    error="Metadata not found in cache",
                )

            return PullResult(
                success=True,
                source=target,
                local_path=cached_path,
                metadata=metadata,
                cached=True,
            )

        # Pull from registry
        console.print(f"[bold blue]Pulling dataset:[/] {target}")

        try:
            output_dir = self.cache.get_dataset_path(org, dataset_name, version)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Try to get metadata to determine strategy
            try:
                manifest = self.storage.get_manifest(target)
                annotations = manifest.get("annotations", {})
                strategy_str = annotations.get(
                    "io.dreadnode.dataset.strategy",
                    StorageStrategy.FILE.value,
                )
                storage_strategy = StorageStrategy(strategy_str)
            except Exception:
                storage_strategy = StorageStrategy.FILE

            # Pull based on strategy
            if storage_strategy == StorageStrategy.FILE:
                data_file, metadata_dict = self.storage.pull_dataset(target, output_dir)
            else:  # VOLUME
                console.print("[dim]Pulling and extracting volume...")
                data_file = self.volume_storage.pull_volume(target, output_dir)
                metadata_dict = {}

            # Create or update metadata
            if metadata_dict:
                metadata = DatasetMetadata(**metadata_dict)
            else:
                # Reconstruct metadata from file
                format = self.format_registry.detect_format(data_file)
                handler = self.format_registry.get_handler(format)
                table = handler.read(data_file)

                metadata = DatasetMetadata(
                    name=dataset_name,
                    org=org,
                    version=version,
                    format=format,
                    compression=CompressionType.SNAPPY,
                    storage_strategy=storage_strategy,
                    size_bytes=data_file.stat().st_size,
                    row_count=table.num_rows,
                    column_count=table.num_columns,
                    checksum=compute_checksum(data_file),
                    schema_json=serialize_schema(table.schema),
                )

            # Update cache
            self.cache.add_dataset(metadata, data_file.parent)

            console.print(f"[bold green]✓[/] Successfully pulled {target}")

            return PullResult(
                success=True,
                source=target,
                local_path=output_dir,
                metadata=metadata,
                cached=False,
            )

        except Exception as e:
            logger.exception("Failed to pull dataset")
            return PullResult(
                success=False,
                source=target,
                local_path=Path(),
                metadata=DatasetMetadata(
                    name=dataset_name,
                    org=org,
                    version=version,
                    format=DatasetFormat.PARQUET,
                    compression=CompressionType.NONE,
                    storage_strategy=StorageStrategy.FILE,
                    size_bytes=0,
                    checksum="",
                ),
                cached=False,
                error=str(e),
            )

    def update_and_push(
        self,
        org: str,
        dataset_name: str,
        current_version: str,
        new_version: str,
        data_path: Path | None = None,
        *,
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> PushResult:
        """
        Update dataset and push changes to registry.

        Use Case 3: Dataset in repo has been updated, track changes and push.

        Args:
            org: Organization name
            dataset_name: Dataset name
            current_version: Current version to update from
            new_version: New version to create
            data_path: Path to updated data (uses cached if None)
            description: Update description
            tags: Additional tags

        Returns:
            PushResult with operation details

        Example:
            >>> manager = DatasetManager("localhost:5000", insecure=True)
            >>> # Pull existing version first
            >>> result = manager.pull_and_load("acme", "sales-data", "1.0.0")
            >>>
            >>> # Make modifications to data...
            >>> updated_path = Path("/data/sales-data-updated.parquet")
            >>>
            >>> # Push as new version
            >>> result = manager.update_and_push(
            ...     "acme",
            ...     "sales-data",
            ...     current_version="1.0.0",
            ...     new_version="1.1.0",
            ...     data_path=updated_path,
            ...     description="Added Q2 data"
            ... )
        """
        org = sanitize_name(org)
        dataset_name = sanitize_name(dataset_name)

        console.print(
            f"[bold blue]Updating dataset:[/] {dataset_name} {current_version} → {new_version}"
        )

        # Get current metadata for lineage
        current_metadata = self.cache.get_metadata(org, dataset_name, current_version)

        # Use cached data if no new path provided
        if data_path is None:
            cached_path = self.cache.get_dataset_path(org, dataset_name, current_version)
            data_files = list(cached_path.glob("*"))
            data_files = [f for f in data_files if f.suffix in {".parquet", ".csv", ".json"}]

            if not data_files:
                return PushResult(
                    success=False,
                    target="",
                    digest=None,
                    size_bytes=0,
                    metadata=DatasetMetadata(
                        name=dataset_name,
                        org=org,
                        version=new_version,
                        format=DatasetFormat.PARQUET,
                        compression=CompressionType.NONE,
                        storage_strategy=StorageStrategy.FILE,
                        size_bytes=0,
                        checksum="",
                    ),
                    error="No data file found in cache",
                )

            data_path = data_files[0]

        # Merge tags
        merged_tags = {}
        if current_metadata:
            merged_tags.update(current_metadata.tags)
        if tags:
            merged_tags.update(tags)

        # Push with lineage tracking
        return self.save_and_push(
            data_path=data_path,
            org=org,
            dataset_name=dataset_name,
            version=new_version,
            format=current_metadata.format if current_metadata else None,
            compression=current_metadata.compression
            if current_metadata
            else CompressionType.SNAPPY,
            storage_strategy=current_metadata.storage_strategy
            if current_metadata
            else StorageStrategy.FILE,
            description=description,
            tags=merged_tags,
            parent_version=current_version,
        )

    def load_table(
        self,
        org: str,
        dataset_name: str,
        version: str,
    ) -> pa.Table:
        """
        Load dataset as PyArrow Table.

        Args:
            org: Organization name
            dataset_name: Dataset name
            version: Dataset version

        Returns:
            PyArrow Table

        Example:
            >>> table = manager.load_table("acme", "sales-data", "1.0.0")
            >>> df = table.to_pandas()
        """
        result = self.pull_and_load(org, dataset_name, version)

        if not result.success:
            raise RuntimeError(f"Failed to load dataset: {result.error}")

        # Find data file
        data_files = list(result.local_path.rglob("*"))
        data_files = [f for f in data_files if f.is_file() and f.name != "metadata.json"]

        if not data_files:
            raise RuntimeError("No data file found")

        handler = self.format_registry.get_handler(result.metadata.format)
        return handler.read(data_files[0])

    def list_versions(
        self,
        org: str,
        dataset_name: str,
    ) -> list[DatasetMetadata]:
        """
        List all versions of a dataset in cache.

        Args:
            org: Organization name
            dataset_name: Dataset name

        Returns:
            List of DatasetMetadata for all versions
        """
        org = sanitize_name(org)
        dataset_name = sanitize_name(dataset_name)

        all_datasets = self.cache.list_datasets(org)
        return [d for d in all_datasets if d.name == dataset_name]

    def delete_version(
        self,
        org: str,
        dataset_name: str,
        version: str,
    ) -> None:
        """
        Delete a version from local cache.

        Args:
            org: Organization name
            dataset_name: Dataset name
            version: Version to delete
        """
        org = sanitize_name(org)
        dataset_name = sanitize_name(dataset_name)
        self.cache.remove_dataset(org, dataset_name, version)
        console.print(f"[bold green]✓[/] Deleted {org}/{dataset_name}:{version} from cache")

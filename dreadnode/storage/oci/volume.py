# volume.py
"""Volume-based storage for datasets."""

from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path

import oras.client

from .types import DatasetMetadata, PushResult, StorageStrategy


class VolumeStorage:
    """Handle volume-based dataset storage."""

    def __init__(self, oci_client: oras.client.OrasClient) -> None:
        """
        Initialize volume storage.

        Args:
            oci_client: ORAS client instance
        """
        self.client = oci_client

    def pack_volume(
        self,
        source_path: Path,
        output_path: Path,
        compression: str = "gz",
    ) -> None:
        """
        Pack a dataset into a tar volume.

        Args:
            source_path: Path to dataset (file or directory)
            output_path: Path for output tar file
            compression: Compression type ('', 'gz', 'bz2', 'xz')
        """
        mode = f"w:{compression}" if compression else "w"

        with tarfile.open(output_path, mode) as tar:
            if source_path.is_file():
                tar.add(source_path, arcname=source_path.name)
            else:
                for item in source_path.rglob("*"):
                    if item.is_file():
                        arcname = item.relative_to(source_path)
                        tar.add(item, arcname=str(arcname))

    def unpack_volume(
        self,
        volume_path: Path,
        output_dir: Path,
    ) -> Path:
        """
        Unpack a tar volume.

        Args:
            volume_path: Path to tar file
            output_dir: Directory to extract to

        Returns:
            Path to extracted content
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(volume_path, "r:*") as tar:
            # Security check: ensure no path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in tar: {member.name}")
            tar.extractall(output_dir)

        # Find the main data file
        files = list(output_dir.rglob("*"))
        data_files = [f for f in files if f.is_file() and f.name != "metadata.json"]

        if not data_files:
            raise ValueError("No data files found in volume")

        return data_files[0]

    def push_volume(
        self,
        source_path: Path,
        metadata: DatasetMetadata,
        target: str,
    ) -> PushResult:
        """
        Push dataset as volume to registry.

        Args:
            source_path: Path to dataset
            metadata: Dataset metadata
            target: Target location

        Returns:
            PushResult with operation details
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            volume_path = Path(tmpdir) / "dataset.tar.gz"

            # Pack into volume
            self.pack_volume(source_path, volume_path, compression="gz")

            # Update metadata with actual size
            metadata.size_bytes = volume_path.stat().st_size
            metadata.storage_strategy = StorageStrategy.VOLUME

            try:
                # Push with volume artifact type
                response = self.client.push(
                    files=[str(volume_path)],
                    target=target,
                )

                digest = None
                if hasattr(response, "headers") and "Docker-Content-Digest" in response.headers:
                    digest = response.headers["Docker-Content-Digest"]

                return PushResult(
                    success=True,
                    target=target,
                    digest=digest,
                    size_bytes=metadata.size_bytes,
                    metadata=metadata,
                )

            except Exception as e:
                return PushResult(
                    success=False,
                    target=target,
                    digest=None,
                    size_bytes=metadata.size_bytes,
                    metadata=metadata,
                    error=str(e),
                )

    def pull_volume(
        self,
        target: str,
        output_dir: Path,
    ) -> Path:
        """
        Pull volume from registry and extract.

        Args:
            target: Source location
            output_dir: Directory to extract to

        Returns:
            Path to extracted dataset
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Pull volume
            files = self.client.pull(target=target, outdir=tmpdir)

            if not files:
                raise ValueError(f"No files retrieved from {target}")

            volume_file = Path(files[0])

            # Unpack volume
            return self.unpack_volume(volume_file, output_dir)

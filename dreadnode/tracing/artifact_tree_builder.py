"""
Tree structure builder for artifacts with directory hierarchy preservation.
Provides efficient concurrent uploads and caching.
"""

import hashlib
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Literal, TypedDict, Union

from dreadnode.storage.artifact_storage import ArtifactStorage

logger = getLogger(__name__)
MAX_CONCURRENT_UPLOADS = 10


class FileNode(TypedDict):
    """
    Represents a file node in the artifact tree.
    Contains metadata about the file, including its name, hash, size, and storage URI.
    """

    type: Literal["file"]
    name: str
    uri: str
    hash: str
    size_bytes: int
    mime_type: str | None
    file_extension: str
    final_real_path: str


class DirectoryNode(TypedDict):
    """
    Represents a directory node in the artifact tree.
    Contains metadata about the directory, including its name, hash, and children nodes.
    """

    type: Literal["dir"]
    name: str
    hash: str
    children: list[Union["DirectoryNode", FileNode]]


@dataclass
class ArtifactTreeBuilder:
    """
    Builds a hierarchical tree structure for artifacts while uploading them to storage.
    Preserves directory structure and handles concurrent uploads.
    """

    storage: ArtifactStorage
    prefix_path: str | None = None

    # Cache for previously processed files to avoid redundant uploads
    processed_files: dict[str, FileNode] = field(default_factory=dict)

    def process_artifact(self, local_uri: str | Path) -> DirectoryNode:
        """
        Process an artifact (file or directory) and build its tree representation.

        Args:
            local_uri: Path to the local file or directory

        Returns:
            Directory tree structure representing the artifact

        Raises:
            FileNotFoundError: If the path doesn't exist
        """
        local_path = Path(local_uri).expanduser().resolve()
        if not local_path.exists():
            raise FileNotFoundError(f"{local_path} does not exist")

        if local_path.is_dir():
            return self._process_directory(local_path)

        return self._process_single_file(local_path)

    def _process_single_file(self, file_path: Path) -> DirectoryNode:
        """
        Process a single file and create a directory structure for it.

        Args:
            file_path: Path to the file to be processed

        Returns:
            DirectoryNode containing the single file
        """
        file_node = self._process_file(file_path)

        file_node["final_real_path"] = f"/{file_path.parent.name}/{file_path.name}"

        dir_name = file_path.parent.name or file_path.name
        return {
            "type": "dir",
            "name": dir_name,
            "hash": file_node["hash"],
            "children": [file_node],
        }

    def _process_directory(self, dir_path: Path) -> DirectoryNode:
        """
        Process a directory and all its contents concurrently.

        This method recursively lists all files in the given directory and processes
        them concurrently using a thread pool. Each file is processed by the `_process_file`
        method, and the results are used to build a hierarchical directory tree structure.

        The number of concurrent threads is dynamically determined based on the number
        of available CPU cores, ensuring efficient resource utilization.

        Args:
            dir_path (Path): Path to the directory to be processed.

        Returns:
            DirectoryNode: A hierarchical tree structure representing the directory and its contents.

        Raises:
            FileNotFoundError: If the directory does not exist.
            OSError: If there is an issue accessing the directory or its contents.
        """
        logger.info("Processing directory: %s", dir_path)

        # Configure concurrent processing
        cpu_count = multiprocessing.cpu_count() or 1
        max_concurrent_uploads = max(MAX_CONCURRENT_UPLOADS, cpu_count * 2)
        logger.info("Using %d threads for concurrent processing", max_concurrent_uploads)

        # Collect all files
        all_files = []
        for root, _, files in os.walk(dir_path):
            root_path = Path(root)
            for file in files:
                file_path = root_path / file
                all_files.append(file_path)

        # Process files concurrently
        file_nodes_by_path = {}
        with ThreadPoolExecutor(max_workers=max_concurrent_uploads) as executor:
            future_to_path = {
                executor.submit(self._process_file, file_path): file_path for file_path in all_files
            }

            for future in as_completed(future_to_path):
                file_path = future_to_path[future]
                try:
                    file_node = future.result()
                    file_nodes_by_path[file_path] = file_node
                except FileNotFoundError:
                    logger.exception("File not found while processing %s", file_path)
                except Exception:
                    logger.exception("Unexpected error processing %s", file_path)

        # Build tree structure
        return self._build_tree_structure(dir_path, file_nodes_by_path)

    def _build_tree_structure(
        self,
        base_dir: Path,
        file_nodes_by_path: dict[Path, FileNode],
    ) -> DirectoryNode:
        """
        Build a hierarchical tree structure from processed files and directories.

        This method constructs a directory tree representation from a dictionary of
        file paths and their corresponding `FileNode` objects, while preserving empty directories.

        Args:
            base_dir (Path): The root directory for the tree structure.
            file_nodes_by_path (dict[Path, FileNode]): A dictionary mapping file paths
                to their corresponding `FileNode` objects.

        Returns:
            DirectoryNode: A hierarchical tree structure representing the directory
            and its contents.

        Example:
            Given the following directory structure:
            ```
            base_dir/
            ├── file1.txt
            ├── subdir1/
            │   ├── file2.txt
            │   └── file3.txt
            └── subdir2/
                └── file4.txt
            ```

            And the [file_nodes_by_path] dictionary:
            {
                Path("base_dir/file1.txt"): FileNode(...),
                Path("base_dir/subdir1/file2.txt"): FileNode(...),
                Path("base_dir/subdir1/file3.txt"): FileNode(...),
                Path("base_dir/subdir2/file4.txt"): FileNode(...),
            }

            The returned tree structure will look like:
            {
                "type": "dir",
                "name": "base_dir",
                "hash": "<hash_of_base_dir>",
                "children": [
                    {
                        "type": "file",
                        "name": "file1.txt",
                        ...
                    },
                    {
                        "type": "dir",
                        "name": "subdir1",
                        "hash": "<hash_of_subdir1>",
                        "children": [
                            {
                                "type": "file",
                                "name": "file2.txt",
                                ...
                            },
                            {
                                "type": "file",
                                "name": "file3.txt",
                                ...
                            }
                        ]
                    },
                    {
                        "type": "dir",
                        "name": "subdir2",
                        "hash": "<hash_of_subdir2>",
                        "children": [
                            {
                                "type": "file",
                                "name": "file4.txt",
                                ...
                            }
                        ]
                    }
                ]
            }
        """
        dir_structure: dict[str, DirectoryNode] = {}

        # Create root node
        root_name = base_dir.name
        root_node: DirectoryNode = {
            "type": "dir",
            "name": root_name,
            "hash": "",  # Will be computed later
            "children": [],
        }
        dir_structure[str(base_dir)] = root_node

        # Create directory structure
        for file_path in file_nodes_by_path:
            try:
                rel_path = file_path.relative_to(base_dir)
                parts = rel_path.parts
            except ValueError:
                logger.warning("File %s is not relative to base directory %s", file_path, base_dir)
                continue

            # File in the root directory
            if len(parts) == 1:
                continue

            # Create parent directories
            current_dir = base_dir
            for part in parts[:-1]:
                next_dir = current_dir / part
                if str(next_dir) not in dir_structure:
                    dir_node: DirectoryNode = {
                        "type": "dir",
                        "name": part,
                        "hash": "",  # Will be computed later
                        "children": [],
                    }
                    dir_structure[str(next_dir)] = dir_node
                    dir_structure[str(current_dir)]["children"].append(dir_node)
                current_dir = next_dir

        # Now add all files to their respective parent directories
        for file_path, file_node in file_nodes_by_path.items():
            parent_dir = file_path.parent
            rel_path = file_path.relative_to(base_dir.parent)
            file_node["final_real_path"] = f"/{rel_path}"

            if str(parent_dir) in dir_structure:
                dir_structure[str(parent_dir)]["children"].append(file_node)
            elif parent_dir == base_dir:
                root_node["children"].append(file_node)

        # Compute directory hashes bottom-up
        self._compute_directory_hashes(base_dir, dir_structure)

        return root_node

    def _compute_directory_hashes(
        self,
        dir_path: Path,
        dir_structure: dict[str, DirectoryNode],
    ) -> str:
        """Compute content-based hash for a directory recursively."""
        dir_node = dir_structure[str(dir_path)]
        child_hashes = []

        for child in dir_node["children"]:
            if child["type"] == "file":
                child_hashes.append(f"{child['hash']}")
            else:
                child_path = dir_path / child["name"]
                child_hash = self._compute_directory_hashes(child_path, dir_structure)
                child_hashes.append(f"{child_hash}")

        child_hashes.sort()  # Ensure consistent hash regardless of order
        hash_input = "|".join(child_hashes)
        dir_hash = hashlib.sha1(hash_input.encode()).hexdigest()[:16]  # noqa: S324

        dir_node["hash"] = dir_hash
        return dir_hash

    def _process_file(self, file_path: Path) -> FileNode:
        """
        Process a single file by hashing and uploading it to storage.

        This method computes a SHA1 hash of the file's contents to uniquely identify it.
        If the file has already been processed (based on the hash), the cached result is
        returned. Otherwise, the file is uploaded to the storage system, and a `FileNode`
        is created to represent the file.

        The method also extracts metadata such as the file's size, MIME type, and extension,
        and determines the target storage path based on the user ID and file hash.

        Args:
            file_path (Path): Path to the file to be processed.

        Returns:
            FileNode: A dictionary representing the processed file, including its metadata
            and storage URI.

        Raises:
            FileNotFoundError: If the file does not exist.
            OSError: If there is an issue accessing the file.
        """
        # Use cache if available
        file_hash = self.storage.compute_file_hash(file_path)

        if file_hash in self.processed_files:
            logger.debug("Using cached result for %s with hash %s", file_path, file_hash)
            cached_node = self.processed_files[file_hash].copy()
            cached_node["name"] = file_path.name
            return cached_node

        # Get file metadata
        file_extension = file_path.suffix
        clean_extension = file_extension.lstrip(".").lower()
        file_name = file_path.name
        file_size = file_path.stat().st_size

        # Determine storage path
        if self.prefix_path:
            prefix = self.prefix_path.rstrip("/")
            target_key = f"{prefix}/artifacts/{file_hash}{file_extension}"
        else:
            raise ValueError("Prefix path is invalid or empty")

        # Store the file
        uri = self.storage.store_file(file_path, target_key)

        # Create and cache file node
        file_node: FileNode = {
            "type": "file",
            "name": file_name,
            "uri": uri,
            "hash": file_hash,
            "size_bytes": file_size,
            "mime_type": self.storage.get_mime_type(file_path),
            "file_extension": clean_extension,
            "final_real_path": target_key,
        }
        self.processed_files[file_hash] = file_node

        return file_node

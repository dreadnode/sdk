"""
Tree structure builder for artifacts with directory hierarchy preservation.
Provides efficient concurrent uploads and caching.
"""

import hashlib
import logging
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypedDict, Union

from dreadnode.storage.artifact_storage import ArtifactStorage

logger = logging.getLogger(__name__)
MAX_CONCURRENT_UPLOADS = 10


class FileNode(TypedDict):
    type: Literal["file"]
    name: str
    hash_filename: str
    uri: str
    sha1: str
    size_bytes: int
    mime_type: str | None
    file_extension: str
    final_real_path: str


class DirectoryNode(TypedDict):
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
        file_node = self._process_file(local_path)
        # Create a directory node with the single file for consistent return type
        dir_name = local_path.parent.name or local_path.name
        return {
            "type": "dir",
            "name": dir_name,
            "hash": hashlib.sha1(dir_name.encode()).hexdigest()[:16],  # noqa: S324
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

        # Process files concurrently
        max_concurrent_uploads = min(MAX_CONCURRENT_UPLOADS, (multiprocessing.cpu_count() or 1) * 2)
        logger.info("Using %d threads for concurrent processing", max_concurrent_uploads)

        # List all files recursively
        all_files = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = Path(root) / file
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
                except Exception as e:
                    logger.exception("Unexpected error processing %s: %s", file_path, e)

        # Build tree structure
        return self._build_tree_structure(dir_path, file_nodes_by_path)

    def _build_tree_structure(
        self,
        base_dir: Path,
        file_nodes_by_path: dict[Path, FileNode],
    ) -> DirectoryNode:
        """
        Build a hierarchical tree structure from processed files.

        This method constructs a directory tree representation from a dictionary of
        file paths and their corresponding `FileNode` objects. It ensures that the
        directory hierarchy is preserved, and each directory and file is represented
        as a node in the tree.

        The root node is created based on the `base_dir`, and all files are added
        to their respective parent directories. If a directory does not exist in
        the tree, it is created dynamically.

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
        root_hash = hashlib.sha1(root_name.encode()).hexdigest()[:16]  # noqa: S324
        root_node: DirectoryNode = {
            "type": "dir",
            "name": root_name,
            "hash": root_hash,
            "children": [],
        }
        dir_structure[str(base_dir)] = root_node

        # Process files and build tree
        for file_path, file_node in file_nodes_by_path.items():
            # Get relative path from base directory
            rel_path = file_path.relative_to(base_dir)
            parts = rel_path.parts

            # Handle files at root level
            if len(parts) == 1:
                root_node["children"].append(file_node)
                continue

            # Create parent directories
            current_dir = base_dir
            for part in parts[:-1]:  # Skip filename
                next_dir = current_dir / part

                if str(next_dir) not in dir_structure:
                    # Create new directory node
                    dir_hash = hashlib.sha1(part.encode()).hexdigest()[:16]  # noqa: S324
                    dir_node: DirectoryNode = {
                        "type": "dir",
                        "name": part,
                        "hash": dir_hash,
                        "children": [],
                    }
                    dir_structure[str(next_dir)] = dir_node

                    # Add to parent
                    if str(current_dir) in dir_structure:
                        dir_structure[str(current_dir)]["children"].append(dir_node)

                current_dir = next_dir

            # Add file to immediate parent
            if str(current_dir) in dir_structure:
                dir_structure[str(current_dir)]["children"].append(file_node)

        return root_node

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
        # Compute file hash
        file_hash = self.storage.compute_file_hash(file_path)
        file_size = file_path.stat().st_size

        # Check cache for this hash
        if file_hash in self.processed_files:
            logger.debug("Using cached result for %s with hash %s", file_path, file_hash)
            cached_node = self.processed_files[file_hash].copy()
            cached_node["name"] = file_path.name
            return cached_node

        # Get file metadata
        file_extension = file_path.suffix
        clean_extension = file_extension.lstrip(".").lower()
        file_name = file_path.name
        hash_filename = f"{file_hash}_{file_name}"

        # Determine storage path
        if self.prefix_path:
            prefix = self.prefix_path.rstrip("/")
            target_key = f"{prefix}/artifacts/{hash_filename}"
        else:
            raise ValueError("Prefix path is invalid or empty")

        # Store the file
        uri = self.storage.store_file(file_path, target_key)

        # Create file node
        file_node: FileNode = {
            "type": "file",
            "name": file_name,
            "hash_filename": hash_filename,
            "uri": uri,
            "sha1": file_hash,
            "size_bytes": file_size,
            "mime_type": self.storage.get_mime_type(file_path),
            "file_extension": clean_extension,
            "final_real_path": target_key,
        }

        # Cache result
        self.processed_files[file_hash] = file_node
        return file_node

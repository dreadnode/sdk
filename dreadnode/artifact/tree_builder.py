"""
Tree structure builder for artifacts with directory hierarchy preservation.
Provides efficient uploads and tree construction for frontend to consume.
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, Union

from dreadnode.artifact.storage import ArtifactStorage
from dreadnode.util import logger


class FileNode(TypedDict):
    """
    Represents a file node in the artifact tree.
    Contains metadata about the file, including its name, uri, size_bytes, and final_real_path.
    """

    type: Literal["file"]
    hash: str
    uri: str
    size_bytes: int
    final_real_path: str


class DirectoryNode(TypedDict):
    """
    Represents a directory node in the artifact tree.
    Contains metadata about the directory, including its dir_path, hash, and children nodes.
    """

    type: Literal["dir"]
    dir_path: str
    hash: str
    children: list[Union["DirectoryNode", FileNode]]


@dataclass
class ArtifactTreeBuilder:
    """
    Builds a hierarchical tree structure for artifacts while uploading them to storage.
    Preserves directory structure and handles efficient uploads.
    """

    storage: ArtifactStorage
    prefix_path: str | None = None

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

        file_node["final_real_path"] = file_path.resolve().as_posix()

        dir_path = file_path.parent.resolve().as_posix()
        return {
            "type": "dir",
            "dir_path": dir_path,
            "hash": file_node["hash"],
            "children": [file_node],
        }

    def _process_directory(self, dir_path: Path) -> DirectoryNode:
        """
        Process a directory and all its contents efficiently.

        Args:
            dir_path: Path to the directory to be processed.

        Returns:
            DirectoryNode: A hierarchical tree structure representing the directory and its contents.
        """
        logger.debug("Processing directory: %s", dir_path)

        all_files: list[Path] = []
        for root, _, files in os.walk(dir_path):
            root_path = Path(root)
            for file in files:
                file_path = root_path / file
                all_files.append(file_path)

        file_hashes = self.storage.compute_file_hashes(all_files)

        source_paths = []
        target_paths = []
        file_nodes_by_path: dict[Path, FileNode] = {}
        file_hash_cache: dict[str, FileNode] = {}

        for file_path in all_files:
            file_path_str = file_path.resolve().as_posix()
            file_hash = file_hashes.get(file_path_str)
            if not file_hash:
                raise ValueError(f"File {file_path} not found in hash computation")

            # Check local cache for duplicates within this directory
            if file_hash in file_hash_cache:
                cached_node = file_hash_cache[file_hash].copy()
                cached_node["final_real_path"] = file_path_str
                file_nodes_by_path[file_path] = cached_node
                continue

            file_extension = file_path.suffix
            file_size = file_path.stat().st_size

            if self.prefix_path:
                prefix = self.prefix_path.rstrip("/")
                target_key = f"{prefix}/artifacts/{file_hash}{file_extension}"
            else:
                raise ValueError("Prefix path is invalid or empty")

            source_paths.append(file_path_str)
            target_paths.append(target_key)

            # Create the file node without URI (will be set after upload)
            file_node: FileNode = {
                "type": "file",
                "uri": "",
                "hash": file_hash,
                "size_bytes": file_size,
                "final_real_path": file_path.resolve().as_posix(),
            }

            file_nodes_by_path[file_path] = file_node
            file_hash_cache[file_hash] = file_node

        if source_paths:
            logger.debug("Uploading %d files in batch", len(source_paths))
            uris = self.storage.batch_upload_files(source_paths, target_paths)

            # Update file nodes with URIs
            for i, file_path_str in enumerate(source_paths):
                file_path = Path(file_path_str)
                if file_path in file_nodes_by_path:
                    file_nodes_by_path[file_path]["uri"] = uris[i]

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
        root_dir_path = base_dir.resolve().as_posix()
        root_node: DirectoryNode = {
            "type": "dir",
            "dir_path": root_dir_path,
            "hash": "",  # Will be computed later
            "children": [],
        }
        dir_structure[root_dir_path] = root_node

        for file_path, file_node in file_nodes_by_path.items():
            try:
                rel_path = file_path.relative_to(base_dir)
                parts = rel_path.parts
            except ValueError:
                logger.debug("File %s is not relative to base directory %s", file_path, base_dir)
                continue

            # File in the root directory
            if len(parts) == 1:
                root_node["children"].append(file_node)
                continue

            # Create parent directories
            current_dir = base_dir
            current_dir_str = current_dir.resolve().as_posix()
            for part in parts[:-1]:
                next_dir = current_dir / part
                next_dir_str = next_dir.resolve().as_posix()
                if next_dir_str not in dir_structure:
                    dir_node: DirectoryNode = {
                        "type": "dir",
                        "dir_path": next_dir_str,
                        "hash": "",  # Will be computed later
                        "children": [],
                    }
                    dir_structure[next_dir_str] = dir_node
                    dir_structure[current_dir_str]["children"].append(dir_node)
                current_dir = next_dir
                current_dir_str = next_dir_str
            # Now add the file to its parent directory
            parent_dir_str = file_path.parent.resolve().as_posix()
            if parent_dir_str in dir_structure:
                dir_structure[parent_dir_str]["children"].append(file_node)
        self._compute_directory_hashes(dir_structure)

        return root_node

    def _compute_directory_hashes(self, dir_structure: dict[str, DirectoryNode]) -> None:
        """
        Compute hashes for all directories in the structure.

        Args:
            dir_structure: Dictionary mapping directory paths to DirectoryNode objects
        """
        parents = self._map_parent_child_relationships(dir_structure)
        leaf_dirs = self._find_leaf_directories(dir_structure, parents)
        self._process_directories_bottom_up(dir_structure, parents, leaf_dirs)

    def _map_parent_child_relationships(
        self,
        dir_structure: dict[str, DirectoryNode],
    ) -> dict[str, str]:
        """
        Create a mapping of parent-child relationships for directories.

        Args:
            dir_structure: Dictionary mapping directory paths to DirectoryNode objects

        Returns:
            A dictionary mapping child directory paths to their parent directory paths.
        """
        parents = {}
        for dir_path, dir_node in dir_structure.items():
            for child in dir_node["children"]:
                if child["type"] == "dir":
                    child_path = child["dir_path"]
                    parents[child_path] = dir_path
        return parents

    def _find_leaf_directories(
        self,
        dir_structure: dict[str, DirectoryNode],
        parents: dict[str, str],
    ) -> set[str]:
        """
        Find leaf directories (those with no directory children).

        Args:
            dir_structure: Dictionary mapping directory paths to DirectoryNode objects
            parents: Dictionary mapping child directory paths to parent directory paths

        Returns:
            A set of leaf directory paths.
        """
        leaf_dirs = set()
        for dir_path in dir_structure:
            if dir_path not in parents.values():
                leaf_dirs.add(dir_path)
        return leaf_dirs

    def _process_directories_bottom_up(
        self,
        dir_structure: dict[str, DirectoryNode],
        parents: dict[str, str],
        leaf_dirs: set[str],
    ) -> None:
        """
        Process directories bottom-up starting from leaf directories.

        Args:
            dir_structure: Dictionary mapping directory paths to DirectoryNode objects
            parents: Dictionary mapping child directory paths to parent directory paths
            leaf_dirs: Set of leaf directory paths
        """
        processed = set()
        while leaf_dirs:
            dir_path = leaf_dirs.pop()
            dir_node = dir_structure[dir_path]

            # Compute hash based on children
            dir_node["hash"] = self._compute_directory_hash(dir_node)

            processed.add(dir_path)

            # Add parent to leaf_dirs if all its children are processed
            if dir_path in parents:
                parent_path = parents[dir_path]
                if self._are_all_children_processed(dir_structure[parent_path], processed):
                    leaf_dirs.add(parent_path)

    def _compute_directory_hash(self, dir_node: DirectoryNode) -> str:
        """
        Compute the hash for a directory based on its children.

        Args:
            dir_node: The DirectoryNode object

        Returns:
            The computed hash as a string.
        """
        child_hashes = [child["hash"] for child in dir_node["children"]]
        child_hashes.sort()  # Ensure consistent hash
        hash_input = "|".join(child_hashes)
        return hashlib.sha1(hash_input.encode()).hexdigest()[:16]  # noqa: S324 # nosec

    def _are_all_children_processed(self, parent_node: DirectoryNode, processed: set[str]) -> bool:
        """
        Check if all children of a parent directory have been processed.

        Args:
            parent_node: The parent DirectoryNode object
            processed: Set of processed directory paths

        Returns:
            True if all children are processed, False otherwise.
        """
        for child in parent_node["children"]:
            if child["type"] == "dir" and child["dir_path"] not in processed:
                return False
        return True

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
        """
        file_hash = self.storage.compute_file_hash(file_path)

        file_extension = file_path.suffix
        file_size = file_path.stat().st_size

        if self.prefix_path:
            prefix = self.prefix_path.rstrip("/")
            target_key = f"{prefix}/artifacts/{file_hash}{file_extension}"
        else:
            raise ValueError("Prefix path is invalid or empty")

        uri = self.storage.store_file(file_path, target_key)

        return {
            "type": "file",
            "uri": uri,
            "hash": file_hash,
            "size_bytes": file_size,
            "final_real_path": file_path.resolve().as_posix(),
        }

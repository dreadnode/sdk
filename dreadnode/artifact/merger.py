"""
Utility for merging artifact tree structures while preserving directory hierarchy.
"""

import hashlib
from pathlib import Path
from typing import cast

from dreadnode.artifact.tree_builder import DirectoryNode, FileNode


class ArtifactMerger:
    """
    Class responsible for merging artifact tree structures.
    Handles overlapping directory structures and efficiently combines artifacts.

    Example:
        ```python
        # Create a merger instance
        merger = ArtifactMerger()

        # Add multiple artifact trees
        merger.add_tree(tree1)  # First tree gets added directly
        merger.add_tree(tree2)  # Second tree gets merged if it overlaps

        # Get the merged result
        merged_trees = merger.get_merged_trees()
        ```
    """

    def __init__(self) -> None:
        self._path_map: dict[str, DirectoryNode | FileNode] = {}
        # Maps file hashes to all matching files
        self._hash_map: dict[str, list[FileNode]] = {}
        self._merged_trees: list[DirectoryNode] = []

    def add_tree(self, new_tree: DirectoryNode) -> None:
        """
        Add a new artifact tree, merging with existing trees if needed.

        This method analyzes the new tree and determines how to integrate it
        with existing trees, handling parent/child relationships and overlaps.

        Args:
            new_tree: New directory tree to add

        Example:
            ```python
            # Add first tree (e.g., /data/audio/sub1)
            merger.add_tree({
                "type": "dir",
                "dir_path": "/data/audio/sub1",
                "hash": "abc123",
                "children": [...]
            })

            # Add parent directory later (e.g., /data/audio)
            # The merger will recognize the relationship and restructure
            merger.add_tree({
                "type": "dir",
                "dir_path": "/data/audio",
                "hash": "def456",
                "children": [...]
            })
            ```
        """
        # First artifact - just add it
        if not self._merged_trees:
            self._merged_trees = [new_tree]
            self._build_maps(new_tree)
            return

        # Get new tree's path
        new_dir_path = new_tree["dir_path"]

        # Check for direct match with existing trees
        for existing_tree in self._merged_trees:
            if existing_tree["dir_path"] == new_dir_path:
                # Same directory - merge them
                self._merge_directory_nodes(existing_tree, new_tree)
                self._build_maps()  # Rebuild maps
                return

        # Check if new tree is parent of any existing trees
        children_to_remove = []
        for existing_tree in self._merged_trees:
            existing_dir_path = existing_tree["dir_path"]

            # New tree is parent of existing tree
            if existing_dir_path.startswith(new_dir_path + "/"):
                rel_path = existing_dir_path[len(new_dir_path) + 1 :].split("/")
                self._place_tree_at_path(new_tree, existing_tree, rel_path)
                children_to_remove.append(existing_tree)

        # Remove trees that are now incorporated into new tree
        if children_to_remove:
            for child in children_to_remove:
                if child in self._merged_trees:
                    self._merged_trees.remove(child)
            self._merged_trees.append(new_tree)
            self._build_maps()  # Rebuild maps
            return

        # Check if new tree is child of an existing tree
        for existing_tree in self._merged_trees:
            existing_dir_path = existing_tree["dir_path"]

            if new_dir_path.startswith(existing_dir_path + "/"):
                rel_path = new_dir_path[len(existing_dir_path) + 1 :].split("/")
                self._place_tree_at_path(existing_tree, new_tree, rel_path)
                self._build_maps()  # Rebuild maps
                return

        # Try to find and handle overlaps
        new_path_map: dict[str, DirectoryNode | FileNode] = {}
        new_hash_map: dict[str, list[FileNode]] = {}
        self._build_path_and_hash_maps(new_tree, new_path_map, new_hash_map)

        # Find common paths between existing and new tree
        path_overlaps = set(self._path_map.keys()) & set(new_path_map.keys())

        if path_overlaps and self._handle_overlaps(path_overlaps, new_path_map):
            # Successfully merged via overlaps
            self._build_maps()  # Rebuild maps
            return

        # If we get here, add new tree as a separate root
        self._merged_trees.append(new_tree)
        self._build_maps()  # Rebuild maps

    def get_merged_trees(self) -> list[DirectoryNode]:
        """
        Get the current merged trees.

        Returns:
            List of merged directory trees

        Example:
            ```python
            # Get the merged trees after adding multiple trees
            trees = merger.get_merged_trees()

            # Typically there will be a single root tree if all added trees are related
            if len(trees) == 1:
                root_tree = trees[0]
                print(f"Root directory: {root_tree['dir_path']}")
            ```
        """
        return self._merged_trees

    def _place_tree_at_path(
        self,
        parent_tree: DirectoryNode,
        child_tree: DirectoryNode,
        path_parts: list[str],
    ) -> None:
        """
        Place child_tree at the specified path under parent_tree.

        This creates any necessary intermediate directories and then merges
        the child tree at the correct location in the parent tree.

        Args:
            parent_tree: The parent tree to place under
            child_tree: The child tree to place
            path_parts: Path components from parent to child

        Example:
            ```python
            # Internal use to place /data/audio/sub1 under /data
            # path_parts would be ['audio', 'sub1']
            self._place_tree_at_path(
                parent_tree=data_tree,  # /data
                child_tree=sub1_tree,   # /data/audio/sub1
                path_parts=['audio', 'sub1']
            )
            ```
        """
        current = parent_tree

        # Navigate to the correct location, creating directories as needed
        for part in path_parts:
            if not part:  # Skip empty path parts
                continue

            # Look for existing directory
            next_node = None
            for child in current["children"]:
                if child["type"] == "dir" and Path(child["dir_path"]).name == part:
                    next_node = child
                    break

            # Create directory if it doesn't exist
            if next_node is None:
                next_dir_path = f"{current['dir_path']}/{part}"
                next_node = {
                    "type": "dir",
                    "dir_path": next_dir_path,
                    "hash": "",
                    "children": [],
                }
                current["children"].append(next_node)

            current = next_node

        # Merge the trees at the final location
        self._merge_directory_nodes(current, child_tree)

    def _handle_overlaps(
        self,
        overlaps: set[str],
        new_path_map: dict[str, DirectoryNode | FileNode],
    ) -> bool:
        """
        Handle overlapping paths between trees.

        This method processes paths that exist in both the existing trees
        and the new tree, merging directories and handling file conflicts.

        Args:
            overlaps: Set of overlapping paths
            new_path_map: Path map for the new tree

        Returns:
            True if the tree was merged, False otherwise

        Example:
            ```python
            # Internal use when two directories have some paths in common
            # but neither is a parent of the other
            overlapping_paths = {'/data/shared/file1.txt', '/data/shared/configs'}
            result = self._handle_overlaps(
                overlaps=overlapping_paths,
                new_path_map={'/data/shared/file1.txt': file_node, ...}
            )
            # If result is True, the trees were successfully merged
            ```
        """
        merged = False

        for path in sorted(overlaps, key=len):
            existing_node = self._path_map.get(path)
            new_node = new_path_map.get(path)

            if not existing_node or not new_node:
                continue

            if existing_node["type"] == "dir" and new_node["type"] == "dir":
                # Both are directories - merge them
                self._merge_directory_nodes(
                    cast("DirectoryNode", existing_node),
                    cast("DirectoryNode", new_node),
                )
                merged = True
            elif existing_node["type"] == "file" and new_node["type"] == "file":
                # Both are files - propagate URIs and update if hash differs
                existing_file = cast("FileNode", existing_node)
                new_file = cast("FileNode", new_node)

                # Always propagate URIs between files with identical hash
                if existing_file["hash"] == new_file["hash"]:
                    self._propagate_uri(existing_file, new_file)
                    merged = True
                else:
                    # Different hash - find the parent directory and update the file
                    for tree in self._merged_trees:
                        if self._update_file_in_tree(tree, existing_file, new_file):
                            merged = True
                            break

        return merged

    def _propagate_uri(self, file1: FileNode, file2: FileNode) -> None:
        """
        Ensure URIs are propagated between files with the same hash.

        If one file has a URI and the other doesn't, the URI will be copied.

        Args:
            file1: First file node
            file2: Second file node

        Example:
            ```python
            # Internal use to ensure URIs are shared between identical files
            # If file1 has a URI but file2 doesn't, file2 will get file1's URI
            self._propagate_uri(
                file1={"type": "file", "uri": "s3://bucket/file.txt", ...},
                file2={"type": "file", "uri": "", ...}
            )
            # After: file2["uri"] == "s3://bucket/file.txt"
            ```
        """
        if not file1["uri"] and file2["uri"]:
            file1["uri"] = file2["uri"]
        elif not file2["uri"] and file1["uri"]:
            file2["uri"] = file1["uri"]

    def _update_file_in_tree(
        self,
        tree: DirectoryNode,
        old_file: FileNode,
        new_file: FileNode,
    ) -> bool:
        """
        Update a file in a directory tree.

        This replaces old_file with new_file in the tree, recursively searching
        if necessary.

        Args:
            tree: The directory tree to search
            old_file: The file to replace
            new_file: The new file

        Returns:
            True if the file was found and updated

        Example:
            ```python
            # Internal use to replace an outdated file with a newer version
            success = self._update_file_in_tree(
                tree=root_tree,
                old_file={"type": "file", "hash": "abc123", ...},
                new_file={"type": "file", "hash": "def456", ...}
            )
            # If success is True, the file was found and replaced
            ```
        """
        for i, child in enumerate(tree["children"]):
            if child is old_file:
                tree["children"][i] = new_file
                return True

            if child["type"] == "dir" and self._update_file_in_tree(
                cast("DirectoryNode", child),
                old_file,
                new_file,
            ):
                return True
        return False

    def _build_maps(self, new_tree: DirectoryNode | None = None) -> None:
        """
        Build or rebuild the path and hash maps.

        This method populates the internal path and hash maps that enable
        efficient lookups during tree merging.

        Args:
            new_tree: Optional new tree to add directly to the maps

        Example:
            ```python
            # Internal use to initialize maps with a new tree
            self._build_maps(new_tree=first_tree)

            # Or to rebuild all maps after changes
            self._build_maps()
            ```
        """
        self._path_map.clear()
        self._hash_map.clear()

        if new_tree:
            self._build_path_and_hash_maps(new_tree, self._path_map, self._hash_map)
        else:
            for tree in self._merged_trees:
                self._build_path_and_hash_maps(tree, self._path_map, self._hash_map)
        self._propagate_uris_by_hash()

    def _propagate_uris_by_hash(self) -> None:
        """
        Ensure all files with the same hash have the same URI.

        This function ensures that if multiple file nodes have the same hash,
        but only some have URIs, the URI is propagated to all instances.
        """
        for file_nodes in self._hash_map.values():
            if len(file_nodes) <= 1:
                continue

            uri = next((node["uri"] for node in file_nodes if node["uri"]), "")
            if not uri:
                continue

            for node in file_nodes:
                if not node["uri"]:
                    node["uri"] = uri

    def _build_path_and_hash_maps(
        self,
        node: DirectoryNode | FileNode,
        path_map: dict[str, DirectoryNode | FileNode],
        hash_map: dict[str, list[FileNode]],
    ) -> None:
        """
        Build both path and hash maps simultaneously.

        This method recursively processes a node (file or directory) and adds
        it to the appropriate maps.

        Args:
            node: The node to process
            path_map: Map of paths to nodes
            hash_map: Map of file hashes to file nodes

        Example:
            ```python
            # Internal use to build maps for a tree
            path_map = {}
            hash_map = {}
            self._build_path_and_hash_maps(
                node=root_tree,
                path_map=path_map,
                hash_map=hash_map
            )
            # After: path_map contains all paths, hash_map contains all file hashes
            ```
        """
        if node["type"] == "dir":
            # Add directory to path map
            dir_node = cast("DirectoryNode", node)
            dir_path = dir_node["dir_path"]
            path_map[dir_path] = dir_node

            # Process children
            for child in dir_node["children"]:
                self._build_path_and_hash_maps(child, path_map, hash_map)
        else:  # File node
            # Add file to path map
            file_node = cast("FileNode", node)
            file_path = file_node["final_real_path"]
            path_map[file_path] = file_node

            # Add file to hash map
            file_hash = file_node["hash"]
            if file_hash not in hash_map:
                hash_map[file_hash] = []
            hash_map[file_hash].append(file_node)

    def _merge_directory_nodes(self, target_dir: DirectoryNode, source_dir: DirectoryNode) -> None:
        """
        Merge contents from source directory into target directory.

        This combines children from both directories, handling duplicates
        and updating files as needed.

        Args:
            target_dir: Directory to merge into
            source_dir: Directory to merge from

        Example:
            ```python
            # Internal use to merge two directory nodes
            self._merge_directory_nodes(
                target_dir={"type": "dir", "dir_path": "/data", "children": [...]},
                source_dir={"type": "dir", "dir_path": "/data", "children": [...]}
            )
            # After: target_dir contains all children from both directories
            ```
        """
        # Delegate file and directory processing to separate methods to reduce branches
        path_to_index, hash_to_index = self._build_indices(target_dir)

        # Process each child from source
        for source_child in source_dir["children"]:
            if source_child["type"] == "dir":
                self._merge_directory_child(
                    target_dir,
                    cast("DirectoryNode", source_child),
                    path_to_index,
                )
            else:  # file
                self._merge_file_child(
                    target_dir,
                    cast("FileNode", source_child),
                    path_to_index,
                    hash_to_index,
                )

        # Update hash
        self._update_directory_hash(target_dir)

    def _build_indices(self, dir_node: DirectoryNode) -> tuple[dict[str, int], dict[str, int]]:
        """
        Build indices for efficient child lookups.

        Returns:
            A tuple of (path_to_index, hash_to_index) dictionaries
        """
        path_to_index: dict[str, int] = {}
        hash_to_index: dict[str, int] = {}

        for i, child in enumerate(dir_node["children"]):
            if child["type"] == "dir":
                path_to_index[cast("DirectoryNode", child)["dir_path"]] = i
            else:  # file
                file_child = cast("FileNode", child)
                path_to_index[file_child["final_real_path"]] = i
                hash_to_index[file_child["hash"]] = i

        return path_to_index, hash_to_index

    def _merge_directory_child(
        self,
        target_dir: DirectoryNode,
        source_dir: DirectoryNode,
        path_to_index: dict[str, int],
    ) -> None:
        """Merge a directory child from source into target directory."""
        dir_path = source_dir["dir_path"]
        if dir_path in path_to_index:
            # Directory exists in both - merge recursively
            index = path_to_index[dir_path]
            existing_child = target_dir["children"][index]
            if existing_child["type"] == "dir":
                self._merge_directory_nodes(
                    cast("DirectoryNode", existing_child),
                    source_dir,
                )
        else:
            # Directory only in source - add to target
            target_dir["children"].append(source_dir)

    def _merge_file_child(
        self,
        target_dir: DirectoryNode,
        source_file: FileNode,
        path_to_index: dict[str, int],
        hash_to_index: dict[str, int],
    ) -> None:
        """Merge a file child from source into target directory."""
        file_path = source_file["final_real_path"]
        file_hash = source_file["hash"]

        if file_path in path_to_index:
            # File exists at same path - update if hash differs
            index = path_to_index[file_path]
            existing_child = target_dir["children"][index]
            if existing_child["hash"] != file_hash:
                target_dir["children"][index] = source_file
            elif existing_child["type"] == "file":
                # Same file - propagate URI if needed
                self._propagate_uri(cast("FileNode", existing_child), source_file)
        elif file_hash in hash_to_index:
            # Same file content exists but at different path
            index = hash_to_index[file_hash]
            existing_child = target_dir["children"][index]
            if existing_child["type"] == "file":
                # Propagate URI if needed
                self._propagate_uri(cast("FileNode", existing_child), source_file)

            if source_file["uri"] and file_hash in self._hash_map:
                for other_file in self._hash_map[file_hash]:
                    if not other_file["uri"]:
                        other_file["uri"] = source_file["uri"]
            target_dir["children"].append(source_file)
        else:
            # File only in source - add to target
            target_dir["children"].append(source_file)

    def _update_directory_hash(self, dir_node: DirectoryNode) -> str:
        """
        Update the hash of a directory based on its children.

        This computes a content-based hash for a directory by combining
        the hashes of all its children.

        Args:
            dir_node: The directory to update

        Returns:
            The updated hash

        Example:
            ```python
            # Internal use to compute directory hash after changes
            new_hash = self._update_directory_hash(
                dir_node={"type": "dir", "children": [...]}
            )
            # After: dir_node["hash"] is updated and returned
            ```
        """
        child_hashes = []

        for child in dir_node["children"]:
            if child["type"] == "file":
                child_hashes.append(cast("FileNode", child)["hash"])
            else:
                child_hash = self._update_directory_hash(cast("DirectoryNode", child))
                child_hashes.append(child_hash)

        child_hashes.sort()  # Ensure consistent hash regardless of order
        hash_input = "|".join(child_hashes)
        dir_hash = hashlib.sha1(hash_input.encode()).hexdigest()[:16]  # noqa: S324 # nosec

        dir_node["hash"] = dir_hash
        return dir_hash

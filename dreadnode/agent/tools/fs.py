import asyncio
import os
import re
import typing as t
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import aiofiles  # type: ignore[import-untyped]
import rigging as rg
from botocore.exceptions import BotoCoreError, ClientError  # type: ignore[import-untyped]
from fsspec import AbstractFileSystem  # type: ignore[import-untyped]
from loguru import logger
from pydantic import PrivateAttr
from upath import UPath

from dreadnode.agent.tools import Toolset, tool_method
from dreadnode.common_types import AnyDict
from dreadnode.meta import Config
from dreadnode.util import shorten_string

MAX_GREP_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


@dataclass
class FilesystemItem:
    """Item in the filesystem"""

    type: t.Literal["file", "dir"]
    name: str
    size: int | None = None
    modified: str | None = None  # Last modified time

    @classmethod
    def from_path(cls, path: "UPath", relative_base: "UPath") -> "FilesystemItem":
        """Create an Item from a UPath.

        Args:
            path: The UPath to create an item from
            relative_base: The base path to calculate relative paths from

        Returns:
            FilesystemItem representing the path

        Raises:
            ValueError: If the path is neither a file nor a directory
        """
        base_path: str = str(relative_base.resolve())
        full_path: str = str(path.resolve())
        relative: str = full_path[len(base_path) :]

        if path.is_dir():
            return cls(type="dir", name=relative, size=None, modified=None)

        if path.is_file():
            return cls(
                type="file",
                name=relative,
                size=path.stat().st_size,
                modified=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S",
                ),
            )

        raise ValueError(f"'{relative}' is not a valid file or directory.")


@dataclass
class GrepMatch:
    """Individual search match"""

    path: str
    line_number: int
    line: str
    context: list[str]


class FilesystemBase(Toolset):
    """
    Base class for filesystem operations with common interface.

    This abstract base class defines the standard interface for filesystem operations
    and provides common utilities like path resolution and validation.
    """

    path: str | Path | UPath = Config(default=Path.cwd(), expose_as=str | Path)
    """Base path to work from."""
    fs_options: AnyDict | None = Config(default=None)
    """Extra options for the universal filesystem."""
    multi_modal: bool = Config(default=False)
    """Enable returning non-text context like images."""
    max_concurrent_reads: int = Config(default=25)
    """Maximum number of concurrent file reads for grep operations."""

    variant: t.Literal["read", "write"] = Config(default="read")

    _fs: AbstractFileSystem = PrivateAttr()
    _upath: "UPath" = PrivateAttr()

    def model_post_init(self, _: t.Any) -> None:
        self._upath = (
            self.path
            if isinstance(self.path, UPath)
            else UPath(str(self.path), **(self.fs_options or {}))
        )
        self.path = self._upath.resolve()
        self._fs = self._upath.fs

    def _resolve(self, path: str) -> "UPath":
        """Resolve a relative path to an absolute UPath within the base path.

        Args:
            path: Relative path to resolve

        Returns:
            Resolved UPath object within the base path

        Raises:
            ValueError: If the resolved path is outside the base path
        """
        full_path: UPath = (self._upath / path.lstrip("/")).resolve()

        # Check if the resolved path starts with the base path
        if not str(full_path).startswith(str(self.path)):
            raise ValueError(f"'{path}' is not accessible.")

        full_path._fs_cached = self._fs  # noqa: SLF001

        return full_path

    def _relative(self, path: "UPath") -> str:
        """Get the path relative to the base path.

        Args:
            path: UPath object to make relative

        Returns:
            String representation of the relative path

        Note:
            Uses string slicing instead of relative_to() due to UPath compatibility issues.
        """
        base_path: str = str(self._upath.resolve())
        full_path: str = str(path.resolve())
        return full_path[len(base_path) :]

    # Methods that must be implemented by subclasses
    # Note: Cannot use @abc.abstractmethod with @tool_method due to decorator conflicts
    # Subclasses must override these methods to provide implementation

    async def read_file(
        self, path: t.Annotated[str, "Path to the file to read"]
    ) -> rg.ContentImageUrl | str:
        """Must be implemented in subclasses"""
        raise NotImplementedError("Subclasses must implement")

    # Common methods that work for all filesystem types (use UPath native methods)

    @tool_method(variants=["read", "write"], catch=True)
    async def ls(
        self,
        path: t.Annotated[str, "Directory path to list"] = "",
    ) -> list[FilesystemItem]:
        """List the contents of a directory."""
        _path = self._resolve(path)

        if not _path.exists():
            raise ValueError(f"'{path}' not found.")

        if not _path.is_dir():
            raise ValueError(f"'{path}' is not a directory.")

        items = await asyncio.to_thread(lambda: list(_path.iterdir()))
        return [FilesystemItem.from_path(item, self._upath) for item in items]

    @tool_method(catch=True)
    async def glob(
        self,
        pattern: t.Annotated[str, "Glob pattern for file matching"],
    ) -> list[FilesystemItem]:
        """
        Returns a list of paths matching a valid glob pattern. The pattern can
        include ** for recursive matching, such as '/path/**/dir/*.py'.
        """
        matches = await asyncio.to_thread(lambda: list(self._upath.glob(pattern)))

        # Check to make sure all matches are within the base path
        for match in matches:
            if not str(match).startswith(str(self._upath)):
                raise ValueError(f"'{pattern}' is not valid.")

        return [FilesystemItem.from_path(match, self._upath) for match in matches]

    @tool_method(variants=["write"], catch=True)
    async def mkdir(
        self,
        path: t.Annotated[str, "Directory path to create"],
    ) -> FilesystemItem:
        """Create a directory and any necessary parent directories."""
        dir_path = self._resolve(path)
        await asyncio.to_thread(lambda: dir_path.mkdir(parents=True, exist_ok=True))

        return FilesystemItem.from_path(dir_path, self._upath)

    @tool_method(variants=["read", "write"], catch=True)
    async def grep(
        self,
        pattern: t.Annotated[str, "Regular expression pattern to search for"],
        path: t.Annotated[str, "File or directory path to search in"],
        *,
        max_results: t.Annotated[int, "Maximum number of results to return"] = 100,
        recursive: t.Annotated[bool, "Search recursively in directories"] = False,
    ) -> list[GrepMatch | str]:
        """
        Search for pattern in files and return matches with line numbers and context.

        For directories, all text files will be searched.
        """
        regex = re.compile(pattern, re.IGNORECASE)

        target_path = self._resolve(path)
        if not target_path.exists():
            raise ValueError(f"'{path}' not found.")

        # Determine files to search
        files_to_search: list[UPath] = []
        if target_path.is_file():
            files_to_search.append(target_path)
        elif target_path.is_dir():
            files_to_search.extend(
                await asyncio.to_thread(
                    lambda: list(target_path.rglob("*") if recursive else target_path.glob("*"))
                ),
            )

        # Filter to files only and check size
        files_to_search = [
            f for f in files_to_search if f.is_file() and f.stat().st_size <= MAX_GREP_FILE_SIZE
        ]

        async def search_file(file_path: UPath) -> list[GrepMatch | str]:
            """Search a single file for matches."""
            file_matches: list[GrepMatch | str] = []
            try:
                # Use the subclass's read_file method
                content = await self.read_file(self._relative(file_path))
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
                elif isinstance(content, rg.ContentImageUrl):
                    # Can't grep images
                    return []

                lines = content.splitlines(keepends=True)

                for i, line in enumerate(lines):
                    if regex.search(line):
                        line_num = i + 1
                        context_start = max(0, i - 1)
                        context_end = min(len(lines), i + 2)
                        context = []

                        for j in range(context_start, context_end):
                            prefix = ">" if j == i else " "
                            line_text = lines[j].rstrip("\r\n")
                            context.append(f"{prefix} {j + 1}: {shorten_string(line_text, 80)}")

                        rel_path = self._relative(file_path)
                        file_matches.append(
                            GrepMatch(
                                path=rel_path,
                                line_number=line_num,
                                line=shorten_string(line.rstrip("\r\n"), 80),
                                context=context,
                            ),
                        )
            except (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                UnicodeDecodeError,
                OSError,
                ValueError,
            ) as e:
                file_matches.append(f"Error occurred while searching file {file_path}: {e}")

            return file_matches

        # Search files in parallel with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent_reads)

        async def search_file_limited(file_path: UPath) -> list[GrepMatch | str]:
            """Search a single file with semaphore to limit concurrency."""
            async with semaphore:
                return await search_file(file_path)

        all_matches: list[GrepMatch | str] = []
        results = await asyncio.gather(
            *[search_file_limited(file_path) for file_path in files_to_search]
        )

        # Flatten results and respect max_results
        for file_matches in results:
            all_matches.extend(file_matches)
            if len(all_matches) >= max_results:
                break

        return all_matches[:max_results]


class LocalFilesystem(FilesystemBase):
    """
    Local filesystem implementation using aiofiles.

    Supports operations on the local disk using async file I/O.
    """

    async def _safe_create_file(self, path: str) -> "UPath":
        """
        Safely create a file and its parent directories if they don't exist.

        Args:
            path: Path to the file to create

        Returns:
            UPath: The resolved path to the created file
        """
        file_path = self._resolve(path)

        parent_path = file_path.parent
        if not parent_path.exists():
            await asyncio.to_thread(lambda: parent_path.mkdir(parents=True, exist_ok=True))

        if not file_path.exists():
            await asyncio.to_thread(file_path.touch)

        return file_path

    @tool_method(variants=["read", "write"], catch=True)
    async def read_file(
        self,
        path: t.Annotated[str, "Path to the file to read"],
    ) -> rg.ContentImageUrl | str:
        """
        Read a file and return its contents.

        Returns:
            - str: The file contents decoded as UTF-8 if possible.
            - rg.ContentImageUrl: If the file is non-text and multi_modal is True.

        Note:
            Callers should be prepared to handle raw bytes if the file is not valid UTF-8 and multi_modal is False.
        """
        _path = self._resolve(path)
        async with aiofiles.open(_path, "rb") as f:
            content = await f.read()

        try:
            return str(content.decode("utf-8"))
        except UnicodeDecodeError as e:
            if self.multi_modal:
                return rg.ContentImageUrl.from_file(path)
            raise ValueError("File is not a valid text file.") from e

    @tool_method(variants=["read", "write"], catch=True)
    async def read_lines(
        self,
        path: t.Annotated[str, "Path to the file to read"],
        start_line: t.Annotated[int, "Start line number (0-indexed)"] = 0,
        end_line: t.Annotated[int, "End line number"] = -1,
    ) -> str:
        """
        Read a partial file and return the contents with optional line numbers.
        Negative line numbers count from the end.
        """
        _path = self._resolve(path)

        if not _path.exists():
            raise ValueError(f"'{path}' not found.")

        if not _path.is_file():
            raise ValueError(f"'{path}' is not a file.")

        async with aiofiles.open(_path) as f:
            lines = await f.readlines()

            if start_line < 0:
                start_line = len(lines) + start_line

            if end_line < 0:
                end_line = len(lines) + end_line + 1

            start_line = max(0, min(start_line, len(lines)))
            end_line = max(start_line, min(end_line, len(lines)))

            return "\n".join(lines[start_line:end_line])

    @tool_method(variants=["write"], catch=True)
    async def write_file(
        self,
        path: t.Annotated[str, "Path to write the file to"],
        contents: t.Annotated[str, "Content to write to the file"],
    ) -> FilesystemItem:
        """Create or overwrite a file with the given contents."""
        _path = await self._safe_create_file(path)
        async with aiofiles.open(_path, "w") as f:
            await f.write(contents)

        return FilesystemItem.from_path(_path, self._upath)

    @tool_method(variants=["write"], catch=True)
    async def write_file_bytes(
        self,
        path: t.Annotated[str, "Path to write the file to"],
        byte_data: t.Annotated[bytes, "Bytes to write to the file"],
    ) -> FilesystemItem:
        """Create or overwrite a file with the given bytes."""
        _path = await self._safe_create_file(path)
        async with aiofiles.open(_path, "wb") as f:
            await f.write(byte_data)

        return FilesystemItem.from_path(_path, self._upath)

    @tool_method(variants=["write"], catch=True)
    async def write_lines(
        self,
        path: t.Annotated[str, "Path to write to"],
        contents: t.Annotated[str, "Content to write"],
        insert_line: t.Annotated[int, "Line number to insert at (negative counts from end)"] = -1,
        mode: t.Annotated[str, "'insert' or 'overwrite'"] = "insert",
    ) -> FilesystemItem:
        """
        Write content to a specific line in the file.
        Mode can be 'insert' to add lines or 'overwrite' to replace lines.
        """
        if mode not in ["insert", "overwrite"]:
            raise ValueError("Invalid mode. Use 'insert' or 'overwrite'")

        _path = await self._safe_create_file(path)

        lines: list[str] = []
        async with aiofiles.open(_path) as f:
            lines = await f.readlines()

        # Normalize line endings in content
        content_lines = [
            line + "\n" if not line.endswith("\n") else line
            for line in contents.splitlines(keepends=False)
        ]

        # Calculate insert position and ensure it's within bounds
        if insert_line < 0:
            insert_line = len(lines) + insert_line + 1

        insert_line = max(0, min(insert_line, len(lines)))

        # Apply the update
        if mode == "insert":
            lines[insert_line:insert_line] = content_lines
        elif mode == "overwrite":
            lines[insert_line : insert_line + len(content_lines)] = content_lines

        async with aiofiles.open(_path, "w") as f:
            await f.writelines(lines)

        return FilesystemItem.from_path(_path, self._upath)

    @tool_method(variants=["write"], catch=True)
    async def cp(
        self,
        src: t.Annotated[str, "Source file"],
        dest: t.Annotated[str, "Destination path"],
    ) -> FilesystemItem:
        """Copy a file to a new location."""
        src_path = self._resolve(src)
        dest_path = self._resolve(dest)

        if not src_path.exists():
            raise ValueError(f"'{src}' not found")

        if not src_path.is_file():
            raise ValueError(f"'{src}' is not a file")

        await asyncio.to_thread(lambda: dest_path.parent.mkdir(parents=True, exist_ok=True))

        async with (
            aiofiles.open(src_path, "rb") as src_file,
            aiofiles.open(dest_path, "wb") as dest_file,
        ):
            content = await src_file.read()
            await dest_file.write(content)

        return FilesystemItem.from_path(dest_path, self._upath)

    @tool_method(variants=["write"], catch=True)
    async def mv(
        self,
        src: t.Annotated[str, "Source path"],
        dest: t.Annotated[str, "Destination path"],
    ) -> FilesystemItem:
        """Move a file or directory to a new location."""
        src_path = self._resolve(src)
        dest_path = self._resolve(dest)

        if not src_path.exists():
            raise ValueError(f"'{src}' not found")

        await asyncio.to_thread(lambda: dest_path.parent.mkdir(parents=True, exist_ok=True))

        await asyncio.to_thread(lambda: src_path.rename(dest_path))

        return FilesystemItem.from_path(dest_path, self._upath)

    @tool_method(variants=["write"], catch=True)
    async def delete(
        self,
        path: t.Annotated[str, "File or directory"],
    ) -> bool:
        """Delete a file or directory."""
        _path = self._resolve(path)
        if not _path.exists():
            raise ValueError(f"'{path}' not found")

        if _path.is_dir():
            await asyncio.to_thread(_path.rmdir)
        else:
            await asyncio.to_thread(_path.unlink)

        return True


class S3Filesystem(FilesystemBase):
    """
    S3 filesystem implementation using aioboto3.

    Supports operations on AWS S3 buckets with async I/O.
    Requires aioboto3 and properly configured AWS credentials.
    """

    def _get_s3_parts(self, path_obj: "UPath") -> tuple[str, str]:
        """Parse S3 path into bucket and key components.

        Args:
            path_obj: UPath object representing an S3 path

        Returns:
            Tuple of (bucket_name, object_key)
        """
        path_str: str = str(path_obj).replace("s3://", "")
        parts: list[str] = path_str.split("/", 1)
        bucket: str = parts[0]
        key: str = parts[1] if len(parts) > 1 else ""
        return bucket, key

    def _get_session(self) -> t.Any:
        """Get aioboto3 session with profile if available.

        Returns:
            aioboto3.Session: An aioboto3 session with optional profile configuration
        """
        try:
            import aioboto3  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "aioboto3 is required for S3 operations. Install with: pip install aioboto3"
            ) from e

        # Try to get profile from fs_options, then environment
        profile: str | None = None
        if self.fs_options:
            profile = self.fs_options.get("profile")
        if not profile:
            profile = os.environ.get("AWS_PROFILE")

        return aioboto3.Session(profile_name=profile) if profile else aioboto3.Session()

    @tool_method(variants=["read", "write"], catch=True)
    async def read_file(
        self,
        path: t.Annotated[str, "Path to the file to read"],
    ) -> str:
        """
        Read a file from S3 and return its contents.

        Returns:
            - str: The file contents decoded as UTF-8 if possible.

        Note:
            multi_modal support for S3 is limited as we can't easily determine
            image types without downloading. Returns bytes for non-UTF-8 content.
        """
        _path = self._resolve(path)
        bucket, key = self._get_s3_parts(_path)

        session = self._get_session()
        async with session.client("s3") as s3_client:
            response = await s3_client.get_object(Bucket=bucket, Key=key)
            content = await response["Body"].read()

        try:
            return str(content.decode("utf-8"))
        except UnicodeDecodeError as e:
            raise ValueError("File is not a valid text file.") from e

    @tool_method(variants=["read", "write"], catch=True)
    async def read_lines(
        self,
        path: t.Annotated[str, "Path to the file to read"],
        start_line: t.Annotated[int, "Start line number (0-indexed)"] = 0,
        end_line: t.Annotated[int, "End line number"] = -1,
    ) -> str:
        """
        Read a partial file from S3 and return the contents.
        Negative line numbers count from the end.
        """
        content = await self.read_file(path)
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        elif isinstance(content, rg.ContentImageUrl):
            raise TypeError("Cannot read lines from non-text content")

        lines = content.splitlines(keepends=True)

        if start_line < 0:
            start_line = len(lines) + start_line

        if end_line < 0:
            end_line = len(lines) + end_line + 1

        start_line = max(0, min(start_line, len(lines)))
        end_line = max(start_line, min(end_line, len(lines)))

        return "".join(lines[start_line:end_line])

    @tool_method(variants=["write"], catch=True)
    async def write_file(
        self,
        path: t.Annotated[str, "Path to write the file to"],
        contents: t.Annotated[str, "Content to write to the file"],
    ) -> FilesystemItem:
        """Create or overwrite a file in S3 with the given contents."""
        _path = self._resolve(path)
        bucket, key = self._get_s3_parts(_path)

        session = self._get_session()
        async with session.client("s3") as s3_client:
            await s3_client.put_object(Bucket=bucket, Key=key, Body=contents.encode("utf-8"))

        # Return FilesystemItem without calling stat (S3 put is async)
        relative = self._relative(_path)
        return FilesystemItem(
            type="file",
            name=relative,
            size=len(contents.encode("utf-8")),
            modified=None,
        )

    @tool_method(variants=["write"], catch=True)
    async def write_file_bytes(
        self,
        path: t.Annotated[str, "Path to write the file to"],
        byte_data: t.Annotated[bytes, "Bytes to write to the file"],
    ) -> FilesystemItem:
        """Create or overwrite a file in S3 with the given bytes."""
        _path = self._resolve(path)
        bucket, key = self._get_s3_parts(_path)

        session = self._get_session()
        async with session.client("s3") as s3_client:
            await s3_client.put_object(Bucket=bucket, Key=key, Body=byte_data)

        # Return FilesystemItem without calling stat (S3 put is async)
        relative = self._relative(_path)
        return FilesystemItem(type="file", name=relative, size=len(byte_data), modified=None)

    @tool_method(variants=["write"], catch=True)
    async def write_lines(
        self,
        path: t.Annotated[str, "Path to write to"],
        contents: t.Annotated[str, "Content to write"],
        insert_line: t.Annotated[int, "Line number to insert at (negative counts from end)"] = -1,
        mode: t.Annotated[str, "'insert' or 'overwrite'"] = "insert",
    ) -> FilesystemItem | str:
        """
        Write content to a specific line in an S3 file.
        Mode can be 'insert' to add lines or 'overwrite' to replace lines.
        """
        if mode not in ["insert", "overwrite"]:
            raise TypeError("Invalid mode. Use 'insert' or 'overwrite'")

        # Read existing content
        try:
            existing_content = await self.read_file(path)
            if isinstance(existing_content, bytes):
                existing_content = existing_content.decode("utf-8")
            elif isinstance(existing_content, rg.ContentImageUrl):
                logger.warning("Cannot write lines to non-text content")
                lines = []
            lines = existing_content.splitlines(keepends=True)
        except FileNotFoundError:
            # File doesn't exist, start with empty lines
            lines = []
        except (PermissionError, IsADirectoryError, ClientError, BotoCoreError, ValueError) as e:
            # File doesn't exist or can't be read, start with empty lines
            return f"Error occurred while trying to write to the supplied filepath {path}: {e}"

        # Normalize line endings in content
        content_lines = [
            line + "\n" if not line.endswith("\n") else line
            for line in contents.splitlines(keepends=False)
        ]

        # Calculate insert position and ensure it's within bounds
        if insert_line < 0:
            insert_line = len(lines) + insert_line + 1

        insert_line = max(0, min(insert_line, len(lines)))

        # Apply the update
        if mode == "insert":
            lines[insert_line:insert_line] = content_lines
        elif mode == "overwrite":
            lines[insert_line : insert_line + len(content_lines)] = content_lines

        # Write back
        new_content = "".join(lines)
        return await self.write_file(path, new_content)

    @tool_method(variants=["write"], catch=True)
    async def cp(
        self,
        src: t.Annotated[str, "Source file"],
        dest: t.Annotated[str, "Destination path"],
    ) -> FilesystemItem:
        """Copy a file to a new location within S3."""
        src_path = self._resolve(src)
        dest_path = self._resolve(dest)

        if not src_path.exists():
            raise ValueError(f"'{src}' not found")

        if not src_path.is_file():
            raise ValueError(f"'{src}' is not a file")

        src_bucket, src_key = self._get_s3_parts(src_path)
        dest_bucket, dest_key = self._get_s3_parts(dest_path)

        session = self._get_session()
        async with session.client("s3") as s3_client:
            # Use S3 copy_object for efficient server-side copy
            copy_source = {"Bucket": src_bucket, "Key": src_key}
            await s3_client.copy_object(CopySource=copy_source, Bucket=dest_bucket, Key=dest_key)

        # Return FilesystemItem without calling stat
        relative = self._relative(dest_path)
        return FilesystemItem(type="file", name=relative, size=None, modified=None)

    @tool_method(variants=["write"], catch=True)
    async def mv(
        self,
        src: t.Annotated[str, "Source path"],
        dest: t.Annotated[str, "Destination path"],
    ) -> FilesystemItem:
        """Move a file to a new location within S3 (copy then delete)."""
        # Copy to destination
        result = await self.cp(src, dest)

        # Delete source
        await self.delete(src)

        return result

    @tool_method(variants=["write"], catch=True)
    async def mkdir(
        self,
        path: t.Annotated[str, "Directory path to create"],
    ) -> FilesystemItem:
        """
        Create a directory marker in S3.

        Note: S3 doesn't have true directories. This creates an empty object
        with a trailing slash to simulate a directory for compatibility.
        """
        _path = self._resolve(path)
        bucket, key = self._get_s3_parts(_path)

        # Ensure key ends with slash for directory marker
        if not key.endswith("/"):
            key += "/"

        session = self._get_session()
        async with session.client("s3") as s3_client:
            # Create empty object with trailing slash
            await s3_client.put_object(Bucket=bucket, Key=key, Body=b"")

        relative = self._relative(_path)
        return FilesystemItem(type="dir", name=relative, size=None, modified=None)

    @tool_method(variants=["write"], catch=True)
    async def delete(
        self,
        path: t.Annotated[str, "File or directory"],
    ) -> bool:
        """Delete a file from S3."""
        _path = self._resolve(path)

        if not _path.exists():
            raise ValueError(f"'{path}' not found")

        bucket, key = self._get_s3_parts(_path)

        session = self._get_session()
        async with session.client("s3") as s3_client:
            await s3_client.delete_object(Bucket=bucket, Key=key)

        return True


def Filesystem(  # noqa: N802
    path: str | Path | UPath, **kwargs: t.Any
) -> LocalFilesystem | S3Filesystem:
    """
    Factory function to create the appropriate filesystem instance based on path.

    Automatically detects the filesystem type from the path protocol and returns
    the corresponding implementation (LocalFilesystem or S3Filesystem).

    Args:
        path: Local path, S3 URL (s3://), or other supported protocol
        **kwargs: Additional arguments passed to the filesystem constructor

    Returns:
        LocalFilesystem for local paths, S3Filesystem for S3 URLs

    Examples:
        >>> # Local filesystem
        >>> fs = Filesystem(path="/tmp/data")
        >>> isinstance(fs, LocalFilesystem)
        True

        >>> # S3 filesystem
        >>> fs = Filesystem(path="s3://my-bucket/data")
        >>> isinstance(fs, S3Filesystem)
        True
    """
    # Check if it's a string starting with s3://
    if isinstance(path, str) and path.startswith("s3://"):
        return S3Filesystem(path=path, **kwargs)

    # Check if it's a UPath with S3 protocol
    if isinstance(path, UPath) and path.protocol in ["s3", "s3a"]:
        return S3Filesystem(path=path, **kwargs)

    # Try to create UPath and check protocol
    try:
        fs_options = kwargs.get("fs_options", {})
        upath = UPath(str(path), **fs_options)
        if upath.protocol in ["s3", "s3a"]:
            return S3Filesystem(path=path, **kwargs)
    except (TypeError, ValueError) as e:
        # If UPath creation fails, fall through to local
        logger.warning(
            f"Upath initialization failed ({type(e).__name__}: {e}), defaulting to local path"
        )
        return LocalFilesystem(path=path, **kwargs)

    # Default to local filesystem
    return LocalFilesystem(path=path, **kwargs)

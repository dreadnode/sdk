import asyncio
import re
import typing as t
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
import rigging as rg
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
        """Create an Item from a UPath"""

        base_path = str(relative_base.resolve())
        full_path = str(path.resolve())
        relative = full_path[len(base_path) :]

        if path.is_dir():
            return cls(type="dir", name=relative, size=None, modified=None)

        if path.is_file():
            return cls(
                type="file",
                name=relative,
                size=path.stat().st_size,
                modified=datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).strftime(
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


class Filesystem(Toolset):
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
    _upath: UPath = PrivateAttr()

    def model_post_init(self, _: t.Any) -> None:
        self._upath = (
            self.path
            if isinstance(self.path, UPath)
            else UPath(str(self.path), **(self.fs_options or {}))
        )
        self.path = self._upath.resolve()
        self._fs = self._upath.fs

    def _resolve(self, path: str) -> "UPath":
        full_path = (self._upath / path.lstrip("/")).resolve()

        # Check if the resolved path starts with the base path
        if not str(full_path).startswith(str(self.path)):
            raise ValueError(f"'{path}' is not accessible.")

        full_path._fs_cached = self._fs  # noqa: SLF001

        return full_path

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
            await asyncio.to_thread(
                lambda: parent_path.mkdir(parents=True, exist_ok=True)
            )

        if not file_path.exists():
            await asyncio.to_thread(file_path.touch)

        return file_path

    def _relative(self, path: "UPath") -> str:
        """
        Get the path relative to the base path.
        """
        # Would prefer relative_to here, but it's very flaky with UPath
        base_path = str(self._upath.resolve())
        full_path = str(path.resolve())
        return full_path[len(base_path) :]

    @tool_method(variants=["read", "write"], catch=True)
    async def read_file(
        self,
        path: t.Annotated[str, "Path to the file to read"],
    ) -> rg.ContentImageUrl | str | bytes:
        """
        Read a file and return its contents.

        Returns:
            - str: The file contents decoded as UTF-8 if possible.
            - rg.ContentImageUrl: If the file is non-text and multi_modal is True.
            - bytes: If the file cannot be decoded as UTF-8 and multi_modal is False.

        Note:
            Callers should be prepared to handle raw bytes if the file is not valid UTF-8 and multi_modal is False.
        """
        _path = self._resolve(path)
        async with aiofiles.open(_path, "rb") as f:
            content = await f.read()

        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            if self.multi_modal:
                return rg.ContentImageUrl.from_file(path)
            return content

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

        async with aiofiles.open(_path, "r") as f:
            lines = await f.readlines()

            if start_line < 0:
                start_line = len(lines) + start_line

            if end_line < 0:
                end_line = len(lines) + end_line + 1

            start_line = max(0, min(start_line, len(lines)))
            end_line = max(start_line, min(end_line, len(lines)))

            return "\n".join(lines[start_line:end_line])

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

    @tool_method(variants=["read", "write"], catch=True)
    async def grep(
        self,
        pattern: t.Annotated[str, "Regular expression pattern to search for"],
        path: t.Annotated[str, "File or directory path to search in"],
        *,
        max_results: t.Annotated[int, "Maximum number of results to return"] = 100,
        recursive: t.Annotated[bool, "Search recursively in directories"] = False,
    ) -> list[GrepMatch]:
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
                    lambda: list(
                        target_path.rglob("*") if recursive else target_path.glob("*")
                    )
                ),
            )

        # Filter to files only and check size
        files_to_search = [
            f
            for f in files_to_search
            if f.is_file() and f.stat().st_size <= MAX_GREP_FILE_SIZE
        ]

        async def search_file(file_path: UPath) -> list[GrepMatch]:
            """Search a single file for matches."""
            file_matches: list[GrepMatch] = []
            try:
                async with aiofiles.open(file_path, "r") as f:
                    lines = await f.readlines()

                for i, line in enumerate(lines):
                    if regex.search(line):
                        line_num = i + 1
                        context_start = max(0, i - 1)
                        context_end = min(len(lines), i + 2)
                        context = []

                        for j in range(context_start, context_end):
                            prefix = ">" if j == i else " "
                            line_text = lines[j].rstrip("\r\n")
                            context.append(
                                f"{prefix} {j + 1}: {shorten_string(line_text, 80)}"
                            )

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
            ) as e:
                logger.warning(f"Error occurred while searching file {file_path}: {e}")

            return file_matches

        # Search files in parallel with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent_reads)

        async def search_file_limited(file_path: UPath) -> list[GrepMatch]:
            """Search a single file with semaphore to limit concurrency."""
            async with semaphore:
                return await search_file(file_path)

        all_matches: list[GrepMatch] = []
        results = await asyncio.gather(
            *[search_file_limited(file_path) for file_path in files_to_search]
        )

        # Flatten results and respect max_results
        for file_matches in results:
            all_matches.extend(file_matches)
            if len(all_matches) >= max_results:
                break

        return all_matches[:max_results]

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
        bytes: t.Annotated[bytes, "Bytes to write to the file"],
    ) -> FilesystemItem:
        """Create or overwrite a file with the given bytes."""
        _path = await self._safe_create_file(path)
        async with aiofiles.open(_path, "wb") as f:
            await f.write(bytes)

        return FilesystemItem.from_path(_path, self._upath)

    @tool_method(variants=["write"], catch=True)
    async def write_lines(
        self,
        path: t.Annotated[str, "Path to write to"],
        contents: t.Annotated[str, "Content to write"],
        insert_line: t.Annotated[
            int, "Line number to insert at (negative counts from end)"
        ] = -1,
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
        async with aiofiles.open(_path, "r") as f:
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
    async def mkdir(
        self,
        path: t.Annotated[str, "Directory path to create"],
    ) -> FilesystemItem:
        """Create a directory and any necessary parent directories."""
        dir_path = self._resolve(path)
        await asyncio.to_thread(lambda: dir_path.mkdir(parents=True, exist_ok=True))

        return FilesystemItem.from_path(dir_path, self._upath)

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

        await asyncio.to_thread(
            lambda: dest_path.parent.mkdir(parents=True, exist_ok=True)
        )

        await asyncio.to_thread(lambda: src_path.rename(dest_path))

        return FilesystemItem.from_path(dest_path, self._upath)

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

        await asyncio.to_thread(
            lambda: dest_path.parent.mkdir(parents=True, exist_ok=True)
        )

        async with (
            aiofiles.open(src_path, "rb") as src_file,
            aiofiles.open(dest_path, "wb") as dest_file,
        ):
            content = await src_file.read()
            await dest_file.write(content)

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

import contextlib
import re
import typing as t
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import rigging as rg
from fsspec import AbstractFileSystem  # type: ignore[import-untyped]
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


class Filesystem(Toolset):
    path: str | Path | UPath = Config(default=Path.cwd(), expose_as=str | Path)
    """Base path to work from."""
    fs_options: AnyDict | None = Config(default=None)
    """Extra options for the universal filesystem."""
    multi_modal: bool = Config(default=False)
    """Enable returning non-text context like images."""

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

    def _safe_create_file(self, path: str) -> "UPath":
        file_path = self._resolve(path)

        parent_path = file_path.parent
        if not parent_path.exists():
            parent_path.mkdir(parents=True, exist_ok=True)

        if not file_path.exists():
            file_path.touch()

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
    def read_file(
        self,
        path: t.Annotated[str, "Path to the file to read"],
    ) -> rg.ContentImageUrl | str:
        """Read a file and return its contents."""
        _path = self._resolve(path)
        content = _path.read_bytes()

        try:
            return content.decode("utf-8")
        except UnicodeDecodeError as e:
            if self.multi_modal:
                return rg.ContentImageUrl.from_file(path)
            raise ValueError("File is not a valid text file.") from e

    @tool_method(variants=["read", "write"], catch=True)
    def read_lines(
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

        with _path.open("r") as f:
            lines = f.readlines()

            if start_line < 0:
                start_line = len(lines) + start_line

            if end_line < 0:
                end_line = len(lines) + end_line + 1

            start_line = max(0, min(start_line, len(lines)))
            end_line = max(start_line, min(end_line, len(lines)))

            return "\n".join(lines[start_line:end_line])

    @tool_method(variants=["read", "write"], catch=True)
    def ls(
        self,
        path: t.Annotated[str, "Directory path to list"] = "",
    ) -> list[FilesystemItem]:
        """List the contents of a directory."""
        _path = self._resolve(path)

        if not _path.exists():
            raise ValueError(f"'{path}' not found.")

        if not _path.is_dir():
            raise ValueError(f"'{path}' is not a directory.")

        items = list(_path.iterdir())
        return [FilesystemItem.from_path(item, self._upath) for item in items]

    @tool_method(catch=True)
    def glob(
        self,
        pattern: t.Annotated[str, "Glob pattern for file matching"],
    ) -> list[FilesystemItem]:
        """
        Returns a list of paths matching a valid glob pattern. The pattern can
        include ** for recursive matching, such as '/path/**/dir/*.py'.
        """
        matches = list(self._upath.glob(pattern))

        # Check to make sure all matches are within the base path
        for match in matches:
            if not str(match).startswith(str(self._upath)):
                raise ValueError(f"'{pattern}' is not valid.")

        return [FilesystemItem.from_path(match, self._upath) for match in matches]

    @tool_method(variants=["read", "write"], catch=True)
    def grep(
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
                list(target_path.rglob("*") if recursive else target_path.glob("*")),
            )

        matches: list[GrepMatch] = []
        for file_path in [f for f in files_to_search if f.is_file()]:
            if len(matches) >= max_results:
                break

            if file_path.stat().st_size > MAX_GREP_FILE_SIZE:
                continue

            with contextlib.suppress(Exception):
                with file_path.open("r") as f:
                    lines = f.readlines()

                for i, line in enumerate(lines):
                    if len(matches) >= max_results:
                        break

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
                        matches.append(
                            GrepMatch(
                                path=rel_path,
                                line_number=line_num,
                                line=shorten_string(line.rstrip("\r\n"), 80),
                                context=context,
                            ),
                        )

        return matches

    @tool_method(variants=["write"], catch=True)
    def write_file(
        self,
        path: t.Annotated[str, "Path to write the file to"],
        contents: t.Annotated[str, "Content to write to the file"],
    ) -> FilesystemItem:
        """Create or overwrite a file with the given contents."""
        _path = self._safe_create_file(path)
        with _path.open("w") as f:
            f.write(contents)

        return FilesystemItem.from_path(_path, self._upath)

    @tool_method(variants=["write"], catch=True)
    def write_lines(
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

        _path = self._safe_create_file(path)

        lines: list[str] = []
        with _path.open("r") as f:
            lines = f.readlines()

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

        with _path.open("w") as f:
            f.writelines(lines)

        return FilesystemItem.from_path(_path, self._upath)

    @tool_method(variants=["write"], catch=True)
    def mkdir(
        self,
        path: t.Annotated[str, "Directory path to create"],
    ) -> FilesystemItem:
        """Create a directory and any necessary parent directories."""
        dir_path = self._resolve(path)
        dir_path.mkdir(parents=True, exist_ok=True)

        return FilesystemItem.from_path(dir_path, self._upath)

    @tool_method(variants=["write"], catch=True)
    def mv(
        self,
        src: t.Annotated[str, "Source path"],
        dest: t.Annotated[str, "Destination path"],
    ) -> FilesystemItem:
        """Move a file or directory to a new location."""
        src_path = self._resolve(src)
        dest_path = self._resolve(dest)

        if not src_path.exists():
            raise ValueError(f"'{src}' not found")

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        src_path.rename(dest_path)

        return FilesystemItem.from_path(dest_path, self._upath)

    @tool_method(variants=["write"], catch=True)
    def cp(
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

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        with src_path.open("rb") as src_file, dest_path.open("wb") as dest_file:
            dest_file.write(src_file.read())

        return FilesystemItem.from_path(dest_path, self._upath)

    @tool_method(variants=["write"], catch=True)
    def delete(
        self,
        path: t.Annotated[str, "File or directory"],
    ) -> bool:
        """Delete a file or directory."""
        _path = self._resolve(path)
        if not _path.exists():
            raise ValueError(f"'{path}' not found")

        if _path.is_dir():
            _path.rmdir()
        else:
            _path.unlink()

        return True

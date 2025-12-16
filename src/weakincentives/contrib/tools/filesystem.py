# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Filesystem protocol and backend implementations for workspace operations.

This module provides a unified `Filesystem` protocol that abstracts over
workspace backends (in-memory VFS, Podman containers, host filesystem) so
tool handlers can perform file operations without coupling to a specific
storage implementation.

Example usage::

    from weakincentives.contrib.tools.filesystem import (
        Filesystem,
        InMemoryFilesystem,
    )

    # Create an in-memory filesystem
    fs = InMemoryFilesystem()

    # Write and read files
    fs.write("src/main.py", "print('hello')")
    result = fs.read("src/main.py")
    assert result.content == "print('hello')"

    # Use glob and grep
    matches = fs.glob("**/*.py")
    grep_results = fs.grep(r"print", path="src")
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final, Literal, Protocol, runtime_checkable

_DEFAULT_READ_LIMIT: Final[int] = 2_000
_MAX_WRITE_LENGTH: Final[int] = 48_000
_MAX_PATH_DEPTH: Final[int] = 16
_MAX_SEGMENT_LENGTH: Final[int] = 80
_MAX_GREP_MATCHES: Final[int] = 1_000
_ASCII: Final[str] = "ascii"


@dataclass(slots=True, frozen=True)
class FileStat:
    """Metadata for a file or directory."""

    path: str
    is_file: bool
    is_directory: bool
    size_bytes: int
    created_at: datetime | None = None
    modified_at: datetime | None = None


@dataclass(slots=True, frozen=True)
class FileEntry:
    """Directory listing entry."""

    name: str
    path: str
    is_file: bool
    is_directory: bool


@dataclass(slots=True, frozen=True)
class GlobMatch:
    """Result from glob operations."""

    path: str
    is_file: bool


@dataclass(slots=True, frozen=True)
class GrepMatch:
    """Result from grep operations."""

    path: str
    line_number: int
    line_content: str
    match_start: int
    match_end: int


@dataclass(slots=True, frozen=True)
class ReadResult:
    """Content returned from read operations."""

    content: str
    path: str
    total_lines: int
    offset: int
    limit: int
    truncated: bool


@dataclass(slots=True, frozen=True)
class WriteResult:
    """Confirmation of a write operation."""

    path: str
    bytes_written: int
    mode: Literal["create", "overwrite", "append"]


@runtime_checkable
class Filesystem(Protocol):
    """Unified filesystem protocol for workspace operations."""

    # --- Read Operations ---

    def read(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int | None = None,
        encoding: str = "utf-8",
    ) -> ReadResult:
        """Read file content with optional pagination.

        Args:
            path: Relative path from workspace root.
            offset: Line number to start reading (0-indexed).
            limit: Maximum lines to return. None means backend default.
            encoding: Text encoding. Only "utf-8" is guaranteed.

        Raises:
            FileNotFoundError: Path does not exist.
            IsADirectoryError: Path is a directory.
            PermissionError: Read access denied.
        """
        ...

    def exists(self, path: str) -> bool:
        """Check if a path exists."""
        ...

    def stat(self, path: str) -> FileStat:
        """Get metadata for a path.

        Raises:
            FileNotFoundError: Path does not exist.
        """
        ...

    def list(self, path: str = ".") -> Sequence[FileEntry]:
        """List directory contents.

        Args:
            path: Directory to list. Defaults to root.

        Raises:
            FileNotFoundError: Path does not exist.
            NotADirectoryError: Path is a file.
        """
        ...

    def glob(
        self,
        pattern: str,
        *,
        path: str = ".",
    ) -> Sequence[GlobMatch]:
        """Match files by glob pattern.

        Args:
            pattern: Glob pattern (e.g., "**/*.py").
            path: Base directory for matching.

        Returns:
            Matching paths sorted by path.
        """
        ...

    def grep(
        self,
        pattern: str,
        *,
        path: str = ".",
        glob: str | None = None,
        max_matches: int | None = None,
    ) -> Sequence[GrepMatch]:
        """Search file contents by regex.

        Args:
            pattern: Regular expression pattern.
            path: Base directory for search.
            glob: Optional file filter pattern.
            max_matches: Limit total matches returned.

        Returns:
            Matches sorted by (path, line_number).
        """
        ...

    # --- Write Operations ---

    def write(
        self,
        path: str,
        content: str,
        *,
        mode: Literal["create", "overwrite", "append"] = "overwrite",
        create_parents: bool = True,
    ) -> WriteResult:
        """Write content to a file.

        Args:
            path: Relative path from workspace root.
            content: UTF-8 text content.
            mode: Write behavior.
                - "create": Fail if file exists.
                - "overwrite": Replace existing content.
                - "append": Add to end of file.
            create_parents: Create parent directories if missing.

        Raises:
            FileExistsError: mode="create" and file exists.
            FileNotFoundError: Parent directory missing and create_parents=False.
            PermissionError: Write access denied.
            ValueError: Content exceeds backend limits.
        """
        ...

    def delete(
        self,
        path: str,
        *,
        recursive: bool = False,
    ) -> None:
        """Delete a file or directory.

        Args:
            path: Path to delete.
            recursive: If True, delete directories and contents.

        Raises:
            FileNotFoundError: Path does not exist.
            IsADirectoryError: Path is directory and recursive=False.
            PermissionError: Delete access denied.
        """
        ...

    def mkdir(
        self,
        path: str,
        *,
        parents: bool = True,
        exist_ok: bool = True,
    ) -> None:
        """Create a directory.

        Args:
            path: Directory path to create.
            parents: Create parent directories if missing.
            exist_ok: Don't raise if directory exists.

        Raises:
            FileExistsError: Path exists and exist_ok=False.
            FileNotFoundError: Parent missing and parents=False.
        """
        ...

    # --- Metadata ---

    @property
    def root(self) -> str:
        """Workspace root path (may be "/" for virtual filesystems)."""
        ...

    @property
    def read_only(self) -> bool:
        """True if write operations are disabled."""
        ...


@dataclass(slots=True)
class _InMemoryFile:
    """Internal representation of a file in memory."""

    content: str
    created_at: datetime
    modified_at: datetime


def _now() -> datetime:
    """Return current UTC time truncated to milliseconds."""
    value = datetime.now(UTC)
    microsecond = value.microsecond - value.microsecond % 1000
    return value.replace(microsecond=microsecond, tzinfo=UTC)


def _normalize_path(path: str) -> str:
    """Normalize a path by removing leading/trailing slashes and cleaning segments."""
    if not path or path in {".", "/"}:
        return ""
    stripped = path.strip().strip("/")
    segments = [s for s in stripped.split("/") if s and s != "."]
    # Process .. segments
    result: list[str] = []
    for segment in segments:
        if segment == "..":
            if result:
                _ = result.pop()
        else:
            result.append(segment)
    return "/".join(result)


def _validate_path(path: str) -> None:
    """Validate path constraints."""
    if not path:
        return
    segments = path.split("/")
    if len(segments) > _MAX_PATH_DEPTH:
        msg = f"Path depth exceeds limit of {_MAX_PATH_DEPTH} segments."
        raise ValueError(msg)
    for segment in segments:
        if len(segment) > _MAX_SEGMENT_LENGTH:
            msg = f"Path segment exceeds limit of {_MAX_SEGMENT_LENGTH} characters."
            raise ValueError(msg)
        try:
            _ = segment.encode(_ASCII)
        except UnicodeEncodeError as err:
            msg = "Path segments must be ASCII."
            raise ValueError(msg) from err


def _is_path_under(path: str, base: str) -> bool:
    """Check if path is under base directory."""
    if not base:
        return True
    return path == base or path.startswith(f"{base}/")


def _glob_match(path: str, pattern: str, base: str) -> bool:
    """Check if path matches glob pattern relative to base."""
    if base:
        if not _is_path_under(path, base):
            return False
        relative = path[len(base) + 1 :] if path != base else ""
    else:
        relative = path
    return fnmatch.fnmatch(relative, pattern)


def _empty_files_dict() -> dict[str, _InMemoryFile]:
    return {}


def _empty_directories_set() -> set[str]:
    return set()


@dataclass(slots=True)
class InMemoryFilesystem:
    """In-memory filesystem implementation.

    Provides a session-scoped in-memory storage that implements the
    Filesystem protocol. State is managed internally by the backend.
    """

    _files: dict[str, _InMemoryFile] = field(default_factory=_empty_files_dict)
    _directories: set[str] = field(default_factory=_empty_directories_set)
    _read_only: bool = False

    def __post_init__(self) -> None:
        # Ensure root directory exists
        self._directories.add("")

    @property
    def root(self) -> str:
        """Workspace root path."""
        return "/"

    @property
    def read_only(self) -> bool:
        """True if write operations are disabled."""
        return self._read_only

    def read(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int | None = None,
        encoding: str = "utf-8",
    ) -> ReadResult:
        """Read file content with optional pagination."""
        del encoding  # Only UTF-8 is supported
        normalized = _normalize_path(path)
        _validate_path(normalized)

        if normalized in self._directories:
            msg = f"Is a directory: {path}"
            raise IsADirectoryError(msg)

        if normalized not in self._files:
            raise FileNotFoundError(path)

        file = self._files[normalized]
        lines = file.content.splitlines(keepends=True)
        total_lines = len(lines)

        actual_limit = limit if limit is not None else _DEFAULT_READ_LIMIT
        start = min(offset, total_lines)
        end = min(start + actual_limit, total_lines)
        selected_lines = lines[start:end]
        content = "".join(selected_lines)
        # Remove trailing newline for consistency
        if content.endswith("\n"):
            content = content[:-1]

        return ReadResult(
            content=content,
            path=normalized or "/",
            total_lines=total_lines,
            offset=start,
            limit=end - start,
            truncated=end < total_lines,
        )

    def exists(self, path: str) -> bool:
        """Check if a path exists."""
        normalized = _normalize_path(path)
        return normalized in self._files or normalized in self._directories

    def stat(self, path: str) -> FileStat:
        """Get metadata for a path."""
        normalized = _normalize_path(path)
        _validate_path(normalized)

        if normalized in self._directories:
            return FileStat(
                path=normalized or "/",
                is_file=False,
                is_directory=True,
                size_bytes=0,
                created_at=None,
                modified_at=None,
            )

        if normalized not in self._files:
            raise FileNotFoundError(path)

        file = self._files[normalized]
        size = len(file.content.encode("utf-8"))
        return FileStat(
            path=normalized,
            is_file=True,
            is_directory=False,
            size_bytes=size,
            created_at=file.created_at,
            modified_at=file.modified_at,
        )

    def list(self, path: str = ".") -> Sequence[FileEntry]:
        """List directory contents."""
        normalized = _normalize_path(path)
        _validate_path(normalized)

        if normalized in self._files:
            msg = f"Not a directory: {path}"
            raise NotADirectoryError(msg)

        if normalized not in self._directories:
            raise FileNotFoundError(path)

        entries: list[FileEntry] = []
        seen: set[str] = set()
        prefix = f"{normalized}/" if normalized else ""

        # Find immediate children
        for file_path in self._files:
            if not _is_path_under(file_path, normalized):
                continue
            relative = file_path[len(prefix) :] if prefix else file_path
            if "/" in relative:
                # It's in a subdirectory
                child_dir = relative.split("/")[0]
                if child_dir not in seen:
                    seen.add(child_dir)
                    entries.append(
                        FileEntry(
                            name=child_dir,
                            path=f"{prefix}{child_dir}" if prefix else child_dir,
                            is_file=False,
                            is_directory=True,
                        )
                    )
            else:
                # Direct child file
                entries.append(
                    FileEntry(
                        name=relative,
                        path=file_path,
                        is_file=True,
                        is_directory=False,
                    )
                )

        # Also check explicit directories
        for dir_path in self._directories:
            if not _is_path_under(dir_path, normalized) or dir_path == normalized:
                continue
            relative = dir_path[len(prefix) :] if prefix else dir_path
            if "/" not in relative and relative not in seen:
                seen.add(relative)
                entries.append(
                    FileEntry(
                        name=relative,
                        path=dir_path,
                        is_file=False,
                        is_directory=True,
                    )
                )

        entries.sort(key=lambda e: e.name)
        return entries

    def glob(
        self,
        pattern: str,
        *,
        path: str = ".",
    ) -> Sequence[GlobMatch]:
        """Match files by glob pattern."""
        normalized_base = _normalize_path(path)
        _validate_path(normalized_base)

        matches = [
            GlobMatch(path=file_path, is_file=True)
            for file_path in self._files
            if _glob_match(file_path, pattern, normalized_base)
        ]
        matches.sort(key=lambda m: m.path)
        return matches

    def grep(
        self,
        pattern: str,
        *,
        path: str = ".",
        glob: str | None = None,
        max_matches: int | None = None,
    ) -> Sequence[GrepMatch]:
        """Search file contents by regex."""
        normalized_base = _normalize_path(path)
        _validate_path(normalized_base)

        try:
            regex = re.compile(pattern)
        except re.error as err:
            msg = f"Invalid regex pattern: {err}"
            raise ValueError(msg) from err

        actual_max = max_matches if max_matches is not None else _MAX_GREP_MATCHES
        matches: list[GrepMatch] = []

        sorted_files = sorted(self._files.items())
        for file_path, file in sorted_files:
            if not _is_path_under(file_path, normalized_base):
                continue
            if glob is not None and not _glob_match(file_path, glob, normalized_base):
                continue

            for line_num, line in enumerate(file.content.splitlines(), start=1):
                match = regex.search(line)
                if match:
                    matches.append(
                        GrepMatch(
                            path=file_path,
                            line_number=line_num,
                            line_content=line,
                            match_start=match.start(),
                            match_end=match.end(),
                        )
                    )
                    if len(matches) >= actual_max:
                        return matches

        return matches

    def write(
        self,
        path: str,
        content: str,
        *,
        mode: Literal["create", "overwrite", "append"] = "overwrite",
        create_parents: bool = True,
    ) -> WriteResult:
        """Write content to a file."""
        if self._read_only:
            msg = "Filesystem is read-only"
            raise PermissionError(msg)

        normalized = _normalize_path(path)
        if not normalized:
            msg = "Cannot write to root directory"
            raise ValueError(msg)

        _validate_path(normalized)

        if len(content) > _MAX_WRITE_LENGTH:
            msg = f"Content exceeds maximum length of {_MAX_WRITE_LENGTH} characters."
            raise ValueError(msg)

        # Check parent directory
        parent = "/".join(normalized.split("/")[:-1])
        if parent and parent not in self._directories:
            if not create_parents:
                raise FileNotFoundError(f"Parent directory does not exist: {parent}")
            # Create parent directories
            self._ensure_parents(parent)

        exists = normalized in self._files
        timestamp = _now()

        if mode == "create" and exists:
            raise FileExistsError(f"File already exists: {path}")

        if mode == "append":
            if exists:
                existing = self._files[normalized]
                final_content = existing.content + content
                created_at = existing.created_at
            else:
                final_content = content
                created_at = timestamp
        else:
            final_content = content
            created_at = self._files[normalized].created_at if exists else timestamp

        self._files[normalized] = _InMemoryFile(
            content=final_content,
            created_at=created_at,
            modified_at=timestamp,
        )

        return WriteResult(
            path=normalized,
            bytes_written=len(final_content.encode("utf-8")),
            mode=mode,
        )

    def delete(
        self,
        path: str,
        *,
        recursive: bool = False,
    ) -> None:
        """Delete a file or directory."""
        if self._read_only:
            msg = "Filesystem is read-only"
            raise PermissionError(msg)

        normalized = _normalize_path(path)
        _validate_path(normalized)

        if normalized in self._files:
            del self._files[normalized]
            return

        if normalized in self._directories:
            # Check if directory has contents
            has_files = any(
                _is_path_under(f, normalized) and f != normalized for f in self._files
            )
            has_subdirs = any(
                _is_path_under(d, normalized) and d != normalized
                for d in self._directories
            )

            if (has_files or has_subdirs) and not recursive:
                msg = f"Directory not empty: {path}"
                raise IsADirectoryError(msg)

            if recursive:
                # Delete all files under this directory
                to_delete = [f for f in self._files if _is_path_under(f, normalized)]
                for f in to_delete:
                    del self._files[f]
                # Delete all subdirectories
                to_delete_dirs = [
                    d
                    for d in self._directories
                    if _is_path_under(d, normalized) and d != normalized
                ]
                for d in to_delete_dirs:
                    self._directories.remove(d)

            if normalized:  # Don't delete root
                self._directories.discard(normalized)
            return

        raise FileNotFoundError(path)

    def mkdir(
        self,
        path: str,
        *,
        parents: bool = True,
        exist_ok: bool = True,
    ) -> None:
        """Create a directory."""
        if self._read_only:
            msg = "Filesystem is read-only"
            raise PermissionError(msg)

        normalized = _normalize_path(path)
        _validate_path(normalized)

        if not normalized:
            return  # Root always exists

        if normalized in self._directories:
            if not exist_ok:
                raise FileExistsError(f"Directory already exists: {path}")
            return

        if normalized in self._files:
            raise FileExistsError(f"A file exists at path: {path}")

        if parents:
            self._ensure_parents(normalized)
        else:
            parent = "/".join(normalized.split("/")[:-1])
            if parent and parent not in self._directories:
                raise FileNotFoundError(f"Parent directory does not exist: {parent}")

        self._directories.add(normalized)

    def _ensure_parents(self, path: str) -> None:
        """Ensure all parent directories exist."""
        segments = path.split("/")
        for i in range(len(segments)):
            parent_path = "/".join(segments[: i + 1])
            if parent_path and parent_path not in self._directories:
                self._directories.add(parent_path)


__all__ = [
    "FileEntry",
    "FileStat",
    "Filesystem",
    "GlobMatch",
    "GrepMatch",
    "InMemoryFilesystem",
    "ReadResult",
    "WriteResult",
]

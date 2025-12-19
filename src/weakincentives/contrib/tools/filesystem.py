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

The protocol uses simple `str` paths throughout - tool handlers convert these
to structured result types (with VfsPath) for serialization to the LLM.

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

import builtins
import fnmatch
import re
import shutil
import subprocess  # nosec: B404
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, Protocol, runtime_checkable
from uuid import UUID, uuid4

from weakincentives.errors import SnapshotRestoreError

_DEFAULT_READ_LIMIT: Final[int] = 2_000
_MAX_WRITE_LENGTH: Final[int] = 48_000
_MAX_PATH_DEPTH: Final[int] = 16
_MAX_SEGMENT_LENGTH: Final[int] = 80
_MAX_GREP_MATCHES: Final[int] = 1_000
_ASCII: Final[str] = "ascii"

#: Pass as `limit` to `Filesystem.read()` to read the entire file without truncation.
READ_ENTIRE_FILE: Final[int] = -1

FileEncoding = Literal["utf-8"]
WriteMode = Literal["create", "overwrite", "append"]


# ---------------------------------------------------------------------------
# Protocol Result Types
# ---------------------------------------------------------------------------


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
    mode: WriteMode


@dataclass(slots=True, frozen=True)
class FilesystemSnapshot:
    """Immutable capture of filesystem state, storable in session.

    Snapshots capture the state of a workspace at a point in time, enabling
    rollback after failed tool invocations or exploratory changes.

    The ``commit_ref`` field stores a git commit hash for disk-backed
    filesystems (HostFilesystem) or an internal version identifier for
    in-memory filesystems (InMemoryFilesystem).
    """

    snapshot_id: UUID
    created_at: datetime
    commit_ref: str
    root_path: str
    tag: str | None = None


# ---------------------------------------------------------------------------
# Filesystem Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Filesystem(Protocol):
    """Unified filesystem protocol for workspace operations.

    All paths are relative strings. Backends normalize paths internally.
    """

    @property
    def mount_point(self) -> str | None:
        """Virtual mount point prefix for path normalization.

        When set (e.g., "/workspace"), absolute paths like "/workspace/file.txt"
        are interpreted as "file.txt" relative to the workspace root. This
        allows models to use absolute paths that match container working
        directories while the underlying filesystem uses relative paths.

        Returns None if no mount point is configured (default behavior).
        """
        ...

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
            limit: Maximum lines to return. None means backend default (2000).
                Use READ_ENTIRE_FILE (-1) to read entire file without truncation.
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


@runtime_checkable
class SnapshotableFilesystem(Filesystem, Protocol):
    """Filesystem that supports snapshot and restore operations.

    This protocol extends ``Filesystem`` with methods for capturing and
    restoring filesystem state. Snapshots are immutable and can be stored
    in session state for later rollback.
    """

    def snapshot(self, *, tag: str | None = None) -> FilesystemSnapshot:
        """Capture current filesystem state.

        Creates an immutable snapshot of the filesystem that can be stored
        in session state and used for later rollback.

        Args:
            tag: Optional human-readable label for the snapshot.

        Returns:
            Immutable snapshot that can be stored in session state.
        """
        ...

    def restore(self, snapshot: FilesystemSnapshot) -> None:
        """Restore filesystem to a previous snapshot.

        Restores the filesystem state to match the given snapshot. This is
        an all-or-nothing operation - partial restores are not supported.

        Args:
            snapshot: The snapshot to restore.

        Raises:
            SnapshotRestoreError: Restore failed (e.g., incompatible snapshot,
                unknown commit reference, or git operation failure).
        """
        ...


# ---------------------------------------------------------------------------
# Shared Utilities
# ---------------------------------------------------------------------------


def normalize_path(path: str) -> str:
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


def validate_path(path: str) -> None:
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


def _now() -> datetime:
    """Return current UTC time truncated to milliseconds."""
    value = datetime.now(UTC)
    microsecond = value.microsecond - value.microsecond % 1000
    return value.replace(microsecond=microsecond, tzinfo=UTC)


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


# ---------------------------------------------------------------------------
# InMemoryFilesystem Implementation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _InMemoryFile:
    """Internal representation of a file in memory."""

    content: str
    created_at: datetime
    modified_at: datetime


@dataclass(slots=True, frozen=True)
class _InMemoryState:
    """Frozen snapshot of in-memory filesystem state."""

    files: Mapping[str, _InMemoryFile]
    directories: frozenset[str]


def _empty_snapshots_dict() -> dict[str, _InMemoryState]:
    return {}


@dataclass(slots=True)
class InMemoryFilesystem:
    """In-memory filesystem implementation.

    Provides a session-scoped in-memory storage that implements the
    Filesystem protocol. State is managed internally by the backend.
    Supports snapshot and restore operations via structural sharing.
    """

    _files: dict[str, _InMemoryFile] = field(default_factory=_empty_files_dict)
    _directories: set[str] = field(default_factory=_empty_directories_set)
    _read_only: bool = False
    _mount_point: str | None = None
    _snapshots: dict[str, _InMemoryState] = field(default_factory=_empty_snapshots_dict)
    _version: int = 0

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

    @property
    def mount_point(self) -> str | None:
        """Virtual mount point prefix for path normalization."""
        return self._mount_point

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
        normalized = normalize_path(path)
        validate_path(normalized)

        if normalized in self._directories:
            msg = f"Is a directory: {path}"
            raise IsADirectoryError(msg)

        if normalized not in self._files:
            raise FileNotFoundError(path)

        file = self._files[normalized]
        lines = file.content.splitlines(keepends=True)
        total_lines = len(lines)

        # READ_ENTIRE_FILE (-1) reads all lines; None uses default window
        if limit == READ_ENTIRE_FILE:
            actual_limit = total_lines
        else:
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
        normalized = normalize_path(path)
        return normalized in self._files or normalized in self._directories

    def stat(self, path: str) -> FileStat:
        """Get metadata for a path."""
        normalized = normalize_path(path)
        validate_path(normalized)

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

    def _collect_file_entries(
        self, normalized: str, prefix: str, seen: set[str]
    ) -> builtins.list[FileEntry]:
        """Collect file entries and implicit directories from files."""
        entries: builtins.list[FileEntry] = []
        for file_path in self._files:
            if not _is_path_under(file_path, normalized):
                continue
            relative = file_path[len(prefix) :] if prefix else file_path
            if "/" in relative:
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
                entries.append(
                    FileEntry(
                        name=relative, path=file_path, is_file=True, is_directory=False
                    )
                )
        return entries

    def _collect_explicit_dir_entries(
        self, normalized: str, prefix: str, seen: set[str]
    ) -> builtins.list[FileEntry]:
        """Collect explicit directory entries not already seen."""
        entries: builtins.list[FileEntry] = []
        for dir_path in self._directories:
            if not _is_path_under(dir_path, normalized) or dir_path == normalized:
                continue
            relative = dir_path[len(prefix) :] if prefix else dir_path
            if "/" not in relative and relative not in seen:
                seen.add(relative)
                entries.append(
                    FileEntry(
                        name=relative, path=dir_path, is_file=False, is_directory=True
                    )
                )
        return entries

    def list(self, path: str = ".") -> Sequence[FileEntry]:
        """List directory contents."""
        normalized = normalize_path(path)
        validate_path(normalized)

        if normalized in self._files:
            msg = f"Not a directory: {path}"
            raise NotADirectoryError(msg)

        if normalized not in self._directories:
            raise FileNotFoundError(path)

        seen: set[str] = set()
        prefix = f"{normalized}/" if normalized else ""
        entries = self._collect_file_entries(normalized, prefix, seen)
        entries.extend(self._collect_explicit_dir_entries(normalized, prefix, seen))
        entries.sort(key=lambda e: e.name)
        return entries

    def glob(
        self,
        pattern: str,
        *,
        path: str = ".",
    ) -> Sequence[GlobMatch]:
        """Match files by glob pattern."""
        normalized_base = normalize_path(path)
        validate_path(normalized_base)

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
        normalized_base = normalize_path(path)
        validate_path(normalized_base)

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

    def _resolve_write_content(
        self,
        normalized: str,
        content: str,
        mode: Literal["create", "overwrite", "append"],
        timestamp: datetime,
    ) -> tuple[str, datetime]:
        """Resolve final content and created_at timestamp for a write."""
        exists = normalized in self._files
        if mode == "append" and exists:
            existing = self._files[normalized]
            return existing.content + content, existing.created_at
        if mode == "append":
            return content, timestamp
        created_at = self._files[normalized].created_at if exists else timestamp
        return content, created_at

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

        normalized = normalize_path(path)
        if not normalized:
            msg = "Cannot write to root directory"
            raise ValueError(msg)

        validate_path(normalized)

        if len(content) > _MAX_WRITE_LENGTH:
            msg = f"Content exceeds maximum length of {_MAX_WRITE_LENGTH} characters."
            raise ValueError(msg)

        parent = "/".join(normalized.split("/")[:-1])
        if parent and parent not in self._directories:
            if not create_parents:
                raise FileNotFoundError(f"Parent directory does not exist: {parent}")
            self._ensure_parents(parent)

        if mode == "create" and normalized in self._files:
            raise FileExistsError(f"File already exists: {path}")

        timestamp = _now()
        final_content, created_at = self._resolve_write_content(
            normalized, content, mode, timestamp
        )

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

    def _delete_directory_contents(self, normalized: str) -> None:
        """Delete all files and subdirectories under a directory."""
        to_delete = [f for f in self._files if _is_path_under(f, normalized)]
        for f in to_delete:
            del self._files[f]
        to_delete_dirs = [
            d
            for d in self._directories
            if _is_path_under(d, normalized) and d != normalized
        ]
        for d in to_delete_dirs:
            self._directories.remove(d)

    def _directory_has_contents(self, normalized: str) -> bool:
        """Check if directory has any files or subdirectories."""
        has_files = any(
            _is_path_under(f, normalized) and f != normalized for f in self._files
        )
        has_subdirs = any(
            _is_path_under(d, normalized) and d != normalized for d in self._directories
        )
        return has_files or has_subdirs

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

        normalized = normalize_path(path)
        validate_path(normalized)

        if normalized in self._files:
            del self._files[normalized]
            return

        if normalized not in self._directories:
            raise FileNotFoundError(path)

        if self._directory_has_contents(normalized) and not recursive:
            msg = f"Directory not empty: {path}"
            raise IsADirectoryError(msg)

        if recursive:
            self._delete_directory_contents(normalized)

        if normalized:  # Don't delete root
            self._directories.discard(normalized)

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

        normalized = normalize_path(path)
        validate_path(normalized)

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

    # --- Snapshot Operations ---

    def snapshot(self, *, tag: str | None = None) -> FilesystemSnapshot:
        """Capture current filesystem state via structural sharing.

        Creates an immutable snapshot by freezing references to the current
        file and directory state. File content strings are shared between
        the active filesystem and snapshots - only modified files allocate
        new memory.

        Args:
            tag: Optional human-readable label for the snapshot.

        Returns:
            Immutable snapshot that can be stored in session state.
        """
        self._version += 1
        commit_ref = f"mem-{self._version}"

        # Freeze current state (O(n) dict copy, but values are shared refs)
        frozen_state = _InMemoryState(
            files=types.MappingProxyType(dict(self._files)),
            directories=frozenset(self._directories),
        )
        self._snapshots[commit_ref] = frozen_state

        return FilesystemSnapshot(
            snapshot_id=uuid4(),
            created_at=datetime.now(UTC),
            commit_ref=commit_ref,
            root_path="/",
            tag=tag,
        )

    def restore(self, snapshot: FilesystemSnapshot) -> None:
        """Restore filesystem state from a snapshot.

        Restores the filesystem state by copying the frozen references
        back to mutable containers. File content strings remain shared.

        Args:
            snapshot: The snapshot to restore.

        Raises:
            SnapshotRestoreError: If the snapshot's commit_ref is not found.
        """
        if snapshot.commit_ref not in self._snapshots:
            msg = f"Unknown snapshot: {snapshot.commit_ref}"
            raise SnapshotRestoreError(msg)

        frozen = self._snapshots[snapshot.commit_ref]
        self._files = dict(frozen.files)  # Mutable copy, shared values
        self._directories = set(frozen.directories)
        # Ensure root directory exists
        self._directories.add("")


# ---------------------------------------------------------------------------
# HostFilesystem Implementation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HostFilesystem:
    """Filesystem backed by a host directory with path restrictions.

    Provides sandboxed access to a root directory on the host filesystem.
    All paths are resolved relative to the root and validated to ensure
    they cannot escape the sandbox via symlinks or path traversal.
    Supports snapshot and restore operations via git commits.
    """

    _root: str
    _read_only: bool = False
    _mount_point: str | None = None
    _git_initialized: bool = False

    @property
    def root(self) -> str:
        """Workspace root path."""
        return self._root

    @property
    def read_only(self) -> bool:
        """True if write operations are disabled."""
        return self._read_only

    @property
    def mount_point(self) -> str | None:
        """Virtual mount point prefix for path normalization."""
        return self._mount_point

    def _resolve_path(self, path: str) -> Path:
        """Resolve a relative path to an absolute path within root.

        Raises:
            PermissionError: If resolved path escapes root directory.
        """
        root_path = Path(self._root).resolve()

        # Handle empty/root path
        if not path or path in {".", "/"}:
            return root_path

        # Join path with root and resolve to handle .. segments
        candidate = (root_path / path).resolve()

        # Ensure the resolved path is within root
        try:
            _ = candidate.relative_to(root_path)
        except ValueError:
            msg = f"Path escapes root directory: {path}"
            raise PermissionError(msg) from None

        return candidate

    def read(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int | None = None,
        encoding: str = "utf-8",
    ) -> ReadResult:
        """Read file content with optional pagination."""
        resolved = self._resolve_path(path)

        if not resolved.exists():
            raise FileNotFoundError(path)

        if resolved.is_dir():
            msg = f"Is a directory: {path}"
            raise IsADirectoryError(msg)

        try:
            with resolved.open(encoding=encoding) as f:
                lines = f.readlines()
        except UnicodeDecodeError as err:
            msg = f"Cannot decode file as {encoding}: {err}"
            raise ValueError(msg) from err

        total_lines = len(lines)
        # READ_ENTIRE_FILE (-1) reads all lines; None uses default window
        if limit == READ_ENTIRE_FILE:
            actual_limit = total_lines
        else:
            actual_limit = limit if limit is not None else _DEFAULT_READ_LIMIT
        start = min(offset, total_lines)
        end = min(start + actual_limit, total_lines)
        selected_lines = lines[start:end]
        content = "".join(selected_lines)

        # Remove trailing newline for consistency
        if content.endswith("\n"):
            content = content[:-1]

        # Return path relative to root
        rel_path = normalize_path(path) or "/"

        return ReadResult(
            content=content,
            path=rel_path,
            total_lines=total_lines,
            offset=start,
            limit=end - start,
            truncated=end < total_lines,
        )

    def exists(self, path: str) -> bool:
        """Check if a path exists."""
        try:
            resolved = self._resolve_path(path)
            return resolved.exists()
        except PermissionError:
            return False

    def stat(self, path: str) -> FileStat:
        """Get metadata for a path."""
        resolved = self._resolve_path(path)

        if not resolved.exists():
            raise FileNotFoundError(path)

        st = resolved.stat()
        is_dir = resolved.is_dir()
        rel_path = normalize_path(path) or "/"

        return FileStat(
            path=rel_path,
            is_file=not is_dir,
            is_directory=is_dir,
            size_bytes=st.st_size if not is_dir else 0,
            created_at=datetime.fromtimestamp(st.st_ctime, tz=UTC),
            modified_at=datetime.fromtimestamp(st.st_mtime, tz=UTC),
        )

    def list(self, path: str = ".") -> Sequence[FileEntry]:
        """List directory contents."""
        resolved = self._resolve_path(path)

        if not resolved.exists():
            raise FileNotFoundError(path)

        if not resolved.is_dir():
            msg = f"Not a directory: {path}"
            raise NotADirectoryError(msg)

        entries: list[FileEntry] = []
        base_path = normalize_path(path)

        for item in resolved.iterdir():
            is_dir = item.is_dir()
            rel_path = f"{base_path}/{item.name}" if base_path else item.name

            entries.append(
                FileEntry(
                    name=item.name,
                    path=rel_path,
                    is_file=not is_dir,
                    is_directory=is_dir,
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
        resolved_base = self._resolve_path(path)
        root_path = Path(self._root)

        if not resolved_base.exists():
            return []

        matches: list[GlobMatch] = []

        for file_path in resolved_base.rglob("*"):
            if not file_path.is_file():
                continue
            # Get path relative to root filesystem
            rel_to_fs_root = str(file_path.relative_to(root_path))
            # Get path relative to search base
            rel_to_base = str(file_path.relative_to(resolved_base))

            if _glob_match(rel_to_base, pattern, ""):
                matches.append(GlobMatch(path=rel_to_fs_root, is_file=True))

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
        resolved_base = self._resolve_path(path)
        root_path = Path(self._root)

        try:
            regex = re.compile(pattern)
        except re.error as err:
            msg = f"Invalid regex pattern: {err}"
            raise ValueError(msg) from err

        actual_max = max_matches if max_matches is not None else _MAX_GREP_MATCHES
        matches: list[GrepMatch] = []

        if not resolved_base.exists():
            return []

        for file_path in sorted(resolved_base.rglob("*")):
            if not file_path.is_file():
                continue

            rel_to_fs_root = str(file_path.relative_to(root_path))
            rel_to_base = str(file_path.relative_to(resolved_base))

            # Apply glob filter if provided
            if glob is not None and not _glob_match(rel_to_base, glob, ""):
                continue

            grep_matches = self._grep_file(
                file_path, regex, rel_to_fs_root, actual_max - len(matches)
            )
            matches.extend(grep_matches)
            if len(matches) >= actual_max:
                break

        return matches

    @staticmethod
    def _grep_file(
        file_path: Path, regex: re.Pattern[str], rel_path: str, max_remaining: int
    ) -> list[GrepMatch]:  # ty: ignore[invalid-type-form]  # ty bug: resolves list to class method
        """Search a single file for regex matches."""
        matches: list[GrepMatch] = []
        try:
            with file_path.open(encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.rstrip("\n")
                    match = regex.search(line)
                    if match:
                        matches.append(
                            GrepMatch(
                                path=rel_path,
                                line_number=line_num,
                                line_content=line,
                                match_start=match.start(),
                                match_end=match.end(),
                            )
                        )
                        if len(matches) >= max_remaining:
                            break
        except (OSError, UnicodeDecodeError):
            # Skip binary or inaccessible files
            pass
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

        normalized = normalize_path(path)
        if not normalized:
            msg = "Cannot write to root directory"
            raise ValueError(msg)

        validate_path(normalized)

        if len(content) > _MAX_WRITE_LENGTH:
            msg = f"Content exceeds maximum length of {_MAX_WRITE_LENGTH} characters."
            raise ValueError(msg)

        resolved = self._resolve_path(path)
        parent_dir = resolved.parent

        # Check parent directory
        if not parent_dir.exists():
            if not create_parents:
                raise FileNotFoundError(
                    f"Parent directory does not exist: {parent_dir}"
                )
            parent_dir.mkdir(parents=True, exist_ok=True)

        exists = resolved.exists()

        if mode == "create" and exists:
            raise FileExistsError(f"File already exists: {path}")

        file_mode = "a" if mode == "append" else "w"

        with resolved.open(file_mode, encoding="utf-8") as f:
            _ = f.write(content)

        # Calculate bytes written
        bytes_written = resolved.stat().st_size

        return WriteResult(
            path=normalized,
            bytes_written=bytes_written,
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

        normalized = normalize_path(path)
        if not normalized:
            msg = "Cannot delete root directory"
            raise PermissionError(msg)

        resolved = self._resolve_path(path)

        if not resolved.exists():
            raise FileNotFoundError(path)

        if resolved.is_file():
            resolved.unlink()
            return

        # It's a directory
        if any(resolved.iterdir()) and not recursive:
            msg = f"Directory not empty: {path}"
            raise IsADirectoryError(msg)

        if recursive:
            shutil.rmtree(resolved)
        else:
            resolved.rmdir()

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

        normalized = normalize_path(path)
        if not normalized:
            return  # Root always exists

        validate_path(normalized)
        resolved = self._resolve_path(path)

        if resolved.exists():
            if resolved.is_file():
                raise FileExistsError(f"A file exists at path: {path}")
            if not exist_ok:
                raise FileExistsError(f"Directory already exists: {path}")
            return

        if parents:
            resolved.mkdir(parents=True, exist_ok=True)
        else:
            if not resolved.parent.exists():
                raise FileNotFoundError(
                    f"Parent directory does not exist: {resolved.parent}"
                )
            resolved.mkdir()

    # --- Snapshot Operations ---

    def _ensure_git(self) -> None:
        """Initialize git repository if needed for snapshot support."""
        if self._git_initialized:
            return

        git_dir = Path(self._root) / ".git"
        if not git_dir.exists():
            _ = subprocess.run(  # nosec B603 B607
                ["git", "init"],
                cwd=self._root,
                check=True,
                capture_output=True,
            )
            # Configure for snapshot use (local config only)
            _ = subprocess.run(  # nosec B603 B607
                ["git", "config", "user.email", "wink@localhost"],
                cwd=self._root,
                check=True,
                capture_output=True,
            )
            _ = subprocess.run(  # nosec B603 B607
                ["git", "config", "user.name", "WINK Snapshots"],
                cwd=self._root,
                check=True,
                capture_output=True,
            )
        self._git_initialized = True

    def snapshot(self, *, tag: str | None = None) -> FilesystemSnapshot:
        """Capture current filesystem state as a git commit.

        Uses git's content-addressed storage for copy-on-write semantics.
        Identical files share storage automatically between snapshots.

        Args:
            tag: Optional human-readable label for the snapshot.

        Returns:
            Immutable snapshot that can be stored in session state.
        """
        self._ensure_git()

        # Stage all changes (including new and deleted files)
        _ = subprocess.run(  # nosec B603 B607
            ["git", "add", "-A"],
            cwd=self._root,
            check=True,
            capture_output=True,
        )

        # Commit (allow empty for idempotent snapshots)
        # Use --no-gpg-sign to avoid issues in environments with signing hooks
        message = tag or f"snapshot-{datetime.now(UTC).isoformat()}"
        commit_result = subprocess.run(  # nosec B603 B607
            ["git", "commit", "-m", message, "--allow-empty", "--no-gpg-sign"],
            cwd=self._root,
            capture_output=True,
            text=True,
        )

        # If commit failed and we don't have any commits yet, we may need
        # to check for empty repo scenario
        if commit_result.returncode != 0:
            # Check if this is because there's nothing to commit
            # and no prior commits exist (empty repo)
            head_check = subprocess.run(  # nosec B603 B607
                ["git", "rev-parse", "--verify", "HEAD"],
                cwd=self._root,
                capture_output=True,
            )
            if head_check.returncode != 0:
                # No HEAD commit exists - create an initial empty commit
                _ = subprocess.run(  # nosec B603 B607
                    ["git", "commit", "--allow-empty", "--no-gpg-sign", "-m", message],
                    cwd=self._root,
                    check=True,
                    capture_output=True,
                )

        # Get commit hash
        result = subprocess.run(  # nosec B603 B607
            ["git", "rev-parse", "HEAD"],
            cwd=self._root,
            check=True,
            capture_output=True,
            text=True,
        )
        commit_ref = result.stdout.strip()

        return FilesystemSnapshot(
            snapshot_id=uuid4(),
            created_at=datetime.now(UTC),
            commit_ref=commit_ref,
            root_path=self._root,
            tag=tag,
        )

    def restore(self, snapshot: FilesystemSnapshot) -> None:
        """Restore filesystem to a previous git commit.

        Performs a hard reset to the snapshot's commit and removes any
        untracked files (including ignored files) for a strict rollback.

        Args:
            snapshot: The snapshot to restore.

        Raises:
            SnapshotRestoreError: If git reset fails (e.g., invalid commit).
        """
        self._ensure_git()

        # Hard reset to the commit (restores tracked files)
        result = subprocess.run(  # nosec B603 B607
            ["git", "reset", "--hard", snapshot.commit_ref],
            cwd=self._root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            msg = f"Failed to restore snapshot: {result.stderr}"
            raise SnapshotRestoreError(msg)

        # Remove untracked files and directories for strict rollback
        # -x removes ignored files too (e.g., cache, logs) for full restore
        _ = subprocess.run(  # nosec B603 B607
            ["git", "clean", "-xfd"],
            cwd=self._root,
            capture_output=True,
        )


__all__ = [
    "READ_ENTIRE_FILE",
    "FileEncoding",
    "FileEntry",
    "FileStat",
    "Filesystem",
    "FilesystemSnapshot",
    "GlobMatch",
    "GrepMatch",
    "HostFilesystem",
    "InMemoryFilesystem",
    "ReadResult",
    "SnapshotableFilesystem",
    "WriteMode",
    "WriteResult",
    "normalize_path",
    "validate_path",
]

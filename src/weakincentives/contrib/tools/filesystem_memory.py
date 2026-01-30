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

"""In-memory filesystem backend for workspace operations.

This module provides an in-memory implementation of the Filesystem protocol,
suitable for testing and session-scoped virtual file storage.

Example usage::

    from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem

    # Create an in-memory filesystem
    fs = InMemoryFilesystem()

    # Write and read files
    fs.write("src/main.py", "print('hello')")
    result = fs.read("src/main.py")
    assert result.content == "print('hello')"
"""

from __future__ import annotations

import re
import types
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from uuid import uuid4

from weakincentives.clock import SYSTEM_CLOCK
from weakincentives.errors import SnapshotRestoreError
from weakincentives.filesystem import (
    DEFAULT_READ_LIMIT,
    MAX_GREP_MATCHES,
    MAX_WRITE_BYTES,
    MAX_WRITE_LENGTH,
    READ_ENTIRE_FILE,
    FileEntry,
    FileStat,
    FilesystemSnapshot,
    GlobMatch,
    GrepMatch,
    ReadBytesResult,
    ReadResult,
    WriteResult,
    glob_match,
    is_path_under,
    normalize_path,
    now,
    validate_path,
)
from weakincentives.filesystem._streams import (
    DefaultTextReader,
    MemoryByteReader,
    MemoryByteWriter,
    TextReader,
)

# Re-export READ_ENTIRE_FILE for direct imports from this module
__all__ = ["InMemoryFilesystem"]


# ---------------------------------------------------------------------------
# Internal Types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _InMemoryFile:
    """Internal representation of a file in memory.

    Files are stored as raw bytes internally. Text operations encode/decode
    using UTF-8, while byte operations work directly with the stored content.
    """

    content: bytes
    created_at: datetime
    modified_at: datetime


@dataclass(slots=True, frozen=True)
class _InMemoryState:
    """Frozen snapshot of in-memory filesystem state."""

    files: Mapping[str, _InMemoryFile]
    directories: frozenset[str]


def _empty_files_dict() -> dict[str, _InMemoryFile]:
    return {}


def _empty_directories_set() -> set[str]:
    return set()


def _empty_snapshots_dict() -> dict[str, _InMemoryState]:
    return {}


# ---------------------------------------------------------------------------
# InMemoryFilesystem Implementation
# ---------------------------------------------------------------------------


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

    def read(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int | None = None,
        encoding: str = "utf-8",
    ) -> ReadResult:
        """Read file content as text with optional pagination."""
        del encoding  # Only UTF-8 is supported
        normalized = normalize_path(path)
        validate_path(normalized)

        if normalized in self._directories:
            msg = f"Is a directory: {path}"
            raise IsADirectoryError(msg)

        if normalized not in self._files:
            raise FileNotFoundError(path)

        file = self._files[normalized]
        try:
            text_content = file.content.decode("utf-8")
        except UnicodeDecodeError as err:
            msg = (
                f"Cannot read '{path}' as text: file contains binary content that "
                "cannot be decoded as utf-8. Use read_bytes() for binary files."
            )
            raise ValueError(msg) from err

        lines = text_content.splitlines(keepends=True)
        total_lines = len(lines)

        # READ_ENTIRE_FILE (-1) reads all lines; None uses default window
        if limit == READ_ENTIRE_FILE:
            actual_limit = total_lines
        else:
            actual_limit = limit if limit is not None else DEFAULT_READ_LIMIT
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

    def read_bytes(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> ReadBytesResult:
        """Read file content as raw bytes with optional pagination."""
        if offset < 0:
            msg = f"offset must be non-negative, got {offset}"
            raise ValueError(msg)
        if limit is not None and limit < 0:
            msg = f"limit must be non-negative, got {limit}"
            raise ValueError(msg)

        normalized = normalize_path(path)
        validate_path(normalized)

        if normalized in self._directories:
            msg = f"Is a directory: {path}"
            raise IsADirectoryError(msg)

        if normalized not in self._files:
            raise FileNotFoundError(path)

        file = self._files[normalized]
        file_size = len(file.content)

        # Calculate actual positions
        actual_offset = min(offset, file_size)
        if limit is not None:
            end = min(actual_offset + limit, file_size)
        else:
            end = file_size

        data = file.content[actual_offset:end]
        bytes_read = len(data)
        truncated = actual_offset + bytes_read < file_size

        return ReadBytesResult(
            content=data,
            path=normalized or "/",
            size_bytes=file_size,
            offset=actual_offset,
            limit=bytes_read,
            truncated=truncated,
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
        return FileStat(
            path=normalized,
            is_file=True,
            is_directory=False,
            size_bytes=len(file.content),
            created_at=file.created_at,
            modified_at=file.modified_at,
        )

    def _collect_file_entries(
        self, normalized: str, prefix: str, seen: set[str]
    ) -> list[FileEntry]:
        """Collect file entries and implicit directories from files."""
        entries: list[FileEntry] = []
        for file_path in self._files:
            if not is_path_under(file_path, normalized):
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
    ) -> list[FileEntry]:
        """Collect explicit directory entries not already seen."""
        entries: list[FileEntry] = []
        for dir_path in self._directories:
            if not is_path_under(dir_path, normalized) or dir_path == normalized:
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
            if glob_match(file_path, pattern, normalized_base)
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

        actual_max = max_matches if max_matches is not None else MAX_GREP_MATCHES
        matches: list[GrepMatch] = []

        sorted_files = sorted(self._files.items())
        for file_path, file in sorted_files:
            if not is_path_under(file_path, normalized_base):
                continue
            if glob is not None and not glob_match(file_path, glob, normalized_base):
                continue

            # Skip binary files that can't be decoded
            try:
                text_content = file.content.decode("utf-8")
            except UnicodeDecodeError:
                continue

            for line_num, line in enumerate(text_content.splitlines(), start=1):
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

    def _resolve_write_bytes_content(
        self,
        normalized: str,
        content: bytes,
        mode: Literal["create", "overwrite", "append"],
        timestamp: datetime,
    ) -> tuple[bytes, datetime]:
        """Resolve final content and created_at timestamp for a byte write."""
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
        """Write text content to a file."""
        if self._read_only:
            msg = "Filesystem is read-only"
            raise PermissionError(msg)

        normalized = normalize_path(path)
        if not normalized:
            msg = "Cannot write to root directory"
            raise ValueError(msg)

        validate_path(normalized)

        if len(content) > MAX_WRITE_LENGTH:
            msg = f"Content exceeds maximum length of {MAX_WRITE_LENGTH} characters."
            raise ValueError(msg)

        parent = "/".join(normalized.split("/")[:-1])
        if parent and parent not in self._directories:
            if not create_parents:
                raise FileNotFoundError(f"Parent directory does not exist: {parent}")
            self._ensure_parents(parent)

        if mode == "create" and normalized in self._files:
            raise FileExistsError(f"File already exists: {path}")

        # Encode text content to bytes for storage
        content_bytes = content.encode("utf-8")

        timestamp = now()
        final_content, created_at = self._resolve_write_bytes_content(
            normalized, content_bytes, mode, timestamp
        )

        self._files[normalized] = _InMemoryFile(
            content=final_content,
            created_at=created_at,
            modified_at=timestamp,
        )

        return WriteResult(
            path=normalized,
            bytes_written=len(content_bytes),
            mode=mode,
        )

    def write_bytes(
        self,
        path: str,
        content: bytes,
        *,
        mode: Literal["create", "overwrite", "append"] = "overwrite",
        create_parents: bool = True,
    ) -> WriteResult:
        """Write raw bytes to a file."""
        if self._read_only:
            msg = "Filesystem is read-only"
            raise PermissionError(msg)

        normalized = normalize_path(path)
        if not normalized:
            msg = "Cannot write to root directory"
            raise ValueError(msg)

        validate_path(normalized)

        if len(content) > MAX_WRITE_BYTES:
            msg = f"Content exceeds maximum size of {MAX_WRITE_BYTES} bytes."
            raise ValueError(msg)

        parent = "/".join(normalized.split("/")[:-1])
        if parent and parent not in self._directories:
            if not create_parents:
                raise FileNotFoundError(f"Parent directory does not exist: {parent}")
            self._ensure_parents(parent)

        if mode == "create" and normalized in self._files:
            raise FileExistsError(f"File already exists: {path}")

        timestamp = now()
        final_content, created_at = self._resolve_write_bytes_content(
            normalized, content, mode, timestamp
        )

        self._files[normalized] = _InMemoryFile(
            content=final_content,
            created_at=created_at,
            modified_at=timestamp,
        )

        return WriteResult(
            path=normalized,
            bytes_written=len(content),
            mode=mode,
        )

    def _delete_directory_contents(self, normalized: str) -> None:
        """Delete all files and subdirectories under a directory."""
        to_delete = [f for f in self._files if is_path_under(f, normalized)]
        for f in to_delete:
            del self._files[f]
        to_delete_dirs = [
            d
            for d in self._directories
            if is_path_under(d, normalized) and d != normalized
        ]
        for d in to_delete_dirs:
            self._directories.remove(d)

    def _directory_has_contents(self, normalized: str) -> bool:
        """Check if directory has any files or subdirectories."""
        has_files = any(
            is_path_under(f, normalized) and f != normalized for f in self._files
        )
        has_subdirs = any(
            is_path_under(d, normalized) and d != normalized for d in self._directories
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
            created_at=SYSTEM_CLOCK.utcnow(),
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

    # --- Streaming Operations ---

    def open_read(self, path: str) -> MemoryByteReader:
        """Open a file for streaming byte reads.

        Returns a ByteReader context manager for chunked reading.
        """
        normalized = normalize_path(path)
        validate_path(normalized)

        if normalized in self._directories:
            msg = f"Is a directory: {path}"
            raise IsADirectoryError(msg)

        if normalized not in self._files:
            raise FileNotFoundError(path)

        file = self._files[normalized]
        return MemoryByteReader.from_bytes(normalized or "/", file.content)

    def open_write(
        self,
        path: str,
        *,
        mode: Literal["create", "overwrite", "append"] = "overwrite",
        create_parents: bool = True,
    ) -> _InMemoryByteWriter:
        """Open a file for streaming byte writes.

        Returns a ByteWriter context manager for chunked writing.
        """
        if self._read_only:
            msg = "Filesystem is read-only"
            raise PermissionError(msg)

        normalized = normalize_path(path)
        if not normalized:
            msg = "Cannot write to root directory"
            raise ValueError(msg)

        validate_path(normalized)

        parent = "/".join(normalized.split("/")[:-1])
        if parent and parent not in self._directories:
            if not create_parents:
                raise FileNotFoundError(f"Parent directory does not exist: {parent}")
            self._ensure_parents(parent)

        if mode == "create" and normalized in self._files:
            raise FileExistsError(f"File already exists: {path}")

        existing_content = None
        if mode == "append" and normalized in self._files:
            existing_content = self._files[normalized].content

        return _InMemoryByteWriter(
            filesystem=self,
            path=normalized,
            mode=mode,
            existing_content=existing_content,
        )

    def open_text(
        self,
        path: str,
        *,
        encoding: str = "utf-8",
    ) -> TextReader:
        """Open a file for streaming text reads with lazy decoding.

        Returns a TextReader context manager that decodes bytes lazily.
        """
        if encoding != "utf-8":
            msg = f"Only 'utf-8' encoding is supported, got: {encoding}"
            raise ValueError(msg)

        byte_reader = self.open_read(path)
        return DefaultTextReader.wrap(byte_reader, encoding=encoding)

    def commit_streaming_write(
        self,
        path: str,
        content: bytes,
        mode: Literal["create", "overwrite", "append"],
    ) -> None:
        """Internal method to commit written bytes to storage.

        Called by _InMemoryByteWriter when the writer is closed.
        """
        timestamp = now()
        if path in self._files:
            created_at = self._files[path].created_at
        else:
            created_at = timestamp

        self._files[path] = _InMemoryFile(
            content=content,
            created_at=created_at,
            modified_at=timestamp,
        )


@dataclass(slots=True)
class _InMemoryByteWriter:
    """ByteWriter that commits to InMemoryFilesystem on close.

    This writer collects bytes in a MemoryByteWriter and commits
    the content to the filesystem when closed.
    """

    filesystem: InMemoryFilesystem
    path: str
    mode: Literal["create", "overwrite", "append"]
    existing_content: bytes | None = None
    _writer: MemoryByteWriter = field(init=False)
    _closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._writer = MemoryByteWriter.create(
            self.path,
            mode=self.mode,
            existing_content=self.existing_content,
        )

    @property
    def bytes_written(self) -> int:
        """Total bytes written (excluding initial append content)."""
        return self._writer.bytes_written

    @property
    def closed(self) -> bool:
        """True if the writer has been closed."""
        return self._closed

    def _check_closed(self) -> None:
        """Raise ValueError if closed."""
        if self._closed:
            msg = "I/O operation on closed file"
            raise ValueError(msg)

    def write(self, data: bytes) -> int:
        """Write bytes to the buffer."""
        self._check_closed()
        return self._writer.write(data)

    def write_all(self, chunks: Iterable[bytes]) -> int:
        """Write all chunks from an iterable."""
        self._check_closed()
        return self._writer.write_all(chunks)

    def __enter__(self) -> _InMemoryByteWriter:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager, committing and closing."""
        self.close()

    def close(self) -> None:
        """Close the writer and commit content to filesystem."""
        if self._closed:
            return
        self._closed = True

        # Commit the written content to the filesystem
        content = self._writer.get_content()
        self.filesystem.commit_streaming_write(self.path, content, self.mode)
        self._writer.close()

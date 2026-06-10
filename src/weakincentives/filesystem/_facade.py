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

"""Concrete ``Filesystem`` facade over a narrow backend.

This module composes all filesystem ergonomics exactly once, over any
:class:`~weakincentives.filesystem.FilesystemBackend`:

- text reads with line pagination (``read``) and byte reads (``read_bytes``)
- one-shot writes with 32 MB convenience caps (``write``/``write_bytes``)
- streaming readers/writers with 64 KB chunks (``open_read``/``open_write``/
  ``open_text``)
- uniform path normalization and validation on every operation
- read-only and root-path guards for all mutations

Backends implement storage primitives only; see ``_backend.py`` for the
contract.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from typing import final

from ._backend import FilesystemBackend, SnapshotRef
from ._path import validate_path
from ._streams import ByteReader, ByteWriter, TextReader
from ._types import (
    DEFAULT_READ_LIMIT,
    MAX_GREP_MATCHES,
    MAX_WRITE_BYTES,
    MAX_WRITE_LENGTH,
    READ_ENTIRE_FILE,
    FileEntry,
    FileStat,
    GlobMatch,
    GrepMatch,
    ReadBytesResult,
    ReadResult,
    WriteMode,
    WriteResult,
    normalize_path,
)


@final
class Filesystem:
    """Ergonomic filesystem API over a narrow storage backend.

    All paths are workspace-relative strings, normalized and validated on
    every operation. Write operations honor the backend's ``read_only``
    flag and the 32 MB convenience caps; streaming operations move data in
    64 KB chunks.

    Construct over a specific backend directly, or use the convenience
    constructors::

        fs = Filesystem.host("/path/to/workspace")
        fs = Filesystem.in_memory()
    """

    def __init__(self, backend: FilesystemBackend) -> None:
        self._backend = backend

    @classmethod
    def host(
        cls,
        root: str | os.PathLike[str],
        *,
        read_only: bool = False,
        git_dir: str | None = None,
    ) -> Filesystem:
        """Create a filesystem over a sandboxed host directory."""
        from ._host import HostBackend

        return cls(HostBackend(root, read_only=read_only, git_dir=git_dir))

    @classmethod
    def in_memory(cls, *, read_only: bool = False) -> Filesystem:
        """Create a filesystem over a session-scoped in-memory backend."""
        from ._memory import MemoryBackend

        return cls(MemoryBackend(read_only=read_only))

    @property
    def backend(self) -> FilesystemBackend:
        """The storage backend serving this filesystem."""
        return self._backend

    @property
    def root(self) -> str:
        """Workspace root path (``"/"`` for virtual backends)."""
        return self._backend.root

    @property
    def read_only(self) -> bool:
        """True if write operations are disabled."""
        return self._backend.read_only

    # --- Path helpers -------------------------------------------------------

    @staticmethod
    def _normalize(path: str) -> str:
        norm = normalize_path(path)
        validate_path(norm)
        return norm

    def _check_writable(self) -> None:
        if self._backend.read_only:
            msg = "Filesystem is read-only"
            raise PermissionError(msg)

    def _writable_file_path(self, path: str) -> str:
        """Validate a mutating file operation; returns the normalized path."""
        self._check_writable()
        norm = self._normalize(path)
        if not norm:
            msg = "Cannot write to root directory"
            raise ValueError(msg)
        return norm

    def _check_parent(self, norm: str, create_parents: bool) -> None:
        """Raise FileNotFoundError when the parent must exist but does not."""
        if create_parents:
            return
        parent = norm.rsplit("/", 1)[0] if "/" in norm else ""
        if parent and not self._stat_or_none(parent):
            raise FileNotFoundError(f"Parent directory does not exist: {parent}")

    def _stat_or_none(self, norm: str) -> FileStat | None:
        try:
            return self._backend.stat(norm)
        except (FileNotFoundError, PermissionError):
            return None

    # --- Read operations ----------------------------------------------------

    def read(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int | None = None,
        encoding: str = "utf-8",
    ) -> ReadResult:
        """Read text content with line-based pagination.

        Args:
            path: Relative path from workspace root.
            offset: Line number to start reading (0-indexed).
            limit: Maximum lines to return; ``None`` uses the default window
                (2000), ``READ_ENTIRE_FILE`` (-1) reads everything.
            encoding: Text encoding. Only ``"utf-8"`` is guaranteed.

        Raises:
            FileNotFoundError: Path does not exist.
            IsADirectoryError: Path is a directory.
            ValueError: Binary content, or negative ``offset``/``limit``.
        """
        if offset < 0:
            msg = f"offset must be non-negative, got {offset}"
            raise ValueError(msg)
        if limit is not None and limit < 0 and limit != READ_ENTIRE_FILE:
            msg = f"limit must be non-negative or READ_ENTIRE_FILE, got {limit}"
            raise ValueError(msg)

        norm = self._normalize(path)
        data = self._backend.read_range(norm, offset=0, length=None)
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError as err:
            msg = (
                f"Cannot read '{path}' as text: file contains binary content that "
                f"cannot be decoded as {encoding}. Use read_bytes() for binary files."
            )
            raise ValueError(msg) from err

        lines = text.splitlines(keepends=True)
        total_lines = len(lines)
        if limit == READ_ENTIRE_FILE:
            actual_limit = total_lines
        else:
            actual_limit = limit if limit is not None else DEFAULT_READ_LIMIT
        start = min(offset, total_lines)
        end = min(start + actual_limit, total_lines)
        content = "".join(lines[start:end]).removesuffix("\n")

        return ReadResult(
            content=content,
            path=norm or "/",
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
        """Read raw bytes with byte-based pagination.

        Raises:
            FileNotFoundError: Path does not exist.
            IsADirectoryError: Path is a directory.
            ValueError: Negative ``offset`` or ``limit``.
        """
        if offset < 0:
            msg = f"offset must be non-negative, got {offset}"
            raise ValueError(msg)
        if limit is not None and limit < 0:
            msg = f"limit must be non-negative, got {limit}"
            raise ValueError(msg)

        norm = self._normalize(path)
        stat = self._backend.stat(norm)
        if stat.is_directory:
            msg = f"Is a directory: {path}"
            raise IsADirectoryError(msg)

        data = self._backend.read_range(norm, offset=offset, length=limit)
        actual_offset = min(offset, stat.size_bytes)

        return ReadBytesResult(
            content=data,
            path=norm or "/",
            size_bytes=stat.size_bytes,
            offset=actual_offset,
            limit=len(data),
            truncated=actual_offset + len(data) < stat.size_bytes,
        )

    def exists(self, path: str) -> bool:
        """Check if a path exists."""
        return self._stat_or_none(self._normalize(path)) is not None

    def stat(self, path: str) -> FileStat:
        """Get metadata for a path.

        Raises:
            FileNotFoundError: Path does not exist.
        """
        return self._backend.stat(self._normalize(path))

    def list(self, path: str = ".") -> Sequence[FileEntry]:
        """List directory contents sorted by name.

        Raises:
            FileNotFoundError: Path does not exist.
            NotADirectoryError: Path is a file.
        """
        norm = self._normalize(path)
        if self._backend.stat(norm).is_file:
            msg = f"Not a directory: {path}"
            raise NotADirectoryError(msg)
        return self._backend.list(norm)

    def glob(self, pattern: str, *, path: str = ".") -> Sequence[GlobMatch]:
        """Match files by glob pattern, sorted by path.

        Returns an empty sequence when the base directory does not exist.
        """
        norm = self._normalize(path)
        base = self._stat_or_none(norm)
        if base is None or base.is_file:
            return ()
        return self._backend.glob(pattern, path=norm)

    def grep(
        self,
        pattern: str,
        *,
        path: str = ".",
        glob: str | None = None,
        max_matches: int | None = None,
    ) -> Sequence[GrepMatch]:
        """Search file contents by regex, sorted by ``(path, line_number)``.

        Raises:
            ValueError: Invalid regex pattern.
        """
        try:
            _ = re.compile(pattern)
        except re.error as err:
            msg = f"Invalid regex pattern: {err}"
            raise ValueError(msg) from err

        norm = self._normalize(path)
        base = self._stat_or_none(norm)
        if base is None or base.is_file:
            return ()
        actual_max = max_matches if max_matches is not None else MAX_GREP_MATCHES
        return self._backend.grep(pattern, path=norm, glob=glob, max_matches=actual_max)

    # --- Write operations ----------------------------------------------------

    def write(
        self,
        path: str,
        content: str,
        *,
        mode: WriteMode = "overwrite",
        create_parents: bool = True,
    ) -> WriteResult:
        """Write text content to a file (capped at 32 MB).

        Raises:
            FileExistsError: ``mode="create"`` and the file exists.
            FileNotFoundError: Parent missing and ``create_parents=False``.
            PermissionError: Filesystem is read-only.
            ValueError: Root path or content over the cap.
        """
        norm = self._writable_file_path(path)
        if len(content) > MAX_WRITE_LENGTH:
            msg = f"Content exceeds maximum length of {MAX_WRITE_LENGTH} characters."
            raise ValueError(msg)
        data = content.encode("utf-8")
        self._check_parent(norm, create_parents)
        self._backend.write(norm, data, mode=mode)
        return WriteResult(path=norm, bytes_written=len(data), mode=mode)

    def write_bytes(
        self,
        path: str,
        content: bytes,
        *,
        mode: WriteMode = "overwrite",
        create_parents: bool = True,
    ) -> WriteResult:
        """Write raw bytes to a file (capped at 32 MB).

        Raises:
            FileExistsError: ``mode="create"`` and the file exists.
            FileNotFoundError: Parent missing and ``create_parents=False``.
            PermissionError: Filesystem is read-only.
            ValueError: Root path or content over the cap.
        """
        norm = self._writable_file_path(path)
        if len(content) > MAX_WRITE_BYTES:
            msg = f"Content exceeds maximum size of {MAX_WRITE_BYTES} bytes."
            raise ValueError(msg)
        self._check_parent(norm, create_parents)
        self._backend.write(norm, content, mode=mode)
        return WriteResult(path=norm, bytes_written=len(content), mode=mode)

    def delete(self, path: str, *, recursive: bool = False) -> None:
        """Delete a file or directory.

        Raises:
            FileNotFoundError: Path does not exist.
            IsADirectoryError: Non-empty directory and ``recursive=False``.
            PermissionError: Read-only filesystem or root path.
        """
        self._check_writable()
        norm = self._normalize(path)
        if not norm:
            msg = "Cannot delete root directory"
            raise PermissionError(msg)
        self._backend.delete(norm, recursive=recursive)

    def mkdir(
        self,
        path: str,
        *,
        parents: bool = True,
        exist_ok: bool = True,
    ) -> None:
        """Create a directory. Root is a no-op.

        Raises:
            FileExistsError: A file occupies the path, or the directory
                exists and ``exist_ok=False``.
            FileNotFoundError: Parent missing and ``parents=False``.
            PermissionError: Filesystem is read-only.
        """
        self._check_writable()
        norm = self._normalize(path)
        if not norm:
            return
        existing = self._stat_or_none(norm)
        if existing is not None:
            if existing.is_file:
                raise FileExistsError(f"A file exists at path: {path}")
            if not exist_ok:
                raise FileExistsError(f"Directory already exists: {path}")
            return
        self._check_parent(norm, parents)
        self._backend.mkdir(norm)

    # --- Streaming operations -------------------------------------------------

    def open_read(self, path: str) -> ByteReader:
        """Open a file for streaming byte reads.

        Raises:
            FileNotFoundError: Path does not exist.
            IsADirectoryError: Path is a directory.
        """
        norm = self._normalize(path)
        stat = self._backend.stat(norm)
        if stat.is_directory:
            msg = f"Is a directory: {path}"
            raise IsADirectoryError(msg)
        return ByteReader(self._backend, norm, stat.size_bytes)

    def open_write(
        self,
        path: str,
        *,
        mode: WriteMode = "overwrite",
        create_parents: bool = True,
    ) -> ByteWriter:
        """Open a file for streaming byte writes.

        The writer buffers bytes and commits them in one backend write on
        ``close()``; aborting (exception inside the ``with`` block) leaves
        the backend untouched.

        Raises:
            FileExistsError: ``mode="create"`` and the file exists.
            FileNotFoundError: Parent missing and ``create_parents=False``.
            PermissionError: Filesystem is read-only.
            ValueError: Root path.
        """
        norm = self._writable_file_path(path)
        if mode == "create" and self._stat_or_none(norm) is not None:
            raise FileExistsError(f"File already exists: {path}")
        self._check_parent(norm, create_parents)
        return ByteWriter(self._backend, norm, mode)

    def open_text(self, path: str, *, encoding: str = "utf-8") -> TextReader:
        """Open a file for streaming text reads with lazy decoding.

        Raises:
            FileNotFoundError: Path does not exist.
            IsADirectoryError: Path is a directory.
            ValueError: Unsupported encoding.
        """
        if encoding != "utf-8":
            msg = f"Only 'utf-8' encoding is supported, got: {encoding}"
            raise ValueError(msg)
        return TextReader(self.open_read(path), encoding)

    # --- Snapshot operations ----------------------------------------------------

    def snapshot(self, *, tag: str | None = None) -> SnapshotRef:
        """Capture current filesystem state as an opaque snapshot ref."""
        return self._backend.snapshot(tag=tag)

    def restore(self, ref: SnapshotRef) -> None:
        """Restore filesystem state from a snapshot ref.

        Raises:
            SnapshotRestoreError: Ref is unknown to the backend or the
                restore failed.
        """
        self._backend.restore(ref)


__all__ = ["Filesystem"]

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

"""Filesystem protocol for workspace operations.

This module provides a unified `Filesystem` protocol that abstracts over
workspace backends (in-memory VFS, Podman containers, host filesystem) so
tool handlers can perform file operations without coupling to a specific
storage implementation.

The protocol uses simple `str` paths throughout - tool handlers convert these
to structured result types for serialization to the LLM.

Core implementation:

- `weakincentives.filesystem.HostFilesystem`: Sandboxed host directory access

Additional implementations in `weakincentives.contrib.tools`:

- `filesystem_memory.InMemoryFilesystem`: Session-scoped in-memory storage
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from ._types import (
    FileEntry,
    FileStat,
    FilesystemSnapshot,
    GlobMatch,
    GrepMatch,
    ReadBytesResult,
    ReadResult,
    WriteResult,
)


@runtime_checkable
class Filesystem(Protocol):
    """Unified filesystem protocol for workspace operations.

    This protocol abstracts over workspace backends (host filesystem, in-memory,
    containers) so tool handlers can perform file operations without coupling
    to a specific storage implementation.

    All paths are relative strings from the workspace root. Backends normalize
    paths internally and validate they don't escape the workspace boundary.

    Implementations:

    - ``HostFilesystem``: Sandboxed access to a host directory
    - ``InMemoryFilesystem``: Session-scoped in-memory storage (in contrib)

    Example::

        def read_config(fs: Filesystem) -> dict[str, Any]:
            if fs.exists("config.json"):
                result = fs.read("config.json")
                return json.loads(result.content)
            return {}
    """

    # --- Read Operations ---

    def read(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int | None = None,
        encoding: str = "utf-8",
    ) -> ReadResult:
        """Read file content as text with optional line-based pagination.

        Note: For binary files or exact file copying, use read_bytes() instead.

        Args:
            path: Relative path from workspace root.
            offset: Line number to start reading (0-indexed). Unlike read_bytes()
                which uses byte offset, this operates on lines.
            limit: Maximum lines to return. None means backend default (2000).
                Use READ_ENTIRE_FILE (-1) to read entire file without truncation.
            encoding: Text encoding. Only "utf-8" is guaranteed.

        Returns:
            ReadResult containing the content and pagination metadata.

        Raises:
            FileNotFoundError: Path does not exist.
            IsADirectoryError: Path is a directory.
            PermissionError: Read access denied.
            ValueError: File contains binary content that cannot be decoded.
        """
        ...

    def read_bytes(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> ReadBytesResult:
        """Read file content as raw bytes with optional byte-based pagination.

        This is the recommended method for copying files, as it preserves
        binary content exactly without encoding/decoding overhead.

        Args:
            path: Relative path from workspace root.
            offset: Byte offset to start reading (0-indexed). Unlike read()
                which uses line offset, this operates on bytes.
            limit: Maximum bytes to return. None means read entire file.

        Returns:
            ReadBytesResult containing the raw bytes and pagination metadata.

        Raises:
            FileNotFoundError: Path does not exist.
            IsADirectoryError: Path is a directory.
            PermissionError: Read access denied.
        """
        ...

    def exists(self, path: str) -> bool:
        """Check if a path exists in the filesystem.

        Args:
            path: Relative path from workspace root.

        Returns:
            True if the path exists (file or directory), False otherwise.
            Returns False for paths that escape the workspace root.
        """
        ...

    def stat(self, path: str) -> FileStat:
        """Get metadata for a file or directory.

        Args:
            path: Relative path from workspace root.

        Returns:
            FileStat containing size, timestamps, and type information.

        Raises:
            FileNotFoundError: Path does not exist.
            PermissionError: Path escapes workspace root.
        """
        ...

    def list(self, path: str = ".") -> Sequence[FileEntry]:
        """List directory contents.

        Args:
            path: Directory to list. Defaults to workspace root.

        Returns:
            Sequence of FileEntry objects sorted by name.

        Raises:
            FileNotFoundError: Path does not exist.
            NotADirectoryError: Path is a file.
            PermissionError: Path escapes workspace root.
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
        """Write text content to a file.

        Args:
            path: Relative path from workspace root.
            content: UTF-8 text content to write.
            mode: Write behavior.
                - "create": Fail if file exists.
                - "overwrite": Replace existing content (default).
                - "append": Add to end of file.
            create_parents: Create parent directories if missing (default True).

        Returns:
            WriteResult confirming path, bytes written, and mode used.

        Raises:
            FileExistsError: mode="create" and file exists.
            FileNotFoundError: Parent directory missing and create_parents=False.
            PermissionError: Write access denied or filesystem is read-only.
            ValueError: Content exceeds backend limits (typically 48000 chars).
        """
        ...

    def write_bytes(
        self,
        path: str,
        content: bytes,
        *,
        mode: Literal["create", "overwrite", "append"] = "overwrite",
        create_parents: bool = True,
    ) -> WriteResult:
        """Write raw bytes to a file.

        This is the recommended method for copying files, as it preserves
        binary content exactly without encoding/decoding overhead.

        Args:
            path: Relative path from workspace root.
            content: Raw byte content to write.
            mode: Write behavior.
                - "create": Fail if file exists.
                - "overwrite": Replace existing content (default).
                - "append": Add to end of file.
            create_parents: Create parent directories if missing (default True).

        Returns:
            WriteResult confirming path, bytes written, and mode used.

        Raises:
            FileExistsError: mode="create" and file exists.
            FileNotFoundError: Parent directory missing and create_parents=False.
            PermissionError: Write access denied or filesystem is read-only.
            ValueError: Content exceeds backend limits (typically 48000 bytes).
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
        """Workspace root path.

        Returns the absolute path to the workspace root directory.
        For virtual filesystems, this may be "/" or an abstract path.
        """
        ...

    @property
    def read_only(self) -> bool:
        """Whether write operations are disabled.

        When True, all write operations (write, write_bytes, delete, mkdir)
        will raise PermissionError.
        """
        ...


@runtime_checkable
class SnapshotableFilesystem(Filesystem, Protocol):
    """Filesystem that supports snapshot and restore operations.

    Extends ``Filesystem`` with methods for capturing and restoring state.
    Snapshots are immutable and can be stored in session state for rollback
    after failed tool invocations or exploratory changes.

    Example::

        # Take snapshot before risky operation
        snapshot = fs.snapshot(tag="before-delete")

        # Perform operation
        fs.delete("important_file.txt")

        # Oops, restore if needed
        fs.restore(snapshot)
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


__all__ = [
    "Filesystem",
    "SnapshotableFilesystem",
]

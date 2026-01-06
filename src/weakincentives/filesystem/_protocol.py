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
    ReadResult,
    WriteResult,
)


@runtime_checkable
class Filesystem(Protocol):
    """Unified filesystem protocol for workspace operations.

    All paths are relative strings. Backends normalize paths internally.
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


__all__ = [
    "Filesystem",
    "SnapshotableFilesystem",
]

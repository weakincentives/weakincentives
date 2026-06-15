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

"""Narrow filesystem backend protocol.

``FilesystemBackend`` is the narrow waist of the filesystem layer: a small
set of primitives that every storage backend implements. All ergonomics
(text decoding, line pagination, streaming readers/writers, size limits)
live once in the :class:`~weakincentives.filesystem.Filesystem` facade,
which is the only intended caller of a backend.

Backend contract
----------------

- Paths arriving at a backend are already normalized by the facade:
  relative, ``/``-separated, no leading/trailing slashes, no ``.`` or
  ``..`` segments. The empty string ``""`` denotes the workspace root.
- The facade enforces read-only mode, path depth/segment limits, write
  size caps, and root-path guards *before* calling a backend. Backends
  only implement storage semantics.
- ``glob`` and ``grep`` are backend methods (not facade compositions) so
  remote backends can execute them server-side.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable
from uuid import UUID

from ._types import FileEntry, FileStat, GlobMatch, GrepMatch, WriteMode


@dataclass(slots=True, frozen=True)
class SnapshotRef:
    """Opaque reference to a filesystem snapshot.

    ``token`` is backend-private state: a git commit reference for the host
    backend, a version key for the in-memory backend. Refs are only
    meaningful to a backend of the kind that created them; restoring a ref
    on the wrong backend raises
    :class:`~weakincentives.errors.SnapshotRestoreError`.
    """

    snapshot_id: UUID
    created_at: datetime
    token: str
    tag: str | None = None


@runtime_checkable
class FilesystemBackend(Protocol):
    """Storage primitives implemented by every filesystem backend.

    See the module docstring for the path and validation contract. Error
    expectations per method are documented inline; backends raise standard
    Python exceptions (``FileNotFoundError``, ``IsADirectoryError``,
    ``FileExistsError``, ``PermissionError``).
    """

    @property
    def root(self) -> str:
        """Workspace root path (``"/"`` for virtual backends)."""
        ...

    @property
    def read_only(self) -> bool:
        """True if write operations are disabled (enforced by the facade)."""
        ...

    def stat(self, path: str) -> FileStat:
        """Get metadata for a path.

        Raises:
            FileNotFoundError: Path does not exist.
        """
        ...

    def list(self, path: str) -> Sequence[FileEntry]:
        """List entries of an existing directory, sorted by name."""
        ...

    def glob(self, pattern: str, *, path: str) -> Sequence[GlobMatch]:
        """Match files under an existing base directory, sorted by path."""
        ...

    def grep(
        self,
        pattern: str,
        *,
        path: str,
        glob: str | None,
        max_matches: int,
    ) -> Sequence[GrepMatch]:
        """Regex-search file contents under an existing base directory.

        Results are ordered by ``(path, line_number)``; binary files are
        skipped.
        """
        ...

    def read_range(
        self,
        path: str,
        *,
        offset: int,
        length: int | None,
    ) -> bytes:
        """Read up to ``length`` bytes starting at ``offset``.

        ``length=None`` reads to EOF. Returns ``b""`` when ``offset`` is at
        or past EOF.

        Raises:
            FileNotFoundError: Path does not exist.
            IsADirectoryError: Path is a directory.
        """
        ...

    def write(self, path: str, data: bytes, *, mode: WriteMode) -> None:
        """Write bytes to a file, creating parent directories as needed.

        The facade rejects writes targeting an existing directory before
        calling this; backends whose files and directories share a namespace
        should still raise ``IsADirectoryError`` defensively.

        Raises:
            FileExistsError: ``mode="create"`` and the file exists.
            IsADirectoryError: A directory occupies the path.
        """
        ...

    def delete(self, path: str, *, recursive: bool) -> None:
        """Delete a file, or a directory when empty or ``recursive=True``.

        Raises:
            FileNotFoundError: Path does not exist.
            IsADirectoryError: Non-empty directory and ``recursive=False``.
        """
        ...

    def mkdir(self, path: str) -> None:
        """Create a directory and any missing parents (idempotent).

        The facade rejects chains that cross an existing file before calling
        this; backends whose files and directories share a namespace should
        still raise ``NotADirectoryError`` defensively.
        """
        ...

    def rename(self, src: str, dst: str) -> None:
        """Move a file or directory (with its contents) from ``src`` to ``dst``.

        ``dst``'s parent directories are created as needed. Renaming is a
        primitive — not facade copy+delete — so backends that can move
        atomically do. The facade rejects a ``dst`` whose parent chain crosses
        an existing file before calling this.

        Raises:
            FileNotFoundError: ``src`` does not exist.
            FileExistsError: ``dst`` already exists.
        """
        ...

    def snapshot(self, *, tag: str | None) -> SnapshotRef:
        """Capture current state as an opaque, immutable snapshot ref."""
        ...

    def restore(self, ref: SnapshotRef) -> None:
        """Restore state captured by :meth:`snapshot`.

        Raises:
            SnapshotRestoreError: Ref is unknown to this backend or the
                restore operation failed.
        """
        ...


__all__ = [
    "FilesystemBackend",
    "SnapshotRef",
]

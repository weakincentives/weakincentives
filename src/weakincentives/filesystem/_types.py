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

"""Core filesystem types and result dataclasses.

This module defines the data structures used by the Filesystem protocol.
These types are used throughout the codebase for filesystem operations.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, Literal
from uuid import UUID

from ..clock import SYSTEM_CLOCK
from ._path import (
    MAX_PATH_DEPTH,
    MAX_SEGMENT_LENGTH,
    normalize_path_string,
    validate_path,
)

DEFAULT_READ_LIMIT: Final[int] = 2_000
MAX_WRITE_LENGTH: Final[int] = 33_554_432  # 32MB
MAX_WRITE_BYTES: Final[int] = 33_554_432  # 32MB
MAX_GREP_MATCHES: Final[int] = 1_000

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
    """Content returned from text read operations."""

    content: str
    path: str
    total_lines: int
    offset: int
    limit: int
    truncated: bool


@dataclass(slots=True, frozen=True)
class ReadBytesResult:
    """Content returned from binary read operations."""

    content: bytes
    path: str
    size_bytes: int
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

    For HostFilesystem, ``git_dir`` stores the external git repository
    location to enable cross-session restore.
    """

    snapshot_id: UUID
    created_at: datetime
    commit_ref: str
    root_path: str
    git_dir: str | None = None
    tag: str | None = None


@dataclass(slots=True, frozen=True)
class SnapshotHistoryEntry:
    """Metadata for a single snapshot in the filesystem history.

    Each entry corresponds to a git commit created by ``snapshot()``.
    The ``tag`` field encodes the tool call context using the format
    ``"pre:{tool_name}:{tool_call_id}"``, from which ``tool_call_id``
    and ``tool_name`` are extracted.
    """

    commit_ref: str
    created_at: str
    tag: str | None = None
    parent_ref: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    files_changed: tuple[str, ...] = ()
    insertions: int = 0
    deletions: int = 0
    rolled_back: bool = False


# ---------------------------------------------------------------------------
# Shared Utility Functions
# ---------------------------------------------------------------------------


def normalize_path(path: str) -> str:
    """Normalize a path by removing leading/trailing slashes and cleaning segments.

    Delegates to the shared implementation in :mod:`weakincentives.filesystem._path`.
    """
    return normalize_path_string(path)


def now() -> datetime:
    """Return current UTC time truncated to milliseconds."""
    value = SYSTEM_CLOCK.utcnow()
    microsecond = value.microsecond - value.microsecond % 1000
    return value.replace(microsecond=microsecond, tzinfo=UTC)


def is_path_under(path: str, base: str) -> bool:
    """Check if path is under base directory."""
    if not base:
        return True
    return path == base or path.startswith(f"{base}/")


def glob_match(path: str, pattern: str, base: str) -> bool:
    """Check if path matches glob pattern relative to base."""
    if base:
        if not is_path_under(path, base):
            return False
        relative = path[len(base) + 1 :] if path != base else ""
    else:
        relative = path
    return fnmatch.fnmatch(relative, pattern)


__all__ = [
    "DEFAULT_READ_LIMIT",
    "MAX_GREP_MATCHES",
    "MAX_PATH_DEPTH",
    "MAX_SEGMENT_LENGTH",
    "MAX_WRITE_BYTES",
    "MAX_WRITE_LENGTH",
    "READ_ENTIRE_FILE",
    "FileEncoding",
    "FileEntry",
    "FileStat",
    "FilesystemSnapshot",
    "GlobMatch",
    "GrepMatch",
    "ReadBytesResult",
    "ReadResult",
    "SnapshotHistoryEntry",
    "WriteMode",
    "WriteResult",
    "glob_match",
    "is_path_under",
    "normalize_path",
    "now",
    "validate_path",
]

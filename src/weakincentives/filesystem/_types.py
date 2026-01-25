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

This module defines the data structures used by the ``Filesystem`` protocol.
All types are immutable frozen dataclasses suitable for storage in session state.

Types are organized into:

- **Result types**: ``ReadResult``, ``ReadBytesResult``, ``WriteResult`` - returned
  by filesystem operations
- **Metadata types**: ``FileStat``, ``FileEntry`` - file and directory information
- **Search types**: ``GlobMatch``, ``GrepMatch`` - pattern matching results
- **Snapshot types**: ``FilesystemSnapshot`` - state capture for rollback

Constants:

- ``DEFAULT_READ_LIMIT``: Default line limit for text reads (2000 lines)
- ``MAX_WRITE_LENGTH``: Maximum characters for text writes (48000)
- ``MAX_WRITE_BYTES``: Maximum bytes for binary writes (48000)
- ``MAX_GREP_MATCHES``: Maximum grep matches returned (1000)
- ``READ_ENTIRE_FILE``: Pass as ``limit`` to read without truncation (-1)
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, Literal
from uuid import UUID

from ._path import (
    MAX_PATH_DEPTH,
    MAX_SEGMENT_LENGTH,
    normalize_path_string,
    validate_path,
)

DEFAULT_READ_LIMIT: Final[int] = 2_000
MAX_WRITE_LENGTH: Final[int] = 48_000
MAX_WRITE_BYTES: Final[int] = 48_000
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
    """Metadata for a file or directory.

    Returned by ``Filesystem.stat()`` to provide file system metadata
    without reading the file contents.

    Attributes:
        path: Normalized path relative to filesystem root.
        is_file: True if this is a regular file.
        is_directory: True if this is a directory.
        size_bytes: File size in bytes (0 for directories).
        created_at: File creation time (may be None for virtual filesystems).
        modified_at: Last modification time (may be None for virtual filesystems).

    Example::

        stat = fs.stat("src/main.py")
        if stat.is_file and stat.size_bytes > 0:
            content = fs.read(stat.path)
    """

    path: str
    is_file: bool
    is_directory: bool
    size_bytes: int
    created_at: datetime | None = None
    modified_at: datetime | None = None


@dataclass(slots=True, frozen=True)
class FileEntry:
    """Directory listing entry returned by ``Filesystem.list()``.

    Represents a single entry in a directory listing, providing both
    the entry name and its full path relative to the filesystem root.

    Attributes:
        name: Entry name without path (e.g., "main.py").
        path: Full path relative to filesystem root (e.g., "src/main.py").
        is_file: True if this entry is a regular file.
        is_directory: True if this entry is a directory.

    Example::

        for entry in fs.list("src"):
            if entry.is_file and entry.name.endswith(".py"):
                print(f"Python file: {entry.path}")
    """

    name: str
    path: str
    is_file: bool
    is_directory: bool


@dataclass(slots=True, frozen=True)
class GlobMatch:
    """Result from ``Filesystem.glob()`` pattern matching.

    Represents a file that matched the glob pattern. Results are sorted
    by path for deterministic ordering.

    Attributes:
        path: Path relative to filesystem root that matched the pattern.
        is_file: True if the match is a file (currently always True).

    Example::

        # Find all Python files in src/
        for match in fs.glob("**/*.py", path="src"):
            content = fs.read(match.path)
    """

    path: str
    is_file: bool


@dataclass(slots=True, frozen=True)
class GrepMatch:
    """Result from ``Filesystem.grep()`` content search.

    Represents a single line that matched the regex pattern, including
    the exact position of the match within the line.

    Attributes:
        path: Path relative to filesystem root containing the match.
        line_number: 1-indexed line number where the match occurred.
        line_content: Full content of the matching line (without newline).
        match_start: Character offset where the match starts (0-indexed).
        match_end: Character offset where the match ends (exclusive).

    Example::

        for match in fs.grep(r"def\\s+\\w+", glob="**/*.py"):
            print(f"{match.path}:{match.line_number}: {match.line_content}")
            # Highlight the matched portion
            highlighted = match.line_content[match.match_start:match.match_end]
    """

    path: str
    line_number: int
    line_content: str
    match_start: int
    match_end: int


@dataclass(slots=True, frozen=True)
class ReadResult:
    """Content returned from ``Filesystem.read()`` text operations.

    Contains the file content along with pagination metadata to support
    reading large files in chunks.

    Attributes:
        content: File content as text (lines joined, trailing newline stripped).
        path: Normalized path relative to filesystem root.
        total_lines: Total number of lines in the file.
        offset: Starting line number of returned content (0-indexed).
        limit: Number of lines actually returned.
        truncated: True if more content exists beyond what was returned.

    Example::

        result = fs.read("large_file.txt", offset=0, limit=100)
        print(result.content)
        if result.truncated:
            # Read next chunk
            next_result = fs.read("large_file.txt", offset=100, limit=100)
    """

    content: str
    path: str
    total_lines: int
    offset: int
    limit: int
    truncated: bool


@dataclass(slots=True, frozen=True)
class ReadBytesResult:
    """Content returned from ``Filesystem.read_bytes()`` binary operations.

    Contains raw bytes along with pagination metadata. Use this for binary
    files or when exact byte-for-byte copying is required.

    Attributes:
        content: Raw bytes read from the file.
        path: Normalized path relative to filesystem root.
        size_bytes: Total size of the file in bytes.
        offset: Starting byte offset of returned content (0-indexed).
        limit: Number of bytes actually returned.
        truncated: True if more content exists beyond what was returned.

    Example::

        # Copy a binary file exactly
        result = fs.read_bytes("image.png")
        fs.write_bytes("copy.png", result.content)

        # Read in chunks for large files
        result = fs.read_bytes("large.bin", offset=0, limit=4096)
        while result.truncated:
            next_offset = result.offset + len(result.content)
            result = fs.read_bytes("large.bin", offset=next_offset, limit=4096)
    """

    content: bytes
    path: str
    size_bytes: int
    offset: int
    limit: int
    truncated: bool


@dataclass(slots=True, frozen=True)
class WriteResult:
    """Confirmation returned from ``Filesystem.write()`` and ``write_bytes()``.

    Provides details about the completed write operation for verification
    and logging purposes.

    Attributes:
        path: Normalized path relative to filesystem root where content was written.
        bytes_written: Number of bytes actually written to the file.
        mode: Write mode used ("create", "overwrite", or "append").

    Example::

        result = fs.write("output.txt", "Hello, World!")
        print(f"Wrote {result.bytes_written} bytes to {result.path}")
    """

    path: str
    bytes_written: int
    mode: WriteMode


@dataclass(slots=True, frozen=True)
class FilesystemSnapshot:
    """Immutable capture of filesystem state for rollback.

    Snapshots capture the complete state of a workspace at a point in time,
    enabling rollback after failed tool invocations or exploratory changes.
    Store snapshots in session state to support undo functionality.

    The implementation varies by filesystem backend:

    - ``HostFilesystem``: Uses git commits for efficient copy-on-write storage.
      The ``git_dir`` field stores the external git repository location.
    - ``InMemoryFilesystem``: Stores a deep copy of the filesystem tree.
      The ``commit_ref`` contains an internal version identifier.

    Attributes:
        snapshot_id: Unique identifier for this snapshot.
        created_at: UTC timestamp when the snapshot was created.
        commit_ref: Backend-specific state reference (git commit hash or version ID).
        root_path: Filesystem root path at time of snapshot.
        git_dir: External git directory path (HostFilesystem only, None for others).
        tag: Optional human-readable label for identifying the snapshot.

    Example::

        # Take a snapshot before risky operations
        snapshot = fs.snapshot(tag="before-refactor")

        # Perform operations...
        fs.write("src/main.py", new_content)

        # Rollback if needed
        if something_went_wrong:
            fs.restore(snapshot)
    """

    snapshot_id: UUID
    created_at: datetime
    commit_ref: str
    root_path: str
    git_dir: str | None = None
    tag: str | None = None


# ---------------------------------------------------------------------------
# Shared Utility Functions
# ---------------------------------------------------------------------------


def normalize_path(path: str) -> str:
    """Normalize a path for consistent filesystem operations.

    Removes leading/trailing slashes, collapses multiple slashes, and resolves
    ``.`` segments. Does not resolve ``..`` segments (use filesystem's resolve
    for that).

    Args:
        path: Raw path string to normalize.

    Returns:
        Normalized path string. Returns empty string for root paths (".", "/", "").

    Example::

        normalize_path("/foo//bar/")  # Returns "foo/bar"
        normalize_path("./src/main.py")  # Returns "src/main.py"
        normalize_path(".")  # Returns ""
    """
    return normalize_path_string(path)


def now() -> datetime:
    """Return current UTC time truncated to milliseconds.

    Provides consistent timestamp precision across filesystem operations.
    Microseconds are rounded down to the nearest millisecond for compatibility
    with systems that don't support microsecond precision.

    Returns:
        Current UTC datetime with microseconds truncated to millisecond precision.
    """
    value = datetime.now(UTC)
    microsecond = value.microsecond - value.microsecond % 1000
    return value.replace(microsecond=microsecond, tzinfo=UTC)


def is_path_under(path: str, base: str) -> bool:
    """Check if a path is under a base directory.

    Used internally to validate that operations stay within expected
    directory boundaries during glob and grep operations.

    Args:
        path: Normalized path to check.
        base: Base directory path. Empty string matches all paths.

    Returns:
        True if path equals base or is a descendant of base.

    Example::

        is_path_under("src/main.py", "src")  # Returns True
        is_path_under("src/main.py", "lib")  # Returns False
        is_path_under("src", "src")  # Returns True
        is_path_under("src/main.py", "")  # Returns True (empty base)
    """
    if not base:
        return True
    return path == base or path.startswith(f"{base}/")


def glob_match(path: str, pattern: str, base: str) -> bool:
    """Check if a path matches a glob pattern relative to a base directory.

    The path is made relative to the base directory before matching against
    the pattern using ``fnmatch`` semantics.

    Args:
        path: Normalized path to test.
        pattern: Glob pattern to match against (e.g., "**/*.py", "*.txt").
        base: Base directory for relative matching. Empty string uses path as-is.

    Returns:
        True if the path matches the pattern when considered relative to base.

    Example::

        glob_match("src/main.py", "*.py", "src")  # Returns True
        glob_match("src/main.py", "**/*.py", "")  # Returns True
        glob_match("lib/util.py", "*.py", "src")  # Returns False (not under src)
    """
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
    "WriteMode",
    "WriteResult",
    "glob_match",
    "is_path_under",
    "normalize_path",
    "now",
    "validate_path",
]

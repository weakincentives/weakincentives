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

"""Filesystem abstraction layer for workspace operations.

This package provides a unified ``Filesystem`` protocol that abstracts over
different storage backends, enabling tool handlers to perform file operations
without coupling to a specific implementation. The abstraction supports both
host-backed (sandboxed directory) and virtual (in-memory) filesystems.

Overview
--------

The filesystem abstraction serves two primary purposes:

1. **Sandboxing**: Restrict agent file access to a designated workspace root,
   preventing path traversal attacks and unauthorized access to system files.

2. **Backend Flexibility**: Allow the same tool code to work with different
   storage backends (host directories, in-memory stores, containers) without
   modification.

Host Filesystem vs Virtual Filesystem
-------------------------------------

**Host Filesystem** (``HostFilesystem``):
    A sandboxed view of an actual directory on the host machine. All paths are
    resolved relative to a root directory, and symlinks/traversal attempts that
    escape the sandbox raise ``PermissionError``. Supports git-based snapshots
    for rollback capability.

**Virtual Filesystem** (``InMemoryFilesystem`` in ``weakincentives.contrib.tools``):
    A session-scoped in-memory filesystem that exists only for the duration of
    an agent session. Useful for testing, evaluation runs, or scenarios where
    persistence is not required.

Protocols
---------

Filesystem
    The core protocol defining read, write, list, glob, and grep operations.
    All paths are relative strings; backends normalize paths internally.

SnapshotableFilesystem
    Extended protocol adding ``snapshot()`` and ``restore()`` methods for
    capturing and rolling back filesystem state.

Implementations
---------------

HostFilesystem
    Sandboxed host directory access. Creates an external git repository for
    snapshot storage (outside the workspace root to prevent agent access).

InMemoryFilesystem
    Available in ``weakincentives.contrib.tools.filesystem_memory``. Provides
    session-scoped in-memory storage with snapshot support.

Result Types
------------

ReadResult
    Text content with pagination metadata (total_lines, offset, limit, truncated).

ReadBytesResult
    Binary content with byte-level pagination metadata.

WriteResult
    Confirmation of write operation (path, bytes_written, mode).

FileStat
    File/directory metadata (size, timestamps, type flags).

FileEntry
    Directory listing entry (name, path, type flags).

GlobMatch
    Glob pattern match result (path, is_file flag).

GrepMatch
    Regex search result (path, line_number, line_content, match positions).

FilesystemSnapshot
    Immutable snapshot for rollback (snapshot_id, commit_ref, timestamps).

Type Aliases
------------

FileEncoding
    Literal type for supported encodings (currently only ``"utf-8"``).

WriteMode
    Literal type for write modes: ``"create"``, ``"overwrite"``, ``"append"``.

Constants
---------

DEFAULT_READ_LIMIT
    Default number of lines returned by ``read()`` (2000).

READ_ENTIRE_FILE
    Pass as ``limit`` to ``read()`` to read entire file without truncation (-1).

MAX_WRITE_LENGTH
    Maximum text content length for ``write()`` (32MB).

MAX_WRITE_BYTES
    Maximum binary content size for ``write_bytes()`` (32MB).

MAX_GREP_MATCHES
    Maximum matches returned by ``grep()`` (1,000).

MAX_PATH_DEPTH
    Maximum allowed path depth (16 segments).

MAX_SEGMENT_LENGTH
    Maximum allowed path segment length (80 characters).

Utility Functions
-----------------

normalize_path(path)
    Normalize a path by removing leading/trailing slashes, collapsing ``..``
    segments, and removing empty/``.`` segments. Returns empty string for root.

validate_path(path)
    Validate path constraints (depth and segment length). Raises ``ValueError``
    if constraints are violated.

is_path_under(path, base)
    Check if a path is contained within a base directory.

glob_match(path, pattern, base)
    Check if a path matches a glob pattern relative to a base directory.

now()
    Return current UTC time truncated to milliseconds.

Usage Examples
--------------

Basic file operations::

    from weakincentives.filesystem import Filesystem, HostFilesystem

    # Create a sandboxed filesystem rooted at a workspace directory
    fs = HostFilesystem(_root="/path/to/workspace")

    # Write a file (creates parent directories by default)
    result = fs.write("src/main.py", "print('hello')")
    print(f"Wrote {result.bytes_written} bytes to {result.path}")

    # Read with pagination
    result = fs.read("src/main.py", offset=0, limit=100)
    print(f"Content: {result.content}")
    print(f"Truncated: {result.truncated}")

    # Read entire file without truncation
    from weakincentives.filesystem import READ_ENTIRE_FILE
    result = fs.read("large_file.txt", limit=READ_ENTIRE_FILE)

Binary file operations::

    # Read binary content
    result = fs.read_bytes("image.png")
    data = result.content  # bytes

    # Write binary content
    fs.write_bytes("output.bin", data, mode="create")

Search operations::

    # Find all Python files
    matches = fs.glob("**/*.py")
    for match in matches:
        print(match.path)

    # Search for a pattern in Python files
    matches = fs.grep(r"def\\s+main", glob="*.py")
    for match in matches:
        print(f"{match.path}:{match.line_number}: {match.line_content}")

Snapshot and restore (HostFilesystem)::

    from weakincentives.filesystem import HostFilesystem

    fs = HostFilesystem(_root="/path/to/workspace")

    # Capture state before risky operation
    snapshot = fs.snapshot(tag="before-refactor")

    # Make changes...
    fs.write("src/main.py", "# broken code")

    # Rollback if something went wrong
    fs.restore(snapshot)

    # Clean up git storage when done
    fs.cleanup()

Writing filesystem-agnostic tools::

    from weakincentives.filesystem import Filesystem, ReadResult

    def count_lines(fs: Filesystem, path: str) -> int:
        \"\"\"Count lines in a file using any filesystem backend.\"\"\"
        result = fs.read(path, limit=-1)  # READ_ENTIRE_FILE
        return result.total_lines

    # Works with HostFilesystem
    host_fs = HostFilesystem(_root="/workspace")
    print(count_lines(host_fs, "src/main.py"))

    # Works with InMemoryFilesystem
    from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
    mem_fs = InMemoryFilesystem()
    mem_fs.write("test.txt", "line1\\nline2\\nline3")
    print(count_lines(mem_fs, "test.txt"))  # Output: 3
"""

from __future__ import annotations

from ._host import HostFilesystem
from ._protocol import Filesystem, SnapshotableFilesystem
from ._types import (
    DEFAULT_READ_LIMIT,
    MAX_GREP_MATCHES,
    MAX_PATH_DEPTH,
    MAX_SEGMENT_LENGTH,
    MAX_WRITE_BYTES,
    MAX_WRITE_LENGTH,
    READ_ENTIRE_FILE,
    FileEncoding,
    FileEntry,
    FileStat,
    FilesystemSnapshot,
    GlobMatch,
    GrepMatch,
    ReadBytesResult,
    ReadResult,
    WriteMode,
    WriteResult,
    glob_match,
    is_path_under,
    normalize_path,
    now,
    validate_path,
)

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
    "Filesystem",
    "FilesystemSnapshot",
    "GlobMatch",
    "GrepMatch",
    "HostFilesystem",
    "ReadBytesResult",
    "ReadResult",
    "SnapshotableFilesystem",
    "WriteMode",
    "WriteResult",
    "glob_match",
    "is_path_under",
    "normalize_path",
    "now",
    "validate_path",
]

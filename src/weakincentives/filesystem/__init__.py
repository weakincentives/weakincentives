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

This package separates filesystem concerns into a narrow waist and one
ergonomic surface:

- :class:`FilesystemBackend` — the narrow protocol (~10 storage primitives)
  every backend implements. Backends stay small: a usable fake fits in
  under 100 lines.
- :class:`Filesystem` — the one concrete facade tools program against. It
  composes text decoding, line pagination, streaming, validation, limits,
  and read-only enforcement once, over any backend.

Backends
--------

:class:`HostBackend`
    Sandboxed host directory. The root is resolved once at construction;
    every operation re-validates that its symlink-resolved target stays
    inside the sandbox. Snapshots are git commits in an external repository
    (outside the workspace root) following a strict rollback contract:
    every file is captured (including ``.gitignore``-matched ones) and
    ``restore`` returns the workspace to the exact snapshot state.

:class:`MemoryBackend`
    Session-scoped in-memory storage with structural-sharing snapshots.
    Useful for tests and evaluation runs.

Snapshots
---------

``snapshot()`` returns an opaque :class:`SnapshotRef`; its ``token`` is
backend-private (a git commit for :class:`HostBackend`, a version key for
:class:`MemoryBackend`). ``restore(ref)`` raises
:class:`~weakincentives.errors.SnapshotRestoreError` for refs the backend
does not recognize.

Usage
-----

Basic file operations::

    from weakincentives.filesystem import Filesystem

    fs = Filesystem.host("/path/to/workspace")  # or Filesystem.in_memory()

    fs.write("src/main.py", "print('hello')")
    result = fs.read("src/main.py", offset=0, limit=100)
    print(result.content, result.truncated)

    for match in fs.grep(r"def\\s+main", glob="*.py"):
        print(f"{match.path}:{match.line_number}: {match.line_content}")

Streaming large files::

    with fs.open_read("large.bin") as reader:
        for chunk in reader.chunks(size=65536):
            process(chunk)

    with fs.open_write("output.bin", mode="create") as writer:
        for chunk in chunks:
            writer.write(chunk)

Snapshot and restore::

    ref = fs.snapshot(tag="before-refactor")
    fs.write("src/main.py", "# broken")
    fs.restore(ref)

Custom backends implement :class:`FilesystemBackend` and wrap themselves in
the facade: ``Filesystem(MyBackend(...))``.
"""

from __future__ import annotations

from ._backend import FilesystemBackend, SnapshotRef
from ._facade import Filesystem
from ._host import HostBackend
from ._memory import MemoryBackend
from ._path import strip_mount_point
from ._streams import (
    DEFAULT_CHUNK_SIZE,
    ByteReader,
    ByteWriter,
    TextReader,
)
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
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_READ_LIMIT",
    "MAX_GREP_MATCHES",
    "MAX_PATH_DEPTH",
    "MAX_SEGMENT_LENGTH",
    "MAX_WRITE_BYTES",
    "MAX_WRITE_LENGTH",
    "READ_ENTIRE_FILE",
    "ByteReader",
    "ByteWriter",
    "FileEncoding",
    "FileEntry",
    "FileStat",
    "Filesystem",
    "FilesystemBackend",
    "GlobMatch",
    "GrepMatch",
    "HostBackend",
    "MemoryBackend",
    "ReadBytesResult",
    "ReadResult",
    "SnapshotRef",
    "TextReader",
    "WriteMode",
    "WriteResult",
    "glob_match",
    "is_path_under",
    "normalize_path",
    "now",
    "strip_mount_point",
    "validate_path",
]

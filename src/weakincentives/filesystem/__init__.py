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

"""Core filesystem protocol and types for workspace operations.

This module provides the `Filesystem` protocol that abstracts over workspace
backends (in-memory VFS, Podman containers, host filesystem) so tool handlers
can perform file operations without coupling to a specific storage implementation.

Example usage::

    from weakincentives.filesystem import Filesystem, ReadResult

    def read_file(fs: Filesystem, path: str) -> ReadResult:
        return fs.read(path)

Implementations are provided in ``weakincentives.contrib.tools``:

- ``InMemoryFilesystem``: Session-scoped in-memory storage
- ``HostFilesystem``: Sandboxed host directory access
"""

from __future__ import annotations

from ._protocol import Filesystem, SnapshotableFilesystem
from ._types import (
    ASCII,
    DEFAULT_READ_LIMIT,
    MAX_GREP_MATCHES,
    MAX_PATH_DEPTH,
    MAX_SEGMENT_LENGTH,
    MAX_WRITE_LENGTH,
    READ_ENTIRE_FILE,
    FileEncoding,
    FileEntry,
    FileStat,
    FilesystemSnapshot,
    GlobMatch,
    GrepMatch,
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
    "ASCII",
    "DEFAULT_READ_LIMIT",
    "MAX_GREP_MATCHES",
    "MAX_PATH_DEPTH",
    "MAX_SEGMENT_LENGTH",
    "MAX_WRITE_LENGTH",
    "READ_ENTIRE_FILE",
    "FileEncoding",
    "FileEntry",
    "FileStat",
    "Filesystem",
    "FilesystemSnapshot",
    "GlobMatch",
    "GrepMatch",
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

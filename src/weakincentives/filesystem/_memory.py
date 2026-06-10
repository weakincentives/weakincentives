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

"""In-memory backend for the filesystem facade.

Implements :class:`~weakincentives.filesystem.FilesystemBackend` primitives
over plain dictionaries, suitable for tests and session-scoped virtual
storage. Snapshots use structural sharing: file contents are shared between
the live state and snapshots, so only modified files allocate new memory.
"""

from __future__ import annotations

import re
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from ..clock import SYSTEM_CLOCK
from ..errors import SnapshotRestoreError
from ._backend import SnapshotRef
from ._types import (
    FileEntry,
    FileStat,
    GlobMatch,
    GrepMatch,
    WriteMode,
    glob_match,
    is_path_under,
    now,
)

__all__ = ["MemoryBackend"]


@dataclass(slots=True)
class _MemoryFile:
    """A file stored as raw bytes with timestamps."""

    content: bytes
    created_at: datetime
    modified_at: datetime


@dataclass(slots=True, frozen=True)
class _MemoryState:
    """Frozen snapshot of in-memory state (contents shared by reference)."""

    files: Mapping[str, _MemoryFile]
    directories: frozenset[str]


class MemoryBackend:
    """Filesystem primitives over in-memory dictionaries.

    All paths are facade-normalized relative strings (``""`` is the root,
    which always exists as a directory).
    """

    def __init__(self, *, read_only: bool = False) -> None:
        super().__init__()
        self._files: dict[str, _MemoryFile] = {}
        self._directories: set[str] = {""}
        self._read_only = read_only
        self._snapshots: dict[str, _MemoryState] = {}
        self._version = 0

    @property
    def root(self) -> str:
        """Workspace root path."""
        return "/"

    @property
    def read_only(self) -> bool:
        """True if write operations are disabled (enforced by the facade)."""
        return self._read_only

    # --- Primitives ---------------------------------------------------------

    def stat(self, path: str) -> FileStat:
        """Get metadata for a path."""
        if path in self._directories:
            return FileStat(
                path=path or "/",
                is_file=False,
                is_directory=True,
                size_bytes=0,
                created_at=None,
                modified_at=None,
            )
        file = self._files.get(path)
        if file is None:
            raise FileNotFoundError(path)
        return FileStat(
            path=path,
            is_file=True,
            is_directory=False,
            size_bytes=len(file.content),
            created_at=file.created_at,
            modified_at=file.modified_at,
        )

    def list(self, path: str) -> Sequence[FileEntry]:
        """List entries of an existing directory, sorted by name."""
        prefix = f"{path}/" if path else ""
        seen: set[str] = set()
        entries: list[FileEntry] = []
        # Files contribute direct children and implicit subdirectories
        for file_path in self._files:
            if not is_path_under(file_path, path):
                continue
            relative = file_path[len(prefix) :]
            if "/" in relative:
                child = relative.split("/")[0]
                if child not in seen:
                    seen.add(child)
                    entries.append(
                        FileEntry(
                            name=child,
                            path=f"{prefix}{child}",
                            is_file=False,
                            is_directory=True,
                        )
                    )
            else:
                entries.append(
                    FileEntry(
                        name=relative,
                        path=file_path,
                        is_file=True,
                        is_directory=False,
                    )
                )
        # Explicitly created directories not already covered
        for dir_path in self._directories:
            if not is_path_under(dir_path, path) or dir_path == path:
                continue
            relative = dir_path[len(prefix) :]
            if "/" not in relative and relative not in seen:
                seen.add(relative)
                entries.append(
                    FileEntry(
                        name=relative,
                        path=dir_path,
                        is_file=False,
                        is_directory=True,
                    )
                )
        entries.sort(key=lambda e: e.name)
        return entries

    def glob(self, pattern: str, *, path: str) -> Sequence[GlobMatch]:
        """Match files under a base directory, sorted by path."""
        matches = [
            GlobMatch(path=file_path, is_file=True)
            for file_path in self._files
            if glob_match(file_path, pattern, path)
        ]
        matches.sort(key=lambda m: m.path)
        return matches

    def grep(
        self,
        pattern: str,
        *,
        path: str,
        glob: str | None,
        max_matches: int,
    ) -> Sequence[GrepMatch]:
        """Regex-search file contents under a base directory."""
        regex = re.compile(pattern)
        matches: list[GrepMatch] = []
        for file_path, file in sorted(self._files.items()):
            if not is_path_under(file_path, path):
                continue
            if glob is not None and not glob_match(file_path, glob, path):
                continue
            try:
                text = file.content.decode("utf-8")
            except UnicodeDecodeError:
                continue
            for line_num, line in enumerate(text.splitlines(), start=1):
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
                    if len(matches) >= max_matches:
                        return matches
        return matches

    def read_range(
        self,
        path: str,
        *,
        offset: int,
        length: int | None,
    ) -> bytes:
        """Read up to ``length`` bytes starting at ``offset``."""
        if path in self._directories:
            msg = f"Is a directory: {path}"
            raise IsADirectoryError(msg)
        file = self._files.get(path)
        if file is None:
            raise FileNotFoundError(path)
        if length is None:
            return file.content[offset:]
        return file.content[offset : offset + length]

    def write(self, path: str, data: bytes, *, mode: WriteMode) -> None:
        """Write bytes to a file, creating parent directories as needed."""
        if path in self._directories:
            # Defensive: files and directories share one namespace here, so a
            # write aimed at a directory would corrupt state (the host backend
            # gets the equivalent error from the OS).
            msg = f"Is a directory: {path}"
            raise IsADirectoryError(msg)
        existing = self._files.get(path)
        if mode == "create" and existing is not None:
            raise FileExistsError(f"File already exists: {path}")
        parent = path.rsplit("/", 1)[0] if "/" in path else ""
        if parent:
            self.mkdir(parent)
        timestamp = now()
        if mode == "append" and existing is not None:
            content = existing.content + data
        else:
            content = data
        created_at = existing.created_at if existing is not None else timestamp
        self._files[path] = _MemoryFile(
            content=content,
            created_at=created_at,
            modified_at=timestamp,
        )

    def delete(self, path: str, *, recursive: bool) -> None:
        """Delete a file, or a directory when empty or ``recursive=True``."""
        if path in self._files:
            del self._files[path]
            return
        if path not in self._directories:
            raise FileNotFoundError(path)
        if recursive:
            for file_path in [f for f in self._files if is_path_under(f, path)]:
                del self._files[file_path]
            for dir_path in [
                d for d in self._directories if is_path_under(d, path) and d != path
            ]:
                self._directories.remove(dir_path)
        elif self._has_contents(path):
            msg = f"Directory not empty: {path}"
            raise IsADirectoryError(msg)
        self._directories.discard(path)

    def _has_contents(self, path: str) -> bool:
        """Check if a directory has any files or subdirectories."""
        return any(is_path_under(f, path) for f in self._files) or any(
            is_path_under(d, path) and d != path for d in self._directories
        )

    def mkdir(self, path: str) -> None:
        """Create a directory and any missing parents (idempotent).

        Raises:
            NotADirectoryError: A file occupies a segment of the chain
                (defensive: files and directories share one namespace here,
                so a directory entry would shadow the file).
        """
        segments = path.split("/")
        for index in range(len(segments)):
            candidate = "/".join(segments[: index + 1])
            if candidate in self._files:
                msg = f"Not a directory: {candidate}"
                raise NotADirectoryError(msg)
            self._directories.add(candidate)

    # --- Snapshots ------------------------------------------------------------

    def snapshot(self, *, tag: str | None = None) -> SnapshotRef:
        """Capture current state via structural sharing."""
        self._version += 1
        token = f"mem-{self._version}"
        self._snapshots[token] = _MemoryState(
            files=types.MappingProxyType(dict(self._files)),
            directories=frozenset(self._directories),
        )
        return SnapshotRef(
            snapshot_id=uuid4(),
            created_at=SYSTEM_CLOCK.utcnow(),
            token=token,
            tag=tag,
        )

    def restore(self, ref: SnapshotRef) -> None:
        """Restore state from a snapshot taken by this backend.

        Raises:
            SnapshotRestoreError: The ref's token is unknown to this backend.
        """
        frozen = self._snapshots.get(ref.token)
        if frozen is None:
            msg = f"Unknown snapshot: {ref.token}"
            raise SnapshotRestoreError(msg)
        self._files = dict(frozen.files)
        self._directories = set(frozen.directories)
        self._directories.add("")

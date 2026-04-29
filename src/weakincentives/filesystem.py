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

"""Filesystem abstraction with snapshot/restore for tool transactions.

Tools register a :class:`Filesystem` as a snapshotable resource so the
spine's transactional tool execution can roll back filesystem mutations
alongside session state.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol, final, runtime_checkable

__all__ = [
    "Filesystem",
    "FilesystemError",
    "FilesystemSnapshot",
    "InMemoryFilesystem",
]


class FilesystemError(Exception):
    """Raised when a filesystem operation cannot complete."""


@runtime_checkable
class Filesystem(Protocol):
    """Minimal filesystem protocol supported by the spine."""

    def read_text(self, path: str) -> str: ...

    def write_text(self, path: str, content: str) -> None: ...

    def exists(self, path: str) -> bool: ...

    def list_dir(self, path: str) -> tuple[str, ...]: ...

    def remove(self, path: str) -> None: ...


@final
@dataclass(frozen=True, slots=True)
class FilesystemSnapshot:
    """Immutable capture of an :class:`InMemoryFilesystem`'s contents."""

    files: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, str]:
        return dict(self.files)


def _normalize(path: str) -> str:
    if not path:
        raise FilesystemError("path must be non-empty")
    posix = PurePosixPath(path)
    normalized = posix.as_posix() if posix.is_absolute() else "/" + posix.as_posix()
    return normalized.rstrip("/") or "/"


@final
class InMemoryFilesystem:
    """Thread-safe in-memory :class:`Filesystem` implementation.

    Implements :class:`weakincentives.core.Snapshotable` so it can take
    part in transactional tool rollback.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.RLock()
        self._files: dict[str, str] = {}

    def read_text(self, path: str) -> str:
        normalized = _normalize(path)
        with self._lock:
            try:
                return self._files[normalized]
            except KeyError as error:
                raise FilesystemError(f"missing file: {normalized}") from error

    def write_text(self, path: str, content: str) -> None:
        normalized = _normalize(path)
        with self._lock:
            self._files[normalized] = content

    def exists(self, path: str) -> bool:
        normalized = _normalize(path)
        with self._lock:
            if normalized in self._files:
                return True
            prefix = normalized.rstrip("/") + "/"
            return any(key.startswith(prefix) for key in self._files)

    def list_dir(self, path: str) -> tuple[str, ...]:
        normalized = _normalize(path)
        prefix = "/" if normalized == "/" else normalized + "/"
        with self._lock:
            children: set[str] = set()
            for key in self._files:
                if not key.startswith(prefix):
                    continue
                remainder = key[len(prefix) :]
                children.add(remainder.split("/", 1)[0])
        return tuple(sorted(children))

    def remove(self, path: str) -> None:
        normalized = _normalize(path)
        with self._lock:
            try:
                del self._files[normalized]
            except KeyError as error:
                raise FilesystemError(
                    f"cannot remove missing file: {normalized}"
                ) from error

    def walk(self) -> Iterator[tuple[str, str]]:
        with self._lock:
            yield from sorted(self._files.items())

    # Snapshotable -----------------------------------------------------

    def snapshot(self) -> FilesystemSnapshot:
        with self._lock:
            return FilesystemSnapshot(files=tuple(sorted(self._files.items())))

    def restore(self, state: FilesystemSnapshot) -> None:
        with self._lock:
            self._files = dict(state.files)

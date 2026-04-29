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

"""Workspace: a filesystem plus its description.

A :class:`Workspace` bundles a :class:`Filesystem` with high-level
helpers tools commonly need: a root path, a digest of the tree's
contents, and a marker for read-only mode. It implements
:class:`Snapshotable` so it participates in transactional rollback.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import final

from weakincentives.filesystem import Filesystem, FilesystemSnapshot

__all__ = [
    "Workspace",
    "WorkspaceDigest",
]


@final
@dataclass(frozen=True, slots=True)
class WorkspaceDigest:
    """Stable summary of a workspace's contents."""

    file_count: int
    total_bytes: int
    sha256: str

    @classmethod
    def of(cls, files: Iterable[tuple[str, str]]) -> WorkspaceDigest:
        materialised = list(files)
        hasher = hashlib.sha256()
        for path, body in sorted(materialised):
            hasher.update(path.encode("utf-8"))
            hasher.update(b"\x00")
            hasher.update(body.encode("utf-8"))
            hasher.update(b"\x01")
        total_bytes = sum(len(body.encode("utf-8")) for _, body in materialised)
        return cls(
            file_count=len(materialised),
            total_bytes=total_bytes,
            sha256=hasher.hexdigest(),
        )


@final
class Workspace:
    """High-level wrapper around a :class:`Filesystem`.

    ``read_only`` simply blocks writes at the workspace level; the
    underlying filesystem is unaffected. Use a fresh workspace per
    request to keep digests stable across retries.
    """

    def __init__(
        self,
        filesystem: Filesystem,
        *,
        name: str = "workspace",
        read_only: bool = False,
    ) -> None:
        super().__init__()
        self._filesystem = filesystem
        self._name = name
        self._read_only = read_only
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def filesystem(self) -> Filesystem:
        return self._filesystem

    @property
    def read_only(self) -> bool:
        return self._read_only

    def read_text(self, path: str) -> str:
        return self._filesystem.read_text(path)

    def write_text(self, path: str, content: str) -> None:
        if self._read_only:
            raise PermissionError(f"workspace {self._name!r} is read-only")
        self._filesystem.write_text(path, content)

    def exists(self, path: str) -> bool:
        return self._filesystem.exists(path)

    def list_dir(self, path: str) -> tuple[str, ...]:
        return self._filesystem.list_dir(path)

    def remove(self, path: str) -> None:
        if self._read_only:
            raise PermissionError(f"workspace {self._name!r} is read-only")
        self._filesystem.remove(path)

    def digest(self) -> WorkspaceDigest:
        snapshot = self._snapshot_filesystem()
        return WorkspaceDigest.of(snapshot.files)

    # ------------------------------------------------------------------
    # Snapshotable -- delegate to the filesystem
    # ------------------------------------------------------------------

    def snapshot(self) -> FilesystemSnapshot:
        with self._lock:
            return self._snapshot_filesystem()

    def restore(self, state: FilesystemSnapshot) -> None:
        with self._lock:
            target = self._filesystem
            restorable = getattr(target, "restore", None)
            if not callable(restorable):
                raise TypeError(
                    f"{type(target).__name__} does not implement Snapshotable"
                )
            _ = restorable(state)

    def _snapshot_filesystem(self) -> FilesystemSnapshot:
        snap = getattr(self._filesystem, "snapshot", None)
        if not callable(snap):
            raise TypeError(f"{type(self._filesystem).__name__} cannot snapshot")
        result = snap()
        if not isinstance(result, FilesystemSnapshot):  # pragma: no cover
            raise TypeError("filesystem.snapshot() must return FilesystemSnapshot")
        return result

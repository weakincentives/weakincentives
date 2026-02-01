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

"""Protocol for state containers that support snapshot and restore."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Snapshotable[SnapshotT](Protocol):
    """Protocol for state containers that support snapshot and restore.

    This protocol defines the interface for objects that can capture their
    current state as an immutable snapshot and restore from a previously
    captured snapshot. Implementations include Session and SnapshotableFilesystem.

    Type parameter ``SnapshotT`` represents the snapshot type returned by
    ``snapshot()`` and accepted by ``restore()``. For example:
    - Session implements Snapshotable[Snapshot]
    - InMemoryFilesystem implements Snapshotable[FilesystemSnapshot]

    Example usage::

        def checkpoint_and_restore(resource: Snapshotable[MySnapshot]) -> None:
            snapshot = resource.snapshot(tag="checkpoint")
            # ... perform operations ...
            resource.restore(snapshot)

    """

    def snapshot(self, *, tag: str | None = None) -> SnapshotT:
        """Capture current state as an immutable snapshot.

        Args:
            tag: Optional human-readable label for the snapshot.
                 Used for debugging and logging purposes.

        Returns:
            An immutable snapshot of the current state that can be
            stored and later passed to ``restore()``.
        """
        ...

    def restore(self, snapshot: SnapshotT) -> None:
        """Restore state from a previously captured snapshot.

        Args:
            snapshot: A snapshot previously returned by ``snapshot()``.

        Raises:
            SnapshotRestoreError: The snapshot is incompatible with
                the current state structure or the restore operation failed.
        """
        ...


__all__ = ["Snapshotable"]

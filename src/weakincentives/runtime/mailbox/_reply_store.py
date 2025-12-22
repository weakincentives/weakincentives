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

"""In-memory implementation of ReplyStore."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from threading import RLock

from ._types import ReplyEntry, ReplyState


class InMemoryReplyStore[T]:
    """In-memory implementation of ReplyStore with thread-safe access.

    Uses a dictionary with RLock synchronization for thread-safe operations.
    Entries are automatically expired based on TTL but not automatically
    cleaned up - use cleanup_expired() for garbage collection.
    """

    __slots__ = ("_entries", "_lock")

    def __init__(self) -> None:
        """Initialize an empty reply store."""
        super().__init__()
        self._entries: dict[str, ReplyEntry[T]] = {}
        self._lock = RLock()

    def create(self, entry_id: str, *, ttl: float) -> bool:
        """Create a new pending reply entry.

        Args:
            entry_id: Unique identifier for the reply.
            ttl: Time-to-live in seconds before the reply expires.

        Returns:
            True if created, False if entry_id already exists.
        """
        with self._lock:
            if entry_id in self._entries:
                return False

            now = datetime.now(UTC)
            self._entries[entry_id] = ReplyEntry(
                id=entry_id,
                value=None,
                state=ReplyState.PENDING,
                created_at=now,
                expires_at=now + timedelta(seconds=ttl),
            )
            return True

    def resolve(self, entry_id: str, value: T) -> bool:
        """Resolve a pending reply with a value.

        Args:
            entry_id: The reply identifier.
            value: The value to resolve with.

        Returns:
            True if resolved, False if not pending.
        """
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                return False

            # Check if expired
            now = datetime.now(UTC)
            if entry.state == ReplyState.PENDING and now >= entry.expires_at:
                self._entries[entry_id] = ReplyEntry(
                    id=entry.id,
                    value=entry.value,
                    state=ReplyState.EXPIRED,
                    created_at=entry.created_at,
                    expires_at=entry.expires_at,
                    resolved_at=entry.resolved_at,
                )
                return False

            if entry.state != ReplyState.PENDING:
                return False

            self._entries[entry_id] = ReplyEntry(
                id=entry.id,
                value=value,
                state=ReplyState.RESOLVED,
                created_at=entry.created_at,
                expires_at=entry.expires_at,
                resolved_at=now,
            )
            return True

    def cancel(self, entry_id: str) -> bool:
        """Cancel a pending reply.

        Args:
            entry_id: The reply identifier.

        Returns:
            True if cancelled, False if not pending.
        """
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None or entry.state != ReplyState.PENDING:
                return False

            # Check if expired
            now = datetime.now(UTC)
            if now >= entry.expires_at:
                self._entries[entry_id] = ReplyEntry(
                    id=entry.id,
                    value=entry.value,
                    state=ReplyState.EXPIRED,
                    created_at=entry.created_at,
                    expires_at=entry.expires_at,
                    resolved_at=entry.resolved_at,
                )
                return False

            self._entries[entry_id] = ReplyEntry(
                id=entry.id,
                value=entry.value,
                state=ReplyState.CANCELLED,
                created_at=entry.created_at,
                expires_at=entry.expires_at,
                resolved_at=entry.resolved_at,
            )
            return True

    def get(self, entry_id: str) -> ReplyEntry[T] | None:
        """Get a reply entry without consuming it.

        Args:
            entry_id: The reply identifier.

        Returns:
            The entry if found, None otherwise.
        """
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                return None

            # Update expired state if needed
            if entry.state == ReplyState.PENDING:
                now = datetime.now(UTC)
                if now >= entry.expires_at:
                    expired_entry = ReplyEntry(
                        id=entry.id,
                        value=entry.value,
                        state=ReplyState.EXPIRED,
                        created_at=entry.created_at,
                        expires_at=entry.expires_at,
                        resolved_at=entry.resolved_at,
                    )
                    self._entries[entry_id] = expired_entry
                    return expired_entry

            return entry

    def consume(self, entry_id: str) -> ReplyEntry[T] | None:
        """Atomically get and delete a reply entry.

        Args:
            entry_id: The reply identifier.

        Returns:
            The entry if found, None otherwise.
        """
        with self._lock:
            entry = self._entries.pop(entry_id, None)
            if entry is None:
                return None

            # Return with updated expired state if needed
            if entry.state == ReplyState.PENDING:
                now = datetime.now(UTC)
                if now >= entry.expires_at:
                    return ReplyEntry(
                        id=entry.id,
                        value=entry.value,
                        state=ReplyState.EXPIRED,
                        created_at=entry.created_at,
                        expires_at=entry.expires_at,
                        resolved_at=entry.resolved_at,
                    )

            return entry

    def delete(self, entry_id: str) -> bool:
        """Delete a reply entry.

        Args:
            entry_id: The reply identifier.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            if entry_id in self._entries:
                del self._entries[entry_id]
                return True
            return False

    def scan_expired(self, *, limit: int = 100) -> Sequence[str]:
        """Scan for expired reply entries.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            Sequence of expired reply IDs.
        """
        with self._lock:
            now = datetime.now(UTC)
            expired: list[str] = []
            for eid, entry in self._entries.items():
                if len(expired) >= limit:
                    break
                if entry.state == ReplyState.EXPIRED or (
                    entry.state == ReplyState.PENDING and now >= entry.expires_at
                ):
                    expired.append(eid)
            return expired

    def cleanup_expired(self, *, limit: int = 100) -> int:
        """Delete expired reply entries.

        Args:
            limit: Maximum number of entries to clean up.

        Returns:
            Number of entries deleted.
        """
        expired_ids = self.scan_expired(limit=limit)
        deleted = 0
        with self._lock:
            for eid in expired_ids:
                if self.delete(eid):
                    deleted += 1
        return deleted


__all__ = ["InMemoryReplyStore"]

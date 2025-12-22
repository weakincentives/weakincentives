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

"""Reply and ReplyChannel implementations."""

from __future__ import annotations

from threading import Condition, RLock
from typing import TYPE_CHECKING

from ._errors import (
    ReplyAlreadySentError,
    ReplyCancelledError,
    ReplyTimeoutError,
)
from ._types import ReplyEntry, ReplyState

if TYPE_CHECKING:
    from ._reply_store import InMemoryReplyStore


class InMemoryReply[T]:
    """In-memory implementation of Reply backed by InMemoryReplyStore.

    Uses condition variables for efficient blocking in wait().
    """

    __slots__ = ("_condition", "_id", "_store")

    def __init__(self, reply_id: str, store: InMemoryReplyStore[T]) -> None:
        """Initialize a reply handle.

        Args:
            reply_id: The reply identifier.
            store: The backing reply store.
        """
        super().__init__()
        self._id = reply_id
        self._store = store
        self._condition = Condition(RLock())

    @property
    def id(self) -> str:
        """Return the reply identifier."""
        return self._id

    def _check_entry_state(self, entry: ReplyEntry[T]) -> T:
        """Check entry state and return value or raise appropriate error.

        This method is only called when entry is not None and not PENDING.

        Args:
            entry: The reply entry to check (must not be None or PENDING).

        Returns:
            The reply value if resolved.

        Raises:
            ReplyTimeoutError: If entry is expired.
            ReplyCancelledError: If entry is cancelled.
        """
        if entry.state == ReplyState.RESOLVED:
            return entry.value  # type: ignore[return-value]
        if entry.state == ReplyState.CANCELLED:
            raise ReplyCancelledError(f"Reply {self._id} was cancelled")
        # EXPIRED state
        raise ReplyTimeoutError(f"Reply {self._id} expired")

    def wait(self, *, timeout: float | None = None) -> T:
        """Block until the reply is ready or timeout expires.

        Args:
            timeout: Maximum time to wait in seconds. None means wait forever.

        Returns:
            The reply value.

        Raises:
            ReplyTimeoutError: If timeout expires before reply is ready.
            ReplyCancelledError: If the reply was cancelled.
        """
        with self._condition:
            # Check initial state
            entry = self._store.get(self._id)
            if entry is not None and entry.state != ReplyState.PENDING:
                return self._check_entry_state(entry)

            if entry is None:
                raise ReplyTimeoutError(f"Reply {self._id} not found")

            # Wait for notification
            while True:
                notified = self._condition.wait(timeout=timeout)
                entry = self._store.get(self._id)

                if entry is not None and entry.state != ReplyState.PENDING:
                    return self._check_entry_state(entry)

                if entry is None or not notified:
                    raise ReplyTimeoutError(f"Timeout waiting for reply {self._id}")

    def poll(self) -> T | None:
        """Non-blocking check for reply readiness.

        Returns:
            The reply value if ready, None otherwise.
        """
        entry = self._store.get(self._id)
        if entry is not None and entry.state == ReplyState.RESOLVED:
            return entry.value
        return None

    def is_ready(self) -> bool:
        """Return True if the reply value is available."""
        entry = self._store.get(self._id)
        return entry is not None and entry.state == ReplyState.RESOLVED

    def is_cancelled(self) -> bool:
        """Return True if the reply was cancelled."""
        entry = self._store.get(self._id)
        return entry is not None and entry.state == ReplyState.CANCELLED

    def cancel(self) -> bool:
        """Cancel the reply.

        Returns:
            True if successfully cancelled, False if already resolved.
        """
        result = self._store.cancel(self._id)
        if result:
            with self._condition:
                self._condition.notify_all()
        return result

    def notify(self) -> None:
        """Notify waiting threads that the reply state has changed.

        This is called by InMemoryReplyChannel when a reply is sent.
        """
        with self._condition:
            self._condition.notify_all()


class InMemoryReplyChannel[T]:
    """In-memory implementation of ReplyChannel.

    Write-once channel that resolves a reply in the backing store.
    """

    __slots__ = ("_id", "_reply", "_sent", "_store")

    def __init__(
        self, reply_id: str, store: InMemoryReplyStore[T], reply: InMemoryReply[T]
    ) -> None:
        """Initialize a reply channel.

        Args:
            reply_id: The reply identifier.
            store: The backing reply store.
            reply: The associated reply handle for notification.
        """
        super().__init__()
        self._id = reply_id
        self._store = store
        self._reply = reply
        self._sent = False

    def send(self, value: T) -> None:
        """Send a reply value through the channel.

        Args:
            value: The reply value to send.

        Raises:
            ReplyAlreadySentError: If a reply was already sent.
        """
        if self._sent:
            raise ReplyAlreadySentError(f"Reply {self._id} was already sent")

        if not self._store.resolve(self._id, value):
            # Could be expired or cancelled
            entry = self._store.get(self._id)
            if entry is not None and entry.state == ReplyState.CANCELLED:
                # Treat cancelled as already sent (can't send to cancelled)
                self._sent = True
                raise ReplyAlreadySentError(
                    f"Reply {self._id} was cancelled, cannot send"
                )
            # For expired or not found, still mark as sent to prevent retries
            self._sent = True
            raise ReplyAlreadySentError(f"Reply {self._id} is no longer pending")

        self._sent = True
        self._reply.notify()

    def is_open(self) -> bool:
        """Return True if the channel can still accept a reply."""
        if self._sent:
            return False
        entry = self._store.get(self._id)
        return entry is not None and entry.state == ReplyState.PENDING


__all__ = ["InMemoryReply", "InMemoryReplyChannel"]

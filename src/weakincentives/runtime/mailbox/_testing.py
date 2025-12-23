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

"""Testing utilities for mailbox implementations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from ._types import (
    MailboxError,
    Message,
    ReceiptHandleExpiredError,
)


@dataclass(slots=True)
class NullMailbox[T]:
    """Mailbox that drops all messages on send and returns empty on receive.

    Useful for tests where you need a Mailbox interface but don't care
    about message delivery.

    Example::

        mailbox: Mailbox[Event] = NullMailbox()
        mailbox.send(Event(...))  # Silently dropped
        assert mailbox.receive() == []
        assert mailbox.approximate_count() == 0
    """

    name: str = "null"

    def send(self, body: T, *, delay_seconds: int = 0) -> str:
        """Accept and discard the message."""
        _ = (self, body, delay_seconds)
        return str(uuid4())

    def receive(
        self,
        *,
        max_messages: int = 1,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
    ) -> Sequence[Message[T]]:
        """Return empty sequence."""
        _ = (self, max_messages, visibility_timeout, wait_time_seconds)
        return []

    def purge(self) -> int:
        """Return zero (nothing to purge)."""
        _ = self
        return 0

    def approximate_count(self) -> int:
        """Return zero."""
        _ = self
        return 0


@dataclass(slots=True)
class CollectingMailbox[T]:
    """Mailbox that stores all sent messages for inspection.

    Useful for asserting what was sent without processing semantics.
    Messages are stored in the ``sent`` list in send order.

    Example::

        mailbox: CollectingMailbox[Event] = CollectingMailbox()
        mailbox.send(Event(type="a"))
        mailbox.send(Event(type="b"))

        assert len(mailbox.sent) == 2
        assert mailbox.sent[0].type == "a"
    """

    name: str = "collecting"
    sent: list[T] = field(init=False)
    """List of all sent message bodies in send order."""

    _message_ids: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.sent = []
        self._message_ids = []

    def send(self, body: T, *, delay_seconds: int = 0) -> str:
        """Store the message body for later inspection."""
        _ = delay_seconds
        msg_id = str(uuid4())
        self.sent.append(body)
        self._message_ids.append(msg_id)
        return msg_id

    def receive(
        self,
        *,
        max_messages: int = 1,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
    ) -> Sequence[Message[T]]:
        """Return empty sequence (collecting only, no consumption)."""
        _ = (self, max_messages, visibility_timeout, wait_time_seconds)
        return []

    def purge(self) -> int:
        """Clear all collected messages."""
        count = len(self.sent)
        self.sent.clear()
        self._message_ids.clear()
        return count

    def approximate_count(self) -> int:
        """Return count of collected messages."""
        return len(self.sent)


@dataclass
class FakeMailbox[T]:
    """Full in-memory implementation with controllable behavior for testing edge cases.

    Extends InMemoryMailbox semantics with methods to simulate failures
    and edge conditions.

    Example::

        mailbox: FakeMailbox[Event] = FakeMailbox()

        # Simulate receipt handle expiry
        msg = mailbox.receive()[0]
        mailbox.expire_handle(msg.receipt_handle)
        with pytest.raises(ReceiptHandleExpiredError):
            msg.acknowledge()

        # Simulate connection failure
        mailbox.set_connection_error(MailboxConnectionError("Redis down"))
        with pytest.raises(MailboxConnectionError):
            mailbox.send(Event(...))
    """

    name: str = "fake"

    _pending: list[tuple[str, T, datetime, int]] = field(init=False, repr=False)
    """List of (id, body, enqueued_at, delivery_count) tuples."""

    _invisible: dict[str, tuple[str, T, datetime, int]] = field(init=False, repr=False)
    """Map of receipt_handle -> (id, body, enqueued_at, delivery_count)."""

    _expired_handles: set[str] = field(init=False, repr=False)
    """Handles that have been manually expired."""

    _connection_error: MailboxError | None = field(default=None, repr=False)
    """Error to raise on next operation."""

    def __post_init__(self) -> None:
        self._pending = []
        self._invisible = {}
        self._expired_handles = set()

    def send(self, body: T, *, delay_seconds: int = 0) -> str:
        """Enqueue a message."""
        _ = delay_seconds
        if self._connection_error is not None:
            raise self._connection_error

        msg_id = str(uuid4())
        self._pending.append((msg_id, body, datetime.now(UTC), 0))
        return msg_id

    def receive(
        self,
        *,
        max_messages: int = 1,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
    ) -> Sequence[Message[T]]:
        """Receive messages from the queue."""
        _ = (visibility_timeout, wait_time_seconds)
        if self._connection_error is not None:
            raise self._connection_error

        messages: list[Message[T]] = []
        count = min(max_messages, len(self._pending))

        for _ in range(count):
            msg_id, body, enqueued_at, delivery_count = self._pending.pop(0)
            receipt_handle = str(uuid4())
            delivery_count += 1

            self._invisible[receipt_handle] = (
                msg_id,
                body,
                enqueued_at,
                delivery_count,
            )

            message = Message(
                id=msg_id,
                body=body,
                receipt_handle=receipt_handle,
                delivery_count=delivery_count,
                enqueued_at=enqueued_at,
                attributes={},
                _acknowledge_fn=lambda h=receipt_handle: self._acknowledge(h),
                _nack_fn=lambda t, h=receipt_handle: self._nack(h, t),
                _extend_fn=lambda t, h=receipt_handle: self._extend(h, t),
            )
            messages.append(message)

        return messages

    def _acknowledge(self, receipt_handle: str) -> None:
        """Delete message from queue."""
        if receipt_handle in self._expired_handles:
            raise ReceiptHandleExpiredError(f"Handle '{receipt_handle}' expired")
        if receipt_handle not in self._invisible:
            raise ReceiptHandleExpiredError(f"Handle '{receipt_handle}' not found")
        del self._invisible[receipt_handle]

    def _nack(self, receipt_handle: str, visibility_timeout: int) -> None:
        """Return message to queue."""
        _ = visibility_timeout
        if receipt_handle in self._expired_handles:
            raise ReceiptHandleExpiredError(f"Handle '{receipt_handle}' expired")
        if receipt_handle not in self._invisible:
            raise ReceiptHandleExpiredError(f"Handle '{receipt_handle}' not found")

        msg_id, body, enqueued_at, delivery_count = self._invisible.pop(receipt_handle)
        self._pending.append((msg_id, body, enqueued_at, delivery_count))

    def _extend(self, receipt_handle: str, timeout: int) -> None:
        """Extend visibility timeout (no-op for fake, just validates handle)."""
        _ = timeout
        if receipt_handle in self._expired_handles:
            raise ReceiptHandleExpiredError(f"Handle '{receipt_handle}' expired")
        if receipt_handle not in self._invisible:
            raise ReceiptHandleExpiredError(f"Handle '{receipt_handle}' not found")

    def purge(self) -> int:
        """Delete all messages."""
        if self._connection_error is not None:
            raise self._connection_error

        count = len(self._pending) + len(self._invisible)
        self._pending.clear()
        self._invisible.clear()
        self._expired_handles.clear()
        return count

    def approximate_count(self) -> int:
        """Return message count."""
        if self._connection_error is not None:
            raise self._connection_error
        return len(self._pending) + len(self._invisible)

    # Test control methods

    def expire_handle(self, receipt_handle: str) -> None:
        """Mark a receipt handle as expired for testing.

        Args:
            receipt_handle: The handle to expire.
        """
        self._expired_handles.add(receipt_handle)

    def set_connection_error(self, error: MailboxError | None) -> None:
        """Set an error to raise on the next operation.

        Args:
            error: Error to raise, or None to clear.
        """
        self._connection_error = error

    def clear_connection_error(self) -> None:
        """Clear any pending connection error."""
        self._connection_error = None

    def inject_message(
        self, body: T, *, msg_id: str | None = None, delivery_count: int = 0
    ) -> str:
        """Inject a message directly into the pending queue.

        Useful for setting up test scenarios without going through send().

        Args:
            body: Message body.
            msg_id: Optional specific message ID.
            delivery_count: Initial delivery count.

        Returns:
            The message ID.
        """
        msg_id = msg_id or str(uuid4())
        self._pending.append((msg_id, body, datetime.now(UTC), delivery_count))
        return msg_id


__all__ = [
    "CollectingMailbox",
    "FakeMailbox",
    "NullMailbox",
]

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
from datetime import datetime
from uuid import uuid4

from weakincentives.runtime.clock import Clock, SystemClock

from ._types import (
    Mailbox,
    MailboxError,
    Message,
    ReceiptHandleExpiredError,
    validate_visibility_timeout,
    validate_wait_time,
)


@dataclass(slots=True)
class NullMailbox[T, R]:
    """Mailbox that drops all messages on send and returns empty on receive.

    Useful for tests where you need a Mailbox interface but don't care
    about message delivery.

    Type parameters:
        T: Message body type.
        R: Reply type (None if no replies expected).

    Example::

        mailbox: Mailbox[Event, None] = NullMailbox()
        mailbox.send(Event(...))  # Silently dropped
        assert mailbox.receive() == []
        assert mailbox.approximate_count() == 0
    """

    name: str = "null"
    """Queue name for identification."""

    _closed: bool = field(default=False, repr=False)

    @property
    def closed(self) -> bool:
        """Return True if mailbox has been closed."""
        return self._closed

    def send(self, body: T, *, reply_to: Mailbox[R, None] | None = None) -> str:
        """Accept and discard the message."""
        _ = (self, body, reply_to)
        return str(uuid4())

    def receive(
        self,
        *,
        max_messages: int = 1,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
    ) -> Sequence[Message[T, R]]:
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

    def close(self) -> None:
        """Mark as closed."""
        self._closed = True


@dataclass(slots=True)
class CollectingMailbox[T, R]:
    """Mailbox that stores all sent messages for inspection.

    Useful for asserting what was sent without processing semantics.
    Messages are stored in the ``sent`` list in send order.

    Type parameters:
        T: Message body type.
        R: Reply type (None if no replies expected).

    Example::

        mailbox: CollectingMailbox[Event, None] = CollectingMailbox()
        mailbox.send(Event(type="a"))
        mailbox.send(Event(type="b"))

        assert len(mailbox.sent) == 2
        assert mailbox.sent[0].type == "a"
    """

    name: str = "collecting"
    """Queue name for identification."""

    sent: list[T] = field(init=False)
    """List of all sent message bodies in send order."""

    _message_ids: list[str] = field(init=False, repr=False)
    _reply_tos: list[Mailbox[R, None] | None] = field(init=False, repr=False)
    _closed: bool = field(default=False, repr=False, init=False)

    def __post_init__(self) -> None:
        self.sent = []
        self._message_ids = []
        self._reply_tos = []

    @property
    def closed(self) -> bool:
        """Return True if mailbox has been closed."""
        return self._closed

    def send(self, body: T, *, reply_to: Mailbox[R, None] | None = None) -> str:
        """Store the message body for later inspection."""
        msg_id = str(uuid4())
        self.sent.append(body)
        self._message_ids.append(msg_id)
        self._reply_tos.append(reply_to)
        return msg_id

    def receive(
        self,
        *,
        max_messages: int = 1,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
    ) -> Sequence[Message[T, R]]:
        """Return empty sequence (collecting only, no consumption)."""
        _ = (self, max_messages, visibility_timeout, wait_time_seconds)
        return []

    def purge(self) -> int:
        """Clear all collected messages."""
        count = len(self.sent)
        self.sent.clear()
        self._message_ids.clear()
        self._reply_tos.clear()
        return count

    def approximate_count(self) -> int:
        """Return count of collected messages."""
        return len(self.sent)

    def close(self) -> None:
        """Mark as closed."""
        self._closed = True


@dataclass
class FakeMailbox[T, R]:
    """Full in-memory implementation with controllable behavior for testing edge cases.

    Extends InMemoryMailbox semantics with methods to simulate failures
    and edge conditions. Uses direct mailbox references for reply routing.

    Type parameters:
        T: Message body type.
        R: Reply type (None if no replies expected).

    Example::

        mailbox: FakeMailbox[Event, Result] = FakeMailbox()
        responses: CollectingMailbox[Result, None] = CollectingMailbox()

        # Send with reply_to mailbox
        mailbox.send(Event(...), reply_to=responses)

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
    """Queue name for identification."""

    clock: Clock = field(default_factory=SystemClock)
    """Clock for generating timestamps."""

    _pending: list[tuple[str, T, datetime, int, Mailbox[R, None] | None]] = field(
        init=False, repr=False
    )
    """List of (id, body, enqueued_at, delivery_count, reply_to) tuples."""

    _invisible: dict[str, tuple[str, T, datetime, int, Mailbox[R, None] | None]] = (
        field(init=False, repr=False)
    )
    """Map of receipt_handle -> (id, body, enqueued_at, delivery_count, reply_to)."""

    _expired_handles: set[str] = field(init=False, repr=False)
    """Handles that have been manually expired."""

    _connection_error: MailboxError | None = field(default=None, repr=False)
    """Error to raise on next operation."""

    _closed: bool = field(default=False, repr=False, init=False)
    """Whether the mailbox has been closed."""

    def __post_init__(self) -> None:
        self._pending = []
        self._invisible = {}
        self._expired_handles = set()

    @property
    def closed(self) -> bool:
        """Return True if mailbox has been closed."""
        return self._closed

    def send(self, body: T, *, reply_to: Mailbox[R, None] | None = None) -> str:
        """Enqueue a message.

        Args:
            body: Message payload.
            reply_to: Mailbox instance for receiving replies.
        """
        if self._connection_error is not None:
            raise self._connection_error

        msg_id = str(uuid4())
        self._pending.append((msg_id, body, self.clock.now(), 0, reply_to))
        return msg_id

    def receive(
        self,
        *,
        max_messages: int = 1,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
    ) -> Sequence[Message[T, R]]:
        """Receive messages from the queue."""
        validate_visibility_timeout(visibility_timeout)
        validate_wait_time(wait_time_seconds)
        if self._connection_error is not None:
            raise self._connection_error

        messages: list[Message[T, R]] = []
        count = min(max_messages, len(self._pending))

        for _ in range(count):
            msg_id, body, enqueued_at, delivery_count, reply_to = self._pending.pop(0)
            receipt_handle = str(uuid4())
            delivery_count += 1

            self._invisible[receipt_handle] = (
                msg_id,
                body,
                enqueued_at,
                delivery_count,
                reply_to,
            )

            message = Message[T, R](
                id=msg_id,
                body=body,
                receipt_handle=receipt_handle,
                delivery_count=delivery_count,
                enqueued_at=enqueued_at,
                reply_to=reply_to,
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

        msg_id, body, enqueued_at, delivery_count, reply_to = self._invisible.pop(
            receipt_handle
        )
        self._pending.append((msg_id, body, enqueued_at, delivery_count, reply_to))

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
        self,
        body: T,
        *,
        msg_id: str | None = None,
        delivery_count: int = 0,
        reply_to: Mailbox[R, None] | None = None,
    ) -> str:
        """Inject a message directly into the pending queue.

        Useful for setting up test scenarios without going through send().

        Args:
            body: Message body.
            msg_id: Optional specific message ID.
            delivery_count: Initial delivery count.
            reply_to: Optional reply_to mailbox.

        Returns:
            The message ID.
        """
        msg_id = msg_id or str(uuid4())
        self._pending.append((msg_id, body, self.clock.now(), delivery_count, reply_to))
        return msg_id

    def close(self) -> None:
        """Mark as closed."""
        self._closed = True


__all__ = [
    "CollectingMailbox",
    "FakeMailbox",
    "NullMailbox",
]

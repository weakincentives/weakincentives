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

"""In-memory mailbox implementation."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import Condition, RLock
from typing import TYPE_CHECKING
from uuid import uuid4

from ._errors import MailboxFullError
from ._message import InMemoryMessage
from ._reply import InMemoryReply, InMemoryReplyChannel
from ._reply_store import InMemoryReplyStore
from ._types import MessageData

if TYPE_CHECKING:
    from ._protocols import Reply


@dataclass(slots=True)
class _QueuedMessage[T, R]:
    """Internal representation of a queued message."""

    id: str
    body: T
    enqueued_at: datetime
    visible_at: datetime
    delivery_count: int
    attributes: dict[str, str]
    reply_channel: InMemoryReplyChannel[R] | None


@dataclass(slots=True)
class _InFlightMessage[T, R]:
    """Internal representation of an in-flight message."""

    queued: _QueuedMessage[T, R]
    receipt_handle: str
    invisible_until: datetime


class InMemoryMailbox[T, R]:
    """In-memory implementation of Mailbox with thread-safe access.

    Uses a deque for the message queue with RLock synchronization.
    Condition variables enable efficient long-polling in receive().

    A background reaper is NOT included - visibility timeout expiration
    is checked lazily during receive() operations.
    """

    __slots__ = (
        "_condition",
        "_in_flight",
        "_lock",
        "_max_size",
        "_queue",
        "_reply_store",
    )

    def __init__(
        self,
        *,
        reply_store: InMemoryReplyStore[R] | None = None,
        max_size: int | None = None,
    ) -> None:
        """Initialize an empty mailbox.

        Args:
            reply_store: Optional reply store for request-reply patterns.
                If not provided, a new InMemoryReplyStore is created.
            max_size: Maximum number of messages. None means unlimited.
        """
        super().__init__()
        self._queue: deque[_QueuedMessage[T, R]] = deque()
        self._in_flight: dict[str, _InFlightMessage[T, R]] = {}
        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._reply_store: InMemoryReplyStore[R] = (
            reply_store if reply_store is not None else InMemoryReplyStore()
        )
        self._max_size = max_size

    def send(self, body: T, *, delay_seconds: int = 0) -> str:
        """Send a message to the mailbox.

        Args:
            body: The message body.
            delay_seconds: Delay before message becomes visible.

        Returns:
            The message ID.

        Raises:
            MailboxFullError: If the mailbox cannot accept more messages.
        """
        message_id = str(uuid4())
        now = datetime.now(UTC)

        with self._condition:
            if self._max_size is not None:
                total = len(self._queue) + len(self._in_flight)
                if total >= self._max_size:
                    raise MailboxFullError(
                        f"Mailbox is full (max_size={self._max_size})"
                    )

            queued: _QueuedMessage[T, R] = _QueuedMessage(
                id=message_id,
                body=body,
                enqueued_at=now,
                visible_at=now + timedelta(seconds=delay_seconds),
                delivery_count=0,
                attributes={},
                reply_channel=None,
            )
            self._queue.append(queued)
            self._condition.notify_all()

        return message_id

    def send_expecting_reply(self, body: T, *, reply_timeout: float = 300) -> Reply[R]:
        """Send a message and return a handle for awaiting the reply.

        Args:
            body: The message body.
            reply_timeout: Timeout in seconds for the reply.

        Returns:
            A Reply handle for awaiting the response.

        Raises:
            MailboxFullError: If the mailbox cannot accept more messages.
        """
        message_id = str(uuid4())
        reply_id = str(uuid4())
        now = datetime.now(UTC)

        # Create reply entry first
        _ = self._reply_store.create(reply_id, ttl=reply_timeout)
        reply = InMemoryReply(reply_id, self._reply_store)
        reply_channel = InMemoryReplyChannel(reply_id, self._reply_store, reply)

        with self._condition:
            if self._max_size is not None:
                total = len(self._queue) + len(self._in_flight)
                if total >= self._max_size:
                    # Clean up the reply entry we created
                    _ = self._reply_store.delete(reply_id)
                    raise MailboxFullError(
                        f"Mailbox is full (max_size={self._max_size})"
                    )

            queued = _QueuedMessage(
                id=message_id,
                body=body,
                enqueued_at=now,
                visible_at=now,
                delivery_count=0,
                attributes={"reply_id": reply_id},
                reply_channel=reply_channel,
            )
            self._queue.append(queued)
            self._condition.notify_all()

        return reply

    def receive(
        self,
        *,
        max_messages: int = 1,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
    ) -> Sequence[InMemoryMessage[T, R]]:
        """Receive messages from the mailbox.

        Args:
            max_messages: Maximum number of messages to receive.
            visibility_timeout: Seconds before unreceived messages become visible.
            wait_time_seconds: Long-poll timeout (0 = immediate return).

        Returns:
            Sequence of received messages.
        """
        deadline = (
            datetime.now(UTC) + timedelta(seconds=wait_time_seconds)
            if wait_time_seconds > 0
            else None
        )

        with self._condition:
            while True:
                # First, return expired in-flight messages to the queue
                self._return_expired_to_queue()

                # Find visible messages
                now = datetime.now(UTC)
                messages: list[InMemoryMessage[T, R]] = []

                # Build a new queue with messages we don't take
                remaining: deque[_QueuedMessage[T, R]] = deque()
                for queued in self._queue:
                    if len(messages) >= max_messages:
                        remaining.append(queued)
                        continue

                    if queued.visible_at <= now:
                        # Take this message
                        receipt_handle = str(uuid4())
                        queued.delivery_count += 1

                        in_flight = _InFlightMessage(
                            queued=queued,
                            receipt_handle=receipt_handle,
                            invisible_until=now + timedelta(seconds=visibility_timeout),
                        )
                        self._in_flight[receipt_handle] = in_flight

                        data: MessageData[T, R] = MessageData(
                            id=queued.id,
                            body=queued.body,
                            receipt_handle=receipt_handle,
                            delivery_count=queued.delivery_count,
                            enqueued_at=queued.enqueued_at,
                            attributes=queued.attributes,
                            reply_channel=queued.reply_channel,
                        )
                        message: InMemoryMessage[T, R] = InMemoryMessage(
                            data,
                            ack_callback=self._acknowledge,
                            nack_callback=self._nack,
                            extend_callback=self._extend_visibility,
                        )
                        messages.append(message)
                    else:
                        remaining.append(queued)

                self._queue = remaining

                if messages:
                    return messages

                # No messages available
                if deadline is None:
                    return []

                # Long poll - wait for messages or timeout
                wait_seconds = (deadline - datetime.now(UTC)).total_seconds()
                if wait_seconds <= 0:
                    return []

                _ = self._condition.wait(timeout=wait_seconds)

    def purge(self) -> int:
        """Remove all messages from the mailbox.

        Returns:
            Number of messages removed.
        """
        with self._condition:
            count = len(self._queue) + len(self._in_flight)
            self._queue.clear()
            self._in_flight.clear()
            return count

    def approximate_count(self) -> int:
        """Return approximate number of messages in the mailbox.

        Returns:
            Approximate message count.
        """
        with self._lock:
            return len(self._queue) + len(self._in_flight)

    def _return_expired_to_queue(self) -> None:
        """Return expired in-flight messages to the queue.

        Must be called with _lock held.
        """
        now = datetime.now(UTC)
        expired_handles: list[str] = []

        for handle, in_flight in self._in_flight.items():
            if in_flight.invisible_until <= now:
                expired_handles.append(handle)

        for handle in expired_handles:
            in_flight = self._in_flight.pop(handle)
            # Reset visibility and add back to queue
            in_flight.queued.visible_at = now
            self._queue.append(in_flight.queued)

    def _acknowledge(self, receipt_handle: str) -> bool:
        """Acknowledge a message and remove it from in-flight.

        Args:
            receipt_handle: The receipt handle from receive().

        Returns:
            True if acknowledged, False if handle not found/expired.
        """
        with self._lock:
            if receipt_handle in self._in_flight:
                del self._in_flight[receipt_handle]
                return True
            return False

    def _nack(self, receipt_handle: str, visibility_timeout: int) -> bool:
        """Return a message to the queue for redelivery.

        Args:
            receipt_handle: The receipt handle from receive().
            visibility_timeout: Seconds before message becomes visible.

        Returns:
            True if nacked, False if handle not found/expired.
        """
        with self._condition:
            in_flight = self._in_flight.pop(receipt_handle, None)
            if in_flight is None:
                return False

            now = datetime.now(UTC)
            in_flight.queued.visible_at = now + timedelta(seconds=visibility_timeout)
            self._queue.append(in_flight.queued)
            self._condition.notify_all()
            return True

    def _extend_visibility(self, receipt_handle: str, timeout: int) -> bool:
        """Extend the visibility timeout for an in-flight message.

        Args:
            receipt_handle: The receipt handle from receive().
            timeout: New visibility timeout in seconds.

        Returns:
            True if extended, False if handle not found/expired.
        """
        with self._lock:
            in_flight = self._in_flight.get(receipt_handle)
            if in_flight is None:
                return False

            now = datetime.now(UTC)
            in_flight.invisible_until = now + timedelta(seconds=timeout)
            return True


__all__ = ["InMemoryMailbox"]

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

"""Thread-safe in-memory mailbox implementation."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from ...clock import SYSTEM_CLOCK, MonotonicClock
from ...threading import BackgroundWorker, Gate, SystemGate
from ..logging import StructuredLogger, get_logger
from ._types import (
    Mailbox,
    MailboxFullError,
    Message,
    ReceiptHandleExpiredError,
    validate_visibility_timeout,
    validate_wait_time,
)

logger: StructuredLogger = get_logger(__name__, context={"component": "mailbox"})


@dataclass
class _InFlightMessage[T, R]:
    """Internal tracking for messages that have been received but not acknowledged."""

    id: str
    body: T
    enqueued_at: datetime
    delivery_count: int
    receipt_handle: str
    reply_to: Mailbox[R, None] | None
    expires_at: float  # time.monotonic() value


class InMemoryMailboxFactory[R]:
    """Factory that creates InMemoryMailbox instances for reply routing.

    Used with RedisMailbox to auto-create in-memory response mailboxes
    when reply_to identifiers need to be resolved. This is useful for
    testing Redis mailbox flows where responses should be collected
    in-memory.

    Example::

        factory = InMemoryMailboxFactory()
        resolver = CompositeResolver(registry={}, factory=factory)
        requests = RedisMailbox(name="requests", client=redis_client, reply_resolver=resolver)
    """

    __slots__ = ("_registry",)

    _registry: dict[str, Mailbox[R, None]]
    """Shared registry for caching created mailboxes."""

    def __init__(self, registry: dict[str, Mailbox[R, None]] | None = None) -> None:
        """Initialize factory with optional shared registry for caching.

        Args:
            registry: Shared registry dict. If provided, created mailboxes are
                cached here to avoid creating duplicates.
        """
        super().__init__()
        self._registry = registry if registry is not None else {}

    def create(self, identifier: str) -> Mailbox[R, None]:
        """Create or return cached InMemoryMailbox for the given identifier.

        Args:
            identifier: Queue name for the mailbox.

        Returns:
            An InMemoryMailbox instance. Cached if registry was provided.
        """
        if identifier in self._registry:  # pragma: no cover - defensive
            return self._registry[identifier]
        mailbox: Mailbox[R, None] = InMemoryMailbox(name=identifier)
        self._registry[identifier] = mailbox
        return mailbox


@dataclass(slots=True)
class InMemoryMailbox[T, R]:
    """Thread-safe in-memory mailbox implementation.

    Messages are stored in memory and lost on process restart.
    Useful for testing and single-process development.

    Characteristics:
    - Thread-safe via Lock
    - FIFO ordering guaranteed
    - Exact message counts
    - No persistence
    - Direct mailbox references for reply routing

    Type parameters:
        T: Message body type.
        R: Reply type (None if no replies expected).

    Example::

        # Create mailboxes
        requests: InMemoryMailbox[Request, Result] = InMemoryMailbox(name="requests")
        responses: InMemoryMailbox[Result, None] = InMemoryMailbox(name="responses")

        # Client: send request with response mailbox
        requests.send(Request(data="hello"), reply_to=responses)

        # Worker: reply routes directly to the response mailbox
        for msg in requests.receive():
            msg.reply(Result(value=42))
            msg.acknowledge()

        # Client: receive response
        for msg in responses.receive():
            print(msg.body)
            msg.acknowledge()
    """

    name: str = "default"
    """Queue name for identification."""

    max_size: int | None = None
    """Maximum queue capacity. None for unlimited."""

    clock: MonotonicClock = field(default=SYSTEM_CLOCK, repr=False)
    """Clock for time-based operations. Defaults to system clock.

    Tests can inject TestClock for deterministic timing without sleeps."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _condition: threading.Condition = field(init=False, repr=False)
    _pending: deque[_InFlightMessage[T, R]] = field(init=False, repr=False)
    _invisible: dict[str, _InFlightMessage[T, R]] = field(init=False, repr=False)
    _delivery_counts: dict[str, int] = field(init=False, repr=False)
    _reaper_worker: BackgroundWorker | None = field(
        default=None, repr=False, init=False
    )
    _closed: bool = field(default=False, repr=False, init=False)
    _stop_reaper: Gate = field(
        default_factory=SystemGate, repr=False, init=False
    )

    def __post_init__(self) -> None:
        self._condition = threading.Condition(self._lock)
        self._pending = deque()
        self._invisible = {}
        self._delivery_counts = {}
        self._start_reaper()

    def _start_reaper(self) -> None:
        """Start background thread to requeue expired messages."""
        self._reaper_worker = BackgroundWorker(
            target=self._reaper_loop,
            daemon=True,
            name=f"mailbox-reaper-{self.name}",
        )
        self._reaper_worker.start()

    def _reaper_loop(self) -> None:
        """Background loop that checks for expired visibility timeouts."""
        while not self._stop_reaper.wait(timeout=0.1):
            self._reap_expired()

    def _reap_expired(self) -> None:
        """Move expired messages from invisible back to pending."""
        now = self.clock.monotonic()
        with self._lock:
            expired_handles = [
                handle
                for handle, msg in self._invisible.items()
                if msg.expires_at <= now
            ]
            for handle in expired_handles:
                msg = self._invisible.pop(handle)
                self._pending.append(msg)
                self._condition.notify_all()

    def send(self, body: T, *, reply_to: Mailbox[R, None] | None = None) -> str:
        """Enqueue a message.

        Args:
            body: Message payload.
            reply_to: Mailbox instance for receiving replies. The reference is
                stored directly and passed to Message.reply_to on receive.

        Returns:
            Message ID (unique within this queue).

        Raises:
            MailboxFullError: Queue capacity exceeded.
        """
        msg_id = str(uuid4())

        with self._lock:
            total = len(self._pending) + len(self._invisible)
            if self.max_size is not None and total >= self.max_size:
                raise MailboxFullError(
                    f"Queue '{self.name}' at capacity ({self.max_size})"
                )

            in_flight = _InFlightMessage[T, R](
                id=msg_id,
                body=body,
                enqueued_at=datetime.now(UTC),
                delivery_count=0,
                receipt_handle="",
                reply_to=reply_to,
                expires_at=0.0,
            )

            self._pending.append(in_flight)
            self._condition.notify_all()

        logger.debug(
            "mailbox.send",
            event="mailbox.send",
            context={
                "mailbox": self.name,
                "message_id": msg_id,
                "reply_to": reply_to.name if reply_to else None,
                "body_type": type(body).__qualname__,
            },
        )
        return msg_id

    def receive(
        self,
        *,
        max_messages: int = 1,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
    ) -> Sequence[Message[T, R]]:
        """Receive messages from the queue.

        Args:
            max_messages: Maximum messages to receive (1-10).
            visibility_timeout: Seconds message remains invisible (0-43200).
            wait_time_seconds: Long poll duration. Zero returns immediately.

        Returns:
            Sequence of messages (may be empty). Returns empty if mailbox closed.

        Raises:
            InvalidParameterError: visibility_timeout or wait_time_seconds out of range.
        """
        validate_visibility_timeout(visibility_timeout)
        validate_wait_time(wait_time_seconds)
        max_messages = min(max(1, max_messages), 10)
        deadline = self.clock.monotonic() + wait_time_seconds

        messages: list[Message[T, R]] = []

        with self._lock:
            while len(messages) < max_messages:
                # Check closed state
                if self._closed:
                    break

                # Try to get a message
                if self._pending:
                    in_flight = self._pending.popleft()

                    # Generate new receipt handle and set expiry
                    receipt_handle = str(uuid4())
                    in_flight.receipt_handle = receipt_handle
                    in_flight.delivery_count += 1
                    in_flight.expires_at = self.clock.monotonic() + visibility_timeout

                    # Track delivery count
                    self._delivery_counts[in_flight.id] = in_flight.delivery_count

                    # Move to invisible
                    self._invisible[receipt_handle] = in_flight

                    # Create message with bound callbacks
                    message = Message[T, R](
                        id=in_flight.id,
                        body=in_flight.body,
                        receipt_handle=receipt_handle,
                        delivery_count=in_flight.delivery_count,
                        enqueued_at=in_flight.enqueued_at,
                        reply_to=in_flight.reply_to,
                        _acknowledge_fn=lambda h=receipt_handle: self._acknowledge(h),
                        _nack_fn=lambda t, h=receipt_handle: self._nack(h, t),
                        _extend_fn=lambda t, h=receipt_handle: self._extend(h, t),
                    )
                    messages.append(message)
                else:
                    # No messages available
                    if wait_time_seconds <= 0:
                        break

                    remaining = deadline - self.clock.monotonic()
                    if remaining <= 0:
                        break

                    # Wait for messages, close signal, or timeout
                    _ = self._condition.wait(timeout=remaining)

        if messages:
            logger.debug(
                "mailbox.receive",
                event="mailbox.receive",
                context={
                    "mailbox": self.name,
                    "message_count": len(messages),
                    "message_ids": [m.id for m in messages],
                },
            )
        return messages

    def _acknowledge(self, receipt_handle: str) -> None:
        """Delete message from queue."""
        with self._lock:
            if receipt_handle not in self._invisible:
                raise ReceiptHandleExpiredError(
                    f"Receipt handle '{receipt_handle}' not found or expired"
                )
            msg = self._invisible.pop(receipt_handle)
            # Clean up delivery count tracking
            _ = self._delivery_counts.pop(msg.id, None)
        logger.debug(
            "mailbox.acknowledge",
            event="mailbox.acknowledge",
            context={
                "mailbox": self.name,
                "message_id": msg.id,
                "receipt_handle": receipt_handle,
            },
        )

    def _nack(self, receipt_handle: str, visibility_timeout: int) -> None:
        """Return message to queue."""
        with self._lock:
            if receipt_handle not in self._invisible:
                raise ReceiptHandleExpiredError(
                    f"Receipt handle '{receipt_handle}' not found or expired"
                )
            msg = self._invisible.pop(receipt_handle)

            if visibility_timeout <= 0:
                # Immediately visible
                self._pending.append(msg)
                self._condition.notify_all()
            else:
                # Delay visibility
                new_handle = str(uuid4())
                msg.receipt_handle = new_handle
                msg.expires_at = self.clock.monotonic() + visibility_timeout
                self._invisible[new_handle] = msg

    def _extend(self, receipt_handle: str, timeout: int) -> None:
        """Extend visibility timeout."""
        with self._lock:
            if receipt_handle not in self._invisible:
                raise ReceiptHandleExpiredError(
                    f"Receipt handle '{receipt_handle}' not found or expired"
                )
            msg = self._invisible[receipt_handle]
            msg.expires_at = self.clock.monotonic() + timeout

    def purge(self) -> int:
        """Delete all messages from the queue.

        Returns:
            Count of messages deleted.
        """
        with self._lock:
            count = len(self._pending) + len(self._invisible)
            self._pending.clear()
            self._invisible.clear()
            self._delivery_counts.clear()
            return count

    def approximate_count(self) -> int:
        """Return exact number of messages in the queue.

        For InMemoryMailbox, this count is exact, not approximate.
        """
        with self._lock:
            return len(self._pending) + len(self._invisible)

    def close(self) -> None:
        """Stop the reaper thread and wake any blocked receivers."""
        # Set closed flag and wake all waiters
        with self._lock:
            self._closed = True
            self._condition.notify_all()

        # Stop reaper worker
        self._stop_reaper.set()
        if self._reaper_worker is not None:  # pragma: no branch
            self._reaper_worker.stop(timeout=1.0)

    @property
    def closed(self) -> bool:
        """Return True if mailbox has been closed."""
        return self._closed


__all__ = ["InMemoryMailbox", "InMemoryMailboxFactory"]

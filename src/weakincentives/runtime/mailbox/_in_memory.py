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
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from ._resolver import CompositeResolver, MailboxResolutionError
from ._types import (
    MailboxFullError,
    Message,
    ReceiptHandleExpiredError,
    ReplyNotAvailableError,
)

if TYPE_CHECKING:
    from ._resolver import MailboxResolver
    from ._types import Mailbox


@dataclass
class _InFlightMessage[T]:
    """Internal tracking for messages that have been received but not acknowledged."""

    id: str
    body: T
    enqueued_at: datetime
    delivery_count: int
    receipt_handle: str
    reply_to: str | None
    expires_at: float  # time.monotonic() value


class InMemoryMailboxFactory[R]:
    """Factory that creates InMemoryMailbox instances for reply routing.

    Used internally by InMemoryMailbox to auto-create response mailboxes
    when reply_to identifiers are used.

    Example::

        factory = InMemoryMailboxFactory()
        resolver = CompositeResolver(registry={}, factory=factory)
        requests = InMemoryMailbox(name="requests", reply_resolver=resolver)
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
        mailbox: Mailbox[R, None] = InMemoryMailbox(
            name=identifier,
            reply_resolver=None,  # Reply mailboxes don't need nested resolution
        )
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
    - Auto-creates response mailboxes for reply_to identifiers

    Type parameters:
        T: Message body type.
        R: Reply type (None if no replies expected).

    Example::

        # Create request mailbox - response mailboxes auto-created
        requests: InMemoryMailbox[Request, Result] = InMemoryMailbox(name="requests")

        # Client: send request, retrieve auto-created response mailbox
        requests.send(Request(data="hello"), reply_to="my-responses")
        responses = requests.resolver.resolve("my-responses")

        # Worker: reply routes automatically
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

    reply_resolver: MailboxResolver[R] | None = None
    """Resolver for reply_to identifiers. If None, a default resolver is created
    that auto-registers mailboxes for reply_to identifiers on send()."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _condition: threading.Condition = field(init=False, repr=False)
    _pending: deque[_InFlightMessage[T]] = field(init=False, repr=False)
    _invisible: dict[str, _InFlightMessage[T]] = field(init=False, repr=False)
    _delivery_counts: dict[str, int] = field(init=False, repr=False)
    _reply_registry: dict[str, Mailbox[R, None]] = field(init=False, repr=False)
    _reaper_thread: threading.Thread | None = field(
        default=None, repr=False, init=False
    )
    _closed: bool = field(default=False, repr=False, init=False)
    _stop_reaper: threading.Event = field(
        default_factory=threading.Event, repr=False, init=False
    )

    def __post_init__(self) -> None:
        self._condition = threading.Condition(self._lock)
        self._pending = deque()
        self._invisible = {}
        self._delivery_counts = {}
        self._reply_registry = {}
        self._start_reaper()
        # Set up default resolver if none provided
        if self.reply_resolver is None:
            factory: InMemoryMailboxFactory[R] = InMemoryMailboxFactory(
                registry=self._reply_registry
            )
            object.__setattr__(
                self,
                "reply_resolver",
                CompositeResolver(registry=self._reply_registry, factory=factory),
            )

    @property
    def resolver(self) -> MailboxResolver[R]:
        """Return the reply resolver for accessing auto-created mailboxes.

        Use this to retrieve mailboxes created for reply_to identifiers::

            requests.send(msg, reply_to="responses")
            responses = requests.resolver.resolve("responses")
        """
        if self.reply_resolver is None:  # pragma: no cover
            raise RuntimeError("No resolver configured")
        return self.reply_resolver

    def _start_reaper(self) -> None:
        """Start background thread to requeue expired messages."""
        self._reaper_thread = threading.Thread(
            target=self._reaper_loop,
            daemon=True,
            name=f"mailbox-reaper-{self.name}",
        )
        self._reaper_thread.start()

    def _reaper_loop(self) -> None:
        """Background loop that checks for expired visibility timeouts."""
        while not self._stop_reaper.wait(timeout=0.1):
            self._reap_expired()

    def _reap_expired(self) -> None:
        """Move expired messages from invisible back to pending."""
        now = time.monotonic()
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

    def send(self, body: T, *, reply_to: str | None = None) -> str:
        """Enqueue a message.

        If reply_to is provided and a default resolver is configured, this
        automatically creates and registers a response mailbox that can be
        retrieved via ``resolver.resolve(reply_to)``.

        Args:
            body: Message payload.
            reply_to: Identifier for response mailbox. Workers resolve this
                via Message.reply(). If using the default resolver, a mailbox
                is auto-created for this identifier.

        Returns:
            Message ID (unique within this queue).

        Raises:
            MailboxFullError: Queue capacity exceeded.
        """
        # Auto-register reply_to mailbox if using default resolver
        if (
            reply_to is not None
            and reply_to not in self._reply_registry
            and self.reply_resolver is not None
        ):
            # Resolve to trigger factory creation and caching
            _ = self.reply_resolver.resolve_optional(reply_to)

        msg_id = str(uuid4())

        with self._lock:
            total = len(self._pending) + len(self._invisible)
            if self.max_size is not None and total >= self.max_size:
                raise MailboxFullError(
                    f"Queue '{self.name}' at capacity ({self.max_size})"
                )

            in_flight = _InFlightMessage(
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
        """
        max_messages = min(max(1, max_messages), 10)
        deadline = time.monotonic() + wait_time_seconds

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
                    in_flight.expires_at = time.monotonic() + visibility_timeout

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
                        _reply_fn=lambda body, rt=in_flight.reply_to: self._reply(
                            rt, body
                        ),
                    )
                    messages.append(message)
                else:
                    # No messages available
                    if wait_time_seconds <= 0:
                        break

                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break

                    # Wait for messages, close signal, or timeout
                    _ = self._condition.wait(timeout=remaining)

        return messages

    def _reply(self, reply_to: str | None, body: R) -> str:
        """Send a reply to the reply_to mailbox."""
        if reply_to is None:  # pragma: no cover
            raise ReplyNotAvailableError("No reply_to specified")
        if (
            self.reply_resolver is None
        ):  # pragma: no cover - always set in __post_init__
            raise ReplyNotAvailableError("No reply_resolver configured")
        try:
            mailbox = self.reply_resolver.resolve(reply_to)
        except MailboxResolutionError as e:
            raise ReplyNotAvailableError(f"Cannot resolve reply_to '{reply_to}'") from e
        return mailbox.send(body)

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
                msg.expires_at = time.monotonic() + visibility_timeout
                self._invisible[new_handle] = msg

    def _extend(self, receipt_handle: str, timeout: int) -> None:
        """Extend visibility timeout."""
        with self._lock:
            if receipt_handle not in self._invisible:
                raise ReceiptHandleExpiredError(
                    f"Receipt handle '{receipt_handle}' not found or expired"
                )
            msg = self._invisible[receipt_handle]
            msg.expires_at = time.monotonic() + timeout

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

        # Stop reaper thread
        self._stop_reaper.set()
        if self._reaper_thread is not None:  # pragma: no branch
            self._reaper_thread.join(timeout=1.0)

    @property
    def closed(self) -> bool:
        """Return True if mailbox has been closed."""
        return self._closed


__all__ = ["InMemoryMailbox", "InMemoryMailboxFactory"]

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

"""In-memory message queue with visibility timeouts and a DLQ.

The :class:`Mailbox` protocol mirrors the SQS / Redis pattern: ``put``
enqueues a message; ``poll`` leases one for processing; ``ack`` deletes
it; ``nack`` returns it to the queue for retry; messages that exceed
their delivery cap land in a :class:`DeadLetterQueue`. The ``InMemory``
implementation is enough to run the full :class:`MailboxWorker` story
in tests, with no third-party dependency.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, final, runtime_checkable

from weakincentives.clock import SYSTEM_CLOCK, MonotonicClock
from weakincentives.logging import get_logger

__all__ = [
    "DeadLetterQueue",
    "InMemoryMailbox",
    "Lease",
    "Mailbox",
    "MailboxError",
    "MailboxWorker",
    "Message",
]

_LOG = get_logger("weakincentives.mailbox")


class MailboxError(Exception):
    """Raised on mailbox protocol violations."""


@final
@dataclass(frozen=True, slots=True)
class Message:
    """A queued payload plus its delivery metadata."""

    id: str
    body: object
    enqueued_at: float
    delivery_count: int = 0


@final
@dataclass(frozen=True, slots=True)
class Lease:
    """A token returned by :meth:`Mailbox.poll`."""

    id: str
    message: Message
    expires_at: float


@runtime_checkable
class Mailbox(Protocol):
    """The minimal interface every mailbox backend implements."""

    def put(self, body: object) -> str: ...

    def poll(self, *, lease_seconds: float) -> Lease | None: ...

    def ack(self, lease_id: str) -> None: ...

    def nack(self, lease_id: str) -> None: ...

    def __len__(self) -> int: ...


@dataclass(slots=True)
class DeadLetterQueue:
    """Stores poison messages that exceeded their retry budget."""

    messages: list[Message] = field(default_factory=lambda: [])
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def add(self, message: Message) -> None:
        with self._lock:
            self.messages.append(message)

    def drain(self) -> tuple[Message, ...]:
        with self._lock:
            drained = tuple(self.messages)
            self.messages.clear()
            return drained

    def __len__(self) -> int:
        with self._lock:
            return len(self.messages)


@final
class InMemoryMailbox:
    """Single-process mailbox with leases, retries, and an optional DLQ."""

    def __init__(
        self,
        *,
        max_deliveries: int = 5,
        dead_letter_queue: DeadLetterQueue | None = None,
        clock: MonotonicClock = SYSTEM_CLOCK,
    ) -> None:
        super().__init__()
        if max_deliveries < 1:
            raise MailboxError("max_deliveries must be >= 1")
        self._max_deliveries = max_deliveries
        self._dlq = dead_letter_queue
        self._clock = clock
        self._lock = threading.RLock()
        self._queue: list[Message] = []
        self._leased: dict[str, tuple[Lease, Message]] = {}

    def put(self, body: object) -> str:
        message = Message(
            id=uuid.uuid4().hex,
            body=body,
            enqueued_at=self._clock.monotonic(),
        )
        with self._lock:
            self._queue.append(message)
        return message.id

    def poll(self, *, lease_seconds: float) -> Lease | None:
        with self._lock:
            self._reclaim_expired_leases()
            if not self._queue:
                return None
            message = self._queue.pop(0)
            updated = Message(
                id=message.id,
                body=message.body,
                enqueued_at=message.enqueued_at,
                delivery_count=message.delivery_count + 1,
            )
            lease = Lease(
                id=uuid.uuid4().hex,
                message=updated,
                expires_at=self._clock.monotonic() + lease_seconds,
            )
            self._leased[lease.id] = (lease, updated)
            return lease

    def ack(self, lease_id: str) -> None:
        with self._lock:
            if lease_id not in self._leased:
                raise MailboxError(f"unknown lease: {lease_id}")
            del self._leased[lease_id]

    def nack(self, lease_id: str) -> None:
        with self._lock:
            entry = self._leased.pop(lease_id, None)
            if entry is None:
                raise MailboxError(f"unknown lease: {lease_id}")
            _, message = entry
            self._return_or_dlq(message)

    def __len__(self) -> int:
        with self._lock:
            return len(self._queue) + len(self._leased)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reclaim_expired_leases(self) -> None:
        now = self._clock.monotonic()
        expired: list[str] = []
        for lease_id, (lease, _message) in self._leased.items():
            if lease.expires_at <= now:
                expired.append(lease_id)
        for lease_id in expired:
            _, message = self._leased.pop(lease_id)
            self._return_or_dlq(message)

    def _return_or_dlq(self, message: Message) -> None:
        if message.delivery_count >= self._max_deliveries:
            if self._dlq is not None:
                self._dlq.add(message)
            else:
                _LOG.warning(
                    "mailbox.dropped_poison_message",
                    message_id=message.id,
                    delivery_count=message.delivery_count,
                )
            return
        self._queue.append(message)


# =============================================================================
# Worker
# =============================================================================


Handler = Callable[[Message], None]


@dataclass(slots=True)
class MailboxWorker:
    """Drive a :class:`Mailbox` with a handler in a single thread.

    Use :meth:`run_until` to step through messages until a stop event
    fires; the worker handles ack/nack automatically. Errors propagate
    after :meth:`stop` is called via the returned exception list, so
    callers can decide whether to escalate.
    """

    mailbox: Mailbox
    handler: Handler
    lease_seconds: float = 30.0
    poll_idle_seconds: float = 0.05
    failures: list[BaseException] = field(default_factory=lambda: [])

    def step(self) -> bool:
        """Process up to one message. Returns True if it processed one."""
        lease = self.mailbox.poll(lease_seconds=self.lease_seconds)
        if lease is None:
            return False
        try:
            self.handler(lease.message)
        except BaseException as error:
            self.failures.append(error)
            self.mailbox.nack(lease.id)
            return True
        self.mailbox.ack(lease.id)
        return True

    def run_until(self, stop: threading.Event) -> None:
        while not stop.is_set():
            processed = self.step()
            if not processed:
                _ = stop.wait(timeout=self.poll_idle_seconds)

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

"""Core mailbox types, protocols, and errors."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from ...errors import WinkError


class MailboxError(WinkError):
    """Base class for mailbox-related errors."""


class ReceiptHandleExpiredError(MailboxError):
    """Receipt handle no longer valid.

    Raised when acknowledge, nack, or extend_visibility is called
    with an expired receipt handle. This occurs when:
    - Visibility timeout expired before operation
    - Message was already acknowledged
    - Message was redelivered (new receipt handle issued)
    """


class MailboxFullError(MailboxError):
    """Queue capacity exceeded.

    SQS: 120,000 in-flight messages (standard) or 20,000 (FIFO)
    Redis: Depends on maxmemory configuration
    InMemory: Configurable max_size parameter
    """


class SerializationError(MailboxError):
    """Message body cannot be serialized or deserialized.

    The body must be JSON-serializable via the standard serde module.
    Complex objects should use FrozenDataclass for automatic serialization.
    """


class MailboxConnectionError(MailboxError):
    """Cannot connect to backend.

    Redis: Connection refused, timeout, authentication failure
    SQS: AWS credentials invalid, network unreachable
    """


@dataclass(frozen=True, slots=True)
class Message[T]:
    """A received message with delivery metadata and lifecycle methods.

    Messages are immutable snapshots of delivery state. The lifecycle methods
    (acknowledge, nack, extend_visibility) operate on the message via the
    bound callback references stored at receive time.
    """

    id: str
    """Unique message identifier within the queue."""

    body: T
    """Deserialized message payload."""

    receipt_handle: str
    """Opaque handle for this specific delivery. Changes on each delivery.
    Required for acknowledge/nack/extend operations."""

    delivery_count: int
    """Number of times this message has been received. First delivery = 1.
    Use for dead-letter logic or debugging redelivery issues."""

    enqueued_at: datetime
    """Timestamp when message was originally sent (UTC)."""

    attributes: Mapping[str, str] = field(default_factory=lambda: dict[str, str]())
    """Backend-specific message attributes (e.g., SQS MessageAttributes)."""

    _acknowledge_fn: Callable[[], None] = field(
        default=lambda: None, repr=False, compare=False
    )
    """Internal callback for acknowledge operation."""

    _nack_fn: Callable[[int], None] = field(
        default=lambda _: None, repr=False, compare=False
    )
    """Internal callback for nack operation."""

    _extend_fn: Callable[[int], None] = field(
        default=lambda _: None, repr=False, compare=False
    )
    """Internal callback for extend_visibility operation."""

    def acknowledge(self) -> None:
        """Delete the message from the queue.

        Call after successfully processing the message. The receipt handle
        must still be valid (message not timed out or already acknowledged).

        Raises:
            ReceiptHandleExpiredError: Handle no longer valid.
        """
        self._acknowledge_fn()

    def nack(self, *, visibility_timeout: int = 0) -> None:
        """Return message to queue immediately or after delay.

        Use when processing fails and the message should be retried.
        Setting ``visibility_timeout=0`` makes the message immediately
        visible to other consumers.

        Args:
            visibility_timeout: Seconds before message becomes visible again.

        Raises:
            ReceiptHandleExpiredError: Handle no longer valid.
        """
        self._nack_fn(visibility_timeout)

    def extend_visibility(self, timeout: int) -> None:
        """Extend the visibility timeout for long-running processing.

        Call periodically during long processing to prevent timeout.
        The new timeout is relative to now, not the original receive time.

        Args:
            timeout: New visibility timeout in seconds from now.

        Raises:
            ReceiptHandleExpiredError: Handle no longer valid.
        """
        self._extend_fn(timeout)


@runtime_checkable
class Mailbox[T](Protocol):
    """Point-to-point message queue with visibility timeout semantics.

    Mailbox provides SQS-compatible semantics for durable, at-least-once
    message delivery. Messages are invisible to other consumers after receive
    until acknowledged, nacked, or visibility times out.
    """

    @abstractmethod
    def send(self, body: T, *, delay_seconds: int = 0) -> str:
        """Enqueue a message, optionally delaying visibility.

        Args:
            body: Message payload (must be serializable).
            delay_seconds: Seconds before message becomes visible (0-900).

        Returns:
            Message ID (unique within this queue).

        Raises:
            MailboxFullError: Queue capacity exceeded (backend-specific).
            SerializationError: Body cannot be serialized.
        """
        ...

    @abstractmethod
    def receive(
        self,
        *,
        max_messages: int = 1,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
    ) -> Sequence[Message[T]]:
        """Receive messages from the queue.

        Received messages become invisible to other consumers for
        ``visibility_timeout`` seconds. Messages must be acknowledged
        before timeout expires or they return to the queue.

        Args:
            max_messages: Maximum messages to receive (1-10).
            visibility_timeout: Seconds message remains invisible (0-43200).
            wait_time_seconds: Long poll duration (0-20). Zero returns
                immediately; positive values block until messages arrive
                or timeout expires.

        Returns:
            Sequence of messages (may be empty if no messages available
            or long poll timed out).
        """
        ...

    @abstractmethod
    def purge(self) -> int:
        """Delete all messages from the queue.

        Returns:
            Approximate count of messages deleted.

        Note:
            SQS enforces 60-second cooldown between purges.
            Redis has no cooldown.
        """
        ...

    @abstractmethod
    def approximate_count(self) -> int:
        """Return approximate number of messages in the queue.

        The count includes both visible and invisible messages.
        Value is eventually consistent (SQS ~1 minute lag, Redis exact).
        """
        ...


__all__ = [
    "Mailbox",
    "MailboxConnectionError",
    "MailboxError",
    "MailboxFullError",
    "Message",
    "ReceiptHandleExpiredError",
    "SerializationError",
]

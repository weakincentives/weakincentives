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
from collections.abc import Callable, Sequence
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


class ReplyNotAvailableError(MailboxError):
    """Cannot resolve reply_to destination.

    Raised when Message.reply() is called but:
    - No reply_to was specified in the original send()
    - The reply_to identifier cannot be resolved by the resolver
    """


class MessageFinalizedError(MailboxError):
    """Message has already been acknowledged or nacked.

    Raised when reply() is called after acknowledge() or nack().
    Once a message is finalized, no further replies are allowed.
    """


class InvalidParameterError(MailboxError):
    """Invalid parameter value provided.

    Raised when timeout parameters are out of valid range:
    - visibility_timeout must be 0-43200 (0 to 12 hours)
    - wait_time_seconds must be non-negative
    """


# SQS-compatible bounds for visibility timeout (0 to 12 hours in seconds)
MAX_VISIBILITY_TIMEOUT = 43200


def validate_visibility_timeout(
    value: int, param_name: str = "visibility_timeout"
) -> None:
    """Validate visibility_timeout is within valid range [0, 43200].

    Args:
        value: The visibility timeout value in seconds.
        param_name: Parameter name for error messages.

    Raises:
        InvalidParameterError: If value is out of range.
    """
    if value < 0:
        raise InvalidParameterError(f"{param_name} must be non-negative, got {value}")
    if value > MAX_VISIBILITY_TIMEOUT:
        raise InvalidParameterError(
            f"{param_name} must be at most {MAX_VISIBILITY_TIMEOUT} seconds, got {value}"
        )


def validate_wait_time(value: int) -> None:
    """Validate wait_time_seconds is non-negative.

    Args:
        value: The wait time value in seconds.

    Raises:
        InvalidParameterError: If value is negative.
    """
    if value < 0:
        raise InvalidParameterError(
            f"wait_time_seconds must be non-negative, got {value}"
        )


@dataclass(slots=True)
class Message[T, R]:
    """A received message with delivery metadata and lifecycle methods.

    Messages are snapshots of delivery state. The lifecycle methods
    (acknowledge, nack, extend_visibility, reply) operate on the message via
    bound callback references stored at receive time.

    Type parameters:
        T: Message body type.
        R: Reply type (None if no replies expected).
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

    reply_to: str | None = None
    """Identifier for response mailbox. Workers resolve this via reply()."""

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

    _reply_fn: Callable[[R], str] = field(
        default=lambda _: "", repr=False, compare=False
    )
    """Internal callback for reply operation."""

    _finalized: bool = field(default=False, repr=False, compare=False)
    """True if message has been acknowledged or nacked."""

    def reply(self, body: R) -> str:
        """Send reply to reply_to destination.

        Multiple replies are allowed before finalization (acknowledge/nack).
        The reply_to identifier is resolved internally via the mailbox's
        reply_resolver.

        Args:
            body: Reply payload to send.

        Returns:
            Message ID of the sent reply.

        Raises:
            MessageFinalizedError: Message already acknowledged or nacked.
            ReplyNotAvailableError: No reply_to specified or cannot resolve.
        """
        if self._finalized:
            raise MessageFinalizedError(
                f"Message '{self.id}' already finalized; cannot reply"
            )
        if self.reply_to is None:
            raise ReplyNotAvailableError(f"Message '{self.id}' has no reply_to")
        return self._reply_fn(body)

    def acknowledge(self) -> None:
        """Delete the message from the queue.

        Call after successfully processing the message. The receipt handle
        must still be valid (message not timed out or already acknowledged).
        Finalizes the message - no further replies allowed.

        Raises:
            ReceiptHandleExpiredError: Handle no longer valid.
        """
        self._acknowledge_fn()
        self._finalized = True

    def nack(self, *, visibility_timeout: int = 0) -> None:
        """Return message to queue immediately or after delay.

        Use when processing fails and the message should be retried.
        Setting ``visibility_timeout=0`` makes the message immediately
        visible to other consumers. Finalizes the message - no further
        replies allowed.

        Args:
            visibility_timeout: Seconds before message becomes visible again (0-43200).

        Raises:
            InvalidParameterError: visibility_timeout out of range.
            ReceiptHandleExpiredError: Handle no longer valid.
        """
        validate_visibility_timeout(visibility_timeout)
        self._nack_fn(visibility_timeout)
        self._finalized = True

    def extend_visibility(self, timeout: int) -> None:
        """Extend the visibility timeout for long-running processing.

        Call periodically during long processing to prevent timeout.
        The new timeout is relative to now, not the original receive time.

        Args:
            timeout: New visibility timeout in seconds from now (0-43200).

        Raises:
            InvalidParameterError: timeout out of range.
            ReceiptHandleExpiredError: Handle no longer valid.
        """
        validate_visibility_timeout(timeout, "timeout")
        self._extend_fn(timeout)

    @property
    def is_finalized(self) -> bool:
        """True if message has been acknowledged or nacked."""
        return self._finalized


@runtime_checkable
class Mailbox[T, R](Protocol):
    """Point-to-point message queue with visibility timeout semantics.

    Mailbox provides SQS-compatible semantics for durable, at-least-once
    message delivery. Messages are invisible to other consumers after receive
    until acknowledged, nacked, or visibility times out.

    Type parameters:
        T: Message body type.
        R: Reply type (None if no replies expected).
    """

    @property
    @abstractmethod
    def closed(self) -> bool:
        """Return True if the mailbox has been closed."""
        ...

    @abstractmethod
    def send(self, body: T, *, reply_to: str | None = None) -> str:
        """Enqueue a message.

        Args:
            body: Message payload (must be serializable).
            reply_to: Identifier for response mailbox. Workers resolve this
                via Message.reply().

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
    ) -> Sequence[Message[T, R]]:
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
            Sequence of messages (may be empty if no messages available,
            long poll timed out, or mailbox closed).
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

    @abstractmethod
    def close(self) -> None:
        """Close the mailbox and release resources.

        After closing:
        - receive() returns empty immediately
        - send() behavior is implementation-defined
        """
        ...


__all__ = [
    "InvalidParameterError",
    "Mailbox",
    "MailboxConnectionError",
    "MailboxError",
    "MailboxFullError",
    "Message",
    "MessageFinalizedError",
    "ReceiptHandleExpiredError",
    "ReplyNotAvailableError",
    "SerializationError",
]

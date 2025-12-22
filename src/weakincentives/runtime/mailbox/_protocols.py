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

"""Protocol definitions for mailbox abstractions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ._types import ReplyEntry


@runtime_checkable
class Reply[T](Protocol):
    """Caller's future-like handle for awaiting responses."""

    @property
    def id(self) -> str:
        """Return the reply identifier."""
        ...

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
        ...

    def poll(self) -> T | None:
        """Non-blocking check for reply readiness.

        Returns:
            The reply value if ready, None otherwise.
        """
        ...

    def is_ready(self) -> bool:
        """Return True if the reply value is available."""
        ...

    def is_cancelled(self) -> bool:
        """Return True if the reply was cancelled."""
        ...

    def cancel(self) -> bool:
        """Cancel the reply.

        Returns:
            True if successfully cancelled, False if already resolved.
        """
        ...


@runtime_checkable
class ReplyChannel[T](Protocol):
    """Consumer's write-once channel for sending responses."""

    def send(self, value: T) -> None:
        """Send a reply value through the channel.

        Args:
            value: The reply value to send.

        Raises:
            ReplyAlreadySentError: If a reply was already sent.
        """
        ...

    def is_open(self) -> bool:
        """Return True if the channel can still accept a reply."""
        ...


@runtime_checkable
class ReplyStore[T](Protocol):
    """Backing storage for reply state with TTL and atomic consume."""

    def create(self, entry_id: str, *, ttl: float) -> bool:
        """Create a new pending reply entry.

        Args:
            entry_id: Unique identifier for the reply.
            ttl: Time-to-live in seconds before the reply expires.

        Returns:
            True if created, False if entry_id already exists.
        """
        ...

    def resolve(self, entry_id: str, value: T) -> bool:
        """Resolve a pending reply with a value.

        Args:
            entry_id: The reply identifier.
            value: The value to resolve with.

        Returns:
            True if resolved, False if not pending.
        """
        ...

    def cancel(self, entry_id: str) -> bool:
        """Cancel a pending reply.

        Args:
            entry_id: The reply identifier.

        Returns:
            True if cancelled, False if not pending.
        """
        ...

    def get(self, entry_id: str) -> ReplyEntry[T] | None:
        """Get a reply entry without consuming it.

        Args:
            entry_id: The reply identifier.

        Returns:
            The entry if found, None otherwise.
        """
        ...

    def consume(self, entry_id: str) -> ReplyEntry[T] | None:
        """Atomically get and delete a reply entry.

        Args:
            entry_id: The reply identifier.

        Returns:
            The entry if found, None otherwise.
        """
        ...

    def delete(self, entry_id: str) -> bool:
        """Delete a reply entry.

        Args:
            entry_id: The reply identifier.

        Returns:
            True if deleted, False if not found.
        """
        ...

    def scan_expired(self, *, limit: int = 100) -> Sequence[str]:
        """Scan for expired reply entries.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            Sequence of expired reply IDs.
        """
        ...

    def cleanup_expired(self, *, limit: int = 100) -> int:
        """Delete expired reply entries.

        Args:
            limit: Maximum number of entries to clean up.

        Returns:
            Number of entries deleted.
        """
        ...


@runtime_checkable
class Message[T, R](Protocol):
    """Message received from a mailbox."""

    @property
    def id(self) -> str:
        """Return the message identifier."""
        ...

    @property
    def body(self) -> T:
        """Return the message body."""
        ...

    @property
    def receipt_handle(self) -> str:
        """Return the receipt handle for acknowledgment."""
        ...

    @property
    def delivery_count(self) -> int:
        """Return the number of times this message has been delivered."""
        ...

    @property
    def enqueued_at(self) -> datetime:
        """Return when the message was enqueued."""
        ...

    @property
    def attributes(self) -> Mapping[str, str]:
        """Return message attributes."""
        ...

    @property
    def reply_channel(self) -> ReplyChannel[R] | None:
        """Return the reply channel if this message expects a reply."""
        ...

    def expects_reply(self) -> bool:
        """Return True if this message expects a reply."""
        ...

    def reply(self, value: R) -> None:
        """Send a reply and acknowledge the message.

        Writes the response to ReplyStore, then acknowledges the message.

        Args:
            value: The reply value.

        Raises:
            NoReplyChannelError: If no reply channel is available.
        """
        ...

    def acknowledge(self) -> bool:
        """Acknowledge message receipt and remove from queue.

        Returns:
            True if acknowledged, False if receipt handle expired.

        Raises:
            ReplyExpectedError: If this message expects a reply.
        """
        ...

    def nack(self, *, visibility_timeout: int = 0) -> bool:
        """Return message to queue for redelivery.

        Args:
            visibility_timeout: Seconds before message becomes visible again.

        Returns:
            True if nacked, False if receipt handle expired.
        """
        ...

    def extend_visibility(self, timeout: int) -> bool:
        """Extend the visibility timeout for this message.

        Args:
            timeout: New visibility timeout in seconds.

        Returns:
            True if extended, False if receipt handle expired.
        """
        ...


@runtime_checkable
class Mailbox[T, R](Protocol):
    """Message queue abstraction with SQS/Redis-compatible semantics."""

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
        ...

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
        ...

    def receive(
        self,
        *,
        max_messages: int = 1,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
    ) -> Sequence[Message[T, R]]:
        """Receive messages from the mailbox.

        Args:
            max_messages: Maximum number of messages to receive.
            visibility_timeout: Seconds before unreceived messages become visible.
            wait_time_seconds: Long-poll timeout (0 = immediate return).

        Returns:
            Sequence of received messages.
        """
        ...

    def purge(self) -> int:
        """Remove all messages from the mailbox.

        Returns:
            Number of messages removed.
        """
        ...

    def approximate_count(self) -> int:
        """Return approximate number of messages in the mailbox.

        Returns:
            Approximate message count.
        """
        ...


__all__ = [
    "Mailbox",
    "Message",
    "Reply",
    "ReplyChannel",
    "ReplyStore",
]

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

"""Testing utilities for mailbox operations."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

from ._errors import ReplyTimeoutError
from ._protocols import Mailbox, Message, Reply
from ._types import MessageData


class ImmediateReply[T]:
    """Reply that resolves instantly with a preset value.

    Useful for testing request-reply patterns without actual async waiting.
    """

    __slots__ = ("_cancelled", "_id", "_value")

    def __init__(self, value: T, *, reply_id: str | None = None) -> None:
        """Initialize an immediately resolved reply.

        Args:
            value: The value to return from wait() and poll().
            reply_id: Optional reply ID. Generated if not provided.
        """
        super().__init__()
        self._id = reply_id if reply_id is not None else str(uuid4())
        self._value = value
        self._cancelled = False

    @property
    def id(self) -> str:
        """Return the reply identifier."""
        return self._id

    def wait(self, *, timeout: float | None = None) -> T:
        """Return the preset value immediately.

        Args:
            timeout: Ignored (reply is always ready).

        Returns:
            The preset reply value.
        """
        del timeout
        return self._value

    def poll(self) -> T | None:
        """Return the preset value.

        Returns:
            The preset reply value.
        """
        return self._value

    def is_ready(self) -> bool:
        """Return True (always ready)."""
        return True

    def is_cancelled(self) -> bool:
        """Return the cancelled state."""
        return self._cancelled

    def cancel(self) -> bool:
        """Mark as cancelled and return True.

        Returns:
            True (always succeeds for testing).
        """
        self._cancelled = True
        return True


class NeverResolvingReply[T]:
    """Reply that never resolves.

    Useful for testing timeout behavior.
    """

    __slots__ = ("_cancelled", "_id")

    def __init__(self, *, reply_id: str | None = None) -> None:
        """Initialize a never-resolving reply.

        Args:
            reply_id: Optional reply ID. Generated if not provided.
        """
        super().__init__()
        self._id = reply_id if reply_id is not None else str(uuid4())
        self._cancelled = False

    @property
    def id(self) -> str:
        """Return the reply identifier."""
        return self._id

    def wait(self, *, timeout: float | None = None) -> T:
        """Always raise ReplyTimeoutError.

        Args:
            timeout: Ignored (always times out).

        Raises:
            ReplyTimeoutError: Always raised.
        """
        raise ReplyTimeoutError(f"Reply {self._id} never resolves")

    def poll(self) -> T | None:
        """Return None (never ready).

        Returns:
            None.
        """
        return None

    def is_ready(self) -> bool:
        """Return False (never ready)."""
        return False

    def is_cancelled(self) -> bool:
        """Return the cancelled state."""
        return self._cancelled

    def cancel(self) -> bool:
        """Mark as cancelled and return True.

        Returns:
            True (always succeeds).
        """
        self._cancelled = True
        return True


class NullMailbox[T, R]:
    """Mailbox that drops messages and returns non-resolving replies.

    Useful for testing fire-and-forget message patterns.
    """

    __slots__ = ("_messages_sent",)

    def __init__(self) -> None:
        """Initialize an empty null mailbox."""
        super().__init__()
        self._messages_sent: list[T] = []

    def send(self, body: T, *, delay_seconds: int = 0) -> str:
        """Accept the message and return a message ID (message is dropped).

        Args:
            body: The message body.
            delay_seconds: Ignored.

        Returns:
            A generated message ID.
        """
        del delay_seconds
        self._messages_sent.append(body)
        return str(uuid4())

    def send_expecting_reply(
        self, body: T, *, reply_timeout: float = 300
    ) -> NeverResolvingReply[R]:
        """Accept the message and return a non-resolving reply.

        Args:
            body: The message body.
            reply_timeout: Ignored.

        Returns:
            A NeverResolvingReply that will timeout on wait().
        """
        del reply_timeout
        self._messages_sent.append(body)
        return NeverResolvingReply()

    def receive(
        self,
        *,
        max_messages: int = 1,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
    ) -> Sequence[Message[T, R]]:
        """Return an empty sequence (no messages available).

        Args:
            max_messages: Ignored.
            visibility_timeout: Ignored.
            wait_time_seconds: Ignored.

        Returns:
            Empty sequence.
        """
        del max_messages, visibility_timeout, wait_time_seconds
        return []

    def purge(self) -> int:
        """Return 0 (nothing to purge).

        Returns:
            0.
        """
        count = len(self._messages_sent)
        self._messages_sent.clear()
        return count

    def approximate_count(self) -> int:
        """Return 0 (always empty).

        Returns:
            0.
        """
        return 0

    @property
    def messages_sent(self) -> Sequence[T]:
        """Return all messages that were sent (for test assertions)."""
        return list(self._messages_sent)


class RecordingMailbox[T, R]:
    """Mailbox that records all sent messages for inspection.

    Wraps another mailbox and records messages before delegating.
    Useful for verifying message contents in tests.
    """

    __slots__ = ("_delegate", "_received_messages", "_sent_messages")

    def __init__(self, delegate: Mailbox[T, R]) -> None:
        """Initialize a recording mailbox.

        Args:
            delegate: The underlying mailbox to delegate to.
        """
        super().__init__()
        self._delegate = delegate
        self._sent_messages: list[T] = []
        self._received_messages: list[MessageData[T, R]] = []

    def send(self, body: T, *, delay_seconds: int = 0) -> str:
        """Record and delegate the send.

        Args:
            body: The message body.
            delay_seconds: Delay before message becomes visible.

        Returns:
            The message ID from the delegate.
        """
        self._sent_messages.append(body)
        return self._delegate.send(body, delay_seconds=delay_seconds)

    def send_expecting_reply(self, body: T, *, reply_timeout: float = 300) -> Reply[R]:
        """Record and delegate the send.

        Args:
            body: The message body.
            reply_timeout: Timeout in seconds for the reply.

        Returns:
            The Reply from the delegate.
        """
        self._sent_messages.append(body)
        return self._delegate.send_expecting_reply(body, reply_timeout=reply_timeout)

    def receive(
        self,
        *,
        max_messages: int = 1,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
    ) -> Sequence[Message[T, R]]:
        """Delegate and record received messages.

        Args:
            max_messages: Maximum number of messages to receive.
            visibility_timeout: Seconds before unreceived messages become visible.
            wait_time_seconds: Long-poll timeout.

        Returns:
            Sequence of received messages.
        """
        messages = self._delegate.receive(
            max_messages=max_messages,
            visibility_timeout=visibility_timeout,
            wait_time_seconds=wait_time_seconds,
        )
        for msg in messages:
            self._received_messages.append(
                MessageData(
                    id=msg.id,
                    body=msg.body,
                    receipt_handle=msg.receipt_handle,
                    delivery_count=msg.delivery_count,
                    enqueued_at=msg.enqueued_at,
                    attributes=msg.attributes,
                    reply_channel=msg.reply_channel,
                )
            )
        return messages

    def purge(self) -> int:
        """Delegate the purge.

        Returns:
            Number of messages removed.
        """
        return self._delegate.purge()

    def approximate_count(self) -> int:
        """Delegate the count.

        Returns:
            Approximate message count.
        """
        return self._delegate.approximate_count()

    @property
    def sent_messages(self) -> Sequence[T]:
        """Return all sent message bodies."""
        return list(self._sent_messages)

    @property
    def received_messages(self) -> Sequence[MessageData[T, R]]:
        """Return all received message data."""
        return list(self._received_messages)


__all__ = [
    "ImmediateReply",
    "NeverResolvingReply",
    "NullMailbox",
    "RecordingMailbox",
]

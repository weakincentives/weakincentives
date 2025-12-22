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

"""Message implementation for mailbox operations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING

from ._errors import NoReplyChannelError, ReplyExpectedError
from ._types import MessageData

if TYPE_CHECKING:
    from ._protocols import ReplyChannel


class InMemoryMessage[T, R]:
    """In-memory message implementation with mailbox callbacks.

    This class wraps MessageData and provides methods that interact with
    the mailbox for acknowledgment and visibility management.
    """

    __slots__ = ("_ack_callback", "_data", "_extend_callback", "_nack_callback")

    def __init__(
        self,
        data: MessageData[T, R],
        *,
        ack_callback: Callable[[str], bool],
        nack_callback: Callable[[str, int], bool],
        extend_callback: Callable[[str, int], bool],
    ) -> None:
        """Initialize a message.

        Args:
            data: The underlying message data.
            ack_callback: Callback for acknowledging the message.
            nack_callback: Callback for returning message to queue.
            extend_callback: Callback for extending visibility timeout.
        """
        super().__init__()
        self._data = data
        self._ack_callback = ack_callback
        self._nack_callback = nack_callback
        self._extend_callback = extend_callback

    @property
    def id(self) -> str:
        """Return the message identifier."""
        return self._data.id

    @property
    def body(self) -> T:
        """Return the message body."""
        return self._data.body

    @property
    def receipt_handle(self) -> str:
        """Return the receipt handle for acknowledgment."""
        return self._data.receipt_handle

    @property
    def delivery_count(self) -> int:
        """Return the number of times this message has been delivered."""
        return self._data.delivery_count

    @property
    def enqueued_at(self) -> datetime:
        """Return when the message was enqueued."""
        return self._data.enqueued_at

    @property
    def attributes(self) -> Mapping[str, str]:
        """Return message attributes."""
        return self._data.attributes

    @property
    def reply_channel(self) -> ReplyChannel[R] | None:
        """Return the reply channel if this message expects a reply."""
        return self._data.reply_channel

    def expects_reply(self) -> bool:
        """Return True if this message expects a reply."""
        return self._data.reply_channel is not None

    def reply(self, value: R) -> None:
        """Send a reply and acknowledge the message.

        Writes the response to ReplyStore, then acknowledges the message.

        Args:
            value: The reply value.

        Raises:
            NoReplyChannelError: If no reply channel is available.
        """
        if self._data.reply_channel is None:
            raise NoReplyChannelError(
                f"Message {self._data.id} does not expect a reply"
            )

        # Send the reply first (this will raise if already sent)
        self._data.reply_channel.send(value)

        # Then acknowledge the message
        _ = self._ack_callback(self._data.receipt_handle)

    def acknowledge(self) -> bool:
        """Acknowledge message receipt and remove from queue.

        Returns:
            True if acknowledged, False if receipt handle expired.

        Raises:
            ReplyExpectedError: If this message expects a reply.
        """
        if self._data.reply_channel is not None:
            raise ReplyExpectedError(
                f"Message {self._data.id} expects a reply; use reply() instead"
            )

        return self._ack_callback(self._data.receipt_handle)

    def nack(self, *, visibility_timeout: int = 0) -> bool:
        """Return message to queue for redelivery.

        Args:
            visibility_timeout: Seconds before message becomes visible again.

        Returns:
            True if nacked, False if receipt handle expired.
        """
        return self._nack_callback(self._data.receipt_handle, visibility_timeout)

    def extend_visibility(self, timeout: int) -> bool:
        """Extend the visibility timeout for this message.

        Args:
            timeout: New visibility timeout in seconds.

        Returns:
            True if extended, False if receipt handle expired.
        """
        return self._extend_callback(self._data.receipt_handle, timeout)


__all__ = ["InMemoryMessage"]

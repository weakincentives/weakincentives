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

"""Message queue abstraction with SQS/Redis-compatible semantics.

This module provides a point-to-point message queue abstraction with visibility
timeout and acknowledgment semantics. Unlike the pub/sub ``Dispatcher``, Mailbox
delivers messages to a single consumer with at-least-once delivery guarantees.

Core abstractions:
    - ``Mailbox``: Protocol for sending and receiving messages
    - ``Message``: Received message with acknowledge/nack/reply methods
    - ``Reply``: Future-like handle for awaiting responses
    - ``ReplyChannel``: Write-once channel for sending responses
    - ``ReplyStore``: Backing storage for reply state

Implementations:
    - ``InMemoryMailbox``: Thread-safe in-memory implementation
    - ``InMemoryReplyStore``: Thread-safe in-memory reply store
    - ``InMemoryMessage``: Message implementation with mailbox callbacks
    - ``InMemoryReply``: Reply with condition variable blocking
    - ``InMemoryReplyChannel``: Reply channel backed by store

Testing utilities:
    - ``NullMailbox``: Drops messages, returns non-resolving replies
    - ``ImmediateReply``: Resolves instantly with preset value
    - ``NeverResolvingReply``: Never resolves (for timeout testing)
    - ``RecordingMailbox``: Records messages for test assertions

Example usage::

    from weakincentives.runtime.mailbox import InMemoryMailbox

    # Create a mailbox
    mailbox: InMemoryMailbox[str, str] = InMemoryMailbox()

    # Fire-and-forget message
    message_id = mailbox.send("hello")

    # Request-reply pattern
    reply = mailbox.send_expecting_reply("request")

    # Consumer side
    for msg in mailbox.receive(max_messages=10, visibility_timeout=30):
        if msg.expects_reply():
            msg.reply("response")
        else:
            msg.acknowledge()

    # Await reply on sender side
    result = reply.wait(timeout=60)
"""

from __future__ import annotations

from ._errors import (
    MailboxError,
    MailboxFullError,
    NoReplyChannelError,
    ReceiptHandleExpiredError,
    ReplyAlreadySentError,
    ReplyCancelledError,
    ReplyError,
    ReplyExpectedError,
    ReplyTimeoutError,
    SerializationError,
)
from ._mailbox import InMemoryMailbox
from ._message import InMemoryMessage
from ._protocols import Mailbox, Message, Reply, ReplyChannel, ReplyStore
from ._reply import InMemoryReply, InMemoryReplyChannel
from ._reply_store import InMemoryReplyStore
from ._testing import (
    ImmediateReply,
    NeverResolvingReply,
    NullMailbox,
    RecordingMailbox,
)
from ._types import MessageData, ReplyEntry, ReplyState

__all__ = [
    "ImmediateReply",
    "InMemoryMailbox",
    "InMemoryMessage",
    "InMemoryReply",
    "InMemoryReplyChannel",
    "InMemoryReplyStore",
    "Mailbox",
    "MailboxError",
    "MailboxFullError",
    "Message",
    "MessageData",
    "NeverResolvingReply",
    "NoReplyChannelError",
    "NullMailbox",
    "ReceiptHandleExpiredError",
    "RecordingMailbox",
    "Reply",
    "ReplyAlreadySentError",
    "ReplyCancelledError",
    "ReplyChannel",
    "ReplyEntry",
    "ReplyError",
    "ReplyExpectedError",
    "ReplyState",
    "ReplyStore",
    "ReplyTimeoutError",
    "SerializationError",
]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})

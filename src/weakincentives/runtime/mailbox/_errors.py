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

"""Error hierarchy for mailbox operations."""

from __future__ import annotations

from ...errors import WinkError


class MailboxError(WinkError):
    """Base class for mailbox-related errors."""


class ReceiptHandleExpiredError(MailboxError):
    """Raised when a receipt handle has expired or is invalid."""


class MailboxFullError(MailboxError):
    """Raised when the mailbox cannot accept more messages."""


class SerializationError(MailboxError):
    """Raised when message serialization or deserialization fails."""


class ReplyError(WinkError):
    """Base class for reply-related errors."""


class ReplyTimeoutError(ReplyError):
    """Raised when waiting for a reply times out."""


class ReplyCancelledError(ReplyError):
    """Raised when a reply has been cancelled."""


class ReplyAlreadySentError(ReplyError):
    """Raised when attempting to send a reply that was already sent."""


class ReplyExpectedError(ReplyError):
    """Raised when acknowledge() is called but a reply is expected."""


class NoReplyChannelError(ReplyError):
    """Raised when reply() is called but no reply channel is available."""


__all__ = [
    "MailboxError",
    "MailboxFullError",
    "NoReplyChannelError",
    "ReceiptHandleExpiredError",
    "ReplyAlreadySentError",
    "ReplyCancelledError",
    "ReplyError",
    "ReplyExpectedError",
    "ReplyTimeoutError",
    "SerializationError",
]

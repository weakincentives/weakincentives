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

"""Message queue abstraction with SQS-compatible semantics.

Mailbox provides point-to-point message delivery with visibility timeout
and explicit acknowledgment. Unlike the pub/sub Dispatcher, Mailbox
delivers messages to one consumer at a time with at-least-once delivery.

**When to use Mailbox:**

- Durable request processing that survives process restarts
- Work distribution across multiple consumers
- Cross-process communication (distributed deployments)
- Tasks requiring acknowledgment and retry on failure

**When to use Dispatcher:**

- Telemetry and observability events
- In-process event notifications
- Fire-and-forget broadcasts to multiple subscribers

Example::

    from weakincentives.runtime.mailbox import InMemoryMailbox, Message

    mailbox: InMemoryMailbox[str] = InMemoryMailbox(name="tasks")

    # Send a message
    msg_id = mailbox.send("process this")

    # Receive and process
    for msg in mailbox.receive(visibility_timeout=60):
        try:
            do_work(msg.body)
            msg.acknowledge()
        except Exception:
            msg.nack(visibility_timeout=30)  # Retry after 30s

See ``specs/MAILBOX.md`` for the complete specification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._in_memory import InMemoryMailbox
from ._testing import CollectingMailbox, FakeMailbox, NullMailbox
from ._types import (
    Mailbox,
    MailboxConnectionError,
    MailboxError,
    MailboxFullError,
    Message,
    ReceiptHandleExpiredError,
    SerializationError,
)

__all__ = [
    "CollectingMailbox",
    "FakeMailbox",
    "InMemoryMailbox",
    "Mailbox",
    "MailboxConnectionError",
    "MailboxError",
    "MailboxFullError",
    "Message",
    "NullMailbox",
    "ReceiptHandleExpiredError",
    "RedisMailbox",
    "SerializationError",
]

# RedisMailbox is only available if the redis package is installed
if TYPE_CHECKING:
    from ._redis import RedisMailbox


def __getattr__(name: str) -> object:
    if name == "RedisMailbox":
        try:
            from ._redis import RedisMailbox
        except ImportError as e:  # pragma: no cover - optional dependency
            msg = (
                "RedisMailbox requires the 'redis' package. "
                "Install with: pip install weakincentives[redis]"
            )
            raise ImportError(msg) from e
        else:
            return RedisMailbox
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )  # pragma: no cover


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})  # pragma: no cover - convenience shim

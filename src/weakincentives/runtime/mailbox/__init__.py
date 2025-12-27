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

    # Setup with reply pattern
    registry: dict[str, InMemoryMailbox] = {}
    responses: InMemoryMailbox[str, None] = InMemoryMailbox(name="responses")
    registry["responses"] = responses

    requests: InMemoryMailbox[str, str] = InMemoryMailbox(
        name="requests",
        reply_resolver=registry.get,
    )

    # Send a request with reply_to
    requests.send("process this", reply_to="responses")

    # Receive and process with reply
    for msg in requests.receive(visibility_timeout=60):
        try:
            result = do_work(msg.body)
            msg.reply_mailbox().send(result)
            msg.acknowledge()
        except ReplyMailboxUnavailableError:
            # No reply_to specified, just process
            msg.acknowledge()
        except Exception:
            msg.nack(visibility_timeout=30)  # Retry after 30s

See ``specs/MAILBOX.md`` for the complete specification.
"""

from __future__ import annotations

from ._in_memory import InMemoryMailbox
from ._testing import CollectingMailbox, FakeMailbox, NullMailbox
from ._types import (
    Mailbox,
    MailboxConnectionError,
    MailboxError,
    MailboxFullError,
    Message,
    ReceiptHandleExpiredError,
    ReplyMailboxUnavailableError,
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
    "ReplyMailboxUnavailableError",
    "SerializationError",
]

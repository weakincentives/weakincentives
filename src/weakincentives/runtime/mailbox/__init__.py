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

    from weakincentives.runtime.mailbox import (
        InMemoryMailbox,
        Message,
        RegistryResolver,
    )

    # Setup with reply routing
    responses = InMemoryMailbox(name="responses")
    resolver = RegistryResolver({"responses": responses})
    requests = InMemoryMailbox(name="requests", reply_resolver=resolver)

    # Send request with reply_to
    requests.send("process this", reply_to="responses")

    # Receive and process with reply
    for msg in requests.receive(visibility_timeout=60):
        try:
            result = do_work(msg.body)
            msg.reply(result)
            msg.acknowledge()
        except Exception:
            msg.nack(visibility_timeout=30)  # Retry after 30s

See ``specs/MAILBOX.md`` for the complete specification.
"""

from __future__ import annotations

from ._in_memory import InMemoryMailbox
from ._resolver import (
    CompositeResolver,
    MailboxFactory,
    MailboxResolutionError,
    MailboxResolver,
    RegistryResolver,
)
from ._testing import CollectingMailbox, FakeMailbox, NullMailbox
from ._types import (
    Mailbox,
    MailboxConnectionError,
    MailboxError,
    MailboxFullError,
    Message,
    MessageFinalizedError,
    ReceiptHandleExpiredError,
    ReplyNotAvailableError,
    SerializationError,
)

__all__ = [
    "CollectingMailbox",
    "CompositeResolver",
    "FakeMailbox",
    "InMemoryMailbox",
    "Mailbox",
    "MailboxConnectionError",
    "MailboxError",
    "MailboxFactory",
    "MailboxFullError",
    "MailboxResolutionError",
    "MailboxResolver",
    "Message",
    "MessageFinalizedError",
    "NullMailbox",
    "ReceiptHandleExpiredError",
    "RegistryResolver",
    "ReplyNotAvailableError",
    "SerializationError",
]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})  # pragma: no cover - convenience shim

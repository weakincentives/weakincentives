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

"""Message queue abstraction with SQS-compatible visibility timeout semantics.

This package provides point-to-point message delivery for durable request
processing. Unlike the pub/sub Dispatcher (which broadcasts events),
Mailbox delivers messages to one consumer at a time with at-least-once
delivery guarantees through visibility timeouts and explicit acknowledgment.

When to Use Mailbox vs Dispatcher
---------------------------------

**Use Mailbox for:**

- Durable request processing that survives process restarts
- Work distribution across multiple consumers
- Cross-process communication (distributed deployments)
- Tasks requiring acknowledgment and retry on failure
- Request/response patterns with reply routing

**Use Dispatcher for:**

- Telemetry and observability events
- In-process event notifications
- Fire-and-forget broadcasts to multiple subscribers
- Metrics collection without delivery guarantees

Message Lifecycle
-----------------

Messages follow this lifecycle:

1. **Send**: Message enters the queue as visible
2. **Receive**: Consumer receives message; it becomes invisible
3. **Process**: Consumer processes the message
4. **Acknowledge**: Message is deleted from queue (success)
5. **Nack**: Message returns to queue for retry (failure)

If the visibility timeout expires before acknowledge/nack, the message
automatically becomes visible again for redelivery.

Request/Reply Pattern
---------------------

Mailbox supports request/reply patterns through the ``reply_to`` parameter.
Workers access ``Message.reply_to`` and call ``Message.reply()`` to send
responses directly to the reply mailbox::

    from weakincentives.runtime.mailbox import InMemoryMailbox

    # Create request and response mailboxes
    requests: InMemoryMailbox[str, str] = InMemoryMailbox(name="requests")
    responses: InMemoryMailbox[str, None] = InMemoryMailbox(name="responses")

    # Client sends request with reply_to
    requests.send("process this", reply_to=responses)

    # Worker processes and replies
    for msg in requests.receive(visibility_timeout=60):
        try:
            result = do_work(msg.body)
            msg.reply(result)  # Sends to msg.reply_to mailbox
            msg.acknowledge()  # Delete from queue
        except Exception:
            msg.nack(visibility_timeout=30)  # Retry after 30s

    # Client receives response
    for msg in responses.receive():
        print(msg.body)
        msg.acknowledge()

Visibility Timeout
------------------

The visibility timeout controls how long a message remains invisible to
other consumers while being processed. Key behaviors:

- **Default**: 30 seconds
- **Range**: 0 to 43,200 seconds (12 hours)
- **Extension**: Use ``msg.extend_visibility(timeout)`` for long processing

When processing takes longer than expected, extend the timeout to prevent
duplicate processing::

    for msg in mailbox.receive(visibility_timeout=300):
        for chunk in process_large_item(msg.body):
            msg.extend_visibility(300)  # Reset timeout

        msg.acknowledge()

Error Handling
--------------

The package defines several exceptions for error handling:

- :exc:`MailboxError` - Base class for all mailbox errors
- :exc:`ReceiptHandleExpiredError` - Handle no longer valid (timeout/ack)
- :exc:`MailboxFullError` - Queue capacity exceeded
- :exc:`SerializationError` - Message body cannot be serialized
- :exc:`MailboxConnectionError` - Cannot connect to backend
- :exc:`ReplyNotAvailableError` - No reply_to specified
- :exc:`MessageFinalizedError` - Message already acknowledged/nacked

Implementations
---------------

**InMemoryMailbox**
    Thread-safe in-memory implementation suitable for single-process
    deployments and testing. Supports all visibility timeout semantics.

**FakeMailbox**
    Testing implementation with methods to simulate failures and edge
    cases. Use for testing error handling and edge conditions::

        fake = FakeMailbox()
        fake.set_connection_error(MailboxConnectionError("Redis down"))
        fake.expire_handle(receipt_handle)  # Simulate timeout

**CollectingMailbox**
    Stores all sent messages for inspection without processing semantics.
    Useful for asserting what was sent in tests.

**NullMailbox**
    Drops all messages silently. Useful as a placeholder when message
    delivery is not needed.

Resolver Pattern
----------------

For distributed deployments, use resolvers to locate mailboxes by name::

    from weakincentives.runtime.mailbox import RegistryResolver, CompositeResolver

    # Register mailboxes by name
    resolver = RegistryResolver()
    resolver.register("requests", requests_mailbox)
    resolver.register("responses", responses_mailbox)

    # Resolve by name
    mailbox = resolver.resolve("requests")

Exports
-------

**Core Types:**
    - :class:`Mailbox` - Protocol for message queue implementations
    - :class:`Message` - Received message with lifecycle methods

**Implementations:**
    - :class:`InMemoryMailbox` - Thread-safe in-memory queue
    - :class:`InMemoryMailboxFactory` - Factory for InMemoryMailbox

**Testing:**
    - :class:`FakeMailbox` - Full implementation with failure injection
    - :class:`CollectingMailbox` - Stores sent messages for inspection
    - :class:`NullMailbox` - Drops all messages silently

**Resolution:**
    - :class:`MailboxResolver` - Protocol for resolving mailboxes by name
    - :class:`MailboxFactory` - Protocol for creating mailboxes
    - :class:`RegistryResolver` - Name-to-mailbox registry
    - :class:`CompositeResolver` - Chain multiple resolvers

**Errors:**
    - :exc:`MailboxError` - Base exception class
    - :exc:`ReceiptHandleExpiredError` - Handle no longer valid
    - :exc:`MailboxFullError` - Queue capacity exceeded
    - :exc:`SerializationError` - Serialization failure
    - :exc:`MailboxConnectionError` - Connection failure
    - :exc:`ReplyNotAvailableError` - No reply_to specified
    - :exc:`MessageFinalizedError` - Already acknowledged/nacked
    - :exc:`InvalidParameterError` - Invalid timeout parameter
    - :exc:`MailboxResolutionError` - Cannot resolve mailbox name

**Utilities:**
    - :func:`validate_visibility_timeout` - Validate timeout range
    - :func:`validate_wait_time` - Validate wait time parameter

See ``specs/MAILBOX.md`` for the complete specification.
"""

from __future__ import annotations

from ._in_memory import InMemoryMailbox, InMemoryMailboxFactory
from ._resolver import (
    CompositeResolver,
    MailboxFactory,
    MailboxResolutionError,
    MailboxResolver,
    RegistryResolver,
)
from ._testing import CollectingMailbox, FakeMailbox, NullMailbox
from ._types import (
    InvalidParameterError,
    Mailbox,
    MailboxConnectionError,
    MailboxError,
    MailboxFullError,
    Message,
    MessageFinalizedError,
    ReceiptHandleExpiredError,
    ReplyNotAvailableError,
    SerializationError,
    validate_visibility_timeout,
    validate_wait_time,
)

__all__ = [
    "CollectingMailbox",
    "CompositeResolver",
    "FakeMailbox",
    "InMemoryMailbox",
    "InMemoryMailboxFactory",
    "InvalidParameterError",
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
    "validate_visibility_timeout",
    "validate_wait_time",
]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})  # pragma: no cover - convenience shim

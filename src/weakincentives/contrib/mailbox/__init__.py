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

"""Redis-backed mailbox implementations for durable inter-agent messaging.

This package provides production-ready mailbox implementations built on Redis,
offering SQS-compatible visibility timeout semantics for reliable message
processing in distributed agent systems.

Dependencies
------------

Requires the ``redis`` package::

    pip install weakincentives[redis]

Supports both standalone Redis and Redis Cluster deployments.

Public Exports
--------------

RedisMailbox
    Durable message queue with visibility timeout semantics. Messages become
    invisible to other consumers when received and automatically reappear if
    not acknowledged within the timeout period. Backed by atomic Lua scripts
    for consistency.

    Type parameters:
        - ``T``: Message body type (must be serializable via ``serde``)
        - ``R``: Reply type (``None`` if no replies expected)

    Key features:
        - Atomic send/receive/acknowledge operations via Lua scripts
        - Visibility timeout with automatic message redelivery
        - Delivery count tracking for dead-letter handling
        - Reply routing via ``reply_to`` parameter
        - Background reaper thread for expired message recovery
        - Redis Cluster compatible (uses hash tags for key locality)

RedisMailboxFactory
    Factory for creating ``RedisMailbox`` instances on a shared Redis client.
    Useful with ``CompositeResolver`` for automatic reply mailbox creation.

    The factory creates send-only mailboxes (no reaper threads) to prevent
    resource leaks when creating ephemeral reply channels.

DEFAULT_TTL_SECONDS
    Default TTL for Redis keys: 259200 seconds (3 days). Keys are refreshed
    on each operation, so active queues persist indefinitely. Set to 0 to
    disable expiration.

Data Structures
---------------

The implementation uses four Redis keys per queue (with hash tags for
cluster compatibility)::

    {queue:name}:pending    # LIST - messages awaiting delivery
    {queue:name}:invisible  # ZSET - in-flight messages scored by expiry
    {queue:name}:data       # HASH - message ID -> serialized body
    {queue:name}:meta       # HASH - delivery counts, timestamps, handles

Formal Verification
-------------------

``RedisMailbox`` includes an embedded TLA+ specification that formally
verifies correctness properties:

- INV-1: Message state exclusivity (pending XOR invisible XOR deleted)
- INV-2/3: Receipt handle validity (stale handles rejected)
- INV-4: Delivery count monotonicity (counts never decrease)
- INV-5: No message loss (messages tracked until acknowledged)

Run ``make verify-formal`` to extract and model-check the specification.

Example Usage
-------------

Basic request/response pattern::

    from redis import Redis
    from weakincentives.contrib.mailbox import RedisMailbox

    client = Redis(host="localhost", port=6379)

    # Create typed mailboxes - type parameters drive deserialization
    requests = RedisMailbox[TaskRequest, TaskResult](
        name="tasks",
        client=client,
    )
    responses = RedisMailbox[TaskResult, None](
        name="results",
        client=client,
    )

    # Producer: send request with reply routing
    requests.send(TaskRequest(task_id="123", payload="data"), reply_to=responses)

    # Consumer: process and reply
    try:
        for msg in requests.receive(visibility_timeout=60, wait_time_seconds=5):
            result = process_task(msg.body)
            msg.reply(result)  # Sends to the responses mailbox
            msg.acknowledge()  # Removes from queue
    finally:
        requests.close()  # Stops reaper thread

Using factory for reply routing::

    from weakincentives.contrib.mailbox import RedisMailbox, RedisMailboxFactory
    from weakincentives.runtime.mailbox import CompositeResolver

    # Factory creates mailboxes on demand for reply resolution
    factory = RedisMailboxFactory(client=redis_client)
    resolver = CompositeResolver(registry={}, factory=factory)

    inbox = RedisMailbox(
        name="inbox",
        client=redis_client,
        reply_resolver=resolver,
    )

    # Messages can reply to any queue name on the same server
    for msg in inbox.receive():
        # reply_to is automatically resolved to a RedisMailbox
        msg.reply(ResponseData(...))
        msg.acknowledge()

Error handling::

    from weakincentives.runtime.mailbox import (
        MailboxFullError,
        ReceiptHandleExpiredError,
        SerializationError,
    )

    try:
        mailbox.send(message)
    except MailboxFullError:
        # Queue at capacity
        pass
    except SerializationError:
        # Message body not serializable
        pass

    for msg in mailbox.receive():
        try:
            process(msg)
            msg.acknowledge()
        except ReceiptHandleExpiredError:
            # Message was redelivered to another consumer
            pass
        except ProcessingError:
            # Return to queue with delay
            msg.nack(visibility_timeout=30)

See Also
--------

- ``specs/MAILBOX.md``: Complete mailbox specification
- ``weakincentives.runtime.mailbox``: Base protocols and error types
"""

from __future__ import annotations

from ._redis import DEFAULT_TTL_SECONDS, RedisMailbox, RedisMailboxFactory

__all__ = ["DEFAULT_TTL_SECONDS", "RedisMailbox", "RedisMailboxFactory"]

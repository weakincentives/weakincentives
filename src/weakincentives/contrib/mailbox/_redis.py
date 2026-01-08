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

"""Redis-backed mailbox implementation with SQS-compatible semantics.

This module provides a durable message queue implementation using Redis.
It supports both standalone Redis and Redis Cluster deployments.

See ``specs/MAILBOX.md`` for the complete specification.
"""

# Pyright suppressions for redis library type stub limitations:
# - Redis[bytes]/RedisCluster[bytes] type args not recognized by stubs
# - register_script() and other methods have incomplete type annotations
# - Script execution returns partially unknown types
# These cannot be fixed without upstream redis-py type stub improvements.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportInvalidTypeArguments=false, reportAttributeAccessIssue=false
# pyright: reportUnknownArgumentType=false, reportUnusedCallResult=false
# pyright: reportOperatorIssue=false

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from weakincentives.formal import (
    Action,
    ActionParameter,
    Invariant,
    StateVar,
    formal_spec,
)
from weakincentives.runtime.mailbox import (
    CompositeResolver,
    MailboxConnectionError,
    MailboxFullError,
    MailboxResolutionError,
    Message,
    ReceiptHandleExpiredError,
    ReplyNotAvailableError,
    SerializationError,
)
from weakincentives.runtime.mailbox._types import (
    validate_visibility_timeout,
    validate_wait_time,
)
from weakincentives.serde import dump, parse

if TYPE_CHECKING:
    from weakincentives.runtime.mailbox import Mailbox, MailboxResolver

# Default TTL for Redis keys: 3 days in seconds.
# Keys are refreshed on each operation, so active queues stay alive indefinitely.
# Set to 0 to disable TTL expiration.
DEFAULT_TTL_SECONDS: int = 259200  # 3 days = 3 * 24 * 60 * 60

# =============================================================================
# Lua Scripts for Atomic Operations
# =============================================================================
# These scripts ensure atomicity of multi-step operations in Redis.
# Each script operates on keys with a common hash tag {queue:name} to
# ensure cluster compatibility.
#
# All scripts use Redis server TIME for visibility calculations to eliminate
# client clock skew as a correctness factor.

# Helper: compute server time as float seconds
_LUA_NOW = """
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
"""

_LUA_SEND = """
-- KEYS: [pending, invisible, data, meta]
-- ARGV: [msg_id, payload, enqueued_at, reply_to, max_size, ttl]
-- reply_to may be empty string for no reply
local max_size = tonumber(ARGV[5])
local ttl = tonumber(ARGV[6])
if max_size and max_size > 0 then
    local pending_n = redis.call('LLEN', KEYS[1])
    local invisible_n = redis.call('ZCARD', KEYS[2])
    if (pending_n + invisible_n) >= max_size then
        return 0
    end
end
redis.call('HSET', KEYS[3], ARGV[1], ARGV[2])
redis.call('HSET', KEYS[4], ARGV[1] .. ':enqueued', ARGV[3])
if ARGV[4] ~= '' then
    redis.call('HSET', KEYS[4], ARGV[1] .. ':reply_to', ARGV[4])
end
redis.call('LPUSH', KEYS[1], ARGV[1])
-- Apply TTL to all keys
if ttl and ttl > 0 then
    redis.call('EXPIRE', KEYS[1], ttl)
    redis.call('EXPIRE', KEYS[2], ttl)
    redis.call('EXPIRE', KEYS[3], ttl)
    redis.call('EXPIRE', KEYS[4], ttl)
end
return 1
"""

_LUA_RECEIVE = """
-- KEYS: [pending, invisible, data, meta]
-- ARGV: [visibility_timeout_seconds, receipt_suffix, ttl]
local msg_id = redis.call('RPOP', KEYS[1])
if not msg_id then return nil end
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local expiry = now + tonumber(ARGV[1])
redis.call('ZADD', KEYS[2], expiry, msg_id)
redis.call('HSET', KEYS[4], msg_id .. ':handle', ARGV[2])
local data = redis.call('HGET', KEYS[3], msg_id)
local count = redis.call('HINCRBY', KEYS[4], msg_id .. ':count', 1)
local enqueued = redis.call('HGET', KEYS[4], msg_id .. ':enqueued')
local reply_to = redis.call('HGET', KEYS[4], msg_id .. ':reply_to')
-- Apply TTL to all keys
local ttl = tonumber(ARGV[3])
if ttl and ttl > 0 then
    redis.call('EXPIRE', KEYS[1], ttl)
    redis.call('EXPIRE', KEYS[2], ttl)
    redis.call('EXPIRE', KEYS[3], ttl)
    redis.call('EXPIRE', KEYS[4], ttl)
end
return {msg_id, data, count, enqueued, reply_to}
"""

_LUA_ACKNOWLEDGE = """
-- KEYS: [invisible, data, meta]
-- ARGV: [msg_id, receipt_suffix, ttl]
local expected = redis.call('HGET', KEYS[3], ARGV[1] .. ':handle')
if expected ~= ARGV[2] then return 0 end
local removed = redis.call('ZREM', KEYS[1], ARGV[1])
if removed == 0 then return 0 end
redis.call('HDEL', KEYS[2], ARGV[1])
redis.call('HDEL', KEYS[3], ARGV[1] .. ':count')
redis.call('HDEL', KEYS[3], ARGV[1] .. ':enqueued')
redis.call('HDEL', KEYS[3], ARGV[1] .. ':handle')
redis.call('HDEL', KEYS[3], ARGV[1] .. ':reply_to')
-- Refresh TTL on remaining keys
local ttl = tonumber(ARGV[3])
if ttl and ttl > 0 then
    redis.call('EXPIRE', KEYS[1], ttl)
    redis.call('EXPIRE', KEYS[2], ttl)
    redis.call('EXPIRE', KEYS[3], ttl)
end
return 1
"""

_LUA_NACK = """
-- KEYS: [invisible, pending, meta, data]
-- ARGV: [msg_id, receipt_suffix, visibility_timeout_seconds, ttl]
local expected = redis.call('HGET', KEYS[3], ARGV[1] .. ':handle')
if expected ~= ARGV[2] then return 0 end
local removed = redis.call('ZREM', KEYS[1], ARGV[1])
if removed == 0 then return 0 end
redis.call('HDEL', KEYS[3], ARGV[1] .. ':handle')
local timeout = tonumber(ARGV[3])
if timeout <= 0 then
    redis.call('LPUSH', KEYS[2], ARGV[1])
else
    local t = redis.call('TIME')
    local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
    redis.call('ZADD', KEYS[1], now + timeout, ARGV[1])
end
-- Refresh TTL on all keys including data
local ttl = tonumber(ARGV[4])
if ttl and ttl > 0 then
    redis.call('EXPIRE', KEYS[1], ttl)
    redis.call('EXPIRE', KEYS[2], ttl)
    redis.call('EXPIRE', KEYS[3], ttl)
    redis.call('EXPIRE', KEYS[4], ttl)
end
return 1
"""

_LUA_EXTEND = """
-- KEYS: [invisible, meta, data]
-- ARGV: [msg_id, receipt_suffix, timeout_seconds, ttl]
local expected = redis.call('HGET', KEYS[2], ARGV[1] .. ':handle')
if expected ~= ARGV[2] then return 0 end
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local expiry = now + tonumber(ARGV[3])
redis.call('ZADD', KEYS[1], 'XX', expiry, ARGV[1])
local score = redis.call('ZSCORE', KEYS[1], ARGV[1])
if not score then return 0 end
-- Refresh TTL on all keys including data
local ttl = tonumber(ARGV[4])
if ttl and ttl > 0 then
    redis.call('EXPIRE', KEYS[1], ttl)
    redis.call('EXPIRE', KEYS[2], ttl)
    redis.call('EXPIRE', KEYS[3], ttl)
end
return 1
"""

_LUA_REAP = """
-- KEYS: [invisible, pending, meta, data]
-- ARGV: [ttl] (computes now from server time)
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local expired = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', now, 'LIMIT', 0, 100)
local count = 0
for i, msg_id in ipairs(expired) do
    redis.call('ZREM', KEYS[1], msg_id)
    redis.call('LPUSH', KEYS[2], msg_id)
    redis.call('HDEL', KEYS[3], msg_id .. ':handle')
    count = count + 1
end
-- Always refresh TTL to keep active queues alive even when no messages expire.
-- This prevents data loss for queues with long visibility timeouts.
local ttl = tonumber(ARGV[1])
if ttl and ttl > 0 then
    redis.call('EXPIRE', KEYS[1], ttl)
    redis.call('EXPIRE', KEYS[2], ttl)
    redis.call('EXPIRE', KEYS[3], ttl)
    redis.call('EXPIRE', KEYS[4], ttl)
end
return count
"""

_LUA_PURGE = """
local pending_count = redis.call('LLEN', KEYS[1])
local invisible_count = redis.call('ZCARD', KEYS[2])
redis.call('DEL', KEYS[1], KEYS[2], KEYS[3], KEYS[4])
return pending_count + invisible_count
"""

if TYPE_CHECKING:
    from redis import Redis
    from redis.cluster import RedisCluster


@dataclass(frozen=True, slots=True)
class _QueueKeys:
    """Redis keys for a named queue with hash tags for cluster compatibility."""

    pending: str
    invisible: str
    data: str
    meta: str

    @classmethod
    def for_queue(cls, name: str) -> _QueueKeys:
        """Create keys for the given queue name."""
        prefix = f"{{queue:{name}}}"
        return cls(
            pending=f"{prefix}:pending",
            invisible=f"{prefix}:invisible",
            data=f"{prefix}:data",
            meta=f"{prefix}:meta",
        )


class RedisMailboxFactory[R]:
    """Factory that creates RedisMailbox instances using a shared Redis client.

    Use with CompositeResolver for automatic reply routing. When a message's
    reply_to identifier is resolved, this factory creates a new RedisMailbox
    connected to the same Redis server.

    Example::

        factory = RedisMailboxFactory(client=redis_client)
        resolver = CompositeResolver(registry={}, factory=factory)
        requests = RedisMailbox(name="requests", client=redis_client, reply_resolver=resolver)

        # Worker can now reply to any queue name
        for msg in requests.receive():
            msg.reply(result)  # Resolves reply_to → new RedisMailbox with same client
            msg.acknowledge()
    """

    __slots__ = ("body_type", "client", "default_ttl")

    client: Redis[bytes] | RedisCluster[bytes]
    """Redis client to use for all created mailboxes."""

    body_type: type[R] | None
    """Optional type hint for message body deserialization."""

    default_ttl: int
    """Default TTL in seconds for all Redis keys."""

    def __init__(
        self,
        client: Redis[bytes] | RedisCluster[bytes],
        *,
        body_type: type[R] | None = None,
        default_ttl: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        """Initialize factory with shared Redis client.

        Args:
            client: Redis client to use for all created mailboxes.
            body_type: Optional type hint for message body deserialization.
            default_ttl: Default TTL in seconds for Redis keys (default: 3 days).
        """
        super().__init__()
        self.client = client
        self.body_type = body_type
        self.default_ttl = default_ttl

    def create(self, identifier: str) -> Mailbox[R, None]:
        """Create a RedisMailbox for the given identifier.

        Args:
            identifier: Queue name for the new mailbox.

        Returns:
            A new RedisMailbox connected to the shared client.

        Note:
            Created mailboxes are send-only: they do not start reaper threads
            and do not support nested reply resolution. This prevents resource
            leaks when creating ephemeral reply mailboxes.
        """
        return RedisMailbox(
            name=identifier,
            client=self.client,
            body_type=self.body_type,
            default_ttl=self.default_ttl,
            _send_only=True,  # Send-only: no reaper thread, no nested resolution
        )


@dataclass(slots=True)
@formal_spec(
    module="RedisMailbox",
    extends=("Integers", "Sequences", "FiniteSets", "TLC"),
    constants={
        "MaxMessages": 2,
        "MaxDeliveries": 2,
        "NumConsumers": 2,
        "VisibilityTimeout": 2,
    },
    state_vars=[
        StateVar(
            "pending", "Seq(MessageId)", "Sequence of message IDs in pending list"
        ),
        StateVar("invisible", "Function", "msg_id -> {expiresAt, handle}"),
        StateVar("data", "Function", "msg_id -> body (or NULL if deleted)"),
        StateVar("handles", "Function", "msg_id -> current valid handle suffix"),
        StateVar("deleted", "Set", "Set of deleted message IDs"),
        StateVar("now", "Nat", "Abstract time counter"),
        StateVar(
            "nextMsgId",
            "Nat",
            "Counter for generating message IDs",
            initial_value="1",
        ),
        StateVar(
            "nextHandle",
            "Nat",
            "Counter for generating handle suffixes",
            initial_value="1",
        ),
        StateVar(
            "consumerState",
            "Function",
            "consumer_id -> {holding, handle}",
            initial_value="[c \\in 1..NumConsumers |-> [holding |-> NULL, handle |-> 0]]",
        ),
        StateVar(
            "deliveryCounts", "Function", "msg_id -> count (persists across requeue)"
        ),
        StateVar(
            "deliveryHistory", "Function", "msg_id -> Seq of (count, handle) for INV-4"
        ),
    ],
    helpers={
        "NULL": "0",
        "InPending(msgId)": r"\E i \in 1..Len(pending): pending[i] = msgId",
        "RemoveKey(f, k)": r"[m \in (DOMAIN f) \ {k} |-> f[m]]",
        "UpdateFunc(f, k, v)": r"[m \in (DOMAIN f) \cup {k} |-> IF m = k THEN v ELSE f[m]]",
    },
    actions=[
        Action(
            name="Send",
            parameters=(ActionParameter("body", "1..MaxMessages"),),
            preconditions=("nextMsgId <= MaxMessages",),
            updates={
                "pending": "Append(pending, nextMsgId)",
                "data": "UpdateFunc(data, nextMsgId, body)",
                "deliveryCounts": "UpdateFunc(deliveryCounts, nextMsgId, 0)",
                "deliveryHistory": "UpdateFunc(deliveryHistory, nextMsgId, <<>>)",
                "nextMsgId": "nextMsgId + 1",
            },
            description="Add a new message to the pending queue (immediate visibility)",
        ),
        Action(
            name="Receive",
            parameters=(ActionParameter("consumer", "1..NumConsumers"),),
            preconditions=(
                "Len(pending) > 0",
                "consumerState[consumer].holding = NULL",
            ),
            updates={
                "pending": "Tail(pending)",
                "invisible": "UpdateFunc(invisible, Head(pending), [expiresAt |-> now + VisibilityTimeout, handle |-> nextHandle])",
                "handles": "UpdateFunc(handles, Head(pending), nextHandle)",
                "deliveryCounts": "[deliveryCounts EXCEPT ![Head(pending)] = @ + 1]",
                "deliveryHistory": "[deliveryHistory EXCEPT ![Head(pending)] = Append(@, [count |-> deliveryCounts[Head(pending)] + 1, handle |-> nextHandle])]",
                "nextHandle": "nextHandle + 1",
                "consumerState": "[consumerState EXCEPT ![consumer] = [holding |-> Head(pending), handle |-> nextHandle]]",
            },
            description="Atomically move message from pending to invisible",
        ),
        Action(
            name="Acknowledge",
            parameters=(ActionParameter("consumer", "1..NumConsumers"),),
            preconditions=(
                "consumerState[consumer].holding /= NULL",
                r"consumerState[consumer].holding \in DOMAIN handles",
                "handles[consumerState[consumer].holding] = consumerState[consumer].handle",
                r"consumerState[consumer].holding \in DOMAIN invisible",
            ),
            updates={
                "invisible": "RemoveKey(invisible, consumerState[consumer].holding)",
                "data": "RemoveKey(data, consumerState[consumer].holding)",
                "handles": "RemoveKey(handles, consumerState[consumer].holding)",
                "deliveryCounts": "RemoveKey(deliveryCounts, consumerState[consumer].holding)",
                r"deleted": r"deleted \cup {consumerState[consumer].holding}",
                "consumerState": "[consumerState EXCEPT ![consumer] = [holding |-> NULL, handle |-> 0]]",
            },
            description="Successfully complete message processing",
        ),
        Action(
            name="AcknowledgeFail",
            parameters=(ActionParameter("consumer", "1..NumConsumers"),),
            preconditions=(
                "consumerState[consumer].holding /= NULL",
                r"consumerState[consumer].holding \notin DOMAIN handles \/ handles[consumerState[consumer].holding] /= consumerState[consumer].handle \/ consumerState[consumer].holding \notin DOMAIN invisible",
            ),
            updates={
                "consumerState": "[consumerState EXCEPT ![consumer] = [holding |-> NULL, handle |-> 0]]",
            },
            description="Acknowledge fails if handle is stale",
        ),
        Action(
            name="Nack",
            parameters=(
                ActionParameter("consumer", "1..NumConsumers"),
                ActionParameter("newTimeout", "0..VisibilityTimeout"),
            ),
            preconditions=(
                "consumerState[consumer].holding /= NULL",
                r"consumerState[consumer].holding \in DOMAIN handles",
                "handles[consumerState[consumer].holding] = consumerState[consumer].handle",
                r"consumerState[consumer].holding \in DOMAIN invisible",
            ),
            updates={
                "pending": "IF newTimeout = 0 THEN Append(pending, consumerState[consumer].holding) ELSE pending",
                "invisible": "IF newTimeout = 0 THEN RemoveKey(invisible, consumerState[consumer].holding) ELSE [invisible EXCEPT ![consumerState[consumer].holding].expiresAt = now + newTimeout, ![consumerState[consumer].holding].handle = 0]",
                "handles": "RemoveKey(handles, consumerState[consumer].holding)",
                "consumerState": "[consumerState EXCEPT ![consumer] = [holding |-> NULL, handle |-> 0]]",
            },
            description="Return message to queue with optional delay",
        ),
        Action(
            name="NackFail",
            parameters=(ActionParameter("consumer", "1..NumConsumers"),),
            preconditions=(
                "consumerState[consumer].holding /= NULL",
                r"consumerState[consumer].holding \notin DOMAIN handles \/ handles[consumerState[consumer].holding] /= consumerState[consumer].handle \/ consumerState[consumer].holding \notin DOMAIN invisible",
            ),
            updates={
                "consumerState": "[consumerState EXCEPT ![consumer] = [holding |-> NULL, handle |-> 0]]",
            },
            description="Nack fails if handle is stale",
        ),
        Action(
            name="Extend",
            parameters=(
                ActionParameter("consumer", "1..NumConsumers"),
                ActionParameter("newTimeout", "1..VisibilityTimeout"),
            ),
            preconditions=(
                "consumerState[consumer].holding /= NULL",
                r"consumerState[consumer].holding \in DOMAIN handles",
                "handles[consumerState[consumer].holding] = consumerState[consumer].handle",
                r"consumerState[consumer].holding \in DOMAIN invisible",
            ),
            updates={
                "invisible": "[invisible EXCEPT ![consumerState[consumer].holding].expiresAt = now + newTimeout]",
            },
            description="Extend visibility timeout for a message",
        ),
        Action(
            name="ExtendFail",
            parameters=(ActionParameter("consumer", "1..NumConsumers"),),
            preconditions=(
                "consumerState[consumer].holding /= NULL",
                r"consumerState[consumer].holding \notin DOMAIN handles \/ handles[consumerState[consumer].holding] /= consumerState[consumer].handle \/ consumerState[consumer].holding \notin DOMAIN invisible",
            ),
            updates={
                "consumerState": "[consumerState EXCEPT ![consumer] = [holding |-> NULL, handle |-> 0]]",
            },
            description="Extend fails if handle is stale or message not in invisible",
        ),
        Action(
            name="ReapOne",
            parameters=(),
            preconditions=(
                r"\E msgId \in DOMAIN invisible: invisible[msgId].expiresAt < now",
            ),
            updates={
                r"pending": r"Append(pending, CHOOSE msgId \in DOMAIN invisible: invisible[msgId].expiresAt < now)",
                r"invisible": r"RemoveKey(invisible, CHOOSE msgId \in DOMAIN invisible: invisible[msgId].expiresAt < now)",
                r"handles": r"RemoveKey(handles, CHOOSE msgId \in DOMAIN invisible: invisible[msgId].expiresAt < now)",
                r"consumerState": r"[c \in DOMAIN consumerState |-> IF consumerState[c].holding = (CHOOSE msgId \in DOMAIN invisible: invisible[msgId].expiresAt < now) THEN [holding |-> NULL, handle |-> 0] ELSE consumerState[c]]",
            },
            description="Move one expired message back to pending",
        ),
        Action(
            name="Tick",
            parameters=(),
            preconditions=(),
            updates={"now": "now + 1"},
            description="Advance abstract time",
        ),
    ],
    invariants=[
        Invariant(
            id="INV-1",
            name="MessageStateExclusive",
            predicate=r"""
\A msgId \in 1..nextMsgId-1:
    LET inPending == InPending(msgId)
        inInvisible == msgId \in DOMAIN invisible
        inDeleted == msgId \in deleted
    IN (inPending /\ ~inInvisible /\ ~inDeleted) \/
       (~inPending /\ inInvisible /\ ~inDeleted) \/
       (~inPending /\ ~inInvisible /\ inDeleted)
""".strip(),
            description="A message must be in exactly one state: pending, invisible, or deleted",
        ),
        Invariant(
            id="INV-2-3",
            name="HandleValidity",
            predicate=r"""
\A c \in 1..NumConsumers:
    LET state == consumerState[c]
    IN state.holding /= NULL =>
        (state.holding \in DOMAIN handles =>
            handles[state.holding] = state.handle)
""".strip(),
            description="Consumers holding a message have a valid handle for it",
        ),
        Invariant(
            id="INV-4",
            name="DeliveryCountMonotonic",
            predicate=r"""
\A msgId \in DOMAIN deliveryHistory:
    LET history == deliveryHistory[msgId]
    IN \A i \in 1..Len(history)-1:
        history[i].count < history[i+1].count
""".strip(),
            description="Delivery counts are strictly increasing",
        ),
        Invariant(
            id="INV-4b",
            name="DeliveryCountPersistence",
            predicate=r"""
\A msgId \in DOMAIN deliveryCounts:
    \A i \in 1..Len(deliveryHistory[msgId]):
        deliveryHistory[msgId][i].count = i
""".strip(),
            description="Delivery counts persist across requeue",
        ),
        Invariant(
            id="INV-5",
            name="NoMessageLoss",
            predicate=r"""
\A msgId \in DOMAIN data:
    LET inPending == InPending(msgId)
        inInvisible == msgId \in DOMAIN invisible
    IN inPending \/ inInvisible
""".strip(),
            description="Every message with data is either pending or invisible",
        ),
        Invariant(
            id="INV-7",
            name="HandleUniqueness",
            predicate=r"""
\A msgId \in DOMAIN deliveryHistory:
    LET history == deliveryHistory[msgId]
    IN \A i, j \in 1..Len(history):
        i /= j => history[i].handle /= history[j].handle
""".strip(),
            description="Each delivery of a message gets a unique handle",
        ),
        Invariant(
            id="INV-8",
            name="PendingNoDuplicates",
            predicate=r"""
\A i, j \in 1..Len(pending):
    i /= j => pending[i] /= pending[j]
""".strip(),
            description="The pending queue contains no duplicate message IDs",
        ),
        Invariant(
            id="INV-9",
            name="DataIntegrity",
            predicate=r"""
\A msgId \in 1..nextMsgId-1:
    (InPending(msgId) \/ msgId \in DOMAIN invisible) => msgId \in DOMAIN data
""".strip(),
            description="Every message in pending or invisible has associated data",
        ),
    ],
    constraint="now <= 2",
)
class RedisMailbox[T, R]:
    """Redis-backed mailbox with SQS-compatible visibility timeout semantics.

    This implementation is formally verified against the embedded TLA+ specification.
    The specification can be extracted and model-checked with:

        make verify-formal

    Supports both standalone Redis and Redis Cluster deployments. Uses Lua scripts
    for atomic operations and a background reaper thread for visibility timeout
    management.

    Type parameters:
        T: Message body type.
        R: Reply type (None if no replies expected).

    Data structures::

        {queue:name}:pending    # LIST - messages awaiting delivery (LPUSH/RPOP)
        {queue:name}:invisible  # ZSET - in-flight messages scored by expiry timestamp
        {queue:name}:data       # HASH - message ID → serialized message body
        {queue:name}:meta       # HASH - message ID:count → delivery count,
                                #        message ID:enqueued → enqueued timestamp,
                                #        message ID:handle → current receipt handle suffix
                                #        message ID:reply_to → reply destination

    Formal Specification:
        The @formal_spec decorator above defines the complete TLA+ state machine
        for this implementation. Key invariants verified:

        - INV-1: Message State Exclusivity (pending XOR invisible XOR deleted)
        - INV-2-3: Receipt Handle Validity (stale handles rejected)
        - INV-4/4b: Delivery Count Monotonicity (counts never decrease)
        - INV-5: No Message Loss (messages tracked until acknowledged)
        - INV-7: Handle Uniqueness (each delivery gets unique handle)
        - INV-8: Pending No Duplicates (no duplicate IDs in pending queue)
        - INV-9: Data Integrity (queued messages have associated data)

        See specs/VERIFICATION.md for complete verification documentation.

    Example::

        from redis import Redis
        from weakincentives.contrib.mailbox import RedisMailbox

        client = Redis(host="localhost", port=6379)
        mailbox: RedisMailbox[MyEvent, MyResult] = RedisMailbox(
            name="events",
            client=client,
        )

        try:
            mailbox.send(MyEvent(data="hello"), reply_to="responses")
            for msg in mailbox.receive(visibility_timeout=60):
                result = process(msg.body)
                msg.reply(result)
                msg.acknowledge()
        finally:
            mailbox.close()
    """

    name: str
    """Queue name for identification. Used in Redis key prefixes."""

    client: Redis[bytes] | RedisCluster[bytes]
    """Redis client instance. Can be standalone Redis or RedisCluster."""

    body_type: type[T] | None = None
    """Optional type hint for message body deserialization."""

    max_size: int | None = None
    """Maximum queue capacity. None for unlimited (subject to Redis maxmemory)."""

    reply_resolver: MailboxResolver[R] | None = None
    """Resolver for reply_to identifiers. If None, a default resolver using
    RedisMailboxFactory is created automatically, enabling reply_to to resolve
    to any queue name on the same Redis server."""

    reaper_interval: float = 1.0
    """Interval in seconds between visibility reaper runs."""

    default_ttl: int = DEFAULT_TTL_SECONDS
    """Default TTL in seconds for all Redis keys (default: 3 days).
    Set to 0 to disable TTL. Keys are refreshed on each operation including
    the background reaper, so active queues stay alive indefinitely."""

    _keys: _QueueKeys = field(init=False, repr=False)
    _scripts: dict[str, Any] = field(init=False, default_factory=dict, repr=False)
    _reaper_thread: threading.Thread | None = field(
        default=None, repr=False, init=False
    )
    _stop_reaper: threading.Event = field(
        default_factory=threading.Event, repr=False, init=False
    )
    _closed: bool = field(default=False, repr=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _send_only: bool = field(default=False, repr=False)
    """Internal flag for send-only mailboxes. Send-only mailboxes don't start
    reaper threads and don't support reply resolution. Used by RedisMailboxFactory
    to prevent resource leaks when creating ephemeral reply mailboxes."""

    def __post_init__(self) -> None:
        """Initialize keys, register Lua scripts, and optionally start reaper thread.

        Send-only mailboxes (created by RedisMailboxFactory for replies) skip
        reaper thread creation and auto-resolver setup to prevent resource leaks.
        """
        object.__setattr__(self, "_keys", _QueueKeys.for_queue(self.name))
        self._register_scripts()

        # Send-only mailboxes don't need reaper threads or reply resolution
        if self._send_only:
            return

        self._start_reaper()
        # Set up default resolver if none provided
        if self.reply_resolver is None:
            factory: RedisMailboxFactory[R] = RedisMailboxFactory(
                client=self.client, default_ttl=self.default_ttl
            )
            object.__setattr__(
                self, "reply_resolver", CompositeResolver(registry={}, factory=factory)
            )

    def _register_scripts(self) -> None:
        """Register Lua scripts with Redis."""
        self._scripts["send"] = self.client.register_script(_LUA_SEND)
        self._scripts["receive"] = self.client.register_script(_LUA_RECEIVE)
        self._scripts["acknowledge"] = self.client.register_script(_LUA_ACKNOWLEDGE)
        self._scripts["nack"] = self.client.register_script(_LUA_NACK)
        self._scripts["extend"] = self.client.register_script(_LUA_EXTEND)
        self._scripts["reap"] = self.client.register_script(_LUA_REAP)
        self._scripts["purge"] = self.client.register_script(_LUA_PURGE)

    def _start_reaper(self) -> None:
        """Start background thread to requeue expired messages."""
        self._reaper_thread = threading.Thread(
            target=self._reaper_loop,
            daemon=True,
            name=f"redis-mailbox-reaper-{self.name}",
        )
        self._reaper_thread.start()

    def _reaper_loop(self) -> None:
        """Background loop that requeues expired visibility timeouts."""
        while not self._stop_reaper.wait(timeout=self.reaper_interval):
            with contextlib.suppress(Exception):
                self._reap_expired()

    def _reap_expired(self) -> int:
        """Move expired messages from invisible back to pending.

        Uses Redis server TIME for consistency across consumers. Also clears
        receipt handles so stale handles cannot acknowledge messages after
        they're redelivered.

        Returns:
            Number of messages requeued.
        """
        keys = [
            self._keys.invisible,
            self._keys.pending,
            self._keys.meta,
            self._keys.data,
        ]
        result = self._scripts["reap"](keys=keys, args=[self.default_ttl])
        return int(result) if result else 0

    def _serialize(self, body: T) -> str:
        """Serialize message body to JSON string."""
        try:
            # dump() works with dataclasses, for primitives use json.dumps
            if hasattr(body, "__dataclass_fields__"):
                return json.dumps(dump(body))
            return json.dumps(body)
        except Exception as e:
            raise SerializationError(f"Failed to serialize message body: {e}") from e

    def _deserialize(self, data: bytes | str) -> T:
        """Deserialize message body from JSON bytes or string.

        Handles both bytes (default Redis client) and str (decode_responses=True).
        """
        try:
            json_str = data.decode("utf-8") if isinstance(data, bytes) else data
            json_data = json.loads(json_str)
            if self.body_type is not None:
                # Use parse() only for dataclass types
                if hasattr(self.body_type, "__dataclass_fields__"):
                    return parse(self.body_type, json_data)
                # For primitive types (str, int, etc.), construct directly
                body_type: Any = self.body_type
                return body_type(json_data)
            # Without a type hint, return raw JSON data
            return cast(T, json_data)
        except Exception as e:
            raise SerializationError(f"Failed to deserialize message body: {e}") from e

    def send(self, body: T, *, reply_to: str | None = None) -> str:
        """Enqueue a message.

        This operation is atomic via Lua script, ensuring no partial state
        on failure.

        Args:
            body: Message payload (must be serializable via serde).
            reply_to: Identifier for response mailbox. Workers resolve this
                via Message.reply().

        Returns:
            Message ID (unique within this queue).

        Raises:
            MailboxFullError: Queue capacity exceeded (checked atomically).
            SerializationError: Body cannot be serialized.
            MailboxConnectionError: Cannot connect to Redis.
        """
        msg_id = str(uuid4())
        serialized = self._serialize(body)
        enqueued_at = datetime.now(UTC).isoformat()

        try:
            keys = [
                self._keys.pending,
                self._keys.invisible,
                self._keys.data,
                self._keys.meta,
            ]
            args = [
                msg_id,
                serialized,
                enqueued_at,
                reply_to or "",  # Empty string for no reply_to
                self.max_size or 0,
                self.default_ttl,
            ]
            result = self._scripts["send"](keys=keys, args=args)

            if result == 0:
                raise MailboxFullError(
                    f"Queue '{self.name}' at capacity ({self.max_size})"
                )

            return msg_id

        except (MailboxFullError, SerializationError):
            raise
        except Exception as e:
            raise MailboxConnectionError(f"Failed to send message: {e}") from e

    def receive(
        self,
        *,
        max_messages: int = 1,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
    ) -> Sequence[Message[T, R]]:
        """Receive messages from the queue.

        Args:
            max_messages: Maximum messages to receive (1-10).
            visibility_timeout: Seconds message remains invisible (0-43200).
            wait_time_seconds: Long poll duration in seconds. Zero returns
                immediately if no messages are available.

        Returns:
            Sequence of messages (may be empty). Returns empty if mailbox closed.

        Raises:
            InvalidParameterError: visibility_timeout or wait_time_seconds out of range.
            MailboxConnectionError: Cannot connect to Redis.
        """
        validate_visibility_timeout(visibility_timeout)
        validate_wait_time(wait_time_seconds)

        if self._closed:
            return []

        max_messages = min(max(1, max_messages), 10)
        messages: list[Message[T, R]] = []
        deadline = time.time() + wait_time_seconds

        try:
            while len(messages) < max_messages and not self._closed:
                remaining = deadline - time.time()
                msg = self._receive_one(visibility_timeout, remaining)
                if msg is not None:
                    messages.append(msg)
                elif wait_time_seconds <= 0 or remaining <= 0:
                    # No wait requested, or timeout expired
                    break

            return messages

        except SerializationError:
            raise
        except Exception as e:
            if self._closed:
                return messages
            raise MailboxConnectionError(f"Failed to receive messages: {e}") from e

    def _receive_one(
        self, visibility_timeout: int, wait_seconds: float
    ) -> Message[T, R] | None:
        """Attempt to receive a single message.

        Uses an atomic Lua script to pop from pending and move to invisible
        in a single operation. Uses Redis server TIME for expiry calculation
        to eliminate client clock skew. For long-polling, polls with 300ms
        intervals to avoid the atomicity gap that would exist with BRPOP +
        pipeline.

        Args:
            visibility_timeout: Seconds message remains invisible.
            wait_seconds: Max seconds to wait for a message.

        Returns:
            A Message if one was received, None otherwise.
        """
        keys = self._keys
        lua_keys = [keys.pending, keys.invisible, keys.data, keys.meta]
        poll_interval = 0.3  # 300ms between poll attempts

        deadline = time.time() + wait_seconds

        while True:
            receipt_suffix = str(uuid4())

            # Atomic Lua script: RPOP + ZADD + metadata in one operation
            # Script computes expiry using Redis server TIME + visibility_timeout
            result = self._scripts["receive"](
                keys=lua_keys,
                args=[visibility_timeout, receipt_suffix, self.default_ttl],
            )

            if result is not None:
                msg_id = (
                    result[0].decode("utf-8")
                    if isinstance(result[0], bytes)
                    else str(result[0])
                )
                data = result[1]
                delivery_count = int(result[2])
                enqueued_raw = result[3]
                # Result tuple: [msg_id, data, count, enqueued, reply_to?]
                reply_to_raw = result[4] if len(result) > 4 else None  # noqa: PLR2004
                break

            # No message available
            remaining = deadline - time.time()
            if remaining <= 0:
                return None

            # Poll again after interval (but don't overshoot deadline)
            time.sleep(min(poll_interval, remaining))

        if data is None:
            # Message data was deleted (shouldn't happen, but handle gracefully)
            return None

        # Parse enqueued timestamp
        if enqueued_raw:
            enqueued_str = (
                enqueued_raw.decode("utf-8")
                if isinstance(enqueued_raw, bytes)
                else str(enqueued_raw)
            )
            enqueued_at = datetime.fromisoformat(enqueued_str)
        else:
            enqueued_at = datetime.now(UTC)

        # Parse reply_to
        reply_to: str | None = None
        if reply_to_raw:
            reply_to = (
                reply_to_raw.decode("utf-8")
                if isinstance(reply_to_raw, bytes)
                else str(reply_to_raw)
            )

        # Compose receipt handle from msg_id and suffix
        receipt_handle = f"{msg_id}:{receipt_suffix}"

        # Deserialize body
        body = self._deserialize(data)

        # Bind callbacks with both msg_id and receipt_suffix for validation
        return Message[T, R](
            id=msg_id,
            body=body,
            receipt_handle=receipt_handle,
            delivery_count=delivery_count,
            enqueued_at=enqueued_at,
            reply_to=reply_to,
            _acknowledge_fn=lambda mid=msg_id, suf=receipt_suffix: self._acknowledge(
                mid, suf
            ),
            _nack_fn=lambda t, mid=msg_id, suf=receipt_suffix: self._nack(mid, suf, t),
            _extend_fn=lambda t, mid=msg_id, suf=receipt_suffix: self._extend(
                mid, suf, t
            ),
            _reply_fn=lambda b, rt=reply_to: self._reply(rt, b),
        )

    def _reply(self, reply_to: str | None, body: R) -> str:
        """Send a reply to the reply_to mailbox."""
        if reply_to is None:
            raise ReplyNotAvailableError("No reply_to specified")
        if self.reply_resolver is None:
            raise ReplyNotAvailableError("No reply_resolver configured")
        try:
            mailbox = self.reply_resolver.resolve(reply_to)
        except MailboxResolutionError as e:
            raise ReplyNotAvailableError(f"Cannot resolve reply_to '{reply_to}'") from e
        return mailbox.send(body)

    def _acknowledge(self, msg_id: str, receipt_suffix: str) -> None:
        """Delete message from queue.

        Args:
            msg_id: The message ID.
            receipt_suffix: The receipt handle suffix for validation.

        Raises:
            ReceiptHandleExpiredError: If the receipt handle doesn't match
                the current delivery (message was redelivered or already acked).
        """
        keys = [self._keys.invisible, self._keys.data, self._keys.meta]
        result = self._scripts["acknowledge"](
            keys=keys, args=[msg_id, receipt_suffix, self.default_ttl]
        )
        if result == 0:
            raise ReceiptHandleExpiredError(
                f"Message '{msg_id}' not found or receipt handle expired"
            )

    def _nack(self, msg_id: str, receipt_suffix: str, visibility_timeout: int) -> None:
        """Return message to queue.

        Uses Redis server TIME for expiry calculation to eliminate clock skew.

        Args:
            msg_id: The message ID.
            receipt_suffix: The receipt handle suffix for validation.
            visibility_timeout: Seconds before message becomes visible again.

        Raises:
            ReceiptHandleExpiredError: If the receipt handle doesn't match
                the current delivery.
        """
        keys = [
            self._keys.invisible,
            self._keys.pending,
            self._keys.meta,
            self._keys.data,
        ]
        # Script computes expiry using Redis server TIME + visibility_timeout
        result = self._scripts["nack"](
            keys=keys,
            args=[msg_id, receipt_suffix, visibility_timeout, self.default_ttl],
        )
        if result == 0:
            raise ReceiptHandleExpiredError(
                f"Message '{msg_id}' not found or receipt handle expired"
            )

    def _extend(self, msg_id: str, receipt_suffix: str, timeout: int) -> None:
        """Extend visibility timeout.

        Uses Redis server TIME for expiry calculation to eliminate clock skew.

        Args:
            msg_id: The message ID.
            receipt_suffix: The receipt handle suffix for validation.
            timeout: New visibility timeout in seconds from now.

        Raises:
            ReceiptHandleExpiredError: If the receipt handle doesn't match
                the current delivery.
        """
        # Script computes expiry using Redis server TIME + timeout
        result = self._scripts["extend"](
            keys=[self._keys.invisible, self._keys.meta, self._keys.data],
            args=[msg_id, receipt_suffix, timeout, self.default_ttl],
        )
        if result == 0:
            raise ReceiptHandleExpiredError(
                f"Message '{msg_id}' not found or receipt handle expired"
            )

    def purge(self) -> int:
        """Delete all messages from the queue.

        Returns:
            Approximate count of messages deleted.

        Note:
            Unlike SQS, Redis has no purge cooldown.
        """
        keys = [
            self._keys.pending,
            self._keys.invisible,
            self._keys.data,
            self._keys.meta,
        ]
        try:
            result = self._scripts["purge"](keys=keys, args=[])
            return int(result) if result else 0
        except Exception as e:
            raise MailboxConnectionError(f"Failed to purge queue: {e}") from e

    def approximate_count(self) -> int:
        """Return exact number of messages in the queue.

        For Redis, this count is exact (unlike SQS which is approximate).
        Includes both visible (pending) and invisible (in-flight) messages.
        """
        try:
            pending = self.client.llen(self._keys.pending)
            invisible = self.client.zcard(self._keys.invisible)
            return pending + invisible
        except Exception as e:
            raise MailboxConnectionError(f"Failed to get queue count: {e}") from e

    def close(self) -> None:
        """Stop the reaper thread if running. Does not close the Redis client.

        For send-only mailboxes, this is a no-op since no reaper thread is started.
        """
        with self._lock:
            self._closed = True

        # Stop reaper thread (not started for send-only mailboxes)
        self._stop_reaper.set()
        if self._reaper_thread is not None:
            self._reaper_thread.join(timeout=2.0)

    @property
    def closed(self) -> bool:
        """Return True if mailbox has been closed."""
        return self._closed


__all__ = ["DEFAULT_TTL_SECONDS", "RedisMailbox", "RedisMailboxFactory"]

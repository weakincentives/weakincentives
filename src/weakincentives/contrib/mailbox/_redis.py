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

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportInvalidTypeArguments=false, reportAttributeAccessIssue=false
# pyright: reportUnknownArgumentType=false, reportUnusedCallResult=false
# pyright: reportArgumentType=false, reportUnnecessaryComparison=false
# pyright: reportGeneralTypeIssues=false, reportUnknownLambdaType=false
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

from weakincentives.runtime.mailbox import (
    MailboxConnectionError,
    MailboxFullError,
    Message,
    ReceiptHandleExpiredError,
    SerializationError,
)
from weakincentives.serde import dump, parse

# =============================================================================
# Lua Scripts for Atomic Operations
# =============================================================================
# These scripts ensure atomicity of multi-step operations in Redis.
# Each script operates on keys with a common hash tag {queue:name} to
# ensure cluster compatibility.

_LUA_RECEIVE = """
local msg_id = redis.call('RPOP', KEYS[1])
if not msg_id then return nil end
redis.call('ZADD', KEYS[2], ARGV[1], msg_id)
redis.call('HSET', KEYS[4], msg_id .. ':handle', ARGV[2])
local data = redis.call('HGET', KEYS[3], msg_id)
local count = redis.call('HINCRBY', KEYS[4], msg_id .. ':count', 1)
local enqueued = redis.call('HGET', KEYS[4], msg_id .. ':enqueued')
return {msg_id, data, count, enqueued}
"""

_LUA_ACKNOWLEDGE = """
local expected = redis.call('HGET', KEYS[3], ARGV[1] .. ':handle')
if expected ~= ARGV[2] then return 0 end
local removed = redis.call('ZREM', KEYS[1], ARGV[1])
if removed == 0 then return 0 end
redis.call('HDEL', KEYS[2], ARGV[1])
redis.call('HDEL', KEYS[3], ARGV[1] .. ':count')
redis.call('HDEL', KEYS[3], ARGV[1] .. ':enqueued')
redis.call('HDEL', KEYS[3], ARGV[1] .. ':handle')
return 1
"""

_LUA_NACK = """
local expected = redis.call('HGET', KEYS[3], ARGV[1] .. ':handle')
if expected ~= ARGV[2] then return 0 end
local removed = redis.call('ZREM', KEYS[1], ARGV[1])
if removed == 0 then return 0 end
redis.call('HDEL', KEYS[3], ARGV[1] .. ':handle')
if tonumber(ARGV[3]) <= 0 then
    redis.call('LPUSH', KEYS[2], ARGV[1])
else
    redis.call('ZADD', KEYS[1], ARGV[3], ARGV[1])
end
return 1
"""

_LUA_EXTEND = """
local expected = redis.call('HGET', KEYS[2], ARGV[1] .. ':handle')
if expected ~= ARGV[2] then return 0 end
redis.call('ZADD', KEYS[1], 'XX', ARGV[3], ARGV[1])
local score = redis.call('ZSCORE', KEYS[1], ARGV[1])
return score and 1 or 0
"""

_LUA_REAP = """
local expired = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1], 'LIMIT', 0, 100)
local count = 0
for i, msg_id in ipairs(expired) do
    redis.call('ZREM', KEYS[1], msg_id)
    redis.call('LPUSH', KEYS[2], msg_id)
    redis.call('HDEL', KEYS[3], msg_id .. ':handle')
    count = count + 1
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


@dataclass(slots=True)
class RedisMailbox[T]:
    """Redis-backed mailbox with SQS-compatible visibility timeout semantics.

    Supports both standalone Redis and Redis Cluster deployments. Uses Lua scripts
    for atomic operations and a background reaper thread for visibility timeout
    management.

    Data structures::

        {queue:name}:pending    # LIST - messages awaiting delivery (LPUSH/RPOP)
        {queue:name}:invisible  # ZSET - in-flight messages scored by expiry timestamp
        {queue:name}:data       # HASH - message ID → serialized message body
        {queue:name}:meta       # HASH - message ID:count → delivery count,
                                #        message ID:enqueued → enqueued timestamp,
                                #        message ID:handle → current receipt handle suffix

    Example::

        from redis import Redis
        from weakincentives.contrib.mailbox import RedisMailbox

        client = Redis(host="localhost", port=6379)
        mailbox: RedisMailbox[MyEvent] = RedisMailbox(
            name="events",
            client=client,
        )

        try:
            mailbox.send(MyEvent(data="hello"))
            for msg in mailbox.receive(visibility_timeout=60):
                process(msg.body)
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

    reaper_interval: float = 1.0
    """Interval in seconds between visibility reaper runs."""

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

    def __post_init__(self) -> None:
        """Initialize keys, register Lua scripts, and start reaper thread."""
        object.__setattr__(self, "_keys", _QueueKeys.for_queue(self.name))
        self._register_scripts()
        self._start_reaper()

    def _register_scripts(self) -> None:
        """Register Lua scripts with Redis."""
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

        Also clears receipt handles so stale handles cannot acknowledge
        messages after they're redelivered.

        Returns:
            Number of messages requeued.
        """
        now = time.time()
        keys = [self._keys.invisible, self._keys.pending, self._keys.meta]
        result = self._scripts["reap"](keys=keys, args=[now])
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

    def _deserialize(self, data: bytes) -> T:
        """Deserialize message body from JSON bytes."""
        try:
            json_data = json.loads(data.decode("utf-8"))
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

    def send(self, body: T, *, delay_seconds: int = 0) -> str:
        """Enqueue a message, optionally delaying visibility.

        Args:
            body: Message payload (must be serializable via serde).
            delay_seconds: Seconds before message becomes visible (0-900).

        Returns:
            Message ID (unique within this queue).

        Raises:
            MailboxFullError: Queue capacity exceeded.
            SerializationError: Body cannot be serialized.
            MailboxConnectionError: Cannot connect to Redis.
        """
        msg_id = str(uuid4())
        serialized = self._serialize(body)
        enqueued_at = datetime.now(UTC).isoformat()

        try:
            # Check capacity if max_size is set
            if self.max_size is not None:
                current = self.approximate_count()
                if current >= self.max_size:
                    raise MailboxFullError(
                        f"Queue '{self.name}' at capacity ({self.max_size})"
                    )

            pipe = self.client.pipeline()

            # Store message data
            pipe.hset(self._keys.data, msg_id, serialized)

            # Store enqueued timestamp in meta
            pipe.hset(self._keys.meta, f"{msg_id}:enqueued", enqueued_at)

            if delay_seconds > 0:
                # Delayed message: add to invisible with future expiry
                expiry_score = time.time() + delay_seconds
                pipe.zadd(self._keys.invisible, {msg_id: expiry_score})
            else:
                # Immediate visibility: add to pending queue
                pipe.lpush(self._keys.pending, msg_id)

            pipe.execute()
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
    ) -> Sequence[Message[T]]:
        """Receive messages from the queue.

        Args:
            max_messages: Maximum messages to receive (1-10).
            visibility_timeout: Seconds message remains invisible (0-43200).
            wait_time_seconds: Long poll duration. Zero returns immediately.
                For Redis, uses BRPOP for efficient blocking when > 0.

        Returns:
            Sequence of messages (may be empty). Returns empty if mailbox closed.

        Raises:
            MailboxConnectionError: Cannot connect to Redis.
        """
        if self._closed:
            return []

        max_messages = min(max(1, max_messages), 10)
        messages: list[Message[T]] = []
        deadline = time.time() + wait_time_seconds

        try:
            while len(messages) < max_messages and not self._closed:
                remaining = deadline - time.time()
                if remaining <= 0 and wait_time_seconds > 0 and not messages:
                    # Timeout expired during long poll
                    break

                # Try to receive a message
                msg = self._receive_one(visibility_timeout, remaining)
                if msg is not None:
                    messages.append(msg)
                elif wait_time_seconds <= 0:
                    # No wait - return immediately
                    break
                elif remaining <= 0:
                    # Timeout expired
                    break

            return messages

        except Exception as e:
            if self._closed:
                return messages
            raise MailboxConnectionError(f"Failed to receive messages: {e}") from e

    def _receive_one(
        self, visibility_timeout: int, wait_seconds: float
    ) -> Message[T] | None:
        """Attempt to receive a single message.

        Args:
            visibility_timeout: Seconds message remains invisible.
            wait_seconds: Max seconds to wait for a message.

        Returns:
            A Message if one was received, None otherwise.
        """
        keys = self._keys
        expiry_score = time.time() + visibility_timeout

        # Generate receipt handle suffix upfront - stored in Redis for validation
        receipt_suffix = str(uuid4())

        # Use blocking pop if waiting, otherwise use Lua script
        if wait_seconds > 0:
            # BRPOP for efficient long polling
            result = self.client.brpop(keys.pending, timeout=int(wait_seconds) or 1)
            if result is None:
                return None

            _, msg_id_bytes = result
            msg_id = msg_id_bytes.decode("utf-8")

            # Now atomically process the message
            pipe = self.client.pipeline()
            pipe.zadd(keys.invisible, {msg_id: expiry_score})
            pipe.hset(keys.meta, f"{msg_id}:handle", receipt_suffix)
            pipe.hget(keys.data, msg_id)
            pipe.hincrby(keys.meta, f"{msg_id}:count", 1)
            pipe.hget(keys.meta, f"{msg_id}:enqueued")
            results = pipe.execute()

            data = results[2]
            delivery_count = results[3]
            enqueued_raw = results[4]
        else:
            # Use Lua script for atomic non-blocking receive
            lua_keys = [keys.pending, keys.invisible, keys.data, keys.meta]
            result = self._scripts["receive"](
                keys=lua_keys, args=[expiry_score, receipt_suffix]
            )

            if result is None:
                return None

            msg_id = (
                result[0].decode("utf-8")
                if isinstance(result[0], bytes)
                else str(result[0])
            )
            data = result[1]
            delivery_count = int(result[2])
            enqueued_raw = result[3]

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

        # Compose receipt handle from msg_id and suffix
        receipt_handle = f"{msg_id}:{receipt_suffix}"

        # Deserialize body
        body = self._deserialize(data)

        # Bind callbacks with both msg_id and receipt_suffix for validation
        return Message(
            id=msg_id,
            body=body,
            receipt_handle=receipt_handle,
            delivery_count=delivery_count,
            enqueued_at=enqueued_at,
            attributes={},
            _acknowledge_fn=lambda mid=msg_id, suf=receipt_suffix: self._acknowledge(
                mid, suf
            ),
            _nack_fn=lambda t, mid=msg_id, suf=receipt_suffix: self._nack(mid, suf, t),
            _extend_fn=lambda t, mid=msg_id, suf=receipt_suffix: self._extend(
                mid, suf, t
            ),
        )

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
        result = self._scripts["acknowledge"](keys=keys, args=[msg_id, receipt_suffix])
        if result == 0:
            raise ReceiptHandleExpiredError(
                f"Message '{msg_id}' not found or receipt handle expired"
            )

    def _nack(self, msg_id: str, receipt_suffix: str, visibility_timeout: int) -> None:
        """Return message to queue.

        Args:
            msg_id: The message ID.
            receipt_suffix: The receipt handle suffix for validation.
            visibility_timeout: Seconds before message becomes visible again.

        Raises:
            ReceiptHandleExpiredError: If the receipt handle doesn't match
                the current delivery.
        """
        new_score = time.time() + visibility_timeout if visibility_timeout > 0 else 0
        keys = [self._keys.invisible, self._keys.pending, self._keys.meta]
        result = self._scripts["nack"](
            keys=keys, args=[msg_id, receipt_suffix, new_score]
        )
        if result == 0:
            raise ReceiptHandleExpiredError(
                f"Message '{msg_id}' not found or receipt handle expired"
            )

    def _extend(self, msg_id: str, receipt_suffix: str, timeout: int) -> None:
        """Extend visibility timeout.

        Args:
            msg_id: The message ID.
            receipt_suffix: The receipt handle suffix for validation.
            timeout: New visibility timeout in seconds from now.

        Raises:
            ReceiptHandleExpiredError: If the receipt handle doesn't match
                the current delivery.
        """
        new_score = time.time() + timeout
        result = self._scripts["extend"](
            keys=[self._keys.invisible, self._keys.meta],
            args=[msg_id, receipt_suffix, new_score],
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
        """Stop the reaper thread. Does not close the Redis client."""
        with self._lock:
            self._closed = True

        # Stop reaper thread
        self._stop_reaper.set()
        if self._reaper_thread is not None:  # pragma: no branch
            self._reaper_thread.join(timeout=2.0)

    @property
    def closed(self) -> bool:
        """Return True if mailbox has been closed."""
        return self._closed


__all__ = ["RedisMailbox"]

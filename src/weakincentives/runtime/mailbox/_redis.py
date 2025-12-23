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

"""Redis-backed mailbox implementation.

Supports both standalone Redis and Redis Cluster deployments.
Uses hash tags to ensure all keys for a queue land on the same shard.

Data Structures:

    {queue:name}:pending    # LIST - messages awaiting delivery (LPUSH/RPOP)
    {queue:name}:invisible  # ZSET - in-flight messages scored by expiry timestamp
    {queue:name}:data       # HASH - message ID → serialized message data
    {queue:name}:meta       # HASH - delivery counts and metadata

The ``{queue:name}`` prefix (with curly braces) is a Redis hash tag ensuring all
keys for a queue hash to the same slot. Required for Cluster mode; harmless in
standalone mode.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from ._types import (
    MailboxConnectionError,
    MailboxFullError,
    Message,
    ReceiptHandleExpiredError,
    SerializationError,
)

if TYPE_CHECKING:
    from typing import Any

    from redis import Redis
    from redis.cluster import RedisCluster

# Lua script for atomic receive operation
# KEYS: [pending, invisible, data, meta]
# ARGV: [visibility_expiry_timestamp]
# Returns: [msg_id, data_json, delivery_count] or nil
_RECEIVE_SCRIPT = """
local msg_id = redis.call('RPOP', KEYS[1])
if not msg_id then return nil end
redis.call('ZADD', KEYS[2], ARGV[1], msg_id)
local data = redis.call('HGET', KEYS[3], msg_id)
local count = redis.call('HINCRBY', KEYS[4], msg_id .. ':count', 1)
return {msg_id, data, count}
"""

# Lua script for atomic acknowledge operation
# KEYS: [invisible, data, meta]
# ARGV: [msg_id, receipt_handle]
# Returns: 1 if acknowledged, 0 if not found/expired
_ACKNOWLEDGE_SCRIPT = """
local current_handle = redis.call('HGET', KEYS[3], ARGV[1] .. ':handle')
if current_handle ~= ARGV[2] then return 0 end
redis.call('ZREM', KEYS[1], ARGV[1])
redis.call('HDEL', KEYS[2], ARGV[1])
redis.call('HDEL', KEYS[3], ARGV[1] .. ':count')
redis.call('HDEL', KEYS[3], ARGV[1] .. ':handle')
redis.call('HDEL', KEYS[3], ARGV[1] .. ':enqueued')
return 1
"""

# Lua script for atomic nack operation
# KEYS: [pending, invisible, meta]
# ARGV: [msg_id, receipt_handle, new_visibility_expiry (0 for immediate)]
# Returns: 1 if nacked, 0 if not found/expired
_NACK_SCRIPT = """
local current_handle = redis.call('HGET', KEYS[3], ARGV[1] .. ':handle')
if current_handle ~= ARGV[2] then return 0 end
redis.call('ZREM', KEYS[2], ARGV[1])
if tonumber(ARGV[3]) == 0 then
    redis.call('LPUSH', KEYS[1], ARGV[1])
else
    redis.call('ZADD', KEYS[2], ARGV[3], ARGV[1])
end
return 1
"""

# Lua script for extending visibility
# KEYS: [invisible, meta]
# ARGV: [msg_id, receipt_handle, new_expiry_timestamp]
# Returns: 1 if extended, 0 if not found/expired
_EXTEND_SCRIPT = """
local current_handle = redis.call('HGET', KEYS[2], ARGV[1] .. ':handle')
if current_handle ~= ARGV[2] then return 0 end
redis.call('ZADD', KEYS[1], 'XX', ARGV[3], ARGV[1])
return 1
"""

# Lua script for reaping expired messages
# KEYS: [pending, invisible, meta]
# ARGV: [current_timestamp, max_count]
# Returns: count of messages requeued
_REAP_SCRIPT = """
local expired = redis.call('ZRANGEBYSCORE', KEYS[2], '-inf', ARGV[1], 'LIMIT', 0, ARGV[2])
local count = 0
for _, msg_id in ipairs(expired) do
    redis.call('ZREM', KEYS[2], msg_id)
    redis.call('HDEL', KEYS[3], msg_id .. ':handle')
    redis.call('LPUSH', KEYS[1], msg_id)
    count = count + 1
end
return count
"""

# Lua script for purge operation
# KEYS: [pending, invisible, data, meta]
# Returns: approximate count of deleted messages
_PURGE_SCRIPT = """
local pending_count = redis.call('LLEN', KEYS[1])
local invisible_count = redis.call('ZCARD', KEYS[2])
redis.call('DEL', KEYS[1], KEYS[2], KEYS[3], KEYS[4])
return pending_count + invisible_count
"""


@dataclass
class _StoredMessage:
    """Internal representation of a stored message."""

    id: str
    body_json: str
    enqueued_at: str  # ISO format


def _serialize_body(body: object) -> str:
    """Serialize message body to JSON string."""
    try:
        return json.dumps(body)
    except (TypeError, ValueError) as e:
        raise SerializationError(f"Cannot serialize message body: {e}") from e


@dataclass(slots=True)
class RedisMailbox[T]:
    """Redis-backed mailbox implementation with visibility timeout semantics.

    Supports both standalone Redis and Redis Cluster. Auto-detects cluster mode
    from the client type.

    Characteristics:
    - Durable (persists across restarts if Redis is configured for persistence)
    - FIFO ordering guaranteed (within a single queue)
    - Exact message counts
    - Thread-safe

    Example::

        from redis import Redis
        from weakincentives.runtime.mailbox import RedisMailbox

        client = Redis(host="localhost", port=6379)
        mailbox: RedisMailbox[dict] = RedisMailbox(name="tasks", client=client)

        try:
            mailbox.send({"task": "process"})
            messages = mailbox.receive(visibility_timeout=60)
            for msg in messages:
                process(msg.body)
                msg.acknowledge()
        finally:
            mailbox.close()

    For Redis Cluster::

        from redis.cluster import RedisCluster

        client = RedisCluster(host="localhost", port=7000)
        mailbox = RedisMailbox(name="tasks", client=client)

    Args:
        name: Queue name. Used to derive Redis key names with hash tags.
        client: A Redis or RedisCluster client instance.
        reaper_interval: Seconds between visibility reaper runs. Default 1.0.
    """

    name: str
    client: Redis | RedisCluster
    reaper_interval: float = 1.0

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _closed: bool = field(default=False, repr=False, init=False)
    _reaper_thread: threading.Thread | None = field(
        default=None, repr=False, init=False
    )
    _stop_reaper: threading.Event = field(
        default_factory=threading.Event, repr=False, init=False
    )

    # Registered Lua scripts (initialized in __post_init__)
    _receive_script: Any = field(default=None, repr=False, init=False)
    _acknowledge_script: Any = field(default=None, repr=False, init=False)
    _nack_script: Any = field(default=None, repr=False, init=False)
    _extend_script: Any = field(default=None, repr=False, init=False)
    _reap_script: Any = field(default=None, repr=False, init=False)
    _purge_script: Any = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        """Register Lua scripts and start reaper thread."""
        self._register_scripts()
        self._start_reaper()

    def _register_scripts(self) -> None:
        """Register Lua scripts with Redis."""
        self._receive_script = self.client.register_script(_RECEIVE_SCRIPT)
        self._acknowledge_script = self.client.register_script(_ACKNOWLEDGE_SCRIPT)
        self._nack_script = self.client.register_script(_NACK_SCRIPT)
        self._extend_script = self.client.register_script(_EXTEND_SCRIPT)
        self._reap_script = self.client.register_script(_REAP_SCRIPT)
        self._purge_script = self.client.register_script(_PURGE_SCRIPT)

    def _start_reaper(self) -> None:
        """Start background thread to requeue expired messages."""
        self._reaper_thread = threading.Thread(
            target=self._reaper_loop,
            daemon=True,
            name=f"redis-mailbox-reaper-{self.name}",
        )
        self._reaper_thread.start()

    def _reaper_loop(self) -> None:
        """Background loop that checks for expired visibility timeouts."""
        while not self._stop_reaper.wait(timeout=self.reaper_interval):
            if self._closed:  # pragma: no cover - race condition
                break
            with contextlib.suppress(Exception):
                self._reap_expired()

    def _reap_expired(self) -> int:
        """Move expired messages from invisible back to pending.

        Returns:
            Number of messages requeued.
        """
        now = time.time()
        keys = [self._key("pending"), self._key("invisible"), self._key("meta")]
        result = self._reap_script(keys=keys, args=[now, 100])
        return int(result) if result else 0

    def _key(self, suffix: str) -> str:
        """Generate a Redis key with hash tag for cluster compatibility.

        Args:
            suffix: Key suffix (pending, invisible, data, meta).

        Returns:
            Full key name like ``{queue:myqueue}:pending``.
        """
        return f"{{queue:{self.name}}}:{suffix}"

    @property
    def closed(self) -> bool:
        """Return True if mailbox has been closed."""
        return self._closed

    def send(self, body: T, *, delay_seconds: int = 0) -> str:
        """Enqueue a message, optionally delaying visibility.

        Args:
            body: Message payload (must be JSON-serializable).
            delay_seconds: Seconds before message becomes visible (0-900).

        Returns:
            Message ID (unique within this queue).

        Raises:
            MailboxFullError: Queue capacity exceeded (Redis maxmemory).
            SerializationError: Body cannot be serialized.
            MailboxConnectionError: Cannot connect to Redis.
        """
        msg_id = str(uuid4())
        enqueued_at = datetime.now(UTC).isoformat()

        # Serialize the body
        body_json = _serialize_body(body)

        # Create stored message
        stored = _StoredMessage(id=msg_id, body_json=body_json, enqueued_at=enqueued_at)
        stored_json = json.dumps(
            {
                "id": stored.id,
                "body": stored.body_json,
                "enqueued_at": stored.enqueued_at,
            }
        )

        try:
            # Store message data
            self.client.hset(self._key("data"), msg_id, stored_json)
            self.client.hset(self._key("meta"), f"{msg_id}:enqueued", enqueued_at)

            if delay_seconds > 0:
                # Put in invisible set with delay as expiry score
                expiry = time.time() + delay_seconds
                self.client.zadd(self._key("invisible"), {msg_id: expiry})
            else:
                # Add to pending list (left side, RPOP from right)
                self.client.lpush(self._key("pending"), msg_id)
        except Exception as e:  # pragma: no cover - error handling
            # Check for OOM errors
            error_str = str(e).lower()
            if "oom" in error_str or "memory" in error_str:
                raise MailboxFullError(f"Redis out of memory: {e}") from e
            raise MailboxConnectionError(f"Redis operation failed: {e}") from e

        return msg_id

    def _parse_received_message(
        self, result: tuple[bytes | str, bytes | str, int]
    ) -> Message[T]:
        """Parse a received message from Redis script result."""
        msg_id, data_json, delivery_count = result
        msg_id = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
        data_json = data_json.decode() if isinstance(data_json, bytes) else data_json

        # Parse stored message and extract body
        stored_data = json.loads(data_json)
        body = json.loads(stored_data["body"])
        enqueued_at = datetime.fromisoformat(stored_data["enqueued_at"])

        # Generate and store receipt handle
        receipt_handle = str(uuid4())
        self.client.hset(self._key("meta"), f"{msg_id}:handle", receipt_handle)

        return Message(
            id=msg_id,
            body=body,
            receipt_handle=receipt_handle,
            delivery_count=int(delivery_count),
            enqueued_at=enqueued_at,
            attributes={},
            _acknowledge_fn=lambda mid=msg_id, rh=receipt_handle: self._acknowledge(
                mid, rh
            ),
            _nack_fn=lambda t, mid=msg_id, rh=receipt_handle: self._nack(mid, rh, t),
            _extend_fn=lambda t, mid=msg_id, rh=receipt_handle: self._extend(
                mid, rh, t
            ),
        )

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

        Returns:
            Sequence of messages (may be empty). Returns empty if mailbox closed.
        """
        if self._closed:
            return []

        max_messages = min(max(1, max_messages), 10)
        deadline = time.monotonic() + wait_time_seconds
        messages: list[Message[T]] = []
        keys = [
            self._key("pending"),
            self._key("invisible"),
            self._key("data"),
            self._key("meta"),
        ]

        while len(messages) < max_messages:
            if self._closed:  # pragma: no cover - race condition
                break

            try:
                expiry = time.time() + visibility_timeout
                result = self._receive_script(keys=keys, args=[expiry])
            except Exception as e:  # pragma: no cover - error handling
                raise MailboxConnectionError(f"Redis receive failed: {e}") from e

            if result:
                messages.append(self._parse_received_message(result))
            elif wait_time_seconds <= 0:
                break
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.1, remaining))

        return messages

    def _acknowledge(self, msg_id: str, receipt_handle: str) -> None:
        """Delete message from queue."""
        keys = [self._key("invisible"), self._key("data"), self._key("meta")]
        try:
            result = self._acknowledge_script(keys=keys, args=[msg_id, receipt_handle])
        except Exception as e:  # pragma: no cover - error handling
            raise MailboxConnectionError(f"Redis acknowledge failed: {e}") from e

        if not result:
            raise ReceiptHandleExpiredError(
                f"Receipt handle '{receipt_handle}' not found or expired"
            )

    def _nack(self, msg_id: str, receipt_handle: str, visibility_timeout: int) -> None:
        """Return message to queue."""
        new_expiry = 0 if visibility_timeout <= 0 else time.time() + visibility_timeout
        keys = [self._key("pending"), self._key("invisible"), self._key("meta")]
        try:
            result = self._nack_script(
                keys=keys, args=[msg_id, receipt_handle, new_expiry]
            )
        except Exception as e:  # pragma: no cover - error handling
            raise MailboxConnectionError(f"Redis nack failed: {e}") from e

        if not result:
            raise ReceiptHandleExpiredError(
                f"Receipt handle '{receipt_handle}' not found or expired"
            )

    def _extend(self, msg_id: str, receipt_handle: str, timeout: int) -> None:
        """Extend visibility timeout."""
        new_expiry = time.time() + timeout
        keys = [self._key("invisible"), self._key("meta")]
        try:
            result = self._extend_script(
                keys=keys, args=[msg_id, receipt_handle, new_expiry]
            )
        except Exception as e:  # pragma: no cover - error handling
            raise MailboxConnectionError(f"Redis extend failed: {e}") from e

        if not result:
            raise ReceiptHandleExpiredError(
                f"Receipt handle '{receipt_handle}' not found or expired"
            )

    def purge(self) -> int:
        """Delete all messages from the queue.

        Returns:
            Count of messages deleted.
        """
        keys = [
            self._key("pending"),
            self._key("invisible"),
            self._key("data"),
            self._key("meta"),
        ]
        try:
            result = self._purge_script(keys=keys, args=[])
            return int(result) if result else 0
        except Exception as e:  # pragma: no cover - error handling
            raise MailboxConnectionError(f"Redis purge failed: {e}") from e

    def approximate_count(self) -> int:
        """Return exact number of messages in the queue.

        For RedisMailbox, this count is exact, not approximate.
        """
        try:
            # Redis returns int for sync clients; cast needed for type checker
            pending: int = self.client.llen(self._key("pending"))  # pyright: ignore[reportAssignmentType]
            invisible: int = self.client.zcard(self._key("invisible"))  # pyright: ignore[reportAssignmentType]
            return pending + invisible
        except Exception as e:  # pragma: no cover - error handling
            raise MailboxConnectionError(f"Redis approximate_count failed: {e}") from e

    def close(self) -> None:
        """Stop the reaper thread.

        Note: This does NOT close the Redis client connection. The caller
        is responsible for managing the Redis client lifecycle.
        """
        with self._lock:
            self._closed = True

        self._stop_reaper.set()
        if self._reaper_thread is not None:  # pragma: no branch
            self._reaper_thread.join(timeout=2.0)


__all__ = ["RedisMailbox"]

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
import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, is_dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast, get_origin
from uuid import uuid4

from weakincentives.formal import formal_spec
from weakincentives.runtime.mailbox import (
    CompositeResolver,
    Mailbox,
    MailboxConnectionError,
    MailboxFullError,
    MailboxResolutionError,
    Message,
    ReceiptHandleExpiredError,
    SerializationError,
    validate_visibility_timeout,
    validate_wait_time,
)
from weakincentives.serde import dump, parse

from ._lua_scripts import (
    LUA_ACKNOWLEDGE,
    LUA_EXTEND,
    LUA_NACK,
    LUA_PURGE,
    LUA_REAP,
    LUA_RECEIVE,
    LUA_SEND,
)
from ._redis_spec import REDIS_MAILBOX_SPEC_KWARGS

if TYPE_CHECKING:
    from redis import Redis
    from redis.cluster import RedisCluster

    from weakincentives.runtime.mailbox import MailboxResolver

_LOGGER = logging.getLogger(__name__)

# Default TTL for Redis keys: 3 days in seconds.
# Keys are refreshed on each operation, so active queues stay alive indefinitely.
# Set to 0 to disable TTL expiration.
DEFAULT_TTL_SECONDS: int = 259200  # 3 days = 3 * 24 * 60 * 60


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
            msg.reply(result)  # Resolves reply_to -> new RedisMailbox with same client
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
@formal_spec(**REDIS_MAILBOX_SPEC_KWARGS)
class RedisMailbox[T, R]:
    """Redis-backed mailbox with SQS-compatible visibility timeout semantics.

    This implementation is formally verified against the embedded TLA+ specification.
    The specification can be extracted and model-checked with ``make verify-formal``.

    Supports both standalone Redis and Redis Cluster deployments. Uses Lua scripts
    for atomic operations and a background reaper thread for visibility timeout
    management.

    Type parameters:
        T: Message body type.
        R: Reply type (None if no replies expected).

    Data structures::

        {queue:name}:pending    # LIST - messages awaiting delivery (LPUSH/RPOP)
        {queue:name}:invisible  # ZSET - in-flight messages scored by expiry timestamp
        {queue:name}:data       # HASH - message ID -> serialized message body
        {queue:name}:meta       # HASH - message ID:count -> delivery count,
                                #        message ID:enqueued -> enqueued timestamp,
                                #        message ID:handle -> current receipt handle suffix
                                #        message ID:reply_to -> reply destination

    Formal Specification:
        Key invariants verified by the ``@formal_spec`` decorator:

        - INV-1: Message State Exclusivity (pending XOR invisible XOR deleted)
        - INV-2-3: Receipt Handle Validity (stale handles rejected)
        - INV-4/4b: Delivery Count Monotonicity (counts never decrease)
        - INV-5: No Message Loss (messages tracked until acknowledged)
        - INV-7: Handle Uniqueness (each delivery gets unique handle)
        - INV-8: Pending No Duplicates (no duplicate IDs in pending queue)
        - INV-9: Data Integrity (queued messages have associated data)

        See ``specs/VERIFICATION.md`` for complete verification documentation.
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
    """Resolver for reconstructing reply mailboxes from names stored in Redis.
    If None, a default resolver using RedisMailboxFactory is created automatically,
    enabling any queue name to be resolved to a mailbox on the same Redis server."""

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
        self._scripts["send"] = self.client.register_script(LUA_SEND)
        self._scripts["receive"] = self.client.register_script(LUA_RECEIVE)
        self._scripts["acknowledge"] = self.client.register_script(LUA_ACKNOWLEDGE)
        self._scripts["nack"] = self.client.register_script(LUA_NACK)
        self._scripts["extend"] = self.client.register_script(LUA_EXTEND)
        self._scripts["reap"] = self.client.register_script(LUA_REAP)
        self._scripts["purge"] = self.client.register_script(LUA_PURGE)

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
                # Use parse() for dataclass types (including generic aliases)
                origin = get_origin(self.body_type)
                if is_dataclass(origin if origin is not None else self.body_type):
                    return parse(self.body_type, json_data)
                # For primitive types (str, int, etc.), construct directly
                body_type: Any = self.body_type
                return body_type(json_data)
            # Without a type hint, return raw JSON data
            return cast(T, json_data)
        except Exception as e:
            raise SerializationError(f"Failed to deserialize message body: {e}") from e

    def send(self, body: T, *, reply_to: Mailbox[R, None] | None = None) -> str:
        """Enqueue a message.

        This operation is atomic via Lua script, ensuring no partial state
        on failure.

        Args:
            body: Message payload (must be serializable via serde).
            reply_to: Mailbox instance for receiving replies. The mailbox name
                is serialized to Redis and resolved on receive.

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
                reply_to.name if reply_to else "",  # Serialize mailbox name
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

        # Parse reply_to name and resolve to mailbox
        reply_to: Mailbox[R, None] | None = None
        if reply_to_raw:
            reply_to_name = (
                reply_to_raw.decode("utf-8")
                if isinstance(reply_to_raw, bytes)
                else str(reply_to_raw)
            )
            # Resolve the mailbox from the stored name
            # If resolution fails, reply_to stays None and Message.reply() will raise
            # ReplyNotAvailableError with a clear message
            if self.reply_resolver is not None:
                try:
                    reply_to = self.reply_resolver.resolve(reply_to_name)
                except MailboxResolutionError:
                    _LOGGER.debug(
                        "Failed to resolve reply_to mailbox '%s' for queue '%s'",
                        reply_to_name,
                        self.name,
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

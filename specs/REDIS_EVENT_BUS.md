# Redis Event Bus Specification

This specification describes the Redis-based `EventBus` implementation for
distributed session event broadcast and main loop orchestration.

## Overview

The `RedisEventBus` provides a distributed, durable alternative to
`InProcessEventBus`. It enables:

- **Cross-process event delivery**: Multiple workers share session state
- **Horizontal scaling**: Main loops distributed across nodes
- **Event persistence**: Optional event durability via Redis Streams
- **Replay capability**: Reconstruct session state from event history

## Design Goals

1. **Protocol compatibility**: Drop-in replacement for `InProcessEventBus`
2. **Deterministic ordering**: Events processed in consistent order
3. **At-least-once delivery**: No silent event loss
4. **Graceful degradation**: Handle Redis unavailability
5. **Observable**: Metrics and health checks exposed

## Non-Goals

- Exactly-once semantics (idempotency is caller responsibility)
- Multi-tenant isolation (use separate Redis instances)
- Event schema evolution (handled by serde layer)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Application Process                          │
│                                                                     │
│  ┌──────────┐    ┌──────────────────┐    ┌───────────────────────┐ │
│  │ Session  │◄──►│  RedisEventBus   │◄──►│  RedisStreamClient    │ │
│  └──────────┘    └──────────────────┘    └───────────────────────┘ │
│                           │                         │               │
│  ┌──────────┐             │                         │               │
│  │ MainLoop │◄────────────┘                         │               │
│  └──────────┘                                       │               │
└─────────────────────────────────────────────────────│───────────────┘
                                                      │
                                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           Redis Server                              │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  Stream: wink:events:{session_id}                               ││
│  │  ┌──────────┬──────────┬──────────┬──────────┬──────────┐       ││
│  │  │ entry-1  │ entry-2  │ entry-3  │ entry-4  │ entry-5  │  ...  ││
│  │  │ Rendered │ Invoked  │ Invoked  │ Invoked  │ Executed │       ││
│  │  └──────────┴──────────┴──────────┴──────────┴──────────┘       ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  Stream: wink:control:{namespace}                               ││
│  │  ┌──────────┬──────────┬──────────┐                             ││
│  │  │ Request  │ Complete │ Request  │  ...                        ││
│  │  └──────────┴──────────┴──────────┘                             ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  Consumer Groups                                                ││
│  │  • wink:cg:{session_id}:telemetry                               ││
│  │  • wink:cg:{namespace}:control                                  ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

## Redis Primitives

### Streams

Redis Streams provide ordered, persistent event storage with consumer group
support:

| Stream Pattern | Purpose | Retention |
|----------------|---------|-----------|
| `wink:events:{session_id}` | Telemetry events per session | Configurable TTL |
| `wink:control:{namespace}` | MainLoop control events | Configurable TTL |

### Consumer Groups

Consumer groups enable competing consumers with acknowledgment:

| Group Pattern | Stream | Purpose |
|---------------|--------|---------|
| `wink:cg:{session_id}:telemetry` | `wink:events:{session_id}` | Session telemetry processing |
| `wink:cg:{namespace}:control` | `wink:control:{namespace}` | MainLoop request distribution |

### Keys

Additional keys for coordination:

| Key Pattern | Type | Purpose |
|-------------|------|---------|
| `wink:session:{session_id}:meta` | Hash | Session metadata |
| `wink:lock:{resource}` | String | Distributed locks |

## Event Serialization

Events are serialized using the existing `weakincentives.serde` module:

```python
from weakincentives.serde import to_dict, from_dict

# Serialize event for Redis
def serialize_event(event: object) -> dict[str, Any]:
    return {
        "type": f"{type(event).__module__}.{type(event).__qualname__}",
        "payload": to_dict(event),
        "timestamp": datetime.now(UTC).isoformat(),
    }

# Deserialize event from Redis
def deserialize_event(data: dict[str, Any]) -> object:
    event_type = _resolve_type(data["type"])
    return from_dict(data["payload"], event_type)
```

### Type Resolution

Event types are resolved from fully qualified names at deserialization time.
Only types in the allowlist are permitted:

```python
ALLOWED_EVENT_TYPES: frozenset[str] = frozenset({
    "weakincentives.runtime.events.PromptRendered",
    "weakincentives.runtime.events.ToolInvoked",
    "weakincentives.runtime.events.PromptExecuted",
    "weakincentives.runtime.main_loop.MainLoopRequest",
    "weakincentives.runtime.main_loop.MainLoopCompleted",
    "weakincentives.runtime.main_loop.MainLoopFailed",
})
```

Custom event types can be registered via configuration.

## Configuration

```python
from dataclasses import dataclass
from datetime import timedelta

@dataclass(slots=True, frozen=True)
class RedisEventBusConfig:
    """Configuration for Redis-backed event bus."""

    # Connection
    redis_url: str = "redis://localhost:6379/0"
    connection_pool_size: int = 10
    socket_timeout: timedelta = timedelta(seconds=5)
    socket_connect_timeout: timedelta = timedelta(seconds=5)

    # Namespacing
    key_prefix: str = "wink"
    namespace: str = "default"

    # Stream behavior
    stream_max_len: int | None = 10_000  # MAXLEN for XADD
    stream_ttl: timedelta | None = timedelta(hours=24)

    # Consumer behavior
    consumer_name: str | None = None  # Auto-generated if None
    block_timeout: timedelta = timedelta(seconds=1)
    batch_size: int = 100
    ack_deadline: timedelta = timedelta(minutes=5)

    # Retry behavior
    max_retries: int = 3
    retry_backoff: timedelta = timedelta(milliseconds=100)

    # Fallback behavior
    fallback_to_in_process: bool = True
    health_check_interval: timedelta = timedelta(seconds=30)

    # Custom event types
    additional_event_types: frozenset[str] = frozenset()
```

## RedisEventBus Implementation

### Protocol Compliance

```python
from weakincentives.runtime.events import EventBus, EventHandler, PublishResult

class RedisEventBus(EventBus):
    """Redis Streams-backed event bus implementation."""

    def __init__(
        self,
        config: RedisEventBusConfig,
        *,
        session_id: UUID | None = None,
    ) -> None:
        self._config = config
        self._session_id = session_id
        self._client = self._create_client()
        self._local_handlers: dict[type[object], list[EventHandler]] = {}
        self._consumer_tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = RLock()

    def subscribe(
        self,
        event_type: type[object],
        handler: EventHandler,
    ) -> None:
        """Register handler for event type.

        Local handlers are invoked synchronously when events arrive
        from Redis. For distributed processing, handlers should be
        registered on all consumer instances.
        """
        with self._lock:
            handlers = self._local_handlers.setdefault(event_type, [])
            handlers.append(handler)
            self._ensure_consumer_for_type(event_type)

    def unsubscribe(
        self,
        event_type: type[object],
        handler: EventHandler,
    ) -> bool:
        """Remove handler. Returns True if found and removed."""
        with self._lock:
            handlers = self._local_handlers.get(event_type, [])
            try:
                handlers.remove(handler)
                return True
            except ValueError:
                return False

    def publish(self, event: object) -> PublishResult:
        """Publish event to Redis Stream.

        The event is added to the appropriate stream based on its type.
        Local handlers are invoked after successful persistence.
        """
        stream_key = self._stream_key_for_event(event)
        serialized = serialize_event(event)

        try:
            entry_id = self._client.xadd(
                stream_key,
                serialized,
                maxlen=self._config.stream_max_len,
            )
        except RedisError as exc:
            return self._handle_publish_failure(event, exc)

        # Invoke local handlers synchronously
        return self._dispatch_to_local_handlers(event, entry_id)
```

### Stream Key Resolution

```python
def _stream_key_for_event(self, event: object) -> str:
    """Determine stream key based on event type."""
    prefix = self._config.key_prefix

    if isinstance(event, (PromptRendered, ToolInvoked, PromptExecuted)):
        # Telemetry events go to session-specific stream
        session_id = getattr(event, "session_id", None) or self._session_id
        if session_id is None:
            raise ValueError("Telemetry event requires session_id")
        return f"{prefix}:events:{session_id}"

    if isinstance(event, (MainLoopRequest, MainLoopCompleted, MainLoopFailed)):
        # Control events go to namespace stream
        return f"{prefix}:control:{self._config.namespace}"

    # Unknown event types go to a catch-all stream
    return f"{prefix}:events:unknown"
```

### Consumer Loop

```python
def _ensure_consumer_for_type(self, event_type: type[object]) -> None:
    """Start consumer task for event type's stream if not running."""
    stream_key = self._stream_key_for_type(event_type)

    if stream_key in self._consumer_tasks:
        return

    group_name = self._consumer_group_for_stream(stream_key)
    consumer_name = self._config.consumer_name or f"consumer-{uuid4()}"

    # Create consumer group if needed
    try:
        self._client.xgroup_create(
            stream_key,
            group_name,
            id="0",
            mkstream=True,
        )
    except ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise

    # Start consumer task
    task = asyncio.create_task(
        self._consume_stream(stream_key, group_name, consumer_name)
    )
    self._consumer_tasks[stream_key] = task

async def _consume_stream(
    self,
    stream_key: str,
    group_name: str,
    consumer_name: str,
) -> None:
    """Consume events from stream and dispatch to handlers."""
    while True:
        try:
            entries = await self._client.xreadgroup(
                group_name,
                consumer_name,
                {stream_key: ">"},
                count=self._config.batch_size,
                block=int(self._config.block_timeout.total_seconds() * 1000),
            )

            for stream, messages in entries:
                for entry_id, data in messages:
                    try:
                        event = deserialize_event(data)
                        self._dispatch_to_local_handlers(event, entry_id)
                        await self._client.xack(stream_key, group_name, entry_id)
                    except Exception:
                        logger.exception(
                            "Failed to process event",
                            entry_id=entry_id,
                            stream=stream_key,
                        )
        except asyncio.CancelledError:
            break
        except RedisError:
            logger.exception("Redis consumer error", stream=stream_key)
            await asyncio.sleep(self._config.retry_backoff.total_seconds())
```

## Session Integration

### Factory Function

```python
from weakincentives.runtime import Session
from weakincentives.runtime.events.redis import RedisEventBus, RedisEventBusConfig

def create_redis_session(
    config: RedisEventBusConfig,
    *,
    session_id: UUID | None = None,
    **kwargs: Any,
) -> Session:
    """Create session with Redis-backed event bus.

    Args:
        config: Redis event bus configuration
        session_id: Optional session ID (generated if None)
        **kwargs: Additional Session arguments

    Returns:
        Session configured with RedisEventBus
    """
    session_id = session_id or uuid4()
    bus = RedisEventBus(config, session_id=session_id)
    return Session(bus=bus, session_id=session_id, **kwargs)
```

### Session Recovery

```python
def recover_session(
    config: RedisEventBusConfig,
    session_id: UUID,
) -> Session:
    """Recover session state from Redis event history.

    Replays all events from the session's stream to reconstruct state.
    """
    bus = RedisEventBus(config, session_id=session_id)
    session = Session(bus=bus, session_id=session_id)

    # Read entire stream
    stream_key = f"{config.key_prefix}:events:{session_id}"
    entries = bus._client.xrange(stream_key, "-", "+")

    # Replay events (handlers already subscribed by Session.__init__)
    for entry_id, data in entries:
        event = deserialize_event(data)
        session._dispatch_event_locally(event)

    return session
```

## MainLoop Integration

### Distributed MainLoop

```python
class DistributedMainLoop[UserRequestT, OutputT](MainLoop[UserRequestT, OutputT]):
    """MainLoop variant for distributed execution via Redis."""

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[OutputT],
        redis_config: RedisEventBusConfig,
        config: MainLoopConfig | None = None,
    ) -> None:
        bus = RedisEventBus(redis_config)
        super().__init__(adapter=adapter, bus=bus, config=config)
        self._redis_config = redis_config

    def handle_request(self, event: object) -> None:
        """Handle request with Redis-backed session."""
        request_event: MainLoopRequest[UserRequestT] = event

        try:
            # Create Redis-backed session for this request
            session = create_redis_session(
                self._redis_config,
                session_id=uuid4(),
            )

            response = self._execute_with_session(
                request_event.request,
                session,
                budget=request_event.budget,
                deadline=request_event.deadline,
                resources=request_event.resources,
            )

            completed = MainLoopCompleted[OutputT](
                request_id=request_event.request_id,
                response=response,
                session_id=session.session_id,
            )
            _ = self._bus.publish(completed)

        except Exception as exc:
            failed = MainLoopFailed(
                request_id=request_event.request_id,
                error=exc,
                session_id=None,
            )
            _ = self._bus.publish(failed)
            raise
```

### Request Distribution

Multiple workers can compete for requests using consumer groups:

```python
# Worker 1
loop1 = DistributedMainLoop(
    adapter=adapter,
    redis_config=RedisEventBusConfig(
        consumer_name="worker-1",
        namespace="code-review",
    ),
)

# Worker 2 (different process/node)
loop2 = DistributedMainLoop(
    adapter=adapter,
    redis_config=RedisEventBusConfig(
        consumer_name="worker-2",
        namespace="code-review",
    ),
)

# Publish request (any worker will handle it)
bus.publish(MainLoopRequest(request=ReviewRequest(...)))
```

## Error Handling

### Publish Failures

```python
def _handle_publish_failure(
    self,
    event: object,
    exc: RedisError,
) -> PublishResult:
    """Handle Redis publish failure."""
    logger.error(
        "Failed to publish event to Redis",
        event_type=type(event).__name__,
        error=str(exc),
    )

    if self._config.fallback_to_in_process:
        # Fall back to local dispatch only
        return self._dispatch_to_local_handlers(event, entry_id=None)

    # Return failure result
    return PublishResult(
        event=event,
        handlers_invoked=(),
        errors=(
            HandlerFailure(
                handler=lambda e: None,  # Placeholder
                error=exc,
            ),
        ),
        handled_count=0,
    )
```

### Consumer Failures

Events that fail processing remain in the pending entries list (PEL). A
background task reclaims stale entries:

```python
async def _reclaim_stale_entries(self, stream_key: str, group_name: str) -> None:
    """Reclaim entries from dead consumers."""
    pending = await self._client.xpending_range(
        stream_key,
        group_name,
        min="-",
        max="+",
        count=100,
    )

    now = datetime.now(UTC)
    for entry in pending:
        idle_time = timedelta(milliseconds=entry["time_since_delivered"])
        if idle_time > self._config.ack_deadline:
            # Claim entry for this consumer
            claimed = await self._client.xclaim(
                stream_key,
                group_name,
                self._consumer_name,
                min_idle_time=int(self._config.ack_deadline.total_seconds() * 1000),
                message_ids=[entry["message_id"]],
            )
            for entry_id, data in claimed:
                try:
                    event = deserialize_event(data)
                    self._dispatch_to_local_handlers(event, entry_id)
                    await self._client.xack(stream_key, group_name, entry_id)
                except Exception:
                    logger.exception("Failed to process reclaimed entry")
```

### Circuit Breaker

```python
from weakincentives.dbc import invariant

@dataclass(slots=True)
class CircuitBreaker:
    """Circuit breaker for Redis connection."""

    failure_threshold: int = 5
    recovery_timeout: timedelta = timedelta(seconds=30)

    _failures: int = 0
    _last_failure: datetime | None = None
    _state: Literal["closed", "open", "half-open"] = "closed"

    @invariant(lambda self: self._failures >= 0)
    def record_failure(self) -> None:
        self._failures += 1
        self._last_failure = datetime.now(UTC)
        if self._failures >= self.failure_threshold:
            self._state = "open"

    def record_success(self) -> None:
        self._failures = 0
        self._state = "closed"

    def allow_request(self) -> bool:
        if self._state == "closed":
            return True
        if self._state == "open":
            if self._last_failure is None:
                return True
            elapsed = datetime.now(UTC) - self._last_failure
            if elapsed >= self.recovery_timeout:
                self._state = "half-open"
                return True
            return False
        # half-open: allow single request
        return True
```

## Health Checks

```python
@dataclass(slots=True, frozen=True)
class RedisEventBusHealth:
    """Health status for Redis event bus."""

    connected: bool
    latency_ms: float | None
    stream_lengths: dict[str, int]
    pending_counts: dict[str, int]
    consumer_lag: dict[str, int]
    last_check: datetime

class RedisEventBus:
    async def health_check(self) -> RedisEventBusHealth:
        """Check Redis connection and stream health."""
        start = datetime.now(UTC)

        try:
            # Ping Redis
            await self._client.ping()
            latency = (datetime.now(UTC) - start).total_seconds() * 1000

            # Get stream lengths
            stream_lengths = {}
            pending_counts = {}
            for stream_key in self._consumer_tasks:
                info = await self._client.xinfo_stream(stream_key)
                stream_lengths[stream_key] = info["length"]

                groups = await self._client.xinfo_groups(stream_key)
                for group in groups:
                    pending_counts[f"{stream_key}:{group['name']}"] = group["pending"]

            return RedisEventBusHealth(
                connected=True,
                latency_ms=latency,
                stream_lengths=stream_lengths,
                pending_counts=pending_counts,
                consumer_lag={},  # Computed from pending
                last_check=datetime.now(UTC),
            )
        except RedisError as exc:
            return RedisEventBusHealth(
                connected=False,
                latency_ms=None,
                stream_lengths={},
                pending_counts={},
                consumer_lag={},
                last_check=datetime.now(UTC),
            )
```

## Testing

### Unit Tests

Use `fakeredis` for unit testing:

```python
import fakeredis
import pytest

@pytest.fixture
def fake_redis() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis()

@pytest.fixture
def redis_bus(fake_redis: fakeredis.FakeRedis) -> RedisEventBus:
    config = RedisEventBusConfig()
    bus = RedisEventBus(config)
    bus._client = fake_redis  # Inject fake client
    return bus

def test_publish_subscribe(redis_bus: RedisEventBus) -> None:
    received: list[object] = []

    def handler(event: object) -> None:
        received.append(event)

    redis_bus.subscribe(ToolInvoked, handler)

    event = ToolInvoked(
        prompt_name="test",
        adapter=AdapterName.OPENAI,
        name="my_tool",
        params={},
        result=ToolResult(message="ok", success=True),
        session_id=uuid4(),
        created_at=datetime.now(UTC),
    )

    result = redis_bus.publish(event)

    assert result.ok
    assert len(received) == 1
    assert received[0] == event
```

### Integration Tests

```python
import pytest
from testcontainers.redis import RedisContainer

@pytest.fixture(scope="module")
def redis_container() -> Generator[RedisContainer, None, None]:
    with RedisContainer() as redis:
        yield redis

@pytest.fixture
def redis_config(redis_container: RedisContainer) -> RedisEventBusConfig:
    return RedisEventBusConfig(
        redis_url=redis_container.get_connection_url(),
    )

@pytest.mark.integration
def test_distributed_session(redis_config: RedisEventBusConfig) -> None:
    """Test session state replication across processes."""
    session_id = uuid4()

    # Create session in "process 1"
    session1 = create_redis_session(redis_config, session_id=session_id)

    # Simulate event in session1
    event = ToolInvoked(...)
    session1.event_bus.publish(event)

    # Recover session in "process 2"
    session2 = recover_session(redis_config, session_id)

    # Verify state replicated
    assert session2[ToolInvoked].latest() == event
```

## Migration Guide

### From InProcessEventBus

1. **Add Redis dependency**:
   ```bash
   uv add redis[hiredis]
   ```

2. **Update session creation**:
   ```python
   # Before
   from weakincentives.runtime import Session, InProcessEventBus

   bus = InProcessEventBus()
   session = Session(bus=bus)

   # After
   from weakincentives.runtime.events.redis import (
       RedisEventBus,
       RedisEventBusConfig,
       create_redis_session,
   )

   config = RedisEventBusConfig(redis_url="redis://localhost:6379/0")
   session = create_redis_session(config)
   ```

3. **Update MainLoop**:
   ```python
   # Before
   bus = InProcessEventBus()
   loop = MyMainLoop(adapter=adapter, bus=bus)

   # After
   config = RedisEventBusConfig(namespace="my-app")
   loop = DistributedMainLoop(adapter=adapter, redis_config=config)
   ```

### Backward Compatibility

The `RedisEventBus` implements the `EventBus` protocol exactly. Code that
accepts `EventBus` works unchanged:

```python
def process_events(bus: EventBus) -> None:
    # Works with both InProcessEventBus and RedisEventBus
    bus.subscribe(ToolInvoked, my_handler)
```

## Performance Considerations

### Latency

Redis adds network latency to every publish operation. For latency-sensitive
workloads:

- Use Redis on the same machine or low-latency network
- Enable pipelining for batch publishes
- Consider `fallback_to_in_process=True` for non-critical events

### Throughput

Redis Streams support high throughput but can become bottlenecks:

- Use `stream_max_len` to cap memory usage
- Partition high-volume streams by session or namespace
- Monitor pending entry counts for backpressure

### Memory

Event streams consume Redis memory:

- Set appropriate `stream_max_len` (default: 10,000 entries)
- Configure `stream_ttl` for automatic cleanup
- Use `XTRIM` periodically for explicit cleanup

## Security

### Connection Security

```python
config = RedisEventBusConfig(
    redis_url="rediss://user:password@redis.example.com:6380/0",
    # TLS enabled via rediss:// scheme
)
```

### Access Control

Use Redis ACL to restrict access:

```redis
ACL SETUSER wink-app on >password ~wink:* +@stream +@connection
```

## Observability

### Metrics

The `RedisEventBus` exposes metrics via structured logging:

| Metric | Type | Description |
|--------|------|-------------|
| `redis_event_bus.publish.count` | Counter | Events published |
| `redis_event_bus.publish.latency` | Histogram | Publish latency |
| `redis_event_bus.publish.error` | Counter | Publish failures |
| `redis_event_bus.consume.count` | Counter | Events consumed |
| `redis_event_bus.consume.latency` | Histogram | Processing latency |
| `redis_event_bus.pending.count` | Gauge | Pending entries |

### Tracing

Events include correlation IDs for distributed tracing:

```python
@FrozenDataclass()
class ToolInvoked:
    # ... existing fields ...
    event_id: UUID = field(default_factory=uuid4)
    trace_id: str | None = None  # Propagated from request context
    span_id: str | None = None
```

## Future Considerations

### Redis Cluster

For horizontal scaling beyond single Redis:

- Hash tags ensure session streams stay on same shard: `wink:{session_id}:events`
- Consumer groups work within single shard
- Cross-shard operations require coordination

### Event Compaction

Long-running sessions accumulate events. Future work:

- Snapshot session state periodically
- Compact events into snapshots
- Prune old events after snapshot

### Multi-Region

For geo-distributed deployments:

- Redis replication for read scaling
- Active-active with conflict resolution
- Event ordering guarantees across regions

# Dead Letter Queue Specification

## Purpose

Dead Letter Queues (DLQs) capture messages that cannot be processed after
repeated attempts. This prevents poison messages from blocking queues while
preserving them for inspection and remediation.

```python
# MainLoop with DLQ
main_loop = MainLoop(
    adapter=adapter,
    requests=requests,
    dlq=DLQPolicy(
        mailbox=dead_letters,
        max_delivery_count=5,
    ),
)

# Messages that fail 5 times are sent to dead_letters instead of retrying
```

**Use DLQ for:** Poison message isolation, failure forensics, manual remediation,
retry orchestration.

**Not for:** Transient failures (use visibility backoff), rate limiting (use
throttling), validation errors (reject immediately).

## Core Types

### DLQPolicy

Policy for dead-letter behavior:

```python
@dataclass(slots=True, frozen=True)
class DLQPolicy[T, R]:
    """Dead letter queue policy.

    Combines destination mailbox with decision logic for when to
    dead-letter failed messages. Subclass to customize behavior.

    Type parameters:
        T: Original message body type.
        R: Original reply type.
    """

    mailbox: Mailbox[DeadLetter[T], None]
    """Destination for dead-lettered messages."""

    max_delivery_count: int = 5
    """Maximum delivery attempts before dead-lettering.

    After this many receive() calls without acknowledge(), the message
    is sent to the DLQ and acknowledged from the source queue.
    """

    include_errors: frozenset[type[Exception]] | None = None
    """Exception types that trigger immediate dead-lettering.

    If set, these exceptions bypass retry and dead-letter immediately.
    None means all exceptions follow the delivery count threshold.
    """

    exclude_errors: frozenset[type[Exception]] | None = None
    """Exception types that never dead-letter.

    These exceptions always retry (respecting visibility backoff).
    Useful for transient network errors that should keep retrying.
    """

    def should_dead_letter(self, message: Message[T, Any], error: Exception) -> bool:
        """Determine if the message should be dead-lettered.

        Override this method for custom dead-letter logic.

        Args:
            message: The failed message.
            error: The exception that caused the failure.

        Returns:
            True to dead-letter, False to retry with backoff.
        """
        error_type = type(error)

        # Excluded errors never dead-letter
        if self.exclude_errors and error_type in self.exclude_errors:
            return False

        # Included errors always dead-letter immediately
        if self.include_errors and error_type in self.include_errors:
            return True

        # Otherwise, check delivery count threshold
        return message.delivery_count >= self.max_delivery_count
```

### DeadLetter

Envelope preserving the original message with failure context:

```python
@dataclass(slots=True, frozen=True)
class DeadLetter[T]:
    """Dead-lettered message with failure metadata."""

    message_id: str
    """Original message ID."""

    body: T
    """Original message body."""

    source_mailbox: str
    """Name of the mailbox the message came from."""

    delivery_count: int
    """Number of delivery attempts before dead-lettering."""

    last_error: str
    """String representation of the final error."""

    last_error_type: str
    """Fully qualified type name of the final error."""

    dead_lettered_at: datetime
    """Timestamp when the message was dead-lettered."""

    first_received_at: datetime
    """Timestamp of the first delivery attempt."""

    request_id: UUID | None = None
    """Request ID if the body is a MainLoopRequest or EvalRequest."""

    reply_to: str | None = None
    """Original reply_to mailbox name, if any."""

    trace_id: str | None = None
    """Trace ID for distributed tracing correlation."""
```

## MainLoop Integration

### Configuration

MainLoop accepts an optional DLQ configuration:

```python
class MainLoop[UserRequestT, OutputT](ABC):
    def __init__(
        self,
        *,
        adapter: ProviderAdapter[OutputT],
        requests: Mailbox[MainLoopRequest[UserRequestT], MainLoopResult[OutputT]],
        resources: ResourceRegistry | None = None,
        config: MainLoopConfig | None = None,
        dlq: DLQPolicy[MainLoopRequest[UserRequestT], MainLoopResult[OutputT]] | None = None,
    ) -> None:
        self._dlq = dlq
        ...
```

### Execution Flow

```
receive() ──► process() ──► reply() ──► acknowledge()
                 │
                 ▼ (exception)
           should_dead_letter()?
                 │
        ┌────────┴────────┐
        ▼ No              ▼ Yes
    nack(backoff)    send_to_dlq()
                          │
                          ▼
                     acknowledge()
```

### Implementation

```python
def _handle_failure(
    self,
    msg: Message[MainLoopRequest[UserRequestT], MainLoopResult[OutputT]],
    error: Exception,
) -> None:
    """Handle message processing failure."""
    # Check if DLQ is configured and policy triggers
    if self._dlq and self._dlq.should_dead_letter(msg, error):
        self._dead_letter(msg, error)
        return

    # Retry with backoff - do NOT send error reply here.
    # The message will be redelivered and may succeed on retry.
    # Only send error replies on terminal outcomes (DLQ or final failure).
    backoff = min(60 * msg.delivery_count, 900)
    msg.nack(visibility_timeout=backoff)


def _dead_letter(
    self,
    msg: Message[MainLoopRequest[UserRequestT], MainLoopResult[OutputT]],
    error: Exception,
) -> None:
    """Send message to dead letter queue."""
    assert self._dlq is not None

    dead_letter = DeadLetter(
        message_id=msg.id,
        body=msg.body,
        source_mailbox=self._requests.name,
        delivery_count=msg.delivery_count,
        last_error=str(error),
        last_error_type=f"{type(error).__module__}.{type(error).__qualname__}",
        dead_lettered_at=datetime.now(UTC),
        first_received_at=msg.enqueued_at,
        request_id=msg.body.request_id,
        reply_to=msg.reply_to.name if msg.reply_to else None,
    )

    # Send error reply - this is a terminal outcome
    try:
        if msg.reply_to:
            msg.reply(MainLoopResult(
                request_id=msg.body.request_id,
                error=f"Dead-lettered after {msg.delivery_count} attempts: {error}",
            ))
    except (ReplyNotAvailableError, MessageFinalizedError):
        pass

    self._dlq.mailbox.send(dead_letter)
    log.warning(
        "Message dead-lettered",
        message_id=msg.id,
        request_id=str(msg.body.request_id),
        delivery_count=msg.delivery_count,
        error_type=dead_letter.last_error_type,
    )

    # Acknowledge to remove from source queue
    msg.acknowledge()
```

## EvalLoop Integration

EvalLoop follows the same pattern:

```python
class EvalLoop[InputT, OutputT, ExpectedT]:
    def __init__(
        self,
        *,
        loop: MainLoop[InputT, OutputT],
        evaluator: Evaluator[OutputT, ExpectedT] | SessionEvaluator[OutputT, ExpectedT],
        requests: Mailbox[EvalRequest[InputT, ExpectedT], EvalResult],
        dlq: DLQPolicy[EvalRequest[InputT, ExpectedT], EvalResult] | None = None,
    ) -> None:
        self._dlq = dlq
        ...
```

The implementation mirrors MainLoop, wrapping `EvalRequest` in `DeadLetter`.

## Usage Examples

### Basic DLQ Setup

```python
from weakincentives.runtime import RedisMailbox, DLQPolicy, DeadLetter
from weakincentives.runtime.main_loop import MainLoop, MainLoopRequest, MainLoopResult

# Create mailboxes
requests: Mailbox[MainLoopRequest[MyRequest], MainLoopResult[MyOutput]] = RedisMailbox(
    name="requests",
    client=redis,
)
dead_letters: Mailbox[DeadLetter[MainLoopRequest[MyRequest]], None] = RedisMailbox(
    name="requests-dlq",
    client=redis,
)

# Configure MainLoop with DLQ
main_loop = MyMainLoop(
    adapter=adapter,
    requests=requests,
    dlq=DLQPolicy(
        mailbox=dead_letters,
        max_delivery_count=5,  # Dead-letter after 5 attempts
    ),
)
```

### Immediate Dead-Lettering for Specific Errors

```python
from weakincentives.prompt import ValidationError
from weakincentives.adapters import ContentPolicyViolation

# Dead-letter validation and policy errors immediately (don't retry)
dlq = DLQPolicy(
    mailbox=dead_letters,
    max_delivery_count=5,
    include_errors=frozenset({
        ValidationError,        # Invalid request format
        ContentPolicyViolation, # Content policy rejection
    }),
)
```

### Exclude Transient Errors

```python
from weakincentives.runtime.mailbox import MailboxConnectionError
from weakincentives.adapters import RateLimitError

# Never dead-letter transient errors (always retry)
dlq = DLQPolicy(
    mailbox=dead_letters,
    max_delivery_count=5,
    exclude_errors=frozenset({
        MailboxConnectionError,  # Network issues
        RateLimitError,          # Rate limiting
        TimeoutError,            # Timeouts
    }),
)
```

### Combined Configuration

```python
# Complex policy: immediate DLQ for some, never DLQ for others, threshold for rest
dlq = DLQPolicy(
    mailbox=dead_letters,
    max_delivery_count=5,
    include_errors=frozenset({ValidationError}),  # Immediate
    exclude_errors=frozenset({RateLimitError}),   # Never
)
```

### Custom DLQ Policy

Subclass `DLQPolicy` to implement custom dead-letter logic:

```python
from weakincentives.runtime import DLQPolicy, DeadLetter, Message

@dataclass(slots=True, frozen=True)
class ErrorBudgetPolicy[T, R](DLQPolicy[T, R]):
    """Dead-letter based on error rate, not just count."""

    error_budget: float = 0.5

    def should_dead_letter(self, message: Message[T, Any], error: Exception) -> bool:
        # Fall back to default behavior for threshold
        if message.delivery_count >= self.max_delivery_count:
            return True

        # Custom logic: dead-letter if error rate exceeds budget
        error_rate = get_error_rate(message.body)
        return error_rate > self.error_budget


main_loop = MyMainLoop(
    adapter=adapter,
    requests=requests,
    dlq=ErrorBudgetPolicy(
        mailbox=dead_letters,
        max_delivery_count=10,
        error_budget=0.3,
    ),
)
```

## Processing Dead Letters

### DLQ Consumer

A separate process handles dead-lettered messages:

```python
def process_dead_letters(
    dlq: Mailbox[DeadLetter[MainLoopRequest[MyRequest]], None],
    *,
    handler: Callable[[DeadLetter[MainLoopRequest[MyRequest]]], None],
) -> None:
    """Process dead letters with custom handler."""
    while True:
        for msg in dlq.receive(wait_time_seconds=20):
            try:
                handler(msg.body)
                msg.acknowledge()
            except Exception as e:
                log.error("DLQ handler failed", error=str(e))
                # DLQ messages should not re-dead-letter
                # Either fix and ack, or leave for manual intervention
                msg.nack(visibility_timeout=3600)  # 1 hour backoff
```

### Alert and Log

```python
def alert_handler(dead_letter: DeadLetter[MainLoopRequest[MyRequest]]) -> None:
    """Alert on dead-lettered messages."""
    log.error(
        "Request dead-lettered",
        message_id=dead_letter.message_id,
        request_id=str(dead_letter.request_id),
        error_type=dead_letter.last_error_type,
        error=dead_letter.last_error,
        delivery_count=dead_letter.delivery_count,
        source=dead_letter.source_mailbox,
    )

    # Send alert to monitoring system
    metrics.increment(
        "dlq.messages",
        tags={
            "source": dead_letter.source_mailbox,
            "error_type": dead_letter.last_error_type,
        },
    )

    # Optionally notify on-call
    if is_critical_request(dead_letter.body):
        pagerduty.alert(
            summary=f"Critical request dead-lettered: {dead_letter.request_id}",
            severity="high",
        )
```

### Replay to Source Queue

```python
def replay_handler(
    dead_letter: DeadLetter[MainLoopRequest[MyRequest]],
    *,
    source: Mailbox[MainLoopRequest[MyRequest], MainLoopResult[MyOutput]],
    reply_resolver: MailboxResolver,
) -> None:
    """Replay dead letter back to source queue."""
    # Reconstruct reply_to if present
    reply_to = None
    if dead_letter.reply_to:
        reply_to = reply_resolver.resolve(dead_letter.reply_to)

    # Re-send to source queue
    source.send(dead_letter.body, reply_to=reply_to)

    log.info(
        "Replayed dead letter",
        message_id=dead_letter.message_id,
        request_id=str(dead_letter.request_id),
    )
```

### Filtered Replay

```python
def selective_replay(
    dlq: Mailbox[DeadLetter[MainLoopRequest[MyRequest]], None],
    source: Mailbox[MainLoopRequest[MyRequest], MainLoopResult[MyOutput]],
    *,
    predicate: Callable[[DeadLetter[MainLoopRequest[MyRequest]]], bool],
    reply_resolver: MailboxResolver,
) -> int:
    """Replay dead letters matching predicate.

    Returns:
        Count of replayed messages.
    """
    replayed = 0

    for msg in dlq.receive(max_messages=10, visibility_timeout=60):
        if predicate(msg.body):
            replay_handler(msg.body, source=source, reply_resolver=reply_resolver)
            msg.acknowledge()
            replayed += 1
        else:
            # Leave non-matching messages
            msg.nack(visibility_timeout=0)

    return replayed


# Replay only timeout errors (transient, likely to succeed now)
count = selective_replay(
    dlq=dead_letters,
    source=requests,
    predicate=lambda dl: "TimeoutError" in dl.last_error_type,
    reply_resolver=resolver,
)
```

## Observability

### Metrics

Track DLQ health with these metrics:

| Metric | Type | Description |
| ------------------------------ | ------- | ------------------------------------ |
| `dlq.messages.sent` | Counter | Messages sent to DLQ |
| `dlq.messages.processed` | Counter | Messages processed from DLQ |
| `dlq.messages.replayed` | Counter | Messages replayed to source |
| `dlq.messages.discarded` | Counter | Messages discarded after processing |
| `dlq.queue.depth` | Gauge | Current DLQ size |
| `dlq.message.age_seconds` | Histogram | Time in DLQ before processing |
| `dlq.processing.duration_ms` | Histogram | DLQ handler execution time |

### Structured Logging

```python
# On dead-letter
log.warning(
    "Message dead-lettered",
    message_id=msg.id,
    request_id=str(request.request_id),
    source_mailbox=source.name,
    delivery_count=msg.delivery_count,
    error_type=f"{type(error).__module__}.{type(error).__qualname__}",
    error_message=str(error),
)

# On DLQ processing
log.info(
    "Processing dead letter",
    message_id=dead_letter.message_id,
    request_id=str(dead_letter.request_id),
    age_seconds=(datetime.now(UTC) - dead_letter.dead_lettered_at).total_seconds(),
    action="replay",  # or "discard", "alert"
)
```

### Health Checks

Include DLQ depth in health probes:

```python
def health_check() -> dict[str, Any]:
    """Health check including DLQ status."""
    dlq_count = dead_letters.approximate_count()

    return {
        "status": "healthy" if dlq_count < 100 else "degraded",
        "dlq": {
            "name": dead_letters.name,
            "depth": dlq_count,
            "threshold": 100,
        },
    }
```

## LoopGroup Integration

LoopGroup can manage DLQ consumers alongside main loops:

```python
from weakincentives.runtime import LoopGroup, DLQConsumer

# Create DLQ consumer as a Runnable
dlq_consumer = DLQConsumer(
    mailbox=dead_letters,
    handler=alert_handler,
)

# Run alongside main loops
group = LoopGroup(
    loops=[main_loop, eval_loop, dlq_consumer],
    health_port=8080,
    watchdog_threshold=720.0,
)
group.run()
```

### DLQConsumer

```python
class DLQConsumer[T]:
    """Runnable consumer for dead letter queues."""

    def __init__(
        self,
        *,
        mailbox: Mailbox[DeadLetter[T], None],
        handler: Callable[[DeadLetter[T]], None],
        visibility_timeout: int = 300,
    ) -> None:
        self._mailbox = mailbox
        self._handler = handler
        self._visibility_timeout = visibility_timeout
        self._running = False
        self._heartbeat = Heartbeat()

    def run(
        self,
        *,
        max_iterations: int | None = None,
        visibility_timeout: int | None = None,
        wait_time_seconds: int = 20,
    ) -> None:
        """Process dead letters until shutdown."""
        self._running = True
        iterations = 0
        vt = visibility_timeout or self._visibility_timeout

        while self._running and (max_iterations is None or iterations < max_iterations):
            self._heartbeat.beat()

            for msg in self._mailbox.receive(
                visibility_timeout=vt,
                wait_time_seconds=wait_time_seconds,
            ):
                try:
                    self._handler(msg.body)
                    msg.acknowledge()
                except Exception as e:
                    log.error(
                        "DLQ handler failed",
                        message_id=msg.id,
                        error=str(e),
                    )
                    # Long backoff for DLQ failures
                    msg.nack(visibility_timeout=3600)

            iterations += 1

    def shutdown(self, *, timeout: float = 30.0) -> bool:
        """Signal shutdown."""
        self._running = False
        return True

    @property
    def running(self) -> bool:
        return self._running

    @property
    def heartbeat(self) -> Heartbeat:
        return self._heartbeat
```

## Error Classification

### Retriable Errors

Errors that should retry with backoff (not dead-letter immediately):

- `MailboxConnectionError` - Network issues
- `RateLimitError` - Provider rate limiting
- `TimeoutError` - Request timeouts
- `VisibilityExpansionRequired` - Normal visibility flow

### Non-Retriable Errors

Errors that should dead-letter immediately (if `include_errors` configured):

- `ValidationError` - Invalid request format
- `ContentPolicyViolation` - Content policy rejection
- `AuthenticationError` - Invalid credentials
- `QuotaExceededError` - Account quota exceeded

### Ambiguous Errors

Errors where the right choice depends on context:

- `ProviderError` - Could be transient or permanent
- `SerializationError` - Usually permanent, but could be version mismatch
- `ResourceNotFoundError` - Might resolve after deployment

## Best Practices

### DLQ Queue Naming

Use consistent naming to identify DLQ relationships:

```python
# Pattern: {source}-dlq
requests = RedisMailbox(name="requests", client=redis)
dead_letters = RedisMailbox(name="requests-dlq", client=redis)

# For multiple queues
eval_requests = RedisMailbox(name="eval-requests", client=redis)
eval_dlq = RedisMailbox(name="eval-requests-dlq", client=redis)
```

### Retention Policy

DLQ messages should have longer retention than source queues:

```python
# Source queue: 3-day TTL (default)
requests = RedisMailbox(name="requests", client=redis, default_ttl=259200)

# DLQ: 30-day TTL for forensics
dead_letters = RedisMailbox(name="requests-dlq", client=redis, default_ttl=2592000)
```

### Avoid DLQ Loops

Never send DLQ consumer failures back to a DLQ:

```python
def safe_dlq_handler(dead_letter: DeadLetter[T]) -> None:
    """Handler that never raises (logs and discards on failure)."""
    try:
        process_dead_letter(dead_letter)
    except Exception as e:
        # Log but don't re-raise - prevents DLQ loops
        log.error(
            "Failed to process dead letter, discarding",
            message_id=dead_letter.message_id,
            error=str(e),
        )
        metrics.increment("dlq.messages.discarded")
```

### Idempotent Replay

Ensure replayed messages are safe to reprocess:

```python
def replay_with_dedup(
    dead_letter: DeadLetter[MainLoopRequest[MyRequest]],
    source: Mailbox[MainLoopRequest[MyRequest], MainLoopResult[MyOutput]],
    seen: set[UUID],
) -> bool:
    """Replay with deduplication."""
    request_id = dead_letter.request_id
    if request_id in seen:
        log.info("Skipping duplicate replay", request_id=str(request_id))
        return False

    seen.add(request_id)
    source.send(dead_letter.body)
    return True
```

## Limitations

- **No automatic replay.** DLQ consumption requires explicit setup.
- **No cross-backend DLQ.** Source and DLQ must use same backend type.
- **No message transformation.** Replay sends original message as-is.
- **No priority.** DLQ messages processed in FIFO order.
- **Single DLQ per loop.** Cannot route different errors to different DLQs.

## Related Specifications

- `specs/MAILBOX.md` - Mailbox protocol and message lifecycle
- `specs/MAIN_LOOP.md` - MainLoop orchestration
- `specs/EVALS.md` - EvalLoop and evaluation framework
- `specs/LIFECYCLE.md` - LoopGroup and graceful shutdown

# Mailboxes and Dead Letter Queues

*Canonical specs: [specs/MAILBOX.md](../specs/MAILBOX.md),
[specs/DLQ.md](../specs/DLQ.md)*

This guide explains how WINK uses message queues (mailboxes) and dead
letter queues to deliver work to agents reliably. It covers the mental
model, delivery guarantees, and how to handle messages that fail
repeatedly.

## Why Mailboxes Exist

AgentLoop needs a way to receive work. You could call `loop.execute()`
directly, but that couples the caller to the worker. If the worker
crashes mid-request, the work is lost. If you want multiple workers
processing from the same queue, you need coordination.

Mailboxes solve this. A `Mailbox[T, R]` is a typed message queue with
SQS-like semantics: point-to-point delivery, visibility timeout, and
explicit acknowledgment. You send a message. A consumer receives it.
The message becomes invisible to other consumers. When processing
succeeds, the consumer acknowledges it and it is deleted. If the
consumer crashes, the visibility timeout expires and the message
reappears for another attempt.

This is the same model as Amazon SQS, Azure Service Bus, or any
durable work queue. If you have used any of those, you already know
how WINK mailboxes work.

## At-Least-Once Delivery

Mailboxes guarantee at-least-once delivery. Every message you send
will be delivered to a consumer at least once. It may be delivered more
than once--if a consumer receives a message, starts processing, and
crashes before acknowledging, the message will be delivered again after
the visibility timeout expires.

This is the right default for agent workloads. The alternative,
exactly-once delivery, requires distributed transactions that add
latency and complexity. At-least-once is simpler, faster, and
sufficient when consumers are idempotent.

**Consumers must be idempotent.** Processing the same message twice
must produce the same result. In practice this means:

- Use the message's `request_id` to deduplicate if your tool handlers
  have side effects (database writes, API calls, file mutations)
- Design your handlers so that re-running them is safe--check whether
  the work was already done before doing it again
- If a handler is naturally idempotent (read-only operations, pure
  computations), you get this for free

AgentLoop handles the common case: if a request is processed twice,
the second run produces the same response. The caller may receive
duplicate replies, which it can deduplicate by `request_id`.

## The Message Lifecycle

A message moves through a simple state machine:

```
send() -> Queued -> receive() -> Invisible -> acknowledge() -> Deleted
                                    |
                            timeout expires -> Redelivery
                                    |
                            nack() -> Delayed Redelivery
```

**Queued.** The message sits in the queue waiting to be received.

**Invisible.** A consumer has received the message. No other consumer
can see it for the duration of the visibility timeout. The consumer
is expected to process it and acknowledge.

**Deleted.** The consumer called `acknowledge()`. The message is gone.

**Redelivery.** Either the visibility timeout expired (consumer
crashed or was too slow) or the consumer explicitly called `nack()`
to return the message. The message goes back to the queue with an
incremented `delivery_count`.

Two important rules:

1. Always send your response before acknowledging. If you acknowledge
   first and then crash before replying, the message is gone and the
   caller never gets a response.
1. Set `visibility_timeout` longer than your maximum expected
   processing time. If it is too short, messages will be redelivered
   while still being processed, causing duplicate work.

## The Reply-To Pattern

Agent workloads are typically request-response: a caller sends a
request and expects a result back. Mailboxes support this with the
`reply_to` parameter.

When you send a message, you can attach a reply mailbox:

```python nocheck
requests.send(Request(...), reply_to=responses)
```

The consumer receives the message and calls `msg.reply(result)` to
send the result back to the caller's mailbox. This decouples the
request and response queues--the worker does not need to know where
replies go at construction time.

For in-memory mailboxes, `reply_to` stores a direct reference. For
Redis mailboxes, it stores the mailbox name and a `MailboxResolver`
reconstructs the mailbox instance on the consumer side. This is
service discovery: the consumer resolves a name to a concrete
mailbox without knowing the caller's topology.

**Dynamic reply queues** are useful when multiple callers share the
same request queue. Each caller creates a unique reply mailbox
(e.g., `client-{uuid}`), attaches it as `reply_to`, and listens on
that mailbox for its response. The worker routes replies
automatically via `msg.reply()`.

## When Messages Fail: Dead Letter Queues

Some messages cannot be processed no matter how many times you retry.
A malformed request, an invalid configuration, a content policy
violation--these will fail on every attempt. Without intervention,
they cycle through receive-fail-redeliver indefinitely, consuming
resources and blocking the queue.

A dead letter queue (DLQ) captures these poison messages. After a
configured number of delivery attempts, the message is moved to the
DLQ instead of being redelivered. This unblocks the source queue
while preserving the failed message for inspection.

```
receive() -> process() -> reply() -> acknowledge()
                |
                v (exception)
          should_dead_letter()?
                |
        +-------+-------+
        v No            v Yes
    nack(backoff)   send_to_dlq() -> acknowledge()
```

Configure a DLQ by attaching a `DLQPolicy` to your AgentLoop:

```python nocheck
dlq = DLQPolicy(mailbox=dead_letters, max_delivery_count=5)
loop = AgentLoop(adapter=adapter, requests=requests, dlq=dlq)
```

After 5 failed deliveries, the message lands in `dead_letters` as a
`DeadLetter` wrapper containing the original body, the error, the
source queue name, and timestamps for forensics.

## Error Classification

Not all errors deserve the same treatment. DLQPolicy lets you
classify errors to route them appropriately:

| Category | Examples | Behavior |
|----------|----------|----------|
| Retriable | `RateLimitError`, `TimeoutError` | Backoff and retry |
| Non-retriable | `ValidationError`, `AuthError` | Immediate DLQ |
| Ambiguous | `ProviderError` | Retry up to limit |

Use `include_errors` to immediately dead-letter errors that will
never succeed on retry:

```python nocheck
dlq = DLQPolicy(
    mailbox=dead_letters,
    include_errors=frozenset({ValidationError, ContentPolicyViolation}),
)
```

Use `exclude_errors` to never dead-letter transient failures that
should always be retried:

```python nocheck
dlq = DLQPolicy(
    mailbox=dead_letters,
    exclude_errors=frozenset({RateLimitError, TimeoutError}),
)
```

The default behavior for unlisted errors is to retry up to
`max_delivery_count`, then dead-letter. This is a reasonable default
for ambiguous failures.

## Processing Dead Letters

Dead letters are not discarded--they sit in the DLQ for inspection
and potential replay. Common patterns:

**Alerting.** A DLQ consumer watches the dead letter mailbox and
sends alerts when messages arrive. This is your signal that something
needs human attention.

**Replay.** After fixing the root cause, replay dead letters back to
the source queue. Deduplicate by `request_id` to handle cases where
the fix was deployed while messages were still being retried.

**Forensics.** Each `DeadLetter` captures the original message body,
the last error and its type, delivery count, timestamps, and
optional trace and request IDs. This is usually enough to diagnose
the failure without reproducing it.

Run a DLQ consumer alongside your agent loops using `LoopGroup`:

```python nocheck
group = LoopGroup(
    loops=[agent_loop, dlq_consumer],
    health_port=8080,
)
```

One important rule: a DLQ handler should never re-dead-letter. If
processing a dead letter fails, back off and retry. A DLQ-of-DLQs
is a sign that something is structurally wrong.

## Lease Extension for Long-Running Work

Agent requests can take minutes. A tool might run a test suite, build
a project, or execute a multi-step evaluation. If the visibility
timeout expires during this work, the message gets redelivered to
another worker, causing duplicate execution.

You could set an enormous visibility timeout, but that creates a
different problem: if the worker genuinely crashes, the message is
stuck invisible for the entire timeout before being redelivered.

WINK solves this with `LeaseExtender`. Instead of a single long
timeout, you start with a moderate timeout and extend it
incrementally as the worker proves it is still making progress.

The key insight is that extension is tied to heartbeats. The
`LeaseExtender` attaches to a message and a heartbeat. When tool
handlers call `context.beat()` during long operations, the lease
extender checks whether enough time has elapsed since the last
extension and, if so, extends the visibility timeout.

```python nocheck
loop = AgentLoop(
    adapter=adapter,
    requests=mailbox,
    config=AgentLoopConfig(
        lease_extender=LeaseExtenderConfig(
            interval=60,     # Extend at most every 60 seconds
            extension=300,   # Each extension adds 5 minutes
        ),
    ),
)
```

This is heartbeat-based, not timer-based. A stuck worker that stops
calling `beat()` will not get its lease extended. The visibility
timeout expires naturally, and the message is redelivered to a
healthy worker. A daemon thread that blindly extends on a timer would
keep a stuck worker's lease alive indefinitely--the heartbeat
approach avoids this.

The `ToolExecutor` beats automatically before and after each tool
execution. Tools that perform long internal operations should add
explicit `context.beat()` calls to prove ongoing liveness.

## Choosing Visibility Timeouts

Getting visibility timeouts right requires thinking about several
interacting timers:

```
visibility_timeout > max_processing_time + safety_margin
watchdog_threshold > wait_time_seconds + max_processing_time
visibility_timeout > watchdog_threshold + max_processing_time
```

A concrete example for an agent that takes up to 10 minutes per
request:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `visibility_timeout` | 1800s (30 min) | Room for retries |
| `max_processing_time` | 600s (10 min) | Longest expected run |
| `watchdog_threshold` | 720s (12 min) | Detects stuck workers |
| `wait_time_seconds` | 30s | Long poll duration |
| `lease_extension` | 300s (5 min) | Incremental extension |
| `lease_interval` | 60s | Extension frequency |

With lease extension enabled, the initial `visibility_timeout` can
be shorter because the lease grows as the worker makes progress.
Without lease extension, set it conservatively high.

## In-Memory vs Redis

WINK ships two mailbox backends:

| Aspect | InMemoryMailbox | RedisMailbox |
|--------|-----------------|--------------|
| Scope | Single process | Multi-process |
| Durability | None | Configurable |
| Reply routing | Direct reference | Resolver-based |
| Use case | Tests, development | Production |

Use `InMemoryMailbox` for tests and single-process development. Use
`RedisMailbox` (in `weakincentives.contrib.mailbox`) when you need
durability or multiple workers processing from the same queue.

Both implementations provide FIFO ordering and the same protocol, so
switching backends requires no changes to your processing logic.

## Next Steps

- [Orchestration](orchestration.md): AgentLoop consumes from
  mailboxes
- [Lifecycle](lifecycle.md): Run loops and DLQ consumers with
  LoopGroup
- [Testing](testing.md): Use `FakeMailbox` and `CollectingMailbox`
  in tests

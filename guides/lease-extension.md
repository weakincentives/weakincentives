# Lease Extension

*Canonical spec: [specs/LEASE_EXTENDER.md](../specs/LEASE_EXTENDER.md)*

When processing messages from a queue, you need to prevent visibility timeout
during long-running operations. Lease extension ties visibility renewal to
proof-of-work: if the worker is actively processing, the lease extends; if
stuck, the lease expires and the message becomes available for reprocessing.

## The Problem

Message queues use visibility timeouts to handle worker failures:

1. Worker receives message, message becomes invisible
1. Worker processes message
1. Worker deletes message on success
1. If worker fails, visibility timeout expires and message reappears

For long-running agent tasks (10+ minutes), you need to extend visibility
during processing. But naive approaches have problems:

| Approach | Problem |
| -------- | ------- |
| Very long initial timeout | Stuck workers block messages for too long |
| Fixed-interval daemon | Keeps extending even when worker is stuck |
| Manual extension calls | Easy to forget, clutters handler code |

## The Heartbeat Solution

WINK solves this with heartbeat-based extension:

1. Tool handlers call `context.beat()` during long operations
1. Each beat potentially extends the message lease
1. No beats = no extensions = lease expires naturally

This correctly handles stuck workers: if processing hangs, beats stop, and the
lease eventually expires.

## Core Types

### LeaseExtenderConfig

```python nocheck
from weakincentives.runtime import LeaseExtenderConfig

config = LeaseExtenderConfig(
    interval=60.0,     # Minimum seconds between extensions
    extension=300,     # Visibility timeout per extension (seconds)
    enabled=True,      # Enable automatic extension
)
```

**Fields:**

- `interval`: Rate limiting—don't extend more often than this
- `extension`: How long each extension adds (relative to now)
- `enabled`: Toggle for testing or environments without queues

### LeaseExtender

Manages lease extension for a message:

```python nocheck
from weakincentives.runtime import LeaseExtender, Heartbeat

extender = LeaseExtender(config=LeaseExtenderConfig(interval=60, extension=300))
heartbeat = Heartbeat()

with extender.attach(msg, heartbeat):
    # Processing happens here
    # Heartbeats from tool execution extend the lease
    pass
```

### Heartbeat

Thread-safe heartbeat tracker with observer pattern:

```python nocheck
from weakincentives.runtime import Heartbeat

heartbeat = Heartbeat()

# Record a heartbeat
heartbeat.beat()

# Check time since last beat
print(heartbeat.elapsed())  # Seconds since last heartbeat

# Register callback (used by LeaseExtender)
heartbeat.add_callback(my_callback)
heartbeat.remove_callback(my_callback)
```

## Heartbeat Propagation

Heartbeats flow from tool handlers up to the lease extender:

```
MainLoop._handle_message()
  └─ lease_extender.attach(msg, heartbeat)
  └─ _execute(heartbeat=heartbeat)
       └─ adapter.evaluate(heartbeat=heartbeat)
            └─ ToolExecutor(heartbeat=heartbeat)
                 └─ ToolContext(heartbeat=heartbeat)
                      └─ handler calls context.beat()
```

When a tool handler calls `context.beat()`:

1. Heartbeat records the beat time
1. All registered callbacks are invoked
1. LeaseExtender's callback checks if extension is needed
1. If `interval` has elapsed, visibility is extended

## Using Heartbeats in Tool Handlers

For long-running operations, call `context.beat()` periodically:

```python nocheck
from weakincentives.prompt import ToolContext, ToolResult


def process_large_dataset(
    params: ProcessParams,
    *,
    context: ToolContext,
) -> ToolResult[ProcessResult]:
    results = []
    for i, item in enumerate(params.items):
        result = process_item(item)
        results.append(result)

        # Beat every 100 items to prove liveness
        if i % 100 == 0:
            context.beat()

    return ToolResult.ok(ProcessResult(results=results))
```

**When to beat:**

- During iteration over large collections
- Between phases of multi-step operations
- After completing significant chunks of work

**When not to beat:**

- On every loop iteration (too frequent, adds overhead)
- In short operations (unnecessary)

## Configuration in MainLoop

Configure lease extension when creating MainLoop:

```python nocheck
from weakincentives.runtime import (
    MainLoop,
    MainLoopConfig,
    LeaseExtenderConfig,
)

loop = MainLoop(
    adapter=adapter,
    requests=mailbox,
    config=MainLoopConfig(
        lease_extender=LeaseExtenderConfig(
            interval=60,      # Extend at most once per minute
            extension=300,    # Each extension adds 5 minutes
        ),
    ),
)
```

MainLoop automatically:

1. Creates a Heartbeat instance
1. Creates a LeaseExtender with your config
1. Attaches the extender to each message during processing
1. Passes the heartbeat through to tool execution

## Configuration in EvalLoop

EvalLoop follows the same pattern:

```python nocheck
from weakincentives.evals import EvalLoop, EvalLoopConfig
from weakincentives.runtime import LeaseExtenderConfig

loop = EvalLoop(
    adapter=adapter,
    evaluator=evaluator,
    requests=mailbox,
    config=EvalLoopConfig(
        lease_extender=LeaseExtenderConfig(interval=30, extension=180),
    ),
)
```

## Error Handling

Lease extension is a reliability optimization, not a correctness requirement:

| Error | Behavior |
| ----- | -------- |
| `ReceiptHandleExpiredError` | Logged as warning; processing continues |
| Network/transient errors | Logged; extension skipped |
| Already attached | Raises `RuntimeError` |

Failed extensions don't abort processing. The work continues, and if the lease
expires, the message will be redelivered to another worker.

## Timeout Calibration

The visibility timeout, extension interval, and watchdog threshold must be
coordinated:

```
visibility_timeout > watchdog_threshold + max_processing_time
extension < visibility_timeout
interval < extension / 2 (ensures extensions happen before expiry)
```

**Example for 10-minute max processing:**

| Parameter | Value | Rationale |
| --------- | ----- | --------- |
| `max_processing_time` | 600s | Longest expected operation |
| `extension` | 300s | Extends by 5 minutes each beat |
| `interval` | 60s | Extends at most once per minute |
| `watchdog_threshold` | 720s | Detects truly stuck workers |
| `visibility_timeout` | 1800s | Initial 30-minute window |

With these settings:

- Active workers extend every minute
- Stuck workers (no beats) are detected in 12 minutes
- Message becomes visible again after 30 minutes if worker dies

## Comparison with Daemon-Based Extension

| Aspect | Daemon Thread | Heartbeat-Based |
| ------ | ------------- | --------------- |
| Extension trigger | Fixed interval | Tool execution |
| Stuck worker | Keeps extending | Lease expires (correct!) |
| Thread overhead | Extra daemon | None |
| Proof-of-work | None | Only extends on activity |

The heartbeat-based approach is superior because it only extends when actual
work is being done, correctly handling stuck workers.

## Testing

Disable lease extension in tests:

```python nocheck
config = LeaseExtenderConfig(enabled=False)
```

Or use a mock message that tracks extension calls:

```python nocheck
from unittest.mock import Mock

msg = Mock()
msg.extend_visibility = Mock()

# After test
assert msg.extend_visibility.call_count == expected_extensions
```

## Next Steps

- [Lifecycle](lifecycle.md): Watchdog monitoring that complements lease extension
- [Orchestration](orchestration.md): MainLoop configuration
- [Evaluation](evaluation.md): EvalLoop configuration

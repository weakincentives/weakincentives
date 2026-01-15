# Message Lease Extender Specification

## Purpose

`LeaseExtender` prevents message visibility timeout during long-running request
processing by extending the lease when heartbeats occur. Ties lease extension to
proof-of-work: if the worker beats, the lease extends; if stuck (no beats), the
lease expires naturally.

**Implementation:** `src/weakincentives/runtime/lease_extender.py` (LeaseExtender,
LeaseExtenderConfig); `main_loop.py` for integration

## Core Types

### LeaseExtenderConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `interval` | `float` | `60.0` | Minimum seconds between extensions |
| `extension` | `int` | `300` | Visibility timeout per extension |
| `enabled` | `bool` | `True` | Enable automatic extension |

### LeaseExtender

Uses context manager pattern to attach/detach from heartbeat:

```python
with lease_extender.attach(msg, heartbeat):
    adapter.evaluate(prompt, session=session, heartbeat=heartbeat)
```

## Heartbeat Propagation

Heartbeat flows through the system:

```
MainLoop._handle_message()
  └─ lease_extender.attach(msg, heartbeat)
  └─ _execute(heartbeat=heartbeat)
       └─ adapter.evaluate(heartbeat=heartbeat)
            └─ ToolExecutor(heartbeat=heartbeat)
                 └─ ToolContext(heartbeat=heartbeat)
                      └─ handler calls context.beat()
```

### ToolContext.beat()

Tool handlers should beat during long operations:

```python
def my_handler(params, *, context: ToolContext) -> ToolResult:
    context.beat()  # Prove liveness
    # ... long operation
```

### Automatic Beating

`ToolExecutor` beats automatically before/after each tool execution. Tools can
add additional beats during long operations.

## MainLoop Integration

```python
loop = MainLoop(
    adapter=adapter,
    requests=mailbox,
    config=MainLoopConfig(
        lease_extender=LeaseExtenderConfig(interval=60, extension=300),
    ),
)
```

## EvalLoop Integration

EvalLoop uses same pattern. Passes its heartbeat to `MainLoop.execute()` so tool
beats extend EvalLoop's message lease.

## Claude Agent SDK Native Tools

Native tools (Bash, Read, Write) beat via SDK hooks since they don't flow
through `ToolExecutor`:

- `PreToolUse` hook: Beat before execution
- `PostToolUse` hook: Beat after execution

## Error Handling

| Error | Behavior |
|-------|----------|
| `ReceiptHandleExpiredError` | Log warning; continue processing |
| Network/transient errors | Log exception; skip extension |

Extension is reliability optimization, not correctness requirement.

## Comparison: Heartbeat-Based vs Daemon Thread

| Aspect | Daemon Thread | Heartbeat-Based |
|--------|---------------|-----------------|
| Extension trigger | Fixed interval | Tool execution |
| Stuck worker | Keeps extending | Lease expires (correct!) |
| Thread overhead | Extra daemon | None |
| Proof-of-work | None | Only extends on activity |

## Related Specifications

- `specs/HEALTH.md` - Heartbeat class definition
- `specs/MAILBOX.md` - Message visibility semantics
- `specs/MAIN_LOOP.md` - MainLoop orchestration

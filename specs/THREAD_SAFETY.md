# Thread Safety Specification

## Purpose

Document concurrency guarantees for multi-threaded usage. Threading is the
primary way to run tool adapters and orchestration without duplicating session
state.

## Guiding Principles

- Prefer deterministic, synchronous delivery over opportunistic concurrency
- Copy-on-write semantics for state transitions
- Single synchronization primitive per shared structure
- Document assumptions where enforced

## Thread-Safe Components

| Component | Protection | Notes |
|-----------|------------|-------|
| `InProcessDispatcher` | RLock on `_handlers` | Handler snapshots taken under lock |
| `Session` | RLock on `_state`, `_reducers` | Copy-on-write inside critical section |
| `LocalPromptOverridesStore` | Per-file locks | Atomic file writes |

## Session Thread Safety

| Operation | Thread-Safe | Notes |
|-----------|-------------|-------|
| `dispatch(event)` | Yes | Serialized via lock |
| `slice[T].all()` | Yes | Returns immutable tuple |
| `slice[T].latest()` | Yes | Point-in-time read |
| `snapshot()` | Yes | Consistent under lock |
| `restore(snapshot)` | Yes | Atomic replacement |
| `reset()` | Yes | Clears atomically |

## Event Dispatcher

- Handler snapshots taken under lock before delivery
- Lock not held during handler execution
- `subscribe`/`unsubscribe` thread-safe

## Non-Guarantees

- User-provided handlers not synchronized
- Example orchestration code may keep mutable state
- Override store lacks inter-process locking

## Anti-Patterns

```python
# WRONG: Blocking in reducer
@reducer(on=MyEvent)
def bad_reducer(state, event):
    time.sleep(10)  # Blocks all dispatches

# WRONG: Recursive dispatch
@reducer(on=MyEvent)
def bad_reducer(state, event):
    session.dispatch(OtherEvent())  # Deadlock
```

## Recommendations

1. One session per logical workflow
1. Use frozen dataclasses for slice contents
1. Keep reducers fast and pure
1. Don't share adapters across threads

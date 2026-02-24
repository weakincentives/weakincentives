# Session Runtime Specification

## Purpose

Sessions provide deterministic, side-effect-free containers for prompt run
lifecycle. Core implementation in `src/weakincentives/runtime/session/session.py`.

## Principles

- **Pure state transitions**: Reducers never mutate; every event returns a SliceOp (Append, Extend, Replace, Clear)
- **Typed dispatch**: Event payloads route by concrete dataclass type
- **Deterministic playback**: Sessions respond only to supported events
- **Publisher isolation**: Handler failures are logged and isolated
- **Explicit dispatchers**: Callers provide a `Dispatcher`; defaults to in-process

## Session Architecture

Session is a thin facade coordinating three specialized subsystems:

| Subsystem | Location | Responsibility |
| --- | --- | --- |
| `SliceStore` | `src/weakincentives/runtime/session/slice_store.py` | Slice storage with policy-based factories |
| `ReducerRegistry` | `src/weakincentives/runtime/session/reducer_registry.py` | Event-to-reducer routing |
| `SessionSnapshotter` | `src/weakincentives/runtime/session/session_snapshotter.py` | Snapshot/restore for transaction rollback |

Thread safety is provided by Session's `RLock`. Subsystems are not thread-safe
on their own and must only be accessed while holding the lock.

### Session Class

Core container at `src/weakincentives/runtime/session/session.py`:

| Method | Description |
| --- | --- |
| `__init__` | Initialize with dispatcher, parent, session_id, created_at, tags, slice_config |
| `dispatch()` | Broadcast event to all matching reducers |
| `__getitem__()` | Slice accessor via `session[Plan]` |
| `install()` | Install declarative state slice with @reducer methods |
| `snapshot()` | Capture immutable state snapshot |
| `reset()` | Clear all slices |
| `restore()` | Restore from snapshot |
| `clone()` | Create new session with different dispatcher/parent |
| `locked()` | Context manager yielding while holding Session's lock |

### SliceStore

At `src/weakincentives/runtime/session/slice_store.py`:

Manages typed slice instances, creating them on-demand using the appropriate
factory based on slice policy. Not thread-safe on its own.

| Method | Description |
| --- | --- |
| `get_or_create(slice_type)` | Get existing slice or create with appropriate factory |
| `select_all(slice_type)` | Return all items in a slice |
| `set_policy(slice_type, policy)` | Set policy for a slice type |
| `get_policy(slice_type)` | Get policy (defaults to STATE) |
| `snapshot_slices()` | Capture snapshot of all slice contents |
| `clear_all(slice_types)` | Clear slices for given types |

### ReducerRegistry

At `src/weakincentives/runtime/session/reducer_registry.py`:

Tracks which reducers should be invoked for each event type, with support for
multiple reducers per event type targeting different slices. Not thread-safe on its own.

| Method | Description |
| --- | --- |
| `register(event_type, reducer, target_slice=)` | Register reducer for event type |
| `get_registrations(event_type)` | Get all registrations for event type |
| `has_registrations(event_type)` | Check if any reducers registered |
| `all_target_slice_types()` | All slice types targeted by reducers |
| `snapshot()` | Snapshot registrations for cloning |
| `copy_from(snapshot)` | Copy registrations from snapshot |

### SessionSnapshotter

At `src/weakincentives/runtime/session/session_snapshotter.py`:

Encapsulates snapshot creation and restoration logic. Works with SliceStore
and ReducerRegistry to gather and apply state under Session's lock.

| Method | Description |
| --- | --- |
| `create_snapshot(parent_id=, children_ids=, tags=, policies=, include_all=)` | Capture immutable snapshot |
| `restore(snapshot, preserve_logs=)` | Restore session slices from snapshot |

## Reducers

Pure functions producing new slices from events at `src/weakincentives/runtime/session/reducers.py`:

| Reducer | Purpose |
| --- | --- |
| `append_all` | Ledger semantics, always appends (default) |
| `upsert_by(key_fn)` | Replace items with matching key |
| `replace_latest` | Store only most recent value |
| `replace_latest_by(key_fn)` | Like `replace_latest` but keyed |

## Query API

Via `SliceAccessor` at `src/weakincentives/runtime/session/slice_accessor.py`:

- `session[Plan].latest()` - Most recent value
- `session[Plan].all()` - All items as tuple
- `session[Plan].where(predicate)` - Filter by callable

## Dispatch API

All mutations go through `session.dispatch()` for auditability:

- **User events**: Route to registered reducers
- **System events**: `InitializeSlice`, `ClearSlice` handled before reducers

Convenience methods dispatch internally:

- `session[Plan].seed(value)` -> `InitializeSlice`
- `session[Plan].clear()` -> `ClearSlice`
- `session[Plan].append(value)` -> dispatch to reducers

### System Mutation Events

Defined at `src/weakincentives/runtime/session/slice_mutations.py`:

| Event | Purpose |
| --- | --- |
| `InitializeSlice[T]` | Replace all values in a slice |
| `ClearSlice[T]` | Remove items (optionally with predicate) |

## Declarative State Slices

The `@reducer` decorator at `src/weakincentives/runtime/session/state_slice.py` enables co-located
reducers on dataclasses:

```python
@dataclass(frozen=True)
class AgentPlan:
    steps: tuple[str, ...] = ()

    @reducer(on=AddStep)
    def add_step(self, event: AddStep) -> Replace["AgentPlan"]:
        new_plan = replace(self, steps=(*self.steps, event.step))
        return Replace((new_plan,))
```

Install with `session.install(AgentPlan)` to auto-register all `@reducer` methods.
Optional `initial` factory enables handling events when slice is empty.

### Constraints

- Class must be frozen dataclass
- Event types defined before slice class
- Methods return SliceOp (e.g., `Replace`, `Append`) containing new state
- One reducer method per event type per slice

## Event System

### InProcessDispatcher

At `runtime/events/__init__.py`:

- `subscribe(event_type, handler)` - Register handler
- `unsubscribe(event_type, handler)` - Remove handler, returns bool
- `dispatch(event)` -> `DispatchResult` with handler results/errors

### Event Types

At `runtime/events/types.py` and `runtime/events/__init__.py`:

| Event | When Emitted |
| --- | --- |
| `PromptRendered` | After render, before provider call |
| `RenderedTools` | After render, correlated with `PromptRendered` via `render_event_id` |
| `PromptExecuted` | After all tools and parsing |
| `ToolInvoked` | After each tool handler |

### DispatchResult

At `runtime/events/types.py`:

- `ok` property - True if no handler errors
- `raise_if_errors()` - Optional strict mode

### Delivery Semantics

- Synchronous on publisher thread
- In-order per dispatcher instance
- Handler exceptions logged and isolated

## Snapshotable Protocol

At `src/weakincentives/runtime/snapshotable.py`:

Generic protocol for state containers that support snapshot and restore.
Implementations include `Session` (`Snapshotable[Snapshot]`) and
`InMemoryFilesystem` (`Snapshotable[FilesystemSnapshot]`).

## Snapshots

Capture/restore via `Session` methods; `Snapshot` class at `src/weakincentives/runtime/session/snapshots.py`:

- `session.snapshot()` -> `Snapshot`
- `snapshot.to_json()` / `Snapshot.from_json()`
- `session.restore(snapshot)`

Serialization uses ISO 8601 timestamps, qualified type names, schema versioning.
Errors: `SnapshotSerializationError`, `SnapshotRestoreError`.

## Session Hierarchy

Sessions form trees for nested orchestration. Use `parent` parameter in constructor.
Traverse with `iter_sessions_bottom_up()`.

## Deadlines

Wall-clock limits via `Deadline` at `src/weakincentives/deadlines.py`:

- Must be timezone-aware, >1s in future
- Checked: before provider calls, before tool execution, during response finalization
- Propagated via `RenderedPrompt.deadline` and `ToolContext.deadline`
- Raises `DeadlineExceededError` -> converted to `PromptEvaluationError` with `phase="request"`, `phase="response"`, or `phase="tool"`

## Budgets

Combined time/token limits via `Budget` and `BudgetTracker` at `src/weakincentives/budget.py`:

| Field | Description |
| --- | --- |
| `deadline` | Optional `Deadline` |
| `max_total_tokens` | Total token limit |
| `max_input_tokens` | Input token limit |
| `max_output_tokens` | Output token limit |

`BudgetTracker` is thread-safe:

- `record_cumulative(evaluation_id, usage)` - Record usage
- `consumed` property - Sum across evaluations
- `check()` - Raise `BudgetExceededError` if breached

Enforcement: after every provider response, after every tool call, on completion.

## Limitations

- **Synchronous reducers**: Keep lightweight
- **Dataclass focus**: Non-dataclass payloads use generic slices
- **No implicit eviction**: State grows; use `replace_latest` when needed
- **No mid-request cancellation**: Limits checked at checkpoints only
- **Clock synchronization**: Deadlines require synchronized UTC

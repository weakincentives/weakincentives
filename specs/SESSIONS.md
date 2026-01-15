# Session Runtime Specification

## Purpose

Sessions provide deterministic, side-effect-free containers for prompt run
lifecycle. Core implementation in `src/weakincentives/runtime/session/session.py`.

## Principles

- **Pure state transitions**: Reducers never mutate; every event produces a new tuple
- **Typed dispatch**: Event payloads route by concrete dataclass type
- **Deterministic playback**: Sessions respond only to supported events
- **Publisher isolation**: Handler failures are logged and isolated
- **Explicit dispatchers**: Callers provide a `Dispatcher`; defaults to in-process

## Session State

### Session Class

Core container at `runtime/session/session.py:159-952`:

| Method | Line | Description |
| --- | --- | --- |
| `__init__` | 186-236 | Initialize with dispatcher, parent, session_id, tags |
| `dispatch()` | 442-472 | Broadcast event to all matching reducers |
| `__getitem__()` | 368-396 | Slice accessor via `session[Plan]` |
| `install()` | 399-435 | Install declarative state slice with @reducer methods |
| `snapshot()` | 696-750 | Capture immutable state snapshot |
| `reset()` | 480-506 | Clear all slices |
| `restore()` | 509-562 | Restore from snapshot |
| `clone()` | 566-610 | Create new session with different dispatcher/parent |

### Reducers

Pure functions producing new slices from events at `runtime/session/reducers.py`:

| Reducer | Purpose |
| --- | --- |
| `append_all` | Ledger semantics, always appends (default) |
| `upsert_by(key_fn)` | Replace items with matching key |
| `replace_latest` | Store only most recent value |
| `replace_latest_by(key_fn)` | Like `replace_latest` but keyed |

### Query API

Via `SliceAccessor` at `runtime/session/slice_accessor.py`:

- `session[Plan].latest()` - Most recent value
- `session[Plan].all()` - All items as tuple
- `session[Plan].where(predicate)` - Filter by callable

### Dispatch API

All mutations go through `session.dispatch()` for auditability:

- **User events**: Route to registered reducers
- **System events**: `InitializeSlice`, `ClearSlice` handled before reducers

Convenience methods dispatch internally:
- `session[Plan].seed(value)` → `InitializeSlice`
- `session[Plan].clear()` → `ClearSlice`
- `session[Plan].append(value)` → dispatch to reducers

### System Mutation Events

Defined at `runtime/session/slices/_ops.py`:

| Event | Purpose |
| --- | --- |
| `InitializeSlice[T]` | Replace all values in a slice |
| `ClearSlice[T]` | Remove items (optionally with predicate) |

## Declarative State Slices

The `@reducer` decorator at `runtime/session/session.py:85-156` enables co-located
reducers on dataclasses:

```python
@dataclass(frozen=True)
class AgentPlan:
    steps: tuple[str, ...] = ()

    @reducer(on=AddStep)
    def add_step(self, event: AddStep) -> "AgentPlan":
        return replace(self, steps=(*self.steps, event.step))
```

Install with `session.install(AgentPlan)` to auto-register all `@reducer` methods.
Optional `initial` factory enables handling events when slice is empty.

### Constraints

- Class must be frozen dataclass
- Event types defined before slice class
- Methods return new instance (validated at runtime)
- One reducer method per event type per slice

## Event System

### InProcessDispatcher

At `runtime/events/__init__.py:56-105`:

- `subscribe(event_type, handler)` - Register handler
- `dispatch(event)` → `DispatchResult` with handler results/errors

### Event Types

At `runtime/events/types.py` and `runtime/events/__init__.py`:

| Event | Line | When Emitted |
| --- | --- | --- |
| `PromptRendered` | `__init__.py:123-136` | After render, before provider call |
| `PromptExecuted` | `__init__.py:109-119` | After all tools and parsing |
| `ToolInvoked` | `types.py:128-154` | After each tool handler |

### DispatchResult

At `runtime/events/types.py:79-109`:
- `ok` property - True if no handler errors
- `raise_if_errors()` - Optional strict mode

### Delivery Semantics

- Synchronous on publisher thread
- In-order per dispatcher instance
- Handler exceptions logged and isolated

## Snapshots

Capture/restore at `runtime/session/session.py:696-750`:

- `session.snapshot()` → `Snapshot`
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
- Raises `DeadlineExceededError` → converted to `PromptEvaluationError` with `phase="deadline"`

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

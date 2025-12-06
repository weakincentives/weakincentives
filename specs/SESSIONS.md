# Session Runtime Specification

## Purpose

Sessions provide a deterministic, side-effect-free container for the full
lifecycle of a prompt run. This specification covers session state management,
event emission and subscription, deadline enforcement, and budget tracking.

## Guiding Principles

- **Pure state transitions**: Reducers never mutate in place; every event
  produces a new tuple.
- **Typed first**: Event payloads route by concrete dataclass type.
- **Deterministic playback**: Sessions respond only to supported events, making
  state easy to reconstruct.
- **Publisher isolation**: Event dispatch is fire-and-forget; handler failures
  are logged and isolated.
- **No implicit globals**: Callers must provide an `EventBus` instance per
  evaluation.

## Session State

### Session

`Session` is an immutable container for accumulated dataclass instances:

```python
class Session:
    def __init__(
        self,
        *,
        bus: EventBus,
        parent: Session | None = None,
        session_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> None: ...

    def register_reducer[T, S](
        self,
        data_type: type[T],
        reducer: TypedReducer[T, S],
        *,
        slice_type: type[S] | None = None,
    ) -> None: ...

    def select_all[S](self, slice_type: type[S]) -> tuple[S, ...]: ...

    def snapshot(self) -> Snapshot: ...

    def rollback(self, snapshot: Snapshot) -> None: ...

    def clone(
        self,
        *,
        bus: EventBus,
        parent: Session | None = None,
    ) -> "Session": ...
```

### Reducers

Pure functions that produce new slices from events:

```python
def reducer(
    slice_values: tuple[S, ...],
    event: DataEvent,
    *,
    context: ReducerContext,
) -> tuple[S, ...]: ...
```

Built-in reducers:

- `append` - Default, dedupes by equality
- `upsert_by(key_fn)` - Replaces items with matching key
- `replace_latest` - Stores only the most recent value

### Selectors

```python
from weakincentives.runtime.session import select_all, select_latest, select_where

latest_plan = select_latest(session, Plan)
all_results = select_all(session, SearchResult)
filtered = select_where(session, Issue, lambda i: i.severity == "high")
```

### Session Hierarchy

Sessions form a tree for nested orchestration:

```python
parent_session = Session(bus=bus)
child_session = Session(bus=bus, parent=parent_session)

# Traverse from leaves up
for session in iter_sessions_bottom_up(root_session):
    snapshot = session.snapshot()
```

## Event System

### Event Bus

In-process pub/sub for prompt lifecycle events:

```python
from weakincentives.runtime.events import InProcessEventBus

bus = InProcessEventBus()
bus.subscribe(PromptExecuted, handler)
result = bus.publish(event)

if not result.ok:
    result.raise_if_errors()  # Optional strict mode
```

### Event Types

**PromptRendered** - After render, before provider call:

```python
@dataclass(slots=True, frozen=True)
class PromptRendered:
    event_id: UUID
    prompt_ns: str
    prompt_key: str
    prompt_name: str | None
    adapter: str
    session_id: UUID | None
    render_inputs: tuple[SupportsDataclass, ...]
    rendered_prompt: str
    created_at: datetime
```

**PromptExecuted** - After all tools and parsing:

```python
@dataclass(slots=True, frozen=True)
class PromptExecuted:
    event_id: UUID
    prompt_name: str
    adapter: str
    result: PromptResponse[Any]
    session_id: UUID | None
    created_at: datetime
```

**ToolInvoked** - After each tool handler:

```python
@dataclass(slots=True, frozen=True)
class ToolInvoked:
    event_id: UUID
    prompt_name: str
    adapter: str
    name: str
    params: SupportsDataclass
    result: ToolResult[Any]
    call_id: str | None
    session_id: UUID | None
    created_at: datetime
```

### Publish Results

```python
@dataclass(slots=True, frozen=True)
class PublishResult:
    event: object
    handlers_invoked: tuple[EventHandler, ...]
    errors: tuple[HandlerFailure, ...]
    handled_count: int

    @property
    def ok(self) -> bool: ...

    def raise_if_errors(self) -> None: ...
```

### Delivery Semantics

- Events delivered synchronously on publisher thread
- In-order delivery per bus instance
- Handler exceptions logged and isolated (unless `raise_if_errors()` called)
- No persistence or cross-process forwarding (implement in subscribers)

## Snapshots

Capture and restore session state:

```python
# Capture
snapshot = session.snapshot()

# Serialize
json_str = snapshot.to_json()

# Restore
loaded = Snapshot.from_json(json_str)
session.rollback(loaded)
```

### Serialization

- Timestamps as timezone-aware ISO 8601
- Types as `"package.module:Class"`
- Items via dataclass serde helpers
- Schema version for compatibility

### Error Handling

- `SnapshotSerializationError` - Unsupported types or payloads
- `SnapshotRestoreError` - Incompatible schema or missing types

## Deadlines

Wall-clock limits for evaluation runs:

```python
from weakincentives.deadlines import Deadline

deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(seconds=30))

response = adapter.evaluate(
    prompt,
    params,
    bus=bus,
    session=session,
    deadline=deadline,
)
```

### Deadline Object

```python
@dataclass(slots=True, frozen=True)
class Deadline:
    expires_at: datetime  # Must be timezone-aware, >1s in future

    def remaining(self, *, now: datetime | None = None) -> timedelta: ...
```

### Enforcement Checkpoints

1. **Before provider calls** - Raise if expired
1. **Before tool execution** - Raise if expired
1. **During response finalization** - Raise if expired
1. **Retry loops** - Re-check before each iteration

### Deadline Propagation

- Stored on `RenderedPrompt.deadline`
- Available via `ToolContext.deadline`
- Subagents inherit parent deadline (use tighter of the two)

### DeadlineExceededError

```python
class DeadlineExceededError(RuntimeError):
    """Tool handler cannot complete before deadline."""
```

Converted to `PromptEvaluationError` with `phase="deadline"` by runtime.

## Budgets

Combined time and token limits:

```python
from weakincentives.budget import Budget, BudgetTracker

budget = Budget(
    deadline=Deadline(expires_at=...),
    max_total_tokens=10000,
    max_input_tokens=8000,
    max_output_tokens=2000,
)

tracker = BudgetTracker(budget)
```

### Budget Object

```python
@dataclass(slots=True, frozen=True)
class Budget:
    deadline: Deadline | None = None
    max_total_tokens: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
```

At least one limit must be set. Token limits must be positive.

### BudgetTracker

Thread-safe tracker for token consumption:

```python
@dataclass
class BudgetTracker:
    budget: Budget

    def record_cumulative(self, evaluation_id: str, usage: TokenUsage) -> None:
        """Record cumulative usage for an evaluation (replaces previous)."""

    @property
    def consumed(self) -> TokenUsage:
        """Sum usage across all evaluations."""

    def check(self) -> None:
        """Raise BudgetExceededError if any limit is breached."""
```

### Enforcement Checkpoints

1. **After every provider response** - Record usage, check limits
1. **After every tool call** - Check limits
1. **On evaluation completion** - Final check

### Subagent Propagation

| Isolation Level | Session/Bus | BudgetTracker |
|-----------------|-------------|---------------|
| NO_ISOLATION | Shared | Shared |
| FULL_ISOLATION | Cloned | Shared |

Budget tracking is always shared so parallel subagents contribute to global
limits.

### BudgetExceededError

```python
@dataclass(slots=True, frozen=True)
class BudgetExceededError(RuntimeError):
    budget: Budget
    consumed: TokenUsage
    exceeded_dimension: str  # "deadline", "total_tokens", etc.
```

Converted to `PromptEvaluationError` with `phase="budget"` by runtime.

## Usage Example

```python
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session, select_latest
from weakincentives.deadlines import Deadline
from weakincentives.budget import Budget, BudgetTracker

# Setup
bus = InProcessEventBus()
session = Session(bus=bus)

# Optional: register custom reducers
session.register_reducer(
    ResearchSummary,
    update_metrics_reducer,
    slice_type=ResearchMetrics,
)

# Configure limits
deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5))
budget = Budget(deadline=deadline, max_total_tokens=50000)
tracker = BudgetTracker(budget)

# Evaluate with constraints
response = adapter.evaluate(
    prompt,
    params,
    bus=bus,
    session=session,
    deadline=deadline,
)

# Query state
latest_plan = select_latest(session, Plan)
all_metrics = select_all(session, ResearchMetrics)

# Snapshot for persistence
snapshot = session.snapshot()
json_str = snapshot.to_json()
```

## Limitations

- **Synchronous reducers**: Run on publisher thread; keep them lightweight
- **Dataclass focus**: Non-dataclass payloads only populate generic slices
- **No implicit eviction**: State grows; use `replace_latest` when needed
- **No mid-request cancellation**: Limits checked at checkpoints only
- **Clock synchronization**: Deadlines require synchronized UTC clocks

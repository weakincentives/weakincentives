# Chapter 5: Sessions

> **Canonical Reference**: See [specs/SESSIONS.md](../specs/SESSIONS.md) for the complete specification.

## Introduction

A `Session` is WINK's answer to "agent memory"—but with a critical constraint:

> **Memory must be deterministic and inspectable.**

Most agent frameworks give you a mutable dictionary or database you modify directly. This works initially, but creates problems at scale:

- **Non-deterministic replays**: Can't reproduce bugs from production
- **Hidden side effects**: State changes scattered throughout the codebase
- **Poor observability**: No audit trail of what changed when
- **Difficult testing**: Hard to assert on state transitions

WINK takes a different approach inspired by Redux and event sourcing: **every state change flows through a pure reducer, and every mutation is recorded as an event**.

Sessions provide:

- **Typed memory slices** - Each dataclass type gets its own slice (`tuple[T, ...]`)
- **Pure reducers** - State transitions are deterministic, side-effect-free functions
- **Event-driven updates** - All mutations go through `dispatch()` for auditability
- **Snapshotting** - Capture and restore entire session state as JSON
- **Query API** - Read state with `latest()`, `all()`, and `where()` filters
- **Hierarchical organization** - Sessions form trees for nested orchestration

This chapter covers WINK's session system from fundamentals to advanced patterns.

## The Mental Model

Think of a session as a **time-series database for agent memory**:

```mermaid
flowchart LR
    subgraph EventStream["Event Stream (Input)"]
        E1["AddStep"]
        E2["CompleteStep"]
        E3["UpdatePlan"]
        E4["AddStep"]
    end

    subgraph Session["Session (State Container)"]
        Reducers["Pure Reducers"]

        subgraph Slices["Typed Slices"]
            PlanSlice["Plan: tuple[Plan, ...]"]
            FactSlice["Fact: tuple[Fact, ...]"]
            MetricSlice["Metric: tuple[Metric, ...]"]
        end
    end

    subgraph Output["Query API"]
        Latest["latest()"]
        All["all()"]
        Where["where(predicate)"]
    end

    E1 --> Reducers
    E2 --> Reducers
    E3 --> Reducers
    E4 --> Reducers

    Reducers --> PlanSlice
    Reducers --> FactSlice
    Reducers --> MetricSlice

    PlanSlice --> Latest
    FactSlice --> All
    MetricSlice --> Where

    style Session fill:#e1f5ff
    style Slices fill:#f0f9ff
```

**Key concepts:**

1. **Events** are dispatched through `session.dispatch(event)`
2. **Reducers** transform events into new slice values
3. **Slices** are immutable tuples stored by type
4. **Queries** read slice contents without mutation

The session never mutates state in place. Reducers return new tuples. This makes snapshots trivial (just serialize the current tuples) and restoration straightforward.

## Creating a Session

The minimal setup requires an event dispatcher:

```python
from weakincentives.runtime import Session, InProcessDispatcher

# Create event bus
bus = InProcessDispatcher()

# Create session
session = Session(bus=bus)
```

### Session Parameters

```python
from datetime import datetime, UTC
from uuid import uuid4

session = Session(
    bus=bus,                          # Event dispatcher (required)
    parent=parent_session,            # Parent session for hierarchies
    session_id=uuid4(),               # Unique ID (generated if omitted)
    created_at=datetime.now(UTC),     # Creation timestamp
    tags={"user": "alice", "env": "prod"},  # Metadata tags
)
```

### Session Properties

```python
# Access session metadata
session.dispatcher        # Event bus for telemetry
session.parent           # Parent session (or None)
session.children         # Tuple of child sessions
session.tags             # Metadata dictionary
session.session_id       # UUID identifier
session.created_at       # Creation timestamp
```

## Query API: Reading State

Access slice contents using the slice accessor `session[Type]`:

```python
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class Fact:
    key: str
    value: str

# Get all items in the slice
facts: tuple[Fact, ...] = session[Fact].all()

# Get the most recent item (or None)
latest_fact: Fact | None = session[Fact].latest()

# Filter items with a predicate
recent_facts = session[Fact].where(lambda f: f.key.startswith("repo_"))

# Check if slice is empty
has_facts = session[Fact].exists()
```

### Query Methods

| Method | Return Type | Description |
|--------|-------------|-------------|
| `all()` | `tuple[T, ...]` | All items in the slice, in order |
| `latest()` | `T \| None` | Most recent item, or `None` if empty |
| `where(predicate)` | `tuple[T, ...]` | Items matching the predicate |
| `exists()` | `bool` | `True` if slice has any items |

All queries are read-only. They never mutate the session.

### Query Examples

```python
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class Issue:
    id: int
    severity: str
    resolved: bool

# Find all high-severity unresolved issues
critical = session[Issue].where(
    lambda i: i.severity == "high" and not i.resolved
)

# Get the most recent issue
latest = session[Issue].latest()
if latest and latest.severity == "critical":
    print(f"Critical issue: {latest.id}")

# Count issues
all_issues = session[Issue].all()
print(f"Total issues: {len(all_issues)}")
```

## Reducers: Pure State Transitions

A **reducer** is a pure function that takes current state and an event, returning a new state:

```
new_state = reducer(current_state, event)
```

This pattern comes from functional programming (popularized by Redux). The core insight: **given the same inputs, you always get the same output**. This makes state changes predictable and debuggable—you can log every event and trace the exact sequence that led to any state.

### Reducer Signature

In WINK, reducers receive a `SliceView[S]` (read-only access to current values) and return a `SliceOp[S]` (describing the mutation to apply):

```python
from weakincentives.runtime.session import SliceView, SliceOp, Append

def my_reducer(state: SliceView[Plan], event: AddStep) -> SliceOp[Plan]:
    return Append(Plan(steps=(event.step,)))
```

### SliceOp Types

Reducers must return one of four operation types:

```mermaid
graph TB
    SliceOp["SliceOp[T]<br/>(Union type)"]

    Append["Append[T]<br/>Add single item"]
    Extend["Extend[T]<br/>Add multiple items"]
    Replace["Replace[T]<br/>Replace all items"]
    Clear["Clear[T]<br/>Remove items"]

    SliceOp --> Append
    SliceOp --> Extend
    SliceOp --> Replace
    SliceOp --> Clear

    style Append fill:#e1ffe1
    style Extend fill:#e1ffe1
    style Replace fill:#ffe1e1
    style Clear fill:#ffe1e1
```

#### Append[T]

Add a single value to the slice. Most efficient for file-backed slices (append-only).

```python
from weakincentives.runtime.session import Append

@dataclass(slots=True, frozen=True)
class LogEntry:
    message: str
    timestamp: datetime

def log_reducer(state: SliceView[LogEntry], event: LogEntry) -> Append[LogEntry]:
    return Append(event)
```

#### Extend[T]

Add multiple values at once.

```python
from weakincentives.runtime.session import Extend

def batch_add_reducer(
    state: SliceView[Fact],
    event: BatchAddFacts,
) -> Extend[Fact]:
    return Extend(tuple(event.facts))
```

#### Replace[T]

Replace the entire slice with new values. Required when transforming existing state.

```python
from weakincentives.runtime.session import Replace

def replace_latest_reducer(
    state: SliceView[Plan],
    event: UpdatePlan,
) -> Replace[Plan]:
    # Keep only the new plan
    return Replace((event.new_plan,))
```

#### Clear[T]

Remove items from the slice. Optionally filter by predicate.

```python
from weakincentives.runtime.session import Clear

def clear_completed_reducer(
    state: SliceView[Task],
    event: ClearCompleted,
) -> Clear[Task]:
    # Remove only completed tasks
    return Clear(predicate=lambda t: t.completed)
```

### Registering Reducers

Register reducers for specific event types on a slice:

```python
from dataclasses import dataclass
from weakincentives.runtime.session import SliceView, Append

@dataclass(slots=True, frozen=True)
class Plan:
    steps: tuple[str, ...]

@dataclass(slots=True, frozen=True)
class AddStep:
    step: str

def add_step_reducer(state: SliceView[Plan], event: AddStep) -> Append[Plan]:
    current = state.latest()
    if current is None:
        return Append(Plan(steps=(event.step,)))

    # Create new plan with additional step
    new_plan = Plan(steps=(*current.steps, event.step))
    return Append(new_plan)

# Register the reducer
session[Plan].register(AddStep, add_step_reducer)

# Now dispatch events
session.dispatch(AddStep(step="Read README"))
session.dispatch(AddStep(step="Run tests"))

# Query the result
plan = session[Plan].latest()
assert plan is not None
assert plan.steps == ("Read README", "Run tests")
```

### Built-in Reducers

WINK provides common reducer patterns:

```python
from weakincentives.runtime.session import (
    append_all,
    replace_latest,
    replace_latest_by,
    upsert_by,
)

# Ledger semantics: always append (default for all slices)
session[Event].register(Event, append_all)

# Keep only the most recent value
session[Config].register(Config, replace_latest)

# Replace by key function
session[User].register(
    UserUpdate,
    replace_latest_by(key=lambda u: u.user_id)
)

# Insert or update by key
session[Document].register(
    DocumentUpdate,
    upsert_by(key=lambda d: d.doc_id)
)
```

### Reducer Flow

```mermaid
sequenceDiagram
    participant Client
    participant Session
    participant Dispatcher
    participant Reducers
    participant Storage

    Client->>Session: dispatch(AddStep("test"))
    Session->>Dispatcher: publish(AddStep)
    Dispatcher->>Reducers: Invoke all matching reducers

    loop For each registered reducer
        Reducers->>Storage: Read current state
        Storage-->>Reducers: SliceView[Plan]
        Reducers->>Reducers: Compute new state
        Reducers->>Storage: Apply SliceOp
        Storage->>Storage: Update immutable tuple
    end

    Session-->>Client: Event dispatched
    Client->>Session: [Plan].latest()
    Session->>Storage: Query slice
    Storage-->>Session: Plan instance
    Session-->>Client: Plan(steps=(...))
```

**Key points:**

1. Events are dispatched to **all registered reducers** for that event type
2. Reducers run synchronously on the dispatcher thread (keep them lightweight)
3. Each reducer receives a read-only view of current state
4. Reducer results are applied immediately to the slice
5. Queries see the updated state after dispatch completes

## Declarative Reducers with @reducer

For complex state slices with many event types, manually registering reducers becomes tedious:

```python
# Imperative style (verbose)
session[Plan].register(AddStep, add_step_reducer)
session[Plan].register(CompleteStep, complete_step_reducer)
session[Plan].register(UpdateStep, update_step_reducer)
session[Plan].register(RemoveStep, remove_step_reducer)
```

The `@reducer` decorator provides a cleaner, declarative style by co-locating reducers as methods on the dataclass:

```python
from dataclasses import dataclass, replace
from weakincentives.runtime.session import reducer, Replace

# Event types
@dataclass(slots=True, frozen=True)
class AddStep:
    step: str

@dataclass(slots=True, frozen=True)
class CompleteStep:
    pass

# Declarative state slice
@dataclass(slots=True, frozen=True)
class AgentPlan:
    steps: tuple[str, ...]
    current_step: int = 0

    @reducer(on=AddStep)
    def add_step(self, event: AddStep) -> Replace["AgentPlan"]:
        """Add a new step to the plan."""
        return Replace((
            replace(self, steps=(*self.steps, event.step)),
        ))

    @reducer(on=CompleteStep)
    def complete_step(self, event: CompleteStep) -> Replace["AgentPlan"]:
        """Mark current step as complete and advance."""
        return Replace((
            replace(self, current_step=self.current_step + 1),
        ))
```

### Installing Declarative Slices

Use `session.install()` to auto-register all `@reducer` methods:

```python
# Install the slice (registers all @reducer methods)
session.install(AgentPlan)

# Seed initial state
session[AgentPlan].seed(AgentPlan(steps=("Research", "Implement")))

# Dispatch events - reducers are automatically invoked
session.dispatch(CompleteStep())

# Query result
plan = session[AgentPlan].latest()
assert plan is not None
assert plan.current_step == 1  # Advanced to next step
```

### Initial State Factory

Provide an `initial` factory to auto-initialize empty slices:

```python
@dataclass(slots=True, frozen=True)
class Counters:
    count: int = 0

    @reducer(on=Increment)
    def increment(self, event: Increment) -> Replace["Counters"]:
        return Replace((
            replace(self, count=self.count + event.amount),
        ))

# Install with initial factory
session.install(Counters, initial=lambda: Counters(count=0))

# Can now dispatch without seeding
session.dispatch(Increment(amount=5))
assert session[Counters].latest().count == 5
```

### Reducer Method Signature

Reducer methods must follow this pattern:

```python
@reducer(on=EventType)
def method_name(self, event: EventType) -> Replace[SelfType]:
    """
    self: Current state (most recent value in slice)
    event: Dispatched event instance
    Returns: Replace[T] with new state tuple
    """
    return Replace((new_instance,))
```

**Important constraints:**

- The class must be a **frozen dataclass**
- Event types must be **defined before** the slice class
- Methods receive **only** `self` and `event` (no `context` parameter)
- Methods must return **`Replace[T]`** wrapping a tuple of new state
- Each event type may have **at most one** reducer method per slice

### When to Use Declarative Reducers

**Use `@reducer` methods when:**

- **Multiple event types** - The slice handles several distinct events with custom logic
- **Complex transformations** - Reducers need access to slice state to compute updates
- **Domain modeling** - The slice represents a domain concept with defined behaviors
- **Co-location** - You want reducer logic close to the data it operates on

**Use built-in reducers when:**

- **Event ledgers** - Recording every event unconditionally (`append_all`)
- **Latest-wins semantics** - Only the most recent value matters (`replace_latest`)
- **Key-based updates** - Upserting by a derived key (`upsert_by`, `replace_latest_by`)
- **Simple patterns** - The logic is generic and doesn't need custom code

### Example: Task Management

```python
from dataclasses import dataclass, replace
from weakincentives.runtime.session import reducer, Replace

# Events
@dataclass(slots=True, frozen=True)
class CreateTask:
    task_id: str
    description: str

@dataclass(slots=True, frozen=True)
class MarkComplete:
    task_id: str

@dataclass(slots=True, frozen=True)
class AddNote:
    task_id: str
    note: str

# State slice with declarative reducers
@dataclass(slots=True, frozen=True)
class Task:
    task_id: str
    description: str
    completed: bool = False
    notes: tuple[str, ...] = ()

    @reducer(on=MarkComplete)
    def mark_complete(self, event: MarkComplete) -> Replace["Task"]:
        if self.task_id == event.task_id:
            return Replace((replace(self, completed=True),))
        return Replace((self,))  # No change

    @reducer(on=AddNote)
    def add_note(self, event: AddNote) -> Replace["Task"]:
        if self.task_id == event.task_id:
            return Replace((replace(self, notes=(*self.notes, event.note)),))
        return Replace((self,))  # No change

# Usage
session.install(Task, initial=lambda: Task(task_id="", description=""))

session.dispatch(CreateTask(task_id="1", description="Write docs"))
session.dispatch(AddNote(task_id="1", note="Started draft"))
session.dispatch(MarkComplete(task_id="1"))

task = session[Task].latest()
assert task is not None
assert task.completed
assert len(task.notes) == 1
```

## Dispatch: The Unified Mutation Surface

All session mutations go through `dispatch()` for auditability and consistency:

```python
# Dispatch custom events
session.dispatch(AddStep(step="Implement feature"))
session.dispatch(UpdateConfig(setting="debug", value=True))

# System events (handled specially by the session)
session.dispatch(InitializeSlice(Plan, (initial_plan,)))
session.dispatch(ClearSlice(Task, predicate=lambda t: t.completed))
```

### Broadcast Semantics

When you dispatch an event, **all registered reducers** for that event type are invoked:

```python
# Register multiple reducers for the same event
session[Plan].register(TaskAssigned, update_plan_reducer)
session[Metrics].register(TaskAssigned, update_metrics_reducer)
session[Audit].register(TaskAssigned, log_audit_reducer)

# Dispatch - all three reducers run
session.dispatch(TaskAssigned(task_id="123", assignee="alice"))
```

This enables clean separation of concerns: different slices can react to the same events independently.

### System Mutation Events

WINK provides system events for common operations:

#### InitializeSlice

Replace all values in a slice:

```python
from weakincentives.runtime.session import InitializeSlice

# Replace entire slice with new values
session.dispatch(InitializeSlice(Plan, (initial_plan,)))

# Convenience method (equivalent)
session[Plan].seed(initial_plan)
```

#### ClearSlice

Remove items from a slice:

```python
from weakincentives.runtime.session import ClearSlice

# Clear all items
session.dispatch(ClearSlice(Task))

# Clear items matching predicate
session.dispatch(ClearSlice(
    Task,
    predicate=lambda t: t.completed
))

# Convenience methods (equivalent)
session[Task].clear()
session[Task].clear(lambda t: t.completed)
```

### Slice Accessor Methods

For convenience, slice accessors provide methods that dispatch events internally:

| Method | Dispatches | Description |
|--------|-----------|-------------|
| `seed(values)` | `InitializeSlice` | Initialize or replace the slice |
| `clear()` | `ClearSlice` | Remove all items |
| `clear(predicate)` | `ClearSlice` | Remove matching items |
| `append(value)` | Custom event | Shorthand for dispatching |
| `register(type, reducer)` | _(none)_ | Register a reducer |

```python
# These are equivalent:
session[Plan].seed(initial_plan)
session.dispatch(InitializeSlice(Plan, (initial_plan,)))

# These are equivalent:
session[Task].clear(lambda t: t.completed)
session.dispatch(ClearSlice(Task, predicate=lambda t: t.completed))
```

### Session-Wide Mutations

Some operations affect the entire session:

```python
# Clear all slices (preserves reducer registrations)
session.reset()

# Restore from snapshot (replaces all slices)
session.restore(snapshot)
```

## Snapshots: Capture and Restore

Sessions can be snapshotted and restored, enabling powerful debugging and fault-tolerance patterns.

### Creating Snapshots

```python
# Capture current state
snapshot = session.snapshot()

# Serialize to JSON string
json_str = snapshot.to_json()

# Save to file
from pathlib import Path
Path("session.json").write_text(json_str)
```

### Restoring Snapshots

```python
# Load from JSON
from weakincentives.runtime.session import Snapshot

json_str = Path("session.json").read_text()
snapshot = Snapshot.from_json(json_str)

# Restore session state
session.restore(snapshot)

# Query restored state
plan = session[Plan].latest()
```

### Snapshot Lifecycle

```mermaid
flowchart TB
    subgraph Creation["1. Snapshot Creation"]
        Session1["Session State"]
        Collect["Collect slices<br/>(filtered by policy)"]
        SnapshotObj["Snapshot object"]
    end

    subgraph Serialization["2. Serialization"]
        ToJSON["to_json()"]
        JSONStr["JSON string"]
        Store["Store to file/database"]
    end

    subgraph Loading["3. Loading"]
        Load["Load from storage"]
        Parse["JSON parsing"]
        Validate["Schema validation"]
        FromJSON["Snapshot.from_json()"]
    end

    subgraph Restoration["4. Restoration"]
        Restored["Snapshot object"]
        Apply["session.restore()"]
        Session2["Session State<br/>(restored)"]
    end

    Session1 --> Collect --> SnapshotObj
    SnapshotObj --> ToJSON --> JSONStr --> Store
    Load --> Parse --> Validate --> FromJSON
    FromJSON --> Restored --> Apply --> Session2

    style SnapshotObj fill:#e1f5ff
    style Restored fill:#e1f5ff
    style Session2 fill:#e1ffe1
```

### Use Cases

**Flight recorder for debugging:**

```python
from pathlib import Path

# Save snapshot after each major operation
snapshot_dir = Path("debug/snapshots")
snapshot_dir.mkdir(exist_ok=True)

counter = 0
def save_debug_snapshot():
    global counter
    counter += 1
    snapshot = session.snapshot()
    path = snapshot_dir / f"step-{counter:03d}.json"
    path.write_text(snapshot.to_json())

# Use during development
session.dispatch(SetupEnvironment())
save_debug_snapshot()  # step-001.json

session.dispatch(RunAnalysis())
save_debug_snapshot()  # step-002.json

session.dispatch(GenerateReport())
save_debug_snapshot()  # step-003.json
```

**Rollback on errors:**

```python
# Save checkpoint before risky operation
checkpoint = session.snapshot()

try:
    # Attempt complex operation
    session.dispatch(StartTransaction())
    session.dispatch(ModifyState())
    session.dispatch(CommitChanges())
except Exception as e:
    # Rollback on failure
    logging.error(f"Operation failed: {e}")
    session.restore(checkpoint)
```

**Attach to bug reports:**

```python
def report_bug(error: Exception):
    # Include snapshot in bug report
    snapshot = session.snapshot()

    bug_report = {
        "error": str(error),
        "timestamp": datetime.now(UTC).isoformat(),
        "session_snapshot": snapshot.to_json(),
    }

    # Send to error tracking service
    send_to_bugtracker(bug_report)
```

### Serialization Format

Snapshots serialize to JSON with the following structure:

```json
{
  "version": "1.0",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2024-01-15T10:30:00Z",
  "slices": [
    {
      "type": "myapp.models:Plan",
      "policy": "STATE",
      "items": [
        {
          "steps": ["Research", "Implement", "Test"],
          "current_step": 1
        }
      ]
    },
    {
      "type": "myapp.models:LogEntry",
      "policy": "LOG",
      "items": [
        {"message": "Started", "timestamp": "2024-01-15T10:25:00Z"},
        {"message": "Analysis complete", "timestamp": "2024-01-15T10:28:00Z"}
      ]
    }
  ]
}
```

**Key properties:**

- **version** - Schema version for compatibility checking
- **session_id** - UUID of the session
- **created_at** - Snapshot creation timestamp (ISO 8601)
- **slices** - Array of typed slices with their items
- **type** - Fully qualified type name (`"module:Class"`)
- **policy** - Slice policy (`"STATE"` or `"LOG"`)

### Error Handling

```python
from weakincentives.runtime.session import (
    SnapshotSerializationError,
    SnapshotRestoreError,
)

# Serialization errors
try:
    snapshot.to_json()
except SnapshotSerializationError as e:
    # Unsupported types or unserializable payloads
    logging.error(f"Cannot serialize: {e}")

# Restore errors
try:
    session.restore(snapshot)
except SnapshotRestoreError as e:
    # Incompatible schema or missing types
    logging.error(f"Cannot restore: {e}")
```

## SlicePolicy: State vs. Logs

Not all slices should roll back the same way. WINK distinguishes between:

- **`SlicePolicy.STATE`** - Working state that should be restored on rollback
- **`SlicePolicy.LOG`** - Append-only history that should be preserved

```mermaid
graph TB
    subgraph StateSlices["STATE Slices (Rollback)"]
        Plan["Plan<br/>(current plan)"]
        Config["Config<br/>(settings)"]
        Workspace["Workspace<br/>(file state)"]
    end

    subgraph LogSlices["LOG Slices (Preserved)"]
        Events["Events<br/>(audit trail)"]
        Metrics["Metrics<br/>(telemetry)"]
        Logs["LogEntries<br/>(debug logs)"]
    end

    Snapshot["snapshot()"] -.->|Captures| StateSlices
    Snapshot -.->|Excludes by default| LogSlices

    SnapshotAll["snapshot(include_all=True)"] -.->|Captures| StateSlices
    SnapshotAll -.->|Captures| LogSlices

    style StateSlices fill:#e1f5ff
    style LogSlices fill:#fff4e1
```

### Default Behavior

By default, `session.snapshot()` captures **only `STATE` slices**:

```python
# Captures only STATE slices (default)
snapshot = session.snapshot()

# Excludes LOG slices by default
# - Event history preserved in session
# - Only working state captured
```

### Including All Slices

To capture everything (including logs):

```python
# Capture STATE + LOG slices
snapshot = session.snapshot(include_all=True)
```

### Setting Slice Policy

Configure policy when registering reducers:

```python
from weakincentives.runtime.session import SlicePolicy

# STATE slice (default) - included in snapshots
session[Plan].set_policy(SlicePolicy.STATE)
session[Plan].register(Plan, replace_latest)

# LOG slice - excluded by default
session[AuditEvent].set_policy(SlicePolicy.LOG)
session[AuditEvent].register(AuditEvent, append_all)
```

### Why This Matters

When debugging, you often want to:

1. **Preserve the full event log** even when rolling back working state
2. **Restore working state** to a known-good checkpoint
3. **Keep audit trails intact** for compliance and debugging

Example:

```python
# Setup slices
session[Plan].set_policy(SlicePolicy.STATE)      # Working state
session[Event].set_policy(SlicePolicy.LOG)       # Audit log

# Do work
session.dispatch(CreatePlan())
session.dispatch(AuditEvent("plan_created"))
checkpoint = session.snapshot()  # Saves Plan, not Event

session.dispatch(ModifyPlan())
session.dispatch(AuditEvent("plan_modified"))

# Error occurs - rollback working state
session.restore(checkpoint)

# Result:
# - Plan is restored to checkpoint
# - Event log still has both audit events
```

### Best Practices

**Use `STATE` policy for:**
- Plans and task lists
- Configuration settings
- File system state
- Model outputs and intermediate results

**Use `LOG` policy for:**
- Audit events
- Telemetry metrics
- Debug log entries
- Tool invocation history

## Session Hierarchy

Sessions form a tree structure for nested orchestration:

```mermaid
graph TB
    Root["Root Session<br/>(main agent loop)"]
    Child1["Child Session<br/>(research subtask)"]
    Child2["Child Session<br/>(analysis subtask)"]
    GrandChild1["Grandchild Session<br/>(document retrieval)"]
    GrandChild2["Grandchild Session<br/>(code analysis)"]

    Root --> Child1
    Root --> Child2
    Child1 --> GrandChild1
    Child2 --> GrandChild2

    style Root fill:#e1f5ff
    style Child1 fill:#f0f9ff
    style Child2 fill:#f0f9ff
    style GrandChild1 fill:#f9fcff
    style GrandChild2 fill:#f9fcff
```

### Creating Child Sessions

```python
# Create parent session
parent_session = Session(bus=bus)

# Create child session
child_session = Session(bus=bus, parent=parent_session)

# Access relationship
assert child_session.parent is parent_session
assert child_session in parent_session.children
```

### Traversing the Hierarchy

```python
from weakincentives.runtime.session import iter_sessions_bottom_up

# Traverse from leaves up to root
for session in iter_sessions_bottom_up(root_session):
    snapshot = session.snapshot()
    print(f"Session {session.session_id}: {len(snapshot.slices)} slices")
```

### Use Cases

**Nested task decomposition:**

```python
def execute_subtask(parent: Session, task: Task) -> Result:
    # Create isolated session for subtask
    child = Session(bus=parent.dispatcher, parent=parent)

    # Execute subtask in child session
    child.dispatch(SetupTask(task))
    result = run_subtask_loop(child)

    # Bubble up results to parent
    parent.dispatch(SubtaskComplete(task_id=task.id, result=result))

    return result
```

**Hierarchical snapshots:**

```python
def save_full_tree(root: Session):
    snapshots = []
    for session in iter_sessions_bottom_up(root):
        snapshots.append(session.snapshot())

    # Save all snapshots
    return {"root": snapshots[-1], "children": snapshots[:-1]}
```

## Integration with Prompts and Adapters

Sessions integrate seamlessly with WINK's prompt and adapter systems (see [Chapter 3: Prompts](03-prompts.md) and [Chapter 6: Adapters](06-adapters.md)).

### Passing Sessions to Adapters

```python
from weakincentives.adapters.openai import OpenAIAdapter

adapter = OpenAIAdapter(model="gpt-4")
session = Session(bus=bus)

# Session receives telemetry events
prompt = Prompt(template).bind(params)

with prompt.resources:
    response = adapter.evaluate(prompt, session=session)

# Query session for telemetry
from weakincentives.runtime.events import ToolInvoked, PromptExecuted

tool_calls = session[ToolInvoked].all()
executions = session[PromptExecuted].all()

print(f"Tools invoked: {len(tool_calls)}")
print(f"Prompts executed: {len(executions)}")
```

### Session-Aware Sections

Some sections require session access (see [Chapter 3: Session-Bound Sections](03-prompts.md#session-bound-sections-and-cloning)):

```python
from weakincentives.contrib.tools import PlanningToolsSection, VfsToolsSection

def build_prompt_template(*, session: Session) -> PromptTemplate[Any]:
    return PromptTemplate(
        ns="agent",
        key="task",
        sections=(
            MarkdownSection(
                title="Instructions",
                key="instructions",
                template="Complete the task step by step.",
            ),
            PlanningToolsSection(session=session),  # Requires session
            VfsToolsSection(session=session),        # Requires session
        ),
    )
```

### Tool Handlers and Session Access

Tool handlers receive sessions via `ToolContext`:

```python
from weakincentives.prompt import Tool, ToolContext, ToolResult

def my_handler(params: MyParams, *, context: ToolContext) -> ToolResult[str]:
    # Access session from context
    session = context.session

    # Query session state
    plan = session[Plan].latest()
    if plan is None:
        return ToolResult.error("No plan found")

    # Dispatch events
    session.dispatch(ToolInvoked(name="my_tool", params=params))

    return ToolResult.ok("Success")
```

## Best Practices

### 1. Keep Reducers Pure and Lightweight

Reducers run synchronously on the dispatcher thread. Avoid:

- ❌ Network calls
- ❌ File I/O
- ❌ Heavy computation
- ❌ Mutable state

Prefer:

- ✅ Simple data transformations
- ✅ Immutable operations
- ✅ Fast, deterministic logic

```python
# Bad: Side effects in reducer
def bad_reducer(state: SliceView[Plan], event: AddStep) -> Append[Plan]:
    # DON'T DO THIS
    requests.post("https://api.example.com/log", json={"step": event.step})
    return Append(Plan(steps=(event.step,)))

# Good: Pure transformation
def good_reducer(state: SliceView[Plan], event: AddStep) -> Append[Plan]:
    return Append(Plan(steps=(event.step,)))
```

### 2. Use Meaningful Slice Policies

Be intentional about `STATE` vs. `LOG`:

```python
# State that should rollback
session[Plan].set_policy(SlicePolicy.STATE)
session[Config].set_policy(SlicePolicy.STATE)
session[Workspace].set_policy(SlicePolicy.STATE)

# Logs that should persist
session[AuditEvent].set_policy(SlicePolicy.LOG)
session[Metric].set_policy(SlicePolicy.LOG)
session[ToolInvoked].set_policy(SlicePolicy.LOG)
```

### 3. Snapshot Regularly During Development

Save snapshots at key milestones:

```python
def checkpoint(name: str):
    snapshot = session.snapshot()
    Path(f"debug/{name}.json").write_text(snapshot.to_json())

# Use during development
session.dispatch(SetupEnvironment())
checkpoint("01-setup")

session.dispatch(RunAnalysis())
checkpoint("02-analysis")

session.dispatch(GenerateReport())
checkpoint("03-report")
```

### 4. Use Declarative Reducers for Domain Models

When a slice has complex behavior, co-locate logic with data:

```python
# Good: Domain model with co-located reducers
@dataclass(slots=True, frozen=True)
class PullRequest:
    pr_id: str
    status: str
    approvals: tuple[str, ...] = ()

    @reducer(on=Approve)
    def approve(self, event: Approve) -> Replace["PullRequest"]:
        if event.pr_id == self.pr_id:
            new_approvals = (*self.approvals, event.reviewer)
            new_status = "approved" if len(new_approvals) >= 2 else self.status
            return Replace((replace(
                self,
                approvals=new_approvals,
                status=new_status,
            ),))
        return Replace((self,))
```

### 5. Query Before Dispatch

Check state before dispatching to avoid invalid transitions:

```python
# Bad: Dispatch blindly
session.dispatch(CompleteTask(task_id="123"))

# Good: Check state first
task = session[Task].latest()
if task and not task.completed:
    session.dispatch(CompleteTask(task_id=task.task_id))
```

### 6. Use Session Hierarchies for Isolation

Create child sessions for subtasks to maintain isolation:

```python
def execute_risky_operation(parent: Session) -> Result:
    # Create isolated session
    child = Session(bus=parent.dispatcher, parent=parent)

    try:
        # Execute in isolation
        result = run_operation(child)

        # Success - bubble up results
        parent.dispatch(OperationComplete(result=result))
        return result
    except Exception as e:
        # Failure - child state is discarded
        parent.dispatch(OperationFailed(error=str(e)))
        raise
```

### 7. Test Reducer Behavior

Write tests for reducer logic:

```python
def test_add_step_reducer():
    # Create test session
    session = Session(bus=InProcessDispatcher())
    session[Plan].register(AddStep, add_step_reducer)

    # Dispatch events
    session.dispatch(AddStep(step="First"))
    session.dispatch(AddStep(step="Second"))

    # Assert final state
    plan = session[Plan].latest()
    assert plan is not None
    assert plan.steps == ("First", "Second")
```

### 8. Handle Missing State Gracefully

Always check for `None` when querying:

```python
# Bad: Assumes state exists
plan = session[Plan].latest()
print(plan.steps)  # TypeError if None

# Good: Handle missing state
plan = session[Plan].latest()
if plan is None:
    print("No plan found")
else:
    print(f"Plan has {len(plan.steps)} steps")
```

## Common Patterns

### Pattern: Latest-Wins Singleton

Keep only the most recent value:

```python
from weakincentives.runtime.session import replace_latest

@dataclass(slots=True, frozen=True)
class Config:
    debug: bool
    timeout: int

session[Config].register(Config, replace_latest)

session.dispatch(Config(debug=False, timeout=30))
session.dispatch(Config(debug=True, timeout=60))

# Only the latest remains
assert session[Config].all() == (Config(debug=True, timeout=60),)
```

### Pattern: Append-Only Event Log

Record every event:

```python
from weakincentives.runtime.session import append_all

@dataclass(slots=True, frozen=True)
class AuditEvent:
    action: str
    timestamp: datetime

session[AuditEvent].set_policy(SlicePolicy.LOG)
session[AuditEvent].register(AuditEvent, append_all)

# All events are preserved
session.dispatch(AuditEvent("login", datetime.now(UTC)))
session.dispatch(AuditEvent("query", datetime.now(UTC)))
session.dispatch(AuditEvent("logout", datetime.now(UTC)))

assert len(session[AuditEvent].all()) == 3
```

### Pattern: Key-Based Upsert

Maintain a collection keyed by ID:

```python
from weakincentives.runtime.session import upsert_by

@dataclass(slots=True, frozen=True)
class User:
    user_id: str
    name: str
    email: str

session[User].register(
    User,
    upsert_by(key=lambda u: u.user_id)
)

session.dispatch(User("1", "Alice", "alice@example.com"))
session.dispatch(User("2", "Bob", "bob@example.com"))
session.dispatch(User("1", "Alice Updated", "newemail@example.com"))

# User 1 was updated, User 2 remains
users = session[User].all()
assert len(users) == 2
assert users[0].name == "Alice Updated"
```

### Pattern: Conditional Updates

Only update if certain conditions are met:

```python
@dataclass(slots=True, frozen=True)
class Counter:
    count: int
    max_count: int

    @reducer(on=Increment)
    def increment(self, event: Increment) -> Replace["Counter"]:
        if self.count < self.max_count:
            return Replace((replace(self, count=self.count + 1),))
        return Replace((self,))  # No change - at max
```

### Pattern: Filtered Clear

Remove items matching a condition:

```python
# Clear completed tasks
session[Task].clear(lambda t: t.completed)

# Clear old entries
cutoff = datetime.now(UTC) - timedelta(days=7)
session[LogEntry].clear(lambda e: e.timestamp < cutoff)
```

## 5.7 Storage Backends and Persistence

> **Canonical Reference**: See [specs/SLICES.md](../specs/SLICES.md) for the complete slice storage specification.

### Introduction

By default, session slices live in memory as immutable tuples. This works well for short-lived sessions, but some scenarios demand persistence:

- **Debugging**: Inspecting session state after a crash
- **Audit trails**: Preserving a complete record of tool invocations
- **Long-running agents**: State that outlives process restarts
- **Distributed systems**: Sharing state across multiple workers

WINK solves this through the **Slice protocol**—an abstraction that decouples session operations from storage backends. Sessions work identically whether slices are stored in memory, on disk, or in a database.

This section covers:

1. The Slice protocol and storage abstraction
2. Built-in backends (MemorySlice, JsonlSlice)
3. Configuring storage per slice policy
4. Performance characteristics and trade-offs
5. Implementing custom backends

### The Slice Protocol

The `Slice[T]` protocol defines a common interface for storage backends. Sessions interact only with this protocol, making backends pluggable:

```python
from typing import Protocol
from collections.abc import Callable, Iterable

@runtime_checkable
class Slice[T: SupportsDataclass](Protocol):
    """Storage backend for immutable tuple-based state."""

    def all(self) -> tuple[T, ...]:
        """Return all items as an immutable tuple."""
        ...

    def latest(self) -> T | None:
        """Return the most recent item, or None if empty."""
        ...

    def append(self, item: T) -> None:
        """Append a single item."""
        ...

    def extend(self, items: Iterable[T]) -> None:
        """Append multiple items."""
        ...

    def replace(self, items: tuple[T, ...]) -> None:
        """Replace all items atomically."""
        ...

    def clear(self, predicate: Callable[[T], bool] | None = None) -> None:
        """Remove items. If predicate is None, clear all."""
        ...

    def snapshot(self) -> tuple[T, ...]:
        """Capture current state for serialization."""
        ...

    def view(self) -> SliceView[T]:
        """Return a lazy readonly view for reducer input."""
        ...
```

#### SliceView: Lazy Access for Reducers

Reducers receive a `SliceView[T]` instead of eager tuples. This enables efficient patterns like append-only reducers that never load existing data:

```python
class SliceView[T: SupportsDataclass](Protocol):
    """Readonly lazy view of slice contents."""

    @property
    def is_empty(self) -> bool:
        """Check if slice is empty without loading data."""
        ...

    def latest(self) -> T | None:
        """Return most recent item."""
        ...

    def all(self) -> tuple[T, ...]:
        """Load all items (expensive for file-backed slices)."""
        ...

    def where(self, predicate: Callable[[T], bool]) -> Iterator[T]:
        """Stream items matching predicate."""
        ...
```

For file-backed slices, `is_empty` can stat the file without parsing, and `latest()` can read just the last line. Append-only reducers avoid I/O entirely.

### Factory Configuration

`SliceFactoryConfig` maps slice policies to storage backends. WINK defines two policies:

- **`SlicePolicy.STATE`**: Working state rolled back on failure (e.g., `Plan`, custom agent state)
- **`SlicePolicy.LOG`**: Append-only records preserved during restore (e.g., `ToolInvoked`, `PromptExecuted`)

```python
from weakincentives.runtime.session import SliceFactoryConfig
from weakincentives.runtime.session.slices import MemorySliceFactory, JsonlSliceFactory
from pathlib import Path

# In-memory state, persistent logs
config = SliceFactoryConfig(
    state_factory=MemorySliceFactory(),
    log_factory=JsonlSliceFactory(base_dir=Path("./session_logs")),
)

session = Session(bus=bus, slice_config=config)
```

Each slice type is assigned a policy when registered. The factory creates appropriate backends automatically:

```python
# Register slice with policy
session[Plan].set_policy(SlicePolicy.STATE)
session[ToolInvoked].set_policy(SlicePolicy.LOG)

# Session uses factories to create backends
# Plan -> MemorySliceFactory -> MemorySlice
# ToolInvoked -> JsonlSliceFactory -> JsonlSlice
```

### MemorySlice: The Default Backend

`MemorySlice` stores data as an in-memory tuple. It's simple, fast, and implements both `Slice` and `SliceView` protocols (serving as its own view since all data is already in memory):

```python
@dataclass(slots=True)
class MemorySlice[T: SupportsDataclass]:
    """In-memory tuple-backed slice."""

    _data: tuple[T, ...] = ()

    # SliceView protocol (readonly)
    @property
    def is_empty(self) -> bool:
        return len(self._data) == 0

    def all(self) -> tuple[T, ...]:
        return self._data

    def latest(self) -> T | None:
        return self._data[-1] if self._data else None

    # Slice protocol (mutable)
    def append(self, item: T) -> None:
        self._data = (*self._data, item)  # O(n) copy

    def replace(self, items: tuple[T, ...]) -> None:
        self._data = items

    def view(self) -> "MemorySlice[T]":
        return self  # Self is its own view
```

#### Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `all()` | O(1) | Direct tuple reference |
| `latest()` | O(1) | Index last element |
| `append()` | O(n) | Tuple expansion creates new tuple |
| `replace()` | O(1) | Reassignment |

Memory usage grows linearly with slice size. For large slices (thousands of items), consider alternative backends.

### JsonlSlice: File-Backed Persistence

`JsonlSlice` stores each item as a JSON line in a file. Reads load the entire file; appends are O(1) file operations. Separate `JsonlSliceView` provides lazy access:

```python
@dataclass(slots=True)
class JsonlSlice[T: SupportsDataclass]:
    """JSONL file-backed slice with caching."""

    path: Path
    item_type: type[T]
    _cache: tuple[T, ...] | None = None

    def append(self, item: T) -> None:
        """O(1) append - writes single line without reading file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                data = dump(item, include_dataclass_type=True)
                f.write(json.dumps(data, separators=(",", ":")) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        self._cache = None  # Invalidate

    def all(self) -> tuple[T, ...]:
        """Load all items, using cache if available."""
        if self._cache is not None:
            return self._cache
        result = self._read_all()
        self._cache = result
        return result

    def view(self) -> JsonlSliceView[T]:
        """Return lazy view for reducer input."""
        return JsonlSliceView(
            path=self.path,
            item_type=self.item_type,
            _slice=self,
        )
```

#### JsonlSliceView: Optimized Access

The view provides lazy operations that avoid loading the entire file:

```python
@dataclass(slots=True)
class JsonlSliceView[T: SupportsDataclass]:
    """Lazy readonly view of JsonlSlice."""

    path: Path
    item_type: type[T]
    _slice: JsonlSlice[T]

    @property
    def is_empty(self) -> bool:
        """O(1) check using file stat."""
        if not self.path.exists():
            return True
        return self.path.stat().st_size == 0

    def __iter__(self) -> Iterator[T]:
        """Stream items without loading all into memory."""
        if not self.path.exists():
            return
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    yield parse(self.item_type, data, allow_dataclass_type=True)

    def latest(self) -> T | None:
        """Return last item. Uses cache if available."""
        if self._slice._cache is not None:
            cache = self._slice._cache
            return cache[-1] if cache else None
        # Could optimize to read only last line
        items = self._slice.all()
        return items[-1] if items else None
```

#### Performance Characteristics

| Operation | Cold (no cache) | Cached | Notes |
|-----------|-----------------|--------|-------|
| `append()` | O(1) | O(1) | Single line write |
| `extend()` | O(k) | O(k) | Write k lines |
| `all()` | O(n) file read | O(1) | Lazy loading with cache |
| `latest()` | O(n) file read | O(1) | Could optimize to O(1) by seeking |
| `is_empty` | O(1) stat | O(1) | No file parsing |
| Memory | O(1) base | O(n) cached | Cache persists until write |

#### File Format

Each line is a self-contained JSON object with type information:

```jsonl
{"__type__":"myapp.events:ToolInvoked","tool":"search","result":"found 3 items"}
{"__type__":"myapp.events:ToolInvoked","tool":"read","result":"file contents..."}
```

The `__type__` field enables polymorphic deserialization when slice types use unions or inheritance.

#### Thread Safety

`JsonlSlice` uses `fcntl.flock()` for file-level locking:

- **Shared locks** (`LOCK_SH`) for reads
- **Exclusive locks** (`LOCK_EX`) for writes

Combined with Session's `RLock`, this provides:
- Session-level atomicity for reducer execution
- File-level atomicity for disk operations

For cross-platform support, consider using the `filelock` library or conditional imports for Windows (`msvcrt.locking`).

### Reducer Performance and SliceOp

Reducers return `SliceOp[T]` to describe mutations. The operation type determines efficiency:

```python
from weakincentives.runtime.session import Append, Replace

# Efficient for file-backed slices
def append_only_reducer(
    view: SliceView[AuditLog],
    event: UserAction,
    *,
    context: ReducerContext,
) -> Append[AuditLog]:
    """Never accesses view - O(1) for JSONL."""
    return Append(AuditLog(action=event.name, timestamp=event.at))

# Requires loading existing data
def transform_reducer(
    view: SliceView[Plan],
    event: AddStep,
    *,
    context: ReducerContext,
) -> Replace[Plan]:
    """Must load to transform - O(n) for JSONL."""
    current = view.latest()
    if current is None:
        new_plan = Plan(steps=(event.step,))
    else:
        new_plan = replace(current, steps=(*current.steps, event.step))
    return Replace((new_plan,))
```

#### Performance by Pattern

| Reducer Pattern | MemorySlice | JsonlSlice |
|-----------------|-------------|------------|
| Append-only (never accesses view) | O(n) copy | **O(1) file append** |
| Replace entire slice | O(1) | O(n) file rewrite |
| Transform existing data | O(1) read + O(n) copy | O(n) file read + O(n) rewrite |

**Key insight**: Append-only reducers (common for logs) become O(1) for file-backed slices.

### Configuration Patterns

#### Default: All In-Memory

```python
# Implicit default - no configuration needed
session = Session(bus=bus)
```

All slices use `MemorySlice`. State lives only in memory.

#### Persistent Logs, In-Memory State

```python
config = SliceFactoryConfig(
    state_factory=MemorySliceFactory(),
    log_factory=JsonlSliceFactory(base_dir=Path("./logs")),
)
session = Session(bus=bus, slice_config=config)
```

`ToolInvoked`, `PromptExecuted`, and other `LOG` slices persist. `Plan` and custom state stay in memory.

#### Temporary Debugging Logs

```python
# JsonlSliceFactory() creates a temporary directory
log_factory = JsonlSliceFactory()
config = SliceFactoryConfig(
    state_factory=MemorySliceFactory(),
    log_factory=log_factory,
)
session = Session(bus=bus, slice_config=config)

# Logs written to /tmp/wink_slices_abc123/
print(f"Debug logs: {log_factory.directory}")
```

Useful for debugging without polluting the working directory.

#### All Persistent

```python
jsonl_factory = JsonlSliceFactory(base_dir=Path("./session_data"))
config = SliceFactoryConfig(
    state_factory=jsonl_factory,
    log_factory=jsonl_factory,
)
session = Session(bus=bus, slice_config=config)
```

Everything persists to disk. Slower but enables crash recovery.

### Snapshots and Restore

Snapshots call `slice.snapshot()` for each slice:

```python
def snapshot(self) -> Snapshot:
    state: dict[SessionSliceType, tuple[SupportsDataclass, ...]] = {}
    for slice_type, slice_instance in self._slices.items():
        policy = self._slice_policies.get(slice_type, SlicePolicy.STATE)
        if policy in (SlicePolicy.STATE, SlicePolicy.BOTH):
            state[slice_type] = slice_instance.snapshot()
    return Snapshot(state=state, ...)
```

For `JsonlSlice`, `snapshot()` returns the current cached tuple. The snapshot is independent of file contents—it's a point-in-time capture for session serialization.

Restore uses `slice.replace()`:

```python
def restore(self, snapshot: Snapshot) -> None:
    for slice_type, items in snapshot.state.items():
        policy = self._slice_policies.get(slice_type, SlicePolicy.STATE)
        if policy == SlicePolicy.LOG:
            continue  # Logs are preserved, not restored
        self._get_or_create_slice(slice_type).replace(items)
```

`LOG` slices are preserved during restore—their file contents remain untouched.

### Implementing Custom Backends

To implement a custom backend, satisfy the `Slice[T]` protocol:

```python
from weakincentives.runtime.session import Slice, SliceView

class SqliteSlice[T: SupportsDataclass]:
    """SQLite-backed slice with indexed queries."""

    def __init__(self, db_path: Path, table_name: str, item_type: type[T]):
        self.db_path = db_path
        self.table_name = table_name
        self.item_type = item_type
        self._ensure_table()

    def all(self) -> tuple[T, ...]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(f"SELECT data FROM {self.table_name} ORDER BY id")
        items = []
        for (json_data,) in cursor:
            data = json.loads(json_data)
            items.append(parse(self.item_type, data, allow_dataclass_type=True))
        conn.close()
        return tuple(items)

    def append(self, item: T) -> None:
        data = dump(item, include_dataclass_type=True)
        json_data = json.dumps(data)
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            f"INSERT INTO {self.table_name} (data) VALUES (?)",
            (json_data,),
        )
        conn.commit()
        conn.close()

    # ... other protocol methods
```

#### Factory for Custom Backend

```python
class SqliteSliceFactory:
    """Factory for SQLite-backed slices."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def create[T: SupportsDataclass](self, slice_type: type[T]) -> SqliteSlice[T]:
        table_name = f"{slice_type.__module__}_{slice_type.__name__}"
        return SqliteSlice(self.db_path, table_name, slice_type)
```

Then configure:

```python
config = SliceFactoryConfig(
    state_factory=MemorySliceFactory(),
    log_factory=SqliteSliceFactory(Path("./logs.db")),
)
```

#### Protocol Compliance Testing

WINK provides abstract test suites for validating backends:

```python
from weakincentives.runtime.session.slices import FilesystemProtocolTests

class TestSqliteSlice(FilesystemProtocolTests):
    def create_slice(self) -> Slice[MyDataclass]:
        return SqliteSlice(Path(":memory:"), "test", MyDataclass)

    # Inherits comprehensive protocol tests
```

### Error Handling

#### File I/O Errors

`JsonlSlice` propagates I/O errors (permission denied, disk full):

```python
try:
    session.dispatch(event)
except OSError as e:
    logger.error(f"Slice persistence failed: {e}")
    # Consider fallback to memory-only mode
```

#### Serialization Errors

Non-serializable fields raise during `dump()`:

```python
@dataclass(frozen=True)
class BadSlice:
    callback: Callable[[], None]  # Cannot serialize

# Raises when JsonlSlice tries to write
session[BadSlice].append(BadSlice(lambda: None))
```

Ensure slice types use only serializable fields (primitives, nested dataclasses, standard collections).

#### Deserialization Errors

Corrupt files or missing types raise during `parse()`:

```python
try:
    items = slice.all()
except (json.JSONDecodeError, TypeError, ValueError) as e:
    logger.error(f"Failed to load {slice.path}: {e}")
    # Consider clearing and starting fresh
    slice.clear()
```

### Recommendations

#### When to Use Each Backend

| Scenario | Recommendation |
|----------|----------------|
| Short-lived sessions (&lt; 1 hour) | `MemorySliceFactory` for everything |
| Debugging/audit needs | `JsonlSliceFactory` for LOG slices only |
| Crash recovery required | `JsonlSliceFactory` for both STATE and LOG |
| Large slices (&gt; 10,000 items) | Consider custom backend (SQLite, Redis) |
| Distributed agents | Custom backend with shared storage |

#### Performance Tips

1. **Use append-only reducers** when possible—they're O(1) for file-backed slices
2. **Enable caching** by reading `all()` once and reusing the result
3. **Batch operations** with `extend()` instead of multiple `append()` calls
4. **Clear old data** periodically to prevent unbounded growth
5. **Use memory for working state**, files for logs

#### Limitations

- **No partial reads**: `all()` loads the entire slice into memory
- **No indexing**: File-backed slices don't support efficient key-based lookup
- **Single-process locking**: `fcntl.flock` doesn't work across NFS or distributed filesystems
- **No streaming writes**: Each append is a separate file operation

For advanced use cases (indexed queries, distributed locking, streaming), implement a custom backend.

### Summary

WINK's slice abstraction decouples session operations from storage. This enables:

- **Flexible persistence**: Choose backends per slice policy
- **Efficient patterns**: Append-only reducers become O(1) for files
- **Custom backends**: Implement the protocol for specialized storage
- **Backward compatibility**: In-memory default matches existing behavior

The protocol, not the implementation, is the contract. Sessions remain agnostic to storage details, and tools work identically across backends.

For most use cases, `MemorySlice` for state and `JsonlSlice` for logs provides the right balance of simplicity and observability. When requirements demand more, the protocol enables extension without modifying core session logic.

## Summary

WINK's session system provides:

- **Deterministic memory** - Pure reducers ensure reproducible state transitions
- **Event-driven updates** - All mutations flow through `dispatch()` for auditability
- **Typed slices** - Each dataclass type gets its own immutable tuple
- **Rich query API** - Access state with `latest()`, `all()`, and `where()`
- **SliceOp types** - `Append`, `Extend`, `Replace`, `Clear` for precise control
- **Declarative reducers** - Co-locate logic with data using `@reducer`
- **Snapshotting** - Capture and restore entire session state as JSON
- **Slice policies** - Distinguish working state (`STATE`) from logs (`LOG`)
- **Hierarchical organization** - Sessions form trees for nested orchestration
- **Pluggable storage** - MemorySlice, JsonlSlice, or custom backends

The session is the foundation of WINK's deterministic architecture. Every state change is explicit, auditable, and reproducible—making agent behavior predictable and debuggable at scale.

## Next Steps

- **[Chapter 4: Tools](04-tools.md)** - Learn about sandboxed, deterministic tool execution
- **[Chapter 6: Adapters](06-adapters.md)** - Integrate with OpenAI, Anthropic, and other providers
- **[Chapter 7: Main Loop](07-main-loop.md)** - Orchestrate long-running agent execution
- **[Chapter 11: Prompt Optimization](11-prompt-optimization.md)** - Use sessions for A/B testing prompts
- **[Chapter 13: Debugging](13-debugging.md)** - Inspect snapshots in the debug UI

---

**Canonical Reference**: See [specs/SESSIONS.md](../specs/SESSIONS.md) for the complete specification, including event system details, deadline enforcement, budget tracking, and snapshot serialization format.

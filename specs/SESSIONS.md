# Session State Specification

## Goals and Scope

- **Purpose**: Offer a deterministic, side-effect-free container that mirrors the
  full lifecycle of a run—prompt rendering, tool invocation, and prompt execution—so
  orchestrators and inspectors can reason about every step.
- **What Sessions cover**: In-process event subscription, typed state accumulation,
  snapshot capture/rollback, and small helper selectors. They intentionally avoid
  execution control, retries, or transport; those belong to the runtime and adapter
  layers.
- **Consumers**: Prompt/runtime adapters that publish events, higher-level planners
  that query accumulated slices, and tooling that needs reproducible snapshots for
  regression or audit scenarios.

## Guiding Principles

- **Pure state transitions**: Reducers never mutate in place; every event produces a
  new tuple so state remains predictable and thread-friendly.
- **Typed first**: Event payloads route by concrete dataclass type to keep slices
  minimal and avoid ad-hoc tagging.
- **Deterministic playback**: Sessions respond only to the supported event set,
  making state easy to reconstruct from a recorded stream.
- **Composable slices**: Multiple reducers per type and the ability to project onto
  alternate slice types keep derivations explicit instead of hiding logic in callers.
- **Snapshot fidelity**: Serialization/rollback must be transparent—restoring state
  never rewires reducers or bus subscriptions.

## Minimal Requirements

1. **Typed routing** – Dataclass payloads/outputs map 1:1 to dedicated slices, while
   every tool invocation is also recorded in the generic `ToolInvoked` slice even
   when the handler fails and returns `value=None`.
1. **Pure reducers** – State transitions are pure
   `(tuple[T, ...], DataEvent[T], *, context=ReducerContext) -> tuple[T, ...]`
   functions. Reducers run synchronously on the publisher thread.
1. **Event bus integration** – Sessions subscribe to `PromptRendered`, `ToolInvoked`,
   and `PromptExecuted` from `weakincentives.runtime.events`. They ignore every
   other event type. Each session and event carries a UUID identifier plus a
   timezone-aware creation timestamp for downstream correlation.
1. **Default behavior that works** – Without custom reducers, the Session appends new
   dataclass values (deduping by equality) and keeps results immutable.
1. **Flexible slices** – Reducers may manage a slice whose element type differs from
   the incoming dataclass value so Sessions can maintain derived aggregates.

## Terminology

- `DataEvent[T]` – Events dispatched to reducers. `ToolInvoked` and
  `PromptExecuted` provide a `.value` field that may reference a dataclass payload
  or `None`. `PromptRendered` exposes itself via `.value` to align with the same
  reducer contract.
- **Slice** – The tuple of accumulated dataclass instances managed for a reducer.
  The slice element type may be the same as `T` or a separate dataclass.
- **Reducer** – A pure function responsible for producing a new slice when a
  matching `DataEvent[T]` arrives.
- **Snapshot** – An immutable value object that captures the mapping of slice
  types to their tuples plus metadata such as timestamps.
- **Snapshot tuple** – The tuple stored for a specific slice within a snapshot;
  it mirrors the reducer-managed slice tuple at capture time.
- **Snapshot tags** – A string-to-string mapping persisted with every snapshot to
  annotate the capture context (e.g., role, scope, orchestration metadata).
- **Session hierarchy** – Sessions can be nested. Each session knows its parent
  (if any) and the set of directly registered child sessions to enable
  tree-aware operations.

## Data Model

```python
from weakincentives.runtime.events import PromptExecuted, PromptRendered, ToolInvoked
from weakincentives.runtime.session import ReducerContext

DataEvent = ToolInvoked | PromptExecuted | PromptRendered

class TypedReducer(Protocol[S]):
    def __call__(
        self,
        slice_values: tuple[S, ...],
        event: DataEvent,
        *,
        context: ReducerContext,
    ) -> tuple[S, ...]: ...
```

Session state is an immutable mapping from a **slice type** to a tuple of dataclass
instances. When the slice type matches the data type, the tuple contains the raw
event values. Reducers may also register an explicit slice type to store derived
structures.

## State Management Behaviors

- **Lifecycle hooks**: Constructing a session immediately wires the event listeners;
  no additional setup steps are required to start collecting data.
- **Immutable slices**: Every dispatch path returns a fresh tuple. Default reducers
  dedupe by equality, but callers can opt into upsert/replace semantics when
  registering reducers.
- **Dispatch ordering**: Events flow through reducers synchronously on the publisher
  thread. When multiple reducers share a data type they are invoked in the order
  they were registered so deterministic derivations are possible.
- **Slice typing**: Each reducer owns its slice. `slice_type` controls the stored
  tuple type and can differ from the incoming event payload type, enabling derived
  aggregates without bolting logic onto callers.
- **Fallback behavior**: If no reducer is registered for a payload type the default
  append reducer is used. If no slice exists for a dispatched event, it is created
  on demand via the registered reducer or default behavior.
- **Hierarchy plumbing**: Sessions optionally accept a parent session reference on
  construction. Children register themselves with the parent, and each session
  exposes its immediate descendants. Hierarchy metadata stays out of snapshots so
  rollback remains local to each session's slices.

## Session Hierarchy

Sessions form a tree to mirror nested orchestration flows. Parent/child links must
be deterministic and immutable after construction so ancestry remains stable during
dispatch. Hierarchy awareness enables tools to walk the tree and perform
topological operations without mutating state.

- **Parent awareness**: A session stores its parent as a read-only reference.
  Passing no parent yields a root session. Parent references are for navigation,
  not state sharing, so reducer registration and event subscriptions remain scoped
  to each session instance.
- **Child registration**: Constructing a session with a parent automatically adds
  it to the parent's direct children set. Children are tracked by identity to avoid
  duplicate registrations and keep traversal deterministic.
- **Snapshot fan-in**: Callers that need a whole-tree snapshot can traverse from
  the leaves upward, capturing each session's snapshot and aggregating the results
  in leaf-first order. Parent sessions do not implicitly serialize child state;
  tooling must walk the tree explicitly when building aggregate reports. The
  `iter_sessions_bottom_up` helper yields sessions from the leaves up to simplify
  traversal without mutating session state.

## Public API Surface

```
src/weakincentives/session/
  session.py      # Session, DataEvent, reducer plumbing
  reducers.py     # append (default), helper factories
  selectors.py    # optional helpers built on top of Session.select_all
```

### Session

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

    def clone(
        self,
        *,
        bus: EventBus,
        parent: Session | None = None,
        session_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> "Session": ...

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


@dataclass(slots=True, frozen=True)
class Snapshot:
    created_at: datetime
    slices: Mapping[type[Any], tuple[Any, ...]]

    def to_json(self) -> str: ...

    @classmethod
    def from_json(cls, raw: str) -> Snapshot: ...
```

### Hierarchy helpers

```python
from weakincentives.runtime.session import iter_sessions_bottom_up
```

- `iter_sessions_bottom_up(root: Session) -> Iterator[Session]` yields sessions
  starting from the leaves up to the provided root session, skipping duplicates
  to protect against accidental cycles.

- Constructing with a bus subscribes to `PromptRendered`, `ToolInvoked`, and
  `PromptExecuted` immediately.

- Multiple reducers may register for the same data type; they run in registration
  order and each maintains its own slice.

- `slice_type` defaults to `data_type`. When provided it controls the tuple type the
  reducer receives and returns.

- Reducers never mutate the previous tuple; they always return a new tuple instance.

- `clone` produces a new Session instance that preserves the current state snapshot
  and reducer registrations. Passing `bus`, `session_id`, or `created_at` overrides
  the copied metadata; omitted parameters reuse the original values. Clones attach
  to the provided event bus without modifying the original session subscription.

- When callers omit `session_id` or `created_at`, the constructor automatically
  generates a UUID (`uuid4()`) and captures `datetime.now(UTC)` respectively. The
  timestamp MUST be timezone-aware; naive datetimes raise `ValueError`.

- `snapshot()` captures the immutable slice mapping without blocking event
  publication and returns a value object representing the captured state.

- `rollback()` replaces the Session's slice mapping with the tuples stored in the
  provided snapshot without emitting new events.

### Reducers

Provide three reducers (each accepts the keyword-only `context` argument and may
ignore it when unused):

1. `append` – Default behavior, dedupes by equality.
1. `upsert_by(key_fn)` – Replaces items that share the same derived key.
1. `replace_latest` – Stores only the most recent value.

`ReducerContext` instances bundle the active `Session` and event bus for the
current invocation. Callers construct a fresh context per event and pass it as the
`context=` keyword argument so reducers can inspect session state or publish their
own events without resorting to global lookups.

### Selectors

A minimal helper module offers:

- `select_all(session, slice_type)` – Returns the tuple slice for the registered type.
- `select_latest(session, slice_type)` – Returns the last item or `None`.
- `select_where(session, slice_type, pred)` – Filters the slice by predicate.

These helpers delegate to `Session.select_all` and perform no caching.

## Event Handling Rules

1. On `PromptRendered`, dispatch the rendered prompt event directly so reducers
   can persist prompt metadata alongside later outputs.
1. On `ToolInvoked`, dispatch the event regardless of success. When
   `result.value` is a dataclass, re-dispatch the same event (with `.value`
   populated) to the payload's concrete type.
1. On `PromptExecuted`, dispatch the event to its dedicated slice. If the
   response contains dataclass outputs, re-dispatch the event with `.value`
   populated for each dataclass item (cloning the event when multiple payloads
   exist).
1. Every normalized `DataEvent` is passed to the reducer chain registered for its
   target type. Each reducer operates on the tuple registered for its `slice_type`.
   If no reducer exists, use the default append reducer.
1. Reducer failures are caught and logged; the slice stays unchanged.

## Snapshot Capture and Rollback

Snapshots add time-travel support so callers can capture the Session's slice
mapping, serialize it for storage or transport, and later roll the Session back to
that exact point. They behave as immutable value objects—copying, storing, and
restoring a snapshot never mutates the original payload or the Session's event
subscriptions. Reducer registrations, event bus wiring, and pending dispatch
state stay outside the snapshot because they are inherent to the Session
instance.

### Serialization Strategy

Snapshots serialize to JSON using only primitive types. Persist metadata fields
(`created_at` as a timezone-aware ISO 8601 string plus an API version) alongside
the captured slices. Each slice entry serializes as a dictionary containing
`"slice_type"`, `"item_type"`, and `"items"`. Type fields use the fully qualified
`"package.module:Class"` form, and items rely on the existing dataclass serde
helpers (`src/weakincentives/serde/`). Reducer slices whose element type differs
from the incoming dataclass must still record the item type explicitly. The
`Snapshot.to_json()` and `Snapshot.from_json()` helpers round-trip the payload; the
deserialized snapshot must compare equal to the original.

### State Capture Semantics

1. `snapshot()` reads the immutable slice mapping and packages it into a new
   `Snapshot` without requiring defensive copies or synchronization.
1. Active reducers finish before capture completes; events published afterward are
   excluded. Callers should snapshot only after their own dispatch loop completes
   to avoid missing in-flight updates.
1. `rollback()` replaces the current slice mapping with the tuples stored in the
   snapshot. Slices present in the Session but absent from the snapshot clear to
   empty tuples. Reducer registrations persist and continue receiving events with
   their restored tuples.

### Error Handling

- `snapshot()` raises `SnapshotSerializationError` when encountering unsupported
  slice types or unserializable payloads and leaves the Session unchanged.
- `rollback()` raises `SnapshotRestoreError` if the snapshot schema version is
  incompatible or if referenced slice types were never registered. On failure the
  Session state remains untouched.

## Known Limitations and Edge Cases

- **Supported events only**: Sessions ignore any runtime events beyond
  `PromptRendered`, `ToolInvoked`, and `PromptExecuted`. Callers that introduce new
  event types must normalize them before publishing or extend the session to route
  them.
- **Synchronous reducers**: Because reducers run on the publisher thread, long-lived
  reducer logic can block downstream listeners. Keep reducers lightweight or move
  heavier work behind new events.
- **Dataclass focus**: Typed slices expect dataclass payloads that are compatible
  with the existing serde helpers. Non-dataclass payloads only populate the generic
  `ToolInvoked` slice and cannot be serialized in snapshots unless wrapped in a
  dataclass.
- **No implicit eviction**: State grows with each event. Callers must register
  reducers that compact or replace entries (e.g., `replace_latest`) when only recent
  values matter.
- **Snapshot compatibility**: Restoring requires the referenced slice types to be
  available in the runtime environment and registered with the session. Incompatible
  schema versions or missing types raise restore errors rather than partially
  applying state.

## Testing Checklist

- Publishing `ToolInvoked` / `PromptExecuted` populates the correct slice exactly once.
- Reducer registration order is respected, and absence of reducers falls back to the
  default append implementation.
- Non-dataclass payloads populate the generic `ToolInvoked` slice while leaving
  dataclass-specific slices untouched.
- Reducers may register a distinct `slice_type` and still receive the correct tuple.
- `Session.snapshot()` / `Session.rollback()` round-trip keeps slice tuples and
  timestamps consistent (timestamp deltas tolerated when comparing equality if
  necessary).
- Reducers registered before snapshot capture remain functional after rollback
  and continue processing future events with the restored tuples.
- Snapshot serialization and restore failures raise the documented exceptions and
  leave Session state untouched.

## Usage Sketch

```python
bus = InProcessEventBus()
session = Session(bus=bus)

# Optional reducer wiring
session.register_reducer(SourceDetails, upsert_by(lambda d: d.source_id))
session.register_reducer(
    ResearchSummary,
    reducer_that_updates_metrics,
    slice_type=ResearchMetrics,
)

# Evaluate prompts/tools with adapters that publish events...

latest_summary = select_latest(session, ResearchSummary)
metrics = select_all(session, ResearchMetrics)
```

Sessions keep the orchestration loop deterministic and easy to test while staying as
small as possible.

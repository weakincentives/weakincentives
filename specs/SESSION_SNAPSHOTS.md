# Session Snapshots Specification

## Overview

Sessions provide pure, deterministic state containers. Snapshots add time-travel
support so callers can capture the full Session state at a given moment,
serialize it for persistence or transport, and later roll the Session back to
that exact point. Snapshots must behave like value objects: copying, storing,
and restoring them never mutates the original snapshot or the Session's
subscribers.

## Target API

```python
class Session:
    def snapshot(self) -> Snapshot: ...
    def rollback(self, snapshot: Snapshot) -> None: ...

@dataclass(slots=True, frozen=True)
class Snapshot:
    session_id: str
    created_at: datetime
    slices: Mapping[type[Any], tuple[Any, ...]]
    reducers: Mapping[type[Any], tuple[RegisteredReducer, ...]]
    pending_events: tuple[DataEvent[Any], ...]
```

- `snapshot()` captures all state at once; it never blocks event publication.
- `rollback()` replaces the Session's state with the snapshot contents without
  emitting new events.
- `Snapshot` instances must be hashable and equality comparable for deterministic
  testing and caching.

## Serialization Strategy

Snapshots must serialize to JSON for transport and storage. The serialized form
contains only primitives: strings, numbers, booleans, and arrays. Use the
existing dataclass serde helpers (`src/weakincentives/serde/`) to convert
payloads into dictionaries. Serialization consists of four layers:

1. **Metadata** – Persist `session_id`, `created_at`, and an API version string.
1. **Slices** – Each registered slice serializes as a dictionary with keys
   `"slice_type"`, `"item_type"`, and `"items"`. The type fields use the fully
   qualified module + class name (`"package.module:Class"`). Items serialize with
   the existing dataclass serde. Reducer slices whose element type differs from
   the data type must still serialize item type explicitly.
1. **Reducers** – Persist registration order for deterministic replay. Each
   reducer entry records the fully qualified reducer callable (module path) and
   the associated `data_type` and `slice_type` using the same notation as slices.
1. **Pending events** – Sessions may buffer events mid-dispatch. Snapshots must
   record any in-flight `DataEvent` values to ensure rollbacks during an active
   publish can resume cleanly. Pending events serialize with the same serde path
   as slice items.

Provide helper functions:

```python
Snapshot.to_json(self) -> str
Snapshot.from_json(cls, raw: str) -> Snapshot
```

Round-tripping `Snapshot -> json -> Snapshot` must preserve equality.

## State Capture Semantics

1. `snapshot()` acquires a read lock on the Session's internal state, clones the
   immutable mappings, and releases the lock before performing serialization.
   Session state stays immutable so snapshots can reuse existing tuples without
   copying.
1. Active reducers finish before the snapshot captures state; new events received
   afterward are not part of the snapshot. Document that callers should snapshot
   only after their own dispatch completes to avoid missing concurrent events.
1. Rollbacks replace the current state with the snapshot payload, including the
   pending events queue. Reducer registrations are restored and re-subscribed to
   the event bus without re-running historical events.
1. Rollback never replays `pending_events`. Instead, the Session resumes handling
   them in order the next time the dispatcher runs.

## Reducer Registration Handling

- Reducer registration order and identities are part of the snapshot. Replaying a
  snapshot must deregister existing reducers, clear reducer state, re-register
  reducers in their original order, and restore each slice tuple.
- Reducer callables must be importable from their module path. If import fails,
  raise `SnapshotRestoreError` with the fully qualified name.
- Snapshots do not capture dynamically created reducers (lambdas, closures). If a
  reducer cannot be serialized (no module + qualname), `snapshot()` raises a
  `SnapshotSerializationError` instructing developers to provide a stable
  reference.

## Error Handling

- `snapshot()` raises `SnapshotSerializationError` for unsupported slice types,
  unserializable dataclass payloads, or reducers without stable identities.
- `rollback()` raises `SnapshotRestoreError` if the snapshot `session_id` differs
  from the Session, if reducer imports fail, or if the serialized schema version
  is incompatible.
- Both methods leave the Session untouched when exceptions occur.

## Testing Checklist

- **Round-trip parity** – `session.snapshot().to_json()` and restoring through
  `Session.rollback(Snapshot.from_json(...))` leaves the Session equal to the
  original state (including slices, reducer order, and pending events).
- **Reducer replay** – Custom reducers registered before the snapshot remain
  functional after rollback and continue receiving events.
- **Error surfaces** – Attempting to snapshot unsupported reducers or restoring
  with invalid data raises the documented exceptions and preserves current state.
- **Concurrency guard** – Simulated concurrent `snapshot()` calls do not corrupt
  state and return independent `Snapshot` values.
- **Pending events** – Rolling back during an active publish preserves pending
  events and resumes processing in order.

## Open Questions

- Should snapshots optionally prune specific slices to reduce payload size?
- Do we need delta snapshots to optimize large histories, or is full capture
  acceptable for near-term workloads?
- Should rollback emit diagnostics or events for observability tooling?

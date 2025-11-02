# Session Snapshots Specification

## Overview

Sessions provide pure, deterministic state containers. Snapshots add time-travel
support so callers can capture the full Session slice mapping at a given moment,
serialize it for persistence or transport, and later roll the Session back to
that exact point. Snapshots must behave like value objects: copying, storing,
and restoring them never mutates the original snapshot or the Session's
subscribers.

Snapshots only capture immutable slice tuples plus a timestamp. Reducer
registrations, event bus wiring, and pending dispatch state stay outside the
snapshot because they are inherent to the Session instance itself (see
`SESSIONS.md`). Rollback simply swaps the Session's stored slices with the
snapshot payload.

## Target API

```python
class Session:
    def snapshot(self) -> Snapshot: ...
    def rollback(self, snapshot: Snapshot) -> None: ...

@dataclass(slots=True, frozen=True)
class Snapshot:
    created_at: datetime
    slices: Mapping[type[Any], tuple[Any, ...]]
```

- `snapshot()` captures all slice state at once; it never blocks event
  publication.
- `rollback()` replaces the Session's slice mapping with the snapshot contents
  without emitting new events.
- `Snapshot` instances must be hashable and equality comparable for deterministic
  testing and caching.

## Serialization Strategy

Snapshots must serialize to JSON for transport and storage. The serialized form
contains only primitives: strings, numbers, booleans, and arrays. Use the
existing dataclass serde helpers (`src/weakincentives/serde/`) to convert
payloads into dictionaries. Serialization consists of two layers:

1. **Metadata** – Persist `created_at` (timezone-aware ISO 8601 string) and an
   API version string.
1. **Slices** – Each registered slice serializes as a dictionary with keys
   `"slice_type"`, `"item_type"`, and `"items"`. The type fields use the fully
   qualified module + class name (`"package.module:Class"`). Items serialize with
   the existing dataclass serde. Reducer slices whose element type differs from
   the data type must still serialize item type explicitly.

Provide helper functions:

```python
Snapshot.to_json(self) -> str
Snapshot.from_json(cls, raw: str) -> Snapshot
```

Round-tripping `Snapshot -> json -> Snapshot` must preserve equality.

## State Capture Semantics

1. `snapshot()` reads the Session's immutable slice mapping and packages it into
   a new `Snapshot`. Because slices are already stored as tuples, capturing a
   snapshot does not require copying or synchronization.
1. Active reducers finish before the snapshot captures state; new events received
   afterward are not part of the snapshot. Document that callers should snapshot
   only after their own dispatch completes to avoid missing in-flight updates.
1. Rollbacks replace the current slice mapping with the snapshot payload. Reducer
   registrations are not serialized; the Session keeps existing reducers and
   simply replaces the tuple values they manage. Slices present in the Session
   but absent from the snapshot clear to empty tuples.

## Error Handling

- `snapshot()` raises `SnapshotSerializationError` for unsupported slice types or
  unserializable dataclass payloads.
- `rollback()` raises `SnapshotRestoreError` if the serialized schema version is
  incompatible or if slice types referenced in the snapshot were never
  registered.
- Both methods leave the Session untouched when exceptions occur.

## Testing Checklist

- **Round-trip parity** – `session.snapshot().to_json()` and restoring through
  `Session.rollback(Snapshot.from_json(...))` leaves the Session equal to the
  original state (same slices, timestamp differences tolerated when comparing
  equality if necessary).
- **Reducer replay** – Custom reducers registered before the snapshot remain
  functional after rollback and continue receiving events with their restored
  slices.
- **Error surfaces** – Attempting to snapshot unsupported slice contents or
  restoring with invalid data raises the documented exceptions and preserves
  current state.

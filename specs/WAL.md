# Write-Ahead Log (WAL) Specification

## Purpose

This spec defines a **Write-Ahead Log** abstraction for `EventBus` instances,
enabling deterministic replay of session state transitions. The WAL captures
every published event in an ordered, durable sequence, allowing reconstruction
of session state from any point in history.

Primary use cases:

- **Debugging**: Replay exact event sequences that led to a bug
- **Testing**: Capture golden event traces for regression tests
- **Recovery**: Restore session state after crash or restart
- **Auditing**: Maintain complete history of agent decisions

## Guiding Principles

- **Append-only**: WAL entries are immutable once written; no updates or deletes
- **Event ordering**: Strict total ordering via monotonic sequence numbers
- **Decoupled storage**: In-memory by default, disk persistence is opt-in
- **Zero-copy replay**: Replay mechanism uses original event instances when possible
- **Minimal overhead**: WAL operations must not significantly impact publish latency

## Goals

1. Capture all events published through an EventBus in sequence order
2. Support full replay from genesis or partial replay from checkpoints
3. Provide in-memory implementation suitable for single-session use
4. Allow optional disk serialization for persistence across restarts
5. Integrate cleanly with existing `EventBus` protocol without modification

## Non-Goals

- Distributed WAL coordination (single-process only)
- Real-time streaming to external systems
- Automatic compaction or garbage collection
- Binary wire format optimization (JSON-based serialization)

---

## Core Concepts

### WAL Entry

Each entry wraps a published event with ordering metadata:

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4


@dataclass(slots=True, frozen=True)
class WALEntry:
    """Immutable record of a published event."""

    sequence: int                    # Monotonic, gap-free sequence number (0-indexed)
    event: object                    # The original event instance
    event_type: str                  # Fully-qualified type name for deserialization
    timestamp: datetime              # When entry was created (not event's created_at)
    entry_id: UUID = field(default_factory=uuid4)

    @property
    def event_type_class(self) -> type[object]:
        """Resolve event_type string back to class. Raises if not importable."""
        ...
```

### WAL Protocol

The core abstraction that any WAL implementation must satisfy:

```python
from typing import Protocol, Iterator, Callable
from collections.abc import Sequence


class WAL(Protocol):
    """Write-ahead log for event replay."""

    @property
    def length(self) -> int:
        """Number of entries in the log."""
        ...

    @property
    def last_sequence(self) -> int | None:
        """Sequence number of most recent entry, or None if empty."""
        ...

    def append(self, event: object) -> WALEntry:
        """
        Append event to log. Returns the created entry.

        Thread-safe. Assigns next sequence number atomically.
        """
        ...

    def get(self, sequence: int) -> WALEntry | None:
        """Retrieve entry by sequence number. Returns None if not found."""
        ...

    def slice(self, start: int = 0, end: int | None = None) -> Sequence[WALEntry]:
        """
        Return entries in range [start, end).

        If end is None, returns all entries from start to end of log.
        """
        ...

    def iter_from(self, sequence: int = 0) -> Iterator[WALEntry]:
        """Iterate entries starting from given sequence number."""
        ...

    def clear(self) -> None:
        """Remove all entries. Use with caution."""
        ...
```

### Checkpoint

A checkpoint marks a known-good state boundary for partial replay:

```python
@dataclass(slots=True, frozen=True)
class WALCheckpoint:
    """Marker for a consistent state boundary."""

    sequence: int                    # Sequence number at checkpoint
    checkpoint_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
```

---

## In-Memory Implementation

The primary implementation stores entries in a thread-safe list:

```python
from threading import RLock
from weakincentives.dbc import require, ensure, invariant


@invariant(lambda self: self._sequence >= 0, "sequence must be non-negative")
class InMemoryWAL:
    """Thread-safe in-memory write-ahead log."""

    def __init__(self) -> None:
        self._entries: list[WALEntry] = []
        self._sequence: int = 0
        self._lock: RLock = RLock()
        self._checkpoints: list[WALCheckpoint] = []

    @property
    def length(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def last_sequence(self) -> int | None:
        with self._lock:
            return self._sequence - 1 if self._entries else None

    @ensure(lambda self, event, result: result.sequence == self._sequence - 1)
    def append(self, event: object) -> WALEntry:
        with self._lock:
            entry = WALEntry(
                sequence=self._sequence,
                event=event,
                event_type=f"{type(event).__module__}.{type(event).__qualname__}",
                timestamp=datetime.utcnow(),
            )
            self._entries.append(entry)
            self._sequence += 1
            return entry

    def get(self, sequence: int) -> WALEntry | None:
        with self._lock:
            if 0 <= sequence < len(self._entries):
                return self._entries[sequence]
            return None

    def slice(self, start: int = 0, end: int | None = None) -> tuple[WALEntry, ...]:
        with self._lock:
            return tuple(self._entries[start:end])

    def iter_from(self, sequence: int = 0) -> Iterator[WALEntry]:
        # Snapshot to avoid holding lock during iteration
        with self._lock:
            snapshot = tuple(self._entries[sequence:])
        yield from snapshot

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._sequence = 0
            self._checkpoints.clear()

    def checkpoint(self, metadata: dict[str, Any] | None = None) -> WALCheckpoint:
        """Create a checkpoint at current position."""
        with self._lock:
            cp = WALCheckpoint(
                sequence=self._sequence - 1 if self._entries else -1,
                metadata=metadata or {},
            )
            self._checkpoints.append(cp)
            return cp

    @property
    def checkpoints(self) -> tuple[WALCheckpoint, ...]:
        with self._lock:
            return tuple(self._checkpoints)
```

---

## EventBus Integration

### WALEventBus Decorator

A decorator that wraps any `EventBus` to automatically log all published events:

```python
from weakincentives.runtime.events import EventBus, PublishResult, EventHandler


class WALEventBus:
    """EventBus wrapper that logs all events to a WAL."""

    def __init__(self, bus: EventBus, wal: WAL | None = None) -> None:
        self._bus = bus
        self._wal = wal or InMemoryWAL()

    @property
    def wal(self) -> WAL:
        """Access the underlying WAL."""
        return self._wal

    def subscribe(self, event_type: type[object], handler: EventHandler) -> None:
        self._bus.subscribe(event_type, handler)

    def unsubscribe(self, event_type: type[object], handler: EventHandler) -> bool:
        return self._bus.unsubscribe(event_type, handler)

    def publish(self, event: object) -> PublishResult:
        # Log BEFORE delivery (write-ahead semantics)
        self._wal.append(event)
        return self._bus.publish(event)
```

### Usage Pattern

```python
from weakincentives.runtime import Session, InProcessEventBus

# Create WAL-enabled bus
inner_bus = InProcessEventBus()
wal = InMemoryWAL()
bus = WALEventBus(inner_bus, wal)

# Session uses the wrapped bus normally
session = Session(bus=bus)

# Later: access event history
for entry in bus.wal.iter_from(0):
    print(f"[{entry.sequence}] {entry.event_type}: {entry.event}")
```

---

## Replay Mechanism

### WALReplayer

Replays WAL entries through an EventBus to reconstruct state:

```python
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ReplayResult:
    """Summary of a replay operation."""

    entries_replayed: int
    start_sequence: int
    end_sequence: int
    errors: tuple[Exception, ...] = ()

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


class WALReplayer:
    """Replays WAL entries through an EventBus."""

    def __init__(self, wal: WAL, bus: EventBus) -> None:
        self._wal = wal
        self._bus = bus

    def replay_all(self) -> ReplayResult:
        """Replay all entries from the beginning."""
        return self.replay_from(0)

    def replay_from(self, sequence: int) -> ReplayResult:
        """Replay entries starting from given sequence."""
        return self.replay_range(sequence, None)

    def replay_from_checkpoint(self, checkpoint: WALCheckpoint) -> ReplayResult:
        """Replay entries after the given checkpoint."""
        return self.replay_from(checkpoint.sequence + 1)

    def replay_range(self, start: int, end: int | None) -> ReplayResult:
        """Replay entries in range [start, end)."""
        errors: list[Exception] = []
        count = 0
        last_seq = start - 1

        for entry in self._wal.slice(start, end):
            try:
                self._bus.publish(entry.event)
                count += 1
                last_seq = entry.sequence
            except Exception as e:
                errors.append(e)

        return ReplayResult(
            entries_replayed=count,
            start_sequence=start,
            end_sequence=last_seq + 1,
            errors=tuple(errors),
        )
```

### Replay Usage

```python
# Capture events during original session
original_bus = WALEventBus(InProcessEventBus())
original_session = Session(bus=original_bus)

# ... run agent, events are logged ...

# Later: replay into fresh session
replay_bus = InProcessEventBus()
replay_session = Session(bus=replay_bus)

replayer = WALReplayer(original_bus.wal, replay_bus)
result = replayer.replay_all()

assert result.ok
# replay_session now has identical state to original_session
```

---

## Disk Persistence

### Serialization Format

WAL entries serialize to JSON Lines (`.jsonl`) format:

```json
{"sequence": 0, "event_type": "weakincentives.runtime.events.PromptRendered", "timestamp": "2024-01-15T10:30:00Z", "entry_id": "...", "event": {...}}
{"sequence": 1, "event_type": "weakincentives.runtime.events.PromptExecuted", "timestamp": "2024-01-15T10:30:01Z", "entry_id": "...", "event": {...}}
```

### WALDumper

Writes WAL contents to disk:

```python
from pathlib import Path
import json
from weakincentives.serde import to_dict, from_dict


class WALDumper:
    """Serializes WAL to disk."""

    def __init__(self, wal: WAL) -> None:
        self._wal = wal

    def dump(self, path: Path) -> int:
        """
        Write all entries to file. Returns number of entries written.

        Overwrites existing file.
        """
        count = 0
        with path.open("w", encoding="utf-8") as f:
            for entry in self._wal.iter_from(0):
                record = {
                    "sequence": entry.sequence,
                    "event_type": entry.event_type,
                    "timestamp": entry.timestamp.isoformat(),
                    "entry_id": str(entry.entry_id),
                    "event": to_dict(entry.event),
                }
                f.write(json.dumps(record) + "\n")
                count += 1
        return count

    def dump_incremental(self, path: Path, from_sequence: int) -> int:
        """Append entries from sequence onward to existing file."""
        count = 0
        with path.open("a", encoding="utf-8") as f:
            for entry in self._wal.iter_from(from_sequence):
                record = {
                    "sequence": entry.sequence,
                    "event_type": entry.event_type,
                    "timestamp": entry.timestamp.isoformat(),
                    "entry_id": str(entry.entry_id),
                    "event": to_dict(entry.event),
                }
                f.write(json.dumps(record) + "\n")
                count += 1
        return count
```

### WALLoader

Loads WAL from disk:

```python
from importlib import import_module


class WALLoader:
    """Deserializes WAL from disk."""

    def load(self, path: Path) -> InMemoryWAL:
        """Load WAL from file. Returns new InMemoryWAL instance."""
        wal = InMemoryWAL()

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                event_class = self._resolve_type(record["event_type"])
                event = from_dict(event_class, record["event"])

                # Reconstruct entry with original metadata
                entry = WALEntry(
                    sequence=record["sequence"],
                    event=event,
                    event_type=record["event_type"],
                    timestamp=datetime.fromisoformat(record["timestamp"]),
                    entry_id=UUID(record["entry_id"]),
                )
                wal._entries.append(entry)
                wal._sequence = entry.sequence + 1

        return wal

    def _resolve_type(self, type_name: str) -> type[object]:
        """Import and return class from fully-qualified name."""
        module_name, class_name = type_name.rsplit(".", 1)
        module = import_module(module_name)
        return getattr(module, class_name)
```

### Persistence Usage

```python
from pathlib import Path

# After session completes
dumper = WALDumper(bus.wal)
dumper.dump(Path("session_trace.jsonl"))

# Later: restore and replay
loader = WALLoader()
restored_wal = loader.load(Path("session_trace.jsonl"))

replay_bus = InProcessEventBus()
replay_session = Session(bus=replay_bus)
replayer = WALReplayer(restored_wal, replay_bus)
replayer.replay_all()
```

---

## Checkpointing Strategies

### Manual Checkpoints

Create checkpoints at meaningful boundaries:

```python
# After tool completes
result = tool_handler(params, context=ctx)
bus.wal.checkpoint(metadata={"tool": tool_name, "success": result.success})

# After each turn
for turn in conversation:
    process_turn(turn)
    bus.wal.checkpoint(metadata={"turn": turn.id})
```

### Automatic Checkpoints

Checkpoint every N events:

```python
class AutoCheckpointWAL:
    """WAL wrapper that auto-checkpoints every N appends."""

    def __init__(self, wal: InMemoryWAL, interval: int = 100) -> None:
        self._wal = wal
        self._interval = interval
        self._since_checkpoint = 0

    def append(self, event: object) -> WALEntry:
        entry = self._wal.append(event)
        self._since_checkpoint += 1
        if self._since_checkpoint >= self._interval:
            self._wal.checkpoint(metadata={"auto": True})
            self._since_checkpoint = 0
        return entry
```

---

## Thread Safety

All WAL operations are thread-safe:

- `append()` uses atomic sequence assignment under lock
- `iter_from()` creates a snapshot to avoid holding lock during iteration
- `slice()` returns immutable tuple, safe to share across threads
- `checkpoint()` is atomic with respect to appends

Concurrent appends are serialized but do not block reads of already-written
entries.

---

## Error Handling

### Append Failures

The in-memory implementation cannot fail on append (barring OOM). Disk-backed
implementations should:

1. Write to temporary file first
2. Fsync before acknowledging
3. Raise `WALWriteError` on I/O failure

### Replay Failures

`ReplayResult.errors` captures exceptions from individual event handlers.
Replay continues through errors by default. For strict replay:

```python
result = replayer.replay_all()
if not result.ok:
    raise ExceptionGroup("Replay failed", result.errors)
```

### Deserialization Failures

`WALLoader` raises if:

- Event type cannot be imported (`ModuleNotFoundError`, `AttributeError`)
- Event data cannot be deserialized (`SerdeError`)

Callers should handle these at load time, not replay time.

---

## Limitations

1. **Memory bound**: `InMemoryWAL` grows unbounded; long-running sessions need
   periodic dumping or external storage
2. **No compaction**: Old entries never removed; implement externally if needed
3. **Single process**: No coordination for distributed replay
4. **Type resolution**: Deserialization requires event classes to be importable
   at load time with matching structure
5. **No encryption**: Disk format is plaintext JSON; encrypt at filesystem level
   if needed

---

## Future Considerations

The following are explicitly out of scope for initial implementation but may be
added later:

- **SQLite backend**: Bounded, queryable storage without memory limits
- **Event filtering**: Replay only specific event types
- **Compression**: Gzip JSON Lines for disk storage
- **Schema versioning**: Handle event class evolution across versions
- **Streaming dump**: Write entries to disk as they're appended

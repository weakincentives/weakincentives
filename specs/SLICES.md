# Slice Storage Specification

## Purpose

Slice protocol abstracts storage backends for session state. Enables different
implementations (in-memory, file-backed) while maintaining Redux-style semantics.
Factories configure per `SlicePolicy`, allowing logs to persist to disk while
keeping working state in memory.

**Implementation:** `src/weakincentives/runtime/session/slices/`

## Guiding Principles

- **Single access pattern**: Operations work identically regardless of backend
- **Policy-driven configuration**: STATE and LOG can use different factories
- **Immutable semantics**: All backends present tuple-like views; mutations create versions
- **Backend-managed persistence**: File backends handle I/O; no session logic required
- **Serialization via serde**: File backends use `weakincentives.serde` for dataclass serialization

## Protocol Definition

### Slice Protocol

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `all()` | - | `tuple[T, ...]` | All items as immutable tuple |
| `latest()` | - | `T \| None` | Most recent item |
| `append(item)` | `item: T` | `None` | Append single item |
| `extend(items)` | `items: Iterable[T]` | `None` | Append multiple items |
| `replace(items)` | `items: tuple[T, ...]` | `None` | Replace all atomically |
| `clear(predicate)` | `predicate: Callable[[T], bool] \| None` | `None` | Remove items (all if None) |
| `__len__()` | - | `int` | Item count |
| `snapshot()` | - | `tuple[T, ...]` | Snapshot for serialization |
| `view()` | - | `SliceView[T]` | Lazy readonly view |

### SliceView Protocol

Lazy readonly view for reducer input, enabling append-only reducers to avoid loading data.

| Method/Property | Returns | Description |
|-----------------|---------|-------------|
| `is_empty` | `bool` | O(1) check without loading |
| `__len__()` | `int` | Item count |
| `__iter__()` | `Iterator[T]` | Lazy iteration |
| `all()` | `tuple[T, ...]` | Load all (expensive) |
| `latest()` | `T \| None` | Most recent item |
| `where(predicate)` | `Iterator[T]` | Filtered iteration |

### SliceOp Types

Reducers return `SliceOp[T]` describing mutations:

| Type | Field | Description |
|------|-------|-------------|
| `Append[T]` | `item: T` | Append single item (O(1) for file-backed) |
| `Extend[T]` | `items: tuple[T, ...]` | Append multiple items |
| `Replace[T]` | `items: tuple[T, ...]` | Replace entire slice |
| `Clear[T]` | `predicate: Callable \| None` | Clear items |

### SliceFactory Protocol

Creates slice backends for dataclass types.

### SliceFactoryConfig

Maps `SlicePolicy` to factories:
- `state_factory`: For STATE slices (rolled back on failure)
- `log_factory`: For LOG slices (preserved during restore)

## Reducer Contract

Reducers receive `SliceView` and return `SliceOp`:

```
reducer(view: SliceView[S], event: Event, *, context: ReducerContext) -> SliceOp[S]
```

### Built-in Reducers

| Reducer | Description | View Access | Performance |
|---------|-------------|-------------|-------------|
| `append_all` | Append event (ledger semantics) | None | O(1) file-backed |
| `replace_latest` | Keep only most recent | None | O(1) |
| `upsert_by(key_fn)` | Upsert by derived key | Full | O(n) |

### Declarative Reducers

The `@reducer` decorator enables method-style reducers on frozen dataclasses.
Methods return `SliceOp` directly. See `runtime/session/state_slice.py`.

## Implementations

### MemorySlice

In-memory tuple-backed slice. O(1) reads, O(n) appends. Serves as its own view.
Default backend for all policies.

### MemorySliceFactory

Creates `MemorySlice` instances. No configuration required.

### JsonlSlice

JSONL file-backed slice for persistent storage. Each item stored as single JSON
line with `__type__` field for polymorphic deserialization.

**Key characteristics:**
- Append-optimized I/O (O(1) appends)
- Lazy view with optimized `is_empty`, `latest()`, streaming iteration
- Write-through cache invalidation
- Thread safety via `fcntl.flock` (POSIX) or `msvcrt.locking` (Windows)

### JsonlSliceFactory

Creates JSONL file-backed slices. Each slice type gets its own file based on
qualified class name.

**Configuration:**
- `base_dir: Path | None` - Directory for slice files (temp dir if None)

## Performance Characteristics

| Operation | MemorySlice | JsonlSlice (cached) | JsonlSlice (cold) |
|-----------|-------------|---------------------|-------------------|
| `all()` | O(1) | O(1) | O(n) file read |
| `latest()` | O(1) | O(1) | O(n) file read |
| `append()` | O(n) copy | **O(1) file append** | **O(1) file append** |
| `replace()` | O(1) | O(n) file write | O(n) file write |
| `view.is_empty` | O(1) | **O(1) file stat** | **O(1) file stat** |

**Key insight**: Append-only reducers become O(1) for file-backed slices.

## Session Integration

Session uses `SliceFactoryConfig` to create slices on demand:

```python
session = Session(
    dispatcher=dispatcher,
    slice_config=SliceFactoryConfig(
        state_factory=MemorySliceFactory(),
        log_factory=JsonlSliceFactory(base_dir=Path("./logs")),
    ),
)
```

## Snapshot and Restore

- `snapshot()` calls `slice.snapshot()` for each slice
- `restore()` uses `slice.replace()` to set state
- LOG slices are preserved (not restored)
- File-backed snapshots load from disk; contents are separate persistence

## Thread Safety

- **MemorySlice**: Delegated to Session's RLock
- **JsonlSlice**: File-level locking (`fcntl.flock`) combined with Session's RLock

## Error Handling

| Error Type | Cause |
|------------|-------|
| `OSError` | File I/O errors (permission, disk full) |
| `json.JSONDecodeError` | Corrupt JSONL files |
| `TypeError`/`ValueError` | Deserialization errors |

## File Format

```jsonl
{"__type__":"myapp.events:ToolInvoked","name":"search","params":{"query":"test"}}
```

The `__type__` field enables polymorphic deserialization for union types.

## Recommendations

- **Short-lived sessions**: MemorySliceFactory for everything
- **Debug/audit needs**: JsonlSliceFactory for LOG slices only
- **Crash recovery**: JsonlSliceFactory for both (accept performance cost)

## Limitations

- **No partial reads**: `all()` loads entire slice
- **No indexing**: No indexed queries on file-backed slices
- **Single-process locking**: `fcntl.flock` doesn't work across NFS
- **Eager caching**: JsonlSlice caches all items; large slices may exhaust memory

## Related Specifications

- `specs/SESSIONS.md` - Session lifecycle, reducers, snapshots
- `specs/DATACLASSES.md` - Serde utilities for serialization
- `specs/THREAD_SAFETY.md` - Concurrency patterns

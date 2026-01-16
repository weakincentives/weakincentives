# Checkpointer Specification

## Purpose

The Checkpointer system provides durable persistence of Session and Filesystem
snapshots to external storage backends (e.g., S3) during agent execution. This
enables recovery from failures, debugging of past runs, and resumption of
long-running tasks from known-good states.

**Implementation location:** `src/weakincentives/runtime/checkpointer/`

## Principles

- **Request-keyed storage.** All checkpoints are organized by `request_id`,
  providing a stable identifier that survives retries and enables correlation
  across execution attempts.
- **Time-based triggers.** Checkpointing is evaluated after each tool call
  against a configurable time threshold, balancing durability with storage
  costs.
- **Backend-agnostic protocol.** Storage backends implement a protocol,
  allowing S3, filesystem, or custom implementations without changing
  application code.
- **Atomic snapshots.** Checkpoints capture both Session and Filesystem state
  atomically using the existing `CompositeSnapshot` infrastructure.
- **Minimal overhead.** Checkpointing only occurs when the trigger condition is
  met; skipped checks impose negligible cost.

## Core Types

### CheckpointStorage Protocol

The storage protocol defines how checkpoints are persisted and retrieved:

```python
from typing import Protocol
from weakincentives.runtime.transactions import CompositeSnapshot


class CheckpointStorage(Protocol):
    """Protocol for checkpoint storage backends."""

    def save(
        self,
        request_id: str,
        snapshot: CompositeSnapshot,
        *,
        attempt: int = 1,
    ) -> CheckpointRef:
        """
        Persist a composite snapshot for the given request.

        Args:
            request_id: Stable identifier from MainLoopRequest.
            snapshot: Atomic capture of session and filesystem state.
            attempt: Delivery attempt number for disambiguation.

        Returns:
            Reference to the stored checkpoint.
        """
        ...

    def load(self, ref: CheckpointRef) -> CompositeSnapshot:
        """
        Load a checkpoint by reference.

        Raises:
            CheckpointNotFoundError: If the checkpoint does not exist.
            CheckpointCorruptedError: If deserialization fails.
        """
        ...

    def load_latest(
        self,
        request_id: str,
        *,
        attempt: int | None = None,
    ) -> CompositeSnapshot | None:
        """
        Load the most recent checkpoint for a request.

        Args:
            request_id: Stable identifier from MainLoopRequest.
            attempt: If provided, filter to specific attempt.

        Returns:
            Most recent snapshot, or None if no checkpoints exist.
        """
        ...

    def list_checkpoints(
        self,
        request_id: str,
        *,
        attempt: int | None = None,
    ) -> Sequence[CheckpointRef]:
        """
        List all checkpoints for a request, ordered by creation time.
        """
        ...

    def delete(self, ref: CheckpointRef) -> None:
        """Delete a checkpoint. Idempotent if already deleted."""
        ...
```

### CheckpointRef

Reference to a stored checkpoint:

```python
@dataclass(frozen=True, slots=True)
class CheckpointRef:
    """Immutable reference to a stored checkpoint."""

    request_id: str
    snapshot_id: UUID
    attempt: int
    created_at: datetime
    storage_key: str  # Backend-specific location (e.g., S3 key)
```

### CheckpointTrigger Protocol

Triggers determine when checkpointing should occur:

```python
class CheckpointTrigger(Protocol):
    """Protocol for checkpoint trigger strategies."""

    def should_checkpoint(
        self,
        *,
        last_checkpoint_at: datetime | None,
        tool_call_count: int,
        run_context: RunContext,
    ) -> bool:
        """
        Evaluate whether a checkpoint should be taken.

        Called after each tool call completes.

        Args:
            last_checkpoint_at: Timestamp of most recent checkpoint, or None.
            tool_call_count: Number of tool calls since last checkpoint.
            run_context: Current execution context.

        Returns:
            True if a checkpoint should be taken.
        """
        ...

    def reset(self) -> None:
        """Reset trigger state (e.g., after checkpoint is taken)."""
        ...
```

### TimeTrigger

The default trigger implementation:

```python
@dataclass(frozen=True, slots=True)
class TimeTrigger:
    """
    Trigger checkpoints based on elapsed time since last checkpoint.

    Default interval is 3 minutes.
    """

    interval: timedelta = field(default_factory=lambda: timedelta(minutes=3))

    def should_checkpoint(
        self,
        *,
        last_checkpoint_at: datetime | None,
        tool_call_count: int,
        run_context: RunContext,
    ) -> bool:
        if last_checkpoint_at is None:
            # No checkpoint yet; only checkpoint after first tool call
            return tool_call_count > 0

        elapsed = datetime.now(UTC) - last_checkpoint_at
        return elapsed >= self.interval

    def reset(self) -> None:
        pass  # Stateless trigger
```

### Checkpointer

The main coordinator that ties storage and triggers together:

```python
@dataclass(slots=True)
class Checkpointer:
    """
    Coordinates checkpoint creation based on trigger conditions.

    Evaluates trigger after each tool call and persists snapshots
    to the configured storage backend.
    """

    storage: CheckpointStorage
    trigger: CheckpointTrigger = field(default_factory=TimeTrigger)
    _last_checkpoint_at: datetime | None = field(default=None, init=False)
    _tool_call_count: int = field(default=0, init=False)

    def on_tool_call_complete(
        self,
        snapshot: CompositeSnapshot,
        run_context: RunContext,
    ) -> CheckpointRef | None:
        """
        Evaluate checkpoint trigger after a tool call completes.

        Args:
            snapshot: Current composite snapshot from tool transaction.
            run_context: Execution context with request_id.

        Returns:
            CheckpointRef if a checkpoint was taken, None otherwise.
        """
        self._tool_call_count += 1

        if not self.trigger.should_checkpoint(
            last_checkpoint_at=self._last_checkpoint_at,
            tool_call_count=self._tool_call_count,
            run_context=run_context,
        ):
            return None

        ref = self.storage.save(
            request_id=run_context.request_id,
            snapshot=snapshot,
            attempt=run_context.attempt,
        )

        self._last_checkpoint_at = datetime.now(UTC)
        self._tool_call_count = 0
        self.trigger.reset()

        return ref

    def load_latest(self, request_id: str) -> CompositeSnapshot | None:
        """Load most recent checkpoint for a request."""
        return self.storage.load_latest(request_id)

    def reset(self) -> None:
        """Reset internal state for a new request."""
        self._last_checkpoint_at = None
        self._tool_call_count = 0
        self.trigger.reset()
```

## Storage Backends

### S3CheckpointStorage

Primary production backend using Amazon S3:

```python
@dataclass(frozen=True, slots=True)
class S3CheckpointStorage:
    """
    S3-backed checkpoint storage.

    Checkpoints are stored with the following key structure:
        {prefix}/{request_id}/attempt-{attempt}/{snapshot_id}.json.gz

    Metadata is stored in S3 object metadata for efficient listing.
    """

    bucket: str
    prefix: str = "checkpoints"
    client: S3Client = field(default_factory=lambda: boto3.client("s3"))
    compression: bool = True

    def save(
        self,
        request_id: str,
        snapshot: CompositeSnapshot,
        *,
        attempt: int = 1,
    ) -> CheckpointRef:
        key = self._build_key(request_id, snapshot.snapshot_id, attempt)
        body = self._serialize(snapshot)

        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            ContentEncoding="gzip" if self.compression else "identity",
            Metadata={
                "request_id": request_id,
                "snapshot_id": str(snapshot.snapshot_id),
                "attempt": str(attempt),
                "created_at": snapshot.metadata.created_at.isoformat(),
            },
        )

        return CheckpointRef(
            request_id=request_id,
            snapshot_id=snapshot.snapshot_id,
            attempt=attempt,
            created_at=snapshot.metadata.created_at,
            storage_key=key,
        )

    def load(self, ref: CheckpointRef) -> CompositeSnapshot:
        try:
            response = self.client.get_object(
                Bucket=self.bucket,
                Key=ref.storage_key,
            )
            return self._deserialize(response["Body"].read())
        except self.client.exceptions.NoSuchKey:
            raise CheckpointNotFoundError(ref)

    def load_latest(
        self,
        request_id: str,
        *,
        attempt: int | None = None,
    ) -> CompositeSnapshot | None:
        refs = self.list_checkpoints(request_id, attempt=attempt)
        if not refs:
            return None
        return self.load(refs[-1])  # Last is most recent

    def list_checkpoints(
        self,
        request_id: str,
        *,
        attempt: int | None = None,
    ) -> Sequence[CheckpointRef]:
        prefix = f"{self.prefix}/{request_id}/"
        if attempt is not None:
            prefix = f"{prefix}attempt-{attempt}/"

        paginator = self.client.get_paginator("list_objects_v2")
        refs: list[CheckpointRef] = []

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                # Parse metadata from key or fetch object metadata
                ref = self._ref_from_object(obj, request_id)
                if ref is not None:
                    refs.append(ref)

        return sorted(refs, key=lambda r: r.created_at)

    def delete(self, ref: CheckpointRef) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=ref.storage_key)

    def _build_key(
        self,
        request_id: str,
        snapshot_id: UUID,
        attempt: int,
    ) -> str:
        ext = ".json.gz" if self.compression else ".json"
        return f"{self.prefix}/{request_id}/attempt-{attempt}/{snapshot_id}{ext}"

    def _serialize(self, snapshot: CompositeSnapshot) -> bytes:
        data = composite_snapshot_to_json(snapshot).encode("utf-8")
        if self.compression:
            return gzip.compress(data)
        return data

    def _deserialize(self, data: bytes) -> CompositeSnapshot:
        if self.compression:
            data = gzip.decompress(data)
        return composite_snapshot_from_json(data.decode("utf-8"))
```

### InMemoryCheckpointStorage

For testing and development:

```python
@dataclass(slots=True)
class InMemoryCheckpointStorage:
    """
    In-memory checkpoint storage for testing.

    Not suitable for production use.
    """

    _store: dict[str, CompositeSnapshot] = field(default_factory=dict)
    _refs: list[CheckpointRef] = field(default_factory=list)

    def save(
        self,
        request_id: str,
        snapshot: CompositeSnapshot,
        *,
        attempt: int = 1,
    ) -> CheckpointRef:
        key = f"{request_id}/{attempt}/{snapshot.snapshot_id}"
        self._store[key] = snapshot

        ref = CheckpointRef(
            request_id=request_id,
            snapshot_id=snapshot.snapshot_id,
            attempt=attempt,
            created_at=snapshot.metadata.created_at,
            storage_key=key,
        )
        self._refs.append(ref)
        return ref

    def load(self, ref: CheckpointRef) -> CompositeSnapshot:
        if ref.storage_key not in self._store:
            raise CheckpointNotFoundError(ref)
        return self._store[ref.storage_key]

    def load_latest(
        self,
        request_id: str,
        *,
        attempt: int | None = None,
    ) -> CompositeSnapshot | None:
        refs = self.list_checkpoints(request_id, attempt=attempt)
        if not refs:
            return None
        return self.load(refs[-1])

    def list_checkpoints(
        self,
        request_id: str,
        *,
        attempt: int | None = None,
    ) -> Sequence[CheckpointRef]:
        refs = [r for r in self._refs if r.request_id == request_id]
        if attempt is not None:
            refs = [r for r in refs if r.attempt == attempt]
        return sorted(refs, key=lambda r: r.created_at)

    def delete(self, ref: CheckpointRef) -> None:
        self._store.pop(ref.storage_key, None)
        self._refs = [r for r in self._refs if r.storage_key != ref.storage_key]

    def clear(self) -> None:
        """Clear all stored checkpoints."""
        self._store.clear()
        self._refs.clear()
```

### FileCheckpointStorage

Local filesystem storage for development and debugging:

```python
@dataclass(frozen=True, slots=True)
class FileCheckpointStorage:
    """
    Local filesystem checkpoint storage.

    Useful for development and debugging. Not recommended for production.
    """

    root: Path
    compression: bool = True

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        request_id: str,
        snapshot: CompositeSnapshot,
        *,
        attempt: int = 1,
    ) -> CheckpointRef:
        path = self._build_path(request_id, snapshot.snapshot_id, attempt)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = composite_snapshot_to_json(snapshot).encode("utf-8")
        if self.compression:
            data = gzip.compress(data)

        path.write_bytes(data)

        return CheckpointRef(
            request_id=request_id,
            snapshot_id=snapshot.snapshot_id,
            attempt=attempt,
            created_at=snapshot.metadata.created_at,
            storage_key=str(path),
        )

    def _build_path(
        self,
        request_id: str,
        snapshot_id: UUID,
        attempt: int,
    ) -> Path:
        ext = ".json.gz" if self.compression else ".json"
        return self.root / request_id / f"attempt-{attempt}" / f"{snapshot_id}{ext}"
```

## Alternative Triggers

### ToolCountTrigger

Checkpoint after a fixed number of tool calls:

```python
@dataclass(slots=True)
class ToolCountTrigger:
    """Trigger checkpoint after N tool calls."""

    threshold: int = 10
    _count: int = field(default=0, init=False)

    def should_checkpoint(
        self,
        *,
        last_checkpoint_at: datetime | None,
        tool_call_count: int,
        run_context: RunContext,
    ) -> bool:
        return tool_call_count >= self.threshold

    def reset(self) -> None:
        pass  # Uses tool_call_count from Checkpointer
```

### CompositeTrigger

Combine multiple triggers with AND/OR logic:

```python
@dataclass(frozen=True, slots=True)
class CompositeTrigger:
    """
    Combine multiple triggers.

    With mode="any", checkpoint if any trigger fires.
    With mode="all", checkpoint only if all triggers fire.
    """

    triggers: tuple[CheckpointTrigger, ...]
    mode: Literal["any", "all"] = "any"

    def should_checkpoint(
        self,
        *,
        last_checkpoint_at: datetime | None,
        tool_call_count: int,
        run_context: RunContext,
    ) -> bool:
        results = [
            t.should_checkpoint(
                last_checkpoint_at=last_checkpoint_at,
                tool_call_count=tool_call_count,
                run_context=run_context,
            )
            for t in self.triggers
        ]

        if self.mode == "any":
            return any(results)
        return all(results)

    def reset(self) -> None:
        for trigger in self.triggers:
            trigger.reset()
```

## Integration with MainLoop

The Checkpointer integrates with MainLoop through the adapter's tool execution
flow:

```python
from weakincentives.runtime import MainLoop, MainLoopConfig
from weakincentives.runtime.checkpointer import (
    Checkpointer,
    S3CheckpointStorage,
    TimeTrigger,
)


# Configure checkpointer
storage = S3CheckpointStorage(bucket="my-agent-checkpoints")
checkpointer = Checkpointer(
    storage=storage,
    trigger=TimeTrigger(interval=timedelta(minutes=3)),
)

# Pass to MainLoop
loop = MainLoop(
    prompt=my_prompt,
    adapter=my_adapter,
    mailbox=my_mailbox,
    config=MainLoopConfig(checkpointer=checkpointer),
)
```

### Adapter Integration

The adapter calls `on_tool_call_complete` after each successful tool execution:

```python
# In adapter tool execution flow (simplified)
def _execute_tool_with_checkpoint(
    self,
    tool_call: ToolCall,
    run_context: RunContext,
    checkpointer: Checkpointer | None,
) -> ToolResult:
    # Execute tool within transaction
    with tool_transaction(session, resources) as snapshot:
        result = self._invoke_handler(tool_call, context)

    # Evaluate checkpoint trigger after successful execution
    if checkpointer is not None and result.success:
        ref = checkpointer.on_tool_call_complete(
            snapshot=snapshot,
            run_context=run_context,
        )
        if ref is not None:
            logger.info(
                "Checkpoint saved",
                request_id=run_context.request_id,
                snapshot_id=str(ref.snapshot_id),
            )

    return result
```

### Resuming from Checkpoint

To resume a request from its last checkpoint:

```python
async def handle_resume(
    request_id: str,
    checkpointer: Checkpointer,
    session: Session,
    resources: ResourceRegistry,
) -> bool:
    """
    Restore session and resources from the latest checkpoint.

    Returns True if a checkpoint was found and restored.
    """
    snapshot = checkpointer.load_latest(request_id)
    if snapshot is None:
        return False

    # Restore using existing transaction infrastructure
    restore_composite_snapshot(snapshot, session, resources)
    return True
```

## Error Handling

### Exception Types

```python
class CheckpointError(Exception):
    """Base class for checkpoint errors."""
    pass


class CheckpointNotFoundError(CheckpointError):
    """Raised when a checkpoint reference cannot be found."""

    def __init__(self, ref: CheckpointRef) -> None:
        self.ref = ref
        super().__init__(f"Checkpoint not found: {ref.storage_key}")


class CheckpointCorruptedError(CheckpointError):
    """Raised when checkpoint data cannot be deserialized."""

    def __init__(self, ref: CheckpointRef, cause: Exception) -> None:
        self.ref = ref
        self.cause = cause
        super().__init__(f"Checkpoint corrupted: {ref.storage_key}: {cause}")


class CheckpointStorageError(CheckpointError):
    """Raised when storage operations fail."""

    def __init__(self, operation: str, cause: Exception) -> None:
        self.operation = operation
        self.cause = cause
        super().__init__(f"Storage operation '{operation}' failed: {cause}")
```

### Failure Modes

| Failure | Behavior |
|---------|----------|
| Storage write fails | Log error, continue execution (checkpoint is best-effort) |
| Storage read fails | Raise `CheckpointStorageError`, caller decides recovery |
| Checkpoint not found | Return `None` from `load_latest`, raise from `load` |
| Corrupted data | Raise `CheckpointCorruptedError` with original cause |
| Trigger evaluation fails | Log error, skip checkpoint, continue execution |

Checkpointing failures are non-fatal by default. The agent continues execution
even if a checkpoint cannot be persisted. This ensures that transient storage
issues do not block agent progress.

## Checkpoint Lifecycle

### Key Structure (S3)

Checkpoints are organized hierarchically by request and attempt:

```
checkpoints/
├── {request_id_1}/
│   ├── attempt-1/
│   │   ├── {snapshot_id_a}.json.gz
│   │   └── {snapshot_id_b}.json.gz
│   └── attempt-2/
│       └── {snapshot_id_c}.json.gz
└── {request_id_2}/
    └── attempt-1/
        └── {snapshot_id_d}.json.gz
```

### Retention

Checkpoint retention is managed externally (e.g., S3 lifecycle policies).
The Checkpointer does not automatically delete old checkpoints. Recommended
retention strategies:

1. **S3 Lifecycle Rules**: Delete checkpoints older than 7 days
2. **Request Completion**: Delete checkpoints when request succeeds
3. **Manual Cleanup**: Periodic job to remove orphaned checkpoints

```python
# Example: Clean up after successful completion
def on_request_complete(
    request_id: str,
    checkpointer: Checkpointer,
    delete_on_success: bool = True,
) -> None:
    if delete_on_success:
        for ref in checkpointer.storage.list_checkpoints(request_id):
            checkpointer.storage.delete(ref)
```

## Configuration

### MainLoopConfig Extension

```python
@dataclass(frozen=True, slots=True)
class MainLoopConfig:
    # ... existing fields ...
    checkpointer: Checkpointer | None = None
```

### Environment-Based Configuration

```python
def checkpointer_from_env() -> Checkpointer | None:
    """
    Create Checkpointer from environment variables.

    CHECKPOINT_ENABLED: "true" to enable (default: disabled)
    CHECKPOINT_BUCKET: S3 bucket name (required if enabled)
    CHECKPOINT_PREFIX: S3 key prefix (default: "checkpoints")
    CHECKPOINT_INTERVAL_SECONDS: Trigger interval (default: 180)
    """
    if os.getenv("CHECKPOINT_ENABLED", "").lower() != "true":
        return None

    bucket = os.environ["CHECKPOINT_BUCKET"]
    prefix = os.getenv("CHECKPOINT_PREFIX", "checkpoints")
    interval = int(os.getenv("CHECKPOINT_INTERVAL_SECONDS", "180"))

    return Checkpointer(
        storage=S3CheckpointStorage(bucket=bucket, prefix=prefix),
        trigger=TimeTrigger(interval=timedelta(seconds=interval)),
    )
```

## Design by Contract

### Checkpointer Invariants

```python
from weakincentives.dbc import invariant, require, ensure

@invariant(
    lambda self: self._tool_call_count >= 0,
    "Tool call count must be non-negative",
)
@invariant(
    lambda self: self._last_checkpoint_at is None
    or self._last_checkpoint_at <= datetime.now(UTC),
    "Last checkpoint time must be in the past",
)
class Checkpointer:
    ...

    @require(lambda snapshot: snapshot is not None, "Snapshot required")
    @require(lambda run_context: run_context.request_id, "Request ID required")
    @ensure(
        lambda result, self: result is None or self._tool_call_count == 0,
        "Tool count reset after checkpoint",
    )
    def on_tool_call_complete(
        self,
        snapshot: CompositeSnapshot,
        run_context: RunContext,
    ) -> CheckpointRef | None:
        ...
```

### Storage Contract

```python
class CheckpointStorage(Protocol):
    @require(lambda request_id: request_id, "Request ID must be non-empty")
    @require(lambda snapshot: snapshot is not None, "Snapshot required")
    @ensure(
        lambda result: result.request_id and result.snapshot_id,
        "Ref must have valid identifiers",
    )
    def save(
        self,
        request_id: str,
        snapshot: CompositeSnapshot,
        *,
        attempt: int = 1,
    ) -> CheckpointRef:
        ...
```

## Limitations

1. **No automatic retry on save failure.** Storage errors are logged but not
   retried. Implement retry logic in the storage backend if needed.

2. **No incremental snapshots.** Each checkpoint captures full state. Large
   sessions may produce substantial checkpoint sizes.

3. **No cross-request deduplication.** Identical states in different requests
   are stored separately.

4. **Filesystem snapshots require git.** `HostFilesystem` checkpoints rely on
   git commits; non-git filesystems are not supported.

5. **No encryption at rest.** Encryption should be configured at the storage
   layer (e.g., S3 SSE).

## Related Specifications

- `SESSIONS.md` — Session lifecycle, snapshots, and restore semantics
- `FILESYSTEM.md` — Filesystem protocol and snapshotable backends
- `MAIN_LOOP.md` — Request handling and tool execution flow
- `RUN_CONTEXT.md` — Request correlation and execution metadata
- `TOOLS.md` — Tool runtime and transaction management

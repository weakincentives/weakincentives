# Debug Bundle Specification

## Purpose

`DebugBundle` consolidates multiple debugging artifacts produced during MainLoop
execution into a single, self-contained archive. This enables simple storage and
retrieval from object storage services (S3, GCS, Azure Blob) while preserving
the full debugging context for post-mortem analysis.

**Problem statement:** A single MainLoop task currently produces three separate
files: session snapshots (`{session_id}.jsonl`), logs (`{session_id}.log`), and
filesystem archives (`{session_id}.zip`). Managing these as independent files
complicates storage, retrieval, and correlation.

**Solution:** Package all artifacts into a single `.winkbundle` archive with a
manifest for metadata queries without full extraction.

## Guiding Principles

- **Complementary**: DebugBundle augments existing debug utilities; it does not
  replace direct file access for local development
- **Self-contained**: A single bundle contains everything needed for debugging
- **Streamable**: Supports incremental writes during long-running executions
- **Queryable**: Manifest enables metadata inspection without full download
- **Object-storage friendly**: Single-file design maps directly to PUT/GET
- **Deterministic**: Same inputs produce identical bundles (for content hashing)

## Bundle Format

### File Extension

`.winkbundle` - A TAR archive with optional gzip compression (`.winkbundle.gz`)

### Directory Structure

```
bundle.winkbundle
├── manifest.json           # Bundle metadata and content index
├── session/
│   └── {session_id}.jsonl  # Session snapshots (one or more)
├── logs/
│   └── {session_id}.log    # Structured logs (JSONL)
├── filesystem/
│   └── {archive_id}.zip    # Filesystem archive (optional)
└── attachments/            # Arbitrary additional files (optional)
    └── ...
```

### Manifest Schema

```json
{
  "version": "1",
  "bundle_id": "uuid",
  "created_at": "2024-01-15T10:30:00+00:00",
  "request_id": "uuid",
  "run_id": "uuid",
  "session_id": "uuid",
  "worker_id": "worker-host1-1234",
  "trace_id": "abc-123",
  "span_id": "xyz-789",
  "status": "completed|failed|partial",
  "error": "optional error message if failed",
  "contents": {
    "session": ["session/{session_id}.jsonl"],
    "logs": ["logs/{session_id}.log"],
    "filesystem": ["filesystem/{archive_id}.zip"],
    "attachments": []
  },
  "metrics": {
    "total_tokens": 12345,
    "input_tokens": 8000,
    "output_tokens": 4345,
    "tool_calls": 15,
    "duration_ms": 45000
  },
  "tags": {
    "app": "code-reviewer",
    "environment": "production"
  }
}
```

### Manifest Fields

| Field | Type | Description |
|-------|------|-------------|
| `version` | `string` | Schema version for forward compatibility |
| `bundle_id` | `UUID` | Unique identifier for this bundle |
| `created_at` | `datetime` | ISO 8601 timestamp when bundle was finalized |
| `request_id` | `UUID` | Correlates with `MainLoopRequest.request_id` |
| `run_id` | `UUID` | Correlates with `RunContext.run_id` |
| `session_id` | `UUID` | Primary session identifier |
| `worker_id` | `string` | Worker that processed the request |
| `trace_id` | `string?` | Distributed trace ID (optional) |
| `span_id` | `string?` | Span ID within trace (optional) |
| `status` | `enum` | `completed`, `failed`, or `partial` |
| `error` | `string?` | Error message if `status=failed` |
| `contents` | `object` | Paths to artifacts within the bundle |
| `metrics` | `object` | Aggregated execution metrics |
| `tags` | `object` | User-defined tags from session |

## Core API

### DebugBundle

```python
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import UUID

@dataclass(slots=True, frozen=True)
class BundleManifest:
    """Immutable manifest describing bundle contents."""

    version: str = "1"
    bundle_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    request_id: UUID | None = None
    run_id: UUID | None = None
    session_id: UUID | None = None
    worker_id: str = ""
    trace_id: str | None = None
    span_id: str | None = None
    status: Literal["completed", "failed", "partial"] = "completed"
    error: str | None = None
    contents: BundleContents = field(default_factory=BundleContents)
    metrics: BundleMetrics = field(default_factory=BundleMetrics)
    tags: Mapping[str, str] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class BundleContents:
    """Index of files within the bundle."""

    session: tuple[str, ...] = ()
    logs: tuple[str, ...] = ()
    filesystem: tuple[str, ...] = ()
    attachments: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class BundleMetrics:
    """Aggregated execution metrics."""

    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    duration_ms: int = 0
```

### DebugBundleWriter

```python
class DebugBundleWriter:
    """Incrementally builds a debug bundle during execution.

    Supports streaming writes for long-running tasks. Files are added
    to the bundle as they become available; the manifest is written
    last when the bundle is finalized.

    Example::

        writer = DebugBundleWriter(target=Path("./bundles"))

        # Add session snapshot
        writer.add_session(session)

        # Add logs (can be called multiple times)
        writer.add_logs(log_path)

        # Add filesystem archive
        writer.add_filesystem(fs)

        # Finalize with metadata
        bundle_path = writer.finalize(
            run_context=run_context,
            status="completed",
            metrics=metrics,
        )
    """

    def __init__(
        self,
        target: Path,
        *,
        bundle_id: UUID | None = None,
        compress: bool = True,
    ) -> None:
        """Initialize a bundle writer.

        Args:
            target: Directory where bundle will be written
            bundle_id: Optional bundle identifier (generated if not provided)
            compress: Whether to gzip compress the bundle (default: True)
        """
        ...

    def add_session(
        self,
        session: Session,
        *,
        include_logs: bool = True,
    ) -> None:
        """Add session snapshot to the bundle.

        Args:
            session: Session to snapshot
            include_logs: Whether to include LOG slices (default: True)
        """
        ...

    def add_session_tree(
        self,
        root_session: Session,
        *,
        include_logs: bool = True,
    ) -> None:
        """Add snapshots for session and all children.

        Args:
            root_session: Root of session tree to snapshot
            include_logs: Whether to include LOG slices (default: True)
        """
        ...

    def add_logs(self, log_path: Path) -> None:
        """Add a log file to the bundle.

        Args:
            log_path: Path to JSONL log file
        """
        ...

    def add_filesystem(
        self,
        fs: Filesystem,
        *,
        archive_id: UUID | None = None,
    ) -> None:
        """Add filesystem archive to the bundle.

        Args:
            fs: Filesystem instance to archive
            archive_id: Optional archive identifier
        """
        ...

    def add_attachment(
        self,
        path: Path,
        *,
        name: str | None = None,
    ) -> None:
        """Add an arbitrary file attachment.

        Args:
            path: Path to file to attach
            name: Name within bundle (defaults to filename)
        """
        ...

    def finalize(
        self,
        *,
        run_context: RunContext | None = None,
        status: Literal["completed", "failed", "partial"] = "completed",
        error: str | None = None,
        metrics: BundleMetrics | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> Path:
        """Finalize the bundle and write to disk.

        Args:
            run_context: Execution context for correlation IDs
            status: Final execution status
            error: Error message if status is "failed"
            metrics: Aggregated metrics
            tags: User-defined tags

        Returns:
            Path to the finalized bundle file
        """
        ...

    @property
    def bundle_id(self) -> UUID:
        """Return the bundle identifier."""
        ...
```

### DebugBundleReader

```python
class DebugBundleReader:
    """Read and extract debug bundles.

    Supports reading manifests without full extraction and selective
    extraction of individual components.

    Example::

        reader = DebugBundleReader(bundle_path)

        # Read manifest only
        manifest = reader.manifest
        print(f"Session: {manifest.session_id}")
        print(f"Status: {manifest.status}")

        # Extract specific components
        reader.extract_session(target_dir)
        reader.extract_logs(target_dir)

        # Full extraction
        reader.extract_all(target_dir)
    """

    def __init__(self, path: Path) -> None:
        """Open a bundle for reading.

        Args:
            path: Path to .winkbundle or .winkbundle.gz file
        """
        ...

    @property
    def manifest(self) -> BundleManifest:
        """Read and return the bundle manifest."""
        ...

    def extract_all(self, target: Path) -> None:
        """Extract all bundle contents to target directory."""
        ...

    def extract_session(self, target: Path) -> list[Path]:
        """Extract session snapshots only."""
        ...

    def extract_logs(self, target: Path) -> list[Path]:
        """Extract log files only."""
        ...

    def extract_filesystem(self, target: Path) -> list[Path]:
        """Extract filesystem archives only."""
        ...

    def extract_attachments(self, target: Path) -> list[Path]:
        """Extract attachment files only."""
        ...

    def iter_logs(self) -> Iterator[dict[str, Any]]:
        """Stream log entries without full extraction."""
        ...

    def iter_session_snapshots(self) -> Iterator[Snapshot]:
        """Stream session snapshots without full extraction."""
        ...
```

## Integration Patterns

### MainLoop Integration

The `MainLoop` base class provides optional bundle creation:

```python
class MainLoop(ABC, Generic[UserRequestT, OutputT]):
    def __init__(
        self,
        *,
        adapter: ProviderAdapter[OutputT],
        dispatcher: ControlDispatcher,
        config: MainLoopConfig | None = None,
        bundle_dir: Path | None = None,  # NEW
    ) -> None:
        self._bundle_dir = bundle_dir
        ...

    def _handle_message(self, message: Message[MainLoopRequest[UserRequestT]]) -> None:
        request_event = message.payload
        run_context = self._build_run_context(request_event, message.delivery_count)

        # Initialize bundle writer if configured
        bundle_writer = None
        if self._bundle_dir:
            bundle_writer = DebugBundleWriter(self._bundle_dir)

        try:
            with collect_all_logs(self._log_path) as collector:
                response, session = self._execute(request_event, run_context)

            # Build bundle on success
            if bundle_writer:
                self._write_bundle(
                    bundle_writer,
                    session=session,
                    log_path=collector.path,
                    run_context=run_context,
                    status="completed",
                )

            result = MainLoopResult(...)
            message.reply(result)

        except Exception as e:
            # Build bundle on failure
            if bundle_writer and session:
                self._write_bundle(
                    bundle_writer,
                    session=session,
                    log_path=collector.path,
                    run_context=run_context,
                    status="failed",
                    error=str(e),
                )
            raise
```

### Explicit Bundle Creation

For custom bundle creation outside MainLoop:

```python
from weakincentives.debug import (
    DebugBundleWriter,
    BundleMetrics,
    collect_all_logs,
)

# Create writer
writer = DebugBundleWriter(target=Path("./bundles"))

# Capture logs during evaluation
with collect_all_logs(f"/tmp/{session.session_id}.log") as collector:
    response = adapter.evaluate(prompt, session=session)

# Build bundle
writer.add_session(session)
writer.add_logs(collector.path)

if fs := prompt.resources.get(Filesystem, None):
    writer.add_filesystem(fs)

# Aggregate metrics from session events
metrics = BundleMetrics(
    total_tokens=response.usage.total_tokens if response.usage else 0,
    tool_calls=len(session[ToolInvoked].all()),
    duration_ms=int((response.completed_at - response.started_at).total_seconds() * 1000),
)

# Finalize
bundle_path = writer.finalize(
    run_context=run_context,
    status="completed",
    metrics=metrics,
    tags={"app": "my-agent"},
)
```

### Object Storage Upload

```python
import boto3
from weakincentives.debug import DebugBundleWriter

# Create and finalize bundle
writer = DebugBundleWriter(target=Path("/tmp/bundles"))
# ... add contents ...
bundle_path = writer.finalize(...)

# Upload to S3
s3 = boto3.client("s3")
key = f"debug-bundles/{bundle_path.name}"
s3.upload_file(str(bundle_path), "my-bucket", key)

# Retrieve later
s3.download_file("my-bucket", key, "/tmp/downloaded.winkbundle.gz")
reader = DebugBundleReader(Path("/tmp/downloaded.winkbundle.gz"))
print(reader.manifest.status)
```

### Manifest-Only Query

For quick status checks without downloading full bundles:

```python
import boto3
import tarfile
import gzip
import json

def read_manifest_from_s3(bucket: str, key: str) -> BundleManifest:
    """Read only the manifest from an S3-stored bundle."""
    s3 = boto3.client("s3")

    # Use range request to fetch first ~64KB (manifest is small)
    response = s3.get_object(
        Bucket=bucket,
        Key=key,
        Range="bytes=0-65535",
    )

    # Decompress and find manifest
    data = gzip.decompress(response["Body"].read())
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
        manifest_info = tar.getmember("manifest.json")
        manifest_file = tar.extractfile(manifest_info)
        return BundleManifest(**json.load(manifest_file))
```

## CLI Commands

### Create Bundle

```bash
# Create bundle from existing debug artifacts
wink bundle create \
    --session ./snapshots/abc123.jsonl \
    --logs ./snapshots/abc123.log \
    --filesystem ./snapshots/abc123.zip \
    --output ./bundles/
```

### Inspect Bundle

```bash
# Show manifest without extraction
wink bundle inspect ./bundles/abc123.winkbundle.gz

# Output:
# Bundle: abc123
# Status: completed
# Session: abc123-def456-...
# Created: 2024-01-15T10:30:00+00:00
# Contents:
#   - session/abc123.jsonl (1.2 MB)
#   - logs/abc123.log (450 KB)
#   - filesystem/abc123.zip (3.4 MB)
# Metrics:
#   - Total tokens: 12,345
#   - Tool calls: 15
#   - Duration: 45.0s
```

### Extract Bundle

```bash
# Extract all contents
wink bundle extract ./bundles/abc123.winkbundle.gz --output ./extracted/

# Extract specific component
wink bundle extract ./bundles/abc123.winkbundle.gz --component session --output ./extracted/
```

### Debug with Bundle

```bash
# Launch debug web UI directly from bundle
wink debug ./bundles/abc123.winkbundle.gz

# Extracts to temp directory and launches debug server
```

## Storage Backend Protocol

For custom storage backends beyond local filesystem:

```python
from typing import Protocol, BinaryIO

class BundleStorage(Protocol):
    """Protocol for bundle storage backends."""

    def put(self, bundle_id: UUID, data: BinaryIO) -> str:
        """Store bundle and return retrieval key."""
        ...

    def get(self, key: str) -> BinaryIO:
        """Retrieve bundle by key."""
        ...

    def get_manifest(self, key: str) -> BundleManifest:
        """Retrieve manifest only (may be optimized)."""
        ...

    def list(
        self,
        *,
        prefix: str | None = None,
        after: datetime | None = None,
        limit: int = 100,
    ) -> list[str]:
        """List bundle keys with optional filters."""
        ...

    def delete(self, key: str) -> None:
        """Delete bundle by key."""
        ...
```

### S3 Implementation

```python
class S3BundleStorage:
    """S3-backed bundle storage.

    Example::

        storage = S3BundleStorage(
            bucket="my-bucket",
            prefix="debug-bundles/",
        )

        # Upload bundle
        with open(bundle_path, "rb") as f:
            key = storage.put(bundle_id, f)

        # Quick manifest check
        manifest = storage.get_manifest(key)

        # Full download
        with storage.get(key) as f:
            reader = DebugBundleReader.from_stream(f)
    """

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        client: S3Client | None = None,
    ) -> None: ...
```

## Naming Conventions

### Bundle Filename

Default: `{bundle_id}.winkbundle.gz`

Configurable patterns:

```python
writer = DebugBundleWriter(
    target=Path("./bundles"),
    filename_pattern="{session_id}_{created_at:%Y%m%d_%H%M%S}.winkbundle.gz",
)
```

Supported variables:

| Variable | Description |
|----------|-------------|
| `{bundle_id}` | Bundle UUID |
| `{session_id}` | Primary session UUID |
| `{request_id}` | Request UUID |
| `{run_id}` | Run UUID |
| `{created_at}` | Creation timestamp (strftime format) |

### S3 Key Pattern

Recommended: `{prefix}/{year}/{month}/{day}/{bundle_id}.winkbundle.gz`

```python
storage = S3BundleStorage(
    bucket="my-bucket",
    prefix="debug-bundles",
    key_pattern="{prefix}/{created_at:%Y/%m/%d}/{bundle_id}.winkbundle.gz",
)
```

## Compression

### Default: gzip

Bundles use gzip compression by default for broad compatibility. The `.gz`
extension indicates compression.

### Compression Levels

```python
writer = DebugBundleWriter(
    target=Path("./bundles"),
    compress=True,
    compression_level=6,  # 1-9, default 6
)
```

| Level | Speed | Size | Use Case |
|-------|-------|------|----------|
| 1 | Fastest | Largest | Real-time streaming |
| 6 | Balanced | Medium | Default |
| 9 | Slowest | Smallest | Archival storage |

### Uncompressed Bundles

For debugging or when downstream systems handle compression:

```python
writer = DebugBundleWriter(
    target=Path("./bundles"),
    compress=False,  # Produces .winkbundle without .gz
)
```

## Size Considerations

### Expected Sizes

| Component | Typical Size | Notes |
|-----------|--------------|-------|
| Manifest | < 1 KB | Always small |
| Session snapshot | 10 KB - 1 MB | Depends on slice count |
| Logs | 100 KB - 10 MB | Scales with duration |
| Filesystem | 0 - 100 MB | Depends on workspace |

### Large Bundle Handling

For bundles exceeding 100 MB:

1. **Streaming writes**: Use `add_*` methods incrementally
2. **Multipart upload**: S3BundleStorage uses multipart for large bundles
3. **Selective extraction**: Use `extract_*` methods for specific components

### Bundle Size Limits

```python
writer = DebugBundleWriter(
    target=Path("./bundles"),
    max_size_bytes=100 * 1024 * 1024,  # 100 MB limit
)

# Raises BundleSizeExceeded if limit would be exceeded
writer.add_filesystem(large_fs)  # May raise
```

## Relationship to Existing Debug Utilities

### Complementary Design

DebugBundle wraps existing utilities rather than replacing them:

| Utility | Direct Use | Bundle Use |
|---------|------------|------------|
| `collect_all_logs` | Local development | Bundled for storage |
| `dump_session` | Local inspection | Bundled with correlation |
| `archive_filesystem` | Quick archive | Bundled with manifest |
| `wink debug` | Local JSONL files | Works with extracted bundles |

### Migration Path

Existing code continues to work unchanged:

```python
# Existing pattern - still works
with collect_all_logs(log_path):
    response = adapter.evaluate(prompt, session=session)
dump_session(session, debug_dir)
archive_filesystem(fs, debug_dir)

# New pattern - adds bundling
writer = DebugBundleWriter(target=bundle_dir)
with collect_all_logs(log_path):
    response = adapter.evaluate(prompt, session=session)
writer.add_session(session)
writer.add_logs(log_path)
writer.add_filesystem(fs)
bundle_path = writer.finalize(run_context=run_context)

# Both approaches produce the same underlying artifacts
```

## Thread Safety

### DebugBundleWriter

- Thread-safe for concurrent `add_*` calls
- Uses internal locking for TAR append operations
- `finalize()` must be called from single thread after all adds complete

### DebugBundleReader

- Fully thread-safe; uses read-only file access
- Multiple readers can access same bundle file

## Error Handling

### Write Errors

```python
try:
    writer.add_filesystem(fs)
    bundle_path = writer.finalize(...)
except BundleWriteError as e:
    logger.error(f"Failed to write bundle: {e}")
    # Partial bundle may exist; cleanup recommended
    writer.cleanup()  # Removes partial bundle file
```

### Read Errors

```python
try:
    reader = DebugBundleReader(path)
    manifest = reader.manifest
except BundleCorruptError as e:
    logger.error(f"Bundle is corrupt: {e}")
except BundleVersionError as e:
    logger.error(f"Unsupported bundle version: {e.version}")
```

### Missing Components

```python
reader = DebugBundleReader(path)
if not reader.manifest.contents.filesystem:
    logger.info("Bundle has no filesystem archive")
else:
    reader.extract_filesystem(target)
```

## Versioning

### Schema Version

The manifest `version` field enables forward compatibility:

- Version "1": Initial schema (this spec)
- Future versions add fields without removing existing ones
- Readers should ignore unknown fields

### Upgrade Path

```python
reader = DebugBundleReader(path)
if reader.manifest.version == "1":
    # Handle v1 bundle
    ...
elif reader.manifest.version == "2":
    # Handle v2 bundle with new fields
    ...
else:
    raise BundleVersionError(reader.manifest.version)
```

## Invariants

1. **Single manifest**: Every bundle has exactly one `manifest.json`
2. **Manifest first**: Manifest appears first in TAR for streaming reads
3. **Immutable bundles**: Once finalized, bundles are never modified
4. **Valid JSON Lines**: All `.jsonl` files contain valid JSON per line
5. **Deterministic output**: Same inputs produce byte-identical bundles
6. **Compression transparency**: Readers handle both compressed and uncompressed

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Create bundle | O(n) | Linear in total content size |
| Read manifest | O(1) | Fixed location in archive |
| Extract component | O(k) | Linear in component size |
| Stream logs | O(1) memory | Constant memory streaming |
| Full extraction | O(n) | Linear in total size |

## Implementation Notes

- Use Python's `tarfile` module with `w:gz` mode for compressed bundles
- Manifest is added first to enable range-request manifest reads
- Session and log files use UTF-8 encoding
- Filesystem archives are stored as-is (already compressed)
- Temporary files are used during creation to enable atomic finalization

## Related Specifications

- `specs/DEBUGGING.md` - Core debugging utilities
- `specs/WINK_DEBUG.md` - Debug web UI
- `specs/SLICES.md` - Session slice storage
- `specs/RUN_CONTEXT.md` - Execution correlation metadata
- `specs/MAIN_LOOP.md` - MainLoop execution lifecycle
- `specs/SESSIONS.md` - Session lifecycle and snapshots

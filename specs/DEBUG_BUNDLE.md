# Debug Bundle Specification

## Purpose

A debug bundle is a self-contained zip archive capturing everything needed to
understand, reproduce, and debug an AgentLoop execution. Bundles unify session
state, logs, filesystem snapshots, configuration, and metrics into a single
portable artifact.

**Implementation:**

- Bundle core: `src/weakincentives/debug/bundle.py`
- Environment capture: `src/weakincentives/debug/environment.py`
- Public API: `src/weakincentives/debug/__init__.py`
- EvalLoop integration: `src/weakincentives/evals/_loop.py`

## Principles

- **Zero-configuration capture**: Automatic bundling during AgentLoop execution
- **Self-contained**: All context for diagnosis lives in one file
- **Portable**: Readable with standard tools (unzip, text editor, jq)
- **Deterministic layout**: Predictable structure enables tooling
- **Streaming writes**: Incremental capture minimizes memory footprint
- **Atomic creation**: Bundle either fully created or not present
- **Graceful degradation**: Capture failures logged, never fail the request

## Bundle Format

All bundles contain a single `debug_bundle/` root directory. Timestamps use
ISO-8601 UTC throughout.

### Directory Layout

```
debug_bundle/
  manifest.json           # Bundle metadata and integrity
  README.txt              # Human-readable navigation guide
  request/
    input.json            # AgentLoop request
    output.json           # AgentLoop response
  session/
    before.jsonl          # Session state before execution
    after.jsonl           # Session state after execution
  logs/
    app.jsonl             # All structured log records during execution (incl. transcript events when available)
  transcript.jsonl        # Transcript entries extracted from logs (when available)
  environment/            # Reproducibility envelope
    system.json           # OS, kernel, arch, CPU, memory
    python.json           # Python version, executable, venv info
    packages.txt          # Installed packages (pip freeze)
    env_vars.json         # Environment variables (filtered/redacted)
    git.json              # Repo root, commit, branch, remotes
    git.diff              # Uncommitted changes (if any)
    command.txt           # argv, working dir, entrypoint
    container.json        # Container runtime info (if applicable)
  config.json             # AgentLoop and adapter configuration
  run_context.json        # Execution context (IDs, tracing)
  metrics.json            # Token usage, timing, budget state
  prompt_overrides.json   # Visibility overrides (if any)
  error.json              # Error details (if failed)
  eval.json               # Eval metadata (EvalLoop only)
  filesystem/             # Workspace snapshot (if captured)
    ...
  filesystem_history/     # Full filesystem change history (if captured)
    history.bundle        # Git bundle containing all snapshot commits
    manifest.json         # Snapshot metadata index
```

### Artifact Requirements

| Artifact | Required | Description |
|----------|----------|-------------|
| `manifest.json` | Yes | Version, bundle ID, file list, integrity checksums |
| `README.txt` | Yes | Generated navigation guide |
| `request/input.json` | Yes | Canonical AgentLoop input |
| `request/output.json` | Yes | Canonical AgentLoop output |
| `session/before.jsonl` | No | Pre-execution session snapshot |
| `session/after.jsonl` | Yes | Post-execution session snapshot |
| `logs/app.jsonl` | Yes | Structured log records with request correlation (and transcript events when emitted) |
| `transcript.jsonl` | No | Transcript entries extracted from logs (present when transcript events exist) |
| `environment/` | No | Reproducibility envelope (system, Python, packages, git) |
| `config.json` | Yes | AgentLoop, adapter, prompt configuration |
| `run_context.json` | Yes | Run/request/session IDs, tracing spans |
| `metrics.json` | Yes | Timing phases, token consumption, budget |
| `prompt_overrides.json` | No | Visibility overrides accumulated during execution |
| `error.json` | No | Exception type, phase, traceback, context |
| `eval.json` | No | Sample ID, experiment, score, judge output |
| `filesystem/` | No | Workspace files preserving directory structure |
| `filesystem_history/` | No | Full filesystem change history from transactional git repo |

### Manifest Schema

```json
{
  "format_version": "1.0.0",
  "bundle_id": "uuid",
  "created_at": "2024-01-15T10:30:00+00:00",
  "request": {
    "request_id": "uuid",
    "session_id": "uuid",
    "status": "success|error",
    "started_at": "...",
    "ended_at": "..."
  },
  "capture": {
    "mode": "full",
    "trigger": "config|env|request",
    "limits_applied": { "filesystem_truncated": false },
    "filesystem_history": true
  },
  "prompt": { "ns": "...", "key": "...", "adapter": "..." },
  "files": ["manifest.json", "..."],
  "integrity": {
    "algorithm": "sha256",
    "checksums": { "request/input.json": "abc123..." }
  },
  "build": { "version": "1.42.0", "commit": "abc123" }
}
```

## API

### BundleConfig

Configuration for bundle creation. See `src/weakincentives/debug/bundle.py`
for the full dataclass definition.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target` | `Path \| None` | `None` | Output directory |
| `max_file_size` | `int` | `10_000_000` | Skip files > 10MB |
| `max_total_size` | `int` | `52_428_800` | Max filesystem capture (50MB) |
| `compression` | `str` | `"deflate"` | Zip compression method |
| `retention` | `BundleRetentionPolicy \| None` | `None` | Local cleanup policy |
| `storage_handler` | `BundleStorageHandler \| None` | `None` | External storage callback |

The `enabled` property returns `True` when `target is not None`. Bundles always
capture full debug information: DEBUG logs, session before+after, and all
filesystem contents within size limits.

### BundleRetentionPolicy

Policy for cleaning up old debug bundles in the target directory.
See `src/weakincentives/debug/bundle.py` for the full dataclass definition.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_bundles` | `int \| None` | `None` | Keep at most N bundles (oldest deleted first) |
| `max_age_seconds` | `int \| None` | `None` | Delete bundles older than this |
| `max_total_bytes` | `int \| None` | `None` | Keep total size under limit (oldest deleted first) |

Retention is applied **after** each bundle is successfully created. If multiple
limits are configured, all are enforced (most restrictive wins). Bundle age is
determined from the `created_at` field in the manifest. Deletion uses TOCTOU
protection via inode/device verification to prevent race conditions.

```python
from weakincentives.debug import BundleConfig, BundleRetentionPolicy

config = BundleConfig(
    target=Path("./debug/"),
    retention=BundleRetentionPolicy(
        max_bundles=10,           # Keep last 10 bundles
        max_age_seconds=86400,    # Delete bundles older than 24 hours
    ),
)
```

### BundleStorageHandler

Runtime-checkable protocol for copying bundles to external storage after creation.
See `src/weakincentives/debug/bundle.py` for the protocol definition.

The handler receives `(bundle_path: Path, manifest: BundleManifest)` and is called
**after** retention policy is applied, so only bundles that survive cleanup are
passed to the handler. Errors are logged but do not propagate (non-blocking).

Example S3 handler:

```python
@dataclass
class S3StorageHandler:
    bucket: str
    prefix: str = "debug-bundles/"

    def store_bundle(self, bundle_path: Path, manifest: BundleManifest) -> None:
        key = f"{self.prefix}{manifest.bundle_id}.zip"
        s3_client.upload_file(str(bundle_path), self.bucket, key)

config = BundleConfig(
    target=Path("./debug/"),
    storage_handler=S3StorageHandler(bucket="my-bucket"),
)
```

### BundleWriter

Context manager for streaming bundle creation. See
`src/weakincentives/debug/bundle.py` for the full class definition.

Key methods: `write_session_before()`, `write_session_after()`,
`write_request_input()`, `write_request_output()`, `capture_logs()` (context
manager), `write_environment()`, `write_filesystem()`,
`write_filesystem_history()`, `write_config()`, `write_run_context()`,
`write_metrics()`, `write_error()`, `write_metadata()`,
`write_prompt_overrides()`, `set_prompt_info()`.

The `write_metadata(name, data)` method provides a generic mechanism for adding
domain-specific metadata (e.g., `eval.json`) without coupling the bundle layer.

```python
with BundleWriter(target="./debug/", bundle_id=run_id) as writer:
    writer.write_session_before(session)
    writer.write_request_input(request)
    with writer.capture_logs():
        response = adapter.evaluate(prompt, session=session)
    writer.write_session_after(session)
    writer.write_request_output(response)
    writer.write_environment()  # Capture reproducibility envelope
    writer.write_filesystem(fs)
    writer.write_filesystem_history(fs)  # Capture full git history
    writer.write_config(config)
    writer.write_run_context(run_context)
    writer.write_metrics(metrics)
    if error:
        writer.write_error(error)
# Bundle finalized on exit: README generated, checksums computed, zip created
```

### DebugBundle

Load and inspect existing bundles. See `src/weakincentives/debug/bundle.py`
for the full class definition.

Properties: `manifest`, `path`, `request_input`, `request_output`,
`session_before`, `session_after`, `logs`, `config`, `run_context`, `metrics`,
`prompt_overrides`, `error`, `eval`, `environment`.

Methods: `load()` (classmethod), `read_file()`, `list_files()`, `extract()`,
`verify_integrity()`.

```python
bundle = DebugBundle.load("./debug/bundle.zip")
print(bundle.manifest)
print(bundle.metrics)
print(bundle.session_after)
```

## AgentLoop Integration

### Configuration

`AgentLoopConfig` gains a `debug_bundle` field. When set, AgentLoop automatically
creates a bundle for each execution:

```python
loop = CodeReviewLoop(
    adapter=adapter,
    requests=requests,
    config=AgentLoopConfig(
        debug_bundle=BundleConfig(target=Path("./debug/")),
    ),
)
```

Per-request override via `AgentLoopRequest.debug_bundle`.

### Execution Flow

```
AgentLoop._handle_message()
  ├─ _build_run_context()
  ├─ BundleWriter(target, bundle_id)  # if configured
  │    ├─ write_session_before()
  │    ├─ write_request_input()
  │    ├─ capture_logs() → adapter.evaluate()
  │    ├─ write_session_after()
  │    ├─ write_environment()
  │    ├─ write_filesystem()
  │    ├─ write_filesystem_history()  # git bundle of snapshot repo
  │    ├─ write_config(), write_run_context(), write_metrics()
  │    ├─ write_error()  # if failed
  │    └─ __exit__() finalizes bundle:
  │         ├─ Generate README, compute checksums, create ZIP
  │         ├─ Atomic rename to target directory
  │         ├─ Apply retention policy (delete old bundles)
  │         └─ Call storage_handler.store_bundle() (if configured)
  └─ Return AgentLoopResult(bundle_path=writer.path)
```

### Post-Creation Lifecycle

After the bundle ZIP is finalized and atomically moved to the target directory:

1. **Retention enforcement**: If `BundleConfig.retention` is set, scan the target
   directory and delete bundles that exceed configured limits. Deletion order is
   oldest-first based on manifest `created_at`.

1. **Storage handler invocation**: If `BundleConfig.storage_handler` is set, call
   `store_bundle(bundle_path, manifest)`. Errors are logged at WARNING level but
   do not fail the request or affect the local bundle.

Both steps are non-blocking: failures are logged but do not propagate exceptions.

### Result Extension

`AgentLoopResult` gains `bundle_path: Path | None` for accessing the created
bundle.

## EvalLoop Integration

EvalLoop wraps AgentLoop for evaluation datasets. When `EvalLoopConfig.debug_bundle`
is set to a `BundleConfig` with a `target` directory, EvalLoop uses
`AgentLoop.execute_with_bundle()` to reuse the standard bundle creation logic with
eval-specific metadata:

1. EvalLoop creates bundle target at `{target}/{request_id}/`
1. AgentLoop creates bundle capturing session, logs, request/response
1. EvalLoop injects `eval.json` with score, experiment, latency
1. Bundle path returned in `EvalResult.bundle_path`

### eval.json Schema

```json
{
  "sample_id": "sample-123",
  "experiment_name": "baseline",
  "score": {
    "value": 0.85,
    "passed": true,
    "reason": "Found expected answer"
  },
  "latency_ms": 1500,
  "error": null
}
```

### EvalLoopConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `lease_extender` | `LeaseExtenderConfig \| None` | `None` | Automatic message visibility extension |
| `debug_bundle` | `BundleConfig \| None` | `None` | Debug bundle creation per sample |

See `src/weakincentives/evals/_loop.py` for the full `EvalLoopConfig` and `EvalLoop`
implementation.

### Example

```python
eval_loop = EvalLoop(
    loop=agent_loop,
    evaluator=exact_match,
    requests=requests_mailbox,
    config=EvalLoopConfig(
        debug_bundle=BundleConfig(target=Path("./eval_bundles/")),
    ),
)

# After evaluation:
# result.bundle_path == Path("./eval_bundles/{request_id}/{bundle_id}_{timestamp}.zip")
```

## CLI

### Commands

```bash
wink debug <bundle.zip>            # Open bundle in web UI
wink debug <directory>             # Open most recent bundle in directory
wink query <bundle.zip> "SELECT * FROM manifest"  # SQL query against bundle
wink query <bundle.zip> --schema   # Show database schema
```

Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `127.0.0.1` | Host interface to bind |
| `--port` | `8000` | Port to bind |
| `--no-open-browser` | | Disable automatic browser open |

### Web UI Panels

| Panel | Source | Description |
|-------|--------|-------------|
| Overview | `manifest.json` | Bundle summary, navigation |
| Request | `request/*.json` | Input/output inspector |
| Slices | `session/*.jsonl` | Session state browser by slice type |
| Logs | `logs/app.jsonl` | Searchable, filterable log viewer |
| Files | `filesystem/` | File tree with content |
| File History | `filesystem_history/` | Timeline of filesystem changes across tool calls |
| Config | `config.json` | Configuration inspector |
| Metrics | `metrics.json` | Performance dashboard |
| Error | `error.json` | Error details (if present) |

### API Routes

| Route | Description |
|-------|-------------|
| `/api/meta` | Bundle metadata summary |
| `/api/manifest` | Full bundle manifest |
| `/api/request/input` | Request input data |
| `/api/request/output` | Request output data |
| `/api/slices/{type}` | Slice items (paginated) |
| `/api/logs` | Log entries (paginated, filterable by level) |
| `/api/files` | Filesystem listing |
| `/api/files/{path}` | File content |
| `/api/history/snapshots` | List all filesystem snapshots (timeline) |
| `/api/history/snapshot/{ref}` | Files at a specific snapshot |
| `/api/history/diff/{from_ref}/{to_ref}` | Diff between two snapshots |
| `/api/history/file/{path}` | Full history of a single file across snapshots |
| `/api/config` | Configuration |
| `/api/metrics` | Metrics |
| `/api/error` | Error details |
| `/api/bundles` | List bundles in directory |
| `/api/switch` | Switch to different bundle |
| `/api/reload` | Reload current bundle |

## Bundle Naming

Standard: `{bundle_id}_{timestamp}.zip`

EvalLoop: `{request_id}/{bundle_id}_{timestamp}.zip`

## Filesystem History

### Background

Tool calls in WINK are transactional. Before each tool execution, the runtime
takes a snapshot of the workspace filesystem by committing all changes to an
external git repository (see `FILESYSTEM.md` and `TOOLS.md`). On tool failure,
the filesystem is restored to the pre-tool snapshot via `git reset --hard`.

This means the external git repository accumulates a commit-per-snapshot history
that records **every filesystem mutation** the agent made during execution.
Each commit is tagged with the tool name and tool-call ID, providing a complete
timeline of how the workspace evolved.

### Capturing History in Bundles

When `BundleWriter.write_filesystem_history(fs)` is called with a
`HostFilesystem` that has an initialized git directory, the writer:

1. **Creates a git bundle** from the external snapshot repository using
   `git bundle create`, capturing the full commit graph.
1. **Generates a snapshot manifest** (`manifest.json`) listing each snapshot
   with metadata: commit ref, timestamp, tag (tool name), and whether the
   snapshot was rolled back or persisted.
1. **Writes both artifacts** to `filesystem_history/` in the debug bundle.

The git bundle format is a standard portable git archive. It can be cloned
locally for full `git log`, `git diff`, and `git show` access without needing
the original repository.

```python
# BundleWriter API
def write_filesystem_history(self, fs: Filesystem) -> None:
    """Write full filesystem change history from transactional git repo.

    Captures the git repository used for tool-call snapshots as a
    portable git bundle. Only applies to HostFilesystem with an
    initialized git directory; no-op for in-memory filesystems.

    Args:
        fs: The filesystem whose snapshot history to capture.
    """
```

### filesystem_history/manifest.json Schema

```json
{
  "format_version": "1.0.0",
  "snapshot_count": 12,
  "snapshots": [
    {
      "commit_ref": "abc123...",
      "created_at": "2024-01-15T10:30:00+00:00",
      "tag": "pre:write_file:call_001",
      "parent_ref": "def456...",
      "message": "pre:write_file:call_001",
      "files_changed": 3,
      "insertions": 45,
      "deletions": 12
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `commit_ref` | `str` | Git commit hash |
| `created_at` | `str` | ISO-8601 UTC timestamp |
| `tag` | `str \| null` | Snapshot tag (typically `"pre:{tool}:{id}"`) |
| `parent_ref` | `str \| null` | Parent commit hash (null for initial) |
| `message` | `str` | Git commit message |
| `files_changed` | `int` | Number of files changed in this snapshot |
| `insertions` | `int` | Lines added |
| `deletions` | `int` | Lines removed |

### Size Control

The git bundle uses git's content-addressed storage, so identical files across
snapshots share storage automatically (copy-on-write). Typical bundle sizes are
much smaller than the sum of all snapshot states.

`BundleConfig` gains a `max_history_size` field (default 100MB) to cap the git
bundle size. If the history exceeds this limit, older commits are pruned using
`git bundle create --since` to keep the most recent history.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_history_size` | `int` | `104_857_600` | Max git bundle size (100MB) |
| `include_history` | `bool` | `True` | Whether to capture filesystem history |

### CLI: wink debug File History

The `wink debug` web UI adds a **File History** panel that provides:

#### Snapshot Timeline

A chronological list of all snapshots showing:

- **Timestamp** and **tag** (tool name and call ID)
- **Diff summary**: files changed, insertions, deletions
- **Rollback indicator**: whether this snapshot was later rolled back (its
  changes were reverted by a subsequent restore)

This lets developers see exactly which tool calls modified the filesystem and
in what order, including tool calls whose changes were undone.

#### File-Level History

Selecting a file shows its full change history across snapshots:

- Each version with the snapshot that introduced the change
- Inline diff between consecutive versions
- Ability to view the file content at any snapshot

#### Snapshot Diff View

Selecting two snapshots shows the full diff between them:

- Added, modified, and deleted files
- Unified diff for each changed file

### History API Routes

| Route | Method | Query Params | Description |
|-------|--------|-------------|-------------|
| `/api/history/snapshots` | GET | `offset`, `limit` | Paginated snapshot timeline |
| `/api/history/snapshot/{ref}` | GET | | File tree at a specific snapshot |
| `/api/history/diff/{from_ref}/{to_ref}` | GET | `path` (optional) | Diff between two snapshots, optionally scoped to one file |
| `/api/history/file/{path}` | GET | `offset`, `limit` | Change history for a single file across all snapshots |

#### Example Responses

**GET /api/history/snapshots**

```json
{
  "total": 12,
  "offset": 0,
  "limit": 50,
  "items": [
    {
      "commit_ref": "abc123",
      "created_at": "2024-01-15T10:30:00+00:00",
      "tag": "pre:write_file:call_001",
      "files_changed": 1,
      "insertions": 20,
      "deletions": 0,
      "rolled_back": false
    },
    {
      "commit_ref": "def456",
      "created_at": "2024-01-15T10:30:05+00:00",
      "tag": "pre:patch_file:call_002",
      "files_changed": 1,
      "insertions": 5,
      "deletions": 3,
      "rolled_back": true
    }
  ]
}
```

**GET /api/history/file/src/main.py**

```json
{
  "path": "src/main.py",
  "versions": [
    {
      "commit_ref": "abc123",
      "created_at": "2024-01-15T10:30:00+00:00",
      "tag": "pre:write_file:call_001",
      "action": "added",
      "diff": "@@ -0,0 +1,20 @@\n+def main():\n+    ..."
    },
    {
      "commit_ref": "ghi789",
      "created_at": "2024-01-15T10:31:00+00:00",
      "tag": "pre:refactor:call_003",
      "action": "modified",
      "diff": "@@ -5,3 +5,5 @@\n-    old_line\n+    new_line\n+    added_line"
    }
  ]
}
```

### Extracting History from a Bundle

The `DebugBundle` class gains a `filesystem_history` property and helper
methods for working with the captured git history:

```python
bundle = DebugBundle.load("./debug/bundle.zip")

# Check if history is available
if bundle.filesystem_history is not None:
    history = bundle.filesystem_history

    # List all snapshots
    for snapshot in history["snapshots"]:
        print(f"{snapshot['created_at']} {snapshot['tag']} "
              f"({snapshot['files_changed']} files)")

# Extract the git bundle for local inspection
bundle.extract_filesystem_history(target=Path("./history_repo/"))
# Now use standard git commands:
#   cd ./history_repo && git log --oneline
#   git diff <ref1> <ref2>
#   git show <ref>:path/to/file
```

### DebugBundle Extensions

| Property/Method | Returns | Description |
|----------------|---------|-------------|
| `filesystem_history` | `dict \| None` | Parsed `filesystem_history/manifest.json` |
| `has_filesystem_history` | `bool` | Whether history was captured |
| `extract_filesystem_history(target)` | `Path` | Clone git bundle to local repo for inspection |

### Manual Inspection

Since the captured history is a standard git bundle, advanced users can extract
and inspect it directly:

```bash
# Extract the bundle
unzip debug_bundle.zip
cd debug_bundle/filesystem_history/

# Clone from the git bundle into a local repo
git clone history.bundle ./workspace_history

# Browse the full timeline
cd workspace_history
git log --oneline --stat

# See what a specific tool call changed
git show <commit-ref>

# Diff between two points in time
git diff <earlier-ref> <later-ref>

# View a file at a specific point
git show <ref>:path/to/file.py
```

## Public API

```python
from weakincentives.debug import (
    # Bundle creation and inspection
    BundleWriter,
    BundleConfig,
    DebugBundle,
    BundleManifest,
    BundleError,
    BundleValidationError,
    # Retention and storage
    BundleRetentionPolicy,
    BundleStorageHandler,
    # Environment capture
    capture_environment,
    EnvironmentCapture,
    SystemInfo,
    PythonInfo,
    GitInfo,
    CommandInfo,
    ContainerInfo,
)
```

## Invariants

1. **Atomic creation**: Bundle either fully created or absent
1. **Immutable**: No modification after creation
1. **Single root**: Zip contains only `debug_bundle/` directory
1. **Valid manifest**: Every bundle has well-formed `manifest.json`
1. **Integrity verification**: Checksums enable artifact verification
1. **Deterministic ordering**: JSONL files ordered by timestamp
1. **Retention before storage**: Retention policy runs before storage handler
1. **Non-blocking lifecycle**: Retention and storage failures never fail the request

## Security Considerations

Bundles may contain sensitive data: API keys in logs, proprietary code in
filesystem snapshots, PII in inputs. Recommendations:

- Review before sharing externally
- Store with appropriate access controls
- Disable filesystem capture via `WEAKINCENTIVES_DEBUG_BUNDLE_NO_FS=1` for sensitive workspaces

## Limitations

- **Filesystem size**: Large workspaces produce large bundles
- **Filesystem history**: History requires HostFilesystem with git; in-memory
  filesystems have no commit history to capture
- **History size**: Long sessions with many tool calls accumulate large git
  histories; use `max_history_size` to cap
- **Memory**: Log capture buffers before writing
- **Concurrency**: One bundle per AgentLoop execution
- **No automatic redaction**: Manual review required for sensitive data

## Related Specifications

- `specs/AGENT_LOOP.md` - Execution flow
- `specs/SESSIONS.md` - Session snapshots
- `specs/LOGGING.md` - Log record format
- `specs/RUN_CONTEXT.md` - Execution metadata
- `specs/FILESYSTEM.md` - Workspace abstraction, snapshot history export
- `specs/TOOLS.md` - Tool transactions, snapshot lifecycle
- `specs/EVALS.md` - Evaluation framework

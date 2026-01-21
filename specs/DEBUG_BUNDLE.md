# Debug Bundle Specification

## Purpose

A debug bundle is a self-contained zip archive capturing everything needed to
understand, reproduce, and debug a MainLoop execution. Bundles unify session
state, logs, filesystem snapshots, configuration, and metrics into a single
portable artifact.

Core implementation at `src/weakincentives/debug/bundle.py`.

## Principles

- **Zero-configuration capture**: Automatic bundling during MainLoop execution
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
    input.json            # MainLoop request
    output.json           # MainLoop response
  session/
    before.jsonl          # Session state before execution
    after.jsonl           # Session state after execution
  logs/
    app.jsonl             # All log records during execution
  config.json             # MainLoop and adapter configuration
  run_context.json        # Execution context (IDs, tracing)
  metrics.json            # Token usage, timing, budget state
  prompt_overrides.json   # Visibility overrides (if any)
  error.json              # Error details (if failed)
  eval.json               # Eval metadata (EvalLoop only)
  filesystem/             # Workspace snapshot (if captured)
    ...
```

### Artifact Requirements

| Artifact | Required | Description |
|----------|----------|-------------|
| `manifest.json` | Yes | Version, bundle ID, file list, integrity checksums |
| `README.txt` | Yes | Generated navigation guide |
| `request/input.json` | Yes | Canonical MainLoop input |
| `request/output.json` | Yes | Canonical MainLoop output |
| `session/before.jsonl` | No | Pre-execution session snapshot |
| `session/after.jsonl` | Yes | Post-execution session snapshot |
| `logs/app.jsonl` | Yes | Structured log records with request correlation |
| `config.json` | Yes | MainLoop, adapter, prompt configuration |
| `run_context.json` | Yes | Run/request/session IDs, tracing spans |
| `metrics.json` | Yes | Timing phases, token consumption, budget |
| `prompt_overrides.json` | No | Visibility overrides accumulated during execution |
| `error.json` | No | Exception type, phase, traceback, context |
| `eval.json` | No | Sample ID, experiment, score, judge output |
| `filesystem/` | No | Workspace files preserving directory structure |

### Manifest Schema

```json
{
  "format_version": "1.1.0",
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
    "limits_applied": {
      "filesystem_truncated": false,
      "logs_truncated": false,
      "tool_outputs_truncated": false
    },
    "redaction": {
      "enabled": false,
      "ruleset": "",
      "patterns": []
    }
  },
  "summary": {
    "duration_ms": 15342,
    "status": "success",
    "error_count": 0,
    "prompt_count": 0,
    "tool_call_count": 0,
    "provider_call_count": 0,
    "token_usage": { "input": 0, "output": 0, "cached": 0 }
  },
  "artifacts": {
    "request_input": {
      "path": "request/input.json",
      "kind": "json",
      "content_type": "application/json",
      "size_bytes": 842,
      "sha256": "..."
    },
    "logs": {
      "path": "logs/app.jsonl",
      "kind": "jsonl",
      "content_type": "application/x-ndjson",
      "size_bytes": 932182,
      "sha256": "...",
      "time_range": { "start": "...", "end": "..." },
      "counts": { "records": 1203, "levels": { "INFO": 900, "ERROR": 3 } }
    },
    "session_after": {
      "path": "session/after.jsonl",
      "kind": "jsonl",
      "content_type": "application/x-ndjson",
      "schema": { "type": "Snapshot", "version": "1.0.0" }
    },
    "filesystem": {
      "path": "filesystem/",
      "kind": "directory",
      "capture": {
        "max_file_size": 10000000,
        "max_total_size": 52428800,
        "excluded_patterns": []
      },
      "counts": {
        "files_captured": 312,
        "files_skipped": 19,
        "total_bytes_captured": 4212345
      }
    }
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

### Artifacts Map

The `artifacts` field provides a catalog of bundle artifacts keyed by logical ID.
Each artifact has:

| Field | Description |
|-------|-------------|
| `path` | Relative path within bundle |
| `kind` | Type: `json`, `jsonl`, `text`, `directory` |
| `content_type` | MIME type (e.g., `application/json`) |
| `size_bytes` | File size in bytes |
| `sha256` | SHA-256 checksum (empty for directories) |
| `index` | Optional pointer to index file |
| `time_range` | Optional start/end timestamps for temporal artifacts |
| `counts` | Optional summary counts (records for logs, files for filesystem) |
| `schema` | Optional schema info for structured data |
| `capture` | Optional capture parameters for filesystem |

Standard logical IDs:

| ID | Path | Description |
|----|------|-------------|
| `request_input` | `request/input.json` | MainLoop request |
| `request_output` | `request/output.json` | MainLoop response |
| `session_before` | `session/before.jsonl` | Pre-execution session |
| `session_after` | `session/after.jsonl` | Post-execution session |
| `logs` | `logs/app.jsonl` | Log records |
| `config` | `config.json` | Configuration |
| `run_context` | `run_context.json` | Execution context |
| `metrics` | `metrics.json` | Timing and usage |
| `filesystem` | `filesystem/` | Workspace snapshot |
| `environment` | `environment/` | Reproducibility envelope |
| `error` | `error.json` | Error details (if failed) |
| `eval` | `eval.json` | Eval metadata (EvalLoop only) |

## API

### BundleConfig

Configuration for bundle creation:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target` | `Path \| None` | `None` | Output directory |
| `max_file_size` | `int` | `10_000_000` | Skip files > 10MB |
| `max_total_size` | `int` | `52_428_800` | Max filesystem capture (50MB) |
| `compression` | `str` | `"deflate"` | Zip compression method |

Bundles always capture full debug information: DEBUG logs, session before+after,
and all filesystem contents within size limits.

### BundleWriter

Context manager for streaming bundle creation:

```python
with BundleWriter(target="./debug/", bundle_id=run_id) as writer:
    writer.write_session_before(session)
    writer.write_request_input(request)
    with writer.capture_logs():
        response = adapter.evaluate(prompt, session=session)
    writer.write_session_after(session)
    writer.write_request_output(response)
    writer.write_filesystem(fs)
    writer.write_config(config)
    writer.write_run_context(run_context)
    writer.write_metrics(metrics)
    if error:
        writer.write_error(error)
# Bundle finalized on exit: README generated, checksums computed, zip created
```

### DebugBundle

Load and inspect existing bundles:

```python
bundle = DebugBundle.load("./debug/bundle.zip")
print(bundle.manifest)
print(bundle.metrics)
print(bundle.session_after)
```

## MainLoop Integration

### Configuration

`MainLoopConfig` gains a `debug_bundle` field. When set, MainLoop automatically
creates a bundle for each execution:

```python
loop = CodeReviewLoop(
    adapter=adapter,
    requests=requests,
    config=MainLoopConfig(
        debug_bundle=BundleConfig(target=Path("./debug/")),
    ),
)
```

Per-request override via `MainLoopRequest.debug_bundle`.

### Execution Flow

```
MainLoop._handle_message()
  ├─ _build_run_context()
  ├─ BundleWriter(target, bundle_id)  # if configured
  │    ├─ write_session_before()
  │    ├─ write_request_input()
  │    ├─ capture_logs() → adapter.evaluate()
  │    ├─ write_session_after()
  │    ├─ write_filesystem()
  │    ├─ write_config(), write_run_context(), write_metrics()
  │    └─ write_error()  # if failed
  └─ Return MainLoopResult(bundle_path=writer.path)
```

### Result Extension

`MainLoopResult` gains `bundle_path: Path | None` for accessing the created
bundle.

## EvalLoop Integration

EvalLoop wraps MainLoop for evaluation datasets. When `EvalLoopConfig.debug_bundle_dir`
is set:

1. MainLoop creates bundle at `{debug_bundle_dir}/{request_id}/{sample_id}.zip`
1. EvalLoop appends `eval.json` with score, experiment, judge output
1. Bundle paths returned in `EvalResult.bundle_paths`

For LLM-as-judge evaluators, both sample and judge execution bundles are
captured:

```python
EvalResult(
    bundle_paths=(
        Path("debug/req/sample-123.zip"),        # Sample execution
        Path("debug/req/sample-123-judge.zip"),  # Judge execution
    ),
)
```

## CLI

### Commands

```bash
wink debug <bundle.zip>            # Open bundle in web UI
wink debug <directory>             # Open most recent bundle in directory
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
| Sessions | `session/*.jsonl` | State browser with diff |
| Logs | `logs/app.jsonl` | Searchable, filterable viewer |
| Files | `filesystem/` | File tree with content |
| Config | `config.json` | Configuration inspector |
| Metrics | `metrics.json` | Performance dashboard |
| Error | `error.json` | Error details (if present) |
| Eval | `eval.json` | Score and judge output (if present) |

### API Routes

| Route | Description |
|-------|-------------|
| `/api/manifest` | Bundle manifest |
| `/api/request/{input\|output}` | Request data |
| `/api/session/{before\|after}` | Session snapshots |
| `/api/session/slices/{type}` | Slice items (paginated) |
| `/api/logs` | Log entries (paginated, filterable) |
| `/api/files`, `/api/files/{path}` | Filesystem listing and content |
| `/api/config`, `/api/metrics` | Configuration and metrics |
| `/api/error`, `/api/eval` | Error and eval details |
| `/api/bundles` | List bundles in directory |
| `/api/switch`, `/api/reload` | Bundle navigation |

## Bundle Naming

Standard: `{request_id}_{timestamp}.zip`

EvalLoop: `{eval_request_id}/{sample_id}_{timestamp}.zip`

## Public API

```python
from weakincentives.debug import (
    BundleWriter,
    BundleConfig,
    DebugBundle,
    BundleManifest,
    BundleError,
    BundleValidationError,
)
```

## Invariants

1. **Atomic creation**: Bundle either fully created or absent
1. **Immutable**: No modification after creation
1. **Single root**: Zip contains only `debug_bundle/` directory
1. **Valid manifest**: Every bundle has well-formed `manifest.json`
1. **Integrity verification**: Checksums enable artifact verification
1. **Deterministic ordering**: JSONL files ordered by timestamp

## Security Considerations

Bundles may contain sensitive data: API keys in logs, proprietary code in
filesystem snapshots, PII in inputs. Recommendations:

- Use `minimal` mode for production incidents
- Review before sharing externally
- Store with appropriate access controls
- Use `exclude_patterns` for sensitive paths

## Limitations

- **Filesystem size**: Large workspaces produce large bundles
- **Memory**: Log capture buffers before writing
- **Concurrency**: One bundle per MainLoop execution
- **No automatic redaction**: Manual review required for sensitive data

## Related Specifications

- `specs/MAIN_LOOP.md` - Execution flow
- `specs/SESSIONS.md` - Session snapshots
- `specs/LOGGING.md` - Log record format
- `specs/RUN_CONTEXT.md` - Execution metadata
- `specs/FILESYSTEM.md` - Workspace abstraction
- `specs/EVALS.md` - Evaluation framework

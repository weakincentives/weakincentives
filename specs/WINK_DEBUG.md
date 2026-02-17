# Wink Debug Bundle Explorer

Specification for the debug bundle viewer tool (`wink debug`).

## Overview

The wink debug tool provides a local browser-based UI for exploring debug bundles
generated during agent execution. Debug bundles capture comprehensive execution
state including session data, logs, task input/output, and filesystem snapshots.

**Primary use case**: Post-mortem analysis of agent runs, especially for:

- Understanding what happened during long-running sessions
- Investigating failures and unexpected behavior
- Reviewing tool call sequences and state evolution

## Current Implementation

### Architecture

```
wink debug [bundle-path]
    │
    ├── HTTP Server (localhost:8000)
    │   ├── /api/meta          - Bundle metadata
    │   ├── /api/slices/:type  - Session slice data
    │   ├── /api/logs          - Log entries (paginated)
    │   ├── /api/logs/facets   - Logger/event/level facets for filters
    │   ├── /api/transcript    - Transcript entries (paginated)
    │   ├── /api/transcript/facets - Source/type facets for filters
    │   ├── /api/request/*     - Task input/output
    │   ├── /api/files         - Filesystem listing
    │   └── /api/files/:path   - File content
    │
    └── Static UI
        ├── index.html
        ├── style.css
        └── app.js
```

### Views

| View | Purpose | Sidebar | Content |
|------|---------|---------|---------|
| **Sessions** | Inspect session state slices | Slice list with filter | Tree viewer with search, depth control |
| **Transcript** | Browse transcript entries | Source/type filters, search | Chat-like transcript stream with details |
| **Logs** | Browse execution logs | Level filters, search | Scrollable log entries |
| **Task** | View request input/output | Input/Output toggle | Tree viewer with depth control |
| **Filesystem** | Browse workspace snapshot | File list with filter | File content viewer |

### Navigation

- **Tabs**: Switch views via numbered tabs (1-5 keyboard shortcuts)
- **Bundle selector**: Dropdown to switch between bundles in directory
- **Keyboard shortcuts**: J/K navigation, / for search, R to reload

### Bundle Contents

A debug bundle (`.zip`) contains:

```
{bundle_id}_{timestamp}.zip
└── debug_bundle/
    ├── manifest.json          # Bundle metadata and integrity
    ├── request/
    │   ├── input.json         # Task input
    │   └── output.json        # Task output
    ├── session/
    │   └── after.jsonl        # Final session state
    ├── logs/
    │   └── app.jsonl          # Structured log entries (includes transcript events)
    ├── transcript.jsonl       # Transcript entries (extracted from logs)
    ├── config.json            # Runtime configuration
    ├── run_context.json       # Execution context
    └── filesystem/            # Workspace snapshot (optional)
        └── ...
```

## Analysis Jobs

An **analysis job** dispatches an agent from the `wink debug` web UI to
produce a structured report for the currently loaded bundle. The user clicks
"Analyze", the server runs an `AgentLoop` in a background thread, and the UI
polls until the report is ready.

```
UI ──POST /api/analysis──→ Server ──→ AgentLoop.execute()
                                          │
UI ──GET /api/analysis/status──→ Server   │  (background thread)
          ↑ poll                          │
          └───────────────────────────────┘
                                          │
UI ──GET /api/analysis/report──→ Server ◄─┘  report persisted
```

### Lifecycle

1. **Dispatch** — `POST /api/analysis`. Server rejects if a job is already
   running (409). Otherwise starts the agent in a background thread and returns
   `202 Accepted` with a `job_id`.

2. **Execution** — Server extracts bundle contents (manifest, request, session
   state, logs, transcript, metrics, error, config) into a context payload.
   Logs are sampled if over a size threshold; filesystem contents are summarized
   as a file listing rather than included verbatim. An `AgentLoop` executes
   with a purpose-built analysis prompt and read-only bundle tools.

3. **Completion** — Report persisted alongside the bundle as
   `{bundle_id}_analysis.json`. Job transitions to `completed`. On agent
   failure, job transitions to `failed` with error details.

4. **Caching** — Reopening the same bundle serves the cached report. Users can
   explicitly re-run to get a fresh report.

### Job States

| State | Description |
|-------|-------------|
| `running` | Agent is executing |
| `completed` | Report ready and persisted |
| `failed` | Agent execution failed; error details available |

### API

| Route | Method | Description |
|-------|--------|-------------|
| `/api/analysis` | `POST` | Dispatch analysis job for current bundle |
| `/api/analysis/status` | `GET` | Job state and elapsed time |
| `/api/analysis/report` | `GET` | Completed report (404 if not ready) |
| `/api/analysis/cancel` | `POST` | Cancel a running job |

**POST /api/analysis** — Request body (all fields optional):

```json
{
  "adapter": "claude_agent_sdk",
  "budget": { "max_tokens": 50000 },
  "focus": ["errors", "timeline", "budget"]
}
```

Response: `202 Accepted` with `{ "job_id": "uuid", "status": "running" }`

**GET /api/analysis/status** — Response:

```json
{
  "job_id": "uuid",
  "status": "running",
  "started_at": "2024-01-15T10:35:00+00:00",
  "elapsed_seconds": 12
}
```

### Analysis Report Schema

```json
{
  "version": "1.0.0",
  "bundle_id": "uuid",
  "analyzed_at": "2024-01-15T10:35:00+00:00",
  "summary": {
    "outcome": "success | partial_success | failure",
    "one_liner": "Short human-readable assessment",
    "duration_assessment": "Timing analysis",
    "key_findings": ["..."]
  },
  "errors": [
    {
      "phase": "tool_execution",
      "event": "filesystem.read_failed",
      "count": 8,
      "first_occurrence": "...",
      "last_occurrence": "...",
      "description": "What happened",
      "impact": "Effect on the run",
      "recommendation": "Suggested fix"
    }
  ],
  "timeline": [
    {
      "timestamp": "...",
      "phase": "planning | tool_execution | budget",
      "description": "What happened",
      "significance": "normal | warning | critical"
    }
  ],
  "tool_analysis": {
    "total_calls": 45,
    "unique_tools": ["..."],
    "failure_rate": 0.18,
    "patterns": ["..."]
  },
  "budget": {
    "tokens_used": 48000,
    "tokens_limit": 50000,
    "utilization": 0.96,
    "assessment": "Human-readable budget analysis"
  },
  "recommendations": [
    {
      "priority": "high | medium | low",
      "category": "budget | tools | prompt | config",
      "title": "Short title",
      "detail": "Actionable detail"
    }
  ]
}
```

### Analysis Prompt

The analysis prompt is a `PromptTemplate[AnalysisReport]` following the
"prompt is the agent" philosophy:

- **System context** — Role as a debugging expert. Structured output
  requirement (the report schema above).

- **Bundle context section** — Extracted bundle data injected as structured
  content the agent can reference directly.

- **Read-only tools** — `read_bundle_file(path)`, `query_logs(filter)`,
  `query_transcript(filter)`, `get_session_slice(type)`. No writes, no
  network.

- **Policy section** — Declarative constraints:
  - MUST produce a valid `AnalysisReport` JSON
  - MUST identify outcome (success/failure/partial)
  - MUST surface all errors with impact assessment
  - MUST NOT speculate about external causes without bundle evidence
  - SHOULD correlate errors with budget/timing impact
  - SHOULD identify tool call anti-patterns (retry storms, redundant reads)

### Persistence

Reports are saved adjacent to the bundle:

```
./debug/
  ├── {bundle_id}_{timestamp}.zip
  └── {bundle_id}_{timestamp}_analysis.json
```

The file wraps the report with metadata:

```json
{
  "_meta": {
    "analyzer_version": "1.0.0",
    "adapter_used": "claude_agent_sdk",
    "tokens_consumed": 12500,
    "wall_time_seconds": 18.4,
    "bundle_checksum": "sha256:abc123..."
  },
  "report": { ... }
}
```

### UI Integration

A new **Analysis** tab (keyboard shortcut: `6`):

| UI State | Display |
|----------|---------|
| No analysis | "Analyze Bundle" button |
| Running | Spinner with elapsed time; poll interval 2s |
| Completed | Rendered report with collapsible sections |
| Failed | Error message with "Retry" button |
| Cached | Report with "Re-analyze" option and timestamp |

Report sections render as collapsible cards: Summary, Errors, Timeline, Tool
Analysis, Budget, Recommendations.

### Configuration

```bash
wink debug <bundle.zip> --analysis-adapter claude_agent_sdk
wink debug <bundle.zip> --no-analysis    # Disable the feature
```

### Invariants

1. At most one analysis job per bundle (concurrent dispatch returns 409)
2. Analysis tools are read-only—never mutate bundle or filesystem
3. Cached report served until explicit re-run
4. Budget-bounded—agent cannot exceed configured token limit
5. Entirely optional—all existing views work without it

## Limitations

The current implementation works well for small debug bundles but becomes
difficult to use for typical production runs:

| Scenario | Challenge |
|----------|-----------|
| 1-2 hour runs | No way to navigate by time |
| Hundreds of tool calls | Events buried in noise |
| Multiple errors | No aggregation or highlighting |
| Cross-cutting concerns | Search is siloed per view |
| State evolution | No way to see how state changed over time |

## References

- Debug bundle format: `specs/DEBUG_BUNDLE.md`
- Logging specification: `specs/LOGGING.md`
- Session state: `specs/SESSIONS.md`
- Agent loop: `specs/AGENT_LOOP.md`
- Provider adapters: `specs/ADAPTERS.md`
- Tool runtime: `specs/TOOLS.md`

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

### Motivation

Manual inspection of debug bundles is effective for small runs but breaks down
for production-scale sessions. A 90-minute run with hundreds of tool calls,
thousands of log entries, and multiple error cascades requires significant human
effort to triage. Analysis jobs let an agent do the heavy lifting—reading the
full bundle contents, correlating events across views, and producing a
structured report—so the human reviewer starts from insight rather than raw
data.

### Concept

An **analysis job** is an agent execution dispatched from the `wink debug` web
UI against the currently loaded bundle. The user clicks "Analyze" (or presses a
keyboard shortcut), the server extracts relevant bundle contents, dispatches an
`AgentLoop` execution with a dedicated analysis prompt, and streams results back
to the UI as they become available.

```
┌─────────────────────────────────────────────────────────┐
│  wink debug UI                                          │
│                                                         │
│  ┌──────────┐   POST /api/analysis                      │
│  │ Analyze  │──────────────────────┐                    │
│  └──────────┘                      ▼                    │
│                           ┌─────────────────┐           │
│                           │  Debug Server   │           │
│                           │  (FastAPI)      │           │
│                           └────────┬────────┘           │
│                                    │                    │
│         ┌──────────────────────────┼──────────────┐     │
│         │  Analysis Job            │              │     │
│         │                          ▼              │     │
│         │  ┌──────────────────────────────────┐   │     │
│         │  │  AgentLoop.execute()             │   │     │
│         │  │  prompt: AnalysisBundlePrompt    │   │     │
│         │  │  adapter: configured provider    │   │     │
│         │  └──────────────────────────────────┘   │     │
│         │         │                               │     │
│         │         ▼                               │     │
│         │  ┌──────────────────────────────────┐   │     │
│         │  │  Analysis Report (JSON)          │   │     │
│         │  └──────────────────────────────────┘   │     │
│         └─────────────────────────────────────────┘     │
│                           │                             │
│         GET /api/analysis/status (poll / SSE)           │
│                           │                             │
│                           ▼                             │
│  ┌──────────────────────────────────────────────┐       │
│  │  Analysis View (new tab)                     │       │
│  │  - Executive summary                         │       │
│  │  - Error analysis                            │       │
│  │  - Timeline of significant events            │       │
│  │  - Tool call patterns                        │       │
│  │  - Recommendations                           │       │
│  └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

### Job Lifecycle

1. **Dispatch**: User triggers analysis from the UI. The server validates that
   no analysis is already running for this bundle, then starts the job in a
   background thread.

2. **Bundle extraction**: The server reads bundle contents into a structured
   context payload: manifest, request input/output, session state, logs
   (sampled if over a size threshold), transcript entries, metrics, error
   details, and config. Filesystem contents are summarized (file listing with
   sizes) rather than included verbatim.

3. **Agent execution**: An `AgentLoop` executes with a purpose-built analysis
   prompt. The prompt instructs the agent to act as a debugging expert,
   producing a structured JSON report. The agent has read-only access to bundle
   contents via tools (no filesystem writes, no network access).

4. **Progress reporting**: The job transitions through states:
   `pending → running → completed | failed`. The UI polls (or receives SSE
   events) to track progress and display incremental status updates.

5. **Report delivery**: On completion, the analysis report is persisted
   alongside the bundle as `{bundle_id}_analysis.json` and returned to the UI
   for rendering in a dedicated Analysis view.

6. **Caching**: Completed analysis reports are cached. Reopening the same
   bundle surfaces the existing report without re-running the agent. Users can
   explicitly re-run analysis to get a fresh report.

### Job States

```
pending ──→ running ──→ completed
                  │
                  └────→ failed
```

| State | Description |
|-------|-------------|
| `pending` | Job accepted, queued for execution |
| `running` | Agent is executing; progress updates available |
| `completed` | Report ready; persisted to disk |
| `failed` | Agent execution failed; error details available |

### Analysis Report Schema

The agent produces a structured JSON report:

```json
{
  "version": "1.0.0",
  "bundle_id": "uuid",
  "analyzed_at": "2024-01-15T10:35:00+00:00",
  "summary": {
    "outcome": "success | partial_success | failure",
    "one_liner": "Agent completed code review but missed 2 of 5 files due to budget exhaustion",
    "duration_assessment": "Run took 47 minutes; 60% of time spent in filesystem operations",
    "key_findings": [
      "Budget exhausted at turn 23 of estimated 30 needed",
      "Repeated file-read failures on locked resources caused 8 retry loops",
      "Final output was incomplete but structurally valid"
    ]
  },
  "errors": [
    {
      "phase": "tool_execution",
      "event": "filesystem.read_failed",
      "count": 8,
      "first_occurrence": "2024-01-15T10:12:00+00:00",
      "last_occurrence": "2024-01-15T10:18:00+00:00",
      "description": "Repeated failures reading locked file 'data/report.csv'",
      "impact": "Agent entered retry loop consuming 15% of total budget",
      "recommendation": "Add file-lock detection to avoid retry storms"
    }
  ],
  "timeline": [
    {
      "timestamp": "2024-01-15T10:05:00+00:00",
      "phase": "planning",
      "description": "Agent began task decomposition",
      "significance": "normal"
    },
    {
      "timestamp": "2024-01-15T10:12:00+00:00",
      "phase": "tool_execution",
      "description": "First filesystem read failure; retry loop begins",
      "significance": "warning"
    },
    {
      "timestamp": "2024-01-15T10:25:00+00:00",
      "phase": "budget",
      "description": "Budget exhausted; agent forced to finalize",
      "significance": "critical"
    }
  ],
  "tool_analysis": {
    "total_calls": 45,
    "unique_tools": ["read_file", "write_file", "run_tests", "search"],
    "failure_rate": 0.18,
    "patterns": [
      "read_file called 22 times (49% of all calls); 8 failures (36% failure rate)",
      "Longest tool execution: run_tests at 12.3 seconds"
    ]
  },
  "budget": {
    "tokens_used": 48000,
    "tokens_limit": 50000,
    "utilization": 0.96,
    "assessment": "Near-complete budget utilization; task was under-budgeted"
  },
  "recommendations": [
    {
      "priority": "high",
      "category": "budget",
      "title": "Increase token budget for code review tasks",
      "detail": "This task exhausted 96% of budget before completion. Consider a 50% budget increase for similar tasks."
    },
    {
      "priority": "medium",
      "category": "tools",
      "title": "Add file availability check before read operations",
      "detail": "8 consecutive read failures suggest the agent should probe file availability before committing to a read."
    }
  ]
}
```

### API Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/api/analysis` | `POST` | Dispatch a new analysis job for the current bundle |
| `/api/analysis/status` | `GET` | Current job state, progress, and partial results |
| `/api/analysis/report` | `GET` | Completed analysis report (404 if not ready) |
| `/api/analysis/cancel` | `POST` | Cancel a running analysis job |
| `/api/analysis/events` | `GET` | SSE stream for real-time progress updates |

#### POST /api/analysis

Request body (all fields optional):

```json
{
  "adapter": "claude_agent_sdk",
  "budget": { "max_tokens": 50000 },
  "focus": ["errors", "timeline", "budget"]
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `adapter` | `string` | Server default | Which provider adapter to use |
| `budget` | `object` | Reasonable default | Token/time budget for the analysis agent |
| `focus` | `string[]` | All sections | Limit analysis to specific report sections |

Response: `202 Accepted` with `{ "job_id": "uuid", "status": "pending" }`

#### GET /api/analysis/status

Response:

```json
{
  "job_id": "uuid",
  "status": "running",
  "started_at": "2024-01-15T10:35:00+00:00",
  "progress": "Analyzing tool call patterns...",
  "elapsed_seconds": 12
}
```

#### GET /api/analysis/events (SSE)

Server-Sent Events stream:

```
event: progress
data: {"status": "running", "progress": "Reading session state..."}

event: progress
data: {"status": "running", "progress": "Correlating error patterns..."}

event: complete
data: {"status": "completed", "report_url": "/api/analysis/report"}
```

### Configuration

Analysis job configuration is provided when starting the `wink debug` server:

```bash
wink debug <bundle.zip> --analysis-adapter claude_agent_sdk
wink debug <bundle.zip> --analysis-budget 50000
wink debug <bundle.zip> --no-analysis    # Disable analysis feature
```

| CLI Option | Default | Description |
|------------|---------|-------------|
| `--analysis-adapter` | Auto-detect | Provider adapter for analysis agent |
| `--analysis-budget` | `50000` tokens | Default token budget for analysis jobs |
| `--no-analysis` | `false` | Disable the analysis job feature entirely |

Programmatic configuration via `DebugServerConfig`:

```python
@dataclass(frozen=True, slots=True)
class AnalysisJobConfig:
    adapter: str = "claude_agent_sdk"
    budget: Budget | None = None
    enabled: bool = True

@dataclass(frozen=True, slots=True)
class DebugServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    open_browser: bool = True
    analysis: AnalysisJobConfig | None = None
```

### Analysis Prompt Design

The analysis prompt follows WINK's "prompt is the agent" philosophy. It is a
`PromptTemplate[AnalysisReport]` with sections that provide:

1. **System context**: Role as a debugging expert analyzing agent execution
   bundles. Structured output requirement (JSON report schema).

2. **Bundle context section**: Injects the extracted bundle data—manifest,
   request, session state, logs, metrics, errors—as structured content the
   agent can reference.

3. **Analysis tools** (read-only):
   - `read_bundle_file(path)` — Read a specific file from the bundle
   - `query_logs(filter)` — Query log entries with filters
   - `query_transcript(filter)` — Query transcript entries
   - `get_session_slice(type)` — Read a specific session slice

4. **Analysis policy section**: Declarative constraints (not a workflow):
   - MUST produce a valid `AnalysisReport` JSON
   - MUST identify the outcome (success/failure/partial)
   - MUST surface all errors with impact assessment
   - MUST NOT speculate about external causes without evidence in the bundle
   - SHOULD correlate errors with budget/timing impact
   - SHOULD identify tool call anti-patterns (retry storms, redundant reads)

### UI Integration

#### Analysis View (New Tab)

A new **Analysis** tab (keyboard shortcut: `6`) appears in the navigation bar.
States:

| UI State | Display |
|----------|---------|
| No analysis | Prompt with "Analyze Bundle" button and description |
| Pending/Running | Progress indicator with status text from SSE stream |
| Completed | Rendered report with expandable sections |
| Failed | Error message with "Retry" button |
| Cached | Report with "Re-analyze" option and timestamp of last analysis |

#### Report Rendering

Each report section maps to a collapsible card:

- **Summary** — Outcome badge, one-liner, key findings as bullet list
- **Errors** — Table with severity indicators, expandable detail rows
- **Timeline** — Vertical timeline with significance-based color coding
  (normal/warning/critical)
- **Tool Analysis** — Bar chart of tool call distribution, failure rate
  highlights
- **Budget** — Utilization gauge, assessment text
- **Recommendations** — Priority-sorted cards with category badges

#### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `6` | Switch to Analysis view |
| `A` | Start analysis (when in Analysis view, no job running) |
| `Esc` | Cancel running analysis |

### Bundle-Side Persistence

Completed analysis reports are saved adjacent to the bundle:

```
./debug/
  ├── {bundle_id}_{timestamp}.zip
  └── {bundle_id}_{timestamp}_analysis.json
```

This allows analysis results to survive server restarts and be shared alongside
bundles. The report file follows the Analysis Report Schema defined above, with
an additional `_meta` envelope:

```json
{
  "_meta": {
    "analyzer_version": "1.0.0",
    "adapter_used": "claude_agent_sdk",
    "tokens_consumed": 12500,
    "wall_time_seconds": 18.4,
    "bundle_checksum": "sha256:abc123..."
  },
  "report": { "...analysis report..." }
}
```

### Security Considerations

- **Read-only tools**: The analysis agent cannot modify the bundle, filesystem,
  or any external state. Tools are strictly read-only against bundle contents.
- **No network access**: The analysis prompt does not include network-capable
  tools. The agent operates exclusively on data already present in the bundle.
- **Sensitive data**: Bundles may contain secrets. The analysis report may
  quote bundle contents verbatim. The report inherits the same access controls
  as the bundle itself.
- **Local execution**: Analysis runs on the same machine as the debug server.
  No bundle data leaves the machine unless explicitly configured otherwise.
- **Budget limits**: Analysis jobs enforce token budgets to prevent runaway
  costs. The default budget is capped and configurable.

### Error Handling

| Failure Mode | Behavior |
|--------------|----------|
| No adapter configured | UI shows setup instructions; "Analyze" button disabled |
| Adapter authentication failure | Job transitions to `failed`; error includes setup guidance |
| Agent timeout / budget exhaustion | Partial report returned if available; status shows `failed` |
| Bundle too large for context | Server pre-filters content; logs sampled; filesystem summarized |
| Concurrent analysis request | 409 Conflict; UI shows existing job status |
| Server restart during job | Job state lost; UI detects disconnect and shows retry option |

### Invariants

1. **At most one analysis job per bundle**: Concurrent dispatch returns 409
2. **Read-only agent**: Analysis tools never mutate bundle or filesystem state
3. **Idempotent caching**: Same bundle always serves cached report until explicit re-run
4. **Budget-bounded**: Analysis agent cannot exceed configured token budget
5. **Graceful degradation**: Analysis feature is entirely optional; all existing
   views work without it

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

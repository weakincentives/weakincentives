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
    ├── config.json            # Runtime configuration
    ├── run_context.json       # Execution context
    └── filesystem/            # Workspace snapshot (optional)
        └── ...
```

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
- Session state: `specs/SESSION.md`

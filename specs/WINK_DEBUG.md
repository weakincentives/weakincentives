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
    ├── HTTP Server (localhost:8765)
    │   ├── /api/meta          - Bundle metadata
    │   ├── /api/slices/:type  - Session slice data
    │   ├── /api/logs          - Log entries (paginated)
    │   ├── /api/request/*     - Task input/output
    │   ├── /api/files         - Filesystem listing
    │   └── /api/file/:path    - File content
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
| **Logs** | Browse execution logs | Level filters, search | Scrollable log entries |
| **Task** | View request input/output | Input/Output toggle | Tree viewer with depth control |
| **Filesystem** | Browse workspace snapshot | File list with filter | File content viewer |

### Navigation

- **Tabs**: Switch views via numbered tabs (1-4 keyboard shortcuts)
- **Bundle selector**: Dropdown to switch between bundles in directory
- **Keyboard shortcuts**: J/K navigation, / for search, R to reload

### Bundle Contents

A debug bundle (`.zip`) contains:

```
{bundle_id}_{timestamp}.zip
├── meta.json           # Bundle metadata
├── config.json         # Runtime configuration
├── run_context.json    # Execution context
├── request_input.json  # Task input
├── request_output.json # Task output (if completed)
├── session_after.json  # Final session state
├── logs.jsonl          # Structured log entries
└── filesystem/         # Workspace snapshot (optional)
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

## Proposed Enhancements

### P0: Global Search

**Goal**: Find anything, anywhere, instantly.

Add a search overlay accessible via `/` from any view:

```
┌─────────────────────────────────────────────────────────────┐
│  🔍 config.yaml                                         ✕   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Logs (12 matches)                                          │
│  ├─ 00:05:23 INFO  Reading config.yaml for settings...      │
│  ├─ 00:05:24 INFO  Config loaded successfully               │
│  └─ 00:23:45 ERROR FileNotFoundError: config.yaml           │
│                                                              │
│  Events (3 matches)                                         │
│  ├─ 00:05:23 file_read("config.yaml") → 234 bytes           │
│  ├─ 00:23:45 file_read("config.yaml") → error               │
│  └─ 00:45:12 file_write("config.yaml") → success            │
│                                                              │
│  Filesystem (1 match)                                       │
│  └─ config.yaml (234 bytes, modified 00:45:12)              │
│                                                              │
│  Session State (2 matches)                                  │
│  ├─ ConfigSlice.path = "config.yaml"                        │
│  └─ FileTracker.files[2] = "config.yaml"                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Behavior**:

- Opens as modal overlay (like current shortcuts overlay)
- Searches across: logs, events, filesystem, session state
- Results grouped by source with match count
- Click result → navigate to that item in appropriate view
- Keyboard navigation: arrow keys, Enter to select, Esc to close

**Implementation notes**:

- Client-side search over already-loaded data (instant)
- For large bundles, may need server-side search endpoint
- Highlight matching text in results
- Remember last search across view switches

### P1: Interactive Timeline

**Goal**: Visualize the run over time, spot patterns, navigate by clicking.

Replace or augment the Overview mini-timeline with a full interactive timeline:

```
┌─────────────────────────────────────────────────────────────┐
│  TIMELINE                                    [Zoom: ──●──]  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  00:00     00:15     00:30     00:45     01:00     01:15    │
│  │─────────│─────────│─────────│─────────│─────────│        │
│                                                              │
│  file_read  ▓▓▓░▓▓▓▓▓░░░▓▓▓▓▓░▓▓▓▓▓▓▓▓░░▓▓▓░░░▓▓▓▓▓        │
│  bash       ░░▓▓▓░░░▓▓▓▓▓░░░░░▓▓▓▓░░░▓▓▓▓▓░░░░░▓▓░░        │
│  file_write ░░░░▓░░░░░▓▓░░░░░░░░▓▓▓░░░░░▓▓░░░░░▓▓░░        │
│  grep       ░░░░░░▓▓▓░░░░▓▓▓▓▓▓░░░░░░░░░░░▓▓▓░░░░░░        │
│                                                              │
│  errors     ░░░░░░░░░░░█░░░░░░░░░░░░█░░░░░░░░░░░░░░        │
│                                                              │
│  ────────────────────────┼──────────────────────────        │
│                       00:32:45                               │
│                       Selected: bash("pytest tests/")        │
│                       Duration: 2.3s | Status: ✓             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Features**:

- Swim lanes by tool type (or grouped by category)
- Error markers as distinct row or overlay
- Zoom control (1m / 5m / 15m / 1h granularity)
- Click to select → show details below
- Drag to select time range → filter other views
- Hover for quick preview

**Implementation notes**:

- Use canvas or SVG for rendering (DOM won't scale)
- Aggregate events into buckets at zoom levels
- Consider virtualization for very long runs

### P1: Error Trail

**Goal**: For each error, automatically show the investigation context.

When viewing an error (in Logs or Timeline), provide an expandable "Error Trail":

```
┌─────────────────────────────────────────────────────────────┐
│  ERROR at 00:23:45                                          │
│  FileNotFoundError: No such file: 'config.yaml'             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ▾ What happened before (last 5 events)                     │
│    00:23:38  bash("cd /workspace/app")        ✓             │
│    00:23:40  file_read("src/main.py")         ✓  234 bytes  │
│    00:23:42  grep("import config", "*.py")    ✓  3 matches  │
│    00:23:44  file_read("config.yaml")         ✗  ← ERROR    │
│                                                              │
│  ▾ State at error                                           │
│    working_directory: "/workspace/app"                      │
│    files_read: ["src/main.py"]                              │
│    current_task: "Load configuration"                       │
│                                                              │
│  ▾ Related logs (±5 seconds)                                │
│    00:23:43 DEBUG Looking for configuration file...         │
│    00:23:44 DEBUG Trying config.yaml                        │
│    00:23:45 ERROR FileNotFoundError: config.yaml            │
│    00:23:45 INFO  Will try alternate locations...           │
│                                                              │
│  ▾ Resolution (what happened after)                         │
│    00:23:48  file_read("/etc/app/config.yaml") ✓            │
│    00:23:49  Continued successfully                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Implementation notes**:

- Auto-generated from event sequence around error timestamp
- "State at error" requires session snapshots or reconstruction
- "Resolution" section only shown if execution continued
- Could be a slide-out panel or inline expansion

### P2: Linked Event Detail

**Goal**: See full context for any event without switching views.

When clicking an event (tool call) anywhere in the UI:

```
┌─────────────────────────────────────────────────────────────┐
│  file_write                                             ✕   │
│  Path: src/utils/config.py                                  │
├─────────────────────────────────────────────────────────────┤
│  Time: 00:34:12    Duration: 45ms    Status: ✓ Success      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ▸ Input Parameters                                         │
│    path: "src/utils/config.py"                              │
│    content: "# Configuration utilities\n\ndef load..."     │
│                                                              │
│  ▸ Output                                                   │
│    bytes_written: 1247                                      │
│    created: false                                           │
│                                                              │
│  ▸ Logs During Execution (2 entries)                        │
│    00:34:12 DEBUG Writing 1247 bytes to config.py           │
│    00:34:12 DEBUG File updated successfully                 │
│                                                              │
│  ▸ State Changes                                            │
│    FileTracker.written_files: +1 entry                      │
│    Metrics.file_writes: 44 → 45                             │
│                                                              │
│  ─────────────────────────────────────────────────────────  │
│  ← bash("black src/")              file_read("setup.py") →  │
│     Previous                              Next               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Implementation notes**:

- Opens as slide-out panel or modal
- Previous/Next navigation for sequential exploration
- State changes require diffing session snapshots
- "Logs During" filters by timestamp window

### P2: Saved Filters & Bookmarks

**Goal**: Remember important filters and mark points of interest.

**Saved Filters**:

```
┌────────────────────────┐
│  Saved Filters         │
│  • Errors only     [x] │
│  • File operations [x] │
│  • Last 30 min     [ ] │
│  + Save current...     │
└────────────────────────┘
```

**Bookmarks**:

```
┌────────────────────────────────────────┐
│  Bookmarks                             │
│  📍 00:23:45 "First config error"      │
│  📍 00:45:12 "Retry logic kicked in"   │
│  📍 01:12:33 "Network issues start"    │
│  + Add bookmark at current position    │
└────────────────────────────────────────┘
```

**Implementation notes**:

- Store in localStorage per bundle ID
- Export/import bookmarks for sharing
- Bookmarks visible as markers on timeline

## Implementation Phases

### Phase 1: Foundation

- [ ] Global Search (across all loaded data)
- [ ] Error highlighting in existing views

### Phase 2: Timeline

- [ ] Interactive Timeline view
- [ ] Zoom and pan controls
- [ ] Click-to-navigate integration

### Phase 3: Context

- [ ] Error Trail auto-generation
- [ ] Linked Event Detail panel
- [ ] Cross-view navigation

### Phase 4: Power Features

- [ ] Saved filters
- [ ] Bookmarks
- [ ] Export filtered views

## Data Requirements

Some features require additional data in the debug bundle:

| Feature | Current Support | Enhancement Needed |
|---------|-----------------|-------------------|
| Duration | ✓ Timestamps in meta | None |
| Tool call counts | ✓ Events in session | None |
| Error list | ✓ Logs have level | None |
| Timeline | ✓ Timestamps on events | Ensure all events have timestamps |
| State changes | ✗ Only final state | Periodic snapshots or event sourcing |
| Event correlation | ✗ No linking | Add correlation IDs to logs |

## API Extensions

New endpoints to support enhanced features:

```
GET /api/search?q=term
    Returns: { logs: [...], events: [...], files: [...], state: [...] }

GET /api/timeline?from=0&to=3600&bucket=60
    Returns: { buckets: [{ time, events: [...] }] }

GET /api/event/:id/context
    Returns: { event, logs_during, state_before, state_after, prev, next }

GET /api/errors
    Returns: [{ error, timestamp, trail: { before, state, logs, after } }]
```

## Success Metrics

The enhanced debug viewer should enable users to:

1. **Find any piece of data in < 5 seconds** (Global Search)
1. **Navigate to any point in time in < 3 clicks** (Timeline)
1. **Understand an error's cause in < 1 minute** (Error Trail)

## References

- Debug bundle format: `specs/DEBUG_BUNDLE.md`
- Logging specification: `specs/LOGGING.md`
- Session state: `specs/SESSION.md`

# WINK Overrides Editor Specification

## Purpose

The `wink overrides` command launches a local web server for editing prompt
overrides extracted from session snapshot files. It provides a browser-based UI
for inspecting prompts that were rendered during a session, editing their
section and tool overrides, and persisting changes to the local override store.

This enables prompt iteration workflows where developers:

1. Run an agent session and capture a snapshot
1. Inspect the actual prompts that were rendered
1. Edit overrides without modifying source files
1. Re-run the session to validate changes

## CLI Contract

```
wink overrides <snapshot_path> [--host HOST] [--port PORT] [--open-browser|--no-open-browser] [--tag TAG] [--store-root PATH]
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `snapshot_path` | Yes | - | Path to a JSONL snapshot file or directory containing snapshots |
| `--host` | No | `127.0.0.1` | Host interface to bind the server |
| `--port` | No | `8001` | Port to bind the server |
| `--open-browser` | No | `True` | Open the default browser automatically |
| `--tag` | No | `latest` | Override tag to edit (e.g., `latest`, `stable`, `v1`) |
| `--store-root` | No | Auto-detect | Root path for `LocalPromptOverridesStore` |

### Global Options

The CLI inherits global options from the `wink` command:

| Option | Default | Description |
|--------|---------|-------------|
| `--log-level` | `None` | Override log level (CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET) |
| `--json-logs` | `True` | Emit structured JSON logs (disable with `--no-json-logs`) |

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Server stopped normally |
| `2` | Snapshot validation failed at startup |
| `3` | Server failed to start |
| `4` | No `PromptRendered` events found in snapshot |

## Prompt Extraction

### PromptDescriptor Discovery

The server extracts `PromptDescriptor` instances from `PromptRendered` events
stored in snapshot slices:

1. Load the snapshot file(s) using the same loader as `wink debug`
1. Locate the `PromptRendered` slice (type: `weakincentives.runtime.events:PromptRendered`)
1. For each `PromptRendered` event with a non-null `descriptor` field:
   - Extract the `PromptDescriptor` (namespace, key, sections, tools)
   - Store the rendered prompt text for reference
   - Deduplicate by `(ns, key)` pair (keep the most recent by `created_at`)

### Extracted Prompt Data

```python
@FrozenDataclass()
class ExtractedPrompt:
    """Prompt metadata extracted from a PromptRendered event."""
    ns: str                           # Prompt namespace
    key: str                          # Prompt key
    name: str | None                  # Human-readable name
    descriptor: PromptDescriptor      # Full descriptor with hashes
    rendered_text: str                # Actual rendered prompt text
    created_at: datetime              # When the prompt was rendered
    event_id: UUID                    # Source event identifier
```

### Deduplication Rules

When multiple `PromptRendered` events share the same `(ns, key)`:

- Use the most recent event (by `created_at`) as the primary
- Log a warning if descriptors differ (hash drift)
- The UI shows all events but edits apply to the current descriptor

## Prerequisites

### Seeded Overrides Required

This tool edits existing override files—it cannot create them from scratch.
Before using `wink overrides`, prompts must be seeded via application code:

```python
from weakincentives.prompt.overrides import LocalPromptOverridesStore

store = LocalPromptOverridesStore()
store.seed(prompt, tag="latest")
```

Seeding requires access to the actual `PromptTemplate` or `Prompt` object, which
contains section templates and tool definitions. The `PromptDescriptor` extracted
from snapshots only contains hashes—not the original content needed to populate
a seed file.

If no override file exists for a prompt, the UI displays the prompt as read-only
with a message indicating that seeding is required.

## Override Resolution

### Loading Existing Overrides

For each extracted prompt, the server resolves existing overrides:

```python
store = LocalPromptOverridesStore(root_path=store_root)
override = store.resolve(descriptor=prompt.descriptor, tag=tag)
```

The resolved override provides:

- Current section overrides (with expected hashes)
- Current tool overrides (with expected contract hashes)
- Validation status (stale hashes are filtered out)

### Override State

```python
@FrozenDataclass()
class PromptOverrideState:
    """Combined state for editing a prompt's overrides."""
    prompt: ExtractedPrompt           # Extracted prompt data
    override: PromptOverride | None   # Current persisted override
    is_seeded: bool                   # Whether an override file exists
    sections: list[SectionState]      # Editable section states
    tools: list[ToolState]            # Editable tool states

@FrozenDataclass()
class SectionState:
    """State for a single section override."""
    path: tuple[str, ...]             # Section path (e.g., ("instructions",))
    number: str                       # Section numbering (e.g., "1.2")
    original_hash: HexDigest          # SHA-256 of original template
    current_body: str | None          # Override body if set
    is_overridden: bool               # Whether an override exists
    is_stale: bool                    # Hash mismatch detected

@FrozenDataclass()
class ToolState:
    """State for a single tool override."""
    name: str                         # Tool name
    path: tuple[str, ...]             # Section path where defined
    original_contract_hash: HexDigest # SHA-256 of tool contract
    current_description: str | None   # Override description if set
    current_param_descriptions: dict[str, str]
    is_overridden: bool
    is_stale: bool
```

## API Routes

### `GET /`

Returns the HTML index page for the overrides editor UI.

### `GET /api/prompts`

Lists all extracted prompts from the snapshot.

```json
[
  {
    "ns": "webapp/agents",
    "key": "code-reviewer",
    "name": "Code Reviewer",
    "section_count": 3,
    "tool_count": 2,
    "is_seeded": true,
    "has_overrides": true,
    "stale_count": 0,
    "created_at": "2024-01-15T10:30:00+00:00"
  }
]
```

### `GET /api/prompts/{encoded_ns}/{prompt_key}`

Returns full override state for a specific prompt.

The `encoded_ns` must be URL-encoded (e.g., `webapp/agents` → `webapp%2Fagents`).

```json
{
  "ns": "webapp/agents",
  "key": "code-reviewer",
  "name": "Code Reviewer",
  "rendered_prompt": {
    "text": "You are a code reviewer.\n\n## Instructions\n\nReview the code...",
    "html": "<p>You are a code reviewer.</p>\n<h2>Instructions</h2>\n<p>Review the code...</p>"
  },
  "created_at": "2024-01-15T10:30:00+00:00",
  "tag": "latest",
  "is_seeded": true,
  "sections": [
    {
      "path": ["instructions"],
      "number": "1",
      "original_hash": "a1b2c3d4...",
      "current_body": null,
      "is_overridden": false,
      "is_stale": false
    }
  ],
  "tools": [
    {
      "name": "search",
      "path": ["tools"],
      "original_contract_hash": "e5f6g7h8...",
      "current_description": "Search the codebase",
      "current_param_descriptions": {"query": "Keywords"},
      "is_overridden": true,
      "is_stale": false
    }
  ]
}
```

### `PUT /api/prompts/{encoded_ns}/{prompt_key}/sections/{encoded_path}`

Updates a section override. The `encoded_path` is the section path joined by
`/` and URL-encoded.

Request body:

```json
{
  "body": "New section content with $variables preserved"
}
```

Response:

```json
{
  "success": true,
  "section": {
    "path": ["instructions"],
    "current_body": "New section content...",
    "is_overridden": true,
    "is_stale": false
  }
}
```

### `DELETE /api/prompts/{encoded_ns}/{prompt_key}/sections/{encoded_path}`

Removes a section override, reverting to the original template.

Response:

```json
{
  "success": true,
  "section": {
    "path": ["instructions"],
    "current_body": null,
    "is_overridden": false,
    "is_stale": false
  }
}
```

### `PUT /api/prompts/{encoded_ns}/{prompt_key}/tools/{tool_name}`

Updates a tool override.

Request body:

```json
{
  "description": "Updated tool description",
  "param_descriptions": {
    "query": "Search keywords to find relevant code"
  }
}
```

### `DELETE /api/prompts/{encoded_ns}/{prompt_key}/tools/{tool_name}`

Removes a tool override.

### `DELETE /api/prompts/{encoded_ns}/{prompt_key}`

Deletes the entire override file for this prompt.

### `GET /api/config`

Returns current configuration.

```json
{
  "tag": "latest",
  "store_root": "/path/to/project",
  "snapshot_path": "/path/to/snapshot.jsonl"
}
```

### `POST /api/reload`

Reloads the snapshot file and refreshes extracted prompts. Does not reload
overrides from disk (override state is always read fresh on each request).

## Web UI

### Layout

The UI follows the same three-panel design as `wink debug`:

1. **Top Bar** (sticky header)

   - Title: "WINK Overrides Editor"
   - Current tag badge (e.g., `latest`)
   - Dark mode toggle (D key)
   - Reload button (R key)
   - Snapshot path display

1. **Sidebar** (left, 400px wide)

   - **Prompts Panel**: List of extracted prompts
     - Each item shows namespace/key, section count, tool count
     - Badge indicates if overrides exist
     - Warning icon if stale overrides detected
     - Click to select and view in content area

1. **Content Area** (main, flexible width)

   - **Prompt Header**: Namespace, key, name, creation timestamp
   - **Rendered Prompt Panel**: Full prompt text with markdown rendering
     - Collapsible panel showing the complete rendered prompt from the snapshot
     - Toggle between rendered markdown view and raw text view
     - Syntax highlighting for code blocks
     - Read-only display for reference while editing overrides
   - **Sections Panel**: Expandable list of sections
     - Each section shows path, number, override status
     - Inline editor for section body (CodeMirror or textarea)
     - "Override" / "Revert" buttons
     - Diff view toggle (show original vs override)
   - **Tools Panel**: Expandable list of tools
     - Each tool shows name, override status
     - Inline editors for description and param descriptions
     - "Override" / "Revert" buttons

### Interaction Patterns

**Selecting a Prompt:**

1. Click a prompt in the sidebar
1. Content area loads full override state
1. Rendered prompt panel shows the full prompt text
1. Sections and tools are shown in expandable accordions

**Viewing the Rendered Prompt:**

1. The rendered prompt panel is expanded by default when selecting a prompt
1. Click the panel header to collapse/expand
1. Toggle "Raw" button to switch between markdown-rendered and raw text views
1. Use the panel as reference while editing section overrides below

**Editing a Section Override:**

1. Expand the section accordion
1. Click "Edit" or the section body area
1. Editor appears with current content (override or original)
1. Edit the content
1. Click "Save" to persist via PUT API
1. UI updates to show "Overridden" badge

**Reverting an Override:**

1. Click "Revert" on an overridden section or tool
1. Confirmation dialog appears
1. On confirm, DELETE API is called
1. UI updates to show original content

**Diff View:**

1. Toggle "Show Diff" on an overridden section
1. Side-by-side or inline diff display
1. Original template on left, override on right
1. Syntax highlighting for both

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `D` | Toggle dark mode |
| `R` | Reload snapshot |
| `P` | Toggle rendered prompt panel collapse |
| `M` | Toggle markdown/raw view in rendered prompt |
| `↑/↓` | Navigate prompt list |
| `Enter` | Select highlighted prompt |
| `Esc` | Close editor / cancel edit |
| `Ctrl+S` | Save current edit |

### Visual Indicators

| State | Indicator |
|-------|-----------|
| Not seeded | Grayed out, "Seed Required" label, read-only |
| No override | Gray text, "Original" label |
| Overridden | Blue badge, "Overridden" label |
| Stale (hash mismatch) | Orange warning icon, "Stale" label |
| Unsaved changes | Yellow dot, "Modified" label |
| Save error | Red border, error message |

## Data Types

### OverridesStore

Thread-safe store managing override state:

```python
class OverridesStore:
    def __init__(
        self,
        snapshot_path: Path,
        *,
        tag: str = "latest",
        store_root: Path | None = None,
        loader: SnapshotLoader,
        logger: StructuredLogger,
    ) -> None: ...

    @property
    def prompts(self) -> tuple[ExtractedPrompt, ...]: ...
    @property
    def tag(self) -> str: ...
    @property
    def snapshot_path(self) -> Path: ...

    def get_prompt_state(
        self, ns: str, key: str
    ) -> PromptOverrideState | None: ...

    def update_section(
        self, ns: str, key: str, path: tuple[str, ...], body: str
    ) -> SectionState: ...

    def delete_section(
        self, ns: str, key: str, path: tuple[str, ...]
    ) -> SectionState: ...

    def update_tool(
        self,
        ns: str,
        key: str,
        tool_name: str,
        description: str | None,
        param_descriptions: dict[str, str],
    ) -> ToolState: ...

    def delete_tool(
        self, ns: str, key: str, tool_name: str
    ) -> ToolState: ...

    def delete_prompt_overrides(self, ns: str, key: str) -> None: ...

    def reload(self) -> None: ...
```

### Error Types

```python
class OverridesEditorError(WinkError, RuntimeError):
    """Base error for overrides editor operations."""

class PromptNotFoundError(OverridesEditorError):
    """Requested prompt not found in extracted prompts."""

class SectionNotFoundError(OverridesEditorError):
    """Requested section path not in prompt descriptor."""

class ToolNotFoundError(OverridesEditorError):
    """Requested tool not in prompt descriptor."""

class PromptNotSeededError(OverridesEditorError):
    """Prompt has no seed file; cannot create or edit overrides."""

class HashMismatchError(OverridesEditorError):
    """Override hash doesn't match current descriptor."""
```

## Persistence

### Write Operations

All write operations use `LocalPromptOverridesStore`:

```python
store = LocalPromptOverridesStore(root_path=store_root)

# Section override
store.set_section_override(
    prompt=extracted_prompt,  # Must implement PromptLike
    tag=tag,
    path=("instructions",),
    body="New content...",
)

# Full override upsert (for tool changes)
store.upsert(descriptor, override)

# Delete
store.delete(ns=ns, prompt_key=key, tag=tag)
```

### PromptLike Adapter

The `ExtractedPrompt` must be wrapped to satisfy `PromptLike`:

```python
class ExtractedPromptAdapter:
    """Adapts ExtractedPrompt to PromptLike protocol."""

    def __init__(self, extracted: ExtractedPrompt) -> None:
        self._extracted = extracted

    @property
    def descriptor(self) -> PromptDescriptor:
        return self._extracted.descriptor
```

### Storage Location

Overrides are stored at:

```
{store_root}/.weakincentives/prompts/overrides/{ns_segments}/{prompt_key}/{tag}.json
```

The `store_root` is determined by:

1. Explicit `--store-root` CLI argument
1. Git repository root containing the snapshot file
1. Parent directory of the snapshot file

## Logging

| Event | Level | Context |
|-------|-------|---------|
| `wink.overrides.start` | INFO | `url`, `snapshot_path`, `tag` |
| `wink.overrides.prompts_extracted` | INFO | `count` |
| `wink.overrides.no_prompts` | WARNING | `snapshot_path` |
| `wink.overrides.section_updated` | INFO | `ns`, `key`, `path` |
| `wink.overrides.section_deleted` | INFO | `ns`, `key`, `path` |
| `wink.overrides.tool_updated` | INFO | `ns`, `key`, `tool_name` |
| `wink.overrides.tool_deleted` | INFO | `ns`, `key`, `tool_name` |
| `wink.overrides.prompt_deleted` | INFO | `ns`, `key` |
| `wink.overrides.reload` | INFO | `snapshot_path` |
| `wink.overrides.error` | ERROR | `error`, `context` |

## Static Assets

The web UI is served from `src/weakincentives/cli/static/overrides/`:

| File | Purpose |
|------|---------|
| `index.html` | Main HTML page |
| `style.css` | Stylesheet (extends base wink styles) |
| `app.js` | Client-side JavaScript |

Static files are mounted at `/static/`.

Shared assets from `src/weakincentives/cli/static/` are also available for
consistent styling with `wink debug`.

## Implementation Notes

- Uses FastAPI for the HTTP server
- Uses uvicorn as the ASGI server
- Shares snapshot loading logic with `wink debug` (`debug_app.load_snapshot`)
- Uses markdown-it for server-side markdown rendering (same as `wink debug`)
- Rendered HTML is generated on-demand when fetching prompt details
- Browser opening uses a 0.2-second timer to avoid blocking server startup
- All write operations acquire filesystem locks via `OverrideFilesystem`
- Override state is always read fresh from disk to avoid stale cache issues
- The extracted prompts are cached in memory until explicit reload

## Workflow Example

```bash
# 1. Seed overrides for prompts you want to edit (one-time setup)
#    This must be done in application code where prompts are defined:
#
#    from weakincentives.prompt.overrides import LocalPromptOverridesStore
#    store = LocalPromptOverridesStore()
#    store.seed(code_reviewer_prompt, tag="stable")

# 2. Run an agent session with snapshot enabled
wink run agent.py --snapshot

# 3. Launch the overrides editor
wink overrides ./snapshots/session_abc123.jsonl --tag stable

# 4. In the browser:
#    - Select "webapp/agents:code-reviewer" prompt
#    - Expand "instructions" section
#    - Edit the content to improve clarity
#    - Click "Save"

# 5. Re-run the session with overrides
wink run agent.py --overrides-tag stable
```

## Future Extensions

- **Bulk operations**: Apply the same edit to multiple prompts
- **Export/import**: Export overrides as portable JSON for sharing
- **Version history**: Track changes to overrides over time
- **A/B comparison**: Compare behavior across different override tags
- **Live reload**: Watch snapshot file for changes and auto-refresh

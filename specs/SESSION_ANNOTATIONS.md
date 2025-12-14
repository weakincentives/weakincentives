# Session Annotations

This spec defines an annotation system for dataclasses stored in session
snapshots. Annotations reduce UI noise by marking which fields are most
important and how they should be rendered.

## Goals

1. **Focus UI on important fields** – Not all fields matter equally; annotations
   mark the subset that deserves prominence.
1. **Explicit rendering hints** – Replace heuristic markdown detection with
   explicit annotations.
1. **Self-describing JSONL files** – Include annotation metadata in the file
   header so tooling can interpret snapshots without hardcoded knowledge.
1. **Zero runtime cost** – Annotations are static metadata extracted at import
   time, not computed per-instance.

## Field Annotations

### Annotation Types

Field annotations are declared via the standard `dataclasses.field(metadata={})`
mechanism with reserved keys:

| Key | Type | Description |
|-----|------|-------------|
| `display` | `Literal["primary", "secondary", "hidden"]` | UI prominence level |
| `format` | `Literal["text", "markdown", "code", "json"]` | Rendering format hint |
| `label` | `str` | Human-readable label (defaults to field name) |
| `description` | `str` | Tooltip/help text (existing convention) |

### Display Levels

- **`primary`** – Always visible, shown prominently (title, status, key output)
- **`secondary`** – Shown in expanded/detail view (timestamps, identifiers)
- **`hidden`** – Internal bookkeeping, omitted from UI (version counters, hashes)

Fields without explicit `display` annotation default to `secondary`.

### Format Hints

- **`text`** – Plain text, escape for HTML display
- **`markdown`** – Render as Markdown (replaces heuristic detection)
- **`code`** – Render with syntax highlighting (monospace, preserve whitespace)
- **`json`** – Pretty-print as JSON with syntax highlighting

Fields without explicit `format` annotation default to `text`.

### Example

```python
from dataclasses import dataclass, field

@dataclass(slots=True, frozen=True)
class PlanStep:
    step_id: int = field(
        metadata={
            "display": "secondary",
            "description": "Stable identifier for the step.",
        }
    )
    title: str = field(
        metadata={
            "display": "primary",
            "label": "Step",
            "description": "Concise summary of the work item.",
        }
    )
    status: str = field(
        metadata={
            "display": "primary",
            "description": "Current progress: pending, in_progress, or done.",
        }
    )
    notes: str = field(
        default="",
        metadata={
            "display": "secondary",
            "format": "markdown",
            "description": "Extended notes with formatting.",
        }
    )
```

## Slice Annotations

Dataclasses that participate in session storage may also declare slice-level
metadata via a class attribute:

```python
@dataclass(slots=True, frozen=True)
class Plan:
    __slice_meta__ = SliceMeta(
        label="Execution Plan",
        description="Tracks objectives and steps for agent execution.",
        icon="clipboard-list",  # Optional icon hint for UI
        sort_key="created_at",  # Default sort field
        sort_order="desc",      # Default sort direction
    )

    objective: str = field(...)
    # ...
```

### SliceMeta Fields

| Field | Type | Description |
|-------|------|-------------|
| `label` | `str` | Human-readable name for the slice type |
| `description` | `str` | Explanation of what this slice contains |
| `icon` | `str \| None` | Icon identifier hint (e.g., Lucide icon name) |
| `sort_key` | `str \| None` | Default field to sort instances by |
| `sort_order` | `Literal["asc", "desc"]` | Default sort direction |

## JSONL Header Format

The first line of a JSONL snapshot file is a **header** containing annotation
metadata. Subsequent lines are snapshot records (unchanged format).

### Header Schema

```json
{
  "header": true,
  "annotation_version": "1",
  "slices": {
    "weakincentives.contrib.tools.Plan": {
      "label": "Execution Plan",
      "description": "Tracks objectives and steps...",
      "icon": "clipboard-list",
      "sort_key": "created_at",
      "sort_order": "desc",
      "fields": {
        "objective": {
          "display": "primary",
          "format": "text",
          "label": "Objective",
          "description": "The goal this plan achieves."
        },
        "status": {
          "display": "primary",
          "format": "text",
          "label": "Status",
          "description": "Current plan state."
        },
        "steps": {
          "display": "primary",
          "format": "json",
          "label": "Steps",
          "description": "Ordered list of plan steps."
        }
      }
    },
    "weakincentives.contrib.tools.PlanStep": {
      "label": "Plan Step",
      "fields": {
        "step_id": {"display": "secondary", "format": "text"},
        "title": {"display": "primary", "format": "text", "label": "Step"},
        "status": {"display": "primary", "format": "text"}
      }
    }
  }
}
```

### Header Detection

The header line is identified by `"header": true`. Parsers that don't understand
annotations can skip this line and process remaining lines as before (backward
compatible).

### Annotation Extraction

At JSONL write time, the system:

1. Collects all unique dataclass types present in the snapshot tree
1. Extracts field annotations from each type's `field.metadata`
1. Extracts slice annotations from `__slice_meta__` if present
1. Writes header as first line, then snapshot lines

## wink debug Integration

### Replacing Markdown Detection

The current `_render_markdown_values()` function uses regex heuristics to detect
markdown. This is replaced by checking the `format` annotation:

```python
# Before (heuristic)
if _looks_like_markdown(value):
    return {"__markdown__": {"text": value, "html": render(value)}}

# After (annotation-driven)
if field_meta.get("format") == "markdown":
    return {"__markdown__": {"text": value, "html": render(value)}}
```

### Rendering Pipeline

1. Load header from first JSONL line
1. Build `field_key → annotation` lookup from header
1. For each snapshot item:
   - Look up field annotations by `(slice_type, field_name)`
   - Apply format transformation based on `format` annotation
   - Filter/sort fields based on `display` level and UI mode

### UI Modes

The debug UI supports display modes:

- **Compact** – Show only `primary` fields
- **Detailed** – Show `primary` and `secondary` fields
- **Raw** – Show all fields including `hidden`

## Annotation Registry

Annotations are registered at module import time via a global registry:

```python
from weakincentives.runtime.annotations import register_annotations

# Called automatically when dataclass module is imported
register_annotations(Plan)
register_annotations(PlanStep)
```

The registry provides:

```python
def get_field_annotations(cls: type) -> dict[str, FieldAnnotation]: ...
def get_slice_meta(cls: type) -> SliceMeta | None: ...
def get_all_registered() -> dict[str, SliceAnnotations]: ...
```

## Migration

### Phase 1: Add Infrastructure

1. Define `FieldAnnotation`, `SliceMeta`, `SliceAnnotations` dataclasses
1. Implement annotation registry
1. Add header writing to `dump_session()`
1. Update `wink debug` to read header and use annotations

### Phase 2: Annotate Core Types

1. Add annotations to `Plan`, `PlanStep`
1. Add annotations to `VfsFile`, `VirtualFileSystem`
1. Add annotations to `AstevalResult`, `AstevalCode`
1. Add annotations to `WorkspaceDigest`
1. Add annotations to `ToolInvoked`, `PromptExecuted`

### Phase 3: Remove Heuristics

1. Remove `_MARKDOWN_PATTERNS` and `_looks_like_markdown()`
1. Remove `_MIN_MARKDOWN_LENGTH` threshold
1. Update tests to use annotation-based detection

## Backward Compatibility

- JSONL files without headers remain loadable (annotations unavailable)
- UI falls back to `secondary` display and `text` format when annotations missing
- Old `wink debug` versions skip header line (starts with `{"header":`)

## Design Decisions

### Why Field Metadata?

Using `dataclasses.field(metadata={})` keeps annotations co-located with field
definitions. No separate registry file to maintain. IDE support for navigation.

### Why JSONL Header?

Embedding annotations in the file makes snapshots self-describing. Tooling
doesn't need access to Python source to interpret fields. Header is optional
for backward compatibility.

### Why Not Instance-Level Annotations?

Instance-level annotations (per-value metadata) would bloat snapshot size and
complicate serialization. Type-level annotations cover the use case of UI
hints without per-instance overhead.

### Why Explicit Format Instead of Heuristics?

Heuristics are fragile. A field containing `# Comment` in code gets
misdetected as markdown. Explicit `format: "code"` is unambiguous.

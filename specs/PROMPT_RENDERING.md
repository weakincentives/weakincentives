# Prompt Rendering Pipeline Specification

## Purpose

This document describes the prompt rendering pipeline in WINK: how sections are
traversed, visibility is computed, templates are normalized, and the final
`RenderedPrompt` is produced. Understanding this pipeline is essential for
building custom sections, debugging rendering issues, and implementing
visibility-based progressive disclosure.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PROMPT RENDERING PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────────┘

   PromptTemplate              Registry                   Parameters
        │                         │                            │
        ▼                         ▼                            ▼
   ┌─────────┐            ┌──────────────┐            ┌──────────────┐
   │ Prompt  │────bind───▶│RegistrySnap- │◀───────────│ param_lookup │
   │Template │            │    shot      │            │ {type: inst} │
   └─────────┘            └──────────────┘            └──────────────┘
                                 │
                                 ▼
                        ┌────────────────┐
                        │PromptRenderer  │
                        │                │
                        │ 1. Iterate     │
                        │ 2. Visibility  │
                        │ 3. Render      │
                        │ 4. Collect     │
                        └────────────────┘
                                 │
                                 ▼
                        ┌────────────────┐
                        │RenderedPrompt │
                        │                │
                        │ .text          │
                        │ .tools         │
                        │ .structured_   │
                        │    output      │
                        └────────────────┘
```

## Component Hierarchy

### 1. PromptTemplate (Coordinate Point)

`PromptTemplate` is an immutable reference to a prompt definition:

```python
from weakincentives.prompt import PromptTemplate

template = PromptTemplate(
    ns="my-namespace",
    key="my-prompt",
    name="my_prompt",
)
```

Properties:

- `ns` - Namespace for grouping related prompts
- `key` - Unique identifier within namespace
- `name` - Human-readable name (derived from key if not provided)

### 2. Prompt (Bound Template)

`Prompt` binds sections and structured output to a template:

```python
from weakincentives import Prompt, MarkdownSection

prompt = Prompt[OutputType](
    ns="my-namespace",
    key="my-prompt",
    sections=[
        MarkdownSection(title="Instructions", template="...", key="instructions"),
    ],
    structured_output=StructuredOutputConfig(OutputType),
)
```

### 3. PromptRegistry / RegistrySnapshot

The registry indexes all sections for efficient traversal:

```python
# Registry is built from prompt.bind()
bound_prompt = prompt.bind(MyParams(value="test"))

# RegistrySnapshot locks the registry state
snapshot = bound_prompt._registry  # Internal access
```

Registry provides:

- `sections` - Flattened section tree as `SectionNode` sequence
- `params_types` - Set of expected parameter types
- `get_section(path)` - Lookup by section path

### 4. PromptRenderer

Orchestrates the rendering process:

```python
renderer = PromptRenderer(
    registry=snapshot,
    structured_output=structured_output_config,
)

rendered = renderer.render(
    param_lookup={MyParams: my_params},
    overrides=None,
    session=session,
)
```

## Rendering Stages

### Stage 1: Section Iteration

The renderer traverses sections in pre-order depth-first order:

```
Root
├── Section A (depth=0)
│   ├── Section A.1 (depth=1)
│   └── Section A.2 (depth=1)
└── Section B (depth=0)
    └── Section B.1 (depth=1)

Traversal: A → A.1 → A.2 → B → B.1
```

Each `SectionNode` contains:

```python
@dataclass
class SectionNode[ParamsT]:
    section: Section[ParamsT]
    path: tuple[str, ...]  # e.g., ("instructions", "context")
    depth: int             # Nesting level (0 = root)
```

### Stage 2: Enabled Filtering

Sections can be conditionally disabled via the `enabled` predicate:

```python
section = MarkdownSection(
    title="Debug Info",
    template="...",
    key="debug",
    enabled=lambda params: params.debug_mode,
)
```

Supported `enabled` signatures:

| Signature | Description |
|-----------|-------------|
| `() -> bool` | Static check |
| `(*, session) -> bool` | Session-aware check |
| `(params) -> bool` | Parameter-dependent check |
| `(params, *, session) -> bool` | Full context check |

Disabled sections and their children are skipped entirely.

### Stage 3: Visibility Computation

Each section has a `visibility` property determining how it renders:

```python
class SectionVisibility(Enum):
    FULL = "full"      # Render complete content
    SUMMARY = "summary"  # Render summary only
```

**Visibility determination order:**

1. **Session overrides** - Check `session[VisibilityOverrides]` for explicit
   override at this section path
2. **Section visibility selector** - Evaluate the section's `visibility`
   property

**Visibility selectors:**

```python
# Static visibility
visibility=SectionVisibility.FULL

# Zero-argument callable
visibility=lambda: SectionVisibility.SUMMARY

# Parameter-dependent
visibility=lambda params: (
    SectionVisibility.FULL if params.detailed else SectionVisibility.SUMMARY
)

# Session-aware
visibility=lambda params, *, session: (
    SectionVisibility.FULL
    if session[UserPrefs].latest().show_details
    else SectionVisibility.SUMMARY
)
```

**Summary rendering behavior:**

When a section renders with `SUMMARY` visibility:

1. The section's `summary` template is used instead of `template`
2. All children are skipped (not rendered)
3. Tools from this section and descendants are not collected
4. A suffix is appended indicating expansion is available

### Stage 4: Template Rendering

Each section type implements `render_body()`:

```python
class MarkdownSection(Section[ParamsT]):
    def render_body(
        self,
        params: ParamsT | None,
        *,
        path: SectionPath,
    ) -> str:
        template = string.Template(self.template)
        return template.safe_substitute(params_dict)
```

**Template substitution:**

```python
template = "Process $item_count items from $source"

# With params: ProcessParams(item_count=42, source="api")
# Renders: "Process 42 items from api"
```

**Override rendering:**

If an override exists for a section path, `render_override()` is called instead:

```python
def render_override(
    self,
    override_body: str,
    params: ParamsT | None,
    *,
    path: SectionPath,
) -> str:
    # Override replaces the entire section body
    return override_body
```

### Stage 5: Tool Collection

Tools are collected from enabled sections with FULL visibility:

```python
section = MarkdownSection(
    title="Search",
    template="Use the search tool...",
    key="search",
    tools=(search_tool, filter_tool),
)
```

**Collection rules:**

| Section State | Tools Collected? |
|---------------|------------------|
| Enabled + FULL visibility | Yes |
| Enabled + SUMMARY visibility | No |
| Disabled | No (section skipped) |

**Tool override application:**

```python
tool_overrides = {
    "search": ToolOverride(
        description="Custom description",
        field_descriptions={"query": "Search query string"},
    ),
}

rendered = renderer.render(
    param_lookup=param_lookup,
    tool_overrides=tool_overrides,
)
```

### Stage 6: Progressive Disclosure Tools

When sections are summarized, special tools are injected:

**`open_sections` tool:**

Injected when summarized sections have tools attached. Allows the model to
request section expansion to access hidden tools:

```python
# Model calls: open_sections(sections=["context.history"])
# Raises: VisibilityExpansionRequired
# Caller retries with expanded visibility
```

**`read_section` tool:**

Injected when summarized sections have no tools. Allows the model to read
section content without full expansion:

```python
# Model calls: read_section(section="context.history")
# Returns: Full section content as string
```

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           RENDERING DATA FLOW                            │
└──────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────┐
                    │  Sections   │
                    │   Tree      │
                    └──────┬──────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   _iter_enabled_       │
              │   sections()           │
              │                        │
              │ For each section:      │
              │ • Check enabled        │
              │ • Resolve params       │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │ effective_visibility() │
              │                        │
              │ 1. Session override?   │
              │ 2. Section selector    │
              └───────────┬────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
   ┌──────────────┐               ┌──────────────┐
   │ FULL         │               │ SUMMARY      │
   │              │               │              │
   │ render_body()│               │ summary +    │
   │ or           │               │ suffix       │
   │ render_      │               │              │
   │ override()   │               │ Skip         │
   │              │               │ children     │
   │ Collect      │               │              │
   │ tools        │               │ No tools     │
   └──────┬───────┘               └──────┬───────┘
          │                               │
          └───────────────┬───────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │   Join sections        │
              │   "\n\n".join(...)     │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │ Inject disclosure      │
              │ tools if needed:       │
              │                        │
              │ • open_sections        │
              │ • read_section         │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │   RenderedPrompt       │
              │                        │
              │ .text                  │
              │ .tools                 │
              │ .tool_param_           │
              │   descriptions         │
              │ .structured_output     │
              │ .descriptor            │
              └────────────────────────┘
```

## RenderedPrompt Structure

The final output contains everything needed for adapter evaluation:

```python
@FrozenDataclass()
class RenderedPrompt[OutputT]:
    text: str
    """Markdown text from all rendered sections."""

    structured_output: StructuredOutputConfig | None
    """Schema for structured output parsing."""

    deadline: Deadline | None
    """Wall-clock deadline for evaluation."""

    descriptor: PromptDescriptor | None
    """Version descriptor for override validation."""

    _tools: tuple[Tool, ...]
    """Tools from enabled FULL-visibility sections."""

    _tool_param_descriptions: Mapping[str, Mapping[str, str]]
    """Description overrides keyed by tool name."""
```

## Visibility Override System

### Session-Based Overrides

Visibility can be controlled at runtime via session state:

```python
from weakincentives.prompt import VisibilityOverrides, SectionVisibility

# Set override
overrides = VisibilityOverrides(
    overrides={
        ("context", "history"): SectionVisibility.FULL,
    }
)
session[VisibilityOverrides].seed(overrides)

# Rendering now uses FULL for context.history regardless of section default
```

### Expansion Request Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      VISIBILITY EXPANSION FLOW                          │
└─────────────────────────────────────────────────────────────────────────┘

   Model                    Adapter                    MainLoop
     │                         │                          │
     │  Call open_sections     │                          │
     │  ──────────────────────▶│                          │
     │                         │                          │
     │                         │  VisibilityExpansion-    │
     │                         │  Required exception      │
     │                         │  ─────────────────────▶  │
     │                         │                          │
     │                         │                          │ Update session
     │                         │                          │ VisibilityOverrides
     │                         │                          │
     │                         │  Re-render with new      │
     │                         │  visibility              │
     │                         │  ◀─────────────────────  │
     │                         │                          │
     │  New prompt with        │                          │
     │  expanded sections      │                          │
     │  ◀──────────────────────│                          │
     │                         │                          │
```

## Normalization

### Section Key Normalization

Section keys are validated and normalized:

```python
# Valid patterns: ^[a-z0-9][a-z0-9._-]{0,63}$
"instructions"      # Valid
"context.history"   # Valid (nested path)
"step-1"            # Valid
"Instructions"      # Invalid (uppercase)
"_private"          # Invalid (starts with underscore)
```

### Visibility Selector Normalization

All visibility selectors are normalized to a common signature:

```python
NormalizedVisibilitySelector = Callable[
    [SupportsDataclass | None, SessionProtocol | None],
    SectionVisibility
]
```

This allows consistent invocation regardless of original signature.

## Error Handling

### PromptValidationError

Raised during prompt construction:

```python
# Duplicate params type
PromptValidationError("Duplicate params type supplied to prompt.")

# Unexpected params type
PromptValidationError("Unexpected params type supplied to prompt.")

# Non-dataclass instance
PromptValidationError("Prompt expects dataclass instances.")
```

### PromptRenderError

Raised during rendering:

```python
# Template substitution failure
PromptRenderError(
    "Failed to render section template.",
    section_path=("instructions",),
    placeholder="$missing_var",
)
```

### VisibilityExpansionRequired

Raised when model requests section expansion:

```python
VisibilityExpansionRequired(
    "Visibility expansion required",
    requested_overrides={
        ("context", "history"): SectionVisibility.FULL,
    },
    reason="Model requested access to tools in summarized section",
    section_keys=("context.history",),
)
```

## Performance Considerations

1. **Registry snapshots are cached** - Building the registry is O(sections);
   subsequent renders reuse the snapshot.

2. **Visibility is computed lazily** - Only computed for enabled sections.

3. **Tool collection is single-pass** - Tools are collected during the main
   traversal, not in a separate pass.

4. **Override lookups are O(1)** - Stored in dictionaries keyed by path tuple.

## Testing Patterns

### Testing Section Rendering

```python
def test_section_renders_with_params():
    section = MarkdownSection(
        title="Test",
        template="Value: $value",
        key="test",
    )

    rendered = section.render_body(
        MyParams(value="hello"),
        path=("test",),
    )

    assert rendered == "# Test\n\nValue: hello"
```

### Testing Visibility Behavior

```python
def test_summary_visibility_skips_children():
    prompt = Prompt[Output](
        ns="test",
        key="test",
        sections=[
            MarkdownSection(
                title="Parent",
                template="Full content",
                summary="Summary only",
                key="parent",
                visibility=SectionVisibility.SUMMARY,
                children=[
                    MarkdownSection(
                        title="Child",
                        template="Should not appear",
                        key="child",
                    ),
                ],
            ),
        ],
    )

    rendered = prompt.bind().render()

    assert "Summary only" in rendered.text
    assert "Should not appear" not in rendered.text
```

### Testing Tool Collection

```python
def test_tools_not_collected_from_summary_sections():
    tool = Tool(name="test_tool", handler=my_handler)

    prompt = Prompt[Output](
        ns="test",
        key="test",
        sections=[
            MarkdownSection(
                title="Section",
                template="...",
                key="section",
                visibility=SectionVisibility.SUMMARY,
                tools=(tool,),
            ),
        ],
    )

    rendered = prompt.bind().render()

    assert tool not in rendered.tools
```

## Related Specifications

- `specs/PROMPTS.md` - Prompt system overview
- `specs/PROMPT_OPTIMIZATION.md` - Override system details
- `specs/TOOLS.md` - Tool definition and execution

# Prompt System Specification

## Purpose

The `Prompt` abstraction centralizes every string template that flows to an LLM
so the codebase has a single, inspectable source for system prompts and per-turn
instructions. This specification covers prompt construction, section composition,
structured output, and progressive disclosure.

## Guiding Principles

- **Type-Safety First**: Every placeholder maps to a dataclass field so issues
  surface early in development.
- **Strict, Predictable Failures**: Validation and render errors fail loudly
  with actionable context.
- **Composable Markdown Structure**: Hierarchical sections with deterministic
  heading levels keep prompts readable.
- **Minimal Templating Surface**: Limit to `Template.substitute` plus boolean
  selectors to prevent complex control flow.
- **Declarative over Imperative**: Prompts describe structure, not logic.

## Core Components

### PromptTemplate and Prompt

`PromptTemplate` is the configuration blueprint that owns a namespace (`ns`),
a required `key`, an optional `name`, and an ordered tree of `Section`
instances:

```python
template = PromptTemplate[OutputType](
    ns="demo",
    key="compose-email",
    name="compose_email",
    sections=[...],
)
```

`Prompt` is a wrapper that binds parameters to a template for rendering:

```python
# Create a Prompt and bind parameters
prompt = Prompt(template).bind(MyParams(field="value"))

# Render the prompt
rendered = prompt.render()
```

**Construction Rules:**

- `ns` and `key` are required and non-empty.
- The `(ns, key)` pair identifies a prompt for versioning and overrides.
- Section keys must match: `^[a-z0-9][a-z0-9._-]{0,63}$`
- Duplicate parameter types are allowed; instances fan out to matching sections.

### Section

Abstract base with metadata, `is_enabled`, `render`, child handling, and
override gating.

```python
class Section(ABC, Generic[ParamsT]):
    title: str
    key: str
    children: tuple[Section, ...] = ()
    tools: tuple[Tool, ...] = ()
    enabled: Callable[[ParamsT], bool] | None = None
    default_params: ParamsT | None = None
    accepts_overrides: bool = True
    visibility: SectionVisibility | Callable[[ParamsT], SectionVisibility] | Callable[[], SectionVisibility] = FULL
```

**Key Behaviors:**

- Sections must be specialized: `MarkdownSection[MyParams]`
- `accepts_overrides=False` excludes sections from the override system
- `visibility` controls full vs. summary rendering

### MarkdownSection

Default concrete section that dedents, strips, and runs `Template.substitute`:

```python
tone_section = MarkdownSection[ToneParams](
    title="Tone",
    key="tone",
    template="Target tone: ${tone}",
    summary="Tone guidance available.",  # Optional for progressive disclosure
)
```

## Rendering

`Prompt.render` accepts dataclass instances as positional arguments matched by
type. Rendering walks the section tree depth-first and produces markdown with
deterministic headings.

### Heading Levels

- Root sections: `##`
- Each depth level adds one `#` (depth 1 = `###`, depth 2 = `####`)
- Headings include numbering with a trailing period after the index: `## 1. Title`, `### 1.1. Subtitle`

### Parameter Lookup

The renderer builds a map of dataclass type to instance. When a section lacks
an override:

1. Use `default_params` if configured
1. Else use the first default for that type
1. Else instantiate with no arguments

Missing required fields raise `PromptRenderError`. Supplying the same
dataclass type more than once is rejected with `PromptValidationError` (the
renderer does **not** fan out duplicate param types).

### RenderedPrompt

```python
@FrozenDataclass()
class RenderedPrompt(Generic[OutputT]):
    text: str
    structured_output: StructuredOutputConfig | None = None
    deadline: Deadline | None = None
    descriptor: PromptDescriptor | None = None

    # Properties derived from structured_output
    @property
    def tools(self) -> tuple[Tool, ...]: ...
    @property
    def tool_param_descriptions(self) -> Mapping[str, Mapping[str, str]]: ...
    @property
    def output_type(self) -> type | None: ...
    @property
    def container(self) -> Literal["object", "array"] | None: ...
    @property
    def allow_extra_keys(self) -> bool | None: ...
```

## Structured Output

Prompts can declare typed outputs via generic specialization:

```python
@dataclass
class Summary:
    title: str
    gist: str

prompt = Prompt[Summary](...)
```

### Declaration

- `Prompt[T]` - JSON object output matching dataclass `T`
- `Prompt[list[T]]` - JSON array of objects matching `T`
- Non-dataclass types raise `PromptValidationError`

Providers must support native structured outputs (JSON schema response format)
for structured output to work correctly.

### Parsing

`parse_structured_output(output_text, rendered)` validates assistant responses:

1. **Extract JSON**: Prefer fenced `json` block, else parse entire message,
   else scan for `{...}` or `[...]`
1. **Validate container**: Object vs. array must match declaration
1. **Validate dataclass**: Required fields, no extra keys (unless allowed),
   conservative type coercions

Failures raise `OutputParseError` with the raw response attached.

### Configuration

```python
prompt = Prompt[Output](
    ...,
    allow_extra_keys=False,  # Default: reject unknown keys
)
```

## Progressive Disclosure

Sections can render with `SUMMARY` visibility to reduce token usage. The
`open_sections` tool lets models request expanded content.

### Section Visibility

```python
class SectionVisibility(Enum):
    FULL = auto()
    SUMMARY = auto()

section = MarkdownSection[Params](
    ...,
    template="Full detailed content...",
    summary="Brief overview available.",
    visibility=SectionVisibility.SUMMARY,
)
```

### Automatic Tool Registration

When any section has `SUMMARY` visibility, the framework injects the
`open_sections` tool:

```python
@dataclass
class OpenSectionsParams:
    section_keys: tuple[str, ...]  # Dot notation for nested sections
    reason: str                     # Why expansion is needed
```

### Exception-Based Signaling

The tool raises `VisibilityExpansionRequired` rather than returning a result:

```python
@dataclass
class VisibilityExpansionRequired(PromptError):
    requested_overrides: Mapping[tuple[str, ...], SectionVisibility]
    reason: str
    section_keys: tuple[str, ...]
```

### Caller Pattern

```python
visibility_overrides = {}
prompt = Prompt(template).bind(*params)

while True:
    try:
        rendered = prompt.render(visibility_overrides=visibility_overrides)
        response = adapter.evaluate(prompt, session=session, bus=bus)
        break
    except VisibilityExpansionRequired as e:
        visibility_overrides.update(e.requested_overrides)
```

### Summary Suffix

Summarized sections automatically append a suffix inviting the model to access
the full content. The suffix varies based on whether the section registers tools:

**Sections with tools:**

```
---
[This section is summarized. To view full content, call `open_sections` with key "context".]
```

**Sections without tools:**

```
---
[This section is summarized. Full content is available at `/context/section-key.md`.]
```

### Auto-Rendering to VFS

When a section subtree renders with `SUMMARY` visibility and does **not**
register any tools (neither the section itself nor any of its descendants),
the framework automatically:

1. **Renders the full content** of the section and its children as a markdown
   file
2. **Writes it to the VFS** at `/context/{section-key}.md`
3. **Appends a suffix** directing the model to read from the filesystem

This optimization avoids the `open_sections` round-trip for content-only
sections. Since no tools need to be collected, there's no need to re-render
the prompt—the model can access the content directly via `read_file`.

**Requirements:**

- A `VfsToolsSection` must be present in the prompt tree (provides the VFS
  session context and `read_file` tool)
- The section must have `visibility=SUMMARY` effective at render time
- The section subtree must register zero tools

**Nested Sections:**

For sections with children, the entire subtree is rendered into a single
markdown file. Heading levels are preserved relative to the section root.

```python
parent = MarkdownSection[Params](
    title="Reference",
    key="reference",
    template="Overview...",
    summary="Reference documentation available.",
    visibility=SectionVisibility.SUMMARY,
    children=[
        MarkdownSection[Params](
            title="API Guide",
            key="api",
            template="API details...",
        ),
        MarkdownSection[Params](
            title="Examples",
            key="examples",
            template="Example code...",
        ),
    ],
)
# Renders to /context/reference.md with:
# ## 1. Reference
# Overview...
# ### 1.1. API Guide
# API details...
# ### 1.2. Examples
# Example code...
```

**Interaction with open_sections:**

If a prompt contains both tool-bearing and tool-free summarized sections:

- Tool-bearing sections: `open_sections` is injected, standard suffix applies
- Tool-free sections: Auto-rendered to VFS, filesystem suffix applies

The two mechanisms coexist. `open_sections` only lists sections that have
tools to collect.

**File Path Convention:**

- Root sections: `/context/{key}.md`
- Nested sections: `/context/{parent-key}.{child-key}.md` (dot-separated path)

## Cloning

Sections expose `clone(**kwargs)` for insertion into new prompts:

```python
cloned = section.clone(session=new_session, bus=new_bus)
```

- Clones are fully decoupled (no shared references)
- Tool-backed sections rewire reducers and handlers to provided instances
- Children are recursively cloned

## Error Handling

### Exception Types

- `PromptValidationError` - Construction failures (missing key, invalid type)
- `PromptRenderError` - Rendering failures (missing params, template errors)
- `OutputParseError` - Structured output validation failures
- `VisibilityExpansionRequired` - Progressive disclosure expansion request

### Validation Rules

- Empty namespace or key: `PromptValidationError`
- Unspecialized section: `PromptValidationError`
- Missing placeholder in dataclass: `PromptValidationError`
- Missing required field at render: `PromptRenderError`
- Template substitution failure: `PromptRenderError`
- Wrong container type in output: `OutputParseError`
- Missing required fields in output: `OutputParseError`

## Usage Example

```python
from dataclasses import dataclass
from weakincentives.prompt import Prompt, MarkdownSection, parse_structured_output

@dataclass
class TaskParams:
    objective: str

@dataclass
class TaskResult:
    summary: str
    steps: list[str]

template = PromptTemplate[TaskResult](
    ns="agents/assistant",
    key="task-planner",
    sections=[
        MarkdownSection[TaskParams](
            title="Task",
            key="task",
            template="Plan the following: ${objective}",
        ),
    ],
)

rendered = Prompt(template).bind(TaskParams(objective="Refactor auth module")).render()

# After adapter evaluation...
result: TaskResult = parse_structured_output(response_text, rendered)
```

## Limitations

- **Dataclass-only inputs**: Non-dataclass params are rejected
- **Limited templating**: Only `Template.substitute` and boolean `enabled`
- **No nested prompts**: Use `children` for reuse, not prompt embedding
- **Single-turn expansion**: Progressive disclosure halts the current turn
- **No partial expansion**: Sections open fully or remain summarized

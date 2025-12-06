# Prompt System Specification

## Purpose

The `Prompt` abstraction centralizes every string template that flows to an LLM
so the codebase has a single, inspectable source for system prompts and per-turn
instructions. This specification covers prompt construction, section composition,
structured output, prompt wrapping for delegation, and progressive disclosure.

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
# Create a Prompt with bound parameters
prompt = Prompt(template, MyParams(field="value"))

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

### Response Format Section

When structured output is declared and `inject_output_instructions=True`
(default), the framework appends a `ResponseFormatSection` keyed as
`response-format`:

```
## Response Format
Return ONLY a single fenced JSON code block. Do not include any text
before or after the block.

The top-level JSON value MUST be an object that matches the fields
of the expected schema. Do not add extra keys.
```

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
    inject_output_instructions=True,  # Default: append format section
    allow_extra_keys=False,           # Default: reject unknown keys
)
```

## Prompt Composition

Delegation composes a new prompt that wraps a parent prompt for subagent
execution.

### Required Layout

1. `# Delegation Summary`
1. `## Response Format` (conditional)
1. `## Parent Prompt (Verbatim)`
1. `## Recap` (optional)

### DelegationPrompt

```python
from weakincentives.prompt.composition import DelegationPrompt, DelegationParams

delegation = DelegationPrompt[ParentOutput, DelegationOutput](
    parent_prompt,
    rendered_parent,
    recap_lines=("Check constraints before planning.",),
)

rendered = delegation.render(
    DelegationParams(
        reason="Investigate filesystem",
        expected_result="Actionable plan",
        may_delegate_further="no",
    ),
    parent=ParentPromptParams(body=rendered_parent.text),  # Optional, defaults to parent text
    recap=RecapParams(...),  # Optional recap section
)
```

### Composition Rules

- Parent prompt is embedded byte-for-byte with markers:
  `<!-- PARENT PROMPT START -->` and `<!-- PARENT PROMPT END -->`
- Tools from parent are inherited, not redeclared
- Response format section appears only when adapter needs fallback instructions
- Truncation is never allowed; abort if size limits prevent embedding

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
prompt = Prompt(template, *params)

while True:
    try:
        rendered = prompt.render(visibility_overrides=visibility_overrides)
        response = adapter.evaluate(prompt, session=session, bus=bus)
        break
    except VisibilityExpansionRequired as e:
        visibility_overrides.update(e.requested_overrides)
```

### Summary Suffix

Summarized sections automatically append:

```
---
[This section is summarized. To view full content, call `open_sections` with key "context".]
```

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

rendered = Prompt(template, TaskParams(objective="Refactor auth module")).render()

# After adapter evaluation...
result: TaskResult = parse_structured_output(response_text, rendered)
```

## Limitations

- **Dataclass-only inputs**: Non-dataclass params are rejected
- **Limited templating**: Only `Template.substitute` and boolean `enabled`
- **No nested prompts**: Use `children` for reuse, not prompt embedding
- **Single-turn expansion**: Progressive disclosure halts the current turn
- **No partial expansion**: Sections open fully or remain summarized

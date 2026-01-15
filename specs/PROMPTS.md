# Prompt System Specification

## Purpose

The `Prompt` abstraction centralizes string templates flowing to LLMs with
resource lifecycle management. Core at `prompt/prompt.py`.

## Principles

- **Type-safety first**: Placeholders map to dataclass fields
- **Strict failures**: Validation/render errors fail loudly
- **Composable markdown**: Hierarchical sections with deterministic headings
- **Resource co-location**: Prompts declare and manage resource lifecycle
- **Minimal templating**: `Template.substitute` plus boolean selectors only
- **Declarative over imperative**: Structure, not logic

## Core Components

### PromptTemplate

At `prompt/prompt.py:94-278`:

| Field | Description |
| --- | --- |
| `ns` | Namespace (required, non-empty) |
| `key` | Identifier (required, non-empty) |
| `name` | Optional display name |
| `sections` | Ordered section tree |
| `resources` | `ResourceRegistry` for dependencies |

Section keys must match: `^[a-z0-9][a-z0-9._-]{0,63}$`

### Prompt

At `prompt/prompt.py:280-562`:

| Method | Description |
| --- | --- |
| `bind(*params, resources=)` | Bind dataclass parameters |
| `render()` | Produce `RenderedPrompt` |
| `resources` property | Access `PromptResources` for lifecycle |

**Note:** `bind()` maintains one instance per dataclass type. Rebinding same type
replaces; providing same type twice in single call rejected.

### Section

Abstract base at `prompt/section.py`:

| Field | Description |
| --- | --- |
| `title` | Display title |
| `key` | Identifier |
| `children` | Child sections |
| `tools` | Tools attached to section |
| `enabled` | Conditional rendering callable |
| `visibility` | FULL or SUMMARY |
| `accepts_overrides` | Whether overridable |

Sections must be specialized: `MarkdownSection[MyParams]`

### MarkdownSection

At `prompt/markdown.py`: Dedents, strips, runs `Template.substitute`.

| Field | Description |
| --- | --- |
| `template` | Template string with `$field` placeholders |
| `summary` | Optional summary for progressive disclosure |

### WorkspaceSection

Provides filesystem access, contributes to prompt resources via `resources()` method.

## Resource Lifecycle

### Collection

Resources collected from (lowest to highest precedence):
1. `PromptTemplate.resources`
2. Section `resources()` methods (depth-first)
3. `bind(resources=...)` at bind time

### Context Manager Protocol

At `prompt/prompt.py` via `PromptResources`:

```python
with prompt.resources:
    fs = prompt.resources.get(Filesystem)
    response = adapter.evaluate(prompt, session=session)
# Resources cleaned up automatically
```

**Key:** `prompt.resources` serves as both context manager and resource accessor.
Accessing outside context raises `RuntimeError`.

### Transactional Tool Execution

Via `runtime/transactions.py`:
- `tool_transaction(session, resource_context, tag)` context manager
- `create_snapshot()` / `restore_snapshot()` for manual control

## Rendering

`Prompt.render()` walks section tree depth-first, producing markdown with
deterministic headings.

### Heading Levels

- Root sections: `##`
- Each depth adds one `#` (depth 1 = `###`)
- Numbered with trailing period: `## 1. Title`, `### 1.1. Subtitle`

### Parameter Lookup

1. Use `default_params` if configured
2. Else use first default for that type
3. Else instantiate with no arguments

Missing required fields raise `PromptRenderError`.

### RenderedPrompt

At `prompt/rendering.py`:

| Property | Description |
| --- | --- |
| `text` | Rendered markdown |
| `tools` | Tool tuple |
| `output_type` | Structured output type |
| `container` | `"object"` or `"array"` |
| `deadline` | Propagated deadline |

## Structured Output

Via generic specialization:

| Declaration | Output |
| --- | --- |
| `PromptTemplate[T]` | JSON object matching dataclass T |
| `PromptTemplate[list[T]]` | JSON array of objects |

Non-dataclass types raise `PromptValidationError`.

### Parsing

`parse_structured_output(output_text, rendered)`:
1. Extract JSON (fenced block preferred, else parse entire message)
2. Validate container type matches declaration
3. Validate dataclass fields, no extra keys (unless allowed)

Failures raise `OutputParseError` with raw response attached.

## Progressive Disclosure

Sections with `visibility=SUMMARY` render abbreviated content until expanded.

### Tools

At `prompt/progressive_disclosure.py`:

| Tool | Purpose | For Sections With |
| --- | --- | --- |
| `open_sections` | Permanently expand (raises `VisibilityExpansionRequired`) | Tools |
| `read_section` | Return content without state change | No tools |

### VisibilityExpansionRequired

Exception at `prompt/__init__.py`:
- `requested_overrides`: Mapping of paths to visibility
- `reason`: Why expansion needed
- `section_keys`: Affected section keys

Adapters catch this, apply overrides to session, retry evaluation.

## Prompt Overrides

Enable prompt iteration without source changes at `prompt/overrides/`.

### Override Targets

| Target | Identifier | Override Type |
| --- | --- | --- |
| Section body | `(path,)` | `SectionOverride` |
| Tool description | `tool_name` | `ToolOverride` |
| Tool param description | `tool_name.param` | `ToolOverride.param_descriptions` |
| Tool example | `tool_name#index` | `ToolExampleOverride` |
| Task example | `path#index` | `TaskExampleOverride` |

**Not overridable:** Tool names, parameter types, section structure, tool availability.

### Descriptor System

Hashes track drift:
- `SectionDescriptor` with `content_hash`
- `ToolDescriptor` with `contract_hash`
- `PromptDescriptor` aggregates all

Hash mismatches indicate stale overrides; filtered on load.

### Store Protocol

`PromptOverridesStore` at `prompt/overrides/store.py`:
- `resolve(descriptor, tag)` - Load, filter stale
- `upsert(descriptor, override)` - Persist
- `seed(prompt, tag)` - Bootstrap from current state

Storage: `.weakincentives/prompts/overrides/{ns}/{key}/{tag}.json`

## Cloning

Sections expose `clone(**kwargs)` for insertion into new prompts. Clones are
fully decoupled; children recursively cloned.

## Error Handling

| Exception | Cause |
| --- | --- |
| `PromptValidationError` | Construction failures (missing key, invalid type) |
| `PromptRenderError` | Missing params, template errors |
| `OutputParseError` | Structured output validation |
| `VisibilityExpansionRequired` | Progressive disclosure expansion |

## Limitations

- **Dataclass-only inputs**: Non-dataclass params rejected
- **Limited templating**: Only `Template.substitute` and boolean `enabled`
- **No nested prompts**: Use `children` for reuse
- **Single-turn expansion**: Progressive disclosure halts current turn
- **Context manager required**: Resources only within `with` block

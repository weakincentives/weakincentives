# Open Sections Tool Specification

## Overview

The `open_sections` tool enables progressive disclosure of prompt content. When a prompt
contains sections rendered with `SUMMARY` visibility, the framework automatically registers
this builtin tool so the model can request expanded views of summarized content. Calling the
tool halts prompt evaluation and signals the caller to retry with the requested visibility
overrides applied.

## Rationale

Large prompts often contain reference material, documentation, or contextual data that is
not always needed for every task. Rendering everything at `FULL` visibility wastes tokens
and can overwhelm the model with irrelevant details. Conversely, omitting content entirely
risks the model lacking critical information when it unexpectedly becomes relevant.

The `SUMMARY` visibility mode provides a middle ground: sections render abbreviated content
that orients the model to what is available without consuming the full token budget. The
`open_sections` tool closes the loop by giving the model agency to expand specific sections
when their details become necessary for the task at hand.

This approach:

- **Reduces prompt size** by default while keeping all content accessible.
- **Preserves model autonomy** by letting the LLM decide when detail is needed.
- **Maintains determinism** because the tool produces a structured signal rather than
  mutating state; the caller controls the retry.

## Scope

- Applies to any `PromptTemplate` containing at least one `Section` with
  `visibility=SectionVisibility.SUMMARY`.
- Covers automatic tool registration, tool schema, handler behavior, and the summary suffix
  convention.
- Does not address multi-turn memory, caching of expanded content, or partial section
  expansion (sections open fully or remain summarized).

## Terminology

- **Summary visibility** – a section rendered with `SectionVisibility.SUMMARY`, displaying
  only its `summary` template text instead of the full body.
- **Open** – transition a section from `SUMMARY` to `FULL` visibility.
- **Visibility override** – a `Mapping[SectionPath, SectionVisibility]` passed to
  `PromptTemplate.render` that overrides the default visibility of specific sections.
- **Progressive disclosure** – the pattern of starting with minimal detail and expanding on
  demand.

## Guiding Principles

- **Opt-in activation**: The tool appears only when the prompt has summarized sections;
  prompts with all-`FULL` visibility do not expose it.
- **Explicit signaling**: The tool does not mutate prompt state. Instead it returns a
  structured result that instructs the caller to retry with specified overrides.
- **Predictable halt**: Invoking `open_sections` always terminates the current evaluation
  turn. The model cannot continue reasoning after calling it within the same turn.
- **Composable with overrides**: Visibility overrides requested by the tool merge with any
  caller-supplied overrides, with tool requests taking precedence.

## Automatic Registration

### Detection

During `PromptTemplate` construction, the framework inspects every registered section. If
at least one section satisfies:

```python
section.visibility == SectionVisibility.SUMMARY
```

the prompt automatically injects the `open_sections` tool into its tool registry. The tool
is registered as a framework-provided builtin with `accepts_overrides=False` to prevent
external optimization from modifying its behavior.

### Injection Point

The tool is appended to the prompt's tool list after all user-defined tools. It does not
belong to any specific section; instead it is owned by the prompt itself and appears in
`RenderedPrompt.tools` whenever the prompt contains summarized sections.

### Conditional Presence

If visibility overrides supplied at render time cause all sections to render at `FULL`
visibility (i.e., no section remains summarized), the tool is excluded from the rendered
tool list for that invocation. This prevents the model from calling a tool that has no
actionable targets.

## Tool Schema

### Parameters

```python
@dataclass(slots=True, frozen=True)
class OpenSectionsParams:
    """Parameters for the open_sections tool."""

    section_keys: tuple[str, ...] = field(
        metadata={
            "description": (
                "Section keys to open. Each key identifies a summarized section. "
                "Use the keys shown in section summaries. Nested sections use "
                "dot notation (e.g., 'parent.child')."
            ),
        },
    )
    reason: str = field(
        metadata={
            "description": (
                "Brief explanation of why these sections need to be expanded. "
                "Helps the caller understand the model's information needs."
            ),
        },
    )
```

- **`section_keys`**: A tuple of section key paths identifying which summarized sections to
  open. Keys use dot notation for nested sections (e.g., `"context.examples"` refers to the
  `examples` child of the `context` section). Only sections currently rendered with
  `SUMMARY` visibility are valid targets.
- **`reason`**: A short explanation (≤256 characters) describing why the model needs the
  expanded content. This aids debugging, logging, and potential caller-side heuristics.

### Result

```python
@dataclass(slots=True, frozen=True)
class OpenSectionsResult:
    """Result payload for the open_sections tool."""

    requested_overrides: dict[tuple[str, ...], str]
    """Mapping of section paths to requested visibility ('full')."""

    def render(self) -> str:
        keys = ", ".join("/".join(p) for p in self.requested_overrides)
        return f"Sections requested for expansion: {keys}. Retry prompt with visibility overrides."
```

- **`requested_overrides`**: A mapping from `SectionPath` tuples to the string `"full"`,
  representing the visibility overrides the caller should apply when retrying.
- The `render()` method produces a human-readable summary for the model's context,
  confirming which sections were requested.

## Handler Behavior

The `open_sections` handler validates inputs and produces a result that signals the caller
to halt and retry:

1. **Parse section keys**: Convert dot-notation strings to `SectionPath` tuples.
2. **Validate targets**: For each requested key, verify that:
   - The section exists in the prompt's registry.
   - The section is currently rendered with `SUMMARY` visibility (accounting for any
     existing visibility overrides).
   - Reject keys pointing to non-existent sections or sections already at `FULL` visibility
     with `ToolValidationError`.
3. **Build override mapping**: Construct `requested_overrides` with each validated path
   mapped to `SectionVisibility.FULL`.
4. **Return halt signal**: Return a `ToolResult` with:
   - `success=True`
   - `value=OpenSectionsResult(requested_overrides=...)`
   - `message` summarizing the requested expansions.

### Halt Semantics

The `open_sections` tool is a **terminal tool**: its invocation ends the current prompt
evaluation turn. Adapters must recognize this tool and:

1. Not execute any subsequent tool calls in the same turn.
2. Return control to the caller with the `OpenSectionsResult` payload.
3. Signal that a retry with visibility overrides is required.

This behavior is analogous to a `yield` or early return—the model's remaining planned
actions are discarded in favor of the caller rerunning with expanded context.

## Caller Responsibilities

When the caller receives an `OpenSectionsResult` from the adapter, it should:

1. **Extract overrides**: Read `requested_overrides` and convert to
   `Mapping[SectionPath, SectionVisibility]`.
2. **Merge with existing overrides**: Combine the tool's requests with any overrides the
   caller was already applying, giving precedence to the tool's requests.
3. **Re-render the prompt**: Call `prompt.render(..., visibility_overrides=merged)`.
4. **Retry evaluation**: Submit the re-rendered prompt to the adapter, continuing the
   conversation with the expanded content now visible.

Callers may implement policies around retry limits, caching of expanded prompts, or
incremental expansion (opening one section at a time vs. batching requests).

## Summary Suffix Convention

To inform the model that summarized sections can be expanded, each section rendered with
`SUMMARY` visibility appends a standardized instruction block to its summary text. This
suffix is injected automatically during rendering and follows this template:

```
---
[This section is summarized. To view full content, call `open_sections` with key "${section_key}".]
```

### Suffix Construction Rules

1. **Separator**: A horizontal rule (`---`) visually separates the summary content from the
   instruction.
2. **Section key**: The dot-notation key path for the section (e.g., `context.examples`).
3. **Instruction text**: A brief, model-directed instruction explaining how to request
   expansion.
4. **Placement**: The suffix is appended after the summary template content but before any
   child content (children of summarized sections are not rendered, so this is effectively
   the final content).

### Customization

Section authors may customize the suffix by providing a `summary_suffix` parameter:

```python
section = MarkdownSection[Params](
    title="Reference Documentation",
    template="...",
    summary="Overview of available APIs and their purposes.",
    summary_suffix=(
        "Call `open_sections` with key '${section_key}' to see detailed API signatures "
        "and usage examples."
    ),
    key="reference",
    visibility=SectionVisibility.SUMMARY,
)
```

The `${section_key}` placeholder is substituted with the section's dot-notation path during
rendering. If `summary_suffix` is not provided, the default template is used.

### Nested Sections

When a parent section is summarized, its children are not rendered. The parent's summary
suffix should indicate that expanding it reveals child content:

```
---
[This section is summarized. Call `open_sections` with key "context" to view full content
including subsections: examples, constraints, history.]
```

The framework automatically appends child section keys to the suffix when the parent has
`SUMMARY` visibility and children exist.

## Integration with Visibility Overrides

The `open_sections` tool works in concert with the existing visibility override system
documented in the rendering pipeline:

```python
rendered = prompt.render(
    *params,
    visibility_overrides={
        ("context",): SectionVisibility.FULL,
        ("reference", "advanced"): SectionVisibility.FULL,
    },
)
```

### Override Precedence

1. **Default visibility**: The section's declared `visibility` attribute.
2. **Caller-supplied overrides**: Visibility overrides passed to `render()`.
3. **Tool-requested overrides**: Overrides requested via `open_sections` on retry.

Tool-requested overrides take precedence because they represent the model's explicit
information needs discovered during evaluation.

### Incremental Expansion

Callers may implement incremental expansion strategies:

- **Eager**: Open all requested sections immediately.
- **Conservative**: Open one section at a time, re-evaluating after each expansion.
- **Batched**: Accumulate requests across turns before expanding.

The tool imposes no policy; callers choose based on latency, cost, and context window
constraints.

## Example Flow

```python
from dataclasses import dataclass
from weakincentives.prompt import (
    MarkdownSection,
    PromptTemplate,
    SectionVisibility,
)

@dataclass
class TaskParams:
    objective: str

@dataclass
class ContextParams:
    project_name: str

# Define a prompt with summarized context
prompt = PromptTemplate(
    ns="agents/assistant",
    key="task-executor",
    name="task_executor",
    sections=[
        MarkdownSection[TaskParams](
            title="Task",
            template="Complete the following: ${objective}",
            key="task",
        ),
        MarkdownSection[ContextParams](
            title="Project Context",
            template="""
            Detailed documentation for ${project_name}:
            - Architecture overview
            - API reference
            - Code conventions
            - Historical decisions
            """,
            summary="Documentation for ${project_name} is available.",
            key="context",
            visibility=SectionVisibility.SUMMARY,
        ),
    ],
)

# Initial render with summarized context
rendered = prompt.render(
    TaskParams(objective="Refactor the authentication module"),
    ContextParams(project_name="WeakIncentives"),
)

# The model sees:
# ## 1 Task
# Complete the following: Refactor the authentication module
#
# ## 2 Project Context
# Documentation for WeakIncentives is available.
# ---
# [This section is summarized. To view full content, call `open_sections` with key "context".]

# Model calls: open_sections(section_keys=("context",), reason="Need architecture details for refactoring")

# Caller receives OpenSectionsResult and retries:
rendered = prompt.render(
    TaskParams(objective="Refactor the authentication module"),
    ContextParams(project_name="WeakIncentives"),
    visibility_overrides={("context",): SectionVisibility.FULL},
)

# The model now sees full context and can proceed with the task.
```

## Telemetry

The `open_sections` tool emits standard `ToolInvoked` events as defined in `specs/EVENTS.md`.
Additional context captured:

- `requested_sections`: List of section keys requested for expansion.
- `reason`: The model's stated reason for expansion.
- `current_visibility_state`: Snapshot of which sections were summarized vs. full at
  invocation time.

Callers may use this telemetry to:

- Analyze which sections are frequently expanded (candidates for `FULL` by default).
- Detect patterns where summarization causes repeated retries (optimization opportunities).
- Audit model behavior for debugging and compliance.

## Limitations and Caveats

- **Single-turn halt**: The tool always terminates the current turn. Models cannot
  speculatively request expansions while continuing other work.
- **No partial expansion**: Sections open fully or remain summarized. There is no mechanism
  to request "more detail" incrementally within a section.
- **Caller cooperation required**: The tool produces a signal; it cannot force the caller
  to retry. Callers that ignore `OpenSectionsResult` will leave the model without requested
  context.
- **Token budget awareness**: The tool does not estimate token impact. Callers should
  implement safeguards if expanding sections risks exceeding context limits.
- **No cross-prompt expansion**: The tool operates within a single prompt's section tree.
  It cannot request content from other prompts or external sources.

## Testing Checklist

- Tool registration tests verifying automatic injection when `SUMMARY` sections exist.
- Tool exclusion tests confirming the tool is absent when all sections are `FULL`.
- Handler validation tests for invalid section keys and already-expanded sections.
- Suffix rendering tests checking default and custom suffix templates.
- Integration tests demonstrating the full flow: initial render → tool call → retry with
  overrides → expanded content visible.
- Telemetry tests asserting `ToolInvoked` events capture expansion metadata.

## Documentation Tasks

- Update `specs/PROMPTS.md` to reference this specification for visibility-related tooling.
- Add examples to `llms.md` demonstrating progressive disclosure patterns.
- Create a tutorial showing best practices for structuring prompts with summarized sections.

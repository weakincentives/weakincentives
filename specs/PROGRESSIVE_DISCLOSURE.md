# Progressive Disclosure Specification

## Overview

The `open_sections` tool enables progressive disclosure of prompt content. When a prompt
contains sections rendered with `SUMMARY` visibility, the framework automatically registers
this builtin tool so the model can request expanded views of summarized content. Calling the
tool raises a `VisibilityExpansionRequired` exception that halts prompt evaluation and
carries the requested visibility overrides for the caller to apply on retry.

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
- **Maintains determinism** because the exception carries structured data rather than
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
- **Exception-based signaling**: The tool raises `VisibilityExpansionRequired` rather than
  returning a result. This provides explicit control flow that cannot be ignored and
  naturally halts evaluation without requiring adapter-level detection.
- **Predictable halt**: Invoking `open_sections` always terminates the current evaluation
  turn via exception. The model cannot continue reasoning after calling it within the same
  turn.
- **Composable with overrides**: Visibility overrides carried by the exception merge with
  any caller-supplied overrides, with tool requests taking precedence.

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

### Exception

The tool raises `VisibilityExpansionRequired` to signal that evaluation should halt and
retry with expanded sections. This exception is part of the `weakincentives.prompt.errors`
module and inherits from `PromptError`:

```python
@dataclass(slots=True)
class VisibilityExpansionRequired(PromptError):
    """Raised when the model requests expansion of summarized sections.

    Callers should catch this exception, extract the visibility overrides,
    and retry prompt evaluation with the requested sections expanded.
    """

    requested_overrides: Mapping[tuple[str, ...], SectionVisibility]
    """Mapping of section paths to requested visibility (always FULL)."""

    reason: str
    """The model's stated reason for needing the expanded content."""

    section_keys: tuple[str, ...]
    """Original section key strings as provided by the model."""

    def __str__(self) -> str:
        keys = ", ".join(".".join(p) for p in self.requested_overrides)
        return f"Visibility expansion required for sections: {keys}. Reason: {self.reason}"
```

- **`requested_overrides`**: A mapping from `SectionPath` tuples to `SectionVisibility.FULL`,
  ready to be passed directly to `prompt.render(..., visibility_overrides=...)`.
- **`reason`**: The model's explanation for why expansion is needed, preserved for logging
  and debugging.
- **`section_keys`**: The original dot-notation keys as provided by the model, useful for
  telemetry and error messages.

## Handler Behavior

The `open_sections` handler validates inputs and raises `VisibilityExpansionRequired` to
halt evaluation:

1. **Parse section keys**: Convert dot-notation strings to `SectionPath` tuples.
1. **Validate targets**: For each requested key, verify that:
   - The section exists in the prompt's registry.
   - The section is currently rendered with `SUMMARY` visibility (accounting for any
     existing visibility overrides).
   - Reject keys pointing to non-existent sections or sections already at `FULL` visibility
     with `ToolValidationError`.
1. **Build override mapping**: Construct `requested_overrides` with each validated path
   mapped to `SectionVisibility.FULL`.
1. **Raise exception**: Raise `VisibilityExpansionRequired` with the override mapping,
   reason, and original section keys.

### Exception Propagation

The `VisibilityExpansionRequired` exception propagates through the adapter's tool dispatch
layer without being caught or converted to a `ToolResult`. This ensures:

1. **Immediate halt**: No subsequent tool calls execute in the same turn.
1. **Clean stack unwinding**: The exception bubbles up to the caller's evaluation loop.
1. **Explicit handling required**: Callers must catch `VisibilityExpansionRequired` or let
   it propagate as an error—there is no silent fallback.

Adapters MUST NOT wrap this exception in `PromptEvaluationError` or convert it to a failed
tool result. The exception is a deliberate control flow mechanism, not an error condition.

```python
# Adapter tool dispatch (pseudocode)
try:
    result = tool.handler(params, context=context)
except VisibilityExpansionRequired:
    raise  # Let it propagate to the caller
except Exception as e:
    # Other exceptions become failed tool results
    result = ToolResult(success=False, message=str(e), value=None)
```

## Caller Responsibilities

Callers must wrap adapter evaluation in a try/except block to handle
`VisibilityExpansionRequired`:

```python
from weakincentives.prompt.errors import VisibilityExpansionRequired

visibility_overrides: dict[tuple[str, ...], SectionVisibility] = {}

while True:
    try:
        rendered = prompt.render(*params, visibility_overrides=visibility_overrides)
        response = adapter.evaluate(rendered, session=session, bus=bus)
        break  # Success
    except VisibilityExpansionRequired as e:
        # Merge requested overrides with existing ones
        visibility_overrides.update(e.requested_overrides)
        # Log the expansion request
        logger.info(f"Expanding sections: {e.section_keys}, reason: {e.reason}")
        # Continue loop to retry with expanded visibility
```

### Handling Steps

1. **Catch the exception**: Handle `VisibilityExpansionRequired` explicitly.
1. **Extract overrides**: Access `e.requested_overrides` directly—it's already in the
   correct format for `prompt.render()`.
1. **Merge with existing overrides**: Combine with any caller-maintained overrides, giving
   precedence to the exception's requests.
1. **Re-render and retry**: Call `prompt.render()` with updated overrides and re-evaluate.

### Policies

Callers may implement policies around:

- **Retry limits**: Cap the number of expansion cycles to prevent infinite loops.
- **Expansion budgets**: Limit total tokens or sections that can be expanded.

## Summary Rendering

When a section is rendered with `SUMMARY` visibility, the framework automatically appends
an instruction block to inform the model how to request expansion. This suffix is injected
during rendering without requiring any configuration from section authors.

### Rendered Format

```
---
[This section is summarized. To view full content, call `open_sections` with key "${section_key}".]
```

- **Separator**: A horizontal rule (`---`) visually separates summary content from the
  instruction.
- **Section key**: The dot-notation key path for the section (e.g., `context.examples`).
- **Placement**: Appended after the summary template content. Since children of summarized
  sections are not rendered, this is the final content for the section.

### Nested Sections

When a parent section is summarized and has children, the suffix enumerates the child
sections that would become visible upon expansion:

```
---
[This section is summarized. Call `open_sections` with key "context" to view full content
including subsections: examples, constraints, history.]
```

The framework automatically detects and lists child section keys.

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
1. **Caller-supplied overrides**: Visibility overrides passed to `render()`.
1. **Tool-requested overrides**: Overrides requested via `open_sections` on retry.

Tool-requested overrides take precedence because they represent the model's explicit
information needs discovered during evaluation.

All requested sections are expanded immediately on retry (eager expansion). The framework
does not support partial or incremental expansion strategies.

## Example Flow

```python
from dataclasses import dataclass
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.prompt import (
    MarkdownSection,
    PromptTemplate,
    SectionVisibility,
)
from weakincentives.prompt.errors import VisibilityExpansionRequired
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session

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
# The model sees:
# ## 1 Task
# Complete the following: Refactor the authentication module
#
# ## 2 Project Context
# Documentation for WeakIncentives is available.
# ---
# [This section is summarized. To view full content, call `open_sections` with key "context".]

# Evaluation loop with exception handling
bus = InProcessEventBus()
session = Session(bus=bus)
adapter = OpenAIAdapter(model="gpt-4o")

params = (
    TaskParams(objective="Refactor the authentication module"),
    ContextParams(project_name="WeakIncentives"),
)
visibility_overrides: dict[tuple[str, ...], SectionVisibility] = {}

while True:
    try:
        rendered = prompt.render(*params, visibility_overrides=visibility_overrides)
        response = adapter.evaluate(rendered, session=session, bus=bus)
        # Success - model completed the task
        break
    except VisibilityExpansionRequired as e:
        # Model called: open_sections(section_keys=("context",), reason="Need architecture details")
        print(f"Expanding: {e.section_keys}, reason: {e.reason}")
        visibility_overrides.update(e.requested_overrides)
        # Loop continues with expanded visibility

# After expansion, the model sees full context and can proceed with the task.
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

- **Single-turn halt**: The tool always terminates the current turn via exception. Models
  cannot speculatively request expansions while continuing other work.
- **No partial expansion**: Sections open fully or remain summarized. There is no mechanism
  to request "more detail" incrementally within a section.
- **Explicit handling required**: The exception propagates unless caught. Callers must
  implement a retry loop or let the exception surface as an error—there is no silent
  fallback behavior.
- **Token budget awareness**: The tool does not estimate token impact. Callers should
  implement safeguards if expanding sections risks exceeding context limits.
- **No cross-prompt expansion**: The tool operates within a single prompt's section tree.
  It cannot request content from other prompts or external sources.
- **Session state preserved**: The exception does not roll back session state from the
  interrupted evaluation. Callers may need to snapshot session state before evaluation if
  rollback semantics are desired.

## Testing Checklist

- Tool registration tests verifying automatic injection when `SUMMARY` sections exist.
- Tool exclusion tests confirming the tool is absent when all sections are `FULL`.
- Handler validation tests for invalid section keys and already-expanded sections.
- Exception tests verifying `VisibilityExpansionRequired` carries correct override mappings.
- Exception propagation tests confirming adapters do not catch or wrap the exception.
- Suffix rendering tests checking the automatic instruction block is appended correctly.
- Nested section tests verifying child keys are listed in the suffix.
- Integration tests demonstrating the full flow: initial render → tool call → exception →
  retry with overrides → expanded content visible.
- Retry loop tests validating typical caller patterns with exception handling.
- Telemetry tests asserting `ToolInvoked` events capture expansion metadata before the
  exception is raised.

## Documentation Tasks

- Update `specs/PROMPTS.md` to reference this specification for visibility-related tooling.
- Add examples to `llms.md` demonstrating progressive disclosure patterns.
- Create a tutorial showing best practices for structuring prompts with summarized sections.

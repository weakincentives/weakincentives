# Progressive Disclosure

*Canonical spec: [specs/PROMPTS.md](../specs/PROMPTS.md) (Progressive Disclosure
section)*

Long prompts are expensive. Progressive disclosure is WINK's first-class
solution: sections can render as summaries by default, and the model can request
expansion when it needs details.

## The Problem

When you're building agents that work with codebases, documentation, or any
substantial context, you face a tension:

- **Full context is expensive**: Token costs add up, and models can get
  distracted by irrelevant details
- **Missing context causes failures**: Without the right information, the model
  makes poor decisions

Progressive disclosure resolves this tension by giving the model control over
what it sees.

## SectionVisibility: FULL vs SUMMARY

A section can have:

- `template`: full content
- `summary`: short content
- `visibility`: constant or callable

If visibility is `SUMMARY` and `summary` is present, WINK renders the summary
instead of the full template.

```python nocheck
from weakincentives.prompt import MarkdownSection, SectionVisibility

section = MarkdownSection(
    title="Reference",
    key="reference",
    template="Very long reference documentation...",
    summary="Reference documentation is available.",
    visibility=SectionVisibility.SUMMARY,
)
```

The model sees "Reference documentation is available." instead of the full text.
This saves tokens while letting the model know the information exists.

## open_sections and read_section

When summarized sections exist, WINK injects builtin tools:

- `open_sections(section_keys, reason)` → raises `VisibilityExpansionRequired`
- `read_section(section_key)` → returns full rendered markdown for that section

**open_sections** permanently expands sections and is used for sections that
register tools. When a section has tools attached, the model needs those tools
to become available—not just the content. `AgentLoop` handles this automatically
by setting overrides and retrying. The model asks to expand a section, AgentLoop
applies the expansion, and evaluation continues with the full content visible
and the section's tools now registered.

**read_section** returns content without changing visibility and is used
exclusively for sections that have no tools attached. The section remains
summarized in subsequent turns. Use this for reference material that the model
only needs temporarily—documentation, examples, or context that doesn't unlock
new capabilities.

## How AgentLoop Handles Expansion

When the model calls `open_sections`:

1. The tool handler raises `VisibilityExpansionRequired`
1. AgentLoop catches the exception
1. AgentLoop applies visibility overrides to the session
1. AgentLoop re-renders the prompt with expanded sections
1. Evaluation continues with the new prompt

You don't need to handle this yourself—AgentLoop does it automatically.

## Visibility Overrides in Session State

Visibility overrides live in the `VisibilityOverrides` session slice and are
applied at render time.

```python nocheck
from weakincentives.prompt import SectionVisibility
from weakincentives.runtime.session import SetVisibilityOverride, VisibilityOverrides

session.dispatch(
    SetVisibilityOverride(path=("reference",), visibility=SectionVisibility.FULL)
)
```

This is what happens under the hood when the model calls `open_sections`.

## When to Use Progressive Disclosure

Use progressive disclosure when:

- You have reference material the model might need, but probably won't
- You want to keep initial prompts lean for faster iteration
- You're working with large codebases where showing everything upfront is
  impractical

The workspace digest in the code review example uses progressive disclosure: the
model sees a summary of the repo structure, and can expand to see full file
contents when needed.

## Example Pattern

```python nocheck
from weakincentives.prompt import PromptTemplate, MarkdownSection, SectionVisibility

template = PromptTemplate(
    ns="docs",
    key="research",
    sections=(
        MarkdownSection(
            title="Instructions",
            key="instructions",
            template="Answer the question using the available sources.",
        ),
        MarkdownSection(
            title="Source: Python Docs",
            key="python-docs",
            template="Full Python documentation text here...",
            summary="Python documentation is available for reference.",
            visibility=SectionVisibility.SUMMARY,
        ),
        MarkdownSection(
            title="Source: API Reference",
            key="api-ref",
            template="Full API reference text here...",
            summary="API reference documentation is available.",
            visibility=SectionVisibility.SUMMARY,
        ),
        MarkdownSection(
            title="Question",
            key="question",
            template="${question}",
        ),
    ),
)
```

The model sees the question and knows documentation is available. It can call
`open_sections(["python-docs"])` if it needs the full Python docs, or
`read_section("api-ref")` to peek at the API reference without permanently
expanding it.

## Next Steps

- [Prompt Overrides](prompt-overrides.md): Iterate on prompts without code
  changes
- [Claude Agent SDK](claude-agent-sdk.md): See progressive disclosure with
  WorkspaceDigestSection
- [Orchestration](orchestration.md): Learn how AgentLoop handles expansion

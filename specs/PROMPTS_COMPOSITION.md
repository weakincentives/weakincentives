# Prompt Wrapping Specification

## Purpose

Delegating agents compose a new prompt when they hand work to a subagent. The only goal of composition is to wrap the
parent prompt with a thin delegation header so the subagent inherits the entire context without alteration. This
specification keeps the wrapper minimal and prescriptive so delegation stays transparent and lossless.

## Core Principle

> **Always deliver the full parent prompt exactly as it was rendered.**
>
> - No sections are removed, reordered, or reflowed.
> - All policies, tool descriptions, transcripts, and variable substitutions from the parent remain intact.
> - The parent prompt is treated as immutable input; composition stops immediately if it cannot be embedded verbatim.

## Required Layout

A subagent prompt MUST follow the structure below:

1. `# Delegation Summary`
1. `## Response Format` (only when fallback instructions are required)
1. `## Parent Prompt (Verbatim)`

Only these sections may appear. Skip the `Response Format` block when fallback instructions are unnecessary, and never add
extra sections unless a future revision explicitly introduces them.

### Delegation Summary

Provide lightweight context to the subagent before showing the parent prompt. Include exactly the following fields:

- **Reason** – a single sentence explaining why the subagent exists.
- **Expected result** – one sentence describing the deliverable.
- **May delegate further?** – `yes` or `no`.

Each field should appear as a bullet item in the order listed above. Do not include extra prose, tables, or subsections.

### Response Format (Conditional)

Most adapters natively request structured outputs. When that support exists, omit this section entirely—the parent prompt
already controls formatting. Only insert the section when the active adapter cannot express the structure through native
API features and the parent expects a schema.

When present, the section MUST:

- Use the default fallback instructions produced by the core renderer. Reproduce the following body without modifications so
  the wrapper matches the behavior of `ResponseFormatSection`:

  ```
  ## Response Format
  Return ONLY a single fenced JSON code block. Do not include any text before or after the block.

  The top-level JSON value MUST be ${article} ${container} that matches the fields of the expected schema${extra_clause}
  ```

- Substitute the `${article}`, `${container}`, and `${extra_clause}` placeholders exactly as the renderer would (for example,
  "an object" with any additional schema notes in `${extra_clause}`).

- Include any field-level descriptions or validation rules that the parent prompt provided so the schema remains clear.

If the parent did not require structured output, do not add this section; the subagent will default to the inherited free-form
instructions.

### Parent Prompt (Verbatim)

Embed the parent prompt exactly as rendered at delegation time. The wrapper MUST:

- Surround the content with the markers `<!-- PARENT PROMPT START -->` and `<!-- PARENT PROMPT END -->` on their own lines.
- Copy the original markdown byte-for-byte; no formatting, whitespace, or heading changes are allowed.
- Abort delegation if size limits would force truncation. There is no fallback path that drops content.

## Tool and Policy Inheritance

All tools, policies, and constraints described in the parent prompt automatically apply to the subagent. The wrapper MUST
NOT restate or reinterpret them. Any future need to override tools or policies requires an update to this specification.

## Reporting Guidance

The `Response Format` section is the only place the wrapper adds reporting guidance. Beyond that summary, the subagent
responds using the instructions already present in the inherited parent prompt. If the parent needs bespoke reporting
requirements, they must be authored in the parent prompt before delegation and mirrored into the `Response Format`
section when composing the wrapper.

## Reference Code Examples

The snippets below demonstrate how to build delegation wrappers with the public prompt primitives. Both examples assume
the parent prompt has already been rendered so its text and structured output metadata can be reused.

### Minimal wrapper when the adapter supports structured outputs

The helper below captures the rendered parent prompt, then instantiates a new prompt that targets
its own structured output type via the `delegation_output_type` parameter.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, TypeVar

from weakincentives.prompt import MarkdownSection, Prompt, Section

ParentOutputT = TypeVar("ParentOutputT")
DelegationOutputT = TypeVar("DelegationOutputT")


@dataclass
class DelegationSummaryParams:
    reason: str
    expected_result: str
    may_delegate_further: str


@dataclass
class DelegationPlan:
    summary: str
    steps: list[str]


@dataclass
class ParentPromptParams:
    body: str


class ParentPromptSection(Section[ParentPromptParams]):
    def __init__(self, *, tools: Sequence[object] | None = None) -> None:
        super().__init__(
            title="Parent Prompt (Verbatim)",
            key="parent-prompt",
            tools=tools,
        )

    def render(self, params: ParentPromptParams, depth: int) -> str:
        heading = "#" * (depth + 2)
        return "\n".join(
            (
                f"{heading} Parent Prompt (Verbatim)",
                "",
                "<!-- PARENT PROMPT START -->",
                params.body,
                "<!-- PARENT PROMPT END -->",
            )
        )


def wrap_prompt_for_delegation(
    parent_prompt: Prompt[ParentOutputT],
    rendered_parent: Any,
    *,
    delegation_output_type: type[DelegationOutputT],
) -> Prompt[DelegationOutputT]:
    summary_section = MarkdownSection[DelegationSummaryParams](
        title="Delegation Summary",
        key="delegation-summary",
        template=(
            "- **Reason** – ${reason}\n"
            "- **Expected result** – ${expected_result}\n"
            "- **May delegate further?** – ${may_delegate_further}"
        ),
    )
    parent_section = ParentPromptSection(tools=rendered_parent.tools)
    prompt_cls: type[Prompt[DelegationOutputT]] = Prompt[delegation_output_type]
    return prompt_cls(
        ns=f"{parent_prompt.ns}.delegation",
        key=f"{parent_prompt.key}-wrapper",
        sections=(summary_section, parent_section),
        inject_output_instructions=False,
        allow_extra_keys=bool(rendered_parent.allow_extra_keys),
    )


# Usage
parent_render = parent_prompt.render(...)
delegation_prompt = wrap_prompt_for_delegation(
    parent_prompt,
    parent_render,
    delegation_output_type=DelegationPlan,
)
rendered_delegation = delegation_prompt.render(
    DelegationSummaryParams(
        reason="Specialize on the filesystem investigation",
        expected_result="Actionable plan for the next commit",
        may_delegate_further="no",
    ),
    ParentPromptParams(body=parent_render.text),
)
```

### Wrapper with fallback response format instructions

When the adapter lacks native structured output support, mirror the parent prompt's schema with a
`ResponseFormatSection` before embedding the parent prompt text. The snippet below continues inside
`wrap_prompt_for_delegation` from the previous example, reusing the same imports and type variables.

```python
from typing import Any

from weakincentives.prompt.response_format import (
    ResponseFormatParams,
    ResponseFormatSection,
)


def build_fallback_response_section(
    rendered_parent: Any,
) -> ResponseFormatSection | None:
    container = rendered_parent.container
    if container is None:
        return None  # Parent prompt did not request structured output.

    article = "an" if container.startswith(tuple("aeiou")) else "a"
    extra_clause = (
        "."
        if rendered_parent.allow_extra_keys
        else ". Do not add extra keys."
    )

    return ResponseFormatSection(
        params=ResponseFormatParams(
            article=article,
            container=container,
            extra_clause=extra_clause,
        )
    )


# Continuing from the earlier wrapper to inject fallback instructions when needed.
response_section = build_fallback_response_section(parent_render)
sections = [summary_section]
if response_section is not None:
    sections.append(response_section)
sections.append(parent_section)
prompt_cls: type[Prompt[DelegationOutputT]] = Prompt[delegation_output_type]
delegation_prompt = prompt_cls(
    ns=f"{parent_prompt.ns}.delegation",
    key=f"{parent_prompt.key}-wrapper",
    sections=tuple(sections),
    inject_output_instructions=False,
    allow_extra_keys=bool(parent_render.allow_extra_keys),
)
```

These examples keep the wrapper logic declarative, reuse the parent's tool catalog, and mirror the structured output
schema whenever the adapter needs explicit JSON instructions. The resulting rendered prompt complies with the required
layout, embeds the parent prompt verbatim, and stays agnostic to additional future sections unless the specification
changes.

## Failure Handling

Composition can fail only when the parent prompt cannot be embedded verbatim. In that case the delegating agent aborts
the delegation attempt, records the error, and returns the failure to the caller.

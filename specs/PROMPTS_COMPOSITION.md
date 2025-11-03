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
1. `## Response Format`
1. `## Parent Prompt (Verbatim)`

Only these sections are required. Additional sections are forbidden until a future revision explicitly introduces them.

### Delegation Summary

Provide lightweight context to the subagent before showing the parent prompt. Include exactly the following fields:

- **Delegation id** – deterministic identifier chosen by the parent.
- **Reason** – a single sentence explaining why the subagent exists.
- **Expected result** – one sentence describing the deliverable.
- **May delegate further?** – `yes` or `no`.

Each field should appear as a bullet item in the order listed above. Do not include extra prose, tables, or subsections.

### Response Format

Describe the exact shape of the subagent's reply. The section MUST:

- Declare the structured schema the parent expects (for example JSON keys, markdown tables, or ordered lists).
- Reproduce any field-level descriptions or validation rules the parent requires.
- Note whether the subagent should include explanatory prose outside the structured output.

When the parent prompt already defines the response contract, copy it verbatim into this section so the subagent sees it
without scanning the entire parent prompt. If the parent did not provide a format, state `Follow free-form response from parent prompt.` so the subagent knows no structured output is required.

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

## Failure Handling

Composition can fail only when the parent prompt cannot be embedded verbatim. In that case the delegating agent aborts
the delegation attempt, records the error alongside the `delegation id`, and returns the failure to the caller.

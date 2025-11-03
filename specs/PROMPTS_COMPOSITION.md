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

- **Delegation id** – deterministic identifier chosen by the parent.
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

## Failure Handling

Composition can fail only when the parent prompt cannot be embedded verbatim. In that case the delegating agent aborts
the delegation attempt, records the error alongside the `delegation id`, and returns the failure to the caller.

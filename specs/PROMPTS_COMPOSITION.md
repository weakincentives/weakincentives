# Prompt Composition Specification

## Purpose

Background agents can delegate portions of work to specialized subagents. Each delegation constructs a new prompt that
wraps the parent conversation so the subagent retains the full problem statement, shared tools, and execution
constraints. This document defines how a **subagent prompt** must compose the **parent prompt** plus any delegation
payload so that the subagent sees a coherent, traceable view of the work to perform.

## Terminology

- **Parent prompt** – the rendered markdown (system + instructions) held by the delegating agent immediately before the
  delegation step. The parent prompt already encapsulates global policies, task framing, known tool affordances, and the
  conversational transcript up to this point.
- **Delegation payload** – the additional structured context collected by the parent when deciding to invoke a subagent
  (for example, a task summary, explicit deliverables, or scoped tool overrides).
- **Subagent prompt** – the rendered markdown passed to the delegated agent. It must contain an auditable copy of the
  parent prompt, the delegation payload, and any extra guidance specific to the subagent's role.

## Composition Goals

1. **Preserve fidelity** – the subagent must receive the exact context the parent relied upon. Information may be
   reorganized but MUST NOT be redacted unless the delegation configuration explicitly removes sections.
2. **Clarify authority** – the subagent needs clear instructions on which policies are inherited, which can be relaxed,
   and which new expectations apply.
3. **Trace tooling** – tools available to the parent remain visible to the subagent unless the parent has explicitly
   revoked access. Tool metadata must explain the provenance so tooling decisions remain debuggable.
4. **Enable nested composition** – subagents must be capable of further delegation using the same composition rules.

## Structural Requirements

The subagent prompt is organized into a deterministic hierarchy of sections so that downstream logging, diffing, and
versioning can reason about the document. The top-level structure MUST be:

1. `# Delegation Overview`
2. `## Parent Prompt` (with the original markdown nested verbatim)
3. `## Delegation Details`
4. `## Subagent Instructions`
5. `## Tooling Context`
6. `## Reporting Requirements`

Each section may contain subsections, but the headings above must exist and appear in the listed order.

### Delegation Overview

Summarize why the subagent was created. Include:

- The parent agent's identifier and prompt key.
- A one sentence reason for delegation.
- The expected completion state (for example "produce a patch", "return research summary").
- Whether the subagent is allowed to perform further delegations.

### Parent Prompt

Embed the parent prompt **exactly as rendered** at delegation time. Implementation MUST:

- Preserve markdown fidelity (no reformatting, reflowing, or stripping of headings).
- Annotate the start and end with HTML comments (`<!-- PARENT PROMPT START -->` / `<!-- PARENT PROMPT END -->`) to make
  automated extraction trivial.
- When the parent prompt exceeds size limits, the delegating agent MUST abort delegation; truncation is forbidden.

### Delegation Details

This section captures structured data that accompanies the delegation call. Recommended layout:

- `### Task Summary` – human readable overview authored by the delegating agent.
- `### Required Outputs` – bullet list of concrete deliverables.
- `### Scope Constraints` – explicit inclusions/exclusions; default to "inherit parent scope" when empty.
- `### Additional Context` – optional subsections (logs, research, files). Each subsection SHOULD include the source and
  timestamp.

### Subagent Instructions

The parent may add extra guidance unique to this subagent. Rules:

- Start with `### Role` describing the delegated persona or capability.
- Follow with `### Execution Principles` using ordered lists for prioritized rules.
- Clarify escalation pathways: when to stop, what blockers to surface, and how to return partial progress.
- Reference inherited policies with links or SectionPaths when available.

### Tooling Context

Document tool availability so the subagent understands what actions remain valid:

- Reproduce the parent's active tool list, including names, identifiers, success semantics, and cost hints.
- For each tool, specify whether access is `inherited`, `restricted`, or `revoked`. Restrictions MUST explain the reason
  and expected alternative workflows.
- Append any delegation-specific tools after the inherited list using the same schema.
- Include environment metadata (working directory, auth scopes, rate limits) when relevant.

### Reporting Requirements

Define how the subagent must respond upon completion:

- Expected response format (structured output schema, markdown headings, etc.).
- Required attachments (diffs, logs, screenshots). Indicate upload mechanisms when applicable.
- Deadlines or keep-alive cadence if the parent requires status updates.
- Guidelines for citing sources or tool output in the final report.

## Context Propagation Rules

1. **Immutability** – parent prompt content is immutable in the subagent prompt. Any mutation must happen via explicit
   delegation payloads (for example, adding a new section under `Delegation Details`).
2. **Selective Redaction** – if confidentiality rules require omission, the parent must supply a structured redaction
   summary noting which sections were withheld and why. The absence of such a summary implies full inheritance.
3. **Tool Overrides** – when the parent modifies tool availability, it must update both the tool list and any related
   policies in `Subagent Instructions` so the subagent understands behavioral changes.
4. **Parameter Threading** – parameters or placeholders present in the parent prompt (for example `${repo_root}`) must be
   preserved with their concrete values. Placeholders cannot be reintroduced in the subagent prompt.

## Logging & Auditing

- Every composed subagent prompt must capture a deterministic `delegation_id` that ties together the parent prompt,
  payload, and tooling snapshot.
- The composition function should emit structured logs referencing the SectionPaths of inherited content to aid in
  debugging nested delegations.
- Storage systems must retain both the parent and subagent prompts so audits can reconstruct delegation chains.

## Failure Handling

If composition fails (missing parent prompt, invalid tooling metadata, oversized payload):

1. Abort the delegation request.
2. Return a structured error to the parent containing: reason, offending field, and remediation hints.
3. Log the failure with the `delegation_id` and parent prompt key.

## Future Considerations

- **Localization** – consider how to represent multilingual parent prompts while preserving meaning in subagent
  translations.
- **Differential Diffs** – explore storing the parent prompt separately and referencing it by hash to reduce payload size
  while still enabling full reconstruction.
- **Tool Capability Negotiation** – allow subagents to request additional tools with a protocol for parent approval.


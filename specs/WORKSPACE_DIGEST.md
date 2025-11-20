# Workspace Digest Specification

## Overview

`WorkspaceDigest` introduces a reusable prompt section that captures a
task-agnostic description of the current workspace. The digest provides a
condensed inventory of files, configuration, and notable behaviors so downstream
prompts can reason about the environment without re-running exploratory steps
every turn. The section should surface a single source of truth regardless of
which adapter executes the prompt by preferring live session state while
supporting prompt overrides as a fallback.

## Goals

- Reuse a consistent workspace summary across prompts instead of repeating file
  system exploration.
- Prefer real-time session data when available while still honoring persisted
  overrides for reproducibility.
- Keep the section task-agnostic so it remains valid across unrelated user
  requests within the same workspace.

## WorkspaceDigest Section

- Implement `WorkspaceDigest` as a first-class `Section` keyed to
  `workspace-digest` (or another caller-chosen key that satisfies the section
  key rules in `specs/PROMPTS.md`).
- The section emits a markdown heading plus the digest body. The body MUST be
  sourced in this order:
  1. **Session snapshot** – read the latest workspace digest payload recorded in
     the active `Session`. Callers SHOULD register a reducer slice dedicated to
     workspace digests so the section can pull the newest tuple entry without
     performing I/O during render time.
  2. **Override fallback** – when no digest exists in the session, resolve a
     template override for the section from the active overrides store (see
     `specs/PROMPT_OVERRIDES.md`). Overrides must be applied exactly as stored,
     including any escaped template tokens.
  3. **Empty state** – if neither source provides content, log a warning to the
     caller and render an explicit placeholder that invites the model to explore
     the workspace before starting further tasks.
- The section is **task agnostic**: it MUST NOT incorporate per-request user
  instructions. It only summarizes workspace structure and defaults for
  long-lived tooling (tests, linters, package managers, container settings,
  etc.).
- Accept prompt overrides (`accepts_overrides=True`) so deployments can pin a
  curated digest when desired. When a session digest exists, it takes priority
  even if an override is present to reflect the freshest observation.

## Adapter `optimize` Method

Adapters MUST expose a new method alongside `evaluate`:

```python
class ProviderAdapter(Protocol):
    def optimize(
        self,
        prompt: Prompt[OutputT],
        *params: object,
        goal_section_key: str,
        chapters_expansion_policy: ChaptersExpansionPolicy = (
            ChaptersExpansionPolicy.ALL_INCLUDED
        ),
        parse_output: bool = True,
        bus: EventBus,
    ) -> PromptResponse[OutputT]:
        ...
```

- The signature mirrors `evaluate` so callers can swap between evaluation and
  optimization without branching on provider capabilities.
- `optimize` constructs and runs an **optimization prompt** analogous to the
  "optimize" REPL command described in `specs/code_reviewer_example.md`:
  - Render the supplied prompt against the provided params and goal section.
  - Execute the provider call, including any tool invocations, until a final
    assistant message is produced.
  - Parse structured output when declared, falling back to plain text.
- Optimization MUST run against a **separate `Session` and `EventBus`** from the
  primary request flow to keep optimization state and events isolated. The
  caller or adapter is responsible for constructing this sandboxed session and
  wiring its bus into the `optimize` invocation.
- The returned `PromptResponse` SHOULD be published as a `PromptExecuted` event
  on the supplied bus, matching `evaluate` semantics.
- Adapters SHOULD surface the resulting optimization content (structured output
  field or text) so callers can persist it as the workspace digest in the
  session and/or overrides store. This enables `WorkspaceDigest` sections to
  surface the optimized digest on subsequent renders without rerunning the
  optimization prompt.

## Integration Notes

- The `code_reviewer_example` will migrate from a bespoke “repository
  instructions” block to the shared `WorkspaceDigest` section. The REPL’s
  optimization command should call `adapter.optimize` with an isolated session
  and bus, then stash the emitted digest into the primary session and override
  store so every turn renders the latest summary without custom wiring.


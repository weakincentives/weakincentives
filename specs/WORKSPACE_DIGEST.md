# Workspace Digest Specification

## Overview

`WorkspaceDigestSection` captures a task-agnostic description of the mounted
workspace so prompts can reuse a single summary instead of repeatedly crawling
the filesystem. The section prefers live session state, gracefully falls back
to prompt overrides, and emits an explicit placeholder (plus a structured log
warning) when no digest exists yet.

## Section Behavior

- The section is keyed to `workspace-digest` by default and renders a markdown
  heading followed by the digest body.
- Resolution order:
  1. **Session snapshot** – read the latest `WorkspaceDigest` entry stored in the
     active `Session` via `session.workspace_digest.latest(section.key)`.
  1. **Override fallback** – when the session slice is empty, use the override
     body provided by the prompt overrides store (overrides are applied exactly
     as stored).
  1. **Placeholder** – emit the default placeholder text _and_ log a warning via
     the workspace-digest logger so operators know optimization has not run.
- The section is intentionally task agnostic; it only describes repo layout,
  tooling defaults, and recurring watch-outs.
- Overrides remain enabled (`accepts_overrides=True`) to support curated digests,
  but any populated session snapshot takes precedence so live optimizations win.
- `clone(session=...)` **requires** a `Session` argument because the section
  always binds to a concrete session instance. Callers must supply a new session
  when cloning for optimization prompts.

## Adapter `optimize` Workflow

`ProviderAdapter` implements `optimize` alongside `evaluate`:

```python
def optimize(
    self,
    prompt: Prompt[OutputT],
    *,
    store_scope: OptimizationScope = OptimizationScope.SESSION,
    overrides_store: PromptOverridesStore | None = None,
    overrides_tag: str | None = None,
    session: SessionProtocol,
    bus_subscribers: Sequence[tuple[type[object], EventHandler]] | None = None,
) -> OptimizationResult: ...
```

- Callers pass the same prompt they would normally evaluate. The adapter scans
  it to locate the `WorkspaceDigestSection` and whichever workspace section is
  present (Podman or VFS).
- The adapter launches a brand-new `Session` for the optimization run, cloning
  session-aware sections (workspace tools, digest) via `section.clone`. Optional
  `bus_subscribers` are registered against the optimization session’s event bus
  so tooling logs mirror the interactive REPL.
- Two small markdown sections—`Optimization Goal` and `Expectations`—lead the
  optimization prompt, followed by `PlanningToolsSection` configured with
  `GOAL_DECOMPOSE_ROUTE_SYNTHESISE`, then the cloned workspace tools and
  `WorkspaceDigestSection`.
- Optimization executes exactly like `evaluate`: the adapter renders the prompt,
  runs provider calls, and parses the structured `_OptimizationResponse`
  dataclass (`digest: str`). When parsing fails, plain text is used.

### Scope & Persistence

- `OptimizationScope.SESSION` (default) stores the digest only in
  `session.workspace_digest`.
- `OptimizationScope.GLOBAL` additionally requires `overrides_store` and
  `overrides_tag`; the adapter resolves the digest section path and writes the
  rendered markdown through `PromptOverridesStore.set_section_override`.
- Both scopes always persist to the active session slice so the digest is
  immediately available to the caller’s main session after optimization.

### OptimizationResult

The concrete dataclass currently includes:

- `response: PromptResponse[Any]`
- `digest: str`
- `scope: OptimizationScope`
- `section_key: str`

Callers can inspect the structured provider response, the stored digest text,
and the section key that was updated.

## Integration Notes

- `code_reviewer_example.py` wires the REPL `optimize` command to
  `ProviderAdapter.optimize`, subscribing loggers to the optimization session so
  prompt bodies and tool invocations remain visible.
- `WorkspaceDigestSection` is the canonical way to surface repo summaries across
  all prompts; bespoke “repository instructions” sections should be replaced by
  this shared component so optimization and overrides behave uniformly.

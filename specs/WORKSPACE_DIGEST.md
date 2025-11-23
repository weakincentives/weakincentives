# Workspace Digest Specification

## Overview

`WorkspaceDigestSection` captures a task-agnostic description of the mounted
workspace so prompts can reuse a single summary instead of repeatedly crawling
the filesystem. The section prefers live session state, gracefully falls back
to prompt overrides, and emits an explicit placeholder (plus a structured log
warning) when no digest exists yet.

## Rationale

- Reduce provider call costs by reusing a vetted workspace summary instead of
  relying on repeated filesystem crawls or ad hoc prompt context.
- Keep the digest bounded to predictable metadata (layout, tooling defaults,
  recurring pitfalls) so downstream prompts remain stable even as tasks vary.
- Preserve human-curated overrides so operators can inject authoritative
  guidance when automated optimization is unavailable or delayed.

## Scope

- Applies to prompts that need repository awareness without embedding
  task-specific instructions.
- Covers only mounted workspace state surfaced through the session or override
  stores; it does **not** attempt to summarize in-flight tool output or
  temporary scratch data.
- Excludes credential material, external service state, and ephemeral
  sandbox-only files unless the optimization process explicitly inserts them.

## Section Behavior

- The section is keyed to `workspace-digest` by default and renders a markdown
  heading followed by the digest body.
- Resolution order:
  1. **Session snapshot** – read the latest `WorkspaceDigest` entry stored in the
     active session via `latest_workspace_digest(session, section.key)`.
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

### Behavior–Data Mapping

- **Session snapshot path** → stores and retrieves the rendered digest text via
  `set_workspace_digest` / `latest_workspace_digest`.
- **Overrides path** → persists the rendered markdown override in
  `PromptOverridesStore` keyed by section path so any prompt can consume the
  same digest without re-optimizing.
- **Placeholder path** → emits the default placeholder string and logs a
  workspace-digest warning; no digest content is recorded in session or
  overrides.

### Data Captured

- Repository layout and notable directories or files required for navigation.
- Default tooling commands and workflows (tests, linting, formatting).
- Known caveats or recurring pitfalls that affect most tasks.
- Optimization metadata: section key, scope, and origin (session vs override).

### Limitations and Caveats

- The digest reflects the workspace at the time of optimization; it is not
  auto-refreshed after filesystem changes unless optimization is rerun.
- Placeholder renders omit contextual hints; operators should treat the warning
  log as a signal to generate a digest before relying on the section.
- Overrides can drift from reality if the workspace changes; prefer session
  snapshots when freshness is critical and clear overrides that are obsolete.
- The section avoids task-specific directives; pair it with task sections when
  additional guidance is required.

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
    optimization_session: Session | None = None,
) -> OptimizationResult: ...
```

- Callers pass the same prompt they would normally evaluate. The adapter scans
  it to locate the `WorkspaceDigestSection` and whichever workspace section is
  present (Podman or VFS).
- The adapter launches a brand-new `Session` for the optimization run, cloning
  session-aware sections (workspace tools, digest) via `section.clone`. Advanced
  callers may pass an existing `optimization_session` when they need to preserve
  a custom event bus or logging context.
- Two small markdown sections—`Optimization Goal` and `Expectations`—lead the
  optimization prompt, followed by `PlanningToolsSection` configured with
  `GOAL_DECOMPOSE_ROUTE_SYNTHESISE`, then the cloned workspace tools and
  `WorkspaceDigestSection`.
- Optimization executes exactly like `evaluate`: the adapter renders the prompt,
  runs provider calls, and parses the structured `_OptimizationResponse`
  dataclass (`digest: str`). When parsing fails, plain text is used.

### Scope & Persistence

- `OptimizationScope.SESSION` (default) stores the digest only in the session’s
  `WorkspaceDigest` slice via `set_workspace_digest`.
- `OptimizationScope.GLOBAL` additionally requires `overrides_store` and
  `overrides_tag`; the adapter resolves the digest section path and writes the
  rendered markdown through `PromptOverridesStore.set_section_override`. After a
  successful write the caller’s session slice is cleared so
  `WorkspaceDigestSection` immediately falls back to the override snapshot.
- Only the session scope keeps the digest cached locally; global scope trades
  the session entry for the persisted override so future renders stay in sync
  after `clear_workspace_digest` removes the stale snapshot.

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
  `ProviderAdapter.optimize`.
- `WorkspaceDigestSection` is the canonical way to surface repo summaries across
  all prompts; bespoke “repository instructions” sections should be replaced by
  this shared component so optimization and overrides behave uniformly.

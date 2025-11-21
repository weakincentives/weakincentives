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
  1. **Override fallback** – when no digest exists in the session, resolve a
     template override for the section from the active overrides store (see
     `specs/PROMPT_OVERRIDES.md`). Overrides must be applied exactly as stored,
     including any escaped template tokens.
  1. **Empty state** – if neither source provides content, log a warning to the
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
        store_scope: OptimizationScope = OptimizationScope.SESSION,
        overrides_store: PromptOverridesStore | None = None,
        overrides_tag: str | None = None,
        focus_areas: MarkdownSection | None = None,
        parse_output: bool = True,
        session: Session,
    ) -> OptimizationResult[OutputT]:
        ...
```

- The signature mirrors `evaluate` so callers can swap between evaluation and
  optimization without branching on provider capabilities. `optimize` returns an
  `OptimizationResult` dataclass that captures the **output** of the inner
  optimization prompt (structured output and digest text), not the input prompt
  itself.
- Callers pass the same user-facing prompt they would evaluate; the adapter
  derives optimization metadata (workspace section, digest section key) by
  inspecting that prompt instead of requiring explicit arguments such as a
  `goal_section_key`.
- `OptimizationResult` SHOULD, at minimum, wrap the executed `PromptResponse`,
  the extracted digest text, and the requested `OptimizationScope` so the
  adapter can persist and callers can forward the optimized workspace summary
  without re-rendering the prompt.
- `optimize` constructs and runs an **optimization prompt** analogous to the
  "optimize" REPL command described in `specs/code_reviewer_example.md`:
  - Render the supplied prompt against the provided params, deriving the
    workspace and digest sections from the prompt structure for optimization.
  - Optionally append the `focus_areas` section to guide the model toward
    specific files, subsystems, or workflows the caller wants prioritized in the
    digest.
  - Execute the provider call, including any tool invocations, until a final
    assistant message is produced.
  - Parse structured output when declared, falling back to plain text.
  - Optimization reads from the supplied `Session` (typically sandboxed from the
    main flow) and does not require an `EventBus` argument. Any event publishing
    should reuse the bus already attached to the session. Adapters MUST persist
    the resulting optimization content (structured output field or text) within
    `optimize` based on the configured scope so the digest is immediately
    available for subsequent prompts without rerunning the optimization
    prompt.

### Optimization Scope

- Introduce `OptimizationScope` as an enum that indicates where optimized
  results are persisted:
  - `SESSION` — store only in the calling session’s workspace-digest slice.
  - `GLOBAL` — store in the prompt overrides store so the digest survives across
    sessions and processes.
- When `store_scope` is `GLOBAL`, callers MUST supply both `overrides_store` and
  `overrides_tag` so the adapter can write the digest to the correct override
  entry. Calls missing either parameter should raise or fail early.
- `optimize` MUST handle persistence internally: it should always write the
  digest to the session slice and, for `GLOBAL` scope, also record the digest in
  the overrides store using the provided tag. Callers should not need to perform
  extra persistence steps.

### Optimization Prompt Assembly

The adapter accepts a normal user prompt and internally builds the optimization
prompt by cloning the incoming structure. It scans the provided prompt to find
the workspace section (VFS or Podman) and the `WorkspaceDigest` section key,
then injects optimization guidance that targets that digest goal. `optimize`
generates its own prompt variant and only extracts the workspace digest section
from the optimized prompt rather than requiring a chapters expansion policy
argument. A typical flow:

```python
def optimize(self, prompt: Prompt[OutputT], *, session: Session, ...):
    goal_section_key = prompt.find_section_key("workspace-digest")
    workspace_section = prompt.find_section(("vfs", "podman"))
    optimization_prompt = prompt.clone_with(
        optimization_sections=[
            MarkdownSection(
                "Optimization Goal",
                "Focus on summarizing the workspace so future prompts can rely on a cached digest.",
            ),
            MarkdownSection(
                "Expectations",
                "List key files, configs, and workflows; avoid task-specific advice.",
            ),
        ],
        focus_areas=focus_areas,
        restricted_sections=[workspace_section, goal_section_key],
    )

    response = self._execute_prompt(
        optimization_prompt,
        session=session,
        parse_output=True,
    )
    digest = response.structured_output.get(goal_section_key) or response.text
    if store_scope is OptimizationScope.GLOBAL:
        assert overrides_store is not None and overrides_tag, "Global scope requires overrides store and tag"
        overrides_store.set_override(
            prompt.namespace,
            prompt.key,
            overrides_tag,
            goal_section_key,
            digest,
        )

    session.workspace_digest.set(goal_section_key, digest)

    return OptimizationResult(
        digest=digest,
        response=response,
        scope=store_scope,
    )
```

An example `OptimizationResult` implementation:

```python
@dataclass
class OptimizationResult(Generic[OutputT]):
    response: PromptResponse[OutputT]
    digest: str
    scope: OptimizationScope
```

- The injected sections provide the optimization instructions that mirror the
  "optimize" REPL command: explore README/docs first, record build/test
  commands, and emit a markdown digest suitable for the `WorkspaceDigest`
  section.
- The adapter is responsible for deriving the digest target from the prompt
  itself (e.g., the `WorkspaceDigest` section key discovered during scanning)
  rather than receiving it as an explicit argument so callers only supply the
  user-facing prompt and params.
- The adapter may use helper methods (e.g., `_execute_prompt`) that mirror
  `evaluate` behavior: render, stream/tool execute, and parse output. The digest
  is read from structured output when available; otherwise the final assistant
  message text is treated as the digest body.
- Persistence happens inside `optimize`; callers only need to provide scope and
  override details. The method records the digest in the session (and overrides
  when requested) so the `WorkspaceDigest` section can reuse it without
  re-optimizing.

## Integration Notes

- The `code_reviewer_example` will migrate from a bespoke “repository
  instructions” block to the shared `WorkspaceDigest` section. The REPL’s
  optimization command should call `adapter.optimize` with a dedicated session
  (no extra bus argument); the method writes the digest into the session and, if
  requested, the override store so every turn renders the latest summary
  without custom wiring. Any legacy `RepositoryOptimizationRequest`/
  `RepositoryOptimizationResponse` classes and related custom parsing should be
  removed in favor of this simpler digest-handling path.

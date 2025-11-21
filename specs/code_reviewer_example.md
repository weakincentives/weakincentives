# Code Reviewer Example

This document outlines the scope, architecture, and runtime behavior of the
`code_reviewer_example.py` program. The example demonstrates how the
`weakincentives` platform assembles an interactive review agent with prompt
overrides, planning tools, and a repository-specific optimization workflow.

## Goals

- Showcase a minimal-yet-complete agent that reviews a mounted repository
  (default: `test-repositories/sunfish`).
- Exercise the core subsystems shipped by the library:
  - Prompt rendering and structured output.
  - Runtime session/event bus plumbing.
  - Planning and workspace tool sections.
  - Prompt overrides managed by `LocalPromptOverridesStore`.
- Provide an interactive REPL that surfaces the agent’s reasoning and allows a
  human operator to steer review tasks.

## High-Level Flow

1. `CodeReviewApp` bootstraps the OpenAI adapter, creates a runtime session, and
   seeds prompt overrides (default tag = session UUID unless
   `CODE_REVIEW_PROMPT_TAG` is set).
1. `build_task_prompt` composes four sections:
   - **Code Review Brief** — static instructions about reviewing mounted code.
   - **Workspace Digest** — task-agnostic workspace summary sourced from the
     session or overrides; replaces the bespoke repository instructions block.
   - **Planning Tools** — exposes PLAN/ACT/REFLECT helpers tied to the session.
   - **Review Request** — injects the user’s turn-by-turn prompt.
1. Each REPL turn renders the prompt, dispatches it through the adapter, then
   prints the structured response and the current planning state.
1. The special `optimize` command launches a dedicated session, renders a
   repository-optimization prompt, and persists the resulting digest in the
   session/overrides for the `WorkspaceDigest` section.

## Key Components

### Prompts

- `build_task_prompt` — main review prompt (namespace
  `examples/code-review:code-review-session`).
- `build_repository_optimization_prompt` — helper prompt that guides the agent
  through README/docs/tooling exploration and yields a Markdown block suitable
  for the `WorkspaceDigest` section.
- Both prompts use `MarkdownSection` instances plus the shared planning/workspace
  sections so the agent can list files, inspect paths, and update multi-step
  plans.

### Overrides

- Overrides are backed by `LocalPromptOverridesStore` and keyed by namespace +
  prompt key + tag.
- `initialize_code_reviewer_runtime` seeds the override file on startup so
  later edits always have a stable hash reference.
- Repository instructions are no longer a bespoke section; override writes
  target the shared `WorkspaceDigest` section so callers persist a single
  digest format.

### Optimization Command

- Trigger: User types `optimize` to refresh the workspace digest.
- Implementation steps:
  1. Create a `Session` dedicated to the optimization prompt so tool
     invocations/events don’t pollute the main session (the attached bus, if
     present, travels with the session implicitly). Any sections borrowed from
     the main prompt must be safe to run with this new session/bus; clone or
     rebuild them if they capture session state.
  1. Invoke the adapter’s `optimize` method with the normal review prompt; no
     bespoke `RepositoryOptimizationResponse` class is needed because the digest
     is treated as markdown content. The adapter scans the prompt to find the
     workspace and `WorkspaceDigest`
     sections (using keys or type references), builds a brand-new optimization
     prompt with a simplified preamble and only the necessary sections, and runs
     the inner optimization flow automatically without rendering the original
     user prompt.
  1. Persist the digest into the main session’s workspace-digest slice and the
     overrides store (using `OptimizationScope.GLOBAL` and a supplied
     `overrides_tag`) so the `WorkspaceDigest` section renders it on subsequent
     turns and across runs.

The optimization prompt is constructed fresh with a concise preamble and only
the workspace-aware sections; the adapter identifies the relevant sections while
building this new prompt. A minimal pseudo-code sketch:

```python
result = adapter.optimize(
    build_task_prompt(user_request, overrides_tag),
    store_scope=OptimizationScope.GLOBAL,
    overrides_store=overrides_store,
    overrides_tag=overrides_tag,
    session=optimization_session,
)
digest_body = result.digest
```

The assistant is instructed to explore the repository (README, docs, workflow
files, default test/build commands) and emit a markdown digest. The digest body
feeds both the session and overrides store so later `WorkspaceDigest` renders do
not need to re-run the optimization flow.

### WorkspaceDigest Simplification Roadmap

- Replace the repository instructions section with the shared `WorkspaceDigest`
  section so the REPL uses the same task-agnostic digest format as other
  orchestrations.
- Remove `RepositoryOptimizationRequest`/`RepositoryOptimizationResponse` and
  the specialized optimization plumbing—optimization simply renders the digest
  prompt, runs `adapter.optimize` against a dedicated session, and writes the
  resulting digest back to session state and overrides.
- Deprecate custom override plumbing by sourcing digest content from the active
  session first, then falling back to prompt overrides—any helper that persists
  digest content should operate directly on the workspace-digest entry consumed
  by the shared section.
- Simplify prompt construction: `build_task_prompt` imports `WorkspaceDigest`
  directly instead of maintaining a local Markdown section and related
  rendering code.

## Running the Example

```bash
uv run python code_reviewer_example.py
```

Environment requirements:

- `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`) must be set.
- Podman is optional; when unavailable the agent falls back to the in-memory VFS
  tool set.

Interactive commands:

- Any non-empty input except the reserved commands is treated as a review task.
- `optimize` refreshes the repository instructions override using the global
  store scope so the digest persists across runs.
- `exit` or `quit` terminates the REPL.

## Testing Strategy

- `tests/test_code_reviewer_example.py` covers:
  - Prompt rendering events and session logging.
  - Default empty repository instructions and override persistence.
  - The optimize command path using a stub adapter that emits canned
    instructions.
  - Escape handling for `$` placeholders in stored overrides.
- `make test` exercises the entire suite with a 100% coverage floor, while
  `make check` runs the full verification pipeline (format, lint, typecheck,
  security, deps, markdown, tests).

## Future Extensions

- Swap the OpenAI adapter for other providers by injecting a different
  `ProviderAdapter`.
- Expose additional commands (e.g., custom repo initialization) by following the
  same pattern as the `optimize` handler.
- Extend the repository instructions section to include structured subsections
  (e.g., “Build”, “Tests”, “Watchouts”) if more granularity is desired.

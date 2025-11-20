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
1. The special `optimize [focus]` command launches an isolated session/bus,
   renders a repository-optimization prompt, and persists the resulting digest
   in the session/overrides for the `WorkspaceDigest` section.

## Key Components

### Prompts

- `build_task_prompt` — main review prompt (namespace
  `examples/code-review:code-review-session`).
- `build_repository_optimization_prompt` — helper prompt that guides the agent
  through README/docs/tooling exploration and yields a Markdown block suitable
  for the repository instructions section.
- Both prompts use `MarkdownSection` instances plus the shared planning/workspace
  sections so the agent can list files, inspect paths, and update multi-step
  plans.

### Overrides

- Overrides are backed by `LocalPromptOverridesStore` and keyed by namespace +
  prompt key + tag.
- `initialize_code_reviewer_runtime` seeds the override file on startup so
  later edits always have a stable hash reference.
- `save_repository_instructions_override` replaces the body of the repository
  instructions section with escaped Markdown (all `$` are doubled) to avoid
  `string.Template` collisions during render time.

### Optimization Command

- Trigger: User types `optimize` with an optional focus string (defaults to
  “Survey README, docs, and key scripts…”).
- Implementation steps:
  1. Create an `InProcessEventBus` and `Session` dedicated to the optimization
     prompt so tool invocations/events don’t pollute the main session.
  1. Render the optimization prompt with `RepositoryOptimizationRequest`.
  1. Invoke the adapter’s `optimize` method (same signature shape as
     `evaluate`) to execute the prompt within the sandboxed session/bus and
     extract `RepositoryOptimizationResponse.instructions` (falling back to
     plain text if structured output is missing).
  1. Persist the Markdown override and print the new instructions to stdout.

### WorkspaceDigest Simplification Roadmap

- Replace the repository instructions section with the shared `WorkspaceDigest`
  section so the REPL uses the same task-agnostic digest format as other
  orchestrations.
- Deprecate custom override plumbing by sourcing digest content from the active
  session first, then falling back to prompt overrides—`save_repository_instructions_override`
  becomes a thin helper that updates the digest entry used by the section.
- Route optimization runs through `adapter.optimize`, publishing the resulting
  digest back into the sandboxed session before persisting to overrides, so the
  main REPL can render the fresh summary without bespoke wiring.
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
- `optimize [focus]` refreshes the repository instructions override.
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

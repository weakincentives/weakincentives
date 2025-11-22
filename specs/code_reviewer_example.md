# Code Reviewer Example

This handbook entry documents the `code_reviewer_example.py` program that ships
alongside the library. The script assembles a full-featured review agent that
demonstrates prompt composition, overrides, workspace tools, and adapter
optimization hooks in one place.

## Objectives

- Mount the `test-repositories/sunfish` fixture read-only and guide the agent
  through code review workflows.
- Exercise core subsystems: prompt rendering, session/event bus plumbing,
  planning tools, workspace digests, Podman/VFS tooling, prompt overrides, and
  adapter-driven optimization prompts.
- Provide an interactive REPL with transparent logging (prompt renderings, tool
  calls, plan snapshots) so operators can steer the review.

## Runtime Architecture

`CodeReviewApp` wires together the adapter, prompt, session, overrides store,
and event bus. The helper `_create_runtime_context` builds these pieces and
subscribes console loggers to `PromptRendered` and `ToolInvoked` events.

On startup:

1. `_ensure_test_repository_available` verifies that
   `PROJECT_ROOT/test-repositories` exists.
1. `build_task_prompt(session=Session())` composes the prompt (details below).
1. A `LocalPromptOverridesStore` is initialized, seeded (once) with the prompt,
   and keyed by namespace + prompt key + tag.
1. `_resolve_override_tag` selects the overrides tag using the order:
   explicit `override_tag` arg → `CODE_REVIEW_PROMPT_TAG` env var → `"latest"`.
1. `CodeReviewApp.run()` prints an intro banner, accepts review prompts until
   EOF/`exit`/`quit`, and emits plan snapshots after every agent response.

Each REPL turn invokes `ProviderAdapter.evaluate` with
`ReviewTurnParams(request=...)` and our session/bus/override state. Responses
are converted into human-readable text by `_render_response_payload`, which
mirrors the `ReviewResponse` dataclass (`summary`, `issues`, `next_steps`). Plan
data is pulled from the session via `select_latest(session, Plan)`.

## Prompt Composition

`build_task_prompt` produces `Prompt[ReviewResponse]` with namespace
`examples/code-review` and key `code-review-session`. Sections render in this
order:

1. **Code Review Brief** (`MarkdownSection[ReviewGuidance]`):
   - Hard-coded template describing tooling, delegation strategy, and output
     format.
   - Ships with `ReviewGuidance.focus` default instructions.
1. **Workspace Digest** (`WorkspaceDigestSection`):
   - Renders cached workspace notes from the session or overrides store.
   - Populated interactively via the `optimize` command.
1. **Subagents** (`SubagentsSection`):
   - Enables `dispatch_subagents` so the agent can parallelize exploration.
1. **Planning Tools** (`PlanningToolsSection`):
   - Uses `PlanningStrategy.PLAN_ACT_REFLECT`, allowing multi-step plans whose
     snapshots we print after every turn.
1. **Workspace Tools**:
   - `_build_workspace_section` picks a `PodmanSandboxSection` when
     `PodmanSandboxSection.resolve_connection()` succeeds; otherwise it falls
     back to `VfsToolsSection`.
   - Both variants mount the `sunfish` repo with
     `SUNFISH_MOUNT_INCLUDE_GLOBS`, `SUNFISH_MOUNT_EXCLUDE_GLOBS`, and a
     600 KB cap to keep prompts concise. Podman inherits the base URL,
     identity, and connection name from the resolved connection.
1. **Review Request** (`MarkdownSection[ReviewTurnParams]`):
   - Echoes `${request}` verbatim so the model always sees the latest operator
     question at the end of the prompt.

All sections accept overrides unless otherwise noted, but only the workspace
digest is expected to receive long-lived overrides.

## Overrides & Optimization

Overrides live in `LocalPromptOverridesStore`, defaulting to
`~/.weakincentives/prompts`. `initialize_code_reviewer_runtime` is a helper
used by tests (and other code) to get a prompt/session/bus/store/tag tuple
without booting the REPL.

The REPL recognizes a single special command:

- `optimize`: calls `ProviderAdapter.optimize` with the base review prompt,
  storing results in the ambient session (`OptimizationScope.SESSION`). Event
  subscribers print the optimization prompt body (`PromptRendered`) and log
  tool invokes (`ToolInvoked`), giving full visibility into the digest refresh.

The shared adapter logic (see `ProviderAdapter.optimize`) builds its own
optimization prompt using two short markdown sections
(`Optimization Goal`, `Expectations`), the planning tools, and whichever
workspace section matches the review prompt (Podman or VFS). The digest body is
persisted via `session.workspace_digest.set` and, when callers request global
scope, via `PromptOverridesStore.set_section_override`.

## Logging & Observability

- `_print_rendered_prompt` prints the entire prompt whenever `PromptRendered`
  fires. The label uses `prompt_name` or `ns:key`.
- `_log_tool_invocation` renders tool params/results/payloads via the serde
  helpers while truncating long strings (`_LOG_STRING_LIMIT = 256`).
- `_render_plan_snapshot` summarizes the latest plan objective, per-step status,
  notes, and details. We call it after every agent response so operators can
  track progress.

## Running the Example

```bash
OPENAI_API_KEY=sk-... uv run python code_reviewer_example.py
```

Environment knobs:

- `OPENAI_API_KEY` is required. `OPENAI_MODEL` defaults to `gpt-5.1`.
- `CODE_REVIEW_PROMPT_TAG` customizes the overrides tag shared across runs.

Interactive commands:

- Non-empty input becomes the next review request.
- `optimize` refreshes the workspace digest in the current session (and prints
  the resulting markdown).
- `exit`/`quit` or an empty line terminates the REPL.

## Testing

- `tests/test_code_reviewer_example.py` covers prompt rendering logs, default
  workspace digest behavior, overrides precedence, the optimize command (via a
  stub adapter), and structured response formatting.
- `tests/test_thread_safety.py` reuses `initialize_code_reviewer_runtime` during
  concurrency checks.
- `make test` / `make check` enforce the repository-wide guarantees
  (formatting, lint, typecheck, security scans, dependency audits, markdown
  lint, and 100 % coverage).

## Future Work

- Swap adapters by passing a different `ProviderAdapter` into `CodeReviewApp`.
- Extend the workspace digest format with structured subsections and persist
  them through overrides.
- Introduce more REPL commands (e.g., “reset overrides”, “rerender prompt”) by
  following the pattern used for `optimize`.

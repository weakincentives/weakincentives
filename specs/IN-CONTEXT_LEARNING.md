# In-Context Learning Optimizer Specification

## Overview

The "optimize" command in `code_reviewer_example.py` demonstrates how an agent can
refresh repository-specific instructions by launching an auxiliary prompt that scans
the workspace, exercises core tools, and summarizes lessons learned for future
turns. This document generalizes that behaviour into a reusable optimizer surface
that orchestrators can wire into any session. The goal is to maximize in-context
learning automatically so downstream prompts inherit fresh, high-signal guidance
about the workspace, available tooling, and prior session history.

## Goals

- Provide a drop-in library component that runs an optimization prompt inside an
  isolated `Session`, mirroring the example's `RepositoryOptimizer`.
- Standardize the strategies used during optimization so every invocation reliably
  explores the filesystem, exercises tool surfaces, and reflects on previous turns.
- Define how optimized instructions are persisted (e.g., stored under the
  `repository-instructions` section override) so subsequent prompt renders can
  consume them without custom glue code.
- Keep the optimizer self-contained: callers provide only the `ProviderAdapter`,
  `Session`, and a high-level objective; the component handles orchestration.

## Non-Goals

- Changing how task prompts ingest instruction overrides (that pipeline already
  exists via `LocalPromptOverridesStore`).
- Adding new tools. The optimizer uses existing workspace, planning, and history
  surfaces; extending the toolset is out of scope.
- Capturing full transcripts of the optimization run; only the structured
  instruction payload is persisted.

## Key Concepts

### Workspace Filesystem Exploration

The optimizer must inspect the mounted repository automatically:

- Use `VfsToolsSection` and/or `PodmanSandboxSection` (depending on the active
  session) to list directories at the workspace root, identify language-specific
  files, and read canonical documents (README, docs, build scripts).
- Encourage breadth-first scans: start with `ls`/`list_dir`, follow with targeted
  `read_file` calls driven by the planning strategy.
- Capture notable observations (languages, frameworks, build/test commands) that
  should flow into the final instructions.

### Tools Exploration

Optimized instructions should also highlight which tools are available and how to
employ them effectively in this repository:

- Run at least one `planning` tool invocation per session (default to
  `PlanningStrategy.PLAN_ACT_REFLECT` so plans include reflection notes).
- Invoke tool metadata inspectors when available (e.g., the tools registry the
  prompt exposes) to understand resource limits, sandbox behaviour, and quota
  constraints.
- Document the discovered tools in the final instructions with actionable usage
  hints ("use `dispatch_subagents` for parallel doc sweeps", "prefer `shell_execute`
  for lightweight build commands", etc.).

### History Retrospectives

Optimization should not happen in a vacuum. When the host session already has
turns on the `EventBus`, the optimizer should analyze them:

- Snapshot the latest turns via `Session.snapshot()` or `select_latest(session)`
  (see `code_reviewer_example` usage) and feed them into the optimization prompt as
  a structured appendix.
- Summaries must call out recurring mistakes or successful tactics so future turns
  keep or avoid them.
- When no history is available, the optimizer should note this explicitly and
  focus solely on workspace/tool discovery.

## API Surface

Add a general-purpose optimizer under `weakincentives.runtime.optimizer` with the
following shape:

```python
@dataclass(slots=True)
class OptimizationObjective:
    focus: str

@dataclass(slots=True)
class OptimizationResponse:
    instructions: str
    workspace_digest: str
    tool_digest: str
    history_digest: str

class InContextLearningOptimizer:
    def __init__(
        self,
        adapter: ProviderAdapter[SupportsDataclass],
        overrides_store: LocalPromptOverridesStore,
        overrides_tag: str,
    ) -> None: ...

    def run(self, *, session: Session, objective: OptimizationObjective) -> OptimizationResponse | None:
        """Launches the optimization prompt in a temporary session and returns structured guidance."""
```

Key rules:

- `run()` constructs a fresh `Session` and `EventBus` (so optimization does not
  mutate the caller's planning state) but receives the caller's `Session` to read
  history snapshots.
- The structured response is optional to allow adapters that can only emit plain
  text; fallback to `Response.text` when `output` is missing, mirroring the example.

## Prompt Composition

The optimizer prompt follows the same layout as `build_repository_optimization_prompt`
with additional sections:

1. **Optimization Brief** – describes the goal, workspace overview, and instructs
   the model to return JSON with `instructions`, `workspace_digest`, `tool_digest`,
   and `history_digest` fields. Remind the model to rely on planning, filesystem,
   and history sections explicitly.
2. **History Retrospective** – a markdown section containing the serialized
   session snapshot (limited to the last N turns; default 3 to cap token usage).
3. **Planning Tools Section** – reuse `PlanningToolsSection` with PAR strategy to
   encourage deliberate exploration.
4. **Subagents Section** – encourages parallel doc/code scans where supported.
5. **Workspace Tools Section** – mount the repo via `PodmanSandboxSection` when
   possible, otherwise fallback to `VfsToolsSection` (same helper as the example).
6. **Optimization Objective** – renders the user-provided focus text.

## Persistence Contract

- After `run()` returns, orchestrators should persist `response.instructions` to
  the `repository-instructions` override (identical to
  `save_repository_instructions_override()` in the example).
- The `workspace_digest`/`tool_digest`/`history_digest` fields are optional
  payloads for logging or higher-level dashboards; they are not automatically
  persisted but should be available to downstream automation if desired.
- The optimizer must trim whitespace and normalize Markdown before returning the
  payload so it can be injected into prompts without reformatting.

## Error Handling

- If the adapter evaluation fails or returns neither structured output nor text,
  `run()` should log the event and return `None` rather than raising.
- Callers are responsible for printing/logging fallback messages ("Optimize command
  produced no instructions") exactly as in the example.

## Testing Guidance

- Add regression tests under `tests/prompt` that snapshot the rendered optimizer
  prompt to ensure the required sections (brief, history, planning, workspace,
  objective) appear in the expected order.
- Exercise `InContextLearningOptimizer` in an integration test that feeds a mock
  adapter returning a deterministic `OptimizationResponse`, verifying that
  `repository-instructions` overrides persist the payload.
- Add unit tests covering history sourcing: populate a fake `Session` with mock
  events, run the optimizer, and assert the history section includes the latest
  turn summaries.

## Usage Pattern

1. Instantiate `InContextLearningOptimizer` alongside the main prompt adapter.
2. Wire an "optimize" (or similarly named) command into the REPL/CLI loop; pass the
   user's focus text into `OptimizationObjective`.
3. Run the optimizer, persist the returned instructions, and optionally display
  them to the user.
4. Future task turns automatically pick up the refreshed instructions via the
   overrides store.

This spec elevates the example command into a reusable primitive so every agent can
benefit from automated in-context learning without bespoke wiring.

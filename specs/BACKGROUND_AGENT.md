# Background Agent Orchestrator Specification

## Overview

The background agent orchestrator owns a long-running objective loop that coordinates prompts, tools, and session
state without performing side effects itself. Callers provide an objective payload and optional guardrails; the
orchestrator drives the LLM through planning, tool execution, and iterative reflection until the objective is complete
or an unrecoverable failure threshold is reached. The workspace surface is the session-scoped virtual filesystem; host
mounts supply read-only context while all edits happen inside the VFS sandbox. Every iteration is deterministic with
respect to the objective, configuration, and recorded events so runs remain auditable and reproducible.

## Goals

1. **Objective focused** – Accept a single objective and pursue it until completion, abandonment, or failure.
1. **Side effect free core** – Never execute external side effects directly; instead instruct tools that surface their
   own validations and send `ToolInvoked` events.
1. **VFS mediated workspace** – Operate exclusively through the virtual filesystem, seeding host context via explicit
   mounts.
1. **Session native** – Persist all state inside `Session` reducers so external callers can monitor progress and feed
   custom analytics without touching private orchestrator internals.
1. **Self-optimising loop** – Incorporate planning, reflection, and retry heuristics that keep the agent on track with
   minimal operator oversight.

Non-goals:

- Multi-objective batching or scheduling.
- Cross-run persistence or checkpointing beyond the session boundary.
- Transport-specific adapters (HTTP, CLI) – these remain caller responsibilities.

## Module Surface

```
src/weakincentives/agent/background/
  __init__.py
  config.py        # dataclasses describing objectives, limits, and feature toggles
  runner.py        # BackgroundAgentRunner, entry point for end users
  loop.py          # control loop implementation and iteration helpers
  state.py         # derived state helpers built on top of Session selectors
  prompts.py       # prompt sections wired for this orchestrator
```

- `BackgroundAgentRunner` exposes a synchronous `run()` API as the recommended entry point.
- Async orchestration is deferred; the first version is single threaded and blocking.
- The package re-exports `BackgroundAgentRunner`, config dataclasses, and result enums from `__all__`.

## Inputs and Outputs

### Objective payload

The runner accepts an immutable `AgentObjective` dataclass:

```python
from typing import Any
from dataclasses import field
from weakincentives.agent.background.config import AgentDeliverable
from weakincentives.prompt import Section
from weakincentives.tools.vfs import HostMount

@dataclass(slots=True, frozen=True)
class AgentObjective:
    objective: str                                      # concise statement of the goal
    deliverable_schema: type[AgentDeliverable]          # dataclass describing structured output
    context_sections: tuple[Section[Any], ...] = field(
        default_factory=tuple
    )                                                   # caller-supplied prompt sections
    host_mounts: tuple[HostMount, ...] = field(
        default_factory=tuple
    )                                                   # initial VFS mounts
```

Strings are normalised to ASCII and capped at 1,024 characters. `deliverable_schema` points at a frozen dataclass that
models the expected structured artefact and must implement the `AgentDeliverable` protocol so the orchestrator can
instantiate and validate the final payload. `context_sections` lets callers pass fully constructed prompt `Section`
instances (for example, Markdown sections with scoped variables). The sections render after the core behaviour block
and before the planning tools. `host_mounts` seeds the virtual filesystem with caller-approved folders (see
`specs/VFS_TOOLS.md`); mounts default to empty when omitted.

### Runtime configuration

`BackgroundAgentConfig` captures operator guardrails:

- `max_iterations` (default: 24) – hard stop for the control loop.
- `max_failures` (default: 8) – consecutive failed tool attempts or reflection loops before aborting.
- `iteration_timeout` (optional `datetime.timedelta`) – upper bound for a single loop iteration, enforced cooperatively.
- `enable_reflection` (default: `True`) – toggles post-action reflection prompts.
- `allow_plan_reset` (default: `True`) – permits the agent to abandon the plan mid-run.
- `objective_schema` – optional callable that validates custom objective dataclasses.
- `deliverable_factory` – optional callable `(AgentSnapshot) -> AgentDeliverable` evaluated when the run terminates. The
  factory is deterministic and raises when it cannot synthesise the deliverable.

### Run result contract

`BackgroundAgentRunner.run()` returns a frozen `AgentRunResult`:

```python
@dataclass(slots=True, frozen=True)
class AgentRunResult:
    status: Literal["completed", "abandoned", "failed"]
    iterations: int
    failures: int
    final_summary: str
    deliverable: AgentDeliverable | None
    session: Session          # session populated during the run
```

`status` reflects the termination condition (objective satisfied, caller-requested stop, or guardrail breach). `final_summary`
contains the latest LLM-provided wrap-up or a synthetic failure report.
`deliverable` holds the structured artefact produced by `deliverable_factory` when present.

`AgentDeliverable` is defined as a frozen protocol/dataclass in `config.py`. `deliverable_schema` on the objective must
implement this protocol so downstream consumers can model domain-specific payloads (for example `CodeReviewSummary`).
The runner does not inspect the payload beyond storing it on the result.

## Custom Deliverables

- The runner evaluates `deliverable_factory` after the loop exits but before returning `AgentRunResult`. The factory may
  rely on `objective.deliverable_schema` to construct the artefact.
- The factory receives the final `AgentSnapshot` and must return an ASCII-safe dataclass instance representing the
  artefact for this run.
- Failures bubble as run failures; the loop does not attempt retries because the artefact synthesis runs post-loop.
- When the factory is absent the runner leaves `deliverable` as `None`.

## Control Loop

The orchestrator operates a deterministic sequence of phases on every iteration:

1. **Assess** – Pull the latest `Plan`, tool outputs, and prompt transcripts from the session to build an
   `AgentSnapshot`. Feed the snapshot into prompt variables.
1. **Decide** – Render the main prompt using `Prompt.render()` and the objective parameters, then let the LLM choose the
   next action (tool call, message, or plan update). The prompt wiring adheres to `specs/PROMPTS.md`.
1. **Act** – When the decision is a tool call, dispatch through the configured tool adapter and publish results onto the
   shared event bus. All file exploration and edits flow through VFS tools initialised with the objective's host mounts.
   Free-form responses are persisted as `PromptExecuted` events.
1. **Reflect** – Optionally run a reflection prompt that evaluates the most recent action and proposes adjustments. This
   phase may request retries when a tool call failed or produced low-quality output.
1. **Check termination** – Evaluate termination guards in order: explicit completion signal from the LLM, plan status
   marked `completed`, exceeding `max_iterations`, exceeding `max_failures`, or reflection-triggered abandonment.

Each iteration appends structured telemetry events so the session reducers stay aligned with the orchestrator state.

## Session Integration

- The runner creates or accepts a preconfigured `Session` hooked to an event bus as described in `specs/SESSIONS.md`.
- `loop.AgentSnapshot` collects:
  - Latest plan (`Plan` dataclass from the planning tool reducers).
  - Last five tool invocations (normalised `ToolData` entries).
  - Pending reflection tasks.
  - Objective metadata.
  - Current virtual filesystem snapshot (`VirtualFileSystem`) maintained by the VFS reducers.
- VFS host mounts are materialised once during runner initialisation via `VfsToolsSection` using
  `objective.host_mounts`.
- Snapshot assembly relies solely on `select_*` helpers; no private state is stored inside the orchestrator.
- Reflection prompts and termination checks consult the snapshot rather than ad hoc instance attributes to ensure the
  run remains replayable.

## Use Case Spotlight: Code Review

The canonical complex objective for the background agent is conducting a code review inside the virtual file system
described in `specs/VFS_TOOLS.md`.

### Setup

1. The caller defines `AgentObjective.objective` as a terse review command (for example, "Review the changes in
   feature/user-session branch for regressions and style violations").
1. `AgentObjective.context_sections` supplies Markdown sections rendering the diff summary, test plan, or reviewer
   checklist derived from the VFS.
1. `AgentObjective.deliverable_schema` is set to the `CodeReviewSummary` dataclass with fields like `verdict`,
   `findings`, and `next_steps`.
1. `AgentObjective.host_mounts` lists read-only folders (for example `HostMount(host_path="diff/")`) preloaded into the
   virtual filesystem so the agent can inspect the patch set.
1. Tool adapters register domain-specific helpers (linters, test runners) that complement the built-in planning and VFS
   suites while preserving the side-effect free policy.

### Expected Loop Behaviour

- First iteration must establish a plan via the planning tool, capturing key inspection steps (map changes, run static
  checks, confirm tests).
- While acting, the agent uses VFS tools exclusively to inspect code, gathers findings, and records them in the plan
  notes.
- Reflection triggers on failed tool invocations (missing files, parse errors) and guides the agent to retry with
  corrected paths.
- Termination happens when the plan marks all review steps as `done` and the final response sets
  `agent_control.status = "completed"`.

### Deliverable

- `deliverable_factory` transforms the final snapshot into a `CodeReviewSummary` with structured findings and a concise
  verdict.
- `AgentRunResult.final_summary` mirrors the human-readable report sent to the caller, while `deliverable` exposes the
  machine-friendly variant for downstream automation.

## Prompt and Tool Wiring

`prompts.py` defines a `BackgroundAgentSection` that:

1. Emits behavioural guidance emphasising the side effect free policy and planning discipline.
1. Embeds the `PlanningToolsSection` (see `specs/PLANNING_TOOL.md`) so the LLM maintains a todo list.
1. Embeds the `VfsToolsSection`, initialised with `objective.host_mounts`, so the agent operates inside the virtual
   filesystem.
1. Registers execution tools declared by caller-supplied adapters; the orchestrator itself ships no domain tools beyond
   planning and VFS.
1. Surfaces a lightweight reflection template triggered by failures when `enable_reflection` is true.

The main prompt follows these rules:

- Always instruct the LLM to keep responses terse, ASCII, and objective-driven.
- Require the agent to maintain or update the plan before acting when no active plan exists.
- Provide a scratchpad section containing the current snapshot rendered from the latest session state.
- Remind the agent that all file access happens through VFS tools; direct host paths are unavailable.
- Return either a tool invocation payload, a plan update, or a final report. Plain chat responses are considered
  terminal outputs and must set `status="completed"` explicitly.

## Retry and Recovery

- Failed tool invocations increment a consecutive failure counter and enter the reflection phase.
- Reflection can request a retry with adjusted parameters. Retries share the same iteration budget.
- After `max_failures`, the orchestrator aborts with `status="failed"` and records a summary referencing the last error.
- Plan resets are permitted only when `allow_plan_reset` is true and the reflection phase justifies the reset.
- Soft timeouts raise a recoverable error that the reflection phase may handle; hard timeouts propagate as failures.

## Safety and Determinism

- All tool invocations must route through adapters that emit `ToolInvoked` events. Direct network or filesystem access
  from the orchestrator is forbidden.
- File operations are confined to the virtual filesystem snapshot initialised with `AgentObjective.host_mounts`; direct
  host writes are never issued.
- Every decision depends on explicit inputs (objective, snapshot, config). Hidden globals or nondeterministic sources
  (random, wall clock) are disallowed unless injected as deterministic seeds.
- Text outputs are trimmed and validated before being written to the session to keep downstream reducers stable.

## Termination Signals

The run terminates when the first of these conditions fires:

1. The LLM sets `agent_control.status = "completed"` in the final response.
1. The plan reducer marks the plan `completed` and reflection confirms no remaining work.
1. `max_iterations` iterations were executed.
1. `max_failures` consecutive action failures occurred.
1. A caller-provided cancellation event is observed (optional future extension).

Termination produces a final reflection prompt that summarises the run and writes to `final_summary`. When abandonment is
intentional (`status="abandoned"`), the summariser must provide a brief rationale.

## Telemetry

- Leverage the existing event bus (`specs/EVENTS.md`) for publishing iteration-scoped telemetry:
  - `BackgroundAgentIterationStarted`
  - `BackgroundAgentIterationCompleted`
  - `BackgroundAgentTerminated`
- Events include the iteration index, decision type, tool identifiers, and termination reason.
- Telemetry dataclasses live alongside the runner and remain optional for first release; the spec reserves the types to
  prevent future collisions.

## Testing Checklist

- Unit tests for snapshot assembly verifying selectors fetch the expected session slices.
- Control loop tests stubbing the LLM adapter to exercise completion, retry, and failure paths.
- Reflection logic tests ensuring retries happen only when reflection approves.
- Integration test that injects the planning tool, executes a multi-step scenario, and asserts plan-driven completion.
- Smoke test confirming run determinism when the same objective, config, and fake LLM responses are replayed.

## Documentation Tasks

- Extend `ROADMAP.md` with a milestone for the background agent orchestrator.
- Add a README section summarising the new agent and linking to this spec.
- Provide an example script under `examples/` that runs a toy objective using a mock tool adapter and prints the final
  summary.

## Future Extensions

- Async runner variant compatible with trio/asyncio.
- Checkpoint export/import for pausing long objectives.
- Adaptive iteration budgeting driven by plan complexity.

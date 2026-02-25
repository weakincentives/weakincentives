# Guardrails Specification

## Purpose

Guardrails ensure agents operate within constraints while preserving reasoning
autonomy. Three complementary mechanisms--tool policies, feedback providers, and
task completion checking--enforce invariants, provide guidance, and verify goals.

**Philosophy:** See `POLICIES_OVER_WORKFLOWS.md` for design rationale.

## Overview

| Mechanism | Role | Enforcement |
|-----------|------|-------------|
| Tool Policies | Gate tool invocations | Hard block (fail-closed) |
| Feedback Providers | Soft guidance over time | Advisory (agent decides) |
| Task Completion | Verify goals before stopping | Block early termination |

All three mechanisms are declared on `PromptTemplate` alongside the tools they
govern. This makes constraints a first-class part of the agent definition—
versionable, reviewable, and portable across adapters.

---

## Tool Policies

**Implementation:** `src/weakincentives/prompt/policy.py`

Policies enforce sequential dependencies between tool invocations—tool B
requires tool A first, unconditionally or keyed by parameter. Denied calls
return an error result without executing; the agent must reason about why.

### Principles

- **Prompt-scoped declaration**: Bound to prompts alongside tools
- **Session-scoped state**: Invocation history in session slices
- **Composable**: Multiple policies can govern same tool; all must allow
- **Fail-closed**: Denied calls return error without executing

### ToolPolicy Protocol

| Method/Property | Description |
|-----------------|-------------|
| `name` | Unique identifier |
| `check(tool, params, *, context)` | Returns `PolicyDecision` |
| `on_result(tool, params, result, *, context)` | Update state after success |

### PolicyDecision

| Field | Type | Description |
|-------|------|-------------|
| `allowed` | `bool` | Whether to proceed |
| `reason` | `str \| None` | Denial explanation |

### PolicyState (Session Slice)

| Field | Type | Description |
|-------|------|-------------|
| `policy_name` | `str` | Policy identifier |
| `invoked_tools` | `frozenset[str]` | Successfully invoked tools |
| `invoked_keys` | `frozenset[tuple[str, str]]` | (tool, key) pairs |

### Built-in Policies

**`SequentialDependencyPolicy`** — Unconditional ordering: tool B requires tool
A to have succeeded first. Configured with a `dependencies` dict mapping each
gated tool to the set of tools that must have succeeded before it.

**`ReadBeforeWritePolicy`** — Parameter-keyed dependency for filesystem tools.
Existing files must be read before overwritten; new files can be created freely.
Tracks read paths in session so the policy persists across tool calls.

Policies attach to sections or prompts via the `policies=` parameter on
`PromptTemplate`. See `src/weakincentives/prompt/policy.py` for full usage.

### Execution Flow

Policy enforcement happens in `BridgedTool` before handler execution.
See `TOOLS.md` (Runtime Dispatch section) for the full dispatch sequence.

### Limitations

- **Synchronous**: Policy checks run on tool execution thread
- **Session-scoped**: No cross-session persistence
- **No rollback notification**: Policies not notified on restore

---

## Feedback Providers

**Implementation:** `src/weakincentives/prompt/feedback.py`

Deliver ongoing progress feedback to agents during unattended execution. Analyze
patterns over time and inject guidance into context for soft course-correction.

Unlike tool policies that gate calls, feedback providers observe trajectory and
produce contextual feedback delivered immediately after tool execution.

### Characteristics

- **Non-blocking**: Guidance, not gates; agent decides response
- **Trigger-based**: Run when conditions met (every N calls/seconds/file created)
- **Immediate delivery**: Inject via hook response, not next render
- **Concurrent evaluation**: All matching providers run, not just first match
- **Independent trigger state**: Each provider maintains its own trigger cadence

### FeedbackTrigger

| Field | Type | Description |
|-------|------|-------------|
| `every_n_calls` | `int \| None` | Run after N tool calls since last feedback from this provider |
| `every_n_seconds` | `float \| None` | Run after N seconds since last feedback from this provider |
| `on_file_created` | `FileCreatedTrigger \| None` | Run once when file created |

Conditions are OR'd together. Each provider maintains independent trigger state.

### FileCreatedTrigger

Triggers when a specified file (`filename` field) is created on the filesystem.
Fires exactly once per session—after initial detection, subsequent tool calls
will not re-trigger even if the file is deleted and recreated. Trigger state
persists in session via `FileCreatedTriggerState` slice; reset clears it.

### Built-in Providers

**`StaticFeedbackProvider`** — Delivers a fixed feedback message. Useful with
`FileCreatedTrigger` for one-time guidance when specific files are detected.
Field: `feedback: str`.

**`DeadlineFeedback`** — Reports remaining time until deadline. Provides
increasingly urgent suggestions as the deadline approaches. Field:
`warning_threshold_seconds` (default 120s; below this threshold, suggestions
are added to the feedback).

Both at `src/weakincentives/prompt/feedback_providers.py`.

### FeedbackProvider Protocol

| Method/Property | Description |
|-----------------|-------------|
| `name` | Unique identifier |
| `should_run(context)` | Additional filtering beyond trigger |
| `provide(context)` | Produce feedback |

### Feedback Output

`Feedback.render()` produces XML-tagged output for structured context injection:

```xml
<feedback provider='Deadline'>
The work so far took 5 minutes. You have 3 minutes remaining.

-> Prioritize completing critical remaining work.
</feedback>
```

Multiple providers produce separate `<feedback>` blocks, joined by blank lines.

### FeedbackContext

| Property/Method | Description |
|-----------------|-------------|
| `session` | Session protocol |
| `prompt` | Prompt protocol |
| `deadline` | Optional deadline |
| `last_feedback` | Most recent feedback for prompt |
| `last_feedback_for_provider(name)` | Most recent feedback from specific provider |
| `tool_call_count` | Total calls for prompt |
| `tool_calls_since_last_feedback()` | Calls since last feedback (any provider) |
| `tool_calls_since_last_feedback_for_provider(name)` | Calls since last feedback from specific provider |
| `recent_tool_calls(n)` | Last N tool calls |

### Execution Flow

At `src/weakincentives/prompt/feedback.py` (`run_feedback_providers`):

1. Tool call completes, `ToolInvoked` dispatched
1. For each provider: check trigger conditions, call `provider.should_run()`
1. Collect all triggered providers, call `provider.provide()` for each
1. Store all feedback in session; mark file creation triggers as fired
1. Render and combine all feedback blocks; return combined text

**All matching providers are evaluated**—not first-match-wins.

### Adapter Integration

| Adapter | Delivery Method |
|---------|-----------------|
| Claude Agent SDK | `PostToolUse` hook `additionalContext` |
| Codex App Server | `append_feedback()` after successful tool calls in `_guardrails.py` |
| ACP / OpenCode | `post_call_hook` on MCP tool server in `_guardrails.py` |

### State Management

| Slice | Purpose |
|-------|---------|
| `ToolInvoked` | Tool invocation log |
| `Feedback` | Feedback history |
| `FileCreatedTriggerState` | Tracks which file creation triggers have fired |

When sessions are reused across prompts, feedback and counts are scoped to the
current prompt via the `prompt_name` field.

### Limitations

- **Synchronous**: Providers block tool completion briefly
- **Text-based**: Agent interprets natural language

---

## Task Completion Checking

**Implementation:** `src/weakincentives/prompt/task_completion.py`

Verify agents complete all assigned tasks before stopping. Critical for ensuring
agents don't prematurely terminate with work incomplete.

### Principles

- **Prompt-scoped declaration**: Bound to prompts alongside tools, policies,
  and feedback providers
- **Provider-agnostic**: Same checker works across all adapters
- **Composable**: Multiple checkers combine via `CompositeChecker`
- **Constraint-aware**: Checking skipped when deadline or budget exhausted

### TaskCompletionResult

Factory methods: `TaskCompletionResult.ok()`, `TaskCompletionResult.incomplete(feedback)`

### TaskCompletionContext

| Field | Type | Description |
|-------|------|-------------|
| `session` | `SessionProtocol` | Session containing state |
| `tentative_output` | `Any` | Output being produced |
| `filesystem` | `Filesystem \| None` | Optional filesystem |
| `adapter` | `ProviderAdapter \| None` | Optional adapter |
| `stop_reason` | `str \| None` | Why agent is stopping |

### Built-in Implementations

**`FileOutputChecker`** — Verifies that required output files exist on the
filesystem. Returns incomplete if any are missing. Fail-closed: if no filesystem
is available and files are required, returns incomplete (cannot verify).

Field: `files: tuple[str, ...]` — paths that must exist for completion.

**`CompositeChecker`** — Combines multiple checkers with configurable logic.
Fields: `checkers: tuple[TaskCompletionChecker, ...]` and `all_must_pass: bool`
(AND vs OR). Short-circuits on first failure/success.

Both at `src/weakincentives/prompt/task_completion.py`.

### Prompt Integration

All three guardrail mechanisms live on the prompt definition:

| Mechanism | PromptTemplate Field | Prompt Property |
|-----------|---------------------|-----------------|
| Tool Policies | `policies` | `policies_for_tool(name)` |
| Feedback Providers | `feedback_providers` | `feedback_providers` |
| Task Completion | `task_completion_checker` | `task_completion_checker` |

### Adapter Integration

Adapters read the checker from `prompt.task_completion_checker` and translate it
into their native hook/stop mechanism:

- **PostToolUse Hook**: If incomplete, adds feedback context
- **Stop Hook**: Returns `needsMoreTurns: True` if incomplete
- **Continuation Loop**: Resolves filesystem from prompt resources so
  file-based checkers can verify output during the message stream
- **Final Verification**: Logs warning if incomplete after execution

### ACK Integration

`AdapterCapabilities.task_completion` gates the `test_task_completion.py`
scenario suite in ACK. Verified behaviors include: blocking early stops when
files are missing, allowing stops when files exist, bypassing checks on deadline
or budget exhaustion, and composite checker logic (AND/OR). See `specs/ACK.md`.

### Operational Notes

- **Default disabled**: Must configure checker on prompt to enable
- **Budget/deadline bypass**: Skipped when exhausted
- **Feedback truncation**: File output checker limits to 3 file paths in message
- **Fail-closed on missing filesystem**: If no filesystem in context and files
  are required, checker returns incomplete

---

## Design Rationale

### Definition Owns Constraints

The prompt definition is the single artifact that is versioned, reviewed, tested,
and ported across adapters. All three guardrail mechanisms—policies, feedback
providers, and task completion—are inherent properties of the agent's goal.
They belong with the definition, not the harness.

### Immediate Delivery (Feedback)

No outer workflow loop; hooks inject guidance into the current turn so the agent
can self-correct without waiting for the next prompt render.

### Store If Delivered

Feedback is stored in session to enable trigger state tracking and debugging via
debug bundles. History also enables per-provider trigger cadence tracking.

### All Matching Providers

All triggered providers run and their feedback is combined, ensuring no guidance
is silently dropped when multiple conditions are met simultaneously.

### Fail-Closed (Policies)

When uncertain, deny. Agent reasons about why and adjusts. This prevents
undetected violations of invariants and makes constraint enforcement observable.

---

## Related Specifications

- `POLICIES_OVER_WORKFLOWS.md` - Design philosophy
- `TOOLS.md` - Tool runtime
- `SESSIONS.md` - Session state, snapshots
- `CLAUDE_AGENT_SDK.md` - SDK adapter integration
- `CODEX_APP_SERVER.md` - Codex adapter guardrails
- `ACP_ADAPTER.md` - ACP/OpenCode adapter guardrails
- `ACK.md` - Adapter Compatibility Kit

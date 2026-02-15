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

______________________________________________________________________

## Tool Policies

**Implementation:** `src/weakincentives/prompt/policy.py`

Policies enforce sequential dependencies between tool invocations. Declares
that tool B requires tool A first--unconditionally or keyed by parameter.

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

#### SequentialDependencyPolicy

Unconditional tool ordering: tool B requires tool A to have succeeded.

```python
policy = SequentialDependencyPolicy(
    dependencies={
        "deploy": frozenset({"test", "build"}),
        "build": frozenset({"lint"}),
    }
)
```

#### ReadBeforeWritePolicy

Parameter-keyed dependency for filesystem tools. Existing files must be read
before overwritten. New files can be created freely.

```python
policy = ReadBeforeWritePolicy()
# write_file("new.txt")      -> OK (doesn't exist)
# write_file("config.yaml")  -> DENIED (exists, not read)
# read_file("config.yaml")   -> OK (records path)
# write_file("config.yaml")  -> OK (was read)
```

### Policy Integration

```python
template = PromptTemplate(
    sections=[
        MarkdownSection(
            tools=[read_file, write_file],
            policies=[ReadBeforeWritePolicy()],
        ),
        MarkdownSection(
            tools=[lint, test, build, deploy],
            policies=[SequentialDependencyPolicy(dependencies={...})],
        ),
    ],
    policies=[...],  # Prompt-level policies
)
```

### Execution Flow

Policy enforcement happens in the adapter's tool execution hooks.
See `TOOLS.md` (Runtime Dispatch section) for the full dispatch sequence.

### Policy State Management

- **Snapshot/restore**: State captured with session snapshots
- **Reset**: `session.reset()` clears policy state
- **Isolation**: Each session has independent state

### Limitations

- **Synchronous**: Policy checks run on tool execution thread
- **Session-scoped**: No cross-session persistence
- **No rollback notification**: Policies not notified on restore

______________________________________________________________________

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

At `src/weakincentives/prompt/feedback.py`:

| Field | Type | Description |
|-------|------|-------------|
| `every_n_calls` | `int \| None` | Run after N tool calls since last feedback from this provider |
| `every_n_seconds` | `float \| None` | Run after N seconds since last feedback from this provider |
| `on_file_created` | `FileCreatedTrigger \| None` | Run once when file created |

Conditions are OR'd together. Trigger state is tracked per-provider to ensure
each provider maintains independent trigger cadences.

### FileCreatedTrigger

At `src/weakincentives/prompt/feedback.py`:

Triggers when a specified file is created on the filesystem. Fires exactly once
per session--after initial detection, subsequent tool calls will not re-trigger
even if the file is deleted and recreated.

| Field | Type | Description |
|-------|------|-------------|
| `filename` | `str` | Path to watch for creation |

#### Behavior

1. After tool execution completes, check if `filename` exists
1. If file exists and trigger has not fired -> fire, mark as fired
1. If file does not exist or trigger already fired -> skip
1. Trigger state persists in session via `FileCreatedTriggerState` slice; reset clears it

### StaticFeedbackProvider

At `src/weakincentives/prompt/feedback_providers.py`:

A built-in provider that delivers a fixed feedback message. Useful with
`FileCreatedTrigger` for one-time guidance when specific files are detected.

| Field | Type | Description |
|-------|------|-------------|
| `feedback` | `str` | Feedback content to deliver |

```python
config = FeedbackProviderConfig(
    provider=StaticFeedbackProvider(
        feedback="AGENTS.md detected. Follow the conventions defined within.",
    ),
    trigger=FeedbackTrigger(
        on_file_created=FileCreatedTrigger(filename="AGENTS.md"),
    ),
)
```

### FeedbackProviderConfig

```python
template = PromptTemplate(
    ...,
    feedback_providers=(
        FeedbackProviderConfig(
            provider=DeadlineFeedback(),
            trigger=FeedbackTrigger(every_n_seconds=30),
        ),
    ),
)
```

### FeedbackProvider Protocol

| Method/Property | Description |
|-----------------|-------------|
| `name` | Unique identifier |
| `should_run(context)` | Additional filtering beyond trigger |
| `provide(context)` | Produce feedback |

### Feedback

At `src/weakincentives/prompt/feedback.py`:

| Field | Type | Description |
|-------|------|-------------|
| `provider_name` | `str` | Source provider |
| `summary` | `str` | Main message |
| `observations` | `tuple[Observation, ...]` | Detailed observations |
| `suggestions` | `tuple[str, ...]` | Recommendations |
| `severity` | `Literal["info", "caution", "warning"]` | Urgency level |
| `timestamp` | `datetime` | When produced |
| `call_index` | `int` | Tool call count when produced |
| `prompt_name` | `str` | Prompt that produced this feedback |

### XML-Style Feedback Rendering

`Feedback.render()` produces XML-tagged output for structured context injection:

```xml
<feedback provider='Deadline'>
The work so far took 5 minutes. You have 3 minutes remaining.

-> Prioritize completing critical remaining work.
</feedback>
```

Multiple providers produce separate `<feedback>` blocks, joined by blank lines.

### FeedbackContext

At `src/weakincentives/prompt/feedback.py`:

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

1. Tool call completes
1. `ToolInvoked` dispatched
1. For each configured provider:
   a. Check trigger conditions (using provider-scoped state)
   b. Call `provider.should_run()`
1. Collect all triggered providers
1. Call `provider.provide()` for each
1. Store all feedback in session
1. Mark file creation triggers as fired
1. Render and combine all feedback blocks
1. Return combined text

**All matching providers are evaluated concurrently** (not first-match-wins).

### Adapter Integration

| Adapter | Delivery Method |
|---------|-----------------|
| Claude Agent SDK | `PostToolUse` hook `additionalContext` |

### Built-in Provider: DeadlineFeedback

At `src/weakincentives/prompt/feedback_providers.py`:

Reports remaining time until deadline.

```python
config = FeedbackProviderConfig(
    provider=DeadlineFeedback(warning_threshold_seconds=120),
    trigger=FeedbackTrigger(every_n_seconds=30),
)
```

Output varies by remaining time. Below threshold adds suggestions.

### State Management

| Slice | Purpose |
|-------|---------|
| `ToolInvoked` | Tool invocation log |
| `Feedback` | Feedback history |
| `FileCreatedTriggerState` | Tracks which file creation triggers have fired |

Feedback stored via `session.dispatch(feedback)` with `append_all` reducer.

When sessions reused across prompts, feedback/counts scoped to current prompt
via `prompt_name` field.

### Public API

```python
from weakincentives.prompt import (
    DeadlineFeedback,        # Built-in provider
    Feedback,                # Dataclass
    FeedbackContext,         # Context
    FeedbackProvider,        # Protocol
    FeedbackProviderConfig,  # Config
    FeedbackTrigger,         # Trigger
    FileCreatedTrigger,      # File creation trigger
    StaticFeedbackProvider,  # Built-in provider
    collect_feedback,        # Primary entry point
)
```

### Limitations

- **Synchronous**: Providers block tool completion briefly
- **Text-based**: Agent interprets natural language

______________________________________________________________________

## Task Completion Checking

**Current implementation:** `src/weakincentives/adapters/claude_agent_sdk/_task_completion.py`

**Target implementation:** `prompt/task_completion.py` (under `src/weakincentives/`)

Verify agents complete all assigned tasks before stopping. Critical for ensuring
agents don't prematurely terminate with work incomplete.

### Motivation for Relocation

Task completion checking is currently configured on the adapter
(`ClaudeAgentSDKClientConfig.task_completion_checker`). This is architecturally
inconsistent with how tool policies and feedback providers are handled: both are
declared on the prompt (definition), not the adapter (harness). The asymmetry
creates three problems:

1. **Definition/harness coupling.** The same agent definition cannot carry its
   completion criteria across adapters. Users must reconfigure task completion
   when switching from Claude Agent SDK to Codex, ACP, or any future adapter.
   This violates the "definition vs harness" separation that is core to WINK.

1. **Incomplete prompt portability.** A `PromptTemplate` captures tools,
   policies, and feedback providers — everything an agent *is* — except for task
   completion. The prompt is not self-describing without the adapter config.

1. **ACK blind spot.** The Adapter Compatibility Kit validates adapter behavior
   against a shared contract. Task completion enforcement is adapter-specific
   behavior that cannot be tested via ACK because the checker lives in the
   adapter config, not in the prompt that ACK scenarios construct.

### Design: Prompt-Scoped Task Completion

Move the `TaskCompletionChecker` declaration from adapter config to
`PromptTemplate`, mirroring the pattern used by `feedback_providers`.

#### New PromptTemplate Field

```python
@FrozenDataclass(slots=False)
class PromptTemplate[OutputT]:
    ns: str
    key: str
    sections: ...
    policies: Sequence[ToolPolicy] = ()
    feedback_providers: Sequence[FeedbackProviderConfig] = ()
    task_completion_checker: TaskCompletionChecker | None = None  # NEW
    ...
```

#### Prompt Instance Forwarding

```python
class Prompt[OutputT]:
    @property
    def task_completion_checker(self) -> TaskCompletionChecker | None:
        """Return task completion checker configured on this prompt."""
        return self.template.task_completion_checker
```

#### Protocol Update

```python
class PromptProtocol[PromptOutputT](Protocol):
    ...
    @property
    def task_completion_checker(self) -> object | None:
        """Return task completion checker if configured."""
        ...
```

### Module Relocation

Move the following types from
`src/weakincentives/adapters/claude_agent_sdk/_task_completion.py` to
a new `task_completion` module under `prompt/`:

| Type | Description |
|------|-------------|
| `TaskCompletionResult` | Frozen dataclass for check results |
| `TaskCompletionContext` | Mutable context for checkers |
| `TaskCompletionChecker` | Runtime-checkable protocol |
| `PlanBasedChecker` | Built-in plan-based implementation |
| `CompositeChecker` | Built-in composite implementation |

The old adapter module re-exports these types for backward compatibility during
transition. The re-exports are removed in the same release (alpha software; no
backward-compatibility shims per project policy).

### Public API

After relocation, the canonical import path is `weakincentives.prompt`:

```python
from weakincentives.prompt import (
    TaskCompletionChecker,    # Protocol
    TaskCompletionContext,    # Context dataclass
    TaskCompletionResult,     # Result dataclass
    PlanBasedChecker,         # Built-in implementation
    CompositeChecker,         # Built-in composition
)
```

### TaskCompletionResult

| Field | Type | Description |
|-------|------|-------------|
| `complete` | `bool` | Whether tasks are complete |
| `feedback` | `str \| None` | Explanation for incomplete |

Factory methods: `TaskCompletionResult.ok()`, `TaskCompletionResult.incomplete(feedback)`

### TaskCompletionContext

| Field | Type | Description |
|-------|------|-------------|
| `session` | `SessionProtocol` | Session containing state |
| `tentative_output` | `Any` | Output being produced |
| `filesystem` | `Filesystem \| None` | Optional filesystem |
| `adapter` | `ProviderAdapter \| None` | Optional adapter |
| `stop_reason` | `str \| None` | Why agent is stopping |

### TaskCompletionChecker Protocol

```python
@runtime_checkable
class TaskCompletionChecker(Protocol):
    def check(self, context: TaskCompletionContext) -> TaskCompletionResult: ...
```

### Built-in Implementations

#### PlanBasedChecker

Checks session `Plan` state for incomplete steps.

```python
checker = PlanBasedChecker(plan_type=Plan)
```

Returns incomplete if plan steps exist with `status != "done"`.

#### CompositeChecker

Combines multiple checkers with configurable logic.

```python
# All must pass
checker = CompositeChecker(
    checkers=(PlanBasedChecker(plan_type=Plan), FileExistsChecker(("output.txt",))),
    all_must_pass=True,
)

# Any can pass
checker = CompositeChecker(checkers=(...), all_must_pass=False)
```

### Prompt Integration (New)

Configure via `PromptTemplate`:

```python
template = PromptTemplate(
    ns="my-agent",
    key="main",
    sections=[...],
    policies=[ReadBeforeWritePolicy()],
    feedback_providers=(
        FeedbackProviderConfig(
            provider=DeadlineFeedback(),
            trigger=FeedbackTrigger(every_n_seconds=30),
        ),
    ),
    task_completion_checker=PlanBasedChecker(plan_type=Plan),  # NEW
)
```

All three guardrail mechanisms now live on the prompt definition:

| Mechanism | PromptTemplate Field | Prompt Property |
|-----------|---------------------|-----------------|
| Tool Policies | `policies` | `policies_for_tool(name)` |
| Feedback Providers | `feedback_providers` | `feedback_providers` |
| Task Completion | `task_completion_checker` | `task_completion_checker` |

### Adapter Integration

Adapters read the checker from the prompt, not from their own config.

#### Resolution Order

When evaluating a prompt, the adapter resolves the checker as follows:

1. `prompt.task_completion_checker` (prompt-scoped, preferred)
1. `client_config.task_completion_checker` (adapter-scoped, deprecated fallback)

If both are set, the prompt-scoped checker wins. A deprecation warning is logged
when the adapter-scoped checker is used and the prompt does not declare one.
When `client_config.task_completion_checker` is set but
`prompt.task_completion_checker` is also set, the prompt wins silently (no
conflict warning — the prompt is the source of truth).

#### Adapter Checker Resolution (Pseudocode)

```python
def _resolve_checker(
    prompt: PromptProtocol,
    client_config: ClaudeAgentSDKClientConfig,
) -> TaskCompletionChecker | None:
    """Resolve task completion checker with prompt-first precedence."""
    prompt_checker = getattr(prompt, "task_completion_checker", None)
    if prompt_checker is not None:
        return prompt_checker

    adapter_checker = client_config.task_completion_checker
    if adapter_checker is not None:
        logger.warning(
            "task_completion.deprecated_adapter_config",
            event="task_completion.deprecated_adapter_config",
            context={"prompt_name": prompt.name},
        )
    return adapter_checker
```

#### Hook Integration

Adapters translate the prompt-declared checker into their hook mechanism.
The hooks themselves are unchanged — only the source of the checker changes.

- **PostToolUse Hook (StructuredOutput)**: If incomplete, adds feedback context
- **Stop Hook**: Returns `needsMoreTurns: True` if incomplete
- **Final Verification**: Logs warning if incomplete after execution

#### Claude Agent SDK

The existing `build_hooks_config` function changes from:

```python
checker = client_config.task_completion_checker
```

to:

```python
checker = _resolve_checker(prompt, client_config)
```

All other hook mechanics remain identical.

#### Other Adapters

For Codex, ACP, and OpenCode adapters, the checker is read from the prompt and
translated into the adapter's stop/continuation mechanism. The adapter-agnostic
protocol ensures each adapter can implement enforcement using its native
primitives.

### Deprecation of Adapter-Scoped Checker

`ClaudeAgentSDKClientConfig.task_completion_checker` is deprecated and will be
removed. The migration path is straightforward — move the checker from the
adapter config to the prompt template.

**Before (deprecated):**

```python
adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        task_completion_checker=PlanBasedChecker(plan_type=Plan),
    ),
)
```

**After:**

```python
template = PromptTemplate(
    ...,
    task_completion_checker=PlanBasedChecker(plan_type=Plan),
)

adapter = ClaudeAgentSDKAdapter()  # No checker config needed
```

### ACK Integration

With the checker on the prompt, ACK scenarios can construct prompts with task
completion checkers and verify enforcement across all adapters.

#### New ACK Capability

Add `task_completion` to `AdapterCapabilities`:

```python
@dataclass(slots=True, frozen=True)
class AdapterCapabilities:
    ...
    # Tier 3: Advanced
    task_completion: bool = True  # NEW
    ...
```

#### New ACK Scenario: `test_task_completion.py`

```
test_task_completion_blocks_early_stop
    Given a prompt with a PlanBasedChecker and a plan tool
    And the plan has incomplete steps
    When adapter.evaluate() is called
    Then the agent does not stop until plan steps are marked done
    Or budget/deadline is exhausted

test_task_completion_allows_stop_when_complete
    Given a prompt with a PlanBasedChecker
    And all plan steps are marked done
    When the agent attempts to stop
    Then the stop is allowed

test_task_completion_skipped_on_deadline_exhaustion
    Given a prompt with a PlanBasedChecker and a near-past deadline
    When adapter.evaluate() is called
    Then task completion checking is bypassed
    And the agent stops despite incomplete tasks

test_task_completion_skipped_on_budget_exhaustion
    Given a prompt with a PlanBasedChecker and an exhausted budget
    When adapter.evaluate() is called
    Then task completion checking is bypassed

test_composite_checker_all_must_pass
    Given a prompt with a CompositeChecker (all_must_pass=True)
    And two checkers where one returns incomplete
    When the agent attempts to stop
    Then the stop is blocked with feedback from the failing checker

test_no_checker_allows_free_stop
    Given a prompt with no task_completion_checker
    When the agent produces output and attempts to stop
    Then the stop is allowed without any completion verification
```

#### Prompt Builder

```python
# integration-tests/ack/scenarios/__init__.py

def build_task_completion_prompt(
    ns: str,
    *,
    checker: TaskCompletionChecker,
) -> tuple[PromptTemplate[object], Tool]:
    """Build a prompt with task completion checking and a plan tool."""
    ...
```

### Unit Test Coverage

Existing unit tests in `tests/adapters/claude_agent_sdk/test_task_completion.py`
and `tests/adapters/claude_agent_sdk/test_verify_task_completion.py` are
relocated to a new `test_task_completion` module under `tests/prompt/`. The tests exercise the
protocol, `PlanBasedChecker`, and `CompositeChecker` in isolation (no adapter
dependency). Adapter-specific hook tests remain in their current location but
are updated to resolve the checker from the prompt.

### Operational Notes

- **Default disabled**: Must configure checker on prompt to enable
- **Budget/deadline bypass**: Skipped when exhausted (unchanged)
- **Feedback truncation**: Plan checker limits to 3 task titles (unchanged)
- **Adapter fallback**: Adapter-scoped config still works but logs deprecation

______________________________________________________________________

## Design Rationale

### Definition Owns Constraints (Task Completion)

The prompt definition is the single artifact that is versioned, reviewed, tested,
and ported across adapters. Task completion criteria are an inherent property of
the agent's goal — they belong with the definition, not the harness.

The precedent is clear: tool policies and feedback providers already live on the
prompt. Task completion was the sole exception, creating an asymmetry that made
agent definitions incomplete without adapter-specific configuration.

### Immediate Delivery (Feedback)

No outer workflow; hooks inject into current turn.

### Store If Delivered

Need history for trigger state and debugging.

### All Matching Providers

All triggered providers run and their feedback is combined, ensuring no guidance
is silently dropped when multiple conditions are met simultaneously.

### No Escalation

Budget provides backstop; feedback is soft guidance.

### Fail-Closed (Policies)

When uncertain, deny. Agent reasons about why and adjusts.

### Observable and Debuggable

Expose reasoning. Denial feedback enables self-correction.

### ACK as Cross-Adapter Verification

By placing the checker on the prompt, ACK can construct task completion scenarios
once and verify enforcement across every adapter. This eliminates the current gap
where task completion enforcement is tested only against the Claude Agent SDK.

______________________________________________________________________

## Migration Summary

### What Moves

| From | To |
|------|-----|
| `adapters/claude_agent_sdk/_task_completion.py` | `prompt/` (new `task_completion` module) |
| `ClaudeAgentSDKClientConfig.task_completion_checker` | `PromptTemplate.task_completion_checker` |
| `tests/adapters/claude_agent_sdk/test_task_completion.py` | `tests/prompt/` (new `test_task_completion` module) |

### What Stays

| Location | Reason |
|----------|--------|
| Hook implementations in `adapters/claude_agent_sdk/_hooks.py` | Adapter-specific translation of checker into SDK hooks |
| `verify_task_completion` in `_result_extraction.py` | Adapter-specific final verification call |
| Adapter-specific hook tests | Test hook behavior, not checker logic |

### What Gets Added

| Location | Content |
|----------|---------|
| `PromptTemplate.task_completion_checker` field | New optional field |
| `Prompt.task_completion_checker` property | Forwards from template |
| `PromptProtocol.task_completion_checker` property | Protocol surface |
| `integration-tests/ack/scenarios/test_task_completion.py` | ACK scenario |
| `AdapterCapabilities.task_completion` | New capability flag |

______________________________________________________________________

## Related Specifications

- `POLICIES_OVER_WORKFLOWS.md` - Design philosophy
- `TOOLS.md` - Tool runtime
- `SESSIONS.md` - Session state, snapshots
- `CLAUDE_AGENT_SDK.md` - SDK adapter integration
- `ACK.md` - Adapter Compatibility Kit

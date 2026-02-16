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

**Implementation:** `weakincentives.prompt.task_completion`

Verify agents complete all assigned tasks before stopping. Critical for ensuring
agents don't prematurely terminate with work incomplete.

### Principles

- **Prompt-scoped declaration**: Bound to prompts alongside tools, policies,
  and feedback providers
- **Provider-agnostic**: Same checker works across all adapters
- **Composable**: Multiple checkers combine via `CompositeChecker`
- **Constraint-aware**: Checking skipped when deadline or budget exhausted

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

#### FileOutputChecker

Verifies that required output files exist on the filesystem. Accepts a list of
file paths; returns incomplete if any are missing.

```python
checker = FileOutputChecker(files=("report.md", "results.json"))
```

| Field | Type | Description |
|-------|------|-------------|
| `files` | `tuple[str, ...]` | Paths that must exist for completion |

**Behavior:**

1. Retrieve `filesystem` from `TaskCompletionContext`
1. If no filesystem available and files are required, return incomplete
   (fail-closed: cannot verify without filesystem access)
1. For each path in `files`, call `filesystem.exists(path)`
1. If all exist, return `TaskCompletionResult.ok()`
1. If any missing, return `TaskCompletionResult.incomplete(feedback)` listing
   the missing files

All file existence checks go through the `Filesystem` abstraction.
`HostFilesystem` resolves both relative and absolute paths (under its root),
so callers can pass absolute workspace paths without bypassing the abstraction.

```python
class FileOutputChecker(TaskCompletionChecker):
    def __init__(self, files: tuple[str, ...]) -> None:
        self._files = files

    def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
        if context.filesystem is None:
            if not self._files:
                return TaskCompletionResult.ok(...)
            return TaskCompletionResult.incomplete(...)

        missing = [f for f in self._files if not context.filesystem.exists(f)]
        if not missing:
            return TaskCompletionResult.ok(...)

        return TaskCompletionResult.incomplete(...)
```

#### CompositeChecker

Combines multiple checkers with configurable logic.

```python
# All must pass (default)
checker = CompositeChecker(
    checkers=(
        FileOutputChecker(files=("output.txt", "summary.md")),
        MyCustomChecker(),
    ),
    all_must_pass=True,
)

# Any can pass
checker = CompositeChecker(checkers=(...), all_must_pass=False)
```

| Field | Type | Description |
|-------|------|-------------|
| `checkers` | `tuple[TaskCompletionChecker, ...]` | Ordered checker sequence |
| `all_must_pass` | `bool` | AND (True) vs OR (False) logic |

Short-circuits: first failure stops evaluation when `all_must_pass=True`;
first success stops evaluation when `all_must_pass=False`.

### Prompt Integration

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
    task_completion_checker=FileOutputChecker(
        files=("report.md", "results.json"),
    ),
)
```

All three guardrail mechanisms live on the prompt definition:

| Mechanism | PromptTemplate Field | Prompt Property |
|-----------|---------------------|-----------------|
| Tool Policies | `policies` | `policies_for_tool(name)` |
| Feedback Providers | `feedback_providers` | `feedback_providers` |
| Task Completion | `task_completion_checker` | `task_completion_checker` |

#### PromptTemplate Field

```python
@FrozenDataclass(slots=False)
class PromptTemplate[OutputT]:
    ns: str
    key: str
    sections: ...
    policies: Sequence[ToolPolicy] = ()
    feedback_providers: Sequence[FeedbackProviderConfig] = ()
    task_completion_checker: TaskCompletionChecker | None = None
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
    def task_completion_checker(self) -> TaskCompletionChecker | None:
        """Return task completion checker if configured."""
        ...
```

### Adapter Integration

Adapters read the checker from `prompt.task_completion_checker` and translate it
into their native hook/stop mechanism.

#### Hook Integration

- **PostToolUse Hook (StructuredOutput)**: If incomplete, adds feedback context
- **Stop Hook**: Returns `needsMoreTurns: True` if incomplete
- **Continuation Loop**: Resolves filesystem from prompt resources so
  file-based checkers can verify output during the message stream
- **Final Verification**: Logs warning if incomplete after execution

#### Claude Agent SDK

The adapter resolves the checker from the prompt via `resolve_checker()`:

```python
checker = resolve_checker(prompt=prompt)
```

The checker is declared on `PromptTemplate.task_completion_checker`. There is
no adapter-level fallback; the prompt is the single source of truth.

#### Other Adapters

For Codex, ACP, and OpenCode adapters, the checker is read from the prompt and
translated into the adapter's stop/continuation mechanism. The adapter-agnostic
protocol ensures each adapter can implement enforcement using its native
primitives.

### ACK Integration

`AdapterCapabilities` includes a `task_completion` flag:

```python
@dataclass(slots=True, frozen=True)
class AdapterCapabilities:
    ...
    # Tier 3: Advanced
    task_completion: bool = True
    ...
```

#### ACK Scenario: `test_task_completion.py`

```
test_task_completion_blocks_early_stop
    Given a prompt with a FileOutputChecker requiring "output.txt"
    And the agent has not yet created the file
    When the agent attempts to stop
    Then the stop is blocked with feedback listing missing files

test_task_completion_allows_stop_when_complete
    Given a prompt with a FileOutputChecker requiring "output.txt"
    And the file exists on the filesystem
    When the agent attempts to stop
    Then the stop is allowed

test_task_completion_skipped_on_deadline_exhaustion
    Given a prompt with a FileOutputChecker and a near-past deadline
    When adapter.evaluate() is called
    Then task completion checking is bypassed
    And the agent stops despite missing output files

test_task_completion_skipped_on_budget_exhaustion
    Given a prompt with a FileOutputChecker and an exhausted budget
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
    """Build a prompt with task completion checking and a file-writing tool."""
    ...
```

### Public API

```python
from weakincentives.prompt import (
    TaskCompletionChecker,    # Protocol
    TaskCompletionContext,    # Context dataclass
    TaskCompletionResult,     # Result dataclass
    FileOutputChecker,        # Built-in: required file existence
    CompositeChecker,         # Built-in: combine multiple checkers
)
```

### Operational Notes

- **Default disabled**: Must configure checker on prompt to enable
- **Budget/deadline bypass**: Skipped when exhausted
- **Feedback truncation**: File output checker limits to 3 file paths in message
- **Fail-closed on missing filesystem**: If no filesystem in context and files
  are required, checker returns incomplete (consistent with tool policy
  fail-closed philosophy)

______________________________________________________________________

## Design Rationale

### Definition Owns Constraints

The prompt definition is the single artifact that is versioned, reviewed, tested,
and ported across adapters. All three guardrail mechanisms -- policies, feedback
providers, and task completion -- are inherent properties of the agent's goal.
They belong with the definition, not the harness.

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

With the checker on the prompt, ACK constructs task completion scenarios once and
verifies enforcement across every adapter.

______________________________________________________________________

## Related Specifications

- `POLICIES_OVER_WORKFLOWS.md` - Design philosophy
- `TOOLS.md` - Tool runtime
- `SESSIONS.md` - Session state, snapshots
- `CLAUDE_AGENT_SDK.md` - SDK adapter integration
- `ACK.md` - Adapter Compatibility Kit

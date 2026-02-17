# Guardrails

*Canonical specs:
[FEEDBACK_PROVIDERS](../specs/FEEDBACK_PROVIDERS.md),
[TASK_COMPLETION](../specs/TASK_COMPLETION.md),
[LEASE_EXTENDER](../specs/LEASE_EXTENDER.md)*

WINK provides three guardrail mechanisms that keep agents on track during
long-running tasks:

| Guardrail | Purpose | Enforcement |
| --- | --- | --- |
| **Feedback providers** | Periodic guidance (time, progress) | Soft — agent decides how to respond |
| **Task completion checkers** | Prevent premature termination | Hard — agent cannot stop until checks pass |
| **Lease extension** | Keep message visibility alive | Automatic — tied to heartbeats |

These mechanisms are complementary. Use them together for production agents
that must finish reliably within resource constraints.

______________________________________________________________________

## Feedback Providers

Feedback providers deliver periodic guidance to agents. Unlike hard
constraints, feedback is soft—the agent receives information and decides what
to do with it.

### When Feedback Helps

Agents working on multi-step tasks can lose track of time, repeat unsuccessful
patterns, or miss context changes. Feedback addresses this by:

- Alerting agents to remaining time before deadlines
- Noting patterns in tool usage (too many calls, repeated failures)
- Surfacing context changes the agent might have missed

### Core Types

The `Feedback` dataclass carries structured information:

```python nocheck
from weakincentives.prompt import Feedback, Observation

feedback = Feedback(
    provider_name="DeadlineFeedback",
    summary="You have 5 minutes remaining to complete the task.",
    observations=(
        Observation(label="elapsed", value="25 minutes"),
        Observation(label="remaining", value="5 minutes"),
    ),
    suggestions=("Consider wrapping up current work.",),
    severity="warning",  # "info", "caution", or "warning"
)
```

`FeedbackTrigger` controls when providers run:

```python nocheck
from weakincentives.prompt import FeedbackTrigger

# Run every 5 tool calls OR every 30 seconds (whichever comes first)
trigger = FeedbackTrigger(every_n_calls=5, every_n_seconds=30)
```

Trigger conditions are OR'd—either condition can fire.

### Built-in: DeadlineFeedback

`DeadlineFeedback` reports remaining time with escalating urgency:

```python nocheck
from weakincentives.prompt import DeadlineFeedback

provider = DeadlineFeedback(warning_threshold_seconds=120)
```

- Returns "info" severity when time is plentiful
- Escalates to "warning" with suggestions when below threshold
- Provides special messaging when deadline has passed

### Configuring on a Prompt

Attach feedback providers to `PromptTemplate`:

```python nocheck
from weakincentives.prompt import (
    PromptTemplate,
    FeedbackProviderConfig,
    FeedbackTrigger,
    DeadlineFeedback,
)

template = PromptTemplate[MyOutput](
    ns="my-namespace",
    key="my-prompt",
    sections=(task_section, tools_section),
    feedback_providers=(
        FeedbackProviderConfig(
            provider=DeadlineFeedback(warning_threshold_seconds=60),
            trigger=FeedbackTrigger(every_n_seconds=30, every_n_calls=5),
        ),
    ),
)
```

### Writing Custom Providers

Implement the `FeedbackProvider` protocol:

```python nocheck
from dataclasses import dataclass
from weakincentives.prompt import Feedback, FeedbackContext, FeedbackProvider


@dataclass(frozen=True)
class ToolUsageMonitor:
    """Alerts when tool call count exceeds threshold."""

    max_calls_without_progress: int = 20

    @property
    def name(self) -> str:
        return "ToolUsageMonitor"

    def should_run(self, *, context: FeedbackContext) -> bool:
        return True

    def provide(self, *, context: FeedbackContext) -> Feedback:
        count = context.tool_call_count
        if count > self.max_calls_without_progress:
            return Feedback(
                provider_name=self.name,
                summary=f"You have made {count} tool calls without completing.",
                suggestions=(
                    "Review your approach.",
                    "Consider whether you're making progress.",
                ),
                severity="caution",
            )
        return Feedback(
            provider_name=self.name,
            summary=f"Tool calls: {count}",
            severity="info",
        )
```

**FeedbackContext** provides:

- `session`: Current session state
- `prompt`: The prompt being executed
- `deadline`: Optional deadline (if configured)
- `last_feedback`: Most recent feedback for this prompt
- `tool_call_count`: Total tool calls for this prompt
- `tool_calls_since_last_feedback()`: Calls since last feedback
- `recent_tool_calls(n)`: Last N tool invocations

### Execution Flow

1. Tool completes execution
1. `ToolInvoked` event dispatched to session
1. Trigger conditions evaluated
1. `provider.should_run()` called for additional filtering
1. `provider.provide()` called if both conditions pass
1. Feedback stored in session and delivered to agent

First matching provider wins—subsequent providers are skipped for that tool
call, preventing feedback overload.

### Adapter Delivery

| Adapter | Delivery mechanism |
| --- | --- |
| Claude Agent SDK | `PostToolUse` hook's `additionalContext` field |
| OpenAI / LiteLLM | Appended to tool result message |

### Prompt Scoping

Feedback is scoped to the prompt that triggered it. Trigger counters (calls
since last feedback, time since last feedback) reset when switching prompts.
This enables safe session reuse across multiple prompt evaluations.

______________________________________________________________________

## Task Completion Checking

Task completion checking prevents agents from stopping prematurely. When an
agent signals it wants to end—by producing structured output or ending its
turn—checkers verify that all required work is actually done.

### The Problem

Agents sometimes declare victory too early:

- Producing output before all plan steps are complete
- Ending turns when required files haven't been created
- Missing edge cases they were supposed to handle

### Core Types

```python nocheck
from weakincentives.adapters.claude_agent_sdk import (
    TaskCompletionResult,
    TaskCompletionContext,
)

# Task is complete
result = TaskCompletionResult.ok()

# Task is incomplete with feedback
result = TaskCompletionResult.incomplete(
    "Steps 'run tests' and 'update docs' are not yet done."
)
```

The `TaskCompletionChecker` protocol requires a single method:

```python nocheck
from typing import Protocol, runtime_checkable


@runtime_checkable
class TaskCompletionChecker(Protocol):
    def check(self, context: TaskCompletionContext) -> TaskCompletionResult: ...
```

### Built-in Checkers

**PlanBasedChecker** verifies all steps in a `Plan` are marked "done":

```python nocheck
from weakincentives.adapters.claude_agent_sdk import PlanBasedChecker
from weakincentives.contrib.tools.planning import Plan

checker = PlanBasedChecker(plan_type=Plan)
```

Graceful degradation: when no plan type is configured, no plan slice exists, or
the plan hasn't been initialized, the checker returns complete. It only enforces
when a plan actually exists.

**CompositeChecker** combines multiple checkers:

```python nocheck
from weakincentives.adapters.claude_agent_sdk import CompositeChecker

checker = CompositeChecker(
    checkers=(plan_checker, file_checker, test_checker),
    all_must_pass=True,  # Default: all must pass; short-circuits on first failure
)
```

Set `all_must_pass=False` to short-circuit on the first success instead.

### Configuration

Enable via `ClaudeAgentSDKClientConfig`:

```python nocheck
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    PlanBasedChecker,
)
from weakincentives.contrib.tools.planning import Plan

adapter = ClaudeAgentSDKAdapter(
    model="claude-opus-4-5-20251101",
    client_config=ClaudeAgentSDKClientConfig(
        permission_mode="bypassPermissions",
        cwd="/workspace",
        task_completion_checker=PlanBasedChecker(plan_type=Plan),
    ),
)
```

### Writing Custom Checkers

```python nocheck
from dataclasses import dataclass
from weakincentives.adapters.claude_agent_sdk import (
    TaskCompletionContext,
    TaskCompletionResult,
)


@dataclass(frozen=True)
class RequiredFilesChecker:
    """Verifies required files exist."""

    required_files: tuple[str, ...]

    def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
        if context.filesystem is None:
            return TaskCompletionResult.ok()

        missing = [
            f for f in self.required_files
            if not context.filesystem.exists(f)
        ]

        if missing:
            return TaskCompletionResult.incomplete(
                f"Missing required files: {', '.join(missing)}"
            )
        return TaskCompletionResult.ok()
```

Combine custom and built-in checkers:

```python nocheck
checker = CompositeChecker(
    checkers=(
        PlanBasedChecker(plan_type=Plan),
        RequiredFilesChecker(("README.md", "tests/test_main.py")),
    ),
    all_must_pass=True,
)
```

### Enforcement Points

**Stop hook** — When the agent attempts to end its turn, the checker runs. If
incomplete, the agent receives feedback and continues.

**Final verification** — After the SDK query completes, the adapter runs
`_verify_task_completion`. If structured output exists but tasks are incomplete,
a `PromptEvaluationError` is raised.

**Resource exhaustion** — Checkers are skipped when deadline or budget is
exhausted. Partial output is acceptable when resources are depleted.

### Feedback Behavior

When tasks are incomplete, the checker returns feedback that is delivered to
the agent, which then continues working. Long task lists are truncated to 3
titles + "..." to prevent noisy feedback.

______________________________________________________________________

## Lease Extension

When processing messages from a queue, you need to prevent visibility timeout
during long-running operations. Lease extension ties visibility renewal to
proof-of-work: if the worker is actively processing, the lease extends; if
stuck, the lease expires and the message becomes available for reprocessing.

### The Problem

Message queues use visibility timeouts to handle worker failures:

1. Worker receives message, message becomes invisible
1. Worker processes message
1. Worker deletes message on success
1. If worker fails, visibility timeout expires and message reappears

For long-running agent tasks (10+ minutes), naive approaches have problems:

| Approach | Problem |
| --- | --- |
| Very long initial timeout | Stuck workers block messages for too long |
| Fixed-interval daemon | Keeps extending even when worker is stuck |
| Manual extension calls | Easy to forget, clutters handler code |

### The Heartbeat Solution

WINK uses heartbeat-based extension:

1. Tool handlers call `context.beat()` during long operations
1. Each beat potentially extends the message lease
1. No beats = no extensions = lease expires naturally

### Core Types

```python nocheck
from weakincentives.runtime import LeaseExtenderConfig

config = LeaseExtenderConfig(
    interval=60.0,     # Minimum seconds between extensions
    extension=300,     # Visibility timeout per extension (seconds)
    enabled=True,      # Enable automatic extension
)
```

Attach the extender to a message during processing:

```python nocheck
from weakincentives.runtime import LeaseExtender, Heartbeat

extender = LeaseExtender(config=LeaseExtenderConfig(interval=60, extension=300))
heartbeat = Heartbeat()

with extender.attach(msg, heartbeat):
    # Processing happens here
    # Heartbeats from tool execution extend the lease
    pass
```

### Heartbeat Propagation

Heartbeats flow from tool handlers up to the lease extender:

```
MainLoop._handle_message()
  └─ lease_extender.attach(msg, heartbeat)
  └─ _execute(heartbeat=heartbeat)
       └─ adapter.evaluate(heartbeat=heartbeat)
            └─ ToolExecutor(heartbeat=heartbeat)
                 └─ ToolContext(heartbeat=heartbeat)
                      └─ handler calls context.beat()
```

When a tool handler calls `context.beat()`:

1. Heartbeat records the beat time
1. All registered callbacks are invoked
1. LeaseExtender's callback checks if extension is needed
1. If `interval` has elapsed, visibility is extended

### Using Heartbeats in Tool Handlers

For long-running operations, call `context.beat()` periodically:

```python nocheck
from weakincentives.prompt import ToolContext, ToolResult


def process_large_dataset(
    params: ProcessParams,
    *,
    context: ToolContext,
) -> ToolResult[ProcessResult]:
    results = []
    for i, item in enumerate(params.items):
        result = process_item(item)
        results.append(result)

        # Beat every 100 items to prove liveness
        if i % 100 == 0:
            context.beat()

    return ToolResult.ok(ProcessResult(results=results))
```

**When to beat:** during iteration over large collections, between phases of
multi-step operations, after completing significant chunks of work.

**When not to beat:** on every loop iteration (too frequent), in short
operations (unnecessary).

### Configuration in MainLoop

```python nocheck
from weakincentives.runtime import MainLoop, MainLoopConfig, LeaseExtenderConfig

loop = MainLoop(
    adapter=adapter,
    requests=mailbox,
    config=MainLoopConfig(
        lease_extender=LeaseExtenderConfig(
            interval=60,      # Extend at most once per minute
            extension=300,    # Each extension adds 5 minutes
        ),
    ),
)
```

MainLoop automatically creates a Heartbeat, creates a LeaseExtender, attaches
the extender to each message during processing, and passes the heartbeat
through to tool execution. EvalLoop works the same way via `EvalLoopConfig`.

### Error Handling

Lease extension is a reliability optimization, not a correctness requirement:

| Error | Behavior |
| --- | --- |
| `ReceiptHandleExpiredError` | Logged as warning; processing continues |
| Network/transient errors | Logged; extension skipped |
| Already attached | Raises `RuntimeError` |

Failed extensions don't abort processing. If the lease expires, the message
will be redelivered to another worker.

### Timeout Calibration

The visibility timeout, extension interval, and watchdog threshold must be
coordinated:

```
visibility_timeout > watchdog_threshold + max_processing_time
extension < visibility_timeout
interval < extension / 2 (ensures extensions happen before expiry)
```

**Example for 10-minute max processing:**

| Parameter | Value | Rationale |
| --- | --- | --- |
| `max_processing_time` | 600s | Longest expected operation |
| `extension` | 300s | Extends by 5 minutes each beat |
| `interval` | 60s | Extends at most once per minute |
| `watchdog_threshold` | 720s | Detects truly stuck workers |
| `visibility_timeout` | 1800s | Initial 30-minute window |

______________________________________________________________________

## Putting It All Together

A production agent typically uses all three guardrails:

```python nocheck
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    PlanBasedChecker,
)
from weakincentives.contrib.tools.planning import Plan
from weakincentives.prompt import (
    DeadlineFeedback,
    FeedbackProviderConfig,
    FeedbackTrigger,
    PromptTemplate,
)
from weakincentives.runtime import (
    LeaseExtenderConfig,
    MainLoop,
    MainLoopConfig,
)

# 1. Feedback: alert the agent to remaining time
template = PromptTemplate[MyOutput](
    ns="my-namespace",
    key="my-prompt",
    sections=(task_section, tools_section),
    feedback_providers=(
        FeedbackProviderConfig(
            provider=DeadlineFeedback(warning_threshold_seconds=120),
            trigger=FeedbackTrigger(every_n_seconds=60, every_n_calls=10),
        ),
    ),
)

# 2. Task completion: don't stop until the plan is done
adapter = ClaudeAgentSDKAdapter(
    model="claude-opus-4-5-20251101",
    client_config=ClaudeAgentSDKClientConfig(
        task_completion_checker=PlanBasedChecker(plan_type=Plan),
    ),
)

# 3. Lease extension: keep the message alive while working
loop = MainLoop(
    adapter=adapter,
    requests=mailbox,
    config=MainLoopConfig(
        lease_extender=LeaseExtenderConfig(interval=60, extension=300),
    ),
)
```

**How they interact:**

- **Feedback** tells the agent "you have 5 minutes left"
- **Task completion** tells the agent "you haven't finished steps 3 and 4"
- **Lease extension** ensures the message stays invisible while both of the
  above are happening

Each addresses a different failure mode. Feedback prevents drift, task
completion prevents premature exit, and lease extension prevents message
duplication.

## Design Philosophy

These guardrails embody the "weak incentives" approach:

1. **Soft over hard**: Feedback informs but doesn't force; the agent retains
   autonomy
1. **Trust but verify**: The agent decides when it thinks it's done; checkers
   validate that assessment
1. **Proof-of-work**: Lease extension only extends on actual activity, not on a
   timer
1. **Graceful degradation**: Missing state doesn't block completion; exhausted
   resources skip checks

## Next Steps

- [Orchestration](orchestration.md): MainLoop and request handling
- [Lifecycle](lifecycle.md): Health checks and watchdog monitoring
- [Claude Agent SDK](claude-agent-sdk.md): Full adapter configuration
- [Sessions](sessions.md): Where feedback and events are stored

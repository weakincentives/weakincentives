# Task Completion Checking

*Canonical spec: [specs/TASK_COMPLETION.md](../specs/TASK_COMPLETION.md)*

Task completion checking prevents agents from stopping prematurely. When an
agent signals it wants to end—by producing structured output or ending its
turn—checkers verify that all required work is actually done.

## The Problem

Agents sometimes declare victory too early:

- Producing output before all plan steps are complete
- Ending turns when required files haven't been created
- Missing edge cases they were supposed to handle

Task completion checkers catch these cases and prompt the agent to continue.

## Core Types

### TaskCompletionResult

The result of checking completion:

```python nocheck
from weakincentives.adapters.claude_agent_sdk import TaskCompletionResult

# Task is complete
result = TaskCompletionResult.ok()

# Task is incomplete with feedback
result = TaskCompletionResult.incomplete(
    "Steps 'run tests' and 'update docs' are not yet done."
)
```

**Fields:**

- `complete`: Whether all tasks are complete
- `feedback`: Natural language explanation (when incomplete)

### TaskCompletionContext

Context passed to checkers:

```python nocheck
from weakincentives.adapters.claude_agent_sdk import TaskCompletionContext

context = TaskCompletionContext(
    session=session,              # Session state
    tentative_output=output,      # Output being produced
    filesystem=fs,                # Optional filesystem access
    adapter=adapter,              # Optional adapter reference
    stop_reason="end_turn",       # Why agent is stopping
)
```

### TaskCompletionChecker Protocol

```python nocheck
from typing import Protocol, runtime_checkable


@runtime_checkable
class TaskCompletionChecker(Protocol):
    def check(self, context: TaskCompletionContext) -> TaskCompletionResult: ...
```

Any class implementing `check()` with the right signature satisfies the
protocol.

## Built-in Checkers

### PlanBasedChecker

Verifies all steps in a `Plan` are marked "done":

```python nocheck
from weakincentives.adapters.claude_agent_sdk import PlanBasedChecker
from weakincentives.contrib.tools.planning import Plan

checker = PlanBasedChecker(plan_type=Plan)
```

**Behavior:**

- Retrieves the Plan from session state
- Checks each step's status
- Returns incomplete if any step has `status != "done"`
- No-op when `plan_type=None` (always returns complete)

**Graceful degradation:**

- No plan type configured → returns complete
- No plan slice in session → returns complete
- Plan not yet initialized → returns complete

This means PlanBasedChecker only enforces completion when a plan actually
exists.

### CompositeChecker

Combines multiple checkers:

```python nocheck
from weakincentives.adapters.claude_agent_sdk import CompositeChecker

checker = CompositeChecker(
    checkers=(plan_checker, file_checker, test_checker),
    all_must_pass=True,  # Default: all checkers must pass
)
```

**Modes:**

- `all_must_pass=True`: All checkers must pass; short-circuits on first failure
- `all_must_pass=False`: Any checker can pass; short-circuits on first success

Combined feedback joins non-None feedback from all evaluated checkers.

## Configuration

Enable task completion checking via `ClaudeAgentSDKClientConfig`:

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

## Writing Custom Checkers

Implement the protocol to create domain-specific checkers:

```python nocheck
from dataclasses import dataclass
from weakincentives.adapters.claude_agent_sdk import (
    TaskCompletionChecker,
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


@dataclass(frozen=True)
class TestsPassedChecker:
    """Verifies tests have passed."""

    def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
        # Check session for test results
        test_results = context.session[TestResults].latest()
        if test_results is None:
            return TaskCompletionResult.incomplete("Tests have not been run.")
        if not test_results.all_passed:
            return TaskCompletionResult.incomplete(
                f"{test_results.failed_count} tests are still failing."
            )
        return TaskCompletionResult.ok()
```

Combine custom checkers with built-in ones:

```python nocheck
checker = CompositeChecker(
    checkers=(
        PlanBasedChecker(plan_type=Plan),
        RequiredFilesChecker(("README.md", "tests/test_main.py")),
        TestsPassedChecker(),
    ),
    all_must_pass=True,
)
```

## Enforcement Points

Task completion is checked at two points:

### 1. Stop Hook (During Execution)

When the agent attempts to end its turn:

- Checker runs via `create_task_completion_stop_hook`
- If incomplete: Agent receives feedback and continues
- If complete: Agent stops normally

### 2. Final Verification (After Execution)

After the SDK query completes:

- `_verify_task_completion` runs in the adapter
- If structured output exists but tasks incomplete: Raises `PromptEvaluationError`
- Catches edge cases the stop hook might miss

### Resource Exhaustion

Checkers are **skipped** when deadline or budget is exhausted:

- Partial output is acceptable when resources are depleted
- The agent did what it could within constraints
- This prevents infinite loops when completion is impossible

## Feedback Behavior

When tasks are incomplete:

1. Checker returns `TaskCompletionResult.incomplete(feedback)`
1. Feedback is delivered to the agent
1. Agent continues working
1. Process repeats until complete or resources exhausted

**Feedback truncation:** Long task lists are limited to 3 titles + "..." to
prevent noisy feedback that obscures actionable information.

## Integration with Planning

Task completion checking works naturally with the planning tools:

```python nocheck
from weakincentives.contrib.tools import PlanningToolsSection, PlanningStrategy
from weakincentives.adapters.claude_agent_sdk import PlanBasedChecker

# Create planning section
planning = PlanningToolsSection(
    session=session,
    strategy=PlanningStrategy.PLAN_ACT_REFLECT,
)

# Configure checker
checker = PlanBasedChecker(plan_type=Plan)

# Agent must complete all plan steps before finishing
```

The agent creates a plan, works through steps, and cannot finish until all
steps are marked done. This externalizes progress tracking and makes completion
verification straightforward.

## Design Philosophy

Task completion checking embodies "trust but verify":

1. **Agent autonomy**: The agent decides when it thinks it's done
1. **Verification**: Checkers validate that assessment
1. **Feedback**: Incomplete work gets specific, actionable feedback
1. **Graceful degradation**: Missing state doesn't block completion

This approach respects agent capabilities while providing guardrails against
premature termination.

## Next Steps

- [Tools](tools.md): Planning tools that integrate with completion checking
- [Claude Agent SDK](claude-agent-sdk.md): Full adapter configuration
- [Lifecycle](lifecycle.md): Deadlines and budgets that affect checking

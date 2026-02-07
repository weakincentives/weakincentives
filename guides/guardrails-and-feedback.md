# Guardrails and Feedback

*Canonical spec: [specs/GUARDRAILS.md](../specs/GUARDRAILS.md)*

This guide explains how WINK keeps agents on track during unattended
execution. Three mechanisms work together: tool policies gate dangerous
actions, feedback providers offer course-correction, and task completion
checkers verify goals before the agent stops.

## The Problem: Agents Without Guardrails

An agent with tools but no constraints is free to do things in any order.
That includes deploying untested code, overwriting files it never read,
or declaring itself done with half the work incomplete. Models are
generally well-intentioned, but they make mistakes--especially in long
sessions where earlier context has scrolled away.

You could embed validation logic in every tool handler, but that scatters
safety concerns across your codebase. When you add a new tool, you have
to remember which checks apply. When requirements change, you hunt
through handlers to update them.

WINK takes a different approach: declare constraints separately from
tools, compose them independently, and enforce them at the runtime level.

## Three Mechanisms, Three Strengths

| Mechanism | What It Does | Enforcement |
|-----------|--------------|-------------|
| Tool Policies | Gate tool calls | Hard block |
| Feedback Providers | Inject guidance | Advisory |
| Task Completion | Verify goals | Block early exit |

These are complementary, not alternatives. Policies prevent the agent
from doing things it should not. Feedback nudges the agent toward better
decisions. Task completion ensures the agent finishes what it started.

## Tool Policies

The [Tools guide](tools.md) covers policy basics and the `ToolPolicy`
protocol. This section focuses on the mental model.

### Constraints, Not Workflows

A policy says "X requires Y" without saying when or how Y should happen.
The agent is free to satisfy the constraint any way it likes. This
preserves reasoning autonomy--the property that makes LLM agents useful
in the first place.

Compare:

- **Workflow:** "Step 1: read file. Step 2: edit file. Step 3: test."
- **Policy:** "A file must be read before it can be overwritten."

The workflow breaks when the agent encounters something unexpected. The
policy holds regardless of what path the agent takes.

### Fail-Closed by Default

When a policy denies a tool call, the tool does not execute. The agent
receives an error message explaining which constraint was violated and
why. This is intentional:

- The agent cannot accidentally bypass constraints
- The denial message gives the agent information to self-correct
- There is no "maybe" state--either the call is allowed or it is not

This mirrors a well-known principle from security engineering: when in
doubt, deny. The agent can always satisfy the prerequisite and retry.

### Composition Through Conjunction

Multiple policies can govern the same tool. All must allow the call for
it to proceed. This means policies compose through logical AND:

- ReadBeforeWritePolicy says "OK" to writing `config.yaml`
  (the file was read earlier)
- SequentialDependencyPolicy says "DENIED" to writing `config.yaml`
  (lint has not run yet)
- Result: call denied

You never need to reason about policy ordering or priority. Each policy
evaluates independently. Adding a new policy cannot weaken an existing
one--it can only add constraints.

### State Lives in the Session

Policies track what has happened (which tools have been called, which
files have been read) via session slices. This means policy state is:

- **Snapshotable:** captured with session snapshots
- **Restorable:** rolled back when tool calls fail
- **Isolated:** each session has independent state

When a tool call fails and the session is restored to its pre-call
snapshot, policy state rolls back too. The agent cannot game the system
by triggering a failure after a side effect.

## Feedback Providers

Policies are binary: allow or deny. But sometimes the agent needs
softer guidance--"you are running low on time," "you seem to be going
in circles," or "a configuration file appeared that you should read."

Feedback providers fill this role. They observe the agent's trajectory
and inject contextual messages after tool calls.

### How Feedback Differs from Policies

| Aspect | Policies | Feedback |
|--------|----------|----------|
| Enforcement | Hard block | Advisory |
| Timing | Before tool call | After tool call |
| Agent response | Must comply | May choose to ignore |
| Purpose | Prevent mistakes | Encourage better decisions |

Feedback is non-blocking. The agent receives the message and decides
what to do with it. This is appropriate for guidance that depends on
judgment: "you have 3 minutes remaining" does not tell the agent what
to do--it provides information for the agent to reason about.

### Triggers Control When Feedback Fires

Each feedback provider has a trigger that determines when it runs.
Triggers are based on:

- **Call count:** run after every N tool calls
- **Elapsed time:** run after N seconds since last feedback
- **File creation:** run once when a specific file appears

Conditions within a trigger are OR'd--any matching condition fires the
provider. Each provider maintains its own trigger cadence independently.

### All Matching Providers Run

When multiple providers trigger simultaneously, all of them produce
feedback. There is no priority or first-match-wins logic. This ensures
no guidance is silently dropped. The agent receives all relevant
feedback combined into a single context injection.

### Built-in Providers

WINK ships two providers:

**DeadlineFeedback** reports remaining time. Configure it with a
trigger interval and a warning threshold:

```python nocheck
FeedbackProviderConfig(
    provider=DeadlineFeedback(warning_threshold_seconds=120),
    trigger=FeedbackTrigger(every_n_seconds=30),
)
```

Below the threshold, the feedback includes suggestions to prioritize
remaining work.

**StaticFeedbackProvider** delivers a fixed message when triggered.
Useful with `FileCreatedTrigger` for one-time guidance:

```python nocheck
FeedbackProviderConfig(
    provider=StaticFeedbackProvider(
        feedback="AGENTS.md detected. Follow the conventions within.",
    ),
    trigger=FeedbackTrigger(
        on_file_created=FileCreatedTrigger(filename="AGENTS.md"),
    ),
)
```

### Custom Providers

Implement the `FeedbackProvider` protocol:

- `name`: unique identifier
- `should_run(context)`: additional filtering beyond the trigger
- `provide(context)`: produce a `Feedback` object with summary,
  observations, suggestions, and severity

The `FeedbackContext` gives you access to the session, the prompt,
the deadline, recent tool calls, and feedback history. This is enough
to detect patterns like repeated failures or stalled progress.

## Task Completion

Policies and feedback operate during execution. Task completion
operates at the boundary--when the agent tries to stop.

### The Problem It Solves

Models sometimes declare victory prematurely. They produce a final
answer with tasks still incomplete, or they stop after encountering
an error without retrying. Without a check, the session ends and
the user discovers the incomplete work later.

Task completion checkers intercept the agent's attempt to finish and
verify that goals are actually met. If not, the agent receives
feedback and continues working.

### How It Works

A `TaskCompletionChecker` examines the session state and the agent's
tentative output. It returns either "complete" or "incomplete with
feedback." The adapter integrates this at two points:

1. **During execution:** if the agent produces structured output while
   tasks remain, the checker injects feedback as additional context
1. **At stop:** if the agent tries to end the turn, the checker can
   request more turns

### Built-in Checkers

**PlanBasedChecker** looks at a `Plan` slice in the session. If any
plan steps have a status other than "done," the agent is told to keep
working. The feedback includes up to three incomplete task titles.

**CompositeChecker** combines multiple checkers. Configure it to
require all checkers to pass, or allow any single checker to pass:

```python nocheck
checker = CompositeChecker(
    checkers=(
        PlanBasedChecker(plan_type=Plan),
        FileExistsChecker(("output.txt",)),
    ),
    all_must_pass=True,
)
```

### Budget and Deadline Override

Task completion checking is automatically skipped when the budget or
deadline is exhausted. There is no point asking the agent to continue
when it has no resources left.

## How the Three Mechanisms Interact

Consider a code review agent:

1. **Tool policies** ensure the agent reads a file before commenting
   on it (ReadBeforeWritePolicy) and runs tests before approving
   changes (SequentialDependencyPolicy).

1. **Feedback providers** remind the agent of the deadline every 30
   seconds and detect when a configuration file like `AGENTS.md`
   appears in the workspace.

1. **Task completion** verifies that all review tasks in the plan are
   marked done before the agent produces its final report.

Policies provide hard safety rails. Feedback provides situational
awareness. Task completion provides goal verification. Together they
create a system where the agent is free to reason about *how* to
accomplish its goals while being constrained on *what* it must not
skip.

## Design Guidance

**Prefer policies over handler validation.** Cross-cutting constraints
belong in policies, not scattered across tool handlers. A policy is
visible, testable, and composable. Validation buried in a handler is
none of these.

**Keep feedback advisory.** If something must be enforced, use a
policy. Feedback is for guidance where the agent's judgment matters.
Mixing enforcement into feedback creates a confusing contract.

**Use task completion for goal verification, not process enforcement.**
Task completion answers "did the agent achieve its goals?" not "did the
agent follow the right steps?" Process enforcement belongs in policies.

**Avoid over-constraining.** If your policies leave only one valid
execution path, you have written a workflow in policy syntax. The point
of policies is to preserve multiple valid paths while ruling out
invalid ones.

## Next Steps

- [Tools](tools.md): Tool contracts, policies, and transactional
  execution
- [Sessions](sessions.md): State management with reducers and events
- [Orchestration](orchestration.md): Request handling with AgentLoop

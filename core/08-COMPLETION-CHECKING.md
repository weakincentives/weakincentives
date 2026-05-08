# Completion Checking

A **completion checker** verifies that the agent's stop condition has
actually been met before allowing termination. It is the third tier of
WINK's control model: policies gate tool calls (block what shouldn't
happen), feedback nudges direction (suggest what should happen), and
completion checking blocks early termination (verify what *did* happen).

______________________________________________________________________

## The problem

Agents declare victory prematurely.

This is a real failure mode, not a hypothetical one. A model asked to
"produce a report and a results file" will sometimes produce only the
report and announce that it is done. A model asked to "fix all failing
tests" will sometimes fix one and stop. Without a verification step, the
agent's self-assessment is the last word. If it's wrong, the run silently
ends in an incomplete state.

Completion checking exists to make "done" objective. The agent's
self-declaration is necessary but not sufficient; the framework also runs
a check, and if the check fails, the agent is told to keep going.

______________________________________________________________________

## What a checker does

A checker runs at the moment the agent attempts to terminate. It receives
context — the session, the tentative output, the filesystem (if the
adapter has one), the stop reason — and returns one of two results:

- **OK.** The stop condition is met. The agent's termination stands.
- **Incomplete, with feedback.** The stop condition is not met. The
  feedback message goes back to the agent, and execution continues.

The check is honest. It does not just look at what the agent claims; it
looks at the world. If the criterion is "produce `report.md`", the
checker asks the filesystem whether the file exists, not the agent whether
it produced one.

______________________________________________________________________

## Why this is a definition-level concern

Whether the agent is *done* is a property of the agent's goal. Not the
harness. Not the model. Not the SDK.

This is the same principle as policies and feedback: anything that
describes *what success looks like* travels with the prompt definition.
The completion checker attaches to the prompt template alongside its
other guardrails, and the same checker runs regardless of which adapter
executes the prompt.

If completion logic lived in the harness, switching runtimes would mean
re-implementing it. If it lived in the prompt text, the model could
evaluate it however it pleased. By making it a first-class definition
artifact, completion behavior is portable, reviewable, and testable.

______________________________________________________________________

## How adapters wire it in

Each runtime exposes its own mechanism for blocking termination — a
"needs more turns" hook, a continuation loop, a stop callback. Adapters
translate the checker into the runtime's native shape. From the
definition's perspective, this is invisible: the same checker runs
everywhere.

The continuation loop has a built-in cap so a buggy or contradictory
checker cannot drive the agent forever. After a fixed number of failed
verifications, the framework gives up and reports the failure.

______________________________________________________________________

## Composition

Checkers compose. A composite checker takes several child checkers and a
combination mode — *all must pass* (AND) or *any must pass* (OR) — and
short-circuits on the first decisive result.

This is the same pattern as policies. A real agent rarely has one
completion criterion; it has several, and the cleanest expression is to
declare each criterion as its own checker and combine them.

For example, a code-review agent might require:

- A summary file exists (file checker).
- The summary contains a non-empty conclusion (content checker).
- A review JSON file exists with a list of issues (file + structured
  output checker).

Each is independently evaluable. The composite says "all of these."

______________________________________________________________________

## Bypass under exhaustion

Completion checking is skipped when the run has exhausted its budget or
deadline. The reason: at that point, the agent isn't choosing to stop —
it is being *stopped*. Continuing the loop would mean keeping the agent
running past its constraints, which is a worse failure mode than
returning incomplete.

When this happens, the framework records that the run terminated due to
exhaustion and the checker is bypassed. Observability surfaces the bypass
so an operator can tell *incomplete-but-validated* apart from
*incomplete-because-time-ran-out*.

______________________________________________________________________

## Fail-closed when verification is impossible

If a checker requires the filesystem to verify completion (e.g., it
checks whether a file exists) and no filesystem is available in the
context, it returns *incomplete*. The reasoning: it cannot verify
completion, and the agent should not stop without verification.

This is the same posture as policies — when uncertain, deny. It avoids
the silent-success failure mode where missing infrastructure leads to
falsely-passed checks.

______________________________________________________________________

## What checkers are not

- **Not output validation.** A structured-output prompt validates the
  shape of the agent's response at parse time. That is a typed-contract
  concern, not a completion concern. A checker asks "did the right
  *effect* happen in the world?", not "is the output well-formed?"
- **Not policies.** A policy gates tool *calls*. A checker gates *stops*.
  They run at different moments and answer different questions.
- **Not feedback.** Feedback is advisory and runs during execution. A
  checker is a hard gate at termination.
- **Not retry.** A checker may cause continuation, but the agent decides
  *what to do next* — the framework only signals "not done yet."

______________________________________________________________________

## Anti-patterns

- **Verbose feedback messages.** The model has to read this to recover.
  Be specific about what is missing. Truncate long lists.
- **Checkers that depend on internal model state.** A check should look
  at the world (files, session events, tentative output), not at the
  model's chain of thought.
- **Per-adapter completion logic.** If the criteria differ by adapter, the
  criteria are wrong. Either the criteria are about the goal (definition)
  or they are operational (harness) — not both.
- **Always-on retry without a cap.** The continuation loop is bounded.
  Treat the cap as a real failure mode and surface it; don't pretend it
  cannot happen.

______________________________________________________________________

## Pointers

- [POLICIES](06-POLICIES.md) — the gating tier (blocks tool calls).
- [FEEDBACK](07-FEEDBACK.md) — the advisory tier (nudges during
  execution).
- [STATE](05-STATE.md) — what a checker can read about the run so far.
- [PRINCIPLES](PRINCIPLES.md) §13 — definition assets carry their own
  constraints.

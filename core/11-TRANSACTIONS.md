# Transactions

Every tool call in WINK is a **transaction**. The framework snapshots
session and resource state before the handler runs and restores those
snapshots on failure. A failed tool leaves no trace of any kind: no
half-applied changes, no inconsistent state, no orphan effects.

This is the property that makes the rest of the system safe.

______________________________________________________________________

## What "transactional" means here

A transaction is the atomic unit: it either commits in full or rolls back
in full. There is no partial success.

For a WINK tool call, the boundary is:

- **Begin.** Snapshot session state and any snapshotable resources
  (filesystems, journaled clients).
- **Execute.** Run the handler. Apply any state mutations through the
  session dispatcher, which routes events to reducers.
- **Commit or rollback.**
  - On success, discard the snapshot. Mutations stand.
  - On failure (handler raised, validation failed, policy denied,
    deadline elapsed mid-call), restore the snapshot. Mutations vanish.

The agent sees the failure as a normal `ToolResult.error(...)`. It does
not see partial state from the failed call.

______________________________________________________________________

## Why this matters

A few important properties fall out.

**Aggressive retry is safe.** If the agent calls a tool, it fails, and the
agent calls it again with different parameters, the second call starts
from the same state as the first. There is no need to "undo" the first
call's effects — they were never applied.

**Policies can be enforced cleanly.** A policy can deny a tool call after
the snapshot is taken, knowing the rollback will restore the world as if
the call never started. The denial is observable; the side-effects are
not.

**Reasoning is local.** When debugging, you do not have to figure out
which mutations from a failed call may have leaked. The answer is none.
Failed calls are absent from history except as failure events.

**Tests are deterministic.** A test that runs a sequence of tool calls
sees exactly the state the calls actually produced. A failed call in the
middle does not contaminate the state observed by later calls.

______________________________________________________________________

## What gets snapshotted

Two layers participate.

- **Session state.** All `STATE`-policy slices are captured. `LOG`-policy
  slices are *not* — the history of what happened must survive failures.
  This is what allows audit logs to record failed calls even when the
  state they produced is rolled back.

- **Snapshotable resources.** Resources that implement the snapshotable
  protocol (the in-memory filesystem, the host filesystem with journaling)
  capture their state at snapshot time. On rollback, they restore. This
  is how filesystem mutations participate in rollback.

Resources that are *not* snapshotable do not participate. A network call
that succeeds before the handler raises has happened in the world; the
framework cannot un-send it. Use idempotency keys, retry logic, or
adapter-level guarantees for those cases — transactions are local to the
process.

______________________________________________________________________

## Where transactions live

The transaction layer is invoked once per tool call by the bridging
machinery between the harness and the handler. The author of a tool
handler does not need to manage transactions explicitly — calling
`return ToolResult.error(...)` or raising an exception both trigger
rollback automatically. The author's job is to express the operation;
atomicity is the framework's job.

The same is true for adapters. Each adapter wraps its tool dispatch in a
transaction. Behavior across adapters is uniform.

______________________________________________________________________

## What survives a failed call

A few things are deliberately preserved across rollback:

- **The failure event itself.** The agent sees the failure as a tool
  result; the framework records the failed invocation in the
  appropriate `LOG`-policy slice; observability surfaces it normally.
- **Resources that are not snapshotable.** External clients, network
  effects, and anything outside the WINK process boundary do not roll
  back.
- **The session's lock and dispatcher.** These are infrastructure, not
  state, and continue functioning across the failure.

This is the right shape: the *story* of what happened survives, while the
*state changes* of failed steps do not.

______________________________________________________________________

## What this enables

Several other concepts only work because transactions exist.

- **Policies fail-closed cleanly.** A policy can deny a call without
  worrying about partial side effects.
- **Read-before-write style invariants.** The "did the read happen?"
  check works because failed reads do not pollute the read-tracking
  slice.
- **Test fixtures.** A test can run a tool, observe its failure, and
  inspect the session — knowing the session reflects only the
  successful operations.
- **Debug bundles.** The pre-call and post-call snapshots are the same
  artifact; rollback is the difference between them.
- **Time-travel debugging.** A snapshot at any point can be restored as
  the starting state for further investigation.

______________________________________________________________________

## What transactions are not

- **Not distributed.** WINK transactions are in-process. They do not
  coordinate across machines.
- **Not a substitute for idempotency.** Outside-the-process effects
  cannot be rolled back. Tools that touch external systems still need
  idempotency keys, retry semantics, or compensating actions.
- **Not visible to the model.** The agent does not see "transaction
  began" or "transaction rolled back." It sees a tool succeeded or
  failed. The transactional layer is implementation detail of *how*
  failures stay clean.
- **Not free.** Snapshotting has a cost — usually small, sometimes
  significant for very large state. Resources that produce expensive
  snapshots should consider whether they need full rollback or whether
  a journal-based approach suffices.

______________________________________________________________________

## Anti-patterns

- **Tools that mutate resources outside the snapshot scope.** A handler
  that creates a temp directory directly with the OS bypasses the
  filesystem resource and the snapshot layer. Use the resource.
- **Long-running handlers without heartbeats.** A handler that holds the
  transaction open for many seconds blocks observability and risks
  visibility-timeout problems with the harness. Heartbeats keep the
  call's lease alive.
- **Tools that do work in a finally block to "make sure it happens".**
  Finally-block side effects often defeat rollback by running on the
  failure path. Put cleanup in resource closeables, not in handlers.

______________________________________________________________________

## Pointers

- [TOOLS](04-TOOLS.md) — how the handler contract is shaped to make
  transactions natural.
- [STATE](05-STATE.md) — slice policies, what survives rollback, what
  doesn't.
- [RESOURCES](09-RESOURCES.md) — how snapshotable resources participate.
- [POLICIES](06-POLICIES.md) — fail-closed denials are clean because
  rollback is automatic.

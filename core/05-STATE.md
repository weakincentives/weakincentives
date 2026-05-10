# State

State in WINK is **event-driven, immutable, and inspectable**. There is no
shared mutable object. There is no in-place update. Every change is an
event; every read returns a snapshot. This is the property that makes
agents debuggable — and the property that makes transactional tool
execution possible at all.

______________________________________________________________________

## The mental model

Think of an agent's state as a ledger of events with derived views.

- The **events** are the source of truth. Each is a typed record
  describing something that happened: a prompt was rendered, a tool was
  invoked, a file was read.
- The **slices** are the derived views. A slice of type `T` answers: *what
  do we know about T given all the events so far?*
- The **reducers** are pure functions that compute slices from events. A
  reducer takes the current slice and a new event, and returns a new
  slice. It never mutates.
- A **session** is the container that holds slices, registers reducers,
  and dispatches events. It is the boundary across which all mutations
  flow.

The whole machinery is inspired by Redux-style architectures, but with
strict typing, snapshot semantics, and transactional rollback baked in.

______________________________________________________________________

## Events

An event is an immutable record. It carries everything needed to
reconstruct the change it represents — including correlation IDs that
tie it back to the prompt render or tool invocation that produced it.
Events are not strings. They are not free-form maps. They are specific
named types: `ToolInvoked`, `FileRead`, `PromptRendered`,
`PromptExecuted`, and so on.

Tools publish events through the session dispatcher. Adapters publish
events at the boundaries of evaluation. Tests publish events to drive
reducers under controlled conditions. In every case, the event is the
*only* way state changes.

______________________________________________________________________

## Slices

A slice is the typed view of one aspect of session state. Where a session
might hold many kinds of information, a slice answers a single question:
"what tools have been invoked so far?", "what files have been read?",
"what is the current plan?".

Slices are addressed by type: you ask the session for the slice of type
`Plan` and get back an accessor that lets you query it.

Slices expose three reading patterns:

- `latest()` — the most recent value (or `None` if empty).
- `all()` — every value in chronological order.
- `where(predicate)` — values matching a filter.

These are the only operations a reader needs. Tests, policies, feedback
providers, and tool handlers all use the same shape.

______________________________________________________________________

## Reducers

A reducer is a pure function bound to one event type, targeting one slice
type. Its job is to return the *operation* that should be applied to the
slice — not to apply it. This is what keeps reducers strictly pure:
they describe change, the framework performs change.

Operations are:

- **Append** — add a single new value (default for ledger-style slices).
- **Extend** — add several values.
- **Replace** — overwrite the entire slice with a new tuple.
- **Clear** — remove items, optionally filtered by predicate.

Reducers are typically declared *on* the slice's record type as
methods, so the data and the rules for transforming it live together.
This mirrors the same co-location principle that puts tools on sections.

______________________________________________________________________

## Slice policies

A slice has a *policy* — `STATE` or `LOG` — that controls how it
participates in transactions and snapshots.

- **STATE slices** are part of the session's working state. They are
  snapshotted, restored, and rolled back as part of transactional tool
  execution. If a tool fails, the rollback returns these slices to their
  pre-call form.
- **LOG slices** are append-only ledgers preserved across rollbacks. The
  history of what *happened* must survive failures, even when the working
  state is rewound. Tool invocations, transcript entries, and event
  records typically use LOG semantics.

The split is what allows the session to behave both as a working
state machine and as an audit log without conflicting requirements.

______________________________________________________________________

## Snapshots

The session can be captured at any point as an immutable snapshot. The
snapshot serializes to JSON, restores cleanly, and respects slice
policies (LOG slices are preserved during restore; STATE slices are
overwritten).

Snapshots have several uses:

- **Transactional rollback.** Each tool invocation is wrapped in a
  snapshot/restore pair so that failures leave no trace.
- **Time-travel debugging.** Capture the session at a known-good point;
  replay later runs against the same starting state.
- **Test fixtures.** Snapshot a session at the moment a regression
  occurred and use it as a deterministic test starting point.
- **Debug bundles.** A failed run can be archived as a snapshot plus its
  surrounding events for offline inspection.

______________________________________________________________________

## Why pure transitions matter

If state mutated in place, transactions would not be safe. Two tools
running back to back could leak partial change. A failure halfway
through a tool would leave the session in a state that is neither "before"
nor "after". Reasoning about correctness would require understanding the
mutation order of every method.

By restricting all change to "publish event → run reducers → produce
operations → apply operations atomically," the framework can:

- Snapshot before, apply, restore on failure.
- Replay event streams and get identical state.
- Reason about correctness one reducer at a time.
- Test reducers without booting up the rest of the framework.

The cost is an extra layer of indirection. The benefit is that *every step
is a checkpoint*.

______________________________________________________________________

## What state is for

State serves four downstream consumers:

- **Policies.** "Was this file read before?" is a session query.
- **Feedback providers.** "How many tool calls since the last
  reminder?" is a session query.
- **Completion checkers.** "Did the agent produce the expected output?"
  is often answerable from session state.
- **Adapters and observability.** Transcripts, debug bundles, and event
  logs are projections of the session.

If a feature needs to ask "what has happened so far?", the answer should
come from the session, not from out-of-band logging or instance variables
on a controller.

______________________________________________________________________

## Sessions are orchestrator-named and durable

A session is named by the orchestrator using a stable identifier it
owns. The name is the durable handle: it survives transport drops,
compute restarts, and orchestrator handoffs. A new orchestrator (or
the same orchestrator on a new connection) can attach to an existing
session by sending its name — the state is already there, on the
sandbox side.

Backend-native session identifiers — provider trace tokens, harness
internal handles — stay behind the protocol. Only the orchestrator's
name crosses the boundary. This is what makes session state portable
across runtimes and across orchestrator instances. A failover that
swaps the orchestrator process does not lose work; the new process
attaches to the same session by name and continues.

Session lifetime is independent of any single connection or compute
instance. (See [Durable Work](19-DURABLE-WORK.md) for the full
treatment of work identity, transport, and reattach.)

______________________________________________________________________

## Anti-patterns

- **Mutating records in place.** Slices are immutable; the framework
  enforces this. Reducers always return new values.
- **Bypassing the dispatcher.** Tools that mutate session-level
  containers directly defeat snapshotting, audit, and transactional
  semantics.
- **State that grows without bound.** Slices grow until you trim them
  via `replace_latest`, `clear`, or a custom reducer. Plan retention
  explicitly.
- **Putting derived state into reducers from outside.** A reducer should
  derive its output from its inputs (current slice + event). Pulling in
  ambient state breaks reproducibility.

______________________________________________________________________

## Pointers

- [TOOLS](04-TOOLS.md) — how tools publish events.
- [TRANSACTIONS](11-TRANSACTIONS.md) — how snapshots wrap tool calls.
- [POLICIES](06-POLICIES.md), [FEEDBACK](07-FEEDBACK.md),
  [COMPLETION-CHECKING](08-COMPLETION-CHECKING.md) — the most common
  consumers of session state.
- [OBSERVABILITY](14-OBSERVABILITY.md) — how snapshots and event streams
  surface to humans.

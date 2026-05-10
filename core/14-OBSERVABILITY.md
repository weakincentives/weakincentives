# Observability

An agent is only as good as the evidence you have about its behavior.
Observability in WINK is not an afterthought layered on with logs; it is
a property of the architecture. Every event is recorded, every state is
snapshotable, every transcript is unified, and every run can be packaged
into a self-contained bundle. Debugging is querying, not re-running.

______________________________________________________________________

## The four artifacts

WINK produces four kinds of observability artifact, each serving a
different question.

- **Events.** Live signal during execution. "What is happening right
  now?"
- **Snapshots.** Immutable captures of state at a moment. "What was
  true here?"
- **Transcripts.** Unified, adapter-agnostic conversation logs. "What
  did the model see and say?"
- **Debug bundles.** Self-contained zip archives packaging the above
  for an entire run. "Show me everything about this execution."

Each artifact is built on the same primitives — events flow through the
session dispatcher, slices accumulate them, snapshots capture state,
transcripts derive from event streams, bundles aggregate the rest. There
is one observability stack, not several.

______________________________________________________________________

## Events

The session dispatcher is a typed publish/subscribe system. Adapters and
tools publish events; reducers consume them; subscribers (logging,
transcripts, custom handlers) observe.

The framework defines a small set of canonical events that mark the
edges of execution: prompt rendered, tools resolved, tool invoked,
prompt executed. Tools and policies publish their own domain events
(file read, plan updated, policy denied) using the same machinery.

Events are immutable, typed, and ordered per dispatcher. Subscribers
that fail are isolated — their errors are logged, but they do not abort
the run. This is what allows observability to layer on without becoming
a reliability dependency.

______________________________________________________________________

## Snapshots

A session can be captured at any moment as an immutable snapshot. The
snapshot serializes to JSON, restores cleanly, and respects slice
policies (`LOG`-policy slices preserved; `STATE`-policy slices
overwritten on restore).

Snapshots are useful in three modes:

- **Transactional rollback.** Each tool call wraps in a snapshot/restore
  pair. (See [Transactions](11-TRANSACTIONS.md).)
- **Time-travel debugging.** Capture the session at a known-interesting
  point; later runs start from the same state.
- **Test fixtures.** Reproduce a regression by snapshotting the moment
  it happened and using that snapshot as the test starting point.

The format is intentionally readable. A snapshot is a small JSON
structure with ISO-8601 timestamps, qualified type names (so polymorphic
events round-trip), and schema versioning. Tools like `jq` work; bespoke
viewers are unnecessary.

______________________________________________________________________

## Transcripts

A transcript is the unified record of a single evaluation: every user
message, every assistant message, every tool call, every result, every
reasoning block, every system event. Across adapters, the *shape* is
identical — common envelope keys, canonical entry types — even though
the underlying runtimes emit different signals.

This uniformity is the key feature. Without it, switching harnesses
would mean re-learning every observability tool. With it, the same
analysis works on a Claude run and a Codex run.

Per-adapter bridges translate runtime-native signals (Claude's JSONL
transcript files, Codex's stdio notifications, ACP's `session/update`
notifications) into the unified format. The translation logic lives in
each adapter; the resulting entries share envelope and type vocabulary.

Transcripts emit as DEBUG-level structured log records. The existing
log infrastructure carries them. There is no separate transcript bus.

______________________________________________________________________

## Debug bundles

A debug bundle is a single zip archive capturing everything needed to
understand, reproduce, and debug a run. Layout is fixed and predictable:

- The request and response.
- The session before and after execution.
- The full structured log stream.
- The transcript, extracted as its own file for convenience.
- Configuration: adapter settings, prompt overrides, run context.
- Metrics: token usage, timing, budget state.
- Environment: OS, language runtime version, version-control commit,
  installed packages.
- Error details, when applicable.
- A workspace filesystem snapshot, when one was used.

The bundle is *self-contained*. Reading it later — possibly on a
different machine, possibly with a different version of WINK installed
— gives a complete picture of what happened. This is what makes
post-mortem analysis tractable.

Bundles can also serve as input. A bundle can be replayed against the
same prompt with different adapter configuration to test fixes; can be
compared against another bundle to detect regression; can be used as a
permanent record of an interesting case.

______________________________________________________________________

## Why this matters

Three properties make observability a structural property, not a feature.

**No truth lives outside the session.** State changes flow through the
dispatcher. There is no "out-of-band" mutation that observability would
miss. Reconstructing state from the event stream produces an exact
match.

**No artifact lives outside the bundle.** Logs, snapshots, transcripts,
config, environment — they all package together. There is no piece of
context that requires "ask the operator who ran it."

**No adapter lives outside the contract.** All adapters publish the same
events, emit the same transcript shape, and produce the same bundle
layout. Observability tooling is portable across runtimes.

These are the conditions under which "debugging is querying" becomes
true. The artifacts are rich enough that asking questions is the
diagnostic method, not running the agent again with extra logging.

______________________________________________________________________

## Across the sandbox boundary

In production, the harness runs in a remote sandbox (see
[Remote Execution](18-REMOTE-EXECUTION.md)). Observability flows
across the protocol boundary.

- **Logs stream from the sandbox.** They are emitted as the sandbox
  runs and ship to the orchestrator over the protocol — not collected
  at the end.
- **Backend events are normalized, not forwarded.** The sandbox
  translates its native event stream into the canonical transcript
  envelope; native runtime types and backend trace tokens stay behind
  the boundary. Transcript entries carry orchestrator-owned
  identifiers, so a reader can correlate transcript and orchestrator
  logs without translation. (See [Durable Work](19-DURABLE-WORK.md)
  for why identity is the orchestrator's.)
- **Bundles assemble from both sides.** The orchestrator contributes
  what it owns: request, configuration, session, run context. The
  sandbox contributes what it owns: filesystem snapshot, transcript
  files, resource metrics. The bundle is finalized on the orchestrator
  side using both contributions.

The bundle layout is identical to local execution. A reader cannot
tell, from the bundle alone, whether the run was local or remote. The
diagnostic skill is the same in either case.

______________________________________________________________________

## Sampling and retention

Producing every event for every run is fine in development but expensive
in production. The framework supports retention and sampling policies
on bundles: keep the last N, drop bundles older than X days, store to
external blob storage instead of local disk. Configuration is
operational, not part of the definition.

The base level — events through the dispatcher, transcripts as DEBUG
logs — is always on. The expensive parts (filesystem snapshots, full
bundles) are opt-in.

______________________________________________________________________

## What observability is not

- **Not a replacement for tests.** Bundles tell you what happened; tests
  tell you what should happen. Both are needed.
- **Not a replacement for monitoring.** Observability captures a single
  run in detail. Monitoring is about aggregate behavior over time. They
  serve different questions.
- **Not free.** Snapshots have a cost; bundles take disk; events require
  serialization. Production deployments tune retention.
- **Not a way to recover state.** A snapshot can restore session state,
  not external state. If a tool sent an email, the email was sent.

______________________________________________________________________

## Anti-patterns

- **Side-channel logging.** Print statements and ad-hoc logs that don't
  flow through the dispatcher are invisible to bundles, tests, and
  transcripts. Use the structured logger or publish an event.
- **Mutable state outside the session.** A controller that holds
  instance variables is invisible to snapshotting. Move it to a slice.
- **Adapter-specific transcript shapes.** Tooling that knows
  per-adapter structure breaks every time a new adapter is added. Use
  the canonical entry types.
- **Bundles that omit context.** A bundle without the prompt, the
  config, or the environment is a memo from a missing person. Capture
  everything.

______________________________________________________________________

## Pointers

- [STATE](05-STATE.md) — the dispatcher and slice substrate.
- [TRANSACTIONS](11-TRANSACTIONS.md) — how snapshots wrap tool calls.
- [ADAPTERS](13-ADAPTERS.md) — how each runtime feeds the unified
  observability stack.
- [AGENT-LOOP](15-AGENT-LOOP.md) — the loop that produces debug
  bundles per execution.
- [EVAL-LOOP](16-EVAL-LOOP.md) — per-sample bundles for evaluation
  runs.
- [PRINCIPLES](PRINCIPLES.md) §7 — inspectability over activity logs.

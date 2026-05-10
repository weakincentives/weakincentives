# Remote Execution

A production-usable WINK adapter is designed to operate against a
**remote sandbox** that runs the agent harness, and a **remote
filesystem** that hosts the agent's working files. The orchestrator
and the sandbox are separate processes — typically separate hosts —
communicating only through a protocol. This is the architectural
posture WINK assumes for production. Local in-process execution is a
*degenerate case* of the same design, useful for development but never
the target.

This is one of the most important design properties of the system.
Skipping it bakes in local-only assumptions that do not transfer to
production deployments.

______________________________________________________________________

## Why remote

Production unattended agents have requirements that the orchestrator
process cannot satisfy on its own.

- **Isolation.** Multi-tenant, untrusted-input, security-sensitive
  agents cannot share a process with the orchestrator. A vulnerability
  in tool execution must not reach the orchestrator's secrets, code,
  or other tenants' data.
- **Scale.** Hundreds or thousands of concurrent evaluations require a
  pool of sandboxes that the orchestrator dispatches to. A single
  process cannot hold them all.
- **Resource limits.** CPU, memory, disk, and network ceilings are
  easier to enforce on a dedicated runtime that the orchestrator
  cannot accidentally exceed.
- **Operational independence.** Sandboxes can be upgraded, restarted,
  or reaped without restarting the orchestrator. The orchestrator
  treats them as fungible workers.
- **Geographic placement.** Sandboxes can run near the data they
  operate on; orchestrators run near the users they serve. Mixing the
  two on one host forces a trade-off neither side wants.

These are not aspirational properties. They are table stakes for
running unattended agents in production, and the architecture of the
adapter and filesystem must respect them from the beginning.

______________________________________________________________________

## The two boundaries

```
┌─────────────────────────────┐                  ┌──────────────────────┐
│  Orchestrator               │                  │  Sandbox             │
│  ─ AgentLoop                │   protocol       │  ─ Harness           │
│  ─ Adapter (RPC client)     │ ◄══════════════► │     (planning loop,  │
│  ─ Session, observability   │                  │     native tools)    │
│  ─ Policies, feedback       │                  │  ─ Filesystem        │
└─────────────────────────────┘                  └──────────────────────┘
```

Two surfaces are remote.

- The **harness** runs in the sandbox: it drives the planning loop,
  invokes native tools, and reports back. The adapter on the
  orchestrator side is its RPC client.
- The **filesystem** lives on (or near) the sandbox. The agent reads
  and writes files through a protocol; the orchestrator does not share
  file descriptors or paths with the sandbox.

Everything else — prompt rendering, session state, policies, feedback,
completion checking, evaluation — runs on the orchestrator side. The
protocol is the only thing crossing the boundary.

______________________________________________________________________

## Local execution is a degenerate case

A local in-process harness — the simplest possible adapter — is the
case where the protocol runs over an in-memory channel and "the
sandbox" is a subprocess on the same host. The protocol is identical
to the remote case; only the transport differs.

Designing for remote means local also works. Designing for local does
not transfer to remote: it bakes in assumptions about shared memory,
shared filesystem, in-process callbacks, and zero-latency operations
that quietly break the moment the sandbox moves to another host.

The rule: never assume local. Even when running in-process, treat the
adapter as if it were a remote client.

______________________________________________________________________

## Implications for adapters

Every adapter operation is **RPC** — a request the adapter sends, a
response the sandbox returns.

- **No shared memory.** Tool params and results serialize across the
  boundary. Whatever the adapter sees in its language is the result of
  decoding what came back over the wire.
- **No in-process callbacks.** The harness cannot call back into the
  orchestrator's address space. When the harness wants to invoke a
  bridged tool, it sends a request; the adapter handles it and replies.
- **Latency is real.** Round-trips count. The adapter does not assume
  free calls — bridged tool dispatch is on the critical path of every
  turn.
- **Network errors are first-class.** They are distinct from tool
  errors. A tool that fails returned a response; a network error means
  no response was received and the adapter must reconcile state with
  the sandbox before retrying.
- **Connection management is real work.** Handshake, keepalive,
  reconnect, drain, and shutdown are part of the adapter's contract,
  not afterthoughts.
- **Streaming is the norm.** Long tool results, verbose transcripts,
  and large file reads all stream. Buffering everything in memory is a
  fallback, not the default.

______________________________________________________________________

## Implications for the filesystem

The filesystem the agent operates on is a **service**, not a directory.

- **Streaming-first protocol.** Reads return iterators of byte chunks;
  writes accept iterators of byte chunks. The convenience layer
  ("read all bytes", "write all bytes") is a wrapper, not the primary
  API.
- **Path-addressed, not handle-addressed.** Every operation is
  independent and identified by path. There are no cross-call file
  handles that assume a persistent connection.
- **Idempotent writes.** A write retried after a network failure must
  produce the same observable result as a successful write. The
  protocol enables this through content addressing or explicit
  transaction tokens, not by hoping the second attempt is a no-op.
- **No path traversal across the boundary.** The orchestrator does not
  ask the sandbox to read `/etc/passwd`. Path validation happens on
  the sandbox side; the orchestrator only sees the paths the protocol
  exposes.
- **Snapshot and restore are protocol operations.** The sandbox
  captures filesystem state on request, returning an opaque token. The
  orchestrator sends the token back to restore. The mechanism does not
  assume the orchestrator can directly inspect or copy the sandbox's
  storage.

______________________________________________________________________

## Implications for workspaces

A workspace is a **staged set of files shipped to the sandbox** before
evaluation begins, not a temporary directory on the orchestrator host.

- **Allowed-roots validation runs before transmission.** The
  orchestrator decides which host files are eligible to ship; the
  sandbox does not see anything else.
- **The sandbox owns workspace cleanup.** When the prompt's resource
  context closes, the sandbox tears down the workspace. The
  orchestrator does not unlink files on the sandbox host directly.
- **Reference counting is protocol-level.** When two prompts share a
  workspace, both ends track the share count via the protocol — not
  via shared filesystem inodes.

______________________________________________________________________

## Implications for skills

Skills are **uploaded to the sandbox** before evaluation begins. The
sandbox mounts them through its own ephemeral-home mechanism. The
orchestrator does not assume the sandbox can read skill bundles from
the orchestrator's local disk.

The skill format is portable across this boundary precisely because it
is a directory of files plus a manifest — no native code, no
in-process state, no cross-process pointers. It ships, mounts, and
runs.

______________________________________________________________________

## Implications for transactions

Filesystem rollback in a remote topology is a **server-side
operation**.

- The sandbox captures its filesystem state when the orchestrator
  signals "snapshot before tool call."
- The sandbox restores from that snapshot when the orchestrator
  signals "tool failed, roll back."
- The orchestrator does not directly manipulate sandbox storage.

Two consequences follow:

- **Network failures during rollback need a cleanup path.** If the
  client connection drops mid-rollback, the sandbox must either
  complete the rollback or unwind the snapshot itself. The
  orchestrator cannot assume the sandbox knows what to do without
  explicit signaling.
- **Snapshots are bounded.** The sandbox has finite storage for
  in-flight snapshots. The protocol carries explicit lifetime
  semantics — snapshots live until commit, rollback, or timeout.

______________________________________________________________________

## Implications for observability

Logs, transcripts, and metrics originate in the sandbox and flow back
to the orchestrator over the protocol.

- **Sandbox-side log streaming.** Logs are emitted as the sandbox
  runs and stream back, not collected at the end.
- **Transcripts are bridged.** The unified transcript format from
  [Observability](14-OBSERVABILITY.md) is fed by a sandbox-side bridge
  that translates native runtime signals into the canonical envelope.
- **Debug bundles assemble across the boundary.** The orchestrator
  collects what it has (request, configuration, session state); the
  sandbox contributes what it has (filesystem snapshot, transcript
  files, resource metrics). The bundle is finalized on the orchestrator
  side using both contributions.

______________________________________________________________________

## Implications for deadlines and budgets

Constraints are declared by the orchestrator and **enforced on both
sides**.

- The sandbox enforces them with hard limits — kills runaway
  processes, returns errors when budgets exhaust.
- The orchestrator enforces them as client-side timeouts — guards
  against a sandbox that fails to enforce them itself.
- Both sides agree on the constraints via the initial handshake.
  Mismatches surface immediately, not after work begins.

A deadline of "two minutes from now" is meaningless without
synchronized clocks. The protocol either ships a relative deadline
(seconds remaining) or both sides agree on an absolute clock; relative
is simpler and avoids skew issues.

______________________________________________________________________

## What this means for adapter authors

- **Define the protocol first.** Decide what the orchestrator sends
  and what the sandbox returns *before* writing the client code.
- **Local-only adapters are a special case.** They are useful for
  development; they are not the architecture. An adapter that cannot
  evolve from local to remote without redesign is the wrong
  abstraction.
- **Never use shared filesystem paths between adapter and harness.**
  If a tool wants to read a file, the read goes through the protocol.
- **Design for streaming.** Long tool results, large file reads, and
  verbose transcripts all need streaming. Decide up front; do not
  retrofit.
- **Treat the network as part of the contract.** Connection failures,
  partial responses, retries, and idempotency are first-order
  concerns, not edge cases.

______________________________________________________________________

## Anti-patterns

- **Shared filesystem paths.** An adapter that hands the harness a
  local path and expects the harness to read it is local-only. It will
  not survive deployment.
- **In-process tool callbacks.** A bridged tool implemented as "the
  harness calls a function in the orchestrator" assumes shared memory.
  Remote harnesses cannot do this.
- **Buffer-everything responses.** A tool that returns a multi-megabyte
  result in a single message blocks streaming and breaks for slow
  networks. Stream by default; buffer only when the payload is small.
- **Implicit handshake.** An adapter that just starts sending requests
  and assumes the sandbox is ready will hang or misbehave on slow
  starts. Protocols have lifecycles; adapters honor them.
- **Treating local as the model.** "It works in development; we'll
  figure out remote later" is how local-only adapters get shipped.
  Remote first; local as a special case.
- **Snapshot tokens that leak orchestrator-side state.** A snapshot
  token must be opaque from the orchestrator's perspective.
  Orchestrators that try to reach into the sandbox's snapshot storage
  break the boundary.

______________________________________________________________________

## What this is not

- **Not a deployment guide.** This doc describes the design posture,
  not how to operate sandboxes in production.
- **Not a protocol specification.** Each adapter binds to a specific
  protocol — Claude Agent SDK, Codex App Server, ACP, and so on. This
  doc captures the properties any such protocol must have.
- **Not an argument against local execution.** Local-in-process
  adapters are useful for development, testing, and lightweight
  scripting. The point is that they are a special case, not the
  architecture.

______________________________________________________________________

## Pointers

- [DEFINITION-VS-HARNESS](01-DEFINITION-VS-HARNESS.md) — why the
  harness is rented in the first place.
- [ADAPTERS](13-ADAPTERS.md) — the adapter contract, designed for
  remote operation.
- [TRANSACTIONS](11-TRANSACTIONS.md) — server-side snapshot and
  rollback semantics.
- [OBSERVABILITY](14-OBSERVABILITY.md) — sandbox-to-orchestrator
  telemetry flow.
- [RESOURCES](09-RESOURCES.md) — how remote resources (filesystem,
  workspace) are bound and resolved.
- [PRINCIPLES](PRINCIPLES.md) §15 — *Remote by design.*

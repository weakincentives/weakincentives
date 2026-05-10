# Durable Work

Two ideas hold together: **work identity is the orchestrator's**, and
**transport is not ownership**. Together they make agent work survive
the things that should not destroy it — a connection drop, a process
restart, a sandbox reboot — while staying explicit about the things
that actually do destroy it.

This is what makes WINK production-usable as an unattended-agent
platform. Without these ideas, every transport hiccup is a correctness
problem. With them, transport is a routine concern that the system
handles by design.

______________________________________________________________________

## The shape of the problem

Production agents do real work for minutes or hours at a time. During
that work:

- Connections drop. Sockets time out. Networks blip.
- Orchestrator processes restart. Deployments roll. Workers come and
  go.
- Sandboxes pause and resume. Compute is reclaimed and reallocated.
- Multiple orchestrators coordinate handoffs.

If any of these events destroyed the work in progress, agents would
constantly be redoing things. Tool calls with side effects would
double-execute. Long evaluations would have to start over after every
network blip.

The system is designed so that none of this happens. Work is a durable
artifact; transport, compute, and connection lifecycles surround it
without affecting it.

______________________________________________________________________

## Three lifecycles

Three lifecycles must remain distinct.

**Compute lifecycle.** A sandbox is started, runs, may sleep, may be
stopped. Compute resources come and go.

**Work lifecycle.** A session is created, accumulates state, hosts
zero or more evaluations, and is eventually deleted. State persists
across compute pauses and orchestrator restarts.

**Connection lifecycle.** An orchestrator connects to a sandbox,
exchanges messages, and disconnects. Connections are short relative
to the work they coordinate.

```
┌───────────────────────────────────────────────────────────────┐
│                                                               │
│  Compute      ──start──┬───────────────────────┬──stop───►    │
│                        │                       │              │
│                        │   ┌───────────────┐   │              │
│  Work         ─────────┼───┤  session(s)   ├───┼──────►       │
│                        │   └───────────────┘   │              │
│                        │   ┌──┐ ┌──┐ ┌──┐      │              │
│  Connection   ─────────┼───┘  └─┘  └─┘  └──────┼──────►       │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

Treating these as one lifecycle is the trap. "Stop the sandbox"
should not delete sessions. "Disconnect" should not cancel
evaluations. "End this evaluation" should not destroy the workspace
files. Each event acts on its own scope and only its own scope.

Sessions outlive connections. Workspace files outlive compute
restarts. Compute can sleep without losing work. Each lifecycle has
its own start, stop, and recovery semantics.

______________________________________________________________________

## Work identity is the orchestrator's

The orchestrator names every durable unit of work — a session, a
request, an evaluation — using a stable identifier it owns.

This is the lever. By having the orchestrator pick the name, three
things become true.

- **Idempotency is explicit.** If the orchestrator sends the same
  identifier twice with the same content, the sandbox returns the
  same result. There is no "is this a retry?" guessing.
- **Reattachment is possible.** A second orchestrator instance can
  pick up an in-flight evaluation by sending the same identifier on a
  new connection. The sandbox already knows what work that name
  refers to.
- **Observability is portable.** A session identifier means the same
  thing whether the harness is Claude, Codex, OpenCode, or something
  not yet shipped. Backend-native handles stay behind the protocol.

The corresponding rule is that **backend identifiers do not cross the
boundary**. The orchestrator never sees the harness's internal session
ID, the provider's trace token, or the model's request handle. It
only sees its own names — the ones it chose — reflected back. This
is what allows the protocol to be portable across runtimes.

______________________________________________________________________

## Idempotent execution

A request is **content-addressed**. Its identity has two parts:

- An orchestrator-supplied identifier (a name the orchestrator
  chose).
- A hash over the request's contract — input, declared tools, model,
  deadline, structured output type, and anything else that could
  change behavior.

This produces three deterministic outcomes:

- **Same identifier + same contract = same work.** A retry returns
  the in-flight or completed result. There is no double execution.
- **Same identifier + different contract = explicit conflict.** The
  sandbox refuses with a clear error. The orchestrator decides
  whether to use a new identifier, abandon the change, or accept the
  original.
- **Different identifier = different work.** Even if the content
  matches, the orchestrator has signaled this is a separate unit.

The contract hash matters because it makes "same retry" meaningful.
Without it, a retry could silently change behavior — different
deadlines, different tool definitions, different output schemas. With
it, the orchestrator either commits to the same intent or signals an
intentional change.

______________________________________________________________________

## Reattach over reconnect

When the orchestrator disconnects from the sandbox, the sandbox does
not destroy the work. It enters a **detach grace window** during
which:

- Active evaluations continue running.
- Pending tool calls remain pending, waiting for fulfillment.
- Sessions remain attached to their compute.
- Workspace files stay in place.

A new orchestrator (or the same orchestrator on a new connection) can
**reattach** by sending the same session identifier. The sandbox
returns a snapshot of where things stand: in-flight evaluations,
unresolved tool calls, recent terminal results. The new orchestrator
picks up.

If the grace window expires without reattach, the sandbox closes out
gracefully: pending tool calls are failed with a clear reason,
in-flight evaluations are recorded as failed for forensics, and the
session's work generation is invalidated. The workspace remains;
durable records remain; only live state is released.

The grace window is a tunable. The principle is that connection drops
in the middle of valuable work are recoverable, not catastrophic.

______________________________________________________________________

## Workspaces are the long-lived data plane

Because the work lifecycle is independent of compute, the workspace
that hosts the agent's files is too. Files written in one evaluation
are still there for the next, even if the sandbox stopped and
restarted in between. Files written before a disconnect are present
after a reattach. The workspace is the place for everything the agent
should be able to read across runs: previous artifacts, accumulated
knowledge, work-in-progress files. Treating it as an ephemeral
scratch directory squanders this.

The transactional snapshot/restore semantics from
[Transactions](11-TRANSACTIONS.md) still apply *within* an evaluation:
a failed tool rolls back workspace mutations to their pre-call state.
The workspace as a whole, though, is durable across evaluations.

(See [Remote Execution](18-REMOTE-EXECUTION.md) for staging,
allowed-roots, and the protocol mechanics.)

______________________________________________________________________

## Tools are the only outbound path

The sandbox's egress is restricted to model and provider traffic.
Application-level outbound — database lookups, API calls, internal
service integration — flows through orchestrator-fulfilled tool calls.
What the agent can reach is exactly what the orchestrator declared
as a tool. There is no implicit network access; no shadow channel.

This makes durable work auditable. Every outbound action is a tool
call, recorded and observable through the same event stream as
everything else; every call carries the orchestrator's identifiers,
not the agent's. (See [Tools](04-TOOLS.md) for the full treatment of
the capability surface.)

______________________________________________________________________

## Tenant separation

Sandboxes are addressed under a tenant scope. Two consequences follow.

- **Identity is namespaced.** A session name in one tenant does not
  collide with the same name in another. The orchestrator can pick
  short, meaningful identifiers without worrying about cross-tenant
  conflicts.
- **Isolation is structural.** A sandbox in one tenant cannot reach
  another tenant's sandboxes, workspaces, or sessions. The
  segmentation is a property of the protocol surface, not a
  convention.

Authentication and authorization run at the tenant boundary. An
orchestrator presents credentials for the tenants it is permitted to
operate in; the protocol denies anything else.

______________________________________________________________________

## What this is not

- **Not a queue replacement.** The orchestrator may or may not be
  driven by a message queue (see [AgentLoop](15-AGENT-LOOP.md)). The
  durable-work properties describe the sandbox-side semantics
  regardless of how the orchestrator is structured.
- **Not a substitute for backups.** Workspace durability is
  operational, not archival. Catastrophic infrastructure failure is
  still possible; important data should be backed up out of band.
- **Not infinite tolerance.** The detach grace window is bounded.
  Sessions can be deleted explicitly. Work has a finite lifetime; the
  point is that *transient* events do not end it prematurely.

______________________________________________________________________

## Anti-patterns

- **Treating disconnect as cancel.** An orchestrator that aborts
  work on every disconnect creates duplicate execution and lost
  progress. Reattach instead.
- **Server-generated identifiers in public payloads.** If the
  orchestrator has to remember a backend's session ID to find its
  own work, the boundary is leaking. The orchestrator's name is the
  only handle.
- **Reusing identifiers with different contracts.** A "retry" with a
  modified deadline or a changed tool list is a new request, not the
  same one. Use a fresh identifier and accept that the sandbox
  cannot pretend otherwise.
- **Confusing compute lifecycle with work lifecycle.** Restarting
  the sandbox does not delete sessions; deleting a session does not
  stop the sandbox. Don't pile these onto each other.
- **Treating the workspace as scratch.** It is durable storage. Use
  it for what should persist; clean it deliberately, not by accident
  of the sandbox restarting.
- **Ambient network access from the agent.** If the agent can reach
  the network outside of declared tools, the capability surface is
  no longer auditable. Restrict egress; surface capability through
  tools.

______________________________________________________________________

## Pointers

- [REMOTE-EXECUTION](18-REMOTE-EXECUTION.md) — the boundary that
  durable work crosses.
- [AGENT-LOOP](15-AGENT-LOOP.md) — how the orchestrator structures
  requests around durable identifiers.
- [STATE](05-STATE.md) — what session state actually contains.
- [TRANSACTIONS](11-TRANSACTIONS.md) — per-tool atomicity inside
  durable evaluations.
- [TOOLS](04-TOOLS.md) — the only sanctioned outbound path.
- [PRINCIPLES](PRINCIPLES.md) §16 — *Work identity is the
  orchestrator's.*
- [PRINCIPLES](PRINCIPLES.md) §17 — *Transport is not ownership.*

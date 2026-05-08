# Tools

A **tool** is the agent's only sanctioned way to take action. Everything
the model wants to *do* — read a file, call an API, mutate state, query a
service — happens through a tool. Everything else — render, reduce,
reason — is pure. This is the side-effect boundary of the system.

______________________________________________________________________

## The capability surface

A tool has:

- A stable name (lowercased, hyphenated identifier).
- A short description.
- A typed parameter record (the model's input shape).
- A typed result record (the value returned to the model and to other
  tools).
- A handler — a function that takes typed parameters and a context, and
  returns a typed result.
- Optional examples that ground the model's understanding of valid usage.
- A flag for whether the description may be overridden by prompt overrides.

The handler signature is uniform: `(params, *, context) → ToolResult`.
That uniformity is what lets adapters bridge tools to any harness without
custom glue per tool.

______________________________________________________________________

## Tools live on sections

Tools do not exist independently. They are attached to sections. A section
that explains "how to query inventory" is the section that declares the
`query_inventory` tool. This co-location is enforced structurally: a tool
without a section has no reachability and no visibility.

The consequence is that *the prompt is the catalog*. There is no separate
listing of tools to consult. Walking the section tree tells you what the
agent can do.

______________________________________________________________________

## ToolResult and the success contract

Tools never abort the run. Whatever happens inside the handler — success,
expected failure, unexpected exception — the framework converts it into a
`ToolResult` and returns it to the model.

A successful result carries:

- A typed `value` (the structured payload).
- An optional `message` forwarded to the model as plain text.
- A success flag set true.

A failure result carries:

- A `message` describing what went wrong, in language the agent can
  reason about.
- An optional partial `value` (often `None`).
- A success flag set false.

The model receives the failure as a normal tool result and decides how to
respond. This is what allows recovery to be a property of the *prompt* —
the agent can read the failure message and replan — instead of a property
of the harness.

______________________________________________________________________

## Why failures don't abort

If a single tool failure crashed the run, the agent would have no chance to
diagnose, retry, or work around it. Models are generally good at reacting
to "your tool call failed because X" if the message is informative.
Treating tool failure as data — not as control flow — preserves the
agent's reasoning path and produces dramatically better recovery behavior.

This requires tool authors to write *informative* failure messages. "File
not found" is fine; "operation failed" is not.

______________________________________________________________________

## Transactional execution

Each tool invocation is an atomic transaction. Before the handler runs, the
framework snapshots session state and any snapshotable resources (e.g., the
filesystem). On failure — handler raised, validation failed, policy denied
— the snapshot is restored and the agent sees the failure cleanly. On
success, the snapshot is discarded and any state mutations stand.

The implication: a failed tool leaves no trace. There are no half-applied
changes to clean up, no inconsistent rows to detect, no partially-written
files. This is what makes aggressive retry safe.

(See [Transactions](11-TRANSACTIONS.md) for the full mechanics.)

______________________________________________________________________

## What a handler can see

The tool context passed to a handler exposes:

- The active prompt and its rendered form.
- The session — for querying state and dispatching events.
- The deadline and budget tracker, when one is in force.
- The resource registry — for accessing dependencies bound at the prompt
  level.
- A heartbeat, for long-running operations to extend their visibility
  timeout.
- The active run context — correlation IDs that tie this tool call to the
  larger evaluation.

Two patterns matter:

- Tools *publish* events through the session dispatcher. They do not
  mutate session state directly. State changes flow through reducers.
- Tools *resolve* their dependencies through the resource context, not
  through global state. A tool that reaches around the resource registry
  to import its dependencies bypasses scoping, snapshotting, and test
  substitution.

______________________________________________________________________

## Examples as documentation

A tool may carry zero or more examples — pairs of input and expected
output. Examples ground the model's understanding of valid usage and serve
as test fixtures. They are validated against the parameter and result
record types at construction; an invalid example is a build-time error,
not a runtime mystery.

______________________________________________________________________

## Bridging to the harness

Different runtimes expose tools in different ways: an in-process MCP
server, a stdio dynamic-tools protocol, an MCP HTTP endpoint, and so on.
Adapters translate the same tool definition into whichever shape the
harness expects. The tool author writes one handler; adapters do the
rest. (See [Adapters](13-ADAPTERS.md).)

______________________________________________________________________

## What tools are not

- **Not workflow steps.** A tool is a capability, not a node in a
  predetermined sequence. The agent decides when to call it.
- **Not the place for orchestration.** A tool that calls another tool is
  rare; chained behavior is the agent's job. Tools are atomic units.
- **Not silent.** A tool that performs side effects without telling the
  model what happened (or what failed) defeats the agent's reasoning loop.
- **Not the home for cross-cutting policy.** "Always check X before doing
  Y" is a *policy*, not a precondition repeated in every handler.

______________________________________________________________________

## Pointers

- [SECTIONS](03-SECTIONS.md) — where tools live and how they reach the
  prompt.
- [TRANSACTIONS](11-TRANSACTIONS.md) — atomicity and rollback.
- [POLICIES](06-POLICIES.md) — gates around tool calls (fail-closed).
- [TYPED-CONTRACTS](12-TYPED-CONTRACTS.md) — the typed-record
  discipline that makes parameter and result types reliable.
- [ADAPTERS](13-ADAPTERS.md) — how tools reach a runtime.
- [STATE](05-STATE.md) — what session state tools can read and what
  events they can publish.

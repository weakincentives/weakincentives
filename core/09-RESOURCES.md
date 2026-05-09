# Resources

A **resource** is anything a tool, section, or adapter needs but should
not construct itself: a filesystem, a database client, an HTTP client, a
budget tracker, a clock. Resources are managed through a typed dependency
injection system with explicit lifetimes and lifecycle hooks. The system
is small, lazy, and built for inspection.

______________________________________________________________________

## The shape

A resource registry is a collection of **bindings**. Each binding
associates:

- A **protocol** (the interface the consumer asks for).
- A **provider** (a function that produces an instance, given a resolver
  for upstream dependencies).
- A **scope** (how often the provider runs).
- An optional **eager** flag (initialize at startup, not on first use).
- An optional **pre-constructed instance** (skip the provider entirely).

A consumer asks the registry for the protocol it needs and gets back an
instance. Lifetimes, ordering, and cleanup are the registry's job.

______________________________________________________________________

## Why dependency injection

Three reasons matter most.

**Testability.** A test substitutes a fake by replacing one binding,
without ad-hoc runtime patching or threading globals. The same code path
runs in production and in tests.

**Lifecycle correctness.** When a prompt's resource context exits, every
resource bound to that prompt cleans up — closeables close, temp
directories disappear, network clients shut down — in reverse
construction order. There is no "finally" block to forget.

**Snapshot integration.** Some resources (the in-memory filesystem, the
host filesystem) are *snapshotable*. Their state participates in
transactional rollback alongside session state. Without explicit
lifecycle binding, this would not be possible.

______________________________________________________________________

## Scopes

Three scopes define when a provider runs.

- **Singleton.** One instance per resource context. The provider runs the
  first time anyone asks for the protocol, and the result is cached for
  the rest of the context. This is the default.
- **Tool call.** A fresh instance for each tool invocation. Useful for
  per-call state — a request tracer, a transaction-scoped client, a
  buffer that should not be reused.
- **Prototype.** A fresh instance every time anyone asks. Useful for
  cheap, stateful builders that should not be shared.

The scope is part of the binding, not the consumer's call. Consumers ask
for "the filesystem" the same way regardless of whether the underlying
binding is singleton or tool-scoped.

______________________________________________________________________

## How tools see resources

A tool handler receives a context that exposes the resource registry
through a stable accessor. The handler asks for what it needs:

- "Give me the filesystem."
- "Give me the budget tracker."
- "Give me the configured rate limiter."

The handler does not reach for a process-wide singleton. It does not
construct its own client. It does not pull anything in from a global. The
explicit access is what makes scope, lifecycle, and substitution work.

______________________________________________________________________

## Where bindings come from

Bindings can be declared at three points, with later layers overriding
earlier ones:

- The **prompt template** declares the resources every binding of that
  prompt should have. This is where stable, agent-wide dependencies live.
- A **section** can contribute resources via its `resources()` method.
  This lets a section that *needs* a temp directory or a filesystem
  declare it without forcing every consumer of the prompt to know.
- A **bind-time override** at the call site can replace any binding for a
  specific prompt instance. This is how tests inject fakes and how
  callers swap implementations per run.

The merge is straightforward: bind-time overrides win, then section
contributions, then template defaults.

______________________________________________________________________

## Lifecycle hooks

A resource may opt into three lifecycle protocols:

- **Closeable.** Implement a `close()` method; the registry calls it
  during context exit. Closes run in reverse construction order so
  dependencies survive their dependents.
- **Post-construct.** Implement a `post_construct()` method; the registry
  calls it once after the provider returns. Useful for side effects that
  should fail loudly if they fail at all (network handshake, schema
  validation). A failed post-construct prevents caching.
- **Snapshotable.** Implement `snapshot()` and `restore()`; the
  transaction layer captures resource state before each tool call and
  restores on failure.

These are *opt-in* — most resources are simple values that need none of
them.

______________________________________________________________________

## Cycles and missing dependencies

The registry resolves dependencies as a graph. If two providers depend on
each other, the system detects the cycle and raises with the path so the
fix is clear. If a provider asks for a protocol that no binding satisfies,
the system raises immediately with the protocol name. There is no silent
"None" returned to the caller.

These checks happen at first use. The registry is lazy by default, so
unused bindings never trigger their providers.

______________________________________________________________________

## Snapshotable resources

Two resources matter especially: the in-memory filesystem (used in tests
and evaluations) and the host filesystem (used in production). Both are
snapshotable. Their state is captured before each tool call and restored
on failure as part of transactional execution.

This is what makes "rollback" mean something tangible. If a tool writes
files and then fails, the filesystem rolls back too — not just the
session. The tool leaves no trace of any kind.

This depends entirely on the resource being managed by the registry.
Resources constructed inside a tool handler, or pulled in from a global,
are invisible to the snapshot layer and cannot participate in rollback.

______________________________________________________________________

## Anti-patterns

- **Module-level singletons.** A globally-scoped client cannot be
  substituted in tests, cannot participate in lifecycle, and cannot be
  snapshotted. Bind it.
- **Tool handlers that construct their own dependencies.** This bypasses
  scoping and substitution. Take the dependency as a resource.
- **Resources whose lifetime is the process.** If your client is a
  process-wide singleton, that is fine — but bind it as a singleton
  resource so tests can replace it.
- **Reaching around the resolver.** Resources should ask the resolver for
  upstream dependencies, not import them. Otherwise the dependency graph
  is invisible to the system.

______________________________________________________________________

## Pointers

- [PROMPT-IS-THE-AGENT](02-PROMPT-IS-THE-AGENT.md) — how prompts own
  their resource lifecycle.
- [SECTIONS](03-SECTIONS.md) — how sections contribute resources.
- [TOOLS](04-TOOLS.md) — how tools access resources through their
  context.
- [TRANSACTIONS](11-TRANSACTIONS.md) — how snapshotable resources
  participate in rollback.
- [AGENT-LOOP](15-AGENT-LOOP.md) — the loop that opens and closes
  the prompt's resource context per execution.
- [PRINCIPLES](PRINCIPLES.md) §11–§12 — scoped resources and injected
  time.

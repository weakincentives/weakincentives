# Glossary

One-line definitions for fast lookup. Each entry links to the concept doc
that explains it in depth.

______________________________________________________________________

## Foundational

- **Agent definition** — The prompt, tools, policies, feedback, and
  completion criteria you own and version. See
  [Definition vs. Harness](01-DEFINITION-VS-HARNESS.md).
- **Execution harness** — The runtime that provides the planning loop,
  sandboxing, retries, and orchestration. See
  [Definition vs. Harness](01-DEFINITION-VS-HARNESS.md).
- **WINK / Weak Incentives** — The agent-definition layer. The bet that a
  well-shaped prompt, typed tools, and explicit constraints make the
  correct path also the easiest path. See
  [Principles](PRINCIPLES.md).

## Prompt structure

- **Prompt template** — The immutable description of an agent: namespace,
  key, section tree, attached policies, attached feedback providers,
  attached completion checker, optional structured output type. See
  [Prompt is the Agent](02-PROMPT-IS-THE-AGENT.md).
- **Prompt** — A template plus its bound runtime parameters; the thing
  you actually evaluate.
- **Section** — The composition unit of a prompt: instructions, tools,
  skills, children, visibility, enabled predicate. See
  [Sections](03-SECTIONS.md).
- **Skill** — A directory bundle (instructions plus optional scripts and
  references) mounted into the runtime. Attached to a section like a
  tool.

## Capability surface

- **Tool** — A typed handler for an action with explicit parameter and
  result record types. The agent's only sanctioned side-effect surface.
  See [Tools](04-TOOLS.md).
- **ToolResult** — The success-or-failure container returned by every
  tool. Failures never abort the run.
- **Tool context** — The bundle the framework hands to a handler:
  prompt, session, deadline, budget, resources, run context.

## Control mechanisms

- **Policy** — A hard guardrail that gates tool invocations. Fail-closed.
  See [Policies](06-POLICIES.md).
- **Feedback provider** — A soft guidance mechanism that injects
  advisory messages during execution. Non-blocking. See
  [Feedback](07-FEEDBACK.md).
- **Completion checker** — A verification step at termination that
  blocks early stops when the agent's stop condition is unmet. See
  [Completion Checking](08-COMPLETION-CHECKING.md).
- **Three-tier control** — The combination of policies, feedback, and
  completion checking; together they constrain tool calls, nudge
  trajectory, and verify termination.

## State

- **Session** — The container that holds slices, registers reducers, and
  dispatches events. The unit across which all mutations flow. See
  [State](05-STATE.md).
- **Event** — A typed record describing something that happened. The
  only mechanism by which state changes.
- **Slice** — A typed view of one aspect of session state, addressed by
  type.
- **Reducer** — A pure function that takes the current slice and an
  event and returns a slice operation.
- **Slice operation** — A description of how to update a slice: append,
  extend, replace, clear.
- **Slice policy** — Either `STATE` (snapshotted, rolls back with
  transactions) or `LOG` (append-only ledger preserved across rollbacks).
- **Snapshot** — An immutable capture of session state, serializable to
  JSON and restorable.

## Resources

- **Resource** — Anything injected into a tool, section, or adapter
  through the registry. See [Resources](09-RESOURCES.md).
- **Binding** — The pairing of a protocol, a provider, and a scope.
- **Scope** — The lifetime of a resource instance: singleton (per
  context), tool-call (per invocation), prototype (per resolution).
- **ResourceRegistry** — The collection of bindings the prompt's resource
  context will resolve.
- **Closeable / PostConstruct / Snapshotable** — Optional lifecycle
  protocols a resource may implement.

## Disclosure

- **Visibility** — A section property: `FULL` or `SUMMARY`. Summary
  withholds the section's tools until expansion. See
  [Progressive Disclosure](10-PROGRESSIVE-DISCLOSURE.md).
- **Visibility expansion** — The mechanism by which an agent requests
  more detail; raises a typed signal that re-renders the prompt.

## Atomicity

- **Transaction** — The atomic boundary around a tool invocation:
  snapshot before, execute, restore on failure, commit on success. See
  [Transactions](11-TRANSACTIONS.md).
- **Rollback** — The restoration of session and snapshotable resources
  to their pre-call state when a tool fails.

## Type discipline

- **Typed record** — Any named, fixed-shape value type with declared
  fields and per-field types — `dataclass`, `struct`, `record`,
  `case class`, etc., depending on the host language.
- **Immutable record** — A record type that cannot be mutated after
  construction. The default shape for almost every type in WINK. See
  [Typed Contracts](12-TYPED-CONTRACTS.md).
- **Validated record** — A record whose direct construction is blocked;
  a factory enforces invariants before instances exist. The validated
  tier of [Typed Contracts](12-TYPED-CONTRACTS.md).
- **Structured output** — A typed return shape declared on the prompt;
  the framework parses model responses into the record.
- **Design by contract** — The optional layer for preconditions,
  postconditions, and invariants attached at function and type
  declarations. Captures semantic constraints the type system alone
  cannot express.

## Harness boundary

- **Adapter** — The bridge between a prompt and a specific execution
  harness. Designed as an RPC client to a remote sandbox. See
  [Adapters](13-ADAPTERS.md).
- **Adapter Compatibility Kit (ACK)** — The unified test suite that
  certifies adapters implement the WINK contract correctly.
- **Throttle policy** — The shared retry-with-backoff configuration
  adapters use to handle rate limits and transient failures.
- **Bridged tool** — A WINK tool wrapped in transactional execution
  glue, ready to be exposed through whatever protocol the harness uses.

## Remote topology

- **Sandbox** — The remote runtime where the harness and filesystem
  live. Reachable from the orchestrator only through a protocol. See
  [Remote Execution](18-REMOTE-EXECUTION.md).
- **Orchestrator** — The process that hosts the AgentLoop, adapter,
  session, and observability. Treats sandboxes as fungible workers.
- **Protocol** — The boundary between orchestrator and sandbox. Same
  shape regardless of transport (in-memory, stdio, network). Local
  execution runs the same protocol over an in-memory channel.
- **Tenant** — Segmentation scope within which sandbox identifiers
  live. Two orchestrators in different tenants cannot collide on
  identifiers and cannot reach each other's sandboxes.
- **Snapshot token** — The opaque handle a sandbox returns when it
  captures filesystem state. The orchestrator sends it back to commit
  or roll back; the orchestrator never inspects the storage directly.
- **Idempotent write** — A write that can be retried after a network
  failure without producing different observable state. Required for
  safe retries across the protocol boundary.

## Durable work

- **Work identity** — The stable identifier the orchestrator owns for
  a unit of work (a session, a request, an evaluation). The only
  durable handle that crosses the protocol boundary. See
  [Durable Work](19-DURABLE-WORK.md).
- **Contract hash** — A hash over a request's contract (input,
  declared tools, model, deadline, structured output type). Combined
  with the orchestrator's identifier to detect duplicate retries vs.
  semantic conflicts.
- **Idempotency conflict** — The result when an identifier is reused
  with a different contract. Surfaced explicitly so the orchestrator
  can choose a fresh identifier or accept the original intent.
- **Detach grace window** — The bounded period after a connection
  drop during which work is preserved for reattach. Pending tool
  calls remain pending; sessions remain attached to compute;
  workspace files stay in place.
- **Reattach** — A new connection picks up an in-flight unit of work
  using the same orchestrator-owned identifier. The sandbox returns
  the current snapshot; the new orchestrator continues from there.
- **Compute lifecycle** — The lifespan of a sandbox: started, runs,
  may sleep, may be stopped. Independent of work and connection
  lifecycles.
- **Work lifecycle** — The lifespan of a session and its
  evaluations. Independent of compute and connection lifecycles.

## Observability

- **Transcript** — A unified, adapter-agnostic log of everything that
  happened during an evaluation. See
  [Observability](14-OBSERVABILITY.md).
- **Transcript entry** — A single record in the transcript, sharing a
  common envelope across all adapters.
- **Debug bundle** — A self-contained zip archive of a run, including
  session, logs, transcript, config, environment, and (optionally)
  filesystem snapshot.
- **Dispatcher** — The publish/subscribe system through which events
  reach reducers and observers.

## Time and lifecycle

- **WallClock / MonotonicClock / Sleeper** — Narrow protocols that
  abstract real time. Production uses real implementations; tests use
  fakes that advance instantly.
- **Deadline** — A wall-clock bound applied to an entire evaluation,
  including all tool calls.
- **Budget** — A composite constraint over time and tokens, tracked
  cumulatively across an evaluation.
- **Heartbeat** — A liveness signal a long-running tool can emit to
  extend its lease.

## Iteration

- **Prompt overrides** — Hash-validated text replacements for sections
  and tool descriptions, applied without source changes. The hash means
  overrides apply only to the version they were authored against. See
  [Prompt Overrides](17-PROMPT-OVERRIDES.md).
- **Override tag** — A named group of overrides under one prompt
  namespace/key. Selected at bind time. Used by experiments for A/B.
- **Descriptor** — The hashed snapshot of a prompt's overridable
  surface. Compared at load to filter stale overrides.
- **Experiment** — A named configuration variant — overrides tag plus
  feature flags — used by the evaluation framework for A/B comparison.
  See [Evaluation](16-EVAL-LOOP.md).

## Orchestration

- **AgentLoop** — The user-facing orchestration shell. Subclass it,
  implement `prepare()`, get back a typed result. Owns prompt resource
  lifecycle, visibility-expansion retries, debug bundling. See
  [AgentLoop](15-AGENT-LOOP.md).
- **AgentLoopRequest / AgentLoopResult** — The typed envelope around
  a single execution: budget, deadline, resources, run context in;
  output, error, session ID, bundle path out.
- **Direct mode** — Calling `execute(request)` synchronously and
  receiving a typed result. The default mode for tests and one-shot
  scripts.
- **Mailbox-driven mode** — Running `run()` to consume requests from
  a message queue and reply via the queue's reply channel. The default
  mode for long-running unattended workers.

## Evaluation

- **Sample** — One test case in a dataset: identifier, input,
  expected output. Generic over both types. See
  [Evaluation](16-EVAL-LOOP.md).
- **Dataset** — An immutable collection of samples, typically loaded
  from a newline-delimited JSON file.
- **Score** — A normalized 0.0–1.0 value plus a binary pass/fail and a
  reason. The output of an evaluator.
- **Evaluator** — A pure function from output and expected to score.
  Composable via `all_of` and `any_of`.
- **Session-aware evaluator** — An evaluator that also receives the
  session, so it can assert on tool usage, token budgets, or any
  custom slice state.
- **EvalLoop** — The orchestrator that wraps AgentLoop, runs each
  sample under each experiment, scores the output, and returns a
  report.
- **EvalReport** — Aggregate statistics across an evaluation run:
  pass rates, mean scores, per-experiment breakdowns, pairwise
  comparisons.

______________________________________________________________________

## Pointers

For deeper treatment, see the numbered concept docs in this folder. For
implementation details — file paths, class signatures, error types — see
the `specs/` folder at the repository root.

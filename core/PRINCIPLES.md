# Principles

These are the rules every part of WINK follows. They are not slogans; they
are operative constraints. When two designs conflict, the one that better
satisfies these principles wins.

______________________________________________________________________

## 1. Own the definition. Rent the harness.

The agent definition — prompt, tools, policies, feedback, completion checks
— is the artifact you version, review, test, and port. The execution
harness — planning loop, sandboxing, retries, scheduling, multi-agent
orchestration — comes from a vendor runtime. The harness will keep changing.
The definition should not. Anything that belongs to the agent's *goal*
belongs in the definition; anything that is a property of *how runtimes
execute* belongs in the harness.

## 2. The prompt is the agent.

There is no separate tool registry, routing layer, or capability table. A
single hierarchical document — sections of instructions, tools, skills, and
nested children — fully determines what the agent can think and do.
Documentation cannot drift from implementation because they are the same
artifact.

## 3. Policies, not workflows.

Encode constraints, not procedures. A policy says *what must hold*; a
workflow says *what to do next*. Policies preserve the agent's reasoning;
workflows fracture on edge cases the author didn't foresee. Use workflows
only when the sequence is genuinely invariant (protocol handshakes) or the
agent lacks reasoning capacity.

## 4. Pure transitions, side effects in tools.

Rendering, state transitions, and reducers are deterministic and pure.
Mutation lives in two places only: tool handlers (which touch the outside
world) and reducers (which return new immutable state). When something goes
wrong, you know exactly where to look.

## 5. Fail closed.

When a constraint cannot be evaluated, deny by default. The agent reasons
about the denial and adjusts. This makes invariants observable instead of
silently violated.

## 6. Typed contracts everywhere.

Parameters, tool calls, tool results, structured output, events, and state
are all typed records — whatever the host language calls them
(dataclass, struct, record, case class). Type mismatches surface at
construction time, not mid-response. A strict static type checker is the
first line of defense, not a nice-to-have.

## 7. Inspectability over activity logs.

Every event is recorded as immutable state. Every snapshot is restorable.
Every adapter emits a unified transcript. Debugging is querying, not
re-running. If you cannot reconstruct a run from its session and bundle, the
run was not really observed.

## 8. Transactional capability.

Each tool call is an atomic transaction. On failure, session and filesystem
state roll back to their pre-call form. Failed tools never leave partial
state. This enables aggressive retry, recovery, and policy enforcement
without bookkeeping.

## 9. Co-locate instructions and capability.

The section that explains *how to use a tool* is the same section that
*provides* that tool. The section that documents an output format is the
same section that declares the output type. Drift becomes structurally
impossible.

## 10. Disclose progressively.

Default to summaries; expand on demand. The agent decides what depth it
needs. Keeping context small keeps reasoning sharp and tokens cheap.

## 11. Resources are scoped, not global.

Dependencies are injected through bindings with explicit lifetimes
(singleton, per-tool-call, per-resolution). Tests substitute fakes by
construction. There are no global singletons that have to be patched at
runtime.

## 12. Time is injected.

No production code reads the wall clock or sleeps directly. Both are
abstracted behind clock and sleeper protocols. Tests advance time instantly;
production runs use real clocks. The same code path drives both.

## 13. Definition assets carry their own constraints.

Policies, feedback providers, and completion checkers live on the prompt
definition — not on the adapter, not on the runtime config. They travel
with the agent across harnesses. Versioned, reviewable, portable.

## 14. The same definition runs on every harness.

If a feature only works on one runtime, it doesn't belong in the definition
layer. Adapter Compatibility Kits exist to *prove* this for each new
adapter.

## 15. Remote by design.

A production-usable adapter assumes the harness and the filesystem are
remote — running in a separate sandbox, reachable only through a
protocol. The protocol is the boundary; local in-process execution is a
degenerate case of the same design. Shared memory, shared file
descriptors, in-process callbacks, and zero-latency assumptions
silently break the moment the sandbox moves to another host. Design for
remote; local follows. Design for local and the architecture does not
transfer.

## 16. Stability is alpha by intent, not by accident.

APIs may still change. We delete unused code rather than carry shims. We do
not preserve backward compatibility for surfaces nobody uses. Old shapes
are removed; clients move to the new shape.

______________________________________________________________________

## On naming

The project is named after a concept from mechanism design: *weak
incentives*. A well-designed system shapes its participants so the easy path
is the correct one. Applied to agents, that means structuring the prompt,
tools, and context so the model's path of least resistance produces the
right behavior. WINK is the bet that *clarity beats constraint* — that
typed contracts, co-located instructions, and inspectable state encourage
correct behavior more reliably than guardrails that try to prevent every
wrong one.

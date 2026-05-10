# Core

This folder is the conceptual foundation of Weak Incentives (WINK). It is a
*library without code*: the durable ideas, the shape of the abstractions, and
the way they fit together — without API surfaces, file paths, or
implementation detail. If you point an agent (or a new contributor) at this
folder, it should walk away with an accurate, opinionated mental model of
what WINK is for and how it thinks.

For implementation depth — class signatures, behaviors, errors — see the
`specs/` folder. `core/` answers *why* and *what*; `specs/` answers *how*.

**Language- and framework-agnostic.** WINK is currently implemented in
Python, but every concept in `core/` is described in language-neutral
terms. "Typed record" means whatever the host language calls a named
fixed-shape value — `dataclass`, `struct`, `record`, `case class`. The
ideas port to Rust, TypeScript, Kotlin, Go, or any other language that
can express typed values, immutability, and dependency injection. If a
concept here can only be expressed in one language, it is the wrong
abstraction.

______________________________________________________________________

## The thesis

An unattended agent has two parts: the **definition** (prompt, tools,
policies, feedback, completion criteria) and the **execution harness**
(planning loop, sandboxing, retries, scheduling). The harness is rented
from a vendor runtime and will keep changing. The definition is what
you actually own. WINK is the layer that makes the definition a
first-class artifact — portable, typed, testable, and stable across
harness changes.

Five commitments do most of the work.

1. **The prompt is the agent.** A single typed hierarchical document —
   sections that bundle instructions and tools — fully determines what
   the agent can think and do. There is no second source of truth.
2. **Policies, not workflows.** Constraints are declared invariants,
   not prescribed sequences. The agent reasons; the framework gates.
3. **Pure transitions, side effects in tools.** State is event-driven
   and immutable. Tools are the only side-effect surface, and every
   tool call is transactional. Failed tools leave no trace.
4. **Definition assets are portable.** The same prompt, tools, policies,
   feedback, and completion checks run across harnesses. Adapters
   absorb the runtime differences.
5. **Designed for remote execution.** The harness and its filesystem
   live in a separate sandbox, reachable only through a protocol. The
   orchestrator owns work identity; transport events do not destroy
   work; compute, work, and connection are three independent
   lifecycles.

These commitments compose. Together they make the definition durable,
portable, observable, and testable — the part of an unattended agent
that should *stay* yours as runtimes evolve.

______________________________________________________________________

## How to read this folder

The order below is the teaching order. The numbers are stable
identifiers, not a strict reading sequence — but a top-to-bottom pass
gives the cleanest mental model. Within each cluster, individual docs
can be read independently.

### The thesis

- [01 DEFINITION-VS-HARNESS](01-DEFINITION-VS-HARNESS.md) — what you
  own vs. what you rent.

### The definition surface (what you build)

- [02 PROMPT-IS-THE-AGENT](02-PROMPT-IS-THE-AGENT.md) — prompts as
  typed hierarchical documents.
- [03 SECTIONS](03-SECTIONS.md) — how prompts are composed.
- [04 TOOLS](04-TOOLS.md) — the capability surface and the only
  outbound path.

### Constraints on behavior (three-tier control)

- [06 POLICIES](06-POLICIES.md) — hard guardrails (fail-closed gates).
- [07 FEEDBACK](07-FEEDBACK.md) — soft guidance during execution.
- [08 COMPLETION-CHECKING](08-COMPLETION-CHECKING.md) — "done means
  done" verification.

### Inside the runtime (state and disciplines)

- [05 STATE](05-STATE.md) — event-driven state, slices, reducers.
- [09 RESOURCES](09-RESOURCES.md) — dependency injection with scoped
  lifetimes.
- [10 PROGRESSIVE-DISCLOSURE](10-PROGRESSIVE-DISCLOSURE.md) —
  token-efficient context expansion.
- [11 TRANSACTIONS](11-TRANSACTIONS.md) — atomic tool execution with
  rollback.
- [12 TYPED-CONTRACTS](12-TYPED-CONTRACTS.md) — records everywhere;
  strict types.

### The harness boundary (reaching a runtime)

- [13 ADAPTERS](13-ADAPTERS.md) — the bridge to a specific harness.
- [18 REMOTE-EXECUTION](18-REMOTE-EXECUTION.md) — the protocol-mediated
  remote topology that adapters and the filesystem assume.
- [19 DURABLE-WORK](19-DURABLE-WORK.md) — work identity, idempotent
  execution, reattach over reconnect, three lifecycles kept distinct.

### Running it (orchestration, iteration, evidence)

- [15 AGENT-LOOP](15-AGENT-LOOP.md) — the orchestration shell you
  subclass to actually run a prompt.
- [16 EVAL-LOOP](16-EVAL-LOOP.md) — datasets, evaluators, experiments,
  A/B comparison.
- [17 PROMPT-OVERRIDES](17-PROMPT-OVERRIDES.md) — hash-validated
  prompt iteration without source changes.
- [14 OBSERVABILITY](14-OBSERVABILITY.md) — events, snapshots,
  transcripts, debug bundles.

### Reference

- [PRINCIPLES](PRINCIPLES.md) — the rules every other doc follows.
- [GLOSSARY](GLOSSARY.md) — one-line definitions for fast lookup.

______________________________________________________________________

## How the concepts connect

```
 ┌──────────────────────────────────────────────────────────┐
 │  THE DEFINITION  (you author; portable; versioned)       │
 │                                                          │
 │      Prompt is the Agent (02)                            │
 │       ├─ Sections (03) compose the prompt                │
 │       │   └─ Tools (04) — the capability surface         │
 │       ├─ Three-tier control                              │
 │       │   ├─ Policies (06)        — fail-closed gates    │
 │       │   ├─ Feedback (07)        — advisory nudges      │
 │       │   └─ Completion (08)      — verify "done"        │
 │       ├─ Resources (09)           — scoped DI            │
 │       └─ Progressive Disclosure (10) — context expansion │
 │                                                          │
 │      Iterate via Prompt Overrides (17)                   │
 └────────────────────────────┬─────────────────────────────┘
                              │
                Adapter (13)  │ — RPC over a protocol
                              ▼
 ┌──────────────────────────────────────────────────────────┐
 │  THE HARNESS  (rented; runs in a remote sandbox)         │
 │                                                          │
 │      Remote Execution (18) — boundary, filesystem,       │
 │                              workspace, skills           │
 │      Durable Work     (19) — orchestrator-owned          │
 │                              identity; three lifecycles; │
 │                              reattach over reconnect     │
 └──────────────────────────────────────────────────────────┘

  Cross-cutting disciplines             Drivers
  ─────────────────────────             ─────────────
  State            (05)                 AgentLoop (15)
  Transactions     (11)                 EvalLoop  (16)
  Typed Contracts  (12)
  Observability    (14)
```

Read the picture as **two strata** — the definition you author at the
top, the harness you rent at the bottom — connected by an **adapter**
that speaks the protocol. The harness lives in a remote sandbox, and
the system is designed for that posture from the start; local
in-process execution is a special case of the same model.

The cross-cutting columns apply on both sides. State, transactions,
typed contracts, and observability are how the system stays
deterministic, recoverable, and inspectable. AgentLoop is the
orchestration shell that drives one evaluation through the adapter;
EvalLoop wraps AgentLoop to run datasets and report results.

______________________________________________________________________

## What `core/` is not

- **Not an API reference.** No class names, signatures, or import paths.
  Map back to `specs/` and the source tree when you need that.
- **Not a tutorial.** It does not walk you through building an agent. See
  `guides/` and `README.md` at the repo root.
- **Not a changelog.** Concepts described here are the durable shape of
  WINK, not the latest delta.
- **Not exhaustive.** Adapter quirks, debug bundle formats, and other
  implementation-bound details intentionally live in `specs/`.

______________________________________________________________________

## Working rule

If a change to the codebase makes a `core/` doc inaccurate, the codebase
is probably moving away from a foundational idea. Pause and decide
whether the core idea is shifting (update `core/`) or whether the change
is local and should be reframed (keep `core/` as the anchor). The
`core/` folder is deliberately small so it can stay accurate.

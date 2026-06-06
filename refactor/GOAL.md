# GOAL — WINK: the Agent-Definition Library for Provider Harnesses

> North star for the refactor. The milestones (`M1.md` … `M16.md`) realize it,
> grouped into releases. **Backwards compatibility is a non-goal**; we improve
> abstractions, delete weight, and accept partial rewrites. Every milestone is
> independently shippable and lands green (`make check`, 100% coverage).

______________________________________________________________________

## Mission

WINK makes the agent **definition** — prompt, tools, policies, feedback, and the
execution environment — a **first-class, portable, versioned artifact** that runs
unchanged on top of any provider agent **harness** (Claude Agent SDK, Codex App
Server, ACP / OpenCode / Gemini).

> You own the **definition**. The harness owns **execution** (planning loop,
> sandboxing, retries, crash recovery, deadlines, budgets). WINK is the seam.

To be *the* library for this, three things must be true, and they are the three
releases below: the definition targets **one consistent environment** (R1), it is
a **portable artifact that behaves identically on every harness** (R2), and its
runs are **observable, reproducible, and policy-governed** (R3).

______________________________________________________________________

## First Principles

1. **Definition vs Harness.** Anything the harness can own (execution, isolation,
   recovery) is *not* in the definition. Anything that defines agent behavior
   (prompt, tools, policies, environment intent) *is*, as data.
1. **The definition is data.** A definition you cannot serialize, version, diff,
   and share is not first-class. Declarative parts are serde values; code
   (handlers) is referenced by stable identity.
1. **Parity is the promise.** The same definition must behave identically on every
   harness, or the abstraction is a lie. Parity is enforced by a conformance kit,
   not hoped for.
1. **Policies over workflows.** Behavior is governed by composable, declarative,
   fail-closed policies evaluated at decision points — not prescriptive scripts.
1. **One environment, one truth.** Every tool call — WINK-bridged and
   harness-native — targets one `Sandbox` (filesystem + shell + egress), local or
   remote, behind one interface.
1. **Narrow capabilities, ergonomic facades.** Backends implement the smallest
   primitive set; convenience composes once on top, never per backend.
1. **Reproducible by construction.** Inject clocks; content-address artifacts;
   record runs; replay deterministically.

______________________________________________________________________

## Naming Taxonomy

Suffixes carry meaning. One idiom per concept, no overlaps:

| Suffix | Means | Examples |
|--------|-------|----------|
| **`*Spec`** | Immutable, declarative description of a resource a **provider/factory materializes** (desired state); lives in the definition | `SandboxSpec` (→ any provisionable resource) |
| **`*Config`** | Tuning knobs for a **component you construct/inject** | `LLMConfig`, `…ClientConfig` |
| **`*Policy`** | **Rules/constraints** evaluated at runtime (fail-closed) | `EgressPolicy`, `NetworkPolicy`, `ThrottlePolicy` |
| **`*Params`** | Typed input to a **tool/section handler** | `ParamsT` dataclasses |
| **`*Request` / `*Result`** | A unit of **work** through a loop + its outcome | `AgentLoopRequest`/`Result`, `EvalRequest`, `ToolResult` |
| **`*Descriptor`** | Lightweight **metadata about an existing thing** | tool/section descriptors |

**Rule for `*Spec`:** if a provider/factory turns it into a live resource
(`provider.open(spec) -> Resource`), it is a `*Spec`. It is *not* a `*Config`
(component knobs), `*Policy` (rules — a spec may *contain* policies, e.g.
`SandboxSpec.egress: EgressPolicy`), `*Params` (tool input), or `*Request` (a
task). The existing `claude_agent_sdk` `SandboxConfig` (OS-isolation knobs) is a
`*Config`; it folds into the sandbox model in R1 (see M4) — there will not be both
a `SandboxSpec` and a `SandboxConfig`.

______________________________________________________________________

## End-State Architecture

```
        DEFINITION  (you own — portable, versioned, serde artifact)
        ┌───────────────────────────────────────────────────────────┐
        │ AgentDefinition: prompt · sections · tools · policies ·     │
        │                  SandboxSpec · feedback · output type       │
        └───────────────────────────┬───────────────────────────────┘
                                     │ bind + negotiate capabilities
        HARNESS  (runtime owns — swappable adapter)
        ┌───────────────────────────▼───────────────────────────────┐
        │ ProviderAdapter  ⟷  HarnessCapabilities (negotiated)        │
        │   opens ▶ Sandbox{ filesystem · shell · egress }            │
        │   bridges ▶ tools/policies   enforces ▶ Policy engine       │
        └───────────────────────────┬───────────────────────────────┘
                                     │ every decision/effect emits events
        OBSERVABILITY  (record · replay · evaluate)
        └─ transcript → debug bundle → ReplayTrace → eval score
```

Key models (detailed in the milestones):

| Concept | Kind | Role | Milestone |
|---------|------|------|-----------|
| `Sandbox` (+`SandboxSpec`, `Filesystem`, `Shell`, `EgressPolicy`) | aggregate + facets | The one environment all tool calls target | R1 (M1–M8) |
| `AgentDefinition` | serde artifact | The portable, versioned, content-addressed definition | M9 |
| `HarnessCapabilities` | negotiated descriptor | What a harness supports ⟷ what a definition requires | M10 |
| Conformance Kit (ACK) | test contract | One behavioral suite every adapter passes | M11 |
| `Policy` (engine) | runtime rules | Composable, fail-closed gating: tools, egress, budget, completion | M14 |
| `ReplayTrace` | recorded artifact | Deterministic record/replay of a run | M13 |

### Egress control (proxy sidecar)

Every `Sandbox` routes outbound traffic through a **proxy sidecar** it owns.
Egress is **default-deny**; `SandboxSpec.egress` seeds the allowlist and
`Sandbox.configure_egress(policy)` reconfigures the sidecar **live, at any time** —
no restart — so the orchestrator can tighten/widen access per phase. It is a
**control-plane** capability (harness/policy-driven, *not* model-facing) and, in
R3, one policy kind in the unified engine (M14).

______________________________________________________________________

## Release Roadmap

### R1 — One Environment

Every tool call targets one consistent, swappable environment (filesystem, shell,
egress), local or remote — the foundational simplification that makes it clean.

| # | Milestone |
|---|-----------|
| [M1](M1.md) | Filesystem: narrow backend + one facade |
| [M2](M2.md) | Shell facet |
| [M3](M3.md) | Sandbox aggregate + provider + `SandboxSpec` (incl. egress) |
| [M4](M4.md) | Sandbox as execution context; transactions over (session, sandbox) |
| [M5](M5.md) | Dissolve `WorkspaceSection`; unify adapters on the sandbox |
| [M6](M6.md) | Remote facets over `SandboxTransport` |
| [M7](M7.md) | Remote provider — Codex in a container + egress proxy |
| [M8](M8.md) | (Optional) Funnel every tool call through the sandbox |

### R2 — Portable Definitions & Harness Parity

The definition becomes a serializable, versioned artifact that behaves
identically on every harness — the core differentiator.

| # | Milestone |
|---|-----------|
| [M9](M9.md) | Portable Agent Definition artifact (serialize · version · diff) |
| [M10](M10.md) | Harness capability negotiation & graceful degradation |
| [M11](M11.md) | Adapter Conformance Kit (parity, CI-enforced) |
| [M12](M12.md) | Runtime de-leak & encapsulation (enables clean serialization) |

### R3 — Trustworthy Runs

Runs are reproducible, measurable, durable, and policy-governed.

| # | Milestone |
|---|-----------|
| [M13](M13.md) | Deterministic record & replay |
| [M14](M14.md) | Policy engine generalization (tools · egress · budget · completion) |
| [M15](M15.md) | Eval & resumable sessions |
| [M16](M16.md) | Observability consolidation (debug bundle / query) |

### Continuous & horizon

A **hardening track** runs alongside, governed by [REVIEW.md](REVIEW.md) and
tracked in [BACKLOG.md](BACKLOG.md): adapter-orchestration consolidation
(co-requisite of R1), type-escape burn-down, prompt-render DRY, and
error-handling/concurrency hygiene. **Already shipped on `main`** (PRs
#1163–#1165): dead-code sweep, `session_id`/de-dynamic access, spec
reconciliation. **Horizon**, past these releases: sub-agent orchestration as a
definition primitive, and multi-harness routing/fallback built on M10.

______________________________________________________________________

## Ground Rules

- **No backwards-compatibility shims.** Delete replaced code outright.
- **Partial rewrites are fine** when they yield a simpler whole.
- **Every milestone lands green** (`make check`, 100% coverage, 10 s/test, 4-layer
  boundaries). Bold within a milestone; never leave the tree broken between them.

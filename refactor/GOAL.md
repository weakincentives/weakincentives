# GOAL — A Core That Ages Well

> North star for the refactor. WINK's durable value is a small set of
> well-designed cores; everything volatile lives at the edges. The milestones
> (`M1.md` … `M9.md`) strengthen the cores in dependency order, grouped into
> three releases. **Backwards compatibility is a non-goal**; we improve
> abstractions, delete weight, and accept partial rewrites. Every milestone lands
> green (`make check`, 100% coverage).

______________________________________________________________________

## Mission

WINK makes the agent **definition** — prompt, tools, policies, feedback, and
environment intent — **versioned, modular code** that is structured for reuse,
**testable in isolation**, and runs on top of any provider agent **harness**
(Claude Agent SDK, Codex App Server, ACP / OpenCode / Gemini).

> You own the **definition**. The harness owns **execution**. WINK is the seam —
> and the seam is deliberately narrow.

______________________________________________________________________

## The 2–3 Year Bet

**What will keep changing:** models (context length, reasoning, computer use),
harnesses (absorbing planning, sub-agents, skills, memory, hosted sandboxes),
and protocols (MCP, ACP, app-server — all young and churning).

**What will not:** an agent needs instructions, capabilities, and constraints
expressed somewhere; side effects must land somewhere controllable; state must be
inspectable and recoverable; runs must be explainable after the fact; and someone
must decide what is allowed.

WINK bets on the second list. The strategy:

1. **Confine volatility.** Everything that changes with model/harness generations
   lives in adapter leaves and *declared capabilities* — never in the cores. A new
   harness is a leaf package plus a capability declaration, nothing else.
1. **Build the durable concerns as five small cores** (below), each with a narrow
   protocol, strict invariants, and isolation tests.
1. **Design to shrink.** When a harness absorbs something WINK used to emulate
   (hosted sandboxes, native sub-agents, native planning), the move is: declare it
   in `AdapterCapabilities`, route the core through it, **delete the emulation**.
   The library gets smaller as platforms mature — that is the aging plan, not a
   failure mode.

______________________________________________________________________

## The Five Cores

| Core | One responsibility | Invariant | Today (to fix) |
|------|--------------------|-----------|----------------|
| **Definition** (`prompt/`) | Express what the agent should do: sections, tools, policies, `WorkspaceConfig`, success criteria | Pure; renders deterministically; **imports no adapter** (checker-enforced) | Independence holds but is incidental; 3 parallel generic-specialization mechanisms; no isolation-test surface |
| **State** (`runtime/session`) | Event-sourced session: pure reducers, snapshots, restore | All mutation via events; snapshot/restore total | Strong design; leaky encapsulation (backcompat shims, `_LoopLike` mirror — hardening H5) |
| **Environment** (`sandbox/`, `filesystem/`) | Where effects land: filesystem, shell, egress — local or remote | Every tool effect lands in one `Sandbox`; rollback = (session, sandbox); egress **default-deny** | The missing core: no shell, no egress, FS-as-DI-resource, harness `cwd` diverges remotely |
| **Contract** (`adapters/`) | The seam: one adapter protocol + declared capabilities + shared turn machinery | No adapter-name conditionals; capability negotiation; new harness = leaf | Largest package (14.5k LOC); turn-loop ×3, guardrails ×2, ephemeral home ×3; no capability negotiation at runtime |
| **Evidence** (`debug/`, transcript, evals) | What happened: events → transcript → bundle → replay → score | Every decision/effect emits one schema'd record; bundles replayable | Three readers of one zip; env schema typed 3×; ~600 LOC duplicated SQL builders; no replay |

Substrate underneath (`types`, `serde`, `dataclasses`, `dbc`, `clock`,
`resources`) is small and healthy; the 4-layer module checker
(`toolchain/checkers/architecture.py`) remains the mechanical enforcement of the
dependency rule: substrate → cores → adapter leaves → CLI/debug tooling.

```
DEFINITION  (code you own)     sections · tools · policies · WorkspaceConfig
      │ bind / negotiate capabilities
CONTRACT    (the narrow seam)  ProviderAdapter + AdapterCapabilities + bridged tools
      │ drives                        │ every effect lands in
STATE       (event-sourced)  ◄──────► ENVIRONMENT  (Sandbox: fs · shell · egress)
      │                               │
      └────────── both emit into ─────┘
EVIDENCE    (one record)       events → transcript → bundle → replay → eval
```

______________________________________________________________________

## Design Tenets (how abstractions age)

Each tenet has a test; a change that fails the test is wrong even if convenient.

1. **Volatility at the edges.** Model/harness-generation churn touches only
   adapter leaves and capability declarations. *Test: a new harness ships — which
   files change? Only `adapters/<new>/` and its capability declaration.*
1. **Narrow waists.** Each core exposes one small protocol; breadth lives in
   facades composed on top, never re-implemented per backend. *Test: a fake can
   implement the protocol in under ~100 lines.*
1. **Capabilities, not conditionals.** Harness differences are declared data,
   negotiated at bind; degrade gracefully or fail closed. *Test:
   `grep 'adapter_name =='` matches nothing outside telemetry.*
1. **Pure core, effects at the edge.** Rendering and reducers are pure; all I/O
   flows through the `Sandbox` or the adapter. *Test: definition unit tests need
   no network and no real disk.*
1. **Data at the boundaries, code inside.** Whatever crosses a boundary —
   `WorkspaceConfig`, events, transcripts, traces, capabilities — is a serde value
   with a schema. The definition itself is code.
1. **One concept per concern.** The naming taxonomy below; no overloaded suffixes.
1. **Delete over deprecate.** Alpha rule, already repo law (`CLAUDE.md`); the
   shrink strategy depends on it.

### Non-goals (restraint is an aging strategy)

- **No workflow/DAG engine** — policies over workflows (`POLICIES_OVER_WORKFLOWS.md`).
- **No raw model-API clients** — harnesses only (`ADAPTERS.md` already says this).
- **No bespoke planning loop** — the harness plans; `AgentLoop` stays a thin driver.
- **No prompt templating DSL** — sections compose in Python; no string-template empire.
- **No second sandbox under a harness's own** — trust one perimeter; compose, don't stack.

______________________________________________________________________

## Naming Taxonomy

Suffixes carry meaning. One idiom per concept, no overlaps:

| Suffix | Means | Examples |
|--------|-------|----------|
| **`*Config`** | Immutable, declarative configuration — of a **component you construct/inject** *or* a **resource a provider materializes** (`provider.open(config) -> Resource`); may contain policies | `WorkspaceConfig`, `LLMConfig`, `…ClientConfig` |
| **`*Policy`** | **Rules/constraints** evaluated at runtime (fail-closed) | `EgressPolicy`, `ThrottlePolicy` |
| **`*Params`** | Typed input to a **tool/section handler** | `ParamsT` dataclasses |
| **`*Request` / `*Result`** | A unit of **work** through a loop + its outcome | `AgentLoopRequest`/`Result`, `ToolResult` |
| **`*Descriptor`** | Lightweight **metadata about an existing thing** | tool/section descriptors |

**On `*Config`:** we do not split out a separate `*Spec` suffix; `*Config` is the
single declarative-input idiom, whether the thing is constructed in-process or
handed to a provider to materialize. A config may *contain* policies
(`WorkspaceConfig.egress: EgressPolicy`). The environment config (now
`WorkspaceConfig`; landed as `SandboxConfig` and renamed at [M3](M3.md))
subsumed the `claude_agent_sdk` isolation `SandboxConfig` (OS-isolation
knobs), which was renamed `IsolationConfig` and refactored into provider
configuration at [M3](M3.md) — one workspace config, not two.

______________________________________________________________________

## Egress & Credentials (proxy sidecar)

Every `Sandbox` routes outbound traffic through a **proxy sidecar** it owns. The
sidecar has two jobs, both reconfigurable **live, at any time** — no restart:

- **Egress** — default-deny. `WorkspaceConfig.egress` seeds the allowlist;
  `Sandbox.configure_egress(policy)` tightens/widens access per phase (e.g. open
  a package registry for a build step, then revoke).
- **Credential injection** — secrets **never enter the sandbox**: not in env
  vars, not on disk, never visible to the model or tools. An `EgressRule` may
  name a credential and how to inject it (e.g. `Authorization: Bearer {secret}`
  toward `api.github.com`); the secret *material* is bound at runtime via
  `Sandbox.configure_credentials(bindings)`, held only in the proxy, and attached
  to allowed outbound requests on the way through. The agent can **use** a
  credential without ever being able to **read** it; rotation is a control-plane
  call. By construction, material never appears in `WorkspaceConfig` (serde values
  carry credential *names* only) nor in evidence (transcripts/bundles).

Both surfaces are **control-plane** capabilities (harness/policy-driven, *not*
model-facing — an agent cannot widen its own egress or mint its own credentials)
and become policy-governed decision points in the unified control plane
([M8](M8.md)).

______________________________________________________________________

## Roadmap

### R1 — The Environment Core (M1–M4)

The missing core, built once: every tool call targets one consistent, swappable
environment — local or remote — and rollback becomes (session, sandbox).

| # | Milestone | Outcome |
|---|-----------|---------|
| [M1](M1.md) | Filesystem narrow waist | One backend protocol + one facade; per-backend duplication deleted |
| [M2](M2.md) | Sandbox aggregate | `Shell` facet, `Sandbox`, `WorkspaceConfig` (incl. egress), local provider |
| [M3](M3.md) | Sandbox as execution context | `ToolContext.sandbox`; transactions = (session, sandbox); `WorkspaceSection` dissolved; adapters unified behind `AgentRuntime` (the bound adapter+prompt+sandbox triple) |
| [M4](M4.md) | Remote sandbox | `SandboxTransport`, remote facets; validated via ACP-over-SSH and Codex-in-container + egress/credential sidecar |

### R2 — The Definition & Contract Cores (M5–M6)

Definitions become composable code you unit-test without a harness; the seam
becomes small, declared, and proven.

| # | Milestone | Outcome |
|---|-----------|---------|
| [M5](M5.md) | Definition composition & isolation testing | Reusable packs; one specialization mechanism; boundary checker; `FakeHarness` |
| [M6](M6.md) | The harness contract | Seam consolidated in `_shared`; `AdapterCapabilities` negotiation; ACK realized |

### R3 — The Evidence Core & Control Plane (M7–M9)

Runs become explainable, replayable, governable, measurable, durable.

| # | Milestone | Outcome |
|---|-----------|---------|
| [M7](M7.md) | One evidence pipeline | Bundle/query consolidated; `ReplayTrace` + deterministic replay |
| [M8](M8.md) | Policies as the one control plane | Tools · egress · budget · completion on one fail-closed engine; decisions are events |
| [M9](M9.md) | Measure & resume | Evals over definitions; checkpoint/resume = (session, sandbox, cursor) |

### Continuous

A **hardening track** runs alongside, governed by [REVIEW.md](REVIEW.md) and
tracked in [BACKLOG.md](BACKLOG.md): type-escape ratchet, prompt-render DRY,
error/concurrency hygiene, runtime de-leak. **Horizon** (post-R3): sub-agent
orchestration as a definition primitive; multi-harness routing using
`AdapterCapabilities` + eval scores — routing, not identical execution.

______________________________________________________________________

## Ground Rules

- **No backwards-compatibility shims.** Delete replaced code outright.
- **Partial rewrites are fine** when they yield a simpler whole.
- **Every milestone lands green** (`make check`, 100% coverage, 10 s/test, 4-layer
  boundaries). Milestones contain ordered **stages**; each stage is independently
  shippable and green.

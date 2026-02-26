# Architecture

This document describes the architecture of WINK (Weak Incentives), the
agent-definition layer for building unattended agents. It covers guiding
principles, system-level decomposition, data flow, and key design invariants.

For implementation details, see the specs in `specs/` and the API reference in
`llms.md`. For terminology, see `GLOSSARY.md`.

______________________________________________________________________

## Thesis

The name "Weak Incentives" comes from mechanism design: a system with the right
incentives is one where participants naturally gravitate toward intended
behavior. Applied to agents, this means shaping the prompt, tools, and context
so the model's easiest path is also the correct one.

WINK is built around a specific bet: **models absorb more of the reasoning loop
every generation; the durable value lives in tools, retrieval, and context
engineering.** Elaborate orchestration graphs that were necessary yesterday work
in a single prompt today. The complexity budget should go toward making the
model's context precise and its tools reliable, not toward routing and
branching.

Concretely:

- **Don't compete at the model layer.** Models and agent frameworks
  commoditize. Treat them as swappable dependencies.
- **Differentiate with your system of record.** Domain knowledge, permissions,
  authoritative data, and business context — these are the defensible assets.
- **Use provider runtimes; own the tools.** Let vendors handle planning,
  orchestration, and retries. Invest in high-quality tools that expose your
  system-of-record capabilities.
- **Build evaluation as your control plane.** Make model and runtime upgrades
  safe via scenario tests and structured-output validation.

______________________________________________________________________

## Definition vs. Harness

The central architectural split. Every unattended agent has two parts, and WINK
draws a hard line between them:

```
┌──────────────────────────────────┐   ┌──────────────────────────────────┐
│     AGENT DEFINITION (yours)     │   │   EXECUTION HARNESS (runtime's)  │
│                                  │   │                                  │
│  Prompt     — decision procedure │   │  Planning/act loop               │
│  Tools      — side-effect surface│   │  Sandboxing & permissions        │
│  Policies   — enforceable gates  │   │  Retries, throttling, backoff    │
│  Feedback   — "done?" criteria   │   │  Deadlines, budgets, recovery    │
│                                  │   │  Multi-agent orchestration       │
└──────────────────────────────────┘   └──────────────────────────────────┘
```

The harness keeps changing — and increasingly comes from vendor runtimes. The
agent definition should not. WINK makes the definition a first-class artifact
you can version, review, test, and port across runtimes via adapters.

Spec: `specs/ADAPTERS.md`

______________________________________________________________________

## Guiding Principles

### 1. The Prompt is the Agent

Most frameworks treat prompts as an afterthought — templates glued to separately
registered tool lists. WINK inverts this: an agent is a single hierarchical
document where each section bundles its own instructions and tools together.

```
PromptTemplate[ReviewResponse]
├── MarkdownSection (guidance)              ← instructions
├── WorkspaceDigestSection                  ← cached codebase summary
├── MarkdownSection (reference docs)        ← progressive disclosure
├── WorkspaceSection                        ← contributes file tools
│   └── (nested workspace docs)
└── MarkdownSection (user request)          ← per-turn parameters
```

Properties that fall out of this design:

| Property | Description |
|----------|-------------|
| **Co-location** | Instructions and tools live in the same section. Documentation can't drift from implementation. |
| **Progressive disclosure** | Nest child sections, default to summaries, expand on demand. The model sees numbered hierarchical headings. |
| **Dynamic scoping** | Each section has an `enabled` predicate. Disable a section and its entire subtree — tools included — vanishes from the prompt. |
| **Typed all the way down** | Sections are parameterized with dataclasses. Placeholders validated at construction time. Tools declare typed params and results. |

Spec: `specs/PROMPTS.md`

### 2. Policies Over Workflows

A workflow encodes _how_ to accomplish a goal — a predetermined sequence that
fractures on edge cases. A policy encodes _what_ the goal requires —
constraints the agent must satisfy while remaining free to find any valid path.

| Aspect | Workflow | Policy |
|--------|----------|--------|
| Specifies | Steps to execute | Constraints to satisfy |
| On unexpected | Fails or branches | Agent reasons |
| Composability | Sequential coupling | Independent conjunction |
| Agent role | Executor | Reasoner |

Good policies are:

- **Declarative** — state what must be true, not how to make it true
- **Composable** — combine via conjunction; all must allow
- **Fail-closed** — when uncertain, deny; let the agent adapt
- **Observable** — explain denials to enable self-correction

Spec: `specs/POLICIES_OVER_WORKFLOWS.md`

### 3. Event-Driven State

All mutations flow through pure reducers processing typed events. State is
immutable and inspectable via snapshots. There is no mutable shared state
outside the session.

### 4. Transactional Tools

Tool calls are atomic transactions. When a tool fails:

1. Session state rolls back to the pre-call snapshot
2. Filesystem changes revert
3. An error result is returned to the LLM with guidance

Failed tools never leave partial state. This enables aggressive retry and
recovery strategies.

### 5. Provider-Agnostic

The same agent definition works across Claude Agent SDK, Codex App Server, and
ACP-compatible agents (OpenCode, Gemini CLI) via the adapter abstraction.

______________________________________________________________________

## Module Architecture

The codebase is organized into four strict layers. Higher layers may depend on
lower layers; reverse imports are forbidden at runtime (`TYPE_CHECKING`-guarded
type-only imports are permitted).

```
┌─────────────────────────────────────┐
│       HIGH-LEVEL (Layer 4)          │  User-facing features
│  evals, contrib, cli, debug         │
└──────────────┬──────────────────────┘
               │ depends on
┌──────────────┴──────────────────────┐
│       ADAPTERS (Layer 3)            │  Provider integrations
└──────────────┬──────────────────────┘
               │ depends on
┌──────────────┴──────────────────────┐
│       CORE (Layer 2)                │  Library primitives
│  prompt, runtime, resources,        │
│  filesystem, serde, skills          │
└──────────────┬──────────────────────┘
               │ depends on
┌──────────────┴──────────────────────┐
│       FOUNDATION (Layer 1)          │  Base types & utilities
│  types, errors, dbc, clock,         │
│  budget, deadlines                  │
└─────────────────────────────────────┘
```

| Layer | Can Import From | Cannot Import From |
|-------|-----------------|-------------------|
| Foundation | stdlib only | Core, Adapters, High-level |
| Core | Foundation, other Core | Adapters, High-level |
| Adapters | Foundation, Core | High-level |
| High-Level | any lower layer | — |

Additional rules:

- Private modules (`_foo.py`) must not be imported outside their package.
- Circular dependencies are broken via protocols or `TYPE_CHECKING`.
- `import time` is banned in production code; all time access goes through
  injected `Clock` protocols.

Layer discipline is enforced by automated checkers that run on every
`make check`.

Spec: `specs/MODULE_BOUNDARIES.md`

______________________________________________________________________

## Core Subsystems

### Prompt System

The prompt system is the heart of WINK. It replaces the common pattern of
separate prompt templates, tool registries, and schema definitions with a single
unified structure.

**PromptTemplate** is an immutable object graph — a tree of sections. Each
section can render markdown, declare typed placeholders, register tools, and
optionally render as a summary for progressive disclosure.

**Prompt** binds runtime configuration: parameter dataclasses that fill template
placeholders, an overrides store for safe iteration, and optionally a session
for dynamic visibility.

**RenderedPrompt** is the final artifact sent to the adapter — deterministic
markdown plus a tool manifest.

Key design decisions:

- Duplicate tool names within a prompt are rejected at render time
- Prompt overrides are hash-validated — when source changes, stale overrides
  stop applying until explicitly updated
- Resources are collected from template, sections, and bind-time sources with
  clear precedence

Spec: `specs/PROMPTS.md`

### Session State

Sessions are deterministic, side-effect-free state containers. A `Session` is a
thin facade coordinating typed slice storage, event-to-reducer routing, and
snapshot/restore for transaction rollback.

**Dispatch model.** All mutations flow through `session.dispatch(event)`. Events
are routed by concrete dataclass type to registered reducers. Reducers are pure
functions that return algebraic `SliceOp` values (Append, Extend, Replace,
Clear) — they never mutate state directly.

**Slice storage.** Slices are typed containers partitioned by policy:

| Policy | Behavior |
|--------|----------|
| STATE | Rolled back on tool failure; participates in snapshot/restore |
| LOG | Preserved during restore; audit trail |

Backends include in-memory (tuple-backed) and file-backed (JSONL with
append-optimized I/O). Factory config selects the backend per policy.

**Query API.** Typed accessors: `session[T].latest()`, `.all()`,
`.where(predicate)`.

**Thread safety.** Session uses a single `RLock`. Reads return immutable tuples;
snapshots are consistent.

Specs: `specs/SESSIONS.md`, `specs/SLICES.md`, `specs/THREAD_SAFETY.md`

### Tool Runtime

Tools are the side-effect boundary. Every capability that modifies the
environment — file writes, API calls, shell commands — is expressed as a tool
with a typed handler.

```python
def handler(params: P, *, context: ToolContext) -> ToolResult[R]:
```

`ToolContext` provides access to the active prompt, session, adapter, resources,
deadline, and heartbeat. Failed tools return errors, never abort evaluation.

**Transactional execution.** Before each tool call, the adapter takes a session
snapshot. On failure, state rolls back. On success, policy state is updated.

**Policy enforcement.** Policies are checked before every tool execution. If any
policy denies the call, the tool returns an error to the LLM explaining why.
The LLM can then reason about how to satisfy the constraint.

Spec: `specs/TOOLS.md`

### Guardrails

Three complementary mechanisms enforce constraints while preserving agent
reasoning autonomy:

| Mechanism | Role | Enforcement |
|-----------|------|-------------|
| **Tool Policies** | Gate tool invocations | Hard block (fail-closed) |
| **Feedback Providers** | Soft guidance over time | Advisory (agent decides) |
| **Task Completion** | Verify goals before stopping | Block early termination |

**Tool Policies** enforce sequential dependencies between tools. Policies are
composable — multiple policies can govern the same tool, and all must allow.
Examples: "tool B requires tool A", "existing files must be read before
overwritten."

**Feedback Providers** observe agent trajectory and inject guidance. They are
trigger-based (every N calls, every N seconds, on file creation), non-blocking,
and all matching providers run concurrently. Delivery is immediate via adapter
hooks.

**Task Completion Checkers** verify that required outputs exist before allowing
the agent to stop. Checkers compose with AND/OR logic. Checking is skipped when
deadline or budget is exhausted.

All three mechanisms are declared on the prompt definition — they are properties
of the agent's goal, not the harness.

Spec: `specs/GUARDRAILS.md`

### Adapters

Adapters bridge agent definitions to execution harnesses. WINK only integrates
with **agentic harnesses** — runtimes that provide planning loops, sandboxing,
and tool orchestration. Direct API calls to model providers are too low-level.

All adapters implement the `ProviderAdapter` protocol with a single `evaluate()`
entry point that accepts a prompt, session, optional deadline, and optional
budget, and returns a typed `PromptResponse`.

**Adapter lifecycle:**

1. Validate prompt context
2. Render prompt into markdown + tool manifest
3. Format for provider wire protocol
4. Execute with throttle protection and deadline checks
5. Parse response, dispatch tool calls transactionally
6. Emit lifecycle events for observability

**Supported harnesses:**

| Harness | Protocol | Custom Tool Bridge |
|---------|----------|--------------------|
| Claude Agent SDK | Claude Code SDK | MCP server |
| Codex App Server | stdio NDJSON | Dynamic tools |
| ACP (OpenCode, Gemini CLI) | JSON-RPC over stdio | MCP HTTP server |

Each adapter has access to the harness's native tools (file ops, shell, search)
while WINK bridges custom tools through the appropriate mechanism.

The **Adapter Compatibility Kit (ACK)** is a unified integration test suite
that validates any adapter against the behavioral contract.

Specs: `specs/ADAPTERS.md`, `specs/CLAUDE_AGENT_SDK.md`,
`specs/CODEX_APP_SERVER.md`, `specs/ACP_ADAPTER.md`

### Resource Registry

Dependency injection with scope-aware lifecycle management. Resources are
declared as bindings (protocol + provider function + scope) collected into an
immutable registry.

| Scope | Lifetime |
|-------|----------|
| SINGLETON | One instance per prompt context |
| TOOL_CALL | Fresh per tool invocation |
| PROTOTYPE | Fresh every access |

Resource lifecycle is owned by the prompt via a context manager. Resources are
collected from template-level, section-level, and bind-time sources with clear
precedence.

Spec: `specs/RESOURCE_REGISTRY.md`

### AgentLoop

`AgentLoop` standardizes the end-to-end agent workflow: receive request, build
prompt, evaluate within resource context, handle visibility expansion, return
result.

**Execution flow:**

1. Receive request (via mailbox or direct `execute()` call)
2. `prepare(request)` → `(Prompt, Session)`
3. Resolve effective settings (budget, deadline, resources)
4. Enter prompt resource context
5. Evaluate with adapter
6. On visibility expansion request: apply overrides, retry (bounded)
7. `finalize()` → post-processed output
8. Clean up prompt resources
9. Return result

The loop is mailbox-driven, supporting durable at-least-once message delivery.

Specs: `specs/AGENT_LOOP.md`, `specs/LIFECYCLE.md`

______________________________________________________________________

## Data Flow

A single evaluation pass follows this path:

```
  User Request
       │
       ▼
  AgentLoop.prepare()
       │  builds Prompt + Session
       ▼
  Prompt.render()
       │  section tree → markdown + tool manifest
       ▼
  Adapter.evaluate()
       │  formats for provider, sends to LLM
       ▼
  ┌─── LLM Response ───┐
  │                     │
  │  text response      │  tool_call request
  │       │             │       │
  │       ▼             │       ▼
  │  Parse output       │  Policy check (all must allow)
  │                     │       │
  │                     │  ┌────┴────┐
  │                     │  │ DENIED  │  → error to LLM → retry
  │                     │  └─────────┘
  │                     │       │ ALLOWED
  │                     │       ▼
  │                     │  Snapshot session
  │                     │       │
  │                     │       ▼
  │                     │  Execute tool handler
  │                     │       │
  │                     │  ┌────┴────┐
  │                     │  │ FAILURE │  → restore snapshot → error to LLM
  │                     │  └─────────┘
  │                     │       │ SUCCESS
  │                     │       ▼
  │                     │  Run feedback providers
  │                     │       │
  │                     │       ▼
  │                     │  Return result + feedback to LLM
  │                     │       │
  │                     │       ▼
  │                     │  (next turn)
  └─────────────────────┘
       │
       ▼
  Task completion check (if configured)
       │
       ▼
  AgentLoopResult (typed output or error)
```

______________________________________________________________________

## Observability

**Event system.** An in-process dispatcher provides publish/subscribe for
lifecycle events. Sessions subscribe to collect prompt renders, tool
invocations, and execution telemetry.

**Transcripts.** A unified, adapter-agnostic log of everything that happens
during evaluation. Chronological entries share a common envelope with canonical
types (assistant message, tool call, tool result, reasoning, system, error).
Emitted as structured log records.

**Debug bundles.** Self-contained zip archives capturing session state, logs,
filesystem snapshots, configuration, transcript, and environment metadata.
Automatic capture during agent loop execution.

**Session snapshots.** Sessions can be snapshotted at any point and serialized.
Snapshots capture all slice contents and metadata. They can be restored for
replay, debugging, or crash recovery.

Specs: `specs/TRANSCRIPT.md`, `specs/DEBUG_BUNDLE.md`

______________________________________________________________________

## Lifecycle & Operations

**Graceful shutdown.** A shutdown coordinator installs SIGTERM/SIGINT handlers
and invokes registered callbacks. Multiple loops run in dedicated threads with
coordinated shutdown — in-flight work completes, queued messages are nacked for
redelivery.

**Health & watchdog.** Optional HTTP health endpoints for Kubernetes probes. A
watchdog daemon monitors heartbeat timestamps and terminates stuck workers.

**Mailbox.** SQS-compatible protocol for durable, at-least-once message
delivery with visibility timeouts and explicit acknowledgment.

Specs: `specs/LIFECYCLE.md`, `specs/HEALTH.md`, `specs/MAILBOX.md`

______________________________________________________________________

## Cross-Cutting Concerns

**Time.** All time-dependent code uses injected `Clock` protocols. Monotonic
time for elapsed measurements; wall-clock time (UTC) for timestamps and
deadlines. Production uses system clock; tests use a fake clock that advances
instantly.

**Serialization.** Custom dataclass serde (`parse`/`dump`) with no third-party
dependencies. Constraint validation via `Annotated` metadata. Polymorphic unions
use a `__type__` discriminator.

**Design by contract.** `@require`, `@ensure`, and `@invariant` decorators.
Always enabled. Preconditions validate inputs; postconditions validate results;
invariants check before/after public methods.

**Filesystem.** A unified protocol accessed through tool context and prompt
resources. Three layers: core operations (exists, stat, glob) → streaming
(byte readers/writers) → convenience (text read/write). Backend-agnostic.

**Skills.** Lightweight, open format for extending agent capabilities. A skill
is a directory with a `SKILL.md` file. Skills attach to sections and follow the
same progressive disclosure rules as tools.

Specs: `specs/CLOCK.md`, `specs/DATACLASSES.md`, `specs/DBC.md`,
`specs/FILESYSTEM.md`, `specs/SKILLS.md`

______________________________________________________________________

## Evaluation & Experimentation

**Evaluation framework.** A minimal framework built on `AgentLoop`. Adds
datasets (JSONL-backed) and scoring. Evaluators are pure functions
`(output, expected) → Score`. Session evaluators can inspect tool usage and
budget compliance.

**Experiments.** Named bundles pairing prompt override tags with feature flags
for A/B testing and optimization runs.

**Prompt overrides.** Hash-addressed override files that replace prompt section
text or tool descriptions without editing source code. When source changes,
stale overrides stop applying until explicitly updated.

Specs: `specs/EVALS.md`, `specs/EXPERIMENTS.md`, `specs/PROMPTS.md`

______________________________________________________________________

## Key Invariants

These hold throughout the system and are enforced by tests, type checkers, and
automated validators:

1. **Layer discipline.** Lower layers never import higher layers at runtime.

2. **Pure reducers.** Reducers return `SliceOp` values. They never mutate
   state, perform I/O, or dispatch events.

3. **Transactional tools.** A failed tool call leaves session state unchanged.

4. **Fail-closed policies.** When uncertain, deny. The agent receives an
   explanation and can adapt.

5. **Typed prompts.** All placeholders resolve to dataclass fields. Mismatches
   fail at construction time, not at LLM request time.

6. **Hash-validated overrides.** Stale overrides are silently skipped, never
   applied to the wrong version.

7. **No global mutable state.** Dependencies are injected explicitly — time,
   filesystem, and state all flow through explicit protocols.

8. **100% test coverage.** Enforced on every build. Design-by-contract
   decorators supplement tests with runtime assertions.

______________________________________________________________________

## Related Documentation

| Resource | Description |
|----------|-------------|
| `README.md` | Project overview, quickstart, usage examples |
| `IDENTITY.md` | WINK's thesis and positioning |
| `llms.md` | Agent-friendly API reference |
| `GLOSSARY.md` | Canonical definitions for all key terms |
| `CLAUDE.md` | Contributor instructions |
| `specs/` | Design specifications |
| `guides/` | How-to material |
| `ROADMAP.md` | Upcoming features |
| `CHANGELOG.md` | Version history |

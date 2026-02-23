# Architecture

This document describes the architecture of WINK (Weak Incentives), the
agent-definition layer for building unattended agents. It covers guiding
principles, system-level decomposition, module boundaries, data flow, and key
design invariants. Cross-references point to detailed specs in `specs/` for
deeper treatment.

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
│  contrib, evals, cli, debug         │
└──────────────┬──────────────────────┘
               │ depends on
┌──────────────┴──────────────────────┐
│       ADAPTERS (Layer 3)            │  Provider integrations
│  adapters                           │
└──────────────┬──────────────────────┘
               │ depends on
┌──────────────┴──────────────────────┐
│       CORE (Layer 2)                │  Library primitives
│  runtime, prompt, resources,        │
│  filesystem, serde, skills,         │
│  formal, debug, optimizers          │
└──────────────┬──────────────────────┘
               │ depends on
┌──────────────┴──────────────────────┐
│       FOUNDATION (Layer 1)          │  Base types & utilities
│  types, errors, dataclasses, dbc,   │
│  deadlines, budget, clock,          │
│  experiment                         │
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
- `import time` is banned in production code (only `clock.py` is exempt); all
  time access goes through injected `Clock` protocols.

Enforcement is automated by three checkers that run as part of `make check`:
`ArchitectureChecker`, `PrivateImportChecker`, `BannedTimeImportsChecker`.

Spec: `specs/MODULE_BOUNDARIES.md`

______________________________________________________________________

## Source Layout

```
src/weakincentives/
├── adapters/              # Layer 3: Provider integrations
│   ├── _shared/           #   Cross-adapter shared code (bridge, signals)
│   ├── claude_agent_sdk/  #   Claude Agent SDK adapter
│   ├── codex_app_server/  #   Codex App Server adapter (stdio JSON-RPC)
│   ├── acp/               #   Generic ACP adapter
│   ├── opencode_acp/      #   OpenCode-specific ACP adapter
│   ├── gemini_acp/        #   Gemini CLI ACP adapter
│   ├── core.py            #   ProviderAdapter protocol, PromptResponse
│   ├── config.py          #   LLMConfig base
│   └── throttle.py        #   Reactive rate limiting
│
├── prompt/                # Layer 2: Prompt system
│   ├── prompt.py          #   PromptTemplate, Prompt
│   ├── section.py         #   Section base class
│   ├── markdown.py        #   MarkdownSection
│   ├── workspace.py       #   WorkspaceSection, HostMount
│   ├── tool.py            #   Tool, ToolContext, ToolExample
│   ├── tool_result.py     #   ToolResult
│   ├── policy.py          #   ToolPolicy protocol, built-in policies
│   ├── feedback.py        #   FeedbackProvider, FeedbackTrigger
│   ├── task_completion.py #   TaskCompletionChecker protocol
│   └── overrides.py       #   Hash-based prompt overrides
│
├── runtime/               # Layer 2: Session & orchestration
│   ├── session/           #   Session, SliceStore, ReducerRegistry
│   │   ├── session.py     #     Session facade
│   │   ├── slice_store.py #     Typed slice storage
│   │   ├── reducer_registry.py  # Event-to-reducer routing
│   │   ├── reducers.py    #     Built-in reducers (append_all, upsert_by, etc.)
│   │   ├── state_slice.py #     @reducer decorator for declarative slices
│   │   ├── slices/        #     Slice backends (memory, JSONL)
│   │   └── ...
│   ├── agent_loop.py      #   AgentLoop orchestration
│   ├── lifecycle.py       #   ShutdownCoordinator, LoopGroup
│   ├── transcript.py      #   TranscriptEmitter, TranscriptEntry
│   ├── mailbox/           #   Mailbox protocol (SQS-compatible)
│   └── events/            #   InProcessDispatcher
│
├── resources/             # Layer 2: Dependency injection
│   ├── binding.py         #   Binding (protocol + provider + scope)
│   ├── registry.py        #   ResourceRegistry (immutable)
│   ├── context.py         #   ScopedResourceContext (mutable cache)
│   └── scope.py           #   SINGLETON, TOOL_CALL, PROTOTYPE
│
├── filesystem/            # Layer 2: Filesystem abstraction
│                          #   Streaming byte-first protocol
│
├── serde/                 # Layer 2: Dataclass serialization
│   ├── parse.py           #   serde.parse(cls, data)
│   ├── dump.py            #   serde.dump(obj)
│   └── schema.py          #   JSON Schema generation
│
├── skills/                # Layer 2: Agent Skills (SKILL.md format)
│
├── evals/                 # Layer 4: Evaluation framework
│                          #   Dataset, EvalLoop, Sample, Score
│
├── contrib/               # Layer 4: Contributed tools
│   └── tools/             #   WorkspaceDigestSection, in-memory FS
│
├── debug/                 # Layer 4: Debug bundles, wink debug UI
│
├── dbc/                   # Layer 1: Design-by-contract decorators
│
├── types/                 # Layer 1: Base type definitions
├── clock.py               # Layer 1: Clock protocols, FakeClock
├── budget.py              # Layer 1: Budget tracking
├── deadlines.py           # Layer 1: Deadline utilities
├── experiment.py          # Layer 1: Experiment for A/B testing
└── errors.py              # Layer 1: Error hierarchy
```

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

Key invariants:

- Section keys match `^[a-z0-9][a-z0-9._-]{0,63}$`
- Tool names match `^[a-z0-9_-]{1,64}$`; descriptions 1-200 chars
- Duplicate tool names within a prompt are rejected at render time
- Prompt overrides are hash-validated — when source changes, stale overrides
  stop applying until explicitly updated

Spec: `specs/PROMPTS.md`

### Session State

Sessions are deterministic, side-effect-free state containers. The `Session`
class is a thin facade coordinating three subsystems:

| Subsystem | Responsibility |
|-----------|----------------|
| `SliceStore` | Typed slice storage with policy-based factories |
| `ReducerRegistry` | Event-to-reducer routing (multiple reducers per event) |
| `SessionSnapshotter` | Snapshot/restore for transaction rollback |

**Dispatch model.** All mutations flow through `session.dispatch(event)`. Events
are routed by concrete dataclass type to registered reducers. Reducers are pure
functions that return `SliceOp` values (`Append`, `Extend`, `Replace`, `Clear`)
— they never mutate state directly.

**Slice storage.** Slices are typed containers partitioned by policy:

| Policy | Behavior |
|--------|----------|
| STATE | Rolled back on tool failure; snapshot/restore participates |
| LOG | Preserved during restore; audit trail |

Two backends: `MemorySlice` (in-memory tuples) and `JsonlSlice` (JSONL
file-backed with append-optimized I/O). Factory config selects the backend per
policy.

**Query API.** Typed accessors: `session[Plan].latest()`, `.all()`,
`.where(predicate)`.

**Thread safety.** Session uses a single `RLock`. All subsystems are accessed
while holding the lock. Reads return immutable tuples; snapshots are consistent.

Specs: `specs/SESSIONS.md`, `specs/SLICES.md`, `specs/THREAD_SAFETY.md`

### Tool Runtime

Tools are the side-effect boundary. Every capability that modifies the
environment — file writes, API calls, shell commands — is expressed as a tool
with a typed handler.

**Handler signature:**

```python
def handler(params: P, *, context: ToolContext) -> ToolResult[R]:
```

`ToolContext` provides access to the active prompt, session, adapter, resources,
deadline, and heartbeat. `ToolResult` has two factories: `ToolResult.ok(value)`
and `ToolResult.error(message)`. Failed tools return errors, never abort
evaluation.

**Transactional execution.** Before each tool call, the adapter takes a session
snapshot. On failure, state rolls back. On success, `policy.on_result()` is
called to update policy state.

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

**Tool Policies** enforce sequential dependencies. Built-in examples:
`SequentialDependencyPolicy` (tool B requires tool A) and
`ReadBeforeWritePolicy` (existing files must be read before overwritten).
Policies are composable — multiple policies can govern the same tool, and all
must allow.

**Feedback Providers** observe agent trajectory and inject guidance. They are
trigger-based (every N calls, every N seconds, on file creation), non-blocking,
and all matching providers run concurrently. Delivery is immediate via adapter
hooks. Examples: `DeadlineFeedback` (remaining time warnings),
`StaticFeedbackProvider` (one-time guidance on file detection).

**Task Completion Checkers** verify that required outputs exist before allowing
the agent to stop. `FileOutputChecker` validates required file existence.
`CompositeChecker` combines multiple checkers with AND/OR logic. Checking is
skipped when deadline or budget is exhausted.

All three mechanisms are declared on the prompt definition — they are properties
of the agent's goal, not the harness.

Spec: `specs/GUARDRAILS.md`

### Adapters

Adapters bridge agent definitions to execution harnesses. WINK only integrates
with **agentic harnesses** — runtimes that provide planning loops, sandboxing,
and tool orchestration. Direct API calls to model providers are too low-level.

All adapters implement the `ProviderAdapter` protocol:

```python
adapter.evaluate(prompt, session=session, deadline=deadline, budget=budget)
  → PromptResponse[OutputT]
```

**Adapter lifecycle:**

1. Validate prompt context — verify within resource context manager
2. Render — `prompt.render()` produces `RenderedPrompt`
3. Format — convert to provider wire format
4. Call — issue request with throttle protection and deadline checks
5. Parse — extract content, dispatch tool calls transactionally
6. Emit — publish lifecycle events (`PromptRendered`, `PromptExecuted`)

**Supported harnesses:**

| Harness | Protocol | Native Tools | Custom Tool Bridge |
|---------|----------|--------------|-------------------|
| Claude Agent SDK | Claude Code SDK | Read, Write, Edit, Bash, Glob, Grep | MCP server |
| Codex App Server | stdio JSON-RPC (NDJSON) | Commands, file changes, web search | Dynamic tools |
| ACP (OpenCode, Gemini CLI) | JSON-RPC over stdio | Commands, file changes, web search | MCP HTTP server |

**Shared adapter code** lives in `adapters/_shared/`: `BridgedTool`
(transactional tool wrapper), `MCPToolExecutionState` (call_id correlation),
`VisibilityExpansionSignal` (progressive disclosure), `run_async()`
(async/sync bridge).

**Adapter Compatibility Kit (ACK)** is a unified integration test suite that
validates any adapter against the behavioral contract: prompt evaluation, tool
bridging, event emission, transcript logging, transactional semantics, and
guardrail enforcement.

Specs: `specs/ADAPTERS.md`, `specs/CLAUDE_AGENT_SDK.md`,
`specs/CODEX_APP_SERVER.md`, `specs/ACP_ADAPTER.md`

### Resource Registry

Dependency injection with scope-aware lifecycle management. Resources are
declared via `Binding` (protocol + provider function + scope) and collected
into an immutable `ResourceRegistry`.

| Scope | Lifetime | Example |
|-------|----------|---------|
| `SINGLETON` | One instance per prompt context | HTTP clients, config |
| `TOOL_CALL` | Fresh per tool invocation | Request tracers |
| `PROTOTYPE` | Fresh every access | Builders, buffers |

Resource lifecycle is owned by the prompt. `prompt.resources` serves as both
context manager and accessor:

```python
with prompt.resources:
    fs = prompt.resources.get(Filesystem)
    response = adapter.evaluate(prompt, session=session)
# Resources cleaned up automatically
```

Resources are collected from three sources in precedence order:
`PromptTemplate.resources` → Section `resources()` methods (depth-first) →
`bind(resources=...)` at bind time.

Spec: `specs/RESOURCE_REGISTRY.md`

### AgentLoop

`AgentLoop` standardizes the end-to-end agent workflow: receive request, build
prompt, evaluate within resource context, handle visibility expansion, publish
result.

**Execution flow:**

1. Receive `AgentLoopRequest` (or direct `execute()` call)
2. `prepare(request)` → `(Prompt, Session)`
3. Resolve effective settings (budget, deadline, resources)
4. Enter prompt resource context
5. Evaluate with adapter
6. On `VisibilityExpansionRequired`: apply overrides, retry (up to 10 times)
7. `finalize(prompt, session, output)` → post-processed `OutputT`
8. `prompt.cleanup()` — release section resources
9. Return `AgentLoopResult`

`AgentLoop` is mailbox-driven: requests arrive via `Mailbox`, results return
via `msg.reply()`. This supports durable, at-least-once message delivery.

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
       │  section tree → RenderedPrompt (markdown + tool manifest)
       ▼
  Adapter.evaluate()
       │  formats for provider, sends to LLM
       ▼
  ┌─── LLM Response ───┐
  │                     │
  │  text response      │  tool_call request
  │       │             │       │
  │       ▼             │       ▼
  │  Parse output       │  Policy check (all policies must allow)
  │                     │       │
  │                     │  ┌────┴────┐
  │                     │  │ DENIED  │  → error result to LLM → retry
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
  AgentLoop.finalize()
       │
       ▼
  AgentLoopResult (typed output or error)
```

______________________________________________________________________

## Observability

### Event System

The `InProcessDispatcher` provides publish/subscribe for lifecycle events.
Sessions subscribe to collect prompt renders, tool invocations, and execution
telemetry. Adapter-emitted events include:

- `PromptRendered` — rendered prompt content and metadata
- `RenderedTools` — tools sent to provider
- `PromptExecuted` — response, token usage, timing
- `ToolInvoked` — individual tool call record
- `Feedback` — feedback provider outputs

### Transcripts

A unified, adapter-agnostic log of everything that happens during `evaluate()`.
Chronological entries share a common envelope (`TranscriptEntry`) with canonical
types (assistant message, tool call, tool result, reasoning, system, error).
Entries are emitted as structured log records, consumed by debug bundles and
`wink query`.

### Debug Bundles

Self-contained zip archives capturing session state, logs, filesystem snapshots,
configuration, transcript, and environment metadata. Zero-configuration:
`AgentLoop` captures them automatically when configured. Readable with standard
tools (unzip, jq, text editor).

### Session Snapshots

Sessions can be snapshotted at any point and serialized to JSONL. Snapshots
capture all slice contents, reducer registrations, and metadata. They can be
restored for replay, debugging, or crash recovery.

Specs: `specs/TRANSCRIPT.md`, `specs/DEBUG_BUNDLE.md`

______________________________________________________________________

## Lifecycle & Operations

### Graceful Shutdown

`ShutdownCoordinator` installs SIGTERM/SIGINT handlers and invokes registered
callbacks. `LoopGroup` runs multiple loops (agent, eval) in dedicated threads
with coordinated shutdown. In-flight work completes; queued messages are nacked
for redelivery.

### Health & Watchdog

`LoopGroup` optionally exposes HTTP health endpoints (`/health/live`,
`/health/ready`) for Kubernetes probes. A `Watchdog` daemon thread monitors
`Heartbeat` timestamps and terminates stuck workers via SIGKILL.

### Mailbox

SQS-compatible protocol for durable, at-least-once message delivery.
`InMemoryMailbox` for testing; `RedisMailbox` for distributed deployments.
Messages have visibility timeouts; expired messages are redelivered.

Specs: `specs/LIFECYCLE.md`, `specs/HEALTH.md`, `specs/MAILBOX.md`

______________________________________________________________________

## Cross-Cutting Concerns

### Time

All time-dependent code uses injected `Clock` protocols. Two domains:

- **Monotonic time** (`float` seconds) — elapsed time, timeouts, rate limiting
- **Wall-clock time** (UTC `datetime`) — timestamps, deadlines, event recording

`SystemClock` for production; `FakeClock` for deterministic testing. Direct
`import time` is banned in production code.

Spec: `specs/CLOCK.md`

### Serialization

`serde.parse(cls, data)` and `serde.dump(obj)` handle dataclass
serialization without third-party dependencies. Constraints via `Annotated`
metadata (`ge`, `le`, `pattern`, etc.). Polymorphic unions use a `__type__`
discriminator field.

Spec: `specs/DATACLASSES.md`

### Design by Contract

`@require`, `@ensure`, and `@invariant` decorators from `weakincentives.dbc`.
Always enabled by default. Preconditions validate inputs; postconditions
validate results; invariants check before/after public methods. Local opt-out
via `dbc_suspended()` context manager.

Spec: `specs/DBC.md`

### Filesystem

Unified `Filesystem` protocol accessed through `ToolContext`. Three-layer
architecture: core operations (exists, stat, glob, grep) → streaming layer
(byte readers/writers) → convenience layer (text read/write). Backend-agnostic:
in-memory VFS, host filesystem, or container-mounted.

Spec: `specs/FILESYSTEM.md`

### Skills

Lightweight, open format for extending agent capabilities. A skill is a
directory containing `SKILL.md` (YAML frontmatter + Markdown instructions).
Skills attach to sections and follow the same progressive disclosure rules as
tools. See [agentskills.io](https://agentskills.io).

Spec: `specs/SKILLS.md`

______________________________________________________________________

## Evaluation & Experimentation

### Evaluation Framework

Minimal framework built on `AgentLoop`. `EvalLoop` wraps `AgentLoop` and adds
datasets and scoring. `Dataset` loads samples from JSONL. Evaluators are pure
functions `(output, expected) → Score`. Session evaluators can inspect tool
usage and budget compliance. Built-in evaluators: `exact_match`, `contains`,
`all_of`, `any_of`, `llm_judge`.

### Experiments

Named experiment bundles that pair prompt overrides tags with feature flags.
Enable A/B testing by routing requests through different override tags.
`EvalReport.compare_experiments()` computes pass-rate deltas and relative
improvements.

### Prompt Overrides

Hash-addressed override files let optimizers or humans replace prompt section
text or tool descriptions without editing source code. Overrides live in
`.weakincentives/prompts/overrides/` and match by namespace, key, and tag.
When source changes, existing overrides stop applying until explicitly updated.

Specs: `specs/EVALS.md`, `specs/EXPERIMENTS.md`, `specs/PROMPTS.md`

______________________________________________________________________

## Key Invariants

These invariants hold throughout the system and are enforced by tests, type
checkers, and automated validators:

1. **Layer discipline.** Lower layers never import higher layers at runtime.
   Automated checker runs on every `make check`.

2. **Pure reducers.** Reducers return `SliceOp` values. They never mutate state,
   perform I/O, or dispatch events.

3. **Transactional tools.** A failed tool call leaves session state unchanged.
   Snapshot-restore wraps every tool execution.

4. **Fail-closed policies.** When a policy cannot determine whether to allow a
   tool call, it denies. The agent receives an explanation and can adapt.

5. **Typed prompts.** All placeholders resolve to dataclass fields. Missing or
   mistyped parameters fail at construction time, not at LLM request time.

6. **Hash-validated overrides.** Prompt overrides carry content hashes. Stale
   overrides are silently skipped, never applied to the wrong version.

7. **No global mutable state.** Dependencies are injected explicitly. Time goes
   through `Clock`. Filesystem goes through `Filesystem`. State goes through
   `Session`.

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
| `CLAUDE.md` | Contributor instructions (style, commands, checklist) |
| `specs/` | Design specifications (44 documents) |
| `guides/` | How-to material (34 guides) |
| `ROADMAP.md` | Upcoming features |
| `CHANGELOG.md` | Version history |

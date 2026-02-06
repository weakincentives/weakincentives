# Glossary

Key concepts that appear throughout Weak Incentives. Each entry links to the
canonical specification for deeper context.

## Binding

Associates a protocol type with a provider function and scope in the
`ResourceRegistry`. Bindings define how resources are constructed and their
lifecycle. See [Resource Registry specification](specs/RESOURCE_REGISTRY.md).

## Dataclass Serde Utilities

Helpers in `weakincentives.serde` parse, validate, and serialize standard library
dataclasses without third-party dependencies. They enable predictable payloads
for prompts, tools, and sessions. Detailed contract in
[Dataclass Serde Utilities](specs/DATACLASSES.md).

## Deadlines

Adapters accept a caller-supplied wall-clock deadline that applies to the entire
evaluation, including all tool calls. Enforcement semantics are defined in the
[Deadlines specification](specs/SESSIONS.md).

## Evaluation Framework

A minimal framework built on AgentLoop for testing agent behavior. Includes
`Dataset` for loading samples, `EvalLoop` for orchestration, and built-in
evaluators like `exact_match`, `contains`, and `llm_judge`. Session evaluators
enable behavioral assertions against tool usage and token budgets. See
[Evals specification](specs/EVALS.md).

## Event Bus

`weakincentives.runtime.events` exposes an in-process publish/subscribe dispatcher that
adapters use to emit prompt lifecycle telemetry. Sessions subscribe to collect
prompt renders, tool invocations, and executions. Refer to
[Prompt Event Emission](specs/SESSIONS.md).

## Formal Spec Decorator

The `@formal_spec` decorator embeds TLA+ specification metadata directly in
Python classes. Includes `StateVar`, `Action`, and `Invariant` declarations that
can be extracted and verified with the TLC model checker. See
[Formal Verification specification](specs/FORMAL_VERIFICATION.md).

## HealthServer

Minimal HTTP server exposing Kubernetes liveness (`/health/live`) and readiness
(`/health/ready`) probes for containerized deployments. Used with `LoopGroup` for
production health monitoring. See [Health specification](specs/HEALTH.md).

## Heartbeat

Thread-safe timestamp tracker for worker liveness. Records when workers last made
progress, enabling `Watchdog` to detect stalled processes. See
[Health specification](specs/HEALTH.md).

## LiteLLM Adapter

A provider adapter that mirrors the OpenAI integration while targeting the
LiteLLM SDK. It keeps evaluation, tooling, and structured-output semantics in
lock-step with the reference adapter. See the
[LiteLLM Adapter specification](specs/ADAPTERS.md).

## LoopGroup

Runs multiple `Runnable` loops (AgentLoop, EvalLoop) in dedicated threads with
coordinated shutdown. Supports optional health endpoints via `HealthServer` and
stuck worker detection via `Watchdog`. See
[Health specification](specs/HEALTH.md).

## Mailbox

Protocol providing SQS-compatible semantics for durable, at-least-once message
delivery between processes. Unlike pub/sub, Mailbox delivers messages
point-to-point with visibility timeout and explicit acknowledgment. Includes
`InMemoryMailbox` for testing and `RedisMailbox` for distributed deployments.
See [Mailbox specification](specs/MAILBOX.md).

## Native OpenAI Structured Outputs

Enhancements to the OpenAI adapter that attach JSON Schema derived from prompt
metadata so the provider returns parsed structured results. Fallbacks preserve
text-based parsing when native support is unavailable. See
[Native OpenAI Structured Outputs](specs/ADAPTERS.md).

## Prompt

The central abstraction for assembling markdown system prompts from typed
sections. Prompts validate placeholders, compose tooling, and optionally declare
structured outputs. Resource lifecycle is now bound directly to `Prompt` via
context manager protocol. Full contract in the
[Prompt specification](specs/PROMPTS.md).

## Prompt Overrides

Hash-addressed override files let optimizers or humans replace prompt sections or
tool definitions without editing source code. Storage layout and hashing rules
are specified in the [Prompts specification](specs/PROMPTS.md).

## Provider Adapter

Adapters bridge prompts to specific model providers, synchronously executing tool
calls and returning typed responses. Shared requirements live in the
[Adapter Evaluation specification](specs/ADAPTERS.md).

## ResourceRegistry

Typed registry for runtime resources passed to adapters and tool handlers.
Supports provider-based lazy construction with scope-aware lifecycle management
via `Binding`, `Scope`, and `ScopedResourceContext`. Resources are resolved via
`context.resources.get(ResourceType)`. See
[Resource Registry specification](specs/RESOURCE_REGISTRY.md).

## Runnable

Protocol for loops that support graceful shutdown. Defines `run()`, `shutdown()`,
`running`, and `heartbeat` properties. Implemented by `AgentLoop` and `EvalLoop`.
See [Health specification](specs/HEALTH.md).

## Scope

Enum controlling resource instance lifetime in `ResourceRegistry`: `SINGLETON`
(one instance per context), `TOOL_CALL` (fresh per tool scope), and `PROTOTYPE`
(fresh on every resolution). See
[Resource Registry specification](specs/RESOURCE_REGISTRY.md).

## Section

Sections are typed building blocks that render markdown, register tools, and can
be composed into prompts. They rely on dataclass payloads and Python
`string.Template` formatting. See [Prompt specification](specs/PROMPTS.md).

## Session

A deterministic state container that subscribes to prompt and tool events,
producing immutable reducer-managed snapshots for each slice. All mutations flow
through `session.dispatch(event)` for a unified, auditable interface. Details in
the [Session State specification](specs/SESSIONS.md).

## Session Evaluators

Evaluators that receive `SessionView` for inspecting agent behavior during
evaluation. Built-in evaluators include `tool_called`, `tool_not_called`,
`all_tools_succeeded`, and `token_usage_under` for asserting tool usage patterns
and budget compliance. See [Evals specification](specs/EVALS.md).

## SessionView

Read-only wrapper around `Session` for safe, immutable access to session state.
Exposes query operations (`all()`, `latest()`, `where()`) but omits mutation
methods. Reducer contexts receive `SessionView` to prevent accidental mutations.
See [Session State specification](specs/SESSIONS.md).

## ShutdownCoordinator

Singleton that installs SIGTERM/SIGINT handlers and invokes registered callbacks
on shutdown. Enables coordinated graceful shutdown across multiple loops in the
same process. See [Health specification](specs/HEALTH.md).

## Slice Storage Backends

Protocol-based storage backends for session slices. `SliceView[T]` provides lazy
read-only access for reducers, while `SliceOp` (algebraic type of `Append`,
`Extend`, `Replace`, `Clear`) represents reducer outputs. Implementations include
`MemorySlice` for in-memory storage and `JsonlSlice` for persistent JSONL files.
See [Slices specification](specs/SLICES.md).

## Structured Output

`Prompt[OutputT]` specializes a prompt with an output dataclass so the runtime
instructs models to return JSON and parses replies into typed objects. Behavior
is defined in [Structured Output via `Prompt[OutputT]`](specs/PROMPTS.md).

## Thread Safety

Identifies components that assume single-threaded use and outlines the
synchronization work required for multi-threaded adapters. Refer to the
[Thread Safety specification](specs/THREAD_SAFETY.md).

## Tool Runtime

Tool definitions attach to sections so prompts advertise callable affordances.
Handlers receive immutable context, return `ToolResult` objects, and follow a
shared success contract. See the [Tool Runtime specification](specs/TOOLS.md).

## Unified Dispatch API

All session mutations flow through `session.dispatch(event)`, providing a
consistent, auditable mutation interface. Convenience methods on slice accessors
(`seed()`, `clear()`) dispatch system events (`InitializeSlice`, `ClearSlice`)
internally. See [Session State specification](specs/SESSIONS.md).

## Watchdog

Daemon thread that monitors `Heartbeat` timestamps and terminates the process via
SIGKILL when workers stall beyond a configured threshold. Prevents stuck workers
from blocking deployments. See [Health specification](specs/HEALTH.md).

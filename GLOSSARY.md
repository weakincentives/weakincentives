# Glossary

Key concepts that appear throughout Weak Incentives. Each entry links to the
canonical specification for deeper context.

## Asteval Section

Provides a deterministic sandbox for evaluating small Python expressions inside a
session-scoped virtual filesystem. The [`AstevalSection`](specs/ASTEVAL.md)
registers the tool, enforces strict globals, captures stdout/stderr, and records
any VFS mutations for traceability.

## Chapter

A chapter groups prompt sections behind a single visibility switch so adapters
determine which regions of a prompt reach the model. Chapters default to the
closed state and wrap existing sections without rewriting them. See the
[Chapters specification](specs/CHAPTERS.md).

## Dataclass Serde Utilities

Helpers in `weakincentives.serde` parse, validate, and serialize standard library
dataclasses without third-party dependencies. They enable predictable payloads
for prompts, tools, and sessions. Detailed contract in
[Dataclass Serde Utilities](specs/DATACLASS_SERDE.md).

## Deadlines

Adapters accept a caller-supplied wall-clock deadline that applies to the entire
evaluation, including nested subagents and tool calls. Enforcement semantics are
defined in the [Deadlines specification](specs/DEADLINES.md).

## Event Bus

`weakincentives.runtime.events` exposes an in-process publish/subscribe bus that
adapters use to emit prompt lifecycle telemetry. Sessions subscribe to collect
prompt renders, tool invocations, and executions. Refer to
[Prompt Event Emission](specs/EVENTS.md).

## LiteLLM Adapter

A provider adapter that mirrors the OpenAI integration while targeting the
LiteLLM SDK. It keeps evaluation, tooling, and structured-output semantics in
lock-step with the reference adapter. See the
[LiteLLM Adapter specification](specs/LITE_LLM_ADAPTER.md).

## MCP Server (`wink`)

The `wink` extra ships a FastMCP-based server that exposes prompt override
workflows over the Model Context Protocol. Packaging, runtime expectations, and
tool wiring live in the [MCP specification](specs/MCP.md).

## Native OpenAI Structured Outputs

Enhancements to the OpenAI adapter that attach JSON Schema derived from prompt
metadata so the provider returns parsed structured results. Fallbacks preserve
text-based parsing when native support is unavailable. See
[Native OpenAI Structured Outputs](specs/NATIVE_OPENAI_STRUCTURED_OUTPUTS.md).

## Planning Tool Suite

The `PlanningToolsSection` registers a todo-list tool suite that keeps a single
session-scoped plan, replacing the current snapshot on each update. Reducers and
data models are documented in the [Planning Tool specification](specs/PLANNING_TOOL.md).

## Planning Strategies

A strategy enum customizes the guidance copy rendered by `PlanningToolsSection`
without altering its tool surface. Available mindsets are cataloged in the
[Planning Strategy Templates specification](specs/PLANNING_STRATEGIES.md).

## Prompt

The central abstraction for assembling markdown system prompts from typed
sections. Prompts validate placeholders, compose tooling, and optionally declare
structured outputs. Full contract in the [Prompt specification](specs/PROMPTS.md).

## Prompt Composition

Delegating agents wrap an existing rendered prompt with a prescriptive header so
child runs inherit the full parent context unchanged. Layout and invariants are
defined in the [Prompt Wrapping specification](specs/PROMPTS_COMPOSITION.md).

## Prompt Overrides

Hash-addressed override files let optimizers or humans replace prompt sections or
tool definitions without editing source code. Storage layout and hashing rules
are specified in the [Prompt Overrides specification](specs/PROMPT_OVERRIDES.md).

## Provider Adapter

Adapters bridge prompts to specific model providers, synchronously executing tool
calls and returning typed responses. Shared requirements live in the
[Adapter Evaluation specification](specs/ADAPTERS.md).

## Section

Sections are typed building blocks that render markdown, register tools, and can
be composed into prompts or chapters. They rely on dataclass payloads and Python
`string.Template` formatting. See [Prompt specification](specs/PROMPTS.md).

## Session

A deterministic state container that subscribes to prompt and tool events,
producing immutable reducer-managed snapshots for each slice. Details in the
[Session State specification](specs/SESSIONS.md).

## Structured Output

`Prompt[OutputT]` specializes a prompt with an output dataclass so the runtime
instructs models to return JSON and parses replies into typed objects. Behavior
is defined in [Structured Output via `Prompt[OutputT]`](specs/STRUCTURED_OUTPUT.md).

## Subagent

The `SubagentsSection` introduces a delegation tool that spawns parallel child
runs sharing the parent's tools, session, and telemetry. Requirements appear in
the [Subagents specification](specs/SUBAGENTS.md).

## Thread Safety

Identifies components that assume single-threaded use and outlines the
synchronization work required for multi-threaded adapters. Refer to the
[Thread Safety specification](specs/THREAD_SAFETY.md).

## Tool Runtime

Tool definitions attach to sections so prompts advertise callable affordances.
Handlers receive immutable context, return `ToolResult` objects, and follow a
shared success contract. See the [Tool Runtime specification](specs/TOOLS.md).

## Virtual Filesystem (VFS)

A session-scoped, copy-on-write filesystem that lets tools read and write files
without touching the host disk. `VfsToolsSection` wires the suite and reducer
integration. Defined in the [Virtual Filesystem Tool specification](specs/VFS_TOOLS.md).

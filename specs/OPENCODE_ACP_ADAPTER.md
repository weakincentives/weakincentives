# OpenCode ACP Adapter Specification

> **Status:** Design specification — not yet implemented.
> The adapter package `src/weakincentives/adapters/opencode_acp/` does not exist
> yet. This document specifies the planned design. All code snippets are
> illustrative; they will be replaced with source file references once
> implementation lands.

> **Adapter name:** `opencode_acp`
> **OpenCode entrypoint:** `opencode acp`
> **ACP protocol:** v1 (JSON-RPC 2.0 over newline-delimited JSON on stdio)
> **Validated against:** `opencode 1.1.59` with `agent-client-protocol 0.8.0`

## Purpose

`OpenCodeACPAdapter` evaluates WINK prompts by delegating execution to **OpenCode**
via its **ACP** (Agent Client Protocol) server. The architecture mirrors the
`ClaudeAgentSDKAdapter`:

| Responsibility | Owner |
|----------------|-------|
| Prompt composition, resource binding, session telemetry | WINK |
| Agentic execution (planning, reasoning, tool calls, file edits) | OpenCode |

WINK receives streamed progress via ACP `session/update` notifications and emits
canonical events: `PromptRendered`, `ToolInvoked`, `PromptExecuted`.

**Planned implementation:** `src/weakincentives/adapters/opencode_acp/`

## Why ACP

The Agent Client Protocol is a **vendor-neutral** standard for client-agent
communication. Where the Codex App Server exposes a Codex-specific stdio
protocol and the Claude Agent SDK exposes an Anthropic-specific Python API, ACP
provides a single protocol implemented by multiple agents (OpenCode, Gemini CLI,
and others).

Key OpenCode capabilities surfaced through ACP:

- **Sessions with mode/model selection:** Persistent conversation state with
  configurable agent modes (`build`, `plan`) and model switching at runtime
- **Native tools:** Command execution, file changes, web search — all streamed
  as typed `ToolCallStart` / `ToolCallProgress` updates
- **MCP server passthrough:** External MCP servers passed on `new_session` are
  connected by the agent, bridging WINK tools without an HTTP server
- **File and terminal capabilities:** Client-side file I/O and terminal
  management via typed `Client` interface methods
- **Model discovery:** `new_session` returns available models and modes,
  enabling runtime validation

For WINK's use case of deeply integrated agent orchestration with session state,
ACP provides the correct abstraction layer — a typed Python SDK with async
context management, rather than raw NDJSON parsing.

## Requirements

### Runtime Dependencies

1. **OpenCode CLI** installed and available on `PATH` as `opencode`
1. **ACP Python SDK**: `agent-client-protocol>=0.8.0`
1. **Claude Agent SDK**: `claude-agent-sdk>=0.1.27` (for shared MCP server infrastructure)
1. WINK (`weakincentives`) runtime

### WINK Packaging

```toml
[project.optional-dependencies]
acp = [
  "agent-client-protocol>=0.8.0",
  "claude-agent-sdk>=0.1.27",
]
```

> **Note:** The `acp` extra does not exist in `pyproject.toml` yet — it will be
> added when implementation begins.

The adapter takes a dependency on `claude-agent-sdk` to reuse its MCP server
infrastructure. This coupling is acceptable because both adapters share the same
tool bridging semantics, and implementing a separate MCP server adds significant
complexity.

The adapter uses lazy imports and raises a helpful error if the `acp` extra is
not installed (following the pattern in `openai.py` and `claude_agent_sdk/*`).

## Architecture

```
WINK Prompt/Session
  └─ OpenCodeACPAdapter.evaluate()
      ├─ Render WINK prompt → markdown text
      ├─ Build in-process MCP server (reuses Claude SDK infrastructure)
      ├─ Expose MCP server over HTTP (same process, localhost)
      ├─ Spawn: opencode acp (stdio JSON-RPC NDJSON)
      ├─ ACP handshake: initialize → session/new (passes HttpMcpServer URL)
      ├─ OpenCode connects to MCP HTTP server → discovers WINK tools
      ├─ Inspect new_session response (models, modes)
      ├─ set_session_model / set_session_mode (best-effort)
      ├─ conn.prompt(text)
      ├─ Stream: session/update notifications
      │    ├─ AgentMessageChunk (assistant output)
      │    ├─ AgentThoughtChunk (reasoning, if configured)
      │    ├─ ToolCallStart / ToolCallProgress (WINK + OpenCode tools)
      │    ├─ AgentPlanUpdate (plan entries)
      │    ├─ AvailableCommandsUpdate (slash commands)
      │    ├─ CurrentModeUpdate (mode changes)
      │    └─ UsageUpdate (token/cost tracking)
      ├─ Drain trailing updates (quiet_period_ms)
      ├─ Model calls structured_output tool to finalize (if required)
      └─ Return PromptResponse(text, output, token_usage)
```

The MCP server runs **in the adapter process** and is exposed over HTTP on
localhost. This is the same pattern used by the Claude Agent SDK adapter — the
`create_mcp_server()` shared infrastructure creates an `mcp.server.Server`
instance with all bridged tools registered. The difference is how the agent
connects:

- **Claude Agent SDK:** The SDK receives the server instance directly as an
  in-process config object (`{'type': 'sdk', 'instance': <Server>}`) and
  manages the transport internally.
- **OpenCode ACP:** OpenCode is a separate subprocess, so the adapter must
  expose the same `mcp.server.Server` instance over HTTP (via
  `StreamableHTTPServerTransport` or equivalent) and pass the URL as an
  `HttpMcpServer` config on `new_session`.

Both approaches use the same `BridgedTool` wrappers and the same
`mcp.server.Server` — only the transport layer differs.

## Module Structure

```
src/weakincentives/adapters/opencode_acp/
  __init__.py
  adapter.py            # OpenCodeACPAdapter
  config.py             # OpenCodeACPClientConfig, OpenCodeACPAdapterConfig
  client.py             # OpenCodeACPClient (ACP Client implementation)
  _state.py             # OpenCodeACPSessionState slice for session reuse
  _events.py            # ACP updates → WINK ToolInvoked mapping
  _structured_output.py # structured_output MCP tool
  _mcp_http.py          # HTTP transport for in-process MCP server
  _async.py             # asyncio helpers
```

Workspace management uses the generic `WorkspaceSection` from
`weakincentives.prompt.workspace` — no adapter-specific workspace module.

MCP tool bridging reuses `create_mcp_server()` from the shared adapter module
`src/weakincentives/adapters/_shared/_bridge.py`. The returned
`mcp.server.Server` instance is exposed over HTTP by `_mcp_http.py`.

## Configuration

### OpenCodeACPClientConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `opencode_bin` | `str` | `"opencode"` | Executable to spawn |
| `opencode_args` | `tuple[str, ...]` | `("acp",)` | Must include `acp` |
| `cwd` | `str \| None` | `None` | Working directory (must be absolute; defaults to `Path.cwd().resolve()`) |
| `env` | `Mapping[str, str] \| None` | `None` | Extra environment variables |
| `suppress_stderr` | `bool` | `True` | Capture stderr for errors (not printed unless debugging) |
| `startup_timeout_s` | `float` | `10.0` | Max time for initialize/session/new |
| `permission_mode` | `Literal["auto", "deny", "prompt"]` | `"auto"` | Response to `session/request_permission` (prompt must not block) |
| `allow_file_reads` | `bool` | `False` | Advertise `readTextFile` capability (only with workspace) |
| `allow_file_writes` | `bool` | `False` | Advertise `writeTextFile` capability |
| `allow_terminal` | `bool` | `False` | Advertise terminal capability (must implement `create_terminal`) |
| `mcp_servers` | `tuple[McpServerConfig, ...]` | `()` | Additional MCP servers (WINK server always added) |
| `reuse_session` | `bool` | `False` | Load/reuse OpenCode session ID |

> **CWD requirement:** ACP requires `cwd` to be an absolute path. If `None`, the
> adapter resolves to `Path.cwd().resolve()`.

> **Capability alignment:** Advertised capabilities in `initialize` must match
> implemented methods. If `readTextFile=True`, implement the method and enforce
> workspace boundaries. If `allow_terminal=True`, implement `create_terminal`.

> **Workspace gating:** If no `WorkspaceSection` is provided, the adapter must
> force `allow_file_reads=False` and `allow_file_writes=False` regardless of
> config to avoid reading or writing the host filesystem.

> **Non-interactive permissions:** `permission_mode="prompt"` must not block.
> The adapter should respond as `deny` and include a reason indicating that
> interactive prompting is not supported.

> **MCP merge:** The adapter always injects its own WINK MCP server. User-provided
> `mcp_servers` are appended; they must not shadow the WINK tool namespace.

### OpenCodeACPAdapterConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode_id` | `str \| None` | `None` | ACP `session/set_mode` (best-effort) |
| `model_id` | `str \| None` | `None` | ACP `session/set_model` (validated against available models) |
| `quiet_period_ms` | `int` | `500` | **Required.** Wait after prompt returns to drain trailing updates |
| `emit_thought_chunks` | `bool` | `False` | Include thoughts in returned text |

> **Model validation:** After `new_session`, the adapter must validate
> `model_id` against `session.models.available_models` if the response includes
> model state. Using an invalid `model_id` with `set_session_model` causes
> OpenCode to return an empty response with `stop_reason="end_turn"` and zero
> content — no error is raised. The adapter must detect this and raise
> `PromptEvaluationError(phase="request")` with a descriptive message listing
> available models.

> **Quiet period is mandatory:** The ACP stdio transport delivers `session/update`
> notifications asynchronously relative to the `prompt()` response. Notifications
> in flight when `PromptResponse` arrives on the wire are lost unless the client
> drains them. Without the quiet period, the adapter may miss final
> `AgentMessageChunk`, `ToolCallProgress`, and `UsageUpdate` notifications. The
> default of `500ms` is based on live validation; `100ms` was observed to be
> insufficient. Reset the timer on each new update; cap total wait by the
> deadline.

> **Mode selection:** OpenCode exposes modes `"build"` (default) and `"plan"`.
> However, `set_session_mode` currently returns "Internal error" in OpenCode
> 1.1.59. The adapter must treat `set_session_mode` failures as non-fatal. Modes
> are reported via `new_session().modes` and via `CurrentModeUpdate`
> notifications.

## Protocol Mapping

### WINK Concepts → ACP Concepts

| WINK Concept | ACP Concept | Adapter Role |
|--------------|-------------|--------------|
| **Prompt** (PromptTemplate + sections + tools) | Session + `prompt()` input text | Render, format, send via `conn.prompt()` |
| **Session** (event-sourced state) | ACP session (persistent, with load/fork/resume) | Map session lifecycle to WINK session events |
| **Tool** (Tool[ParamsT, ResultT]) | MCP tool via `HttpMcpServer` on `new_session` | Bridge via `create_bridged_tools()` + `create_mcp_server()` + HTTP transport |
| **Tool Execution** (transactional) | `ToolCallStart` / `ToolCallProgress` updates | Map terminal `ToolCallProgress` → `ToolInvoked` |
| **Output** (structured dataclass) | MCP `structured_output` tool call | Register tool, validate response, deserialize |
| **Events** (PromptRendered, ToolInvoked, PromptExecuted) | `session/update` notifications | Translate and dispatch |
| **Deadline** | `conn.cancel(session_id=...)` notification | Enforce with timer + cancel + kill |
| **Budget** | `PromptResponse.usage` + `UsageUpdate` notifications | Record usage from both sources |

### ACP Update Types → WINK Events

At `src/weakincentives/adapters/opencode_acp/_events.py`:
`dispatch_tool_invoked()` maps terminal ACP updates to `ToolInvoked`:

| ACP Update Type | WINK Event | Notes |
|-----------------|------------|-------|
| `AgentMessageChunk` | Text accumulation | Concatenated for `PromptResponse.text` |
| `AgentThoughtChunk` | Text accumulation (if `emit_thought_chunks`) | Prepended to response text when configured |
| `ToolCallStart` | (tracking) | Records tool call ID, title, kind, initial status |
| `ToolCallProgress` (`status="completed"`) | `ToolInvoked` with `success=True` | `title` field contains tool name |
| `ToolCallProgress` (`status="failed"`) | `ToolInvoked` with `success=False` | |
| `AgentPlanUpdate` | (informational) | Logged; plan entries available via `SessionSnapshot.plan_entries` |
| `AvailableCommandsUpdate` | (informational) | Logged |
| `CurrentModeUpdate` | (informational) | Logged; tracks mode changes |
| `UsageUpdate` | Token usage tracking | `cost`, `size`, `used` fields |
| `ConfigOptionUpdate` | (ignored) | Agent-specific config changes |
| `SessionInfoUpdate` | (ignored) | Agent-specific session metadata |

**ToolCallStart `status` values:** `"pending"` | `"in_progress"` | `"completed"` | `"failed"`

**ToolCallStart `title` field:** Contains the tool name (e.g. `"bash"`,
`"edit_file"`). Note: the field is `title`, not `name`. For MCP tools, OpenCode
prefixes the tool name with the server name: `"{server_name}_{tool_name}"`
(e.g. `"wink-tools_get_secret_number"`).

**Deduplication:** Skip `ToolInvoked` for bridged WINK tools using explicit MCP
server metadata when available; fall back to the `mcp__wink__` prefix if needed.
`BridgedTool` already emitted the event.

## Workspace Management

### WorkspaceSection

At `src/weakincentives/prompt/workspace.py` (exported from `weakincentives.prompt`):

- Accepts `HostMount` tuples, `allowed_host_roots`, max-bytes budgets
- Materializes temporary directory with copied files (with glob filtering,
  symlink safety, and byte budget enforcement)
- Exposes `temp_dir` for `OpenCodeACPClientConfig.cwd`
- Renders a provider-agnostic summary of mounts and budgets
- Exposes cleanup via `.cleanup()` with reference counting for cloned sections
- Provides `workspace_fingerprint` for session reuse validation
- Binds a `HostFilesystem` resource scoped to the temp directory

**Workspace types** (from `weakincentives.prompt`):

| Type | Description |
|------|-------------|
| `WorkspaceSection` | Generic workspace section for all adapters |
| `HostMount` | Mount configuration (host_path, mount_path, include_glob, exclude_glob, max_bytes, follow_symlinks) |
| `HostMountPreview` | Summary of materialized mount |
| `WorkspaceBudgetExceededError` | Mount exceeds byte budget |
| `WorkspaceSecurityError` | Mount violates security constraints |

## ACP Client Implementation

### OpenCodeACPClient

Implements `acp.interfaces.Client`. All methods are `async`.

| Method | ACP Protocol Message | Description |
|--------|---------------------|-------------|
| `session_update(session_id, update)` | `session/update` notification | Wrap in `SessionNotification`, feed to `SessionAccumulator` |
| `request_permission(options, session_id, tool_call)` | `session/request_permission` request | Auto-respond per `permission_mode` (prompt → deny) |
| `read_text_file(path, session_id, ...)` | `fs/read_text_file` request | Read within workspace (if capability advertised) |
| `write_text_file(content, path, session_id)` | `fs/write_text_file` request | Write within workspace (if capability advertised) |
| `create_terminal(command, session_id, ...)` | `terminal/create` request | Raise `RequestError.method_not_found` unless `allow_terminal=True` |
| `terminal_output(session_id, terminal_id)` | `terminal/output` request | Raise `RequestError.method_not_found` unless implemented |
| `release_terminal(session_id, terminal_id)` | `terminal/release` request | Raise `RequestError.method_not_found` unless implemented |
| `wait_for_terminal_exit(session_id, terminal_id)` | `terminal/wait_for_exit` request | Raise `RequestError.method_not_found` unless implemented |
| `kill_terminal(session_id, terminal_id)` | `terminal/kill` request | Raise `RequestError.method_not_found` unless implemented |
| `ext_method(method, params)` | `_*` extension request | Raise `RequestError.method_not_found` |
| `ext_notification(method, params)` | `_*` extension notification | Ignore silently |

> **Important:** `RequestError` is imported from `acp` (top-level), not
> `acp.schema`. Use `RequestError.method_not_found(method_name)` to signal
> unsupported capabilities.

Use `acp.spawn_agent_process()` to spawn OpenCode and connect:

```python
from acp import spawn_agent_process

async with spawn_agent_process(
    client,
    client_config.opencode_bin,
    *client_config.opencode_args,
    cwd=resolved_cwd,
    env=merged_env,
) as (conn, proc):
    # conn is ClientSideConnection
    # proc is asyncio.subprocess.Process
    ...
```

### SessionAccumulator

Use `acp.contrib.session_state.SessionAccumulator` to merge updates into a
queryable snapshot.

**Critical:** `SessionAccumulator.apply()` accepts `SessionNotification`, **not**
raw update objects. The `session_update` callback receives raw updates; the
adapter must wrap them:

```python
from acp.schema import SessionNotification

async def session_update(self, session_id: str, update: ..., **kwargs) -> None:
    notification = SessionNotification(sessionId=session_id, update=update)
    self.accumulator.apply(notification)
```

Calling `apply()` with a raw update raises `TypeError`. Calling `snapshot()`
before any `apply()` raises `SessionSnapshotUnavailableError`.

**SessionSnapshot fields:**

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | `str` | Current session ID |
| `tool_calls` | `dict[str, ToolCallView]` | Tool calls indexed by ID |
| `plan_entries` | `tuple[PlanEntry, ...]` | Agent plan steps |
| `current_mode_id` | `str \| None` | Active mode |
| `available_commands` | `tuple[AvailableCommand, ...]` | Slash commands |
| `user_messages` | `tuple[UserMessageChunk, ...]` | User message history |
| `agent_messages` | `tuple[AgentMessageChunk, ...]` | Agent message history |
| `agent_thoughts` | `tuple[AgentThoughtChunk, ...]` | Agent reasoning chunks |

### Initialize Response Inspection

The `InitializeResponse` exposes agent capabilities that the adapter should
inspect to determine available features:

```python
init_resp = await conn.initialize(
    protocol_version=PROTOCOL_VERSION,
    client_capabilities=ClientCapabilities(
        fs=FileSystemCapability(
            read_text_file=allow_file_reads,
            write_text_file=allow_file_writes,
        ),
        terminal=allow_terminal,
    ),
    client_info=Implementation(
        name="wink", title="WINK", version="0.1.0",
    ),
)
```

**InitializeResponse fields:**

| Field | Type | Description |
|-------|------|-------------|
| `protocol_version` | `int` | Protocol version (must be `1`) |
| `agent_info` | `Implementation \| None` | Agent name, title, version |
| `agent_capabilities` | `AgentCapabilities \| None` | Feature flags |
| `auth_methods` | `list[AuthMethod] \| None` | Available auth methods |

**AgentCapabilities fields:**

| Field | Type | Description |
|-------|------|-------------|
| `load_session` | `bool \| None` | Whether `load_session` is supported |
| `mcp_capabilities` | `McpCapabilities \| None` | HTTP/SSE MCP support |
| `prompt_capabilities` | `PromptCapabilities \| None` | Audio/image/embedded support |
| `session_capabilities` | `SessionCapabilities \| None` | Fork/list/resume support |

### New Session Response Inspection

The `NewSessionResponse` includes model and mode discovery:

```python
session = await conn.new_session(cwd=resolved_cwd, mcp_servers=mcp_configs)

# Model discovery
if session.models:
    available_models = session.models.available_models  # list[ModelInfo]
    current_model = session.models.current_model_id     # str

# Mode discovery
if session.modes:
    available_modes = session.modes.available_modes      # list[SessionMode]
    current_mode = session.modes.current_mode_id         # str
```

**NewSessionResponse fields:**

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | `str` | Session identifier |
| `models` | `SessionModelState \| None` | Available models + current selection |
| `modes` | `SessionModeState \| None` | Available modes + current selection |
| `config_options` | `list[SessionConfigOption] \| None` | Agent-specific config |

**ModelInfo fields:** `model_id` (str), `name` (str), `description` (str | None)

**SessionMode fields:** `id` (str), `name` (str), `description` (str | None)

## Structured Output

When `rendered.output_type is not None`, the adapter uses an **MCP tool-based
approach** rather than prompt augmentation.

### The `structured_output` Tool

Register a special MCP tool that the model must call to finalize. The `data`
field accepts any JSON value (object or array) to support array-root schemas:

```python
# _structured_output.py
from typing import Any

@FrozenDataclass()
class StructuredOutputParams:
    data: Any

def structured_output_handler(
    params: StructuredOutputParams,
    *,
    context: ToolContext,
) -> ToolResult[None]:
    # Validate against rendered.output_type
    # Store in adapter state
    # Return success or validation error
```

### Schema Generation

Use WINK's schema generation via `weakincentives.serde.schema()` to produce the
JSON Schema from the output type. This handles array containers
(`rendered.container == "array"`) and extra keys policy
(`rendered.allow_extra_keys`). The exact helper may be extracted into
`weakincentives.adapters._shared` as part of implementation.

### Tool Description

```
Call this tool to submit your final structured output.
The data must conform to the following JSON schema:
{json_schema}
```

### Retrieval

After `conn.prompt()` returns and trailing updates are drained:

1. Check if model called `structured_output`
1. If called: retrieve validated output
1. If not called or validation failed: raise `PromptEvaluationError(phase="response")`

`structured_output` should emit a `ToolInvoked` event like any other bridged tool.

## MCP Tool Bridging

### Reusing Shared Adapter Infrastructure

The adapter reuses the **shared MCP bridge** extracted into
`src/weakincentives/adapters/_shared/` (see `_shared/__init__.py` for exports):

```python
from weakincentives.adapters._shared import (
    BridgedTool,
    VisibilityExpansionSignal,
    create_bridged_tools,
    create_mcp_server,
)
```

This is the same infrastructure used by both the Claude Agent SDK adapter and
the Codex App Server adapter.

Benefits:

- Direct access to WINK session state and resources
- Full transactional semantics without IPC
- Proven, tested implementation shared across adapters

### Reused Components

| Component | Source | Purpose |
|-----------|--------|---------|
| `BridgedTool` | `adapters/_shared/_bridge.py` | Transactional tool wrapper |
| `create_bridged_tools()` | `adapters/_shared/_bridge.py` | Factory for BridgedTool |
| `create_mcp_server()` | `adapters/_shared/_bridge.py` | In-process MCP server |
| `VisibilityExpansionSignal` | `adapters/_shared/_visibility_signal.py` | Exception propagation |
| `tool_transaction()` | `runtime/transactions.py` | Snapshot/restore |

> **Important:** Pass `adapter_name="opencode_acp"` to `create_bridged_tools()`
> to ensure `ToolInvoked` events are labeled correctly.

### Tool Bridging Flow

```
1. Render prompt → rendered.tools
2. create_bridged_tools(..., adapter_name="opencode_acp")
3. Create structured_output tool if output_type declared
4. create_mcp_server(bridged_tools + structured_output_tool)
5. Expose MCP server over HTTP on localhost (StreamableHTTPServerTransport)
6. Pass MCP server URL as HttpMcpServer on new_session(mcp_servers=[...])
7. OpenCode connects to MCP HTTP server
8. Tool call → in-process BridgedTool.__call__()
   ├─ Snapshot session state
   ├─ Execute handler
   ├─ Dispatch ToolInvoked
   └─ Rollback on failure
```

### BridgedTool Semantics

Each invocation:

1. **Snapshot** — Capture session and resource state
1. **Execute** — Call handler with parsed parameters
1. **Dispatch** — Emit `ToolInvoked` event
1. **Rollback** — Restore snapshot on failure

Handles: parameter parsing (`serde.parse()`), result formatting,
`VisibilityExpansionRequired` capture, deadline/budget enforcement.

### Visibility Expansion

When a tool raises `VisibilityExpansionRequired`:

1. `BridgedTool` catches and stores in `VisibilityExpansionSignal`
1. Returns non-error result explaining expansion need
1. After `conn.prompt()` and update draining, adapter checks signal
1. If set, re-raises to caller for re-render

## Execution Flow

### 1. Budget/Deadline Setup

- Create `BudgetTracker` if budget provided
- Derive deadline from argument or `budget.deadline`
- Raise `PromptEvaluationError(phase="request")` if already expired

### 2. Render Prompt

At `adapter.py`: `_render_prompt()`:

1. `prompt.render(session=session)` → `RenderedPrompt` (text + tools + output_type)
1. Resolve CWD and bind `HostFilesystem` resource if prompt has no filesystem
1. Emit `PromptRendered`

### 3. Start MCP Server

At `adapter.py`: `_build_mcp_server()`:

1. `create_bridged_tools(rendered.tools, adapter_name="opencode_acp", ...)`
1. Create `structured_output` tool if `output_type` declared
1. `create_mcp_server(all_tools)` → `mcp.server.Server` instance
1. Expose server over HTTP on localhost → `HttpMcpServer` config

### 4. Spawn OpenCode

At `adapter.py`: `_execute_protocol()`:

```python
from acp import spawn_agent_process

async with spawn_agent_process(
    client,
    client_config.opencode_bin,
    *client_config.opencode_args,
    cwd=resolved_cwd,
    env=merged_env,
) as (conn, proc):
    ...
```

The subprocess communicates over stdio. Pass `cwd` as a `spawn_agent_process`
kwarg for the process working directory, and separately as the `cwd` parameter
on `new_session` for the ACP session working directory (they should match).

### 5. Initialize

At `adapter.py`: `_initialize()`:

```python
from acp import PROTOCOL_VERSION
from acp.schema import (
    ClientCapabilities,
    FileSystemCapability,
    Implementation,
)

init_resp = await asyncio.wait_for(
    conn.initialize(
        protocol_version=PROTOCOL_VERSION,
        client_capabilities=ClientCapabilities(
            fs=FileSystemCapability(
                read_text_file=allow_file_reads,
                write_text_file=allow_file_writes,
            ),
            terminal=allow_terminal,
        ),
        client_info=Implementation(
            name="wink", title="WINK", version="...",
        ),
    ),
    timeout=client_config.startup_timeout_s,
)
```

Inspect `init_resp.agent_capabilities.load_session` to determine whether
`load_session` is available for session reuse. On timeout, raise
`PromptEvaluationError(phase="request")`.

### 6. Create or Load Session

At `adapter.py`: `_create_or_load_session()`:

```python
if reuse_session and state and state.cwd == resolved_cwd:
    try:
        await conn.load_session(
            session_id=state.session_id,
            cwd=resolved_cwd,
            mcp_servers=mcp_configs,
        )
        session_id = state.session_id
    except Exception:
        # Fall back to new session
        session_resp = await conn.new_session(cwd=resolved_cwd, mcp_servers=mcp_configs)
        session_id = session_resp.session_id
else:
    session_resp = await conn.new_session(cwd=resolved_cwd, mcp_servers=mcp_configs)
    session_id = session_resp.session_id
```

After `new_session`, store session state and inspect the response:

- Extract `session_resp.models.available_models` for model validation
- Extract `session_resp.modes.available_modes` for mode validation
- Store via `session.seed(OpenCodeACPSessionState(...))`

### 7. Set Mode/Model (Best-Effort with Validation)

At `adapter.py`: `_configure_session()`:

- **Model:** If `model_id` configured, validate against
  `session_resp.models.available_models`. If the model ID is not in the list,
  raise `PromptEvaluationError(phase="request")` immediately — do not proceed
  to `set_session_model` with an invalid ID. If valid, call
  `conn.set_session_model(session_id=..., model_id=...)`. Treat `RequestError`
  as non-fatal (log and continue with default model).

- **Mode:** If `mode_id` configured, call
  `conn.set_session_mode(session_id=..., mode_id=...)`. Treat all errors as
  non-fatal — OpenCode 1.1.59 returns "Internal error" for `set_session_mode`.
  Log the failure and continue.

### 8. Prompt

At `adapter.py`: `_send_prompt()`:

```python
from acp import text_block

prompt_resp = await asyncio.wait_for(
    conn.prompt(
        session_id=session_id,
        prompt=[text_block(rendered_text)],
    ),
    timeout=remaining_deadline,
)
```

The `session_update()` callback on `OpenCodeACPClient` captures streaming
updates asynchronously during the prompt call.

### 9. Drain Trailing Updates

At `adapter.py`: `_drain_updates()`:

**This step is mandatory.** The ACP stdio transport delivers `session/update`
notifications asynchronously. Notifications in flight when `PromptResponse`
arrives on the wire are lost unless the client actively drains them.

Wait until no new updates arrive for `quiet_period_ms`; reset the timer on each
new update. Cap total drain time by the deadline. A minimum quiet period of
`500ms` is recommended based on live validation.

### 10. Extract Results

At `adapter.py`: `_extract_results()`:

**Text:** Concatenate `AgentMessageChunk` content from the
`SessionAccumulator` snapshot (`snapshot.agent_messages`). Prepend
`AgentThoughtChunk` text if `emit_thought_chunks` is configured
(`snapshot.agent_thoughts`).

**Tool events:** Map terminal `ToolCallProgress` updates to `ToolInvoked`:

| ACP `ToolCallProgress.status` | Action |
|-------------------------------|--------|
| `"completed"` | `ToolInvoked` with `success=True` |
| `"failed"` | `ToolInvoked` with `success=False` |
| `"pending"` / `"in_progress"` | (intermediate, no event) |

**Deduplication:** Skip `ToolInvoked` for bridged WINK tools using explicit MCP
server metadata when available; fall back to the `mcp__wink__` prefix if needed.
`BridgedTool` already emitted the event.

**Stop reason handling:** `PromptResponse.stop_reason` must be inspected:

| `stop_reason` | Action |
|---------------|--------|
| `"end_turn"` | Normal completion |
| `"max_tokens"` | Log warning; response may be truncated |
| `"max_turn_requests"` | Log warning; agent hit turn limit |
| `"refusal"` | Raise `PromptEvaluationError(phase="response")` |
| `"cancelled"` | Raise `PromptEvaluationError(phase="request")` or `DeadlineExceededError` |

### 11. Structured Output

If declared, retrieve from `structured_output` tool invocation.
Raise `PromptEvaluationError(phase="response")` if missing or invalid.

### 12. PromptExecuted

Emit event (includes `TokenUsage` if available) and return
`PromptResponse(text=..., output=...)`.

## Cancellation

If deadline expires during a prompt:

1. Send `conn.cancel(session_id=...)` — ACP defines this as a **notification** (no response expected)
1. Wait briefly for the prompt to return (bounded)
1. Kill subprocess if needed
1. Raise `PromptEvaluationError(phase="request")` or `DeadlineExceededError`

## Error Handling

### Error Phases

| Phase | When |
|-------|------|
| `"request"` | Spawn, initialize, session/new, set_session_model (invalid), or prompt fails |
| `"response"` | Structured output missing or invalid; `stop_reason="refusal"` |
| `"tool"` | Bridged tool execution failure |
| `"budget"` | Token/cost budget exceeded |

### PromptResponse Error Mapping

When `prompt()` returns successfully, inspect `stop_reason`:

| `stop_reason` | WINK Action |
|---------------|-------------|
| `"end_turn"` | Normal — extract text and output |
| `"max_tokens"` | Warn; proceed with truncated output |
| `"max_turn_requests"` | Warn; proceed with partial output |
| `"refusal"` | `PromptEvaluationError(phase="response")` |
| `"cancelled"` | `PromptEvaluationError(phase="request")` or `DeadlineExceededError` |

### Empty Response Detection

When `stop_reason="end_turn"` but zero `AgentMessageChunk` updates were
received, the response is empty. This occurs when:

- An invalid `model_id` was accepted by `set_session_model` but the model
  cannot generate output
- The agent encountered an internal error not surfaced via `stop_reason`

The adapter must detect empty responses and raise
`PromptEvaluationError(phase="response")` with diagnostic details (model ID,
session ID, update count, stderr tail).

### Error Payload

Include in payload: stderr tail (bounded, e.g., last 8k), ACP error details,
`stop_reason`, recent session updates (bounded).

Tool telemetry errors: log but don't crash.

## Events

| Event | When |
|-------|------|
| `PromptRendered` | After render, before `conn.prompt()` |
| `ToolInvoked` | Each bridged tool call + each native OpenCode tool (command, file change) |
| `PromptExecuted` | After update draining completes (includes `TokenUsage` if available) |

## Token Usage

### PromptResponse.usage

The `PromptResponse` returned by `conn.prompt()` includes a `Usage` object:

| ACP `Usage` field | WINK `TokenUsage` field |
|-------------------|------------------------|
| `input_tokens` | `input_tokens` |
| `output_tokens` | `output_tokens` |
| `cached_read_tokens` | `cached_tokens` |
| `thought_tokens` | (logged, not mapped) |
| `total_tokens` | (computed property in WINK) |

`cached_write_tokens` is not currently populated by OpenCode.

### UsageUpdate Notifications

OpenCode also sends `UsageUpdate` notifications via `session/update` during
prompt execution. These provide running cost and size tracking:

| `UsageUpdate` field | Description |
|---------------------|-------------|
| `cost` | `Cost` object (if available) |
| `size` | Total context size |
| `used` | Tokens used in current turn |

The adapter should prefer `PromptResponse.usage` for the final per-turn usage
(more complete), and use `UsageUpdate` for budget tracking during execution.

`TokenUsage.total_tokens` is a computed property (not stored).

## Session State Storage

WINK sessions use **typed dataclass slices**, not string keys. For session reuse:

```python
# _state.py
from weakincentives.dataclasses import FrozenDataclass

@FrozenDataclass()
class OpenCodeACPSessionState:
    """Stores OpenCode session ID and workspace fingerprint for reuse."""
    session_id: str
    cwd: str
    workspace_fingerprint: str | None
```

Usage:

```python
from pathlib import Path

# Store after session/new (use the resolved cwd passed to session/new)
resolved_cwd = client_config.cwd or str(Path.cwd().resolve())
session.seed(
    OpenCodeACPSessionState(
        session_id=result.session_id,
        cwd=resolved_cwd,
        workspace_fingerprint=workspace_fingerprint,
    )
)

# Retrieve for session/load
state = session[OpenCodeACPSessionState].latest()
if state is not None:
    opencode_session_id = state.session_id
    cached_cwd = state.cwd
    cached_workspace_fingerprint = state.workspace_fingerprint
```

When `reuse_session=True`, only reuse the session if `cached_cwd` and
`cached_workspace_fingerprint` match the current workspace. If they differ or
`session/load` fails, fall back to `session/new` and overwrite the stored state.
Compute `workspace_fingerprint` from mount config and budgets (stable ordering)
so reuse is deterministic.

## Testing

### Unit Tests

- Mock ACP agent (echo style via `run_agent`)
- Verify `PromptRendered`, `PromptExecuted` emitted once
- Verify `ToolInvoked` on terminal `ToolCallProgress` status
- Verify tool deduplication (no double events for bridged WINK tools)
- Verify structured output retrieval via MCP tool
- Verify `permission_mode="prompt"` denies without blocking
- Verify file read/write disabled when no workspace configured
- Verify `SessionAccumulator` wrapping with `SessionNotification`
- Verify empty response detection (zero `AgentMessageChunk`)
- Verify model validation against `new_session().models.available_models`
- Verify `stop_reason` mapping for all five values

### Integration Tests

Skip unless `opencode` on PATH:

- Spawn `opencode acp` in temp workspace
- Simple prompt, verify response text and `stop_reason`
- Tool invocation, verify `ToolInvoked`
- Model switching via `set_session_model`, verify response
- `PromptResponse.usage` populated with token counts
- Session reuse via `load_session`

### Security Tests

- `read_text_file`/`write_text_file` reject paths outside workspace
- `allowed_host_roots` enforced

## Usage Examples

### Basic

```python
from weakincentives import Prompt, PromptTemplate, MarkdownSection
from weakincentives.runtime import Session, InProcessDispatcher
from weakincentives.adapters.opencode_acp import (
    OpenCodeACPAdapter,
    OpenCodeACPAdapterConfig,
    OpenCodeACPClientConfig,
)

bus = InProcessDispatcher()
session = Session(dispatcher=bus)

template = PromptTemplate(
    ns="demo",
    key="opencode",
    sections=(
        MarkdownSection(
            title="Task",
            key="task",
            template="List the files in the repo and summarize.",
        ),
    ),
)
prompt = Prompt(template)

adapter = OpenCodeACPAdapter(
    adapter_config=OpenCodeACPAdapterConfig(
        model_id="openai/gpt-5.1-codex-mini",
    ),
    client_config=OpenCodeACPClientConfig(
        cwd="/absolute/path/to/workspace",
        permission_mode="auto",
        allow_file_reads=True,
        allow_file_writes=False,
    ),
)

with prompt.resources:
    resp = adapter.evaluate(prompt, session=session)

print(resp.text)
```

### With Workspace Isolation

```python
from weakincentives.adapters.opencode_acp import (
    OpenCodeACPAdapter,
    OpenCodeACPClientConfig,
)
from weakincentives.prompt import WorkspaceSection, HostMount

workspace = WorkspaceSection(
    session=session,
    mounts=(HostMount(host_path="/abs/path/to/repo", mount_path="repo"),),
    allowed_host_roots=("/abs/path/to",),
)

adapter = OpenCodeACPAdapter(
    client_config=OpenCodeACPClientConfig(
        cwd=str(workspace.temp_dir),
        allow_file_reads=True,
        allow_file_writes=True,
    ),
)
```

### Structured Output

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Summary:
    title: str
    files: list[str]
    line_count: int

template = PromptTemplate[Summary](
    ns="demo",
    key="summarize",
    sections=(
        MarkdownSection(
            title="Task",
            key="task",
            template="Summarize the repository structure.",
        ),
    ),
)

prompt = Prompt(template)
with prompt.resources:
    resp = adapter.evaluate(prompt, session=session)

summary: Summary = resp.output  # Typed structured output
```

## Non-Goals (v1)

- Full OpenCode config management (providers, keys)
- Perfect Claude sandboxing parity (workspace isolation is primary boundary)
- Session fork/resume/list (ACP supports these; add when needed)
- Authentication flows (OpenCode inherits host-level credentials)
- `ConfigOptionUpdate` / `SessionInfoUpdate` processing

## Design Decisions

### Why MCP Tool Bridging (Not Native ACP Tools)

ACP does not define a "dynamic tool" protocol equivalent to what Codex provides.
The tool bridging mechanism in ACP is **MCP servers passed on `new_session`**.
The agent connects to the MCP server and discovers tools via the standard MCP
protocol. This is the intended ACP approach and works cleanly because:

- WINK already has a proven `create_mcp_server()` implementation
- The `claude-agent-sdk` dependency is already required for the Claude adapter
- OpenCode's `McpCapabilities` confirms HTTP and SSE MCP support
- No custom protocol or agent-specific tool registration needed

The alternative — a custom stdio-based tool call/response protocol — would
require agent-specific code and is not part of the ACP specification.

### Why MCP `structured_output` Tool (Not Native)

ACP does not define a native structured output mechanism like Codex's
`outputSchema`. The MCP tool approach provides:

- Schema validation in-process with full WINK type system access
- Consistent behavior across all agents (any ACP agent that supports MCP tools
  can use structured output)
- Proven pattern from the Claude Agent SDK adapter

### Why `quiet_period_ms` Is Mandatory

The ACP Python SDK's stdio transport delivers `session/update` notifications as
NDJSON lines on stdout, interleaved with JSON-RPC responses. When
`PromptResponse` arrives, the SDK resolves the `prompt()` future immediately.
Any notifications still in the OS pipe buffer or not yet read by the transport
layer are effectively lost from the caller's perspective.

Live validation with OpenCode 1.1.59 confirmed that the final
`AgentMessageChunk` and `UsageUpdate` are routinely in flight when
`PromptResponse` arrives. Without draining, the adapter returns truncated text
and no usage data. A quiet period of 500ms was sufficient in all tested
scenarios; 100ms was not.

### Why Token Usage Is Tracked

The ACP SDK provides token usage in two places:

1. `PromptResponse.usage` — per-turn breakdown (input, output, cached, thought tokens)
2. `UsageUpdate` notifications — running cost and size during execution

Both are populated by OpenCode. Tracking usage is effectively free and enables
WINK budget enforcement. This was initially listed as a non-goal based on the
assumption that ACP might not expose usage reliably — live validation confirmed
it does.

## Appendix: Protocol Reference

### Validated with Probe Scripts

All protocol details in this spec were validated against `opencode 1.1.59` with
`agent-client-protocol 0.8.0` using probe scripts in
`/tmp/acp-validation-scripts/`. Key findings:

- `spawn_agent_process` communicates over stdio (not HTTP)
- `PROTOCOL_VERSION` is `1` (integer)
- `ClientCapabilities` uses typed Pydantic models (`FileSystemCapability`), not raw dicts
- `SessionAccumulator.apply()` requires `SessionNotification` wrappers
- `ToolCallStart` uses `title` for tool name, not `name`
- `ToolCallStart.status` accepts `"pending"` | `"in_progress"` | `"completed"` | `"failed"`
- `set_session_mode` returns "Internal error" from OpenCode (modes work via `new_session`)
- `set_session_model` accepts invalid model IDs without error; prompt returns empty
- `RequestError` is in `acp` top-level, not `acp.schema`

### Available Models (OpenCode 1.1.59)

Models observed via `new_session().models.available_models`:

**OpenAI:** `openai/gpt-5.3-codex`, `openai/gpt-5.2-codex`, `openai/gpt-5.2`,
`openai/gpt-5.1-codex-mini`, `openai/gpt-5.1-codex-max`, `openai/gpt-5.1-codex`

**OpenCode Zen:** `opencode/big-pickle`, `opencode/gpt-5-nano`,
`opencode/minimax-m2.5-free`, `opencode/kimi-k2.5-free`

Most models support variants: `low`, `medium`, `high`, `xhigh` (appended as
`model_id/variant`). Model availability depends on auth type and plan.

### Available Modes (OpenCode 1.1.59)

| Mode ID | Description |
|---------|-------------|
| `build` | Default agent. Executes tools based on configured permissions. |
| `plan` | Plan mode. Disallows all edit tools. |

### MCP Server Config Format

The adapter exposes the in-process MCP server over HTTP and passes its URL to
OpenCode as an `HttpMcpServer`:

```python
from acp.schema import HttpMcpServer

HttpMcpServer(
    url="http://127.0.0.1:{port}/mcp",
    name="wink-tools",
    headers=[],
    type="http",
)
```

Fields: `url` (str, required), `name` (str, required), `headers`
(list[HttpHeader], required — pass `[]` when no auth needed), `type`
(Literal["http"], required).

OpenCode also supports `McpServerStdio` and `SseMcpServer` per its
`McpCapabilities`, but `HttpMcpServer` is used here because the MCP server
runs in the adapter process and OpenCode is a separate subprocess — stdio
would require spawning yet another process.

## Related Specifications

- `specs/ADAPTERS.md` — Provider adapter protocol
- `specs/CLAUDE_AGENT_SDK.md` — Reference adapter architecture (shared bridge)
- `specs/CODEX_APP_SERVER.md` — Sibling adapter using shared bridge
- `specs/PROMPTS.md` — Prompt system
- `specs/SESSIONS.md` — Session state and events
- `specs/TOOLS.md` — Tool registration and policies
- `specs/WORKSPACE.md` — Workspace management

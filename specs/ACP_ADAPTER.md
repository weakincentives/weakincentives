# ACP Adapter Specification

> **Status:** Implemented.
> **Package:** `src/weakincentives/adapters/acp/`
> **Adapter name:** `acp`
> **ACP protocol:** v1 (JSON-RPC 2.0 over newline-delimited JSON on stdio)
> **Tested against:** `agent-client-protocol 0.8.0`

## Purpose

`ACPAdapter` is a generic adapter for evaluating WINK prompts via any
ACP-compatible agent binary. It implements the full Agent Client Protocol flow
(spawn → initialize → new_session → prompt → drain) and bridges WINK tools via
an in-process MCP HTTP server.

Agent-specific behavior (model validation, error tolerance, empty response
detection) is delegated to subclass hooks, making this a reusable base for
OpenCode, Gemini CLI, and future ACP agents.

| Responsibility | Owner |
|----------------|-------|
| Prompt composition, resource binding, session telemetry | WINK |
| Agentic execution (planning, reasoning, tool calls, file edits) | ACP agent |

WINK receives progress via ACP `session/update` notifications and emits
canonical events: `PromptRendered`, `ToolInvoked`, `PromptExecuted`.

## Why ACP

The Agent Client Protocol is a **vendor-neutral** standard for client-agent
communication. Where the Codex App Server exposes a Codex-specific stdio
protocol and the Claude Agent SDK exposes an Anthropic-specific Python API, ACP
provides a single protocol implemented by multiple agents.

Key capabilities surfaced through ACP:

- **Sessions with mode/model selection:** Persistent conversation state with
  configurable agent modes and model switching at runtime
- **Native tools:** Command execution, file changes, web search — all streamed
  as typed `ToolCallStart` / `ToolCallProgress` updates
- **MCP server passthrough:** External MCP servers passed on `new_session` are
  connected by the agent, bridging WINK tools without a separate process
- **File and terminal capabilities:** Client-side file I/O and terminal
  management via typed `Client` interface methods
- **Model discovery:** `new_session` returns available models and modes,
  enabling runtime validation

## Requirements

### Runtime Dependencies

1. An ACP-compatible agent binary installed and available on `PATH`
1. **ACP Python SDK**: `agent-client-protocol>=0.8.0`
1. **Claude Agent SDK**: `claude-agent-sdk>=0.1.27` (for shared MCP server infrastructure)
1. WINK (`weakincentives`) runtime

### WINK Packaging

See `pyproject.toml` — the `acp` optional dependency group:

```toml
acp = [
  "agent-client-protocol>=0.8.0",
  "claude-agent-sdk>=0.1.27",
]
```

All ACP imports are lazy (inside method bodies) with a helpful `ImportError`
message if the extra is not installed.

## Architecture

```
WINK Prompt/Session
  └─ ACPAdapter.evaluate()
      ├─ Render WINK prompt → markdown text
      ├─ Build in-process MCP server (reuses Claude SDK infrastructure)
      ├─ Expose MCP server over HTTP (same process, localhost)
      ├─ Spawn: <agent_bin> <agent_args> (stdio JSON-RPC NDJSON)
      ├─ ACP handshake: initialize → session/new (passes HttpMcpServer URL)
      ├─ Agent connects to MCP HTTP server → discovers WINK tools
      ├─ set_session_model / set_session_mode (best-effort)
      ├─ conn.prompt(text)
      ├─ Stream: session/update notifications
      │    ├─ AgentMessageChunk (assistant output)
      │    ├─ AgentThoughtChunk (reasoning, if configured)
      │    ├─ ToolCallStart / ToolCallProgress (WINK + agent tools)
      │    └─ UsageUpdate (token/cost tracking)
      ├─ Drain trailing updates (quiet_period_ms)
      ├─ Model calls structured_output tool to finalize (if required)
      └─ Return PromptResponse(text, output, token_usage)
```

The MCP server runs **in the adapter process** and is exposed over HTTP on
localhost. The `create_mcp_server()` shared infrastructure creates an
`mcp.server.Server` instance with all bridged tools registered. Unlike the
Claude Agent SDK adapter (which passes the server instance in-process), ACP
agents are separate subprocesses, so the adapter wraps the server with HTTP
transport and passes the URL as an `HttpMcpServer` config on `new_session`.

## Module Structure

```
src/weakincentives/adapters/acp/
  __init__.py              # Public exports
  adapter.py               # ACPAdapter (full ACP protocol flow)
  config.py                # ACPClientConfig, ACPAdapterConfig
  client.py                # ACPClient (generic Client implementation)
  _state.py                # ACPSessionState slice
  _events.py               # ACP updates → WINK events mapping
  _structured_output.py    # structured_output MCP tool
  _mcp_http.py             # HTTP transport for in-process MCP server
  _async.py                # asyncio helpers (re-export run_async)
```

## Configuration

### ACPClientConfig

Defined at `src/weakincentives/adapters/acp/config.py:29`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `agent_bin` | `str` | `"opencode"` | Executable to spawn |
| `agent_args` | `tuple[str, ...]` | `("acp",)` | Arguments for the agent binary |
| `cwd` | `str \| None` | `None` | Working directory (absolute; defaults to `Path.cwd().resolve()`) |
| `env` | `Mapping[str, str] \| None` | `None` | Extra environment variables (merged with `os.environ`) |
| `suppress_stderr` | `bool` | `True` | Capture stderr for errors |
| `startup_timeout_s` | `float` | `10.0` | Max time for initialize/session/new |
| `permission_mode` | `Literal["auto", "deny", "prompt"]` | `"auto"` | Response to permission requests |
| `allow_file_reads` | `bool` | `False` | Advertise `readTextFile` capability |
| `allow_file_writes` | `bool` | `False` | Advertise `writeTextFile` capability |
| `allow_terminal` | `bool` | `False` | Advertise terminal capability |
| `mcp_servers` | `tuple[Any, ...]` | `()` | Additional MCP servers (WINK server always added first) |
| `reuse_session` | `bool` | `False` | Load/reuse session ID |

> **CWD requirement:** ACP requires `cwd` to be an absolute path. If `None`, the
> adapter resolves to the prompt's `HostFilesystem` root or `Path.cwd().resolve()`.

> **Capability alignment:** Advertised capabilities in `initialize` must match
> implemented methods in `ACPClient`.

> **Non-interactive permissions:** `permission_mode="prompt"` is treated as `deny`
> since the adapter cannot block for interactive prompting.

### ACPAdapterConfig

Defined at `src/weakincentives/adapters/acp/config.py:64`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode_id` | `str \| None` | `None` | ACP `session/set_mode` (best-effort) |
| `model_id` | `str \| None` | `None` | ACP `session/set_model` |
| `quiet_period_ms` | `int` | `500` | Wait after prompt returns to drain trailing updates |
| `emit_thought_chunks` | `bool` | `False` | Include `AgentThoughtChunk` text in returned output |

> **Quiet period is mandatory.** See [Quiet Period Design Decision](#why-quiet_period_ms-is-mandatory).

## ACP Client Implementation

Defined at `src/weakincentives/adapters/acp/client.py:31`.

`ACPClient` implements the `acp.interfaces.Client` protocol. All methods are `async`.

| Method | ACP Protocol Message | Behavior |
|--------|---------------------|----------|
| `session_update` | `session/update` notification | Tracks message/thought/tool chunks and last update time |
| `request_permission` | `session/request_permission` | Auto-approve for `"auto"`, deny otherwise |
| `read_text_file` | `fs/read_text_file` | Read within workspace root (if capability advertised) |
| `write_text_file` | `fs/write_text_file` | Write within workspace root (if capability advertised) |
| `create_terminal` | `terminal/create` | Raise `RequestError.method_not_found` |
| `terminal_output` | `terminal/output` | Raise `RequestError.method_not_found` |
| `release_terminal` | `terminal/release` | Raise `RequestError.method_not_found` |
| `wait_for_terminal_exit` | `terminal/wait_for_exit` | Raise `RequestError.method_not_found` |
| `kill_terminal` | `terminal/kill` | Raise `RequestError.method_not_found` |
| `ext_method` | `_*` extension request | Raise `RequestError.method_not_found` |
| `ext_notification` | `_*` extension notification | Ignore silently |

The client tracks `message_chunks`, `thought_chunks`, and `tool_call_tracker`
for result extraction after the prompt completes. `last_update_time` (monotonic)
is updated on each `session_update` for quiet period drain.

> **Important:** `RequestError` is imported from `acp` (top-level), not
> `acp.schema`. Use `RequestError.method_not_found(method_name)`.

## MCP HTTP Server

Defined at `src/weakincentives/adapters/acp/_mcp_http.py:35`.

`MCPHttpServer` wraps an `mcp.server.Server` instance with HTTP transport
(Starlette + uvicorn on a daemon thread). Used to expose WINK bridged tools to
the ACP agent subprocess.

Key methods:

- `start()` — Picks a free port, starts uvicorn on a daemon thread
- `stop()` — Signals shutdown, joins thread
- `url` — Returns `http://127.0.0.1:{port}/mcp`
- `to_http_mcp_server()` — Returns `acp.schema.HttpMcpServer` for `new_session`

## Structured Output

When `rendered.output_type is not None`, the adapter registers a special MCP
tool that the model must call to finalize structured output.

### StructuredOutputTool

Defined at `src/weakincentives/adapters/acp/_structured_output.py:60`.

Duck-typed MCP tool with `.name`, `.description`, `.input_schema`, `__call__`.
Compatible with `create_mcp_server()`.

### Factory

`create_structured_output_tool()` at
`src/weakincentives/adapters/acp/_structured_output.py:109` builds the tool
from the output type, using `weakincentives.serde.schema()` for JSON Schema
generation.

### Retrieval

After `conn.prompt()` returns and trailing updates are drained, the adapter
checks `StructuredOutputCapture.called`. If the tool was invoked, it parses the
captured data. If not called, it falls back to parsing the accumulated text.
See `ACPAdapter._resolve_structured_output()` at
`src/weakincentives/adapters/acp/adapter.py:588`.

## MCP Tool Bridging

The adapter reuses the **shared MCP bridge** from
`src/weakincentives/adapters/_shared/_bridge.py`:

| Component | Purpose |
|-----------|---------|
| `BridgedTool` | Transactional tool wrapper |
| `create_bridged_tools()` | Factory for BridgedTool instances |
| `create_mcp_server()` | In-process MCP server creation |
| `VisibilityExpansionSignal` | Exception propagation from tool calls |

### Tool Bridging Flow

See `ACPAdapter._prepare_tools()` at
`src/weakincentives/adapters/acp/adapter.py:281` and
`ACPAdapter._execute_protocol()` at
`src/weakincentives/adapters/acp/adapter.py:380`.

1. Render prompt → `rendered.tools`
1. `create_bridged_tools(...)` → list of `BridgedTool`
1. Create `structured_output` tool if `output_type` declared
1. `create_mcp_server(all_tools)` → `mcp.server.Server` instance
1. Expose via `MCPHttpServer` on localhost
1. Pass URL as `HttpMcpServer` on `new_session(mcp_servers=[...])`
1. Agent connects to MCP HTTP server
1. Tool call → in-process `BridgedTool.__call__()`

### Visibility Expansion

When a bridged tool raises `VisibilityExpansionRequired`:

1. `BridgedTool` catches and stores in `VisibilityExpansionSignal`
1. Returns non-error result explaining expansion need
1. After drain, adapter checks signal and re-raises if set

See `src/weakincentives/adapters/acp/adapter.py:442-444`.

## Execution Flow

The main entry point is `ACPAdapter.evaluate()` at
`src/weakincentives/adapters/acp/adapter.py:93`.

### 1. Budget/Deadline Setup

Create `BudgetTracker` if budget provided. Derive deadline from argument or
`budget.deadline`. Raise `PromptEvaluationError(phase="request")` if expired.

### 2. Render Prompt

`_evaluate_async()` at `src/weakincentives/adapters/acp/adapter.py:129`:
renders the prompt, resolves CWD, binds `HostFilesystem` if needed, dispatches
`PromptRendered`.

### 3. Resolve CWD

`_resolve_cwd()` at `src/weakincentives/adapters/acp/adapter.py:185`:

1. Use config `cwd` if set
1. Else use prompt's `HostFilesystem` root
1. Else fall back to `Path.cwd().resolve()`
1. Create temp directory if no cwd source and no filesystem

### 4. Build Tools and MCP Server

`_prepare_tools()` at `src/weakincentives/adapters/acp/adapter.py:281` and
`_run_acp()` at `src/weakincentives/adapters/acp/adapter.py:208`.

### 5. Execute Protocol

`_execute_protocol()` at `src/weakincentives/adapters/acp/adapter.py:380`:

1. Lazy import `acp.spawn_agent_process` and `acp.schema.HttpMcpServer`
1. Start `MCPHttpServer`
1. Spawn agent subprocess via `spawn_agent_process()`
1. `_handshake()` — initialize + new_session
1. `_configure_session()` — set model/mode (best-effort)
1. `conn.prompt()` with rendered text
1. `_drain_quiet_period()` — wait for trailing updates
1. Check visibility signal
1. Dispatch `ToolInvoked` for tracked tool calls
1. Extract text and token usage
1. Return `(text, usage)` tuple

### 6. Finalize Response

`_finalize_response()` at `src/weakincentives/adapters/acp/adapter.py:323`:
resolves structured output, records budget, dispatches `PromptExecuted`,
returns `PromptResponse`.

## Events

### ACP Update Types → WINK Events

Defined at `src/weakincentives/adapters/acp/_events.py:37`.

| ACP Update Type | WINK Event | Notes |
|-----------------|------------|-------|
| `AgentMessageChunk` | Text accumulation | Concatenated for `PromptResponse.text` |
| `AgentThoughtChunk` | Text accumulation (if `emit_thought_chunks`) | Prepended to response text |
| `ToolCallStart` | (tracking) | Records tool call ID and title |
| `ToolCallProgress` (`completed`) | `ToolInvoked(success=True)` | Via `dispatch_tool_invoked()` |
| `ToolCallProgress` (`failed`) | `ToolInvoked(success=False)` | Via `dispatch_tool_invoked()` |

### WINK Events Emitted

| Event | When |
|-------|------|
| `PromptRendered` | After render, before `conn.prompt()` |
| `ToolInvoked` | Each bridged tool call + each agent-native tool |
| `PromptExecuted` | After update draining completes (includes `TokenUsage`) |

## Token Usage

`extract_token_usage()` at `src/weakincentives/adapters/acp/_events.py:80`
maps ACP `Usage` to WINK `TokenUsage`:

| ACP `Usage` field | WINK `TokenUsage` field |
|-------------------|------------------------|
| `input_tokens` | `input_tokens` |
| `output_tokens` | `output_tokens` |
| `cached_read_tokens` | `cached_tokens` |
| `thought_tokens` | (logged, not mapped) |

## Session State

`ACPSessionState` at `src/weakincentives/adapters/acp/_state.py:23`:
frozen dataclass with `session_id`, `cwd`, `workspace_fingerprint` for
session reuse.

## Subclass Hooks

`ACPAdapter` exposes three protected methods for agent-specific behavior:

| Hook | Location | Default |
|------|----------|---------|
| `_validate_model(model_id, available_models)` | `adapter.py:632` | No-op |
| `_handle_mode_error(error)` | `adapter.py:635` | Log warning |
| `_detect_empty_response(client, prompt_resp)` | `adapter.py:643` | No-op |

Subclasses override these for agent-specific quirks. See
`specs/OPENCODE_ADAPTER.md` for the OpenCode implementation.

## Error Handling

### Error Phases

| Phase | When |
|-------|------|
| `"request"` | Spawn, initialize, session/new, or prompt fails |
| `"response"` | Structured output missing or invalid |
| `"tool"` | Bridged tool execution failure |
| `"budget"` | Token/cost budget exceeded |

Unexpected exceptions from the ACP protocol flow are wrapped in
`PromptEvaluationError`. `VisibilityExpansionRequired` and
`PromptEvaluationError` propagate without wrapping.
See `src/weakincentives/adapters/acp/adapter.py:258-267`.

## Design Decisions

### Why MCP Tool Bridging (Not Native ACP Tools)

ACP does not define a "dynamic tool" protocol. Tool bridging uses **MCP servers
passed on `new_session`**. The agent connects and discovers tools via MCP. This
reuses the proven `create_mcp_server()` infrastructure shared with other adapters.

### Why MCP `structured_output` Tool

ACP does not define a native structured output mechanism. The MCP tool approach
provides in-process schema validation with the full WINK type system, consistent
behavior across all ACP agents.

### Why `quiet_period_ms` Is Mandatory

The ACP stdio transport delivers `session/update` notifications as NDJSON lines
interleaved with JSON-RPC responses. When `PromptResponse` arrives, the SDK
resolves the `prompt()` future immediately. Notifications still in the pipe
buffer are lost from the caller's perspective. The quiet period drain waits
until no new updates arrive for the configured duration.

## Testing

Tests at `tests/adapters/acp/`:

| File | Coverage |
|------|----------|
| `test_adapter.py` | Unit tests for `ACPAdapter` methods |
| `test_adapter_protocol.py` | Full protocol flow tests with mocked ACP |
| `test_client.py` | `ACPClient` method tests |
| `test_config.py` | Configuration dataclass tests |
| `test_events.py` | Event dispatch and token usage extraction |
| `test_mcp_http.py` | `MCPHttpServer` lifecycle tests |
| `test_structured_output.py` | `StructuredOutputTool` tests |
| `test_state.py` | `ACPSessionState` tests |
| `conftest.py` | Mock ACP types and fixtures |

## Related Specifications

- `specs/ADAPTERS.md` — Provider adapter protocol
- `specs/OPENCODE_ADAPTER.md` — OpenCode-specific subclass
- `specs/CLAUDE_AGENT_SDK.md` — Reference adapter architecture (shared bridge)
- `specs/CODEX_APP_SERVER.md` — Sibling adapter using shared bridge
- `specs/TOOLS.md` — Tool registration and policies

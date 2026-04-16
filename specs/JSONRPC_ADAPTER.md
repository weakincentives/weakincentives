# JSON-RPC Provider Adapter Specification

> **Status:** Implemented.
> **Package:** `src/weakincentives/adapters/jsonrpc/`
> **Adapter name:** `jsonrpc` (base; concrete subclasses provide own names)
> **Protocol:** JSON-RPC over newline-delimited JSON (stdio) or WebSocket
> **First concrete subclass:** Codex App Server (`codex_app_server`)

## Purpose

`JsonRpcAdapter` is a generic base adapter for evaluating WINK prompts via any
agent that speaks a turn-based JSON-RPC protocol over stdio or WebSocket. It
extracts the common protocol lifecycle from the Codex App Server adapter into a
reusable base class, following the same pattern as `ACPAdapter` for
ACP-compatible agents.

| Responsibility | Owner |
|----------------|-------|
| Prompt composition, resource binding, session telemetry | WINK (base adapter) |
| Transport lifecycle (spawn, connect, read/write) | `JsonRpcClient` |
| Protocol orchestration (init, turn loop, stream, interrupt) | Base adapter + hooks |
| Provider-specific handshake, notification processing, error mapping | Concrete subclass |
| Agentic execution (planning, reasoning, tool calls, file edits) | Remote agent |

## Why a Generic JSON-RPC Adapter

The Codex App Server adapter implements a protocol pattern that is not
Codex-specific:

1. **Initialize handshake** — client sends capabilities, server acknowledges
1. **Create session** — establish a conversation context (thread, session)
1. **Turn-based execution** — send prompt text, stream responses, complete
1. **Bidirectional tool calls** — server sends requests back to client for
   tool execution
1. **Deadline enforcement** — client sends interrupt on timeout
1. **Token tracking** — server reports usage via notifications

Any agent that implements this pattern over JSON-RPC can reuse the base adapter.
The Codex App Server is the first concrete subclass. Future providers using
similar JSON-RPC protocols (with different method names and payload shapes) can
extend `JsonRpcAdapter` with minimal effort.

### Comparison with ACP Adapter

| Aspect | ACP Adapter | JSON-RPC Adapter |
|--------|-------------|------------------|
| **Protocol** | ACP (JSON-RPC 2.0 with header) | JSON-RPC (header optional) |
| **Tool bridging** | MCP HTTP server (out-of-process) | Dynamic tools (in-process) |
| **Dependencies** | `agent-client-protocol`, `mcp`, `uvicorn` | None (stdlib only) |
| **Structured output** | MCP tool | Provider-native (e.g. `outputSchema`) |
| **Subclass pattern** | Hook methods on `ACPAdapter` | Hook methods on `JsonRpcAdapter` |

## Architecture

```
WINK Prompt/Session
  └─ JsonRpcAdapter.evaluate()          [base class]
      ├─ Render WINK prompt → markdown text
      ├─ create_bridged_tools() → BridgedTool list
      ├─ _build_tool_specs() → provider format  [hook]
      ├─ _build_output_schema() → schema         [hook]
      ├─ _setup_environment() → env + cleanup     [hook]
      ├─ _create_client() → JsonRpcClient         [hook]
      ├─ client.start()                           [generic]
      ├─ _initialize_session()                    [hook]
      ├─ Turn loop (max 10 rounds):               [generic]
      │    ├─ _start_turn()                       [hook]
      │    ├─ Stream messages:                    [generic loop]
      │    │    ├─ Server request → _handle_server_request()  [hook]
      │    │    ├─ Notification → _process_notification()     [hook]
      │    │    ├─ Apply result (delta/text/usage/done)       [generic]
      │    │    └─ Visibility signal check                    [generic]
      │    ├─ Deadline watchdog → _send_interrupt() [hook]
      │    └─ check_task_completion()              [generic]
      ├─ Build PromptResponse                      [generic]
      └─ Dispatch PromptExecuted                   [generic]
```

## Module Structure

```
src/weakincentives/adapters/jsonrpc/
  __init__.py              # Public exports
  adapter.py               # JsonRpcAdapter (base class with hooks)
  client.py                # JsonRpcClient (transport-aware JSON-RPC client)
  config.py                # JsonRpcClientConfig (base config)
  _types.py                # Generic JSON-RPC message TypedDicts
  _protocol.py             # Generic turn loop, message consumption, watchdog
  _response.py             # Response building, structured output, events
  _async.py                # asyncio helpers (re-export run_async)
```

## JsonRpcClient

Extracted from `CodexAppServerClient`. A transport-agnostic bidirectional
JSON-RPC client that handles message framing, request/response correlation,
and server-initiated requests.

Defined at `src/weakincentives/adapters/jsonrpc/client.py`.

### Transport Modes

| Mode | Description | Config |
|------|-------------|--------|
| **stdio** | Spawn subprocess, NDJSON over pipes | `bin_path` + `bin_args` |
| **managed WS** | Spawn subprocess with `--listen ws://`, connect | `bin_path` + `bin_ws_args` |
| **external WS** | Connect to existing server | `remote_url` |

### Public API

- `start()` — Establish the transport
- `stop()` — Tear down the transport
- `send_request(method, params, timeout)` — Send request, await response
- `send_notification(method, params)` — Send notification (no response)
- `send_response(request_id, result)` — Respond to server-initiated request
- `read_messages()` — Async iterator of notifications and server requests
- `stderr_output` — Captured stderr (stdio mode only)

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bin_path` | `str` | *(required)* | Binary to spawn (managed modes) |
| `bin_args` | `tuple[str, ...]` | *(required)* | Args for stdio mode |
| `bin_ws_args` | `tuple[str, ...]` | *(required)* | Args prefix for managed WS |
| `env` | `Mapping[str, str] \| None` | `None` | Extra environment variables |
| `suppress_stderr` | `bool` | `True` | Capture stderr quietly |
| `transport` | `Literal["stdio", "websocket"]` | `"stdio"` | Wire protocol |
| `remote_url` | `str \| None` | `None` | External WS URL |
| `ws_auth_token` | `str \| None` | `None` | Bearer token for WS auth |

## JsonRpcClientConfig

Base configuration for `JsonRpcClient` instances. Provider-specific adapters
extend this with additional fields. The binary fields (`bin_path`, `bin_args`,
`bin_ws_args`) are **required** — there are no defaults so that a second
provider cannot accidentally launch the wrong binary.

Defined at `src/weakincentives/adapters/jsonrpc/config.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `bin_path` | `str` | *(required)* | Binary to spawn |
| `bin_args` | `tuple[str, ...]` | *(required)* | Args for stdio |
| `bin_ws_args` | `tuple[str, ...]` | *(required)* | Args prefix for managed WS |
| `transport` | `Literal["stdio", "websocket"]` | `"stdio"` | Wire protocol |
| `remote_url` | `str \| None` | `None` | External WS URL |
| `ws_auth_token` | `str \| None` | `None` | Bearer token for WS auth |
| `cwd` | `str \| None` | `None` | Working directory |
| `env` | `Mapping[str, str] \| None` | `None` | Extra environment variables |
| `suppress_stderr` | `bool` | `True` | Capture stderr |
| `startup_timeout_s` | `float` | `10.0` | Max time for initialize handshake |
| `transcript` | `bool` | `True` | Emit transcript entries |
| `transcript_emit_raw` | `bool` | `True` | Include raw JSON in transcript |

## JsonRpcAdapter

Abstract base class for JSON-RPC provider adapters.

Defined at `src/weakincentives/adapters/jsonrpc/adapter.py`.

### Provided Behavior (Generic)

The base class handles:

1. **Budget/deadline setup** — create tracker, validate deadline
1. **Prompt rendering** — `prompt.render(session=session)`
1. **Event dispatch** — `PromptRendered`, `RenderedTools`, `PromptExecuted`
1. **CWD resolution** — config, filesystem, or temp directory
1. **Tool bridging** — `create_bridged_tools()` from shared bridge
1. **Turn loop** — continuation with task completion checking (max 10 rounds)
1. **Deadline enforcement** — watchdog task with interrupt hook
1. **Message consumption** — generic loop calling hooks for processing
1. **Visibility expansion** — signal check and re-raise
1. **Response building** — structured output parsing, budget recording

### Required Hooks (Abstract)

| Hook | Signature | Purpose |
|------|-----------|---------|
| `_adapter_name` | `() -> str` | Canonical adapter identifier |
| `_create_client` | `(...) -> JsonRpcClient` | Construct configured client |
| `_initialize_session` | `(client, ...) -> str` | Handshake + session creation; returns session ID |
| `_start_turn` | `(client, session_id, text, ...) -> Any` | Start a turn; returns turn state |
| `_send_interrupt` | `(client, session_id, turn_state) -> None` | Send interrupt for deadline |
| `_process_notification` | `(message, ...) -> (kind, value) \| None` | Process a notification |
| `_handle_server_request` | `(client, message, tool_lookup, ...) -> None` | Handle server request |
| `_build_tool_specs` | `(bridged_tools) -> list[dict]` | Convert tools to provider format |
| `_build_output_schema` | `(rendered) -> dict \| None` | Build output schema |

### Optional Hooks (Default Behavior)

| Hook | Default | Purpose |
|------|---------|---------|
| `_setup_environment` | `(None, config_env)` | Setup environment (e.g. ephemeral home) |
| `_cleanup_environment` | no-op | Cleanup environment after evaluation |
| `_create_transcript_bridge` | `None` | Create transcript bridge |
| `_on_notification_for_transcript` | no-op | Forward notification to bridge |
| `_on_tool_call_for_transcript` | no-op | Forward tool call to bridge |
| `_on_tool_result_for_transcript` | no-op | Forward tool result to bridge |

### Notification Result Types

`_process_notification()` returns `(kind, value)` tuples:

| Kind | Meaning | How Applied |
|------|---------|-------------|
| `"delta"` | Append text to accumulator | `accumulated += value` |
| `"text"` | Replace accumulated text | `accumulated = value` |
| `"usage"` | Token usage update | Extract from message params |
| `"done"` | Turn complete | Break loop |
| `"error"` | Turn failed | Raise `PromptEvaluationError` |
| `"interrupted"` | Turn interrupted | Raise `PromptEvaluationError` |

## Generic Protocol Flow

Defined at `src/weakincentives/adapters/jsonrpc/_protocol.py`.

### execute_protocol()

Orchestrates the full lifecycle:

1. `client.start()`
1. Create transcript bridge (optional)
1. `_initialize_session()` → session ID
1. **Turn continuation loop** (max 10 rounds):
   a. `_start_turn()` → turn state
   b. Create deadline watchdog (calls `_send_interrupt()` on timeout)
   c. **Message consumption loop**:
   - Server requests → `_handle_server_request()`
   - Notifications → `_process_notification()` → apply result
   - Break on turn complete or visibility signal
     d. Accumulate text and usage
     e. `check_task_completion()` → continue or break
1. Check visibility signal
1. Return `(accumulated_text, usage)`

### Generic Tool Call Handler

The base adapter provides `handle_tool_call()` which:

1. Extracts tool name and arguments from params
1. Looks up `BridgedTool` by name
1. Executes via `asyncio.to_thread(bridged_tool, arguments)`
1. Calls `_format_tool_response(success, content_items)` (hook)
1. Appends feedback if applicable
1. Sends response via `client.send_response()`

Subclasses provide:

- `_extract_tool_call_info(params)` — extract name and arguments
- `_format_tool_response(success, content_items)` — format response dict

### Token Usage Extraction

Subclasses implement `_extract_token_usage(params)` to map provider-specific
usage fields to WINK's `TokenUsage`.

## Response Building

Defined at `src/weakincentives/adapters/jsonrpc/_response.py`.

Shared between all JSON-RPC subclasses:

1. Parse structured output via `parse_structured_output(text, rendered)`
1. Record budget if tracker provided
1. Build `PromptResponse(prompt_name, text, output)`
1. Dispatch `PromptExecuted` event
1. Log completion with duration and token info

## Error Handling

### Error Phases

| Phase | When |
|-------|------|
| `"request"` | Transport connect, handshake, session/turn start fails |
| `"response"` | Structured output missing or invalid; turn failed |
| `"tool"` | Bridged tool execution failure |
| `"budget"` | Token/cost budget exceeded |

Provider-specific error mapping is implemented in subclass hooks.

## Codex App Server Subclass

After this refactoring, `CodexAppServerAdapter` extends `JsonRpcAdapter` and
implements the following hooks:

| Hook | Codex Implementation |
|------|---------------------|
| `_adapter_name()` | `"codex_app_server"` |
| `_create_client()` | `JsonRpcClient(bin_path="codex", bin_args=("app-server",), ...)` |
| `_initialize_session()` | `initialize` + `initialized` + `authenticate` + `thread/start` |
| `_start_turn()` | `turn/start` with effort, summary, personality, outputSchema |
| `_send_interrupt()` | `turn/interrupt` with threadId and turnId |
| `_process_notification()` | Route `item/agentMessage/delta`, `item/completed`, etc. |
| `_handle_server_request()` | Route `item/tool/call`, approval requests |
| `_build_tool_specs()` | `bridged_tools_to_dynamic_specs()` |
| `_build_output_schema()` | `build_output_schema()` with `openai_strict_schema()` |
| `_setup_environment()` | `CodexEphemeralHome` for skill installation |
| `_extract_token_usage()` | Map `tokenUsage.last` to WINK `TokenUsage` |

## Design Decisions

### Why Extract from Codex (Not Build from Scratch)

The Codex App Server adapter is the most mature JSON-RPC adapter with
comprehensive test coverage. Extracting the generic layer from working code
ensures correctness and preserves test coverage. The refactoring follows the
same evolutionary pattern as `ACPAdapter` (extracted from OpenCode-specific
code to support multiple ACP agents).

### Why Turn-Based Protocol Abstraction

The turn-based lifecycle (init → session → turn → stream → complete) is the
natural abstraction for agentic JSON-RPC protocols. It's specific enough to
provide real value (vs. raw JSON-RPC) but general enough to accommodate
different provider semantics (different method names, payload shapes, auth
flows).

### Why Dynamic Tools (Not MCP)

JSON-RPC providers that support bidirectional requests can execute WINK tools
directly — the server sends a request, the client executes the tool in-process,
and responds on the same channel. This avoids the MCP HTTP server overhead
required by ACP. The `_format_tool_response()` hook allows each provider to
define its own response format.

### Why Parameterized Client (Not Subclass)

`JsonRpcClient` is parameterized via constructor arguments (bin_path, bin_args,
etc.) rather than requiring subclassing. The transport layer is truly generic —
the only provider-specific aspect is which binary to spawn and with what
arguments. Parameterization is simpler and avoids unnecessary class hierarchy.

## Testing

Tests at `tests/adapters/jsonrpc/`:

| File | Coverage |
|------|----------|
| `test_client.py` | JsonRpcClient transport tests |
| `test_client_io.py` | Message I/O tests |
| `test_client_ws.py` | WebSocket transport tests |
| `test_protocol.py` | Generic protocol flow tests |
| `test_response.py` | Response building tests |
| `test_types.py` | Type definition tests |

Codex-specific tests remain at `tests/adapters/codex_app_server/` and test
the hook implementations.

## Related Specifications

- `specs/ADAPTERS.md` — Provider adapter protocol
- `specs/CODEX_APP_SERVER.md` — Codex App Server subclass
- `specs/ACP_ADAPTER.md` — Analogous generic adapter for ACP
- `specs/TOOLS.md` — Tool registration and policies
- `specs/SESSIONS.md` — Session state and events

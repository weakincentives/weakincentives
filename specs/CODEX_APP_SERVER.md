# Codex App Server Adapter Specification

> **Adapter name:** `codex_app_server`
> **Codex entrypoint:** `codex app-server`
> **Protocol:** JSON-RPC (without `"jsonrpc":"2.0"` header)
> **Transports:** stdio (NDJSON) or WebSocket (one message per text frame)
> **Validated against:** `codex-cli 0.118.0` with ChatGPT auth

## Purpose

`CodexAppServerAdapter` evaluates WINK prompts by delegating execution to
**Codex** via its **app-server** (the same interface powering the Codex VS Code
extension). The architecture mirrors the `ClaudeAgentSDKAdapter`:

| Responsibility | Owner |
|----------------|-------|
| Prompt composition, resource binding, session telemetry | WINK |
| Agentic execution (planning, reasoning, tool calls, file edits, commands) | Codex |

WINK receives streamed progress via app-server `item/*` and `turn/*`
notifications and emits canonical events: `PromptRendered`, `ToolInvoked`,
`PromptExecuted`.

**Implementation:** `src/weakincentives/adapters/codex_app_server/`

## Why the App Server

The Codex App Server is an **agentic harness** — it provides planning loops,
tool orchestration, sandboxing, approval flows, and crash recovery. This
qualifies it as an execution harness under WINK's design philosophy (see
`specs/ADAPTERS.md`). The app-server protocol exposes the full Codex agent
lifecycle, making it suitable for deep product integrations while keeping the
agent definition (prompts, tools, policies) portable.

Key Codex capabilities surfaced through the app-server:

- **Threads and turns:** Persistent conversation state with fork/resume
- **Native tools:** Command execution, file changes, web search, image viewing
- **Custom tools:** Dynamic tools (lightweight, in-process) and external MCP
  servers (subprocess or URL)
- **Sandboxing:** Configurable sandbox policies (read-only, workspace-write,
  full access, external sandbox)
- **Approval flows:** Command and file change approvals
- **Structured output:** Native `outputSchema` on `turn/start`

## Requirements

### Runtime Dependencies

1. **Codex CLI** installed and available on `PATH` (stdio mode), **or** a
   reachable Codex App Server instance (WebSocket mode)
1. WINK (`weakincentives`) runtime
1. `weakincentives[codex-ws]` extra (WebSocket mode only — provides `websockets`)

No additional Python dependencies beyond WINK for stdio mode. The adapter
reuses `BridgedTool` and `create_bridged_tools()` from the shared adapter
module at `src/weakincentives/adapters/_shared/_bridge.py` — it does not
require the `claude-agent-sdk` package at runtime.

## Transport Modes

The adapter supports two transport modes that carry identical JSON-RPC
messages. The transport determines **how** messages are framed and **who**
manages the Codex process lifecycle.

### stdio (default)

The adapter spawns `codex app-server` as a child process and communicates via
newline-delimited JSON (NDJSON) over stdin/stdout pipes. This is the original
transport and the simplest way to run Codex locally.

| Aspect | Details |
|--------|---------|
| **Framing** | One JSON object per `\n`-delimited line |
| **Process lifecycle** | Adapter spawns and terminates the subprocess |
| **Clients per process** | One (1:1 mapping) |
| **Auth** | Inherits host-level credentials from `~/.codex/` |
| **Stderr** | Captured for diagnostics |
| **Backpressure** | None |

### Managed WebSocket

The adapter spawns `codex app-server --listen ws://127.0.0.1:PORT` as a child
process, waits for it to start listening, then connects via WebSocket. This
gives the benefits of WebSocket framing (text frames, no newline parsing)
while retaining local process management.

| Aspect | Details |
|--------|---------|
| **Framing** | One JSON object per WebSocket text frame |
| **Process lifecycle** | Adapter spawns and terminates the subprocess |
| **Clients per process** | One (adapter is the sole client) |
| **Auth** | Inherits host-level credentials from `~/.codex/` |
| **Stderr** | Captured for diagnostics |
| **Backpressure** | `-32001` error code when server request queue is full |

### External WebSocket

The adapter connects to a Codex App Server that is already running and
listening on a `ws://` or `wss://` endpoint. **The adapter does not spawn or
manage the server process.** This mode is designed for:

- **Remote execution:** Connecting to a Codex App Server running on a different
  machine over the network.
- **Shared servers:** Multiple WINK adapter instances (or other clients) can
  connect to the same server concurrently. Each WebSocket connection gets its
  own independent session.
- **Long-lived servers:** The server lifecycle is decoupled from any single
  client. The server continues running when clients disconnect.
- **Managed infrastructure:** The Codex App Server may be deployed and
  supervised by external infrastructure (systemd, Kubernetes, etc.). WINK does
  not need to know how to start or stop it.

| Aspect | Details |
|--------|---------|
| **Framing** | One JSON object per WebSocket text frame |
| **Process lifecycle** | **Not managed by the adapter** — server is external |
| **Clients per server** | Multiple concurrent connections, each independent |
| **Auth** | `Authorization: Bearer TOKEN` header during WebSocket upgrade |
| **Stderr** | Not available (no subprocess) |
| **Backpressure** | `-32001` error code when server request queue is full |

**Binary frames** are silently ignored. **Invalid JSON** is silently ignored
and the connection survives.

### Protocol Equivalence

All three transport modes carry identical JSON-RPC messages. All protocol-level
code works unchanged across transports:

- `_protocol.py` — initialize, authenticate, thread, turn, stream
- `_events.py` — notification → event mapping
- `_response.py` — response building, structured output
- `_guardrails.py` — feedback, task completion
- `_transcript.py` — transcript bridge
- `_schema.py` — schema transforms

The only code that differs is the transport layer in `client.py`.

## Architecture

```
WINK Prompt/Session
  └─ CodexAppServerAdapter.evaluate()
      ├─ Render WINK prompt → markdown text
      ├─ create_bridged_tools() → BridgedTool list
      ├─ Convert to DynamicToolSpec list [{name, description, inputSchema}]
      ├─ Connect transport:
      │    ├─ stdio: Spawn codex app-server, pipe stdin/stdout
      │    └─ websocket: Connect to ws://host:port
      ├─ Handshake: initialize (experimentalApi) → initialized
      ├─ thread/start (model, cwd, sandbox, dynamicTools)
      ├─ turn/start (text input, outputSchema if structured)
      ├─ Stream: item/*, turn/* notifications
      │    ├─ item/agentMessage/delta (params.delta — assistant output)
      │    ├─ item/started + item/completed (commands, file changes, MCP tools)
      │    ├─ item/tool/call → execute BridgedTool in-process → respond
      │    ├─ item/reasoning/* (reasoning summaries)
      │    ├─ thread/tokenUsage/updated (token tracking)
      │    └─ turn/completed (final status)
      ├─ Parse JSON output if outputSchema was provided
      └─ Return PromptResponse(text, output)
```

## Module Structure

```
src/weakincentives/adapters/codex_app_server/
  __init__.py
  adapter.py                # CodexAppServerAdapter (high-level orchestration)
  config.py                 # CodexAppServerClientConfig, CodexAppServerModelConfig
  client.py                 # CodexAppServerClient (transport-aware JSON-RPC client)
  _schema.py                # Schema transforms (DynamicToolSpec, OpenAI strict schema)
  _protocol.py              # JSON-RPC protocol orchestration (init, auth, thread, turn, stream)
  _response.py              # Response building and structured output parsing
  _events.py                # Codex item/turn notifications → WINK ToolInvoked mapping
  _transcript.py            # Transcript bridging for Codex notifications
  _async.py                 # asyncio helpers
  _guardrails.py            # Feedback providers, task completion checking
  _ephemeral_home.py        # CodexEphemeralHome (skill installation)
```

Tool bridging reuses `BridgedTool` and `create_bridged_tools()` from
`src/weakincentives/adapters/_shared/_bridge.py`.

## Configuration

### CodexAppServerClientConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `transport` | `Transport` | `"stdio"` | Wire protocol: `"stdio"` or `"websocket"` |
| `codex_bin` | `str` | `"codex"` | Executable to spawn (managed modes only) |
| `remote_url` | `str \| None` | `None` | WebSocket URL of an external server (e.g. `ws://10.0.1.5:4500`). When set, no subprocess is spawned |
| `ws_auth_token` | `str \| None` | `None` | Bearer token for WebSocket auth. Sent as `Authorization: Bearer TOKEN` during upgrade |
| `cwd` | `str \| None` | `None` | Working directory (must be absolute; defaults to `Path.cwd().resolve()`) |
| `env` | `Mapping[str, str] \| None` | `None` | Extra environment variables for the subprocess (managed modes only) |
| `suppress_stderr` | `bool` | `True` | Capture stderr for debugging (stdio mode only) |
| `startup_timeout_s` | `float` | `10.0` | Max time for initialize handshake |
| `approval_policy` | `ApprovalPolicy` | `"never"` | How to handle command/file approvals |
| `sandbox_mode` | `SandboxMode \| None` | `None` | Sandbox mode for `thread/start` |
| `auth_mode` | `CodexAuthMode \| None` | `None` | Authentication configuration. None inherits host credentials |
| `mcp_servers` | `dict[str, McpServerConfig] \| None` | `None` | Additional external MCP servers |
| `ephemeral` | `bool` | `False` | If true, thread is not persisted to disk |
| `client_name` | `str` | `"wink"` | Client identifier for `initialize` |
| `client_version` | `str` | `"0.1.0"` | Client version for `initialize` |
| `transcript` | `bool` | `True` | Emit transcript entries during evaluation |
| `transcript_emit_raw` | `bool` | `True` | Include raw notification JSON in `raw` field |

**Transport selection:**

- `transport="stdio"` (default), `remote_url=None`: spawn subprocess, NDJSON
  over stdin/stdout pipes.
- `transport="websocket"`, `remote_url=None`: spawn subprocess with
  `--listen ws://127.0.0.1:PORT`, connect via WebSocket. The subprocess is
  managed by the adapter.
- `remote_url` set (any `transport` value): connect to an external server via
  WebSocket. **No subprocess is spawned or managed.** `codex_bin`, `env`, and
  `suppress_stderr` are ignored.

> **CWD requirement:** `thread/start` requires `cwd` to be an absolute path. If
> `None`, the adapter resolves to `Path.cwd().resolve()`. When connecting to a
> remote server, `cwd` refers to a path on the **remote** machine — the caller
> is responsible for ensuring it exists and is accessible.

> **Approval handling:** `approval_policy="never"` means the adapter auto-accepts
> all approvals. For non-interactive WINK execution, `"never"` is the default
> since there is no human to prompt.

> **Tool namespace:** WINK bridged tools are registered as dynamic tools.
> User-provided `mcp_servers` are passed to Codex via `config.mcp_servers` on
> `thread/start`. External MCP tool names must not collide with WINK tool names.

#### Transport-Specific Fields

| Field | stdio | managed WS | external WS | Notes |
|-------|-------|------------|-------------|-------|
| `codex_bin` | Used | Used | Ignored | No subprocess in external mode |
| `env` | Used | Used | Ignored | No subprocess env in external mode |
| `suppress_stderr` | Used | Ignored | Ignored | Only stdio uses piped stderr |
| `remote_url` | Ignored | Ignored | Required | URL of external server |
| `ws_auth_token` | Ignored | Optional | Optional | Bearer token for WS upgrade |

#### ApprovalPolicy

```python
ApprovalPolicy = Literal["never", "untrusted", "on-failure", "on-request"]
```

| Value | Behavior |
|-------|----------|
| `"never"` | Auto-accept all approvals (no human gating) |
| `"untrusted"` | Approval required for non-trusted commands |
| `"on-failure"` | Approval required after command failure |
| `"on-request"` | Approval required on every action |

#### SandboxMode

```python
SandboxMode = Literal["read-only", "workspace-write", "danger-full-access"]
```

Sent as a string on `thread/start` via the `sandbox` field. The response
returns the object form (e.g. `{"type": "dangerFullAccess"}`).

Codex also supports a `sandboxPolicy` override on `turn/start` with richer
options (`writableRoots`, `networkAccess`, `excludeSlashTmp`,
`excludeTmpdirEnvVar`), but the adapter does not expose this in v1 — the
thread-level `SandboxMode` string is sufficient.

#### CodexAuthMode

`CodexAuthMode = ApiKeyAuth | ExternalTokenAuth`. `ApiKeyAuth(api_key: str)` uses
an API key; `ExternalTokenAuth(id_token: str, access_token: str)` uses ChatGPT
OAuth tokens. At `src/weakincentives/adapters/codex_app_server/config.py`.

Authentication is performed after `initialize` via `account/login/start`. When
`auth_mode` is `None`, the adapter skips authentication and assumes the Codex
environment is already authenticated. In stdio mode this means the CLI inherits
host-level credentials from `~/.codex/`. In WebSocket mode the remote server
must be pre-authenticated or the caller must provide `auth_mode`.

#### WebSocket Authentication

Two layers of authentication exist in WebSocket mode:

1. **Transport-level auth** (`ws_auth_token`): A bearer token sent during the
   WebSocket upgrade handshake via the `Authorization: Bearer TOKEN` header.
   This authenticates the client to the Codex App Server process itself. The
   server must be started with `--ws-auth capability-token --ws-token-file PATH`
   for this to be enforced. On loopback connections, transport auth is optional.

1. **Session-level auth** (`auth_mode`): The `account/login/start` request
   sent after `initialize`. This authenticates the session with the upstream
   model provider (OpenAI). Same mechanism as stdio mode.

These are independent — a server may require transport auth without session
auth (if pre-authenticated), or session auth without transport auth (loopback).

### CodexAppServerModelConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `"gpt-5.4"` | Codex model identifier |
| `effort` | `ReasoningEffort \| None` | `None` | Reasoning effort |
| `summary` | `ReasoningSummary \| None` | `None` | Summary preference |
| `personality` | `Personality \| None` | `None` | Response personality |

```python
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
ReasoningSummary = Literal["auto", "concise", "detailed", "none"]
Personality = Literal["none", "friendly", "pragmatic"]
```

**Note:** `seed`, `stop`, `presence_penalty`, `frequency_penalty` are not
supported by the Codex app-server and are not fields on this config.

## Transport Layer

### CodexAppServerClient

The client manages transport connectivity and provides a typed interface for
the JSON-RPC protocol. It operates in one of two modes depending on
configuration.

At `src/weakincentives/adapters/codex_app_server/client.py`:

**Public API** (identical across both transports):

- `start()` — Establish the transport (spawn subprocess or connect WebSocket)
- `stop()` — Tear down the transport (terminate subprocess or close WebSocket)
- `send_request(method, params, timeout)` — Send request and await response by
  matching `id`; returns the `result` field
- `send_notification(method, params=None)` — Send notification (no `id`, no
  response expected); `params` is optional
- `send_response(request_id, result)` — Send response to a server-initiated
  request
- `read_messages()` — Async iterator yielding notifications and server requests
  (responses are consumed internally by `send_request`)
- `stderr_output` — Captured stderr output (stdio mode only; empty in WS mode)

### stdio Transport Internals

Each message is a single JSON object terminated by `\n`. The client assigns
incrementing integer `id` fields to requests and correlates responses by `id`.

- `_write(msg)` — `json.dumps(msg, separators=(",",":"))` + `\n` to stdin
- `_read_loop()` — Reads lines from stdout, parses JSON, calls `_route_message()`
- `_stderr_loop()` — Buffers up to 1000 stderr lines for diagnostics

### WebSocket Transport Internals

Each message is a single JSON object sent as one WebSocket text frame. No
newline delimiter. The read/write interface matches stdio but uses
`ws.send(frame)` / `ws.recv()` instead of pipe I/O.

- `_write(msg)` — `json.dumps(msg, separators=(",",":"))` sent as WS text frame
- `_read_loop()` — Receives WS text frames, parses JSON, calls `_route_message()`
- No stderr capture (no subprocess)

**Connection headers:** When `ws_auth_token` is configured, the client passes
`Authorization: Bearer TOKEN` as an additional header during the WebSocket
upgrade.

### Message Routing

The client demultiplexes inbound messages into three streams (same logic for
both transports):

1. **Responses** — messages with an `id` field matching a pending request
1. **Notifications** — messages with a `method` field and no `id`
1. **Server requests** — messages with both `method` and `id` (approval
   requests and dynamic tool calls); the client must respond with a matching `id`

Server-initiated requests require the client to respond promptly. Dynamic tool
calls (`item/tool/call`) and approval requests both follow this pattern.

**Wire format note:** The Codex protocol omits the `"jsonrpc": "2.0"` header —
do not include it.

## Protocol Mapping

### WINK Concepts → Codex App Server Concepts

| WINK Concept | Codex App Server Concept | Adapter Role |
|--------------|--------------------------|--------------|
| **Prompt** (PromptTemplate + sections + tools) | Thread + Turn input text | Render, format, send via `turn/start` |
| **Session** (event-sourced state) | Thread (persistent conversation) | Map thread/turn lifecycle to session events |
| **Tool** (Tool[ParamsT, ResultT]) | Dynamic tool via `item/tool/call` | Bridge via `create_bridged_tools()` + `DynamicToolSpec` |
| **Tool Execution** (transactional) | Item (`commandExecution`, `fileChange`, `mcpToolCall`) | Map `item/completed` → `ToolInvoked` |
| **Output** (structured dataclass) | Native `outputSchema` on `turn/start` | Parse JSON from delta text, deserialize |
| **Events** (PromptRendered, ToolInvoked, PromptExecuted) | `item/*`, `turn/*` notifications | Translate and dispatch |
| **Deadline** | Turn interrupt via `turn/interrupt` | Enforce with timer + interrupt |
| **Budget** | Per-model token tracking | Record usage from `thread/tokenUsage/updated` |

### Codex Item Types → WINK Events

At `src/weakincentives/adapters/codex_app_server/_events.py`:
`dispatch_item_tool_invoked()` maps completed Codex items to `ToolInvoked`:

| Codex Item Type | WINK Event | Tool Name |
|-----------------|------------|-----------|
| `commandExecution` (completed) | `ToolInvoked` | `codex:command` |
| `fileChange` (completed) | `ToolInvoked` | `codex:file_change` |
| `mcpToolCall` (completed) | `ToolInvoked` | `codex:mcp:{tool}` |
| `webSearch` (completed) | `ToolInvoked` | `codex:web_search` |
| `item/tool/call` (server request) | `ToolInvoked` | (via BridgedTool) |
| `agentMessage` | Text accumulation | Concatenated for `PromptResponse.text` |
| `reasoning` | (informational) | Logged if configured |
| `contextCompaction` | (informational) | Logged |

### Dual Notification System

Codex emits notifications in two parallel namespaces:

- **`item/*`, `turn/*`, `thread/*`** — standardized v2 protocol (use these)
- **`codex/event/*`** — legacy v1 events (ignore; same content, different shape)

The adapter should only process v2 notifications.

## Structured Output

When `rendered.output_type is not None`, the adapter uses Codex's **native
`outputSchema`** parameter on `turn/start`. This constrains the model's final
message to valid JSON conforming to the schema.

### Schema Generation

At `src/weakincentives/adapters/codex_app_server/_schema.py`:
`build_output_schema()` generates the JSON schema via `serde.schema()` and
then applies `openai_strict_schema()` to make it compatible with OpenAI/Codex
structured output requirements: adds `additionalProperties: false` on all
object types and lists all properties in `required`. For `container="array"`
prompts, wraps in an `{"items": [...]}` envelope.

### Passing the Schema

The schema is passed as the `outputSchema` field on `turn/start`.

### Retrieval

After `turn/completed`:

1. Parse the accumulated `agentMessage` delta text
1. Deserialize via `parse_structured_output(text, rendered)` from
   `weakincentives.prompt.structured_output`
1. If parsing fails: raise `PromptEvaluationError(phase="response")`

No MCP tool is needed — the model produces valid JSON directly in its response
text when `outputSchema` is provided.

## Tool Bridging via Dynamic Tools

### Why Dynamic Tools

Dynamic tools are the simplest mechanism for exposing WINK tools to Codex.
When the model calls a dynamic tool, Codex sends an `item/tool/call` server
request **back over the same transport channel** to the adapter process. The
adapter executes the `BridgedTool` in-process with full access to session
state and resources, then responds. No subprocess, no HTTP server, no extra
dependencies.

The entire integration is:

1. Convert `BridgedTool` list to `DynamicToolSpec` list (3-line function)
1. Pass `dynamicTools` on `thread/start` (requires `experimentalApi` on
   `initialize`)
1. Handle `item/tool/call` in the message loop (same pattern as approval
   handling)

### Reused Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `BridgedTool` | `adapters/_shared/_bridge.py` | Transactional tool wrapper |
| `create_bridged_tools()` | `adapters/_shared/_bridge.py` | Factory for BridgedTool |
| `VisibilityExpansionSignal` | `adapters/_shared/_visibility_signal.py` | Exception propagation |
| `tool_transaction()` | `runtime/transactions.py` | Snapshot/restore |

> **Important:** Pass `adapter_name="codex_app_server"` to
> `create_bridged_tools()` to ensure `ToolInvoked` events are labeled correctly.

### DynamicToolSpec Conversion

At `src/weakincentives/adapters/codex_app_server/_schema.py`:
`bridged_tools_to_dynamic_specs()` converts each `BridgedTool` to a dict
with `name`, `description`, and `inputSchema` keys.

### Tool Bridging Flow

```
1. Render prompt → rendered.tools
2. create_bridged_tools(..., adapter_name="codex_app_server")
3. bridged_tools_to_dynamic_specs(bridged_tools)
4. thread/start with dynamicTools=[...] (requires experimentalApi)
5. Model calls tool → Codex sends item/tool/call server request
6. Adapter executes BridgedTool.__call__()
   ├─ Snapshot session state
   ├─ Execute handler
   ├─ Dispatch ToolInvoked
   └─ Rollback on failure
7. Adapter responds with DynamicToolCallResponse
```

### item/tool/call Handling

At `src/weakincentives/adapters/codex_app_server/_protocol.py`:
`handle_tool_call()` processes `item/tool/call` server requests:

1. Extract `tool` name and `arguments` from `params` (handles string arguments
   via `json.loads`)
1. Look up `BridgedTool` by name; respond with error if unknown
1. Execute via `asyncio.to_thread(bridged_tool, arguments)`
1. Convert MCP result to `DynamicToolCallResponse` format:
   `{"success": bool, "contentItems": [{"type": "inputText", "text": str}]}`

### External MCP Servers

User-provided MCP servers (not WINK tools) are passed to Codex via
`config.mcp_servers` on `thread/start` as stdio or HTTP entries.

Tool calls to external MCP servers appear as `mcpToolCall` items (not
`item/tool/call`) and are mapped to `ToolInvoked` events.

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
1. After `turn/completed`, adapter checks signal
1. If set, re-raises to caller for re-render

## Workspace Management

### WorkspaceSection

At `src/weakincentives/prompt/workspace.py` (exported from `weakincentives.prompt`):

- Accepts `HostMount` tuples, `allowed_host_roots`, max-bytes budgets
- Materializes temporary directory with copied files (with glob filtering,
  symlink safety, and byte budget enforcement)
- Exposes `temp_dir` for `CodexAppServerClientConfig.cwd`
- Renders a provider-agnostic summary of mounts and budgets
- Exposes cleanup via `.cleanup()` with reference counting for cloned sections
- Provides `workspace_fingerprint` for session reuse validation
- Binds a `HostFilesystem` resource scoped to the temp directory

> **Remote servers:** When using WebSocket mode to connect to a remote Codex
> App Server, `cwd` refers to a directory on the remote machine. Workspace
> materialization (temp dirs, file copies) happens locally and the resulting
> path is sent to the remote server. The caller must ensure the remote server
> has access to the specified path — typically by using a shared filesystem,
> pre-staging files, or specifying a path that already exists on the remote.

## Execution Flow

### 1. Budget/Deadline Setup

- Create `BudgetTracker` if budget provided
- Derive deadline from argument or `budget.deadline`
- Raise `PromptEvaluationError(phase="request")` if already expired

### 2. Render Prompt

1. `prompt.render(session=session)` → `RenderedPrompt` (text + tools + output_type)
1. Resolve CWD and bind `HostFilesystem` resource if prompt has no filesystem
1. Emit `PromptRendered`

### 3. Build Dynamic Tool Specs

1. `create_bridged_tools(rendered.tools, adapter_name="codex_app_server", ...)`
1. `bridged_tools_to_dynamic_specs(bridged_tools)` → `DynamicToolSpec` list

### 4. Establish Transport

**stdio mode** (`transport="stdio"`, `remote_url=None`):

`CodexAppServerClient.start()` spawns the subprocess via
`asyncio.create_subprocess_exec(codex_bin, "app-server", ...)` with stdin,
stdout, and stderr pipes. A background read loop and stderr capture loop are
started as asyncio tasks.

**Managed WebSocket mode** (`transport="websocket"`, `remote_url=None`):

`CodexAppServerClient.start()` spawns the subprocess via
`asyncio.create_subprocess_exec(codex_bin, "app-server", "--listen", "ws://127.0.0.1:PORT", ...)`, waits for it to accept TCP connections, then
connects via `websockets.connect(...)`. A background read loop and stderr
capture loop are started as asyncio tasks.

**External WebSocket mode** (`remote_url` set):

`CodexAppServerClient.start()` connects to the specified WebSocket URL via
`websockets.connect(remote_url, ...)`. If `ws_auth_token` is configured, the
`Authorization: Bearer TOKEN` header is included in the upgrade request. A
background read loop is started as an asyncio task. No subprocess is spawned.
No stderr capture is available.

### 5. Initialize Handshake

At `src/weakincentives/adapters/codex_app_server/_protocol.py`:
`execute_protocol()` sends the `initialize` request with
`capabilities: {experimentalApi: true}` (enables dynamic tools on
`thread/start`), then sends an `initialized` notification. The
`startup_timeout_s` config controls the handshake timeout.

The server rejects all methods before `initialize`. Repeated `initialize` calls
return `Already initialized`.

### 6. Authenticate (Optional)

At `src/weakincentives/adapters/codex_app_server/_protocol.py`:
`authenticate()` sends `account/login/start` if `auth_mode` is configured:

- `ApiKeyAuth` — `{"type": "apiKey", "apiKey": ...}`
- `ExternalTokenAuth` — `{"type": "chatgptAuthTokens", "idToken": ..., "accessToken": ...}`

The login is synchronous — `account/login/start` returns its result directly.
On error, raise `PromptEvaluationError(phase="request")`.

### 7. Start Thread

At `src/weakincentives/adapters/codex_app_server/_protocol.py`:
`create_thread()` sends `thread/start` with `model`, `cwd`,
`approvalPolicy`, and `ephemeral`. Optional fields `sandbox`,
`dynamicTools`, and `config.mcp_servers` are included only when configured.
Returns `result["thread"]["id"]`.

### 8. Start Turn

At `src/weakincentives/adapters/codex_app_server/_protocol.py`:
`start_turn()` sends `turn/start` with `threadId` and `input` (as
`[{"type": "text", "text": prompt_text}]`). Optional fields `effort`,
`summary`, `personality`, and `outputSchema` are included only when set.
Returns `result["turn"]["id"]`.

### 9. Stream Notifications

At `src/weakincentives/adapters/codex_app_server/_protocol.py`:
`stream_turn()` reads messages via `client.read_messages()` until
`turn/completed`. A deadline watchdog task sends `turn/interrupt` if the
deadline expires.

**Server requests** (both `id` and `method`) are dispatched by
`handle_server_request()`:

- `item/tool/call` — execute `BridgedTool` in-process via `handle_tool_call()`
- `item/commandExecution/requestApproval` / `item/fileChange/requestApproval` —
  auto-respond per approval policy
- Unknown methods — respond with empty result

**Notifications** (method only) are processed by `process_notification()`:

- `item/agentMessage/delta` — accumulate assistant text
- `item/completed` — dispatch `ToolInvoked` for `commandExecution`,
  `fileChange`, `mcpToolCall`, `webSearch`; capture final `agentMessage` text
- `thread/tokenUsage/updated` — extract `TokenUsage`
- `turn/completed` — check status for errors/interruption, signal done

**Note:** WINK bridged tools arrive as `item/tool/call` server requests (same
as approval requests). External MCP tools arrive as `mcpToolCall` notification
items. The two paths are distinct — no deduplication needed.

### 10. Handle Approvals

At `src/weakincentives/adapters/codex_app_server/_protocol.py`:
`handle_server_request()` auto-responds to approval requests:

| Policy | Decision |
|--------|----------|
| `"never"` | `"accept"` |
| `"on-failure"` | `"accept"` |
| `"untrusted"` | `"decline"` |
| `"on-request"` | `"decline"` |

For non-interactive WINK execution, `"never"` and `"on-failure"` accept all
approvals. `"untrusted"` and `"on-request"` decline, since there is no human
to prompt for approval.

### 11. Extract Results

**Text:** Use the final `agentMessage` item from `item/completed`, or the
accumulated delta text.

**Tool events:** Map `item/completed` notifications to `ToolInvoked`:

| Codex Item Status | Action |
|-------------------|--------|
| `completed` | `ToolInvoked` with `success=True` |
| `failed` | `ToolInvoked` with `success=False` |
| `declined` | `ToolInvoked` with `success=False` |

### 12. Structured Output

If `outputSchema` was set on `turn/start`, the accumulated delta text contains
valid JSON conforming to the schema. Parse and deserialize via
`parse_structured_output(text, rendered)` from
`weakincentives.prompt.structured_output`.
Raise `PromptEvaluationError(phase="response")` if parsing fails.

### 13. PromptExecuted

Emit event and return `PromptResponse(text=..., output=...)`.

## Cancellation

If deadline expires during a turn:

1. Send `turn/interrupt` — `{"threadId": thread_id, "turnId": turn_id}`
1. Wait for `turn/completed` with `status: "interrupted"` (bounded wait)
1. Tear down transport (kill subprocess or close WebSocket)
1. Raise `PromptEvaluationError(phase="request")` or `DeadlineExceededError`

## Error Handling

### Error Phases

| Phase | When |
|-------|------|
| `"request"` | Transport connect, initialize, auth, thread/start, or turn/start fails |
| `"response"` | Structured output missing or invalid; turn completes with `status: "failed"` |
| `"tool"` | Bridged tool execution failure |
| `"budget"` | Token budget exceeded |

### WebSocket-Specific Errors

| Condition | Handling |
|-----------|----------|
| Connection refused | `PromptEvaluationError(phase="request")` — server not reachable |
| HTTP 401 on upgrade | `PromptEvaluationError(phase="request")` — invalid or missing `ws_auth_token` |
| Connection closed (code 1006) | `PromptEvaluationError(phase="request")` — server terminated |
| `-32001` overload error | Retry with backoff, then `PromptEvaluationError(phase="request")` |
| WebSocket frame error | Log and continue — connection survives |

### Turn Failure Mapping

At `src/weakincentives/adapters/codex_app_server/_events.py`:
`map_codex_error_phase()` maps `codexErrorInfo` to WINK error phases.
When `turn/completed` has `status: "failed"`:

| Codex Error | WINK Action |
|-------------|-------------|
| `contextWindowExceeded` | `PromptEvaluationError(phase="response")` |
| `usageLimitExceeded` | `PromptEvaluationError(phase="budget")` |
| `httpConnectionFailed` | `PromptEvaluationError(phase="request")` |
| `unauthorized` | `PromptEvaluationError(phase="request")` |
| `badRequest` | `PromptEvaluationError(phase="request")` |
| `sandboxError` | `PromptEvaluationError(phase="tool")` |
| `responseTooManyFailedAttempts` | `PromptEvaluationError(phase="request")` |
| `responseStreamConnectionFailed` | `PromptEvaluationError(phase="request")` |
| `responseStreamDisconnected` | `PromptEvaluationError(phase="request")` |
| `threadRollbackFailed` | `PromptEvaluationError(phase="response")` |
| `internalServerError` | `PromptEvaluationError(phase="response")` |
| `modelCap` (object with `model`, `reset_after_seconds`) | `PromptEvaluationError(phase="budget")` |
| `other` / unknown | `PromptEvaluationError(phase="response")` |

Include in payload: stderr tail (stdio mode only, bounded to last 8k), Codex
error details, `codexErrorInfo`, and `additionalDetails`.

Tool telemetry errors: log but don't crash.

## Skill Installation

`CodexEphemeralHome` at `src/weakincentives/adapters/codex_app_server/_ephemeral_home.py`
manages a temporary HOME directory for Codex skill discovery. Skills from
`RenderedPrompt.skills` are mounted at `$HOME/.agents/skills/<name>/`, which is
the path Codex scans for user-scoped skills.

The ephemeral home provides environment overrides:

- `HOME` — points to the ephemeral directory (skill discovery)
- `CODEX_HOME` — points to the original `~/.codex` (auth / config)

> **WebSocket mode:** Skill installation writes to the local filesystem and
> modifies subprocess environment variables. In WebSocket mode (no subprocess),
> the ephemeral home environment overrides do not apply. Skills must be
> pre-installed on the remote server or provided via `mcp_servers`.

## Guardrails

The Codex adapter supports the full guardrails stack declared on the prompt:

- **Tool policies**: Enforced in `BridgedTool` before handler execution
- **Feedback providers**: Collected after successful tool calls via
  `append_feedback()` in `_guardrails.py` and appended as additional content
- **Task completion**: Continuation loop in `execute_protocol` (max 10 rounds)
  re-prompts the agent when the checker reports incomplete with feedback

Implementation: `adapters/codex_app_server/_guardrails.py`

## Events

| Event | When |
|-------|------|
| `PromptRendered` | After render, before `turn/start` |
| `RenderedTools` | After render, correlated with `PromptRendered` via `render_event_id` |
| `ToolInvoked` | Each bridged tool call + each native Codex tool (command, file change) |
| `PromptExecuted` | After `turn/completed` (includes `TokenUsage` if available) |

## Token Usage

The `thread/tokenUsage/updated` notification provides detailed usage data:

```json
{
  "method": "thread/tokenUsage/updated",
  "params": {
    "threadId": "...",
    "turnId": "...",
    "tokenUsage": {
      "last": {
        "inputTokens": 8260,
        "outputTokens": 35,
        "cachedInputTokens": 0,
        "reasoningOutputTokens": 0,
        "totalTokens": 8295
      },
      "total": { "...same fields..." },
      "modelContextWindow": 258400
    }
  }
}
```

At `src/weakincentives/adapters/codex_app_server/_events.py`:
`extract_token_usage()` maps the `last` breakdown to WINK's `TokenUsage`:

| Codex field | WINK `TokenUsage` field |
|-------------|------------------------|
| `inputTokens` | `input_tokens` |
| `outputTokens` | `output_tokens` |
| `cachedInputTokens` | `cached_tokens` |

`TokenUsage.total_tokens` is a computed property (not stored).
The adapter uses the `last` breakdown for per-turn usage.

## Testing

### Unit Tests

- Mock Codex app-server (echo-style NDJSON over stdio)
- Verify `PromptRendered`, `PromptExecuted` emitted once
- Verify `ToolInvoked` for `commandExecution` and `fileChange` items
- Verify dynamic tool call handling (`item/tool/call` → BridgedTool → response)
- Verify structured output via `outputSchema`
- Verify approval auto-response per policy
- Verify `turn/interrupt` on deadline expiry
- Verify thread resume with session state
- Verify authentication flows (API key, external tokens)

### WebSocket Transport Tests

- Verify WebSocket connect, initialize, thread/turn lifecycle
- Verify dynamic tool calls over WebSocket (bidirectional server requests)
- Verify transport-level auth (bearer token, rejection on 401)
- Verify reconnection behavior (new session per connection)
- Verify graceful handling of server termination (close code 1006)
- Verify backpressure error `-32001` handling
- Verify multiple concurrent connections get independent sessions

### Integration Tests

Skip unless `codex` on PATH:

- Spawn `codex app-server` in temp workspace
- Simple prompt, verify response
- Dynamic tool invocation, verify `ToolInvoked`
- Thread resume, verify continuity
- WebSocket mode: spawn with `--listen ws://`, connect, full lifecycle

### Security Tests

- Workspace paths restrict file operations
- `allowed_host_roots` enforced
- Sandbox policy correctly propagated
- WebSocket auth tokens not logged or leaked in error messages

## Non-Goals (v1)

- Full Codex review integration (`review/start`) — can be added later
- ChatGPT browser OAuth flow — requires interactive browser
- Per-turn `sandboxPolicy` overrides — thread-level `SandboxMode` is sufficient
- Apps/connectors (`app/list`) — can be added later
- Configuration management (`config/*`) — Codex handles its own config
- Multi-thread management — one thread per `evaluate()` call
- WebSocket connection pooling — one connection per `evaluate()` call
- Automatic reconnection with session resumption — the adapter creates a new
  session on each `evaluate()` call, so reconnection is not needed within a
  single evaluation

## Design Decisions

### Why App Server over Codex SDK

The Codex SDK (`codex-sdk`) is designed for automation and CI jobs — fire and
forget. The app-server protocol provides:

1. **Streaming progress** — item-level granularity for real-time visibility
1. **Thread persistence** — resume conversations across `evaluate()` calls
1. **Approval flows** — programmatic approval handling
1. **Full lifecycle control** — initialize, authenticate, configure per-thread

For WINK's use case of deeply integrated agent orchestration with session state,
the app-server protocol is the correct abstraction.

### Why Dual Transport

stdio is simple and works well for local, single-client usage. WebSocket
enables deployment topologies where the Codex App Server runs on dedicated
infrastructure:

- **GPU servers** with local model access
- **Shared clusters** serving multiple WINK agents
- **Cloud deployments** where WINK runs separately from Codex

The protocol is identical across transports, so the transport choice is purely
an operational decision. Agent definitions, tools, and policies are unaffected.

### Why Dynamic Tools for WINK Tools

Dynamic tools are the simplest mechanism for bridging WINK tools to Codex:

- **Zero dependencies** — no `mcp`, `starlette`, `uvicorn`, or HTTP server
- **In-process execution** — `BridgedTool.__call__()` runs in the adapter
  process with full access to session state, resources, and transactional
  snapshots
- **Same pattern as approvals** — `item/tool/call` server requests are handled
  identically to approval requests in the message loop
- **3-line conversion** — `bridged_tools_to_dynamic_specs()` converts
  BridgedTool to DynamicToolSpec with no schema transformation

The `experimentalApi` capability required by dynamic tools is a single flag on
`initialize` and the protocol is stable — it powers the Codex VS Code
extension's tool integration.

### Why Auto-Accept Approvals by Default

WINK agents run programmatically without a human in the loop. Approval gates
are designed for interactive use (VS Code extension). For WINK:

- `"never"` (auto-accept) is the safe default for trusted workspaces
- Sandbox policy is the primary security boundary
- Callers can opt into `"on-request"` for maximum approval gating

## Known Limitations (WebSocket Transport)

The following limitations apply to the current WebSocket transport
implementation and are expected to be addressed in future iterations.

### Bearer token over plaintext `ws://`

When `ws_auth_token` is set, the adapter sends the `Authorization: Bearer`
header on any URL including unencrypted `ws://` endpoints. Over a non-loopback
network this exposes the capability token to interception. Callers connecting
to remote servers should use `wss://` until the adapter enforces TLS for
non-loopback bearer auth.

### Implicit local `cwd` in external mode

When `remote_url` is set (external WebSocket mode), the adapter's
`_resolve_cwd()` logic still falls back to creating a local temp directory or
using the local `Path.cwd()`. That local path is then sent to the remote
server's `thread/start`, where it is unlikely to be valid. Callers using
external mode **must** set an explicit `cwd` that exists on the remote server.
A future version should reject implicit `cwd` resolution when `remote_url` is
configured.

### Hard-coded managed-WS startup timeout

The managed WebSocket startup waits a fixed 5 seconds (50 retries × 0.1 s) for
the subprocess to begin accepting TCP connections. This limit is independent
of `startup_timeout_s`, so raising the configured timeout does not extend the
TCP-ready wait. A future version should plumb `startup_timeout_s` into the
transport bring-up path.

## Security Considerations (WebSocket Transport)

### Transport-level authentication

The Codex App Server supports `--ws-auth capability-token` and
`--ws-auth signed-bearer-token`. When deploying a server accessible beyond
loopback, **always** enable one of these modes and use `wss://`.

### Token handling

`ws_auth_token` is stored in `CodexAppServerClientConfig` as a plain string.
It is never logged by the adapter, but callers should treat the config object
as sensitive and avoid serializing it to logs or debug bundles.

### Sandbox boundaries in external mode

When connecting to a remote Codex server, the sandbox policy (`sandbox_mode`)
is enforced by the **remote** server, not the adapter. The adapter has no
way to verify that the remote server honours the requested policy. Operators
must ensure the remote server is configured with appropriate sandbox
restrictions independently.

### Skill installation

`CodexEphemeralHome` installs skills by writing to a local filesystem and
setting `HOME`/`CODEX_HOME` environment variables on the subprocess. In
external WebSocket mode there is no subprocess, so skills provided via
`RenderedPrompt.skills` are not installed. Skills must be pre-installed on
the remote server or provided through `mcp_servers`.

## Appendix: Protocol Reference

### Validated Behaviors

All protocol details in this spec were validated against `codex-cli 0.118.0`.

#### stdio Transport (validated)

- NDJSON framing, no `"jsonrpc"` header, integer `id` correlation
- Initialize → initialized handshake; double-initialize rejected
- Dynamic tools via `experimentalApi` capability
- `item/tool/call` bidirectional server requests
- Approval auto-response, `turn/interrupt` on deadline
- Structured output via `outputSchema`

#### WebSocket Transport (validated)

- One JSON object per WebSocket text frame (no `\n` delimiter)
- Protocol messages identical to stdio — same JSON-RPC objects
- Multiple concurrent clients: each connection gets an independent session
- Server exits cleanly when terminated; clients receive close code 1006
- Binary frames: silently ignored by server
- Invalid JSON: silently ignored, connection survives for subsequent messages
- Transport auth: `Authorization: Bearer TOKEN` header on upgrade, enforced
  only when server started with `--ws-auth`
- Loopback connections: auth not required unless explicitly configured
- Reconnection: new connection creates a new session (no session resumption)
- Dynamic tools and bidirectional tool calls work identically to stdio

### Available Models (ChatGPT auth)

`gpt-5.4` (default), `gpt-5.3-codex`, `gpt-5.2-codex`, `gpt-5.1-codex-max`,
`gpt-5.2`, `gpt-5.1-codex-mini`. Model availability depends on auth type and plan.

### MCP Server Config Formats

Codex supports subprocess (`{"command": "...", "args": [...]}`) and HTTP
(`{"url": "http://..."}`) transports on `config.mcp_servers`, passed via
`thread/start`.

### Codex App Server CLI

```
codex app-server [OPTIONS]

Options:
  --listen <URL>              Transport: stdio:// (default) or ws://IP:PORT
  --ws-auth <MODE>            Auth mode: capability-token | signed-bearer-token
  --ws-token-file <PATH>      Token file for capability-token auth
  --ws-shared-secret-file     Shared secret for signed JWT auth
  --ws-issuer <ISSUER>        Expected JWT issuer
  --ws-audience <AUDIENCE>    Expected JWT audience
  -c, --config <key=value>    Override config.toml values
  --enable/--disable <FEAT>   Toggle features
```

## Related Specifications

- `specs/ADAPTERS.md` — Provider adapter protocol
- `specs/CLAUDE_AGENT_SDK.md` — Reference adapter architecture
- `specs/ACP_ADAPTER.md` — Generic ACP adapter
- `specs/OPENCODE_ADAPTER.md` — OpenCode ACP adapter
- `specs/PROMPTS.md` — Prompt system
- `specs/SESSIONS.md` — Session state and events
- `specs/TOOLS.md` — Tool registration and policies
- `specs/WORKSPACE.md` — Workspace management

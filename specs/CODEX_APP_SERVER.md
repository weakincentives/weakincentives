# Codex App Server Adapter Specification

> **Adapter name:** `codex_app_server`
> **Codex entrypoint:** `codex app-server`
> **Protocol:** JSON-RPC 2.0 (without `"jsonrpc":"2.0"` header) over newline-delimited JSON on stdio
> **Validated against:** `codex-cli 0.98.0` with ChatGPT auth

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
lifecycle over stdio, making it suitable for deep product integrations while
keeping the agent definition (prompts, tools, policies) portable.

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

1. **Codex CLI** installed and available on `PATH` as `codex`
1. WINK (`weakincentives`) runtime

No additional Python dependencies beyond WINK. The adapter reuses `BridgedTool`
and `create_bridged_tools()` from the Claude Agent SDK adapter module (already
in the WINK codebase), but does not require the `claude-agent-sdk` package at
runtime — tool bridging uses Codex's native dynamic tools protocol.

The adapter uses lazy imports and raises a helpful error if `codex` is not
found on PATH.

## Architecture

```
WINK Prompt/Session
  └─ CodexAppServerAdapter.evaluate()
      ├─ Render WINK prompt → markdown text
      ├─ create_bridged_tools() → BridgedTool list
      ├─ Convert to DynamicToolSpec list [{name, description, inputSchema}]
      ├─ Spawn: codex app-server (stdio NDJSON)
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
  adapter.py                # CodexAppServerAdapter
  config.py                 # CodexAppServerClientConfig, CodexAppServerModelConfig
  client.py                 # CodexAppServerClient (stdio JSON-RPC client)
  workspace.py              # CodexWorkspaceSection
  _state.py                 # CodexAppServerSessionState slice for thread reuse
  _events.py                # Codex item/turn notifications → WINK ToolInvoked mapping
  _async.py                 # asyncio helpers for stdio NDJSON processing
```

Tool bridging reuses `BridgedTool` and `create_bridged_tools()` from
`src/weakincentives/adapters/claude_agent_sdk/_bridge.py`.

## Configuration

### CodexAppServerClientConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `codex_bin` | `str` | `"codex"` | Executable to spawn |
| `cwd` | `str \| None` | `None` | Working directory (must be absolute; defaults to `Path.cwd().resolve()`) |
| `env` | `Mapping[str, str] \| None` | `None` | Extra environment variables |
| `suppress_stderr` | `bool` | `True` | Capture stderr for debugging (not printed unless debugging) |
| `startup_timeout_s` | `float` | `10.0` | Max time for initialize handshake |
| `approval_policy` | `ApprovalPolicy` | `"never"` | How to handle command/file approvals |
| `sandbox_mode` | `SandboxMode \| None` | `None` | Sandbox mode for `thread/start` |
| `auth_mode` | `CodexAuthMode \| None` | `None` | Authentication configuration |
| `reuse_thread` | `bool` | `False` | Resume existing thread ID from session state |
| `mcp_servers` | `dict[str, McpServerConfig] \| None` | `None` | Additional external MCP servers |
| `ephemeral` | `bool` | `False` | If true, thread is not persisted to disk |
| `client_name` | `str` | `"wink"` | Client identifier for `initialize` |
| `client_version` | `str` | `"0.1.0"` | Client version for `initialize` |

> **CWD requirement:** `thread/start` requires `cwd` to be an absolute path. If
> `None`, the adapter resolves to `Path.cwd().resolve()`.

> **Approval handling:** `approval_policy="never"` means the adapter auto-accepts
> all approvals. For non-interactive WINK execution, `"never"` is the default
> since there is no human to prompt.

> **Tool namespace:** WINK bridged tools are registered as dynamic tools.
> User-provided `mcp_servers` are passed to Codex via `config.mcp_servers` on
> `thread/start`. External MCP tool names must not collide with WINK tool names.

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

```python
CodexAuthMode = ApiKeyAuth | ExternalTokenAuth

@FrozenDataclass()
class ApiKeyAuth:
    api_key: str

@FrozenDataclass()
class ExternalTokenAuth:
    id_token: str
    access_token: str
```

Authentication is performed after `initialize` via `account/login/start`. When
`auth_mode` is `None`, the adapter skips authentication and assumes the Codex
CLI environment is already authenticated (the default — Codex inherits host-level
credentials from `~/.codex/`).

### CodexAppServerModelConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `"gpt-5.1-codex"` | Codex model identifier |
| `effort` | `ReasoningEffort \| None` | `None` | Reasoning effort |
| `summary` | `ReasoningSummary \| None` | `None` | Summary preference |
| `personality` | `Personality \| None` | `None` | Response personality |

```python
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
ReasoningSummary = Literal["auto", "concise", "detailed", "none"]
Personality = Literal["none", "friendly", "pragmatic"]
```

**Note:** `seed`, `stop`, `presence_penalty`, `frequency_penalty` are not
supported by the Codex app-server — raises `ValueError` if provided.

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

| Codex Item Type | WINK Event | Notes |
|-----------------|------------|-------|
| `commandExecution` (completed) | `ToolInvoked` | `success` from exit code |
| `fileChange` (completed) | `ToolInvoked` | `success` from status |
| `mcpToolCall` (completed) | `ToolInvoked` | External MCP servers only |
| `item/tool/call` (server request) | `ToolInvoked` | WINK bridged tools via dynamic tools |
| `agentMessage` | Text accumulation | Concatenated for `PromptResponse.text` |
| `reasoning` | (informational) | Logged if configured |
| `webSearch` | `ToolInvoked` | Optional native tool tracking |
| `contextCompaction` | (informational) | Logged |

### Dual Notification System

Codex emits notifications in two parallel namespaces:

- **`item/*`, `turn/*`, `thread/*`** — standardized v2 protocol (use these)
- **`codex/event/*`** — legacy v1 events (ignore; same content, different shape)

The adapter should only process v2 notifications.

## Session State Storage

WINK sessions use **typed dataclass slices**. For thread reuse:

```python
# _state.py
from weakincentives.dataclasses import FrozenDataclass

@FrozenDataclass()
class CodexAppServerSessionState:
    """Stores Codex thread ID and workspace fingerprint for reuse."""
    thread_id: str
    cwd: str
    workspace_fingerprint: str | None
```

Usage:

```python
from pathlib import Path

# Store after thread/start
resolved_cwd = client_config.cwd or str(Path.cwd().resolve())
session.seed(
    CodexAppServerSessionState(
        thread_id=result["thread"]["id"],
        cwd=resolved_cwd,
        workspace_fingerprint=workspace_fingerprint,
    )
)

# Retrieve for thread/resume
state = session[CodexAppServerSessionState].latest()
if state is not None:
    thread_id = state.thread_id
    cached_cwd = state.cwd
    cached_workspace_fingerprint = state.workspace_fingerprint
```

When `reuse_thread=True`, only reuse the thread if `cached_cwd` and
`cached_workspace_fingerprint` match the current workspace. If they differ or
`thread/resume` fails, fall back to `thread/start` and overwrite the stored
state. Compute `workspace_fingerprint` from mount config and budgets (stable
ordering) so reuse is deterministic.

## Workspace Management

### CodexWorkspaceSection

Create by **extracting** generic mount/copy logic from
`src/weakincentives/adapters/claude_agent_sdk/workspace.py`:

- Accepts `HostMount` tuples, `allowed_host_roots`, max-bytes budgets
- Materializes temporary directory with copied files
- Exposes `temp_dir` for `CodexAppServerClientConfig.cwd`
- Renders a provider-agnostic summary of mounts and budgets
- Exposes cleanup via `.cleanup()` or a context manager
- Provides `workspace_fingerprint` for session reuse validation

> **Do not** reuse `ClaudeAgentWorkspaceSection` directly — its template
> contains Claude-specific wording. Extract the machinery and create a new
> section with neutral text.

**Shared types:**

| Type | Description |
|------|-------------|
| `HostMount` | Mount configuration (host_path, mount_path, globs, max_bytes) |
| `HostMountPreview` | Summary of materialized mount |
| `WorkspaceBudgetExceededError` | Mount exceeds byte budget |
| `WorkspaceSecurityError` | Mount violates security constraints |

## Stdio JSON-RPC Client

### CodexAppServerClient

The client manages the `codex app-server` subprocess and provides a typed
interface over the NDJSON stdio protocol.

```python
class CodexAppServerClient:
    """Bidirectional JSON-RPC client for the Codex app-server."""

    def __init__(
        self,
        codex_bin: str = "codex",
        env: Mapping[str, str] | None = None,
        suppress_stderr: bool = True,
    ) -> None: ...

    async def start(self) -> None:
        """Spawn codex app-server subprocess."""

    async def stop(self) -> None:
        """Terminate subprocess gracefully."""

    async def send_request(
        self, method: str, params: dict[str, Any], timeout: float | None = None
    ) -> Any:
        """Send JSON-RPC request and await response by matching id."""

    async def send_notification(self, method: str, params: dict[str, Any]) -> None:
        """Send JSON-RPC notification (no id, no response expected)."""

    async def send_response(self, request_id: int, result: dict[str, Any]) -> None:
        """Send response to a server-initiated request."""

    async def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        """Yield all messages from stdout (responses, notifications, server requests)."""
```

**Wire format:** Each message is a single JSON object terminated by `\n`. The
client assigns incrementing integer `id` fields to requests and correlates
responses by `id`. Notifications (no `id`) are routed to subscribers.

**Important:** The Codex protocol omits the `"jsonrpc": "2.0"` header — do
not include it.

### Message Routing

The client must demultiplex stdout into three streams:

1. **Responses** — messages with an `id` field matching a pending request
1. **Notifications** — messages with a `method` field and no `id`
1. **Server requests** — messages with both `method` and `id` (approval
   requests and dynamic tool calls); the client must respond with a matching `id`

Server-initiated requests require the client to respond promptly. Dynamic tool
calls (`item/tool/call`) and approval requests both follow this pattern.

## Structured Output

When `rendered.output_type is not None`, the adapter uses Codex's **native
`outputSchema`** parameter on `turn/start`. This constrains the model's final
message to valid JSON conforming to the schema.

### Schema Generation

Use WINK's existing `serde.schema()`:

```python
from weakincentives.serde import schema

json_schema = schema(rendered.output_type)
```

### Passing the Schema

```python
result = send_request("turn/start", {
    "threadId": thread_id,
    "input": [{"type": "text", "text": rendered_text}],
    "outputSchema": json_schema,
    ...
})
```

### Retrieval

After `turn/completed`:

1. Parse the accumulated `agentMessage` delta text as JSON
1. Deserialize via `serde.parse(output_type, parsed_json)`
1. If parsing fails: raise `PromptEvaluationError(phase="response")`

No MCP tool is needed — the model produces valid JSON directly in its response
text when `outputSchema` is provided.

## Tool Bridging via Dynamic Tools

### Why Dynamic Tools

Dynamic tools are the simplest mechanism for exposing WINK tools to Codex.
When the model calls a dynamic tool, Codex sends an `item/tool/call` server
request **back over the same stdio channel** to the adapter process. The
adapter executes the `BridgedTool` in-process with full access to session
state and resources, then responds. No subprocess, no HTTP server, no extra
dependencies.

The entire integration is:

1. Convert `BridgedTool` list to `DynamicToolSpec` list (3-line function)
1. Pass `dynamicTools` on `thread/start` (requires `experimentalApi` on
   `initialize`)
1. Handle `item/tool/call` in the stdio message loop (same pattern as approval
   handling)

### Reused Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `BridgedTool` | `claude_agent_sdk/_bridge.py` | Transactional tool wrapper |
| `create_bridged_tools()` | `claude_agent_sdk/_bridge.py` | Factory for BridgedTool |
| `VisibilityExpansionSignal` | `claude_agent_sdk/_visibility_signal.py` | Exception propagation |
| `tool_transaction()` | `runtime/transactions.py` | Snapshot/restore |

> **Important:** Pass `adapter_name="codex_app_server"` to
> `create_bridged_tools()` to ensure `ToolInvoked` events are labeled correctly.

### DynamicToolSpec Conversion

```python
def bridged_tools_to_dynamic_specs(
    tools: tuple[BridgedTool, ...],
) -> list[dict[str, Any]]:
    """Convert BridgedTool list to Codex DynamicToolSpec format."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": tool.input_schema,
        }
        for tool in tools
    ]
```

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

When the adapter receives an `item/tool/call` server request:

```python
# Server request: {"id": 42, "method": "item/tool/call", "params": {...}}
tool_name = params["tool"]
arguments = params["arguments"]
call_id = params["callId"]

bridged_tool = tool_lookup[tool_name]
mcp_result = bridged_tool(arguments)
# mcp_result: {"content": [{"type": "text", "text": "..."}], "isError": bool}

# Convert to DynamicToolCallResponse format
send_response(request_id, {
    "success": not mcp_result.get("isError", False),
    "contentItems": [
        {"type": "inputText", "text": item["text"]}
        for item in mcp_result.get("content", [])
        if item.get("type") == "text"
    ],
})
```

The `DynamicToolCallResponse` format (validated against the JSON schema):

```python
{
    "success": bool,
    "contentItems": [
        {"type": "inputText", "text": str}    # text content
        | {"type": "inputImage", "imageUrl": str}  # image content
    ],
}
```

### External MCP Servers

User-provided MCP servers (not WINK tools) are passed to Codex via
`config.mcp_servers` on `thread/start`. These can be subprocess-based or
URL-based:

```python
"config": {
    "mcp_servers": {
        "user-stdio-server": {
            "command": "/path/to/server",
            "args": ["--flag"],
        },
        "user-http-server": {
            "url": "http://localhost:8080/mcp",
        },
    },
}
```

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

## Execution Flow

### 1. Budget/Deadline Setup

- Create `BudgetTracker` if budget provided
- Derive deadline from argument or `budget.deadline`
- Raise `PromptEvaluationError(phase="request")` if already expired

### 2. Render Prompt

1. `prepare_adapter_conversation(...)` → `AdapterRenderContext`
1. Emit `PromptRendered`

### 3. Build Dynamic Tool Specs

1. `create_bridged_tools(rendered.tools, adapter_name="codex_app_server", ...)`
1. `bridged_tools_to_dynamic_specs(bridged_tools)` → `DynamicToolSpec` list

### 4. Spawn Codex App Server

```python
proc = subprocess.Popen(
    [codex_bin, "app-server"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE if suppress_stderr else None,
    cwd=resolved_cwd,
    env=merged_env,
)
```

### 5. Initialize Handshake

```python
# experimentalApi enables dynamicTools on thread/start
send_request("initialize", {
    "clientInfo": {
        "name": client_name,    # "wink"
        "title": "WINK Agent",
        "version": client_version,
    },
    "capabilities": {
        "experimentalApi": True,
    },
})
# Response: {"result": {"userAgent": "wink/0.98.0 (...)"}}

# Notification (no response expected)
send_notification("initialized")
```

The server rejects all methods before `initialize`. Repeated `initialize` calls
return `Already initialized`.

### 6. Authenticate (Optional)

If `auth_mode` is provided:

```python
# API key auth
send_request("account/login/start", {
    "type": "apiKey",
    "apiKey": auth_mode.api_key,
})

# External token auth
send_request("account/login/start", {
    "type": "chatgptAuthTokens",
    "idToken": auth_mode.id_token,
    "accessToken": auth_mode.access_token,
})
```

The login is synchronous — `account/login/start` returns its result directly.
On error, raise `PromptEvaluationError(phase="request")`.

For `ExternalTokenAuth`, the adapter must handle
`account/chatgptAuthTokens/refresh` server requests — respond with refreshed
tokens or raise an error.

### 7. Start or Resume Thread

```python
# New thread
thread_params = {
    "model": model_config.model,
    "cwd": resolved_cwd,
    "approvalPolicy": approval_policy,
    "sandbox": sandbox_mode,
    "ephemeral": ephemeral,
    "dynamicTools": dynamic_tool_specs,  # WINK bridged tools
}
if additional_mcp_servers:
    thread_params["config"] = {"mcp_servers": additional_mcp_servers}

result = send_request("thread/start", thread_params)
thread_id = result["thread"]["id"]

# Resume existing thread
result = send_request("thread/resume", {
    "threadId": cached_thread_id,
})
```

Store thread ID via `session.seed(CodexAppServerSessionState(...))`.

### 8. Start Turn

```python
turn_params = {
    "threadId": thread_id,
    "input": [{"type": "text", "text": rendered_text}],
    "effort": model_config.effort,
    "summary": model_config.summary,
    "personality": model_config.personality,
}
if output_schema is not None:
    turn_params["outputSchema"] = output_schema

result = send_request("turn/start", turn_params)
turn_id = result["turn"]["id"]
```

### 9. Stream Notifications

After `turn/start`, keep reading stdout for all messages:

```python
async for message in client.read_messages():
    # Server requests have both "id" and "method"
    if "id" in message and "method" in message:
        match message["method"]:
            case "item/tool/call":
                # Dynamic tool call — execute BridgedTool in-process
                execute_dynamic_tool_call(message)
            case "item/commandExecution/requestApproval":
                respond_to_approval(message["id"], message["params"])
            case "item/fileChange/requestApproval":
                respond_to_approval(message["id"], message["params"])
            case "account/chatgptAuthTokens/refresh":
                handle_token_refresh(message["id"], message["params"])
        continue

    # Notifications have "method" but no "id"
    method = message["method"]
    params = message["params"]

    match method:
        case "item/agentMessage/delta":
            accumulated_text += params.get("delta", "")

        case "item/completed":
            item = params["item"]
            match item["type"]:
                case "commandExecution":
                    dispatch_tool_invoked(item)
                case "fileChange":
                    dispatch_tool_invoked(item)
                case "mcpToolCall":
                    dispatch_tool_invoked(item)  # external MCP only
                case "agentMessage":
                    accumulated_text = item.get("text", accumulated_text)

        case "thread/tokenUsage/updated":
            record_token_usage(params)

        case "turn/completed":
            final_turn = params["turn"]
            break
```

**Note:** WINK bridged tools arrive as `item/tool/call` server requests (same
as approval requests). External MCP tools arrive as `mcpToolCall` notification
items. The two paths are distinct — no deduplication needed.

### 10. Handle Approvals

When the server sends an approval request (a JSON-RPC request with `id`):

```python
def respond_to_approval(request_id: int, params: dict) -> None:
    match approval_policy:
        case "never":
            send_response(request_id, {"decision": "accept"})
        case "on-request":
            send_response(request_id, {"decision": "decline"})
        case "untrusted" | "on-failure":
            send_response(request_id, {"decision": "accept"})
```

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
`serde.parse(output_type, json.loads(text))`.
Raise `PromptEvaluationError(phase="response")` if parsing fails.

### 13. PromptExecuted

Emit event and return `PromptResponse(text=..., output=...)`.

## Cancellation

If deadline expires during a turn:

1. Send `turn/interrupt` — `{"threadId": thread_id, "turnId": turn_id}`
1. Wait for `turn/completed` with `status: "interrupted"` (bounded wait)
1. Kill subprocess if needed
1. Raise `PromptEvaluationError(phase="request")` or `DeadlineExceededError`

## Error Handling

### Error Phases

| Phase | When |
|-------|------|
| `"request"` | Spawn, initialize, auth, thread/start, or turn/start fails |
| `"response"` | Structured output missing or invalid; turn completes with `status: "failed"` |
| `"tool"` | Bridged tool execution failure |
| `"budget"` | Token budget exceeded |

### Turn Failure Mapping

When `turn/completed` has `status: "failed"`, map `codexErrorInfo` to WINK
error handling:

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

Include in payload: stderr tail (bounded, e.g., last 8k), Codex error details,
`codexErrorInfo`, and `additionalDetails`.

Tool telemetry errors: log but don't crash.

## Events

| Event | When |
|-------|------|
| `PromptRendered` | After render, before `turn/start` |
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

Map to WINK's `TokenUsage`:

```python
@FrozenDataclass()
class TokenUsage:
    prompt_tokens: int        # from inputTokens
    completion_tokens: int    # from outputTokens
    total_tokens: int         # from totalTokens
```

Use the `last` breakdown for per-turn usage and `total` for cumulative thread
usage.

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

### Integration Tests

Skip unless `codex` on PATH:

- Spawn `codex app-server` in temp workspace
- Simple prompt, verify response
- Dynamic tool invocation, verify `ToolInvoked`
- Thread resume, verify continuity

### Security Tests

- Workspace paths restrict file operations
- `allowed_host_roots` enforced
- Sandbox policy correctly propagated

## Usage Example

```python
from weakincentives import Prompt, PromptTemplate, MarkdownSection
from weakincentives.runtime import Session, InProcessDispatcher
from weakincentives.adapters.codex_app_server import (
    CodexAppServerAdapter,
    CodexAppServerClientConfig,
    CodexAppServerModelConfig,
)

bus = InProcessDispatcher()
session = Session(dispatcher=bus)

template = PromptTemplate(
    ns="demo",
    key="codex",
    sections=(
        MarkdownSection(
            title="Task",
            key="task",
            template="List the files in the repo and summarize.",
        ),
    ),
)
prompt = Prompt(template)

adapter = CodexAppServerAdapter(
    model_config=CodexAppServerModelConfig(
        model="gpt-5.1-codex",
        effort="medium",
    ),
    client_config=CodexAppServerClientConfig(
        cwd="/absolute/path/to/workspace",
        approval_policy="never",
        sandbox_mode="workspace-write",
    ),
)

with prompt.resources:
    resp = adapter.evaluate(prompt, session=session)

print(resp.text)
```

### With Authentication

```python
from weakincentives.adapters.codex_app_server import ApiKeyAuth

adapter = CodexAppServerAdapter(
    client_config=CodexAppServerClientConfig(
        auth_mode=ApiKeyAuth(api_key="sk-..."),
        cwd="/absolute/path/to/workspace",
    ),
)
```

### With Workspace Isolation

```python
from weakincentives.adapters.codex_app_server import (
    CodexAppServerAdapter,
    CodexAppServerClientConfig,
    CodexWorkspaceSection,
)
from weakincentives.adapters.claude_agent_sdk.workspace import HostMount

workspace = CodexWorkspaceSection(
    session=session,
    mounts=(HostMount(host_path="/abs/path/to/repo", mount_path="repo"),),
    allowed_host_roots=("/abs/path/to",),
)

adapter = CodexAppServerAdapter(
    client_config=CodexAppServerClientConfig(
        cwd=str(workspace.temp_dir),
        sandbox_mode="workspace-write",
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

- Full Codex review integration (`review/start`) — can be added later
- Codex skill invocation — can be added later
- ChatGPT browser OAuth flow — requires interactive browser
- Per-turn `sandboxPolicy` overrides — thread-level `SandboxMode` is sufficient
- Apps/connectors (`app/list`) — can be added later
- Configuration management (`config/*`) — Codex handles its own config
- Multi-thread management — one thread per `evaluate()` call

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

### Why Dynamic Tools for WINK Tools

Dynamic tools are the simplest mechanism for bridging WINK tools to Codex:

- **Zero dependencies** — no `mcp`, `starlette`, `uvicorn`, or HTTP server
- **In-process execution** — `BridgedTool.__call__()` runs in the adapter
  process with full access to session state, resources, and transactional
  snapshots
- **Same pattern as approvals** — `item/tool/call` server requests are handled
  identically to approval requests in the stdio message loop
- **3-line conversion** — `bridged_tools_to_dynamic_specs()` converts
  BridgedTool to DynamicToolSpec with no schema transformation

The alternative — running an in-process MCP HTTP server via
`StreamableHTTPServerTransport` — was prototyped and validated (see
`scratch/codex_probes/probe_20_mcp_http_bridge.py`). It works but requires
`claude-agent-sdk`, `mcp`, `starlette`, `uvicorn`, port allocation, background
threads, and HTTP server lifecycle management. Dynamic tools achieve the same
result with none of that complexity.

The `experimentalApi` capability required by dynamic tools is a single flag on
`initialize` and the protocol is stable — it powers the Codex VS Code
extension's tool integration.

### Why Auto-Accept Approvals by Default

WINK agents run programmatically without a human in the loop. Approval gates
are designed for interactive use (VS Code extension). For WINK:

- `"never"` (auto-accept) is the safe default for trusted workspaces
- Sandbox policy is the primary security boundary
- Callers can opt into `"on-request"` for maximum approval gating

## Appendix: Protocol Reference

### Validated with Probes

All protocol details in this spec were validated against `codex-cli 0.98.0`
using probe scripts in `scratch/codex_probes/`. Key findings are documented in
`scratch/codex_probes/FINDINGS.md`. The end-to-end driver
(`scratch/codex_probes/codex_code_reviewer_driver.py`) exercises the full
flow: initialize, thread/start with dynamic tools, turn/start with
outputSchema, item/tool/call handling, structured output parsing, and
turn/completed.

### Available Models (ChatGPT auth)

`gpt-5.2-codex`, `gpt-5.3-codex`, `gpt-5.1-codex-max`, `gpt-5.2`,
`gpt-5.1-codex-mini`. Model availability depends on auth type and plan.

### MCP Server Config Formats

Codex supports two MCP server transport types on `config.mcp_servers`:

```python
# Subprocess (stdio)
{"command": "/path/to/server", "args": ["--flag"]}

# HTTP (streamable)
{"url": "http://localhost:8080/mcp"}
```

Both are passed via `thread/start` → `config.mcp_servers`.

## Related Specifications

- `specs/ADAPTERS.md` — Provider adapter protocol
- `specs/CLAUDE_AGENT_SDK.md` — Reference adapter architecture
- `specs/OPENCODE_ACP_ADAPTER.md` — Second adapter, similar patterns
- `specs/PROMPTS.md` — Prompt system
- `specs/SESSIONS.md` — Session state and events
- `specs/TOOLS.md` — Tool registration and policies
- `specs/WORKSPACE.md` — Workspace management

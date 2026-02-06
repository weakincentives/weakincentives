# Codex App Server Adapter Specification

> **Adapter name:** `codex_app_server`
> **Codex entrypoint:** `codex app-server`
> **Protocol:** JSON-RPC 2.0 (without `"jsonrpc":"2.0"` header) over newline-delimited JSON on stdio

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
- **MCP tool bridging:** Custom tools via MCP servers
- **Sandboxing:** Configurable sandbox policies (read-only, workspace-write,
  full access, external sandbox)
- **Approval flows:** Command and file change approvals
- **Reviews:** Automated code review via `review/start`
- **Skills:** Codex skill invocation

## Requirements

### Runtime Dependencies

1. **Codex CLI** installed and available on `PATH` as `codex`
1. **Claude Agent SDK**: `claude-agent-sdk>=0.1.15` (for MCP server infrastructure)
1. WINK (`weakincentives`) runtime

### WINK Packaging

```toml
[project.optional-dependencies]
codex = [
  "claude-agent-sdk>=0.1.15",
]
```

The adapter takes a dependency on `claude-agent-sdk` to reuse its MCP server
infrastructure for tool bridging. This coupling is acceptable because both
adapters share the same tool bridging semantics, and implementing a separate
MCP server adds significant complexity.

The adapter uses lazy imports and raises a helpful error if the `codex` extra is
not installed (following the pattern in `claude_agent_sdk/*`).

## Architecture

```
WINK Prompt/Session
  └─ CodexAppServerAdapter.evaluate()
      ├─ Render WINK prompt → markdown text
      ├─ Start in-process MCP server (reuses Claude SDK infrastructure)
      ├─ Spawn: codex app-server (stdio NDJSON)
      ├─ Handshake: initialize → initialized notification
      ├─ thread/start (model, cwd, sandbox, approval policy)
      ├─ turn/start (text input + MCP server config)
      ├─ Stream: item/*, turn/* notifications
      │    ├─ item/agentMessage/delta (assistant output)
      │    ├─ item/started + item/completed (tool calls, commands, file changes)
      │    ├─ turn/plan/updated (agent planning)
      │    └─ turn/completed (final status)
      ├─ Model calls structured_output MCP tool to finalize (if required)
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
  _structured_output.py     # structured_output MCP tool
  _async.py                 # asyncio helpers for stdio NDJSON processing
  _approval.py              # Approval request handling
```

MCP tool bridging reuses `create_mcp_server()` from
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
| `sandbox_policy` | `SandboxPolicy \| None` | `None` | Sandbox configuration for threads |
| `auth_mode` | `CodexAuthMode \| None` | `None` | Authentication configuration |
| `personality` | `str \| None` | `None` | Codex personality setting |
| `reuse_thread` | `bool` | `False` | Resume existing thread ID from session state |
| `mcp_servers` | `tuple[McpServerConfig, ...] \| None` | `None` | Additional MCP servers (WINK server always added) |
| `client_name` | `str` | `"wink"` | Client identifier for `initialize` |
| `client_version` | `str` | `"0.1.0"` | Client version for `initialize` |

> **CWD requirement:** `thread/start` requires `cwd` to be an absolute path. If
> `None`, the adapter resolves to `Path.cwd().resolve()`.

> **Approval handling:** `approval_policy="never"` means the adapter auto-accepts
> all approvals. `"always"` means auto-decline. `"unlessTrusted"` means accept
> for trusted tools, decline otherwise. For non-interactive WINK execution,
> `"never"` is the default since there is no human to prompt.

> **MCP merge:** The adapter always injects its own WINK MCP server for bridged
> tools. User-provided `mcp_servers` are passed to `thread/start`; they must not
> shadow the WINK tool namespace.

#### ApprovalPolicy

```python
ApprovalPolicy = Literal["never", "always", "unlessTrusted"]
```

| Value | Behavior |
|-------|----------|
| `"never"` | Auto-accept all approvals (no human gating) |
| `"always"` | Auto-decline all approvals |
| `"unlessTrusted"` | Maps to Codex `approvalPolicy: "unlessTrusted"` on `thread/start` |

#### SandboxPolicy

```python
@FrozenDataclass()
class SandboxPolicy:
    type: Literal["readOnly", "workspaceWrite", "dangerFullAccess", "externalSandbox"]
    writable_roots: tuple[str, ...] = ()
    network_access: bool = True
```

Maps directly to the Codex `sandboxPolicy` parameter on `thread/start` and
`turn/start`.

#### CodexAuthMode

```python
CodexAuthMode = ApiKeyAuth | ChatGptAuth | ExternalTokenAuth

@FrozenDataclass()
class ApiKeyAuth:
    api_key: str

@FrozenDataclass()
class ChatGptAuth:
    """Uses browser-based ChatGPT OAuth flow."""

@FrozenDataclass()
class ExternalTokenAuth:
    id_token: str
    access_token: str
```

Authentication is performed after `initialize` via `account/login/start`. When
`auth_mode` is `None`, the adapter skips authentication and assumes the Codex
CLI environment is already authenticated.

### CodexAppServerModelConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `"gpt-5.1-codex"` | Codex model identifier |
| `effort` | `Literal["low", "medium", "high"] \| None` | `None` | Reasoning effort |
| `summary` | `Literal["concise", "detailed"] \| None` | `None` | Summary preference |

**Note:** `seed`, `stop`, `presence_penalty`, `frequency_penalty` are not
supported by the Codex app-server — raises `ValueError` if provided.

## Protocol Mapping

### WINK Concepts → Codex App Server Concepts

| WINK Concept | Codex App Server Concept | Adapter Role |
|--------------|--------------------------|--------------|
| **Prompt** (PromptTemplate + sections + tools) | Thread + Turn input text | Render, format, send via `turn/start` |
| **Session** (event-sourced state) | Thread (persistent conversation) | Map thread/turn lifecycle to session events |
| **Tool** (Tool[ParamsT, ResultT]) | MCP tool via server | Bridge via `create_mcp_server()` |
| **Tool Execution** (transactional) | Item (`mcpToolCall`, `commandExecution`, `fileChange`) | Map `item/completed` → `ToolInvoked` |
| **Output** (structured dataclass) | MCP `structured_output` tool call | Parse, deserialize, return |
| **Events** (PromptRendered, ToolInvoked, PromptExecuted) | `item/*`, `turn/*` notifications | Translate and dispatch |
| **Deadline** | Turn interrupt via `turn/interrupt` | Enforce with timer + interrupt |
| **Budget** | Per-model token tracking | Record usage from `thread/tokenUsage/updated` |

### Codex Item Types → WINK Events

| Codex Item Type | WINK Event | Notes |
|-----------------|------------|-------|
| `commandExecution` (completed) | `ToolInvoked` | `success` from exit code |
| `fileChange` (completed) | `ToolInvoked` | `success` from status |
| `mcpToolCall` (completed) | `ToolInvoked` | Deduplicate bridged WINK tools |
| `agentMessage` | Text accumulation | Concatenated for `PromptResponse.text` |
| `plan` | (informational) | Logged if configured |
| `reasoning` | (informational) | Logged if configured |
| `webSearch` | `ToolInvoked` | Optional native tool tracking |
| `contextCompaction` | (informational) | Logged |

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

    async def read_notifications(self) -> AsyncIterator[dict[str, Any]]:
        """Yield server-initiated notifications from stdout."""
```

**Wire format:** Each message is a single JSON object terminated by `\n`. The
client assigns incrementing integer `id` fields to requests and correlates
responses by `id`. Notifications (no `id`) are routed to subscribers.

**Important:** The Codex protocol omits the `"jsonrpc": "2.0"` header — do
not include it.

### Message Routing

The client must demultiplex stdout into two streams:

1. **Responses** — messages with an `id` field matching a pending request
1. **Notifications** — messages with a `method` field and no `id`
1. **Server requests** — messages with both `method` and `id` (e.g., approval
   requests); the client must respond with a matching `id`

Server-initiated requests (approvals) require the client to respond promptly.
Use `approval_policy` to auto-respond without blocking.

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

Use WINK's existing logic for compatibility:

```python
from weakincentives.adapters.response_parser import build_json_schema_response_format

schema_format = build_json_schema_response_format(rendered, prompt_name)
json_schema = schema_format["json_schema"]["schema"]
```

This handles array containers (`rendered.container == "array"`) and extra keys
policy (`rendered.allow_extra_keys`).

### Tool Description

```
Call this tool to submit your final structured output.
The data must conform to the following JSON schema:
{json_schema}
```

### Retrieval

After `turn/completed`:

1. Check if model called `structured_output` via the MCP tool
1. If called: retrieve validated output
1. If not called or validation failed: raise `PromptEvaluationError(phase="response")`

`structured_output` emits a `ToolInvoked` event like any other bridged tool.

## MCP Tool Bridging

### Reusing Claude Agent SDK Infrastructure

The adapter reuses the **exact same MCP server** from the Claude Agent SDK:

```python
from weakincentives.adapters.claude_agent_sdk._bridge import (
    BridgedTool,
    create_bridged_tools,
    create_mcp_server,
)
```

Benefits:

- Direct access to WINK session state and resources
- Full transactional semantics without IPC
- Proven, tested implementation

### Reused Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `BridgedTool` | `claude_agent_sdk/_bridge.py` | Transactional tool wrapper |
| `create_bridged_tools()` | `claude_agent_sdk/_bridge.py` | Factory for BridgedTool |
| `create_mcp_server()` | `claude_agent_sdk/_bridge.py` | In-process MCP server |
| `VisibilityExpansionSignal` | `claude_agent_sdk/_visibility_signal.py` | Exception propagation |
| `tool_transaction()` | `runtime/transactions.py` | Snapshot/restore |

> **Important:** Pass `adapter_name="codex_app_server"` to
> `create_bridged_tools()` to ensure `ToolInvoked` events are labeled correctly.

### Tool Bridging Flow

```
1. Render prompt → rendered.tools
2. create_bridged_tools(..., adapter_name="codex_app_server")
3. Create structured_output tool if output_type declared
4. create_mcp_server(bridged_tools + structured_output_tool)
5. thread/start with MCP server configuration
6. Codex connects to in-process MCP server
7. Tool call → BridgedTool.__call__()
   ├─ Snapshot session state
   ├─ Execute handler
   ├─ Dispatch ToolInvoked
   └─ Rollback on failure
```

### MCP Server Advertisement

The WINK MCP server is advertised to Codex via `thread/start` or `turn/start`
parameters. The exact mechanism depends on how Codex discovers MCP servers —
either via the `dynamicTools` parameter or through configuration-level MCP
server registration.

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

### 3. Start MCP Server

1. `create_bridged_tools(rendered.tools, adapter_name="codex_app_server", ...)`
1. Create `structured_output` tool if `output_type` declared
1. `create_mcp_server(all_tools)`

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
# Request
send_request("initialize", {
    "clientInfo": {
        "name": client_name,    # "wink"
        "title": "WINK Agent",
        "version": client_version,
    }
})

# Notification (no response expected)
send_notification("initialized", {})
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

Wait for `account/login/completed` notification. On failure, raise
`PromptEvaluationError(phase="request")`.

For `ExternalTokenAuth`, the adapter must handle
`account/chatgptAuthTokens/refresh` server requests — respond with refreshed
tokens or raise an error. The adapter stores a callback for token refresh.

### 7. Start or Resume Thread

```python
# New thread
result = send_request("thread/start", {
    "model": model_config.model,
    "cwd": resolved_cwd,
    "approvalPolicy": approval_policy,
    "sandbox": sandbox_policy.type if sandbox_policy else None,
    "sandboxPolicy": {
        "type": sandbox_policy.type,
        "writableRoots": list(sandbox_policy.writable_roots),
        "networkAccess": sandbox_policy.network_access,
    } if sandbox_policy else None,
    "personality": personality,
})
thread_id = result["thread"]["id"]

# Resume existing thread
result = send_request("thread/resume", {
    "threadId": cached_thread_id,
    "personality": personality,
})
```

Store thread ID via `session.seed(CodexAppServerSessionState(...))`.

### 8. Start Turn

```python
result = send_request("turn/start", {
    "threadId": thread_id,
    "input": [{"type": "text", "text": rendered_text}],
    "cwd": resolved_cwd,
    "model": model_config.model,
    "effort": model_config.effort,
    "summary": model_config.summary,
})
turn_id = result["turn"]["id"]
```

### 9. Stream Notifications

After `turn/start`, keep reading stdout for notifications:

```python
async for notification in client.read_notifications():
    method = notification["method"]
    params = notification["params"]

    match method:
        case "item/agentMessage/delta":
            accumulated_text += params.get("text", "")

        case "item/started":
            item = params["item"]
            # Track in-progress items

        case "item/completed":
            item = params["item"]
            match item["type"]:
                case "commandExecution":
                    dispatch_tool_invoked(item)
                case "fileChange":
                    dispatch_tool_invoked(item)
                case "mcpToolCall":
                    if not is_wink_bridged_tool(item):
                        dispatch_tool_invoked(item)
                case "agentMessage":
                    accumulated_text = item.get("text", accumulated_text)

        case "item/commandExecution/requestApproval":
            respond_to_approval(params)

        case "item/fileChange/requestApproval":
            respond_to_approval(params)

        case "thread/tokenUsage/updated":
            record_token_usage(params)

        case "turn/completed":
            final_turn = params["turn"]
            break
```

### 10. Handle Approvals

When the server sends an approval request (a JSON-RPC request with `id`):

```python
def respond_to_approval(request_id: int, params: dict) -> None:
    match approval_policy:
        case "never":
            # Auto-accept
            send_response(request_id, {"decision": "accept"})
        case "always":
            # Auto-decline
            send_response(request_id, {"decision": "decline"})
        case "unlessTrusted":
            # Codex handles this via approvalPolicy on thread/start
            send_response(request_id, {"decision": "accept"})
```

### 11. Extract Results

**Text:** Use the final `agentMessage` item from `turn/completed`, or the
accumulated delta text.

**Tool events:** Map `item/completed` notifications to `ToolInvoked`:

| Codex Item Status | Action |
|-------------------|--------|
| `completed` | `ToolInvoked` with `success=True` |
| `failed` | `ToolInvoked` with `success=False` |
| `declined` | `ToolInvoked` with `success=False` |

**Deduplication:** Skip `ToolInvoked` for bridged WINK tools. When an
`mcpToolCall` item's `server` field identifies the WINK MCP server, skip it —
the `BridgedTool` already emitted the event.

### 12. Structured Output

If declared, retrieve from `structured_output` tool invocation.
Raise `PromptEvaluationError(phase="response")` if missing or invalid.

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
| `ContextWindowExceeded` | `PromptEvaluationError(phase="response")` |
| `UsageLimitExceeded` | `PromptEvaluationError(phase="budget")` |
| `HttpConnectionFailed` | `PromptEvaluationError(phase="request")` |
| `Unauthorized` | `PromptEvaluationError(phase="request")` |
| `BadRequest` | `PromptEvaluationError(phase="request")` |
| `SandboxError` | `PromptEvaluationError(phase="tool")` |
| `ResponseTooManyFailedAttempts` | `PromptEvaluationError(phase="request")` |
| Others | `PromptEvaluationError(phase="response")` |

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

The `thread/tokenUsage/updated` notification provides usage data during a turn.
Map to WINK's `TokenUsage`:

```python
@FrozenDataclass()
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

If Codex does not expose per-turn token breakdowns, emit `TokenUsage` with
available fields and zero for unknowns.

## Testing

### Unit Tests

- Mock Codex app-server (echo-style NDJSON over stdio)
- Verify `PromptRendered`, `PromptExecuted` emitted once
- Verify `ToolInvoked` for `commandExecution` and `fileChange` items
- Verify tool deduplication (no double events for bridged MCP tools)
- Verify structured output retrieval
- Verify approval auto-response per policy
- Verify `turn/interrupt` on deadline expiry
- Verify thread resume with session state
- Verify authentication flows (API key, external tokens)

### Integration Tests

Skip unless `codex` on PATH:

- Spawn `codex app-server` in temp workspace
- Simple prompt, verify response
- Tool invocation, verify `ToolInvoked`
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
    SandboxPolicy,
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
        sandbox_policy=SandboxPolicy(
            type="workspaceWrite",
            writable_roots=("/absolute/path/to/workspace",),
            network_access=True,
        ),
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
    SandboxPolicy,
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
        sandbox_policy=SandboxPolicy(
            type="workspaceWrite",
            writable_roots=(str(workspace.temp_dir),),
        ),
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
- ChatGPT browser OAuth flow — requires interactive browser, not suitable for
  programmatic usage
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

### Why Reuse Claude Agent SDK MCP Infrastructure

The MCP server infrastructure in `claude_agent_sdk/_bridge.py` provides:

- Transactional tool execution with snapshot/rollback
- Proper `ToolInvoked` event emission
- Visibility expansion signal handling
- Parameter parsing via `serde.parse()`

Reimplementing this would duplicate ~600 lines of battle-tested code. The
coupling is acceptable because both adapters share identical tool bridging
semantics.

### Why Auto-Accept Approvals by Default

WINK agents run programmatically without a human in the loop. Approval gates
are designed for interactive use (VS Code extension). For WINK:

- `"never"` (auto-accept) is the safe default for trusted workspaces
- Sandbox policy is the primary security boundary
- Callers can opt into `"always"` (auto-decline) for read-only analysis

## Related Specifications

- `specs/ADAPTERS.md` — Provider adapter protocol
- `specs/CLAUDE_AGENT_SDK.md` — Reference adapter architecture
- `specs/OPENCODE_ACP_ADAPTER.md` — Second adapter, similar patterns
- `specs/PROMPTS.md` — Prompt system
- `specs/SESSIONS.md` — Session state and events
- `specs/TOOLS.md` — Tool registration and policies
- `specs/WORKSPACE.md` — Workspace management

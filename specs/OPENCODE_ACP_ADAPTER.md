# OpenCode ACP Adapter Specification

> **Status:** Design specification — not yet implemented.
> The adapter package `src/weakincentives/adapters/opencode_acp/` does not exist
> yet. This document specifies the planned design. All code snippets are
> illustrative; they will be replaced with source file references once
> implementation lands.

> **Adapter name:** `opencode_acp`
> **OpenCode entrypoint:** `opencode acp`
> **ACP protocol:** v1 (JSON-RPC 2.0 over newline-delimited JSON on stdio)

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

## Requirements

### Runtime Dependencies

1. **OpenCode CLI** installed and available on `PATH` as `opencode`
1. **ACP Python SDK**: `agent-client-protocol>=0.7.1`
1. **Claude Agent SDK**: `claude-agent-sdk>=0.1.27` (for shared MCP server infrastructure)
1. WINK (`weakincentives`) runtime

### WINK Packaging

```toml
[project.optional-dependencies]
acp = [
  "agent-client-protocol>=0.7.1",
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
      ├─ Start in-process MCP server (reuses Claude SDK infrastructure)
      ├─ Spawn: opencode acp (stdio JSON-RPC NDJSON)
      ├─ ACP handshake: initialize → session/new
      ├─ session/prompt (text + MCP server config)
      ├─ Stream: session/update notifications
      │    ├─ agent_message_chunk (assistant output)
      │    ├─ tool_call / tool_call_update (WINK + OpenCode tools)
      │    └─ thoughts, plan, commands
      ├─ Model calls structured_output tool to finalize (if required)
      └─ Return PromptResponse(text, output)
```

## Module Structure

```
src/weakincentives/adapters/opencode_acp/
  __init__.py
  adapter.py            # OpenCodeACPAdapter
  config.py             # OpenCodeACPClientConfig, OpenCodeACPAdapterConfig
  client.py             # OpenCodeACPClient (ACP Client implementation)
  workspace.py          # OpenCodeWorkspaceSection
  _state.py             # OpenCodeACPSessionState slice for session reuse
  _events.py            # ACP updates → WINK ToolInvoked mapping
  _structured_output.py # structured_output MCP tool
  _async.py             # asyncio helpers
```

MCP tool bridging reuses `create_mcp_server()` from the shared adapter module
`src/weakincentives/adapters/_shared/_bridge.py` (re-exported by
`src/weakincentives/adapters/claude_agent_sdk/_bridge.py`).

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

> **Workspace gating:** If no `OpenCodeWorkspaceSection` is provided, the adapter
> must force `allow_file_reads=False` and `allow_file_writes=False` regardless of
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
| `model_id` | `str \| None` | `None` | ACP `session/set_model` (best-effort) |
| `quiet_period_ms` | `int` | `100` | Wait after prompt to drain trailing updates |
| `emit_thought_chunks` | `bool` | `False` | Include thoughts in returned text |

> **Model selection:** The ACP protocol defines an unstable `session/set_model`
> method. OpenCode **may not implement it**. The adapter attempts the call but
> treats any failure (including "method not found") as non-fatal.

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

## Workspace Management

### OpenCodeWorkspaceSection

Reuse the generic `WorkspaceSection` from `weakincentives.prompt.workspace`,
which consolidates the workspace types that were previously duplicated across
adapter modules.

The workspace section should:

- Accept `HostMount` tuples, `allowed_host_roots`, max-bytes budgets
- Materialize a temporary directory with copied files
- Expose `temp_dir` for `OpenCodeACPClientConfig.cwd`
- Render a provider-agnostic summary of mounts and budgets
- Expose cleanup via `.cleanup()` or a context manager
- Provide `workspace_fingerprint` for session reuse validation

**Workspace types** (from `weakincentives.prompt`):

| Type | Description |
|------|-------------|
| `WorkspaceSection` | Generic workspace section for all adapters |
| `HostMount` | Mount configuration (host_path, mount_path, globs, max_bytes) |
| `HostMountPreview` | Summary of materialized mount |
| `WorkspaceBudgetExceededError` | Mount exceeds byte budget |
| `WorkspaceSecurityError` | Mount violates security constraints |

## ACP Client Implementation

### OpenCodeACPClient

Implements `acp.interfaces.Client`:

| Method | Description |
|--------|-------------|
| `session_update(...)` | Feed updates to `SessionAccumulator` |
| `request_permission(...)` | Auto-respond per `permission_mode` (prompt -> deny) |
| `read_text_file(...)` | Read within workspace (if capability advertised) |
| `write_text_file(...)` | Write within workspace (if capability advertised) |
| `create_terminal(...)` | Return not-supported unless `allow_terminal=True` and implemented |

Use `acp.spawn_agent_process()` to spawn OpenCode and connect.

### SessionAccumulator

Use `acp.contrib.session_state.SessionAccumulator` to merge updates:

- Tracks tool calls and final merged state
- Records agent message chunks
- Optionally records thought chunks

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

After `conn.prompt()` returns:

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
5. conn.new_session(cwd=..., mcp_servers=[mcp_config])
6. OpenCode connects to MCP server
7. Tool call → in-process BridgedTool.__call__()
   ├─ Snapshot session state
   ├─ Execute handler
   ├─ Dispatch ToolInvoked
   └─ Rollback on failure
```

### BridgedTool Semantics

Each invocation:

1. **Snapshot** - Capture session and resource state
1. **Execute** - Call handler with parsed parameters
1. **Dispatch** - Emit `ToolInvoked` event
1. **Rollback** - Restore snapshot on failure

Handles: parameter parsing (`serde.parse()`), result formatting,
`VisibilityExpansionRequired` capture, deadline/budget enforcement.

### Visibility Expansion

When a tool raises `VisibilityExpansionRequired`:

1. `BridgedTool` catches and stores in `VisibilityExpansionSignal`
1. Returns non-error result explaining expansion need
1. After `conn.prompt()`, adapter checks signal
1. If set, re-raises to caller for re-render

## Execution Flow

### 1. Budget/Deadline Setup

- Create `BudgetTracker` if budget provided
- Derive deadline from argument or `budget.deadline`
- Raise `PromptEvaluationError(phase="request")` if already expired

### 2. Render Prompt

1. `prompt.render(session=session)` → `RenderedPrompt`
1. Emit `PromptRendered`

### 3. Start MCP Server

1. `create_bridged_tools(rendered.tools, adapter_name="opencode_acp", ...)`
1. Create `structured_output` tool if `output_type` declared
1. `create_mcp_server(all_tools)`

### 4. Spawn OpenCode

```python
from acp import spawn_agent_process
# opencode acp
```

Pass `cwd` via `session/new` or `session/load` only; do not rely on CLI flags.

### 5. Initialize + Session

```python
from acp import PROTOCOL_VERSION

conn.initialize(protocol_version=PROTOCOL_VERSION, client_capabilities={
    "fs": {"readTextFile": allow_file_reads, "writeTextFile": allow_file_writes},
    "terminal": allow_terminal,
})

if reuse_session and session[OpenCodeACPSessionState].latest():
    conn.load_session(session_id=..., cwd=..., mcp_servers=[...])
else:
    conn.new_session(cwd=..., mcp_servers=[...])
```

Store session ID, `cwd`, and `workspace_fingerprint` via
`session.seed(OpenCodeACPSessionState(...))`.

### 6. Set Mode/Model (Best-Effort)

- `session/set_mode` if `mode_id` configured (ignore errors)
- `session/set_model` if `model_id` configured (ignore errors)

### 7. Prompt

```python
conn.prompt(session_id=..., prompt=[TextContentBlock(text=rendered_text)])
```

`session_update()` captures streaming updates.

### 8. Drain Updates

Wait until no updates for `quiet_period_ms`; reset the timer on each update and
cap total wait by the deadline.

### 9. Extract Results

**Text:** Concatenate `agent_messages` chunks. Include thoughts if configured.

**Tool events:** Map ACP `tool_call_update` to `ToolInvoked`:

| ACP Status | Action |
|------------|--------|
| `completed` | `ToolInvoked` with `success=True` |
| `failed` | `ToolInvoked` with `success=False` |

**Deduplication:** Skip `ToolInvoked` for bridged WINK tools using explicit MCP
server metadata when available; fall back to the `mcp__wink__` prefix if needed.
`BridgedTool` already emitted the event.

### 10. Structured Output

If declared, retrieve from `structured_output` tool invocation.
Raise `PromptEvaluationError(phase="response")` if missing or invalid.

### 11. PromptExecuted

Emit event and return `PromptResponse(text=..., output=...)`.

## Cancellation

If deadline expires:

1. Send `conn.cancel(session_id=...)` — ACP defines this as a **notification** (no response)
1. Kill subprocess if needed
1. Raise `PromptEvaluationError(phase="request")` or `DeadlineExceededError`

## Error Handling

| Phase | When |
|-------|------|
| `"request"` | Spawn, initialize, session/new, or prompt fails |
| `"response"` | Structured output missing or invalid |

Include in payload: stderr tail (bounded, e.g., last 8k), ACP error details,
recent session updates (bounded).

Tool telemetry errors: log but don't crash.

## Testing

### Unit Tests

- Mock ACP agent (echo style)
- Verify `PromptRendered`, `PromptExecuted` emitted once
- Verify `ToolInvoked` on terminal status
- Verify tool deduplication (no double events)
- Verify structured output retrieval
- Verify `permission_mode="prompt"` denies without blocking
- Verify file read/write disabled when no workspace configured

### Integration Tests

Skip unless `opencode` on PATH:

- Spawn `opencode acp` in temp workspace
- Simple prompt, verify response
- Tool invocation, verify `ToolInvoked`

### Security Tests

- `read_text_file`/`write_text_file` reject paths outside workspace
- `allowed_host_roots` enforced

## Usage Example

```python
from weakincentives import Prompt, PromptTemplate, MarkdownSection
from weakincentives.runtime import Session, InProcessDispatcher
from weakincentives.adapters.opencode_acp import (
    OpenCodeACPAdapter,
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
    client_config=OpenCodeACPClientConfig(
        cwd="/absolute/path/to/workspace",
        permission_mode="auto",
        allow_file_reads=True,
        allow_file_writes=False,
    )
)

with prompt.resources:
    resp = adapter.evaluate(prompt, session=session)

print(resp.text)
```

## Non-Goals (v1)

- Token usage accounting (unless ACP exposes reliably)
- Full OpenCode config management (providers, keys)
- Perfect Claude sandboxing parity (workspace isolation is primary boundary)

## Related Specifications

- `specs/ADAPTERS.md` — Provider adapter protocol
- `specs/CLAUDE_AGENT_SDK.md` — Reference adapter architecture (shared bridge)
- `specs/CODEX_APP_SERVER.md` — Sibling adapter using shared bridge
- `specs/PROMPTS.md` — Prompt system
- `specs/SESSIONS.md` — Session state and events
- `specs/TOOLS.md` — Tool registration and policies

# OpenCode ACP Adapter Specification

> **Adapter name:** `opencode_acp`
> **OpenCode entrypoint:** `opencode acp`
> **ACP protocol:** v1 (JSON-RPC 2.0 over newline-delimited JSON on stdio)

## Purpose

`OpenCodeACPAdapter` evaluates prompts by delegating execution to **OpenCode** via
its **ACP** (Agent Client Protocol) server (`opencode acp`). Similar in architecture
to `ClaudeAgentSDKAdapter`:

- WINK handles **prompt composition**, **resource binding**, and **session telemetry**
- OpenCode handles **agentic execution** (planning, multi-step reasoning, tool calls, file edits)
- WINK receives streamed progress via ACP `session/update` notifications and emits canonical events:
  `PromptRendered`, `ToolInvoked`, `PromptExecuted`

This adapter targets "coding agent" workflows where you want OpenCode's behavior and
tools while retaining WINK's prompt system, session state, reducers, and orchestration.

**Implementation:** `src/weakincentives/adapters/opencode_acp/`

## Requirements

### Runtime Dependencies

1. **OpenCode CLI** installed and available on `PATH` as `opencode`
1. **ACP Python SDK**: `agent-client-protocol>=0.7.1`
1. WINK (`weakincentives`) runtime

### WINK Packaging

Add a new optional extra in `pyproject.toml`:

```toml
[project.optional-dependencies]
acp = [
  "agent-client-protocol>=0.7.1",
]
```

The adapter module uses lazy imports and raises a helpful error if the `acp` extra
is not installed (following the pattern in `openai.py` and `claude_agent_sdk/*`).

## Architecture

```
WINK Prompt/Session
  └─ OpenCodeACPAdapter.evaluate()
      ├─ Render WINK prompt (markdown)
      ├─ Spawn: `opencode acp`  (stdio JSON-RPC NDJSON)
      ├─ ACP handshake: initialize → session/new (or session/load)
      ├─ Optional: session/set_mode, session/set_model (best-effort)
      ├─ session/prompt (text + attachments)
      ├─ Stream in: session/update notifications
      │    ├─ agent_message_chunk (assistant output)
      │    ├─ tool_call / tool_call_update (native + MCP tools)
      │    └─ other (thoughts, plan, commands)
      └─ Return PromptResponse(text, output)
```

OpenCode runs its own internal SDK client when started via `opencode acp`.
The WINK adapter treats OpenCode as the "provider".

## Module Structure

```
src/weakincentives/adapters/opencode_acp/
  __init__.py
  adapter.py          # OpenCodeACPAdapter
  config.py           # OpenCodeACPClientConfig / OpenCodeACPAdapterConfig
  client.py           # OpenCodeACPClient (ACP Client implementation)
  workspace.py        # OpenCodeWorkspaceSection (generic mount logic)
  _state.py           # OpenCodeACPSessionState dataclass for session reuse
  _events.py          # Mapping ACP updates → WINK ToolInvoked
  _mcp.py             # MCP server process spawning
  _async.py           # asyncio/run helpers
```

MCP tool bridging reuses components from `src/weakincentives/adapters/claude_agent_sdk/_bridge.py`.

## Adapter Protocol

The adapter implements `ProviderAdapter` from `src/weakincentives/adapters/core.py`:

```python
class OpenCodeACPAdapter(ProviderAdapter[OutputT]):
    def evaluate(
        self,
        prompt: Prompt[OutputT],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
        heartbeat: Heartbeat | None = None,
        run_context: RunContext | None = None,
    ) -> PromptResponse[OutputT]: ...
```

### Semantics

- **OpenCode executes tools internally** for OpenCode-native tools.
- **WINK tools are exposed via MCP** and execute in a subprocess with WINK's
  transactional semantics (see MCP Tool Bridging section).
- WINK emits `ToolInvoked` events for telemetry (with deduplication rules below).
- **Structured output**: OpenCode ACP does not expose native JSON schema response format.
  The adapter enforces structured output via **prompt augmentation** and text parsing.

## Configuration

### OpenCodeACPClientConfig

Controls process spawn and ACP session parameters.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `opencode_bin` | `str` | `"opencode"` | Executable to spawn |
| `opencode_args` | `tuple[str, ...]` | `("acp",)` | Must include `acp` |
| `cwd` | `str \| None` | `None` | Working directory for ACP session (must be absolute if provided; defaults to `Path.cwd().resolve()`) |
| `env` | `Mapping[str, str] \| None` | `None` | Extra env for process |
| `suppress_stderr` | `bool` | `True` | Capture/stash stderr for debug |
| `startup_timeout_s` | `float` | `10.0` | Max time for initialize/session/new |
| `permission_mode` | `Literal["auto", "deny", "prompt"]` | `"auto"` | How to respond to `session/request_permission` |
| `allow_file_reads` | `bool` | `True` | Advertise `readTextFile` capability |
| `allow_file_writes` | `bool` | `False` | Advertise `writeTextFile` capability (conservative default) |
| `allow_terminal` | `bool` | `False` | Advertise terminal capability |
| `mcp_servers` | `tuple[McpServerConfig, ...]` | `()` | Extra MCP servers to register |
| `reuse_session` | `bool` | `False` | Whether to load/reuse an OpenCode session id |

> **Note:** ACP requires `cwd` to be an **absolute path**. If `cwd` is `None`, the
> adapter resolves it to `Path.cwd().resolve()` before passing to `session/new`.

### OpenCodeACPAdapterConfig

Controls adapter-level behavior (not OpenCode's LLM settings).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode_id` | `str \| None` | `None` | ACP `session/set_mode` after session creation |
| `model_id` | `str \| None` | `None` | ACP `session/set_model` (best-effort; see note) |
| `require_structured_output_text` | `bool` | `True` | For WINK structured output parsing |
| `quiet_period_ms` | `int` | `100` | Wait after prompt returns to drain trailing updates |
| `emit_thought_chunks` | `bool` | `False` | Whether to include thought chunks in returned text |

> **Note:** The ACP protocol defines an unstable `session/set_model` method. OpenCode
> **may or may not implement it**. The adapter attempts to call it if `model_id` is
> set, but treats any failure (including "method not found") as non-fatal.

### Session Reuse Storage

WINK sessions use **typed dataclass slices**, not string keys. For session reuse,
define a dataclass in `_state.py`:

```python
from weakincentives.dataclasses import FrozenDataclass

@FrozenDataclass()
class OpenCodeACPSessionState:
    """Stores OpenCode session ID for reuse across adapter calls."""
    session_id: str
```

Storage and retrieval:

```python
# Store after session/new
session.seed(OpenCodeACPSessionState(session_id=result.session_id))

# Retrieve for session/load
state = session[OpenCodeACPSessionState].latest()
if state is not None:
    opencode_session_id = state.session_id
```

## Workspace Management

OpenCode runs locally and reads/writes files under its session `cwd`. For production
patterns (multi-tenant, safety, reproducibility), use an isolated workspace directory.

### OpenCodeWorkspaceSection

Create `OpenCodeWorkspaceSection` by **extracting** the generic filesystem-mount
machinery from `src/weakincentives/adapters/claude_agent_sdk/workspace.py`:

- Accepts `HostMount` tuples, `allowed_host_roots`, and max-bytes budgets
- Materializes a temporary directory with copied files
- Exposes `temp_dir` path for use as `OpenCodeACPClientConfig.cwd`

> **Important:** Do **not** reuse `ClaudeAgentWorkspaceSection` directly—its rendered
> template text contains Claude-specific wording ("Claude Code provides direct access…").
> Extract the mount/copy logic into a shared helper and create `OpenCodeWorkspaceSection`
> with neutral or OpenCode-appropriate wording.

**Key types (shared with Claude adapter):**

| Type | Description |
|------|-------------|
| `HostMount` | Configuration for mounting host files into workspace |
| `HostMountPreview` | Summary of a materialized mount |
| `WorkspaceBudgetExceededError` | Mount exceeds byte budget |
| `WorkspaceSecurityError` | Mount violates security constraints |

## ACP Client Implementation

The ACP process is the *agent*; WINK implements an ACP *client*.

### OpenCodeACPClient

Implement `acp.interfaces.Client` and provide:

| Method | Description |
|--------|-------------|
| `session_update(...)` | Capture streamed updates; feed a `SessionAccumulator` |
| `request_permission(...)` | Respond automatically based on `permission_mode` |
| `read_text_file(...)` | Operate within workspace root (if `allow_file_reads=True`) |
| `write_text_file(...)` | Operate within workspace root (if `allow_file_writes=True`) |
| `create_terminal(...)` | Stub unless `allow_terminal=True` |

> **Capability alignment:** The capabilities advertised in `initialize` must match
> what you actually implement. If you advertise `readTextFile=True`, you must
> implement the method and enforce workspace boundary correctly.

Use the ACP SDK's `spawn_agent_process()` to spawn OpenCode and connect.

### SessionAccumulator

Use `acp.contrib.session_state.SessionAccumulator` to merge `SessionNotification`
updates into a snapshot:

- Tracks tool calls and their final merged state
- Records agent message chunks (assistant output)
- Optionally records thought chunks

## Prompt Translation

### Base Prompt

Render the WINK prompt via the standard pipeline:

1. Use `prepare_adapter_conversation(...)` from `src/weakincentives/adapters/rendering.py`
   to get `AdapterRenderContext` with `rendered.text` and `render_inputs`
1. Emit `PromptRendered` immediately after render
1. Send the prompt to OpenCode as a single ACP `TextContentBlock`

### Structured Output

If the prompt declares structured output (`rendered.output_type is not None`):

**1. Generate schema** using WINK's existing logic to ensure compatibility:

```python
from weakincentives.adapters.response_parser import build_json_schema_response_format

# This handles:
# - Array container wrapping (rendered.container == "array")
# - Extra keys policy (rendered.allow_extra_keys)
# - Proper schema generation via serde.schema()
schema_format = build_json_schema_response_format(rendered, prompt_name)
json_schema = schema_format["json_schema"]["schema"]
```

**2. Append output contract** to the rendered markdown:

````markdown
## Output Format (MANDATORY)

Return ONLY valid JSON in a ```json``` fenced block.

Schema:
```json
{ ... extracted schema ... }
```
````

**3. Parse after completion** using WINK's structured output parser:

```python
from weakincentives.prompt.structured_output import parse_structured_output

parsed = parse_structured_output(text, rendered)
```

On parse failure, raise `PromptEvaluationError(phase="response")`.

### Attachments / Resources (Optional)

OpenCode's ACP agent supports `resource_link` and embedded `resource` blocks.
Map WINK file resources to ACP blocks conservatively:

- Local files in workspace: `ResourceContentBlock(type="resource_link", uri="file:///...")`
- Embedded small text: `EmbeddedResourceContentBlock(type="resource", resource={...text...})`

Use the workspace root; avoid leaking host paths when using isolation.

## Execution Flow

### 1. Budget/Deadline Setup

- If `budget` provided without `budget_tracker`, create one
- Derive effective deadline from `deadline` argument or `budget.deadline`
- If deadline already expired, raise `PromptEvaluationError(phase="request")`

### 2. Render + PromptRendered Event

Render once. Emit `PromptRendered` (at `src/weakincentives/runtime/events/__init__.py`) with:

| Field | Type | Value |
|-------|------|-------|
| `prompt_ns` | `str` | Prompt namespace |
| `prompt_key` | `str` | Prompt key |
| `prompt_name` | `str \| None` | Full prompt name (may be None) |
| `adapter` | `AdapterName` | `"opencode_acp"` |
| `session_id` | `UUID \| None` | Session identifier |
| `render_inputs` | `tuple[Any, ...]` | Prompt params |
| `rendered_prompt` | `str` | Final text (including output contract) |
| `created_at` | `datetime` | Timestamp |

### 3. Spawn OpenCode ACP

```python
from acp import spawn_agent_process
```

Spawn command:

- `opencode_bin` + `opencode_args` (must include `"acp"`)
- Pass `--cwd <cwd>` as an arg
- Set subprocess working directory to `cwd` for consistency

Capture stderr if `suppress_stderr=True` for debugging/tracing.

### 4. Initialize + Session

- Call `conn.initialize(protocol_version=1, client_capabilities=...)`
  - fs: `readTextFile` and `writeTextFile` per config
  - terminal: only if `allow_terminal=True`

Then:

- If `reuse_session=True` and WINK session contains stored `OpenCodeACPSessionState`:
  - Attempt `conn.load_session(cwd=cwd, mcp_servers=[...], session_id=...)`
  - Fall back to `conn.new_session(...)` on failure
- Else call `conn.new_session(cwd=cwd, mcp_servers=[...])`

Store returned `sessionId` via `session.seed(OpenCodeACPSessionState(...))` if
`reuse_session=True`.

### 5. Set Mode/Model (Best-Effort)

If `adapter_config.mode_id` is set:

- Call `conn.set_session_mode(mode_id=..., session_id=...)`
- Ignore `method_not_found` / request errors (non-fatal)

If `adapter_config.model_id` is set:

- Attempt `conn.set_session_model(model_id=..., session_id=...)`
- Treat **any failure** as non-fatal (OpenCode may not implement this method)

### 6. Prompt

Call `conn.prompt(session_id=..., prompt=[text_block(prompt_text), ...attachments])`

While the request is in flight, OpenCode streams `session/update` notifications.
`OpenCodeACPClient.session_update()` captures them.

### 7. Drain Trailing Updates

After `conn.prompt()` returns:

- Wait until no new session updates for `quiet_period_ms` (default 100ms)
- Cap wait by remaining deadline

Implementation: track `last_update_monotonic`, loop with short sleeps until quiet.

### 8. Extract Result Text and Tool Events

**Text:**
Concatenate the accumulator's `agent_messages` chunks in arrival order.
Optionally include thought chunks if `emit_thought_chunks=True`.

**Tool events → WINK ToolInvoked:**

OpenCode emits:

- `tool_call` (start): `toolCallId`, `title`, `status="pending"`, `rawInput`
- `tool_call_update` (progress/completion): updated status and `rawOutput`

Mapping rules:

| ACP Status | WINK Action |
|------------|-------------|
| `"completed"` | Emit `ToolInvoked` with `ToolResult(success=True, ...)` |
| `"failed"` | Emit `ToolInvoked` with `ToolResult(success=False, ...)` |

**Deduplication rule:** If the tool name indicates it's a bridged WINK tool
(e.g., prefix `mcp__wink__`), do **not** emit `ToolInvoked` from ACP updates—
the `BridgedTool` execution already dispatched a richer event. Only emit
ACP-derived telemetry for OpenCode-native tools.

`ToolInvoked` fields (at `src/weakincentives/runtime/events/types.py`):

| Field | Source |
|-------|--------|
| `name` | Tool title or canonical name |
| `params` | `rawInput` as dict |
| `call_id` | `toolCallId` |
| `rendered_output` | Best-effort string from ACP tool content |

> **Note:** This is telemetry. WINK is not responsible for tool correctness—OpenCode owns execution.

### 9. Structured Output Parse

If prompt declares structured output, parse from `result_text`:

- Success → `PromptResponse(output=parsed)`
- Failure → raise `PromptEvaluationError(phase="response")` with raw text in `provider_payload`

### 10. PromptExecuted Event

Emit `PromptExecuted` (at `src/weakincentives/runtime/events/__init__.py`) with:

| Field | Value |
|-------|-------|
| `result` | `PromptResponse` |
| `usage` | `None` (unless reliable usage source available) |
| `run_context` | Pass through if provided |

Return:

```python
PromptResponse(prompt_name=..., text=result_text, output=parsed_or_none)
```

## MCP Tool Bridging

MCP tool bridging is **required** for exposing WINK tools to OpenCode. OpenCode
supports MCP servers via ACP `session/new` `mcpServers` parameter.

### ACP MCP Server Model

ACP's `mcpServers` configuration describes **server specs** that the **agent spawns**:

```python
McpServerStdio(
    name="wink",
    command="python",
    args=["-m", "weakincentives.adapters.opencode_acp.mcp_server", ...],
    env={...},
)
```

The adapter cannot pass already-open file descriptors or in-process pipes to OpenCode.
OpenCode spawns the MCP server as a subprocess and connects to it via stdio.

### Reusing Claude Agent SDK Bridge Components

The MCP server subprocess reuses tool bridging infrastructure from the Claude Agent SDK:

| Component | Location | Purpose |
|-----------|----------|---------|
| `BridgedTool` | `claude_agent_sdk/_bridge.py` | Wraps WINK tools with transactional semantics |
| `create_bridged_tools()` | `claude_agent_sdk/_bridge.py` | Factory for creating BridgedTool instances |
| `VisibilityExpansionSignal` | `claude_agent_sdk/_visibility_signal.py` | Thread-safe exception propagation |
| `tool_transaction()` | `runtime/transactions.py` | Atomic snapshot/restore for tool calls |

> **Important:** When calling `create_bridged_tools(...)`, pass `adapter_name="opencode_acp"`
> to ensure `ToolInvoked` events are labeled correctly (the function defaults to
> `"claude_agent_sdk"`).

### MCP Server Subprocess

The adapter spawns a dedicated MCP server process that OpenCode connects to:

```
src/weakincentives/adapters/opencode_acp/mcp_server.py
```

This process:

1. Receives tool definitions via command-line args or a rendezvous file
1. Registers tools using the MCP protocol (stdio JSON-RPC)
1. Executes tools using `BridgedTool` semantics
1. Communicates `VisibilityExpansionRequired` signals back to the parent adapter

### Tool Bridging Flow

```
1. Adapter renders prompt → rendered.tools
2. Serialize tool definitions to a rendezvous file or args
3. Configure MCP server spec:
   McpServerStdio(
       name="wink",
       command=sys.executable,
       args=["-m", "weakincentives.adapters.opencode_acp.mcp_server",
             "--tools-file", rendezvous_path, ...],
   )
4. Pass config to ACP session/new:
   conn.new_session(cwd=..., mcp_servers=[mcp_config])
5. OpenCode spawns MCP server subprocess
6. OpenCode calls MCP tool → mcp_server.py handles it:
   ├─ Deserialize tool definition
   ├─ Create BridgedTool with adapter_name="opencode_acp"
   ├─ Execute with transactional semantics
   └─ Return result via MCP protocol
7. Adapter receives tool_call updates via ACP session/update
```

### BridgedTool Execution Semantics

Each `BridgedTool` invocation (from `claude_agent_sdk/_bridge.py`):

1. **Snapshot** - Capture session state and resource context before execution
1. **Execute** - Call the WINK tool handler with parsed parameters
1. **Dispatch** - Emit `ToolInvoked` event with result/error
1. **Rollback** - On failure, restore snapshot (transactional semantics)

The same `BridgedTool` class handles:

- Parameter parsing via `serde.parse()`
- Result formatting for MCP response
- `VisibilityExpansionRequired` exception capture
- Deadline and budget enforcement

### Visibility Expansion

When a bridged tool raises `VisibilityExpansionRequired`:

1. `BridgedTool` catches the exception in the MCP server subprocess
1. Writes signal to a shared file or returns a special marker in the result
1. After `conn.prompt()` completes, adapter checks for the signal
1. If signal set, re-raises `VisibilityExpansionRequired` to caller

This matches the Claude Agent SDK adapter's behavior.

### Session State Synchronization

Since the MCP server runs in a **separate process**, it cannot directly access
the WINK session state. Options for v1:

1. **Stateless tools only**: Tools that don't require session state work out of the box
1. **Snapshot serialization**: Serialize session snapshot to the rendezvous file;
   MCP server restores and uses it (complex, deferred to v2)

For v1, recommend stateless tools or tools that use external storage (files, databases).

## Cancellation and Deadlines

If `effective_deadline` expires while waiting:

- Send `conn.cancel(session_id=...)` (ACP `session/cancel`)
- Kill subprocess if it doesn't exit quickly
- Raise `PromptEvaluationError(phase="request")` or `DeadlineExceededError`

> **Note:** ACP defines `session/cancel` as a **notification** (no response expected).
> The implementation should not wait for a reply.

## Error Handling

Wrap failures as `PromptEvaluationError` (at `src/weakincentives/adapters/core.py`)
with accurate `phase`:

| Phase | When |
|-------|------|
| `"request"` | Spawn / initialize / session/new / prompt request fails |
| `"response"` | Cannot parse structured output |

Include provider payload when useful:

- stderr tail
- ACP RequestError details
- Last N session updates (optional, size-limited)

Tool telemetry mapping errors should be logged but not crash the run.

## Testing

### Unit Tests (No OpenCode Required)

1. Use ACP Python SDK's `examples/echo_agent.py` style agent to simulate updates
1. Verify:
   - `PromptRendered` emitted once
   - `PromptExecuted` emitted once
   - Tool updates produce `ToolInvoked` when terminal status observed
   - Structured output parsing works when agent returns JSON-in-fence
   - Tool event deduplication (bridged tools not double-logged)

### Integration Tests (OpenCode Optional)

Mark as skipped unless `opencode` is on PATH:

- Spawn `opencode acp` in temp workspace
- Send simple prompt ("say hi")
- Verify at least one `agent_message_chunk` and returned text
- Optional: ask OpenCode to run read-only tool, assert `ToolInvoked` emitted

### Security Tests

- Ensure `read_text_file` / `write_text_file` reject paths outside workspace root
- Ensure `allowed_host_roots` enforced when mounting

## Usage Example

```python
from weakincentives import Prompt, PromptTemplate, MarkdownSection
from weakincentives.runtime import Session, InProcessDispatcher
from weakincentives.adapters.opencode_acp import (
    OpenCodeACPAdapter,
    OpenCodeACPClientConfig,
)

bus = InProcessDispatcher()
session = Session(bus=bus)

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

# Optional workspace isolation (recommended)
# with OpenCodeWorkspaceSection(...) as ws:
#     client_config = OpenCodeACPClientConfig(cwd=str(ws.temp_dir), ...)

adapter = OpenCodeACPAdapter(
    client_config=OpenCodeACPClientConfig(
        cwd="/abs/path/to/workspace",  # Must be absolute
        permission_mode="auto",
        allow_file_writes=False,  # safe default for exploration
    )
)

with prompt.resources:
    resp = adapter.evaluate(prompt, session=session)

print(resp.text)
```

## Non-Goals (v1)

- Accurate token usage accounting (unless OpenCode/ACP exposes it reliably)
- Full OpenCode configuration management (providers, keys, etc.)—defer to OpenCode's own config
- Perfect parity with Claude's sandboxing model—workspace isolation is the primary safety boundary
- Stateful tools in MCP subprocess (deferred to v2 with session serialization)

## Related Specifications

- `specs/ADAPTERS.md` - Provider adapter protocol and lifecycle
- `specs/CLAUDE_AGENT_SDK.md` - Reference adapter with similar architecture
- `specs/PROMPTS.md` - Prompt system and rendering
- `specs/SESSIONS.md` - Session state and events
- `specs/TOOLS.md` - Tool registration and policies

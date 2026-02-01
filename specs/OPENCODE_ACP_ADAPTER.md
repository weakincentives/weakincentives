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

## Architecture

```
WINK Prompt/Session
  └─ OpenCodeACPAdapter.evaluate()
      ├─ Render WINK prompt (markdown)
      ├─ Spawn: `opencode acp`  (stdio JSON-RPC NDJSON)
      ├─ ACP handshake: initialize → session/new (or session/load)
      ├─ Optional: session/set_mode, session/set_model
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
  workspace.py        # OpenCodeWorkspaceSection
  _events.py          # Mapping ACP updates → WINK ToolInvoked
  _async.py           # asyncio/run helpers
```

Optional MCP bridge (for exposing WINK tools to OpenCode):

```
src/weakincentives/mcp/
  server.py           # stdio MCP server: exposes WINK tools as MCP tools
  bridge.py           # Tool event semantics bridge
```

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

- **Tools are not executed by WINK** for this adapter. OpenCode executes tools internally.
- WINK emits `ToolInvoked` events derived from ACP tool-call updates (telemetry only).
- **Structured output**: OpenCode ACP does not expose native JSON schema response format.
  The adapter enforces structured output via **prompt augmentation** and text parsing.

## Configuration

### OpenCodeACPClientConfig

Controls process spawn and ACP session parameters.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `opencode_bin` | `str` | `"opencode"` | Executable to spawn |
| `opencode_args` | `tuple[str, ...]` | `("acp",)` | Must include `acp` |
| `cwd` | `str \| None` | `None` | Working directory for ACP session |
| `env` | `Mapping[str, str] \| None` | `None` | Extra env for process |
| `suppress_stderr` | `bool` | `True` | Capture/stash stderr for debug |
| `startup_timeout_s` | `float` | `10.0` | Max time for initialize/session/new |
| `permission_mode` | `Literal["auto", "deny", "prompt"]` | `"auto"` | How to respond to `session/request_permission` |
| `allow_file_writes` | `bool` | `True` | Controls ACP fs/write capability |
| `allow_terminal` | `bool` | `False` | Advertise terminal capability |
| `mcp_servers` | `tuple[McpServerSpec, ...]` | `()` | Extra MCP servers to register |
| `reuse_session` | `bool` | `False` | Whether to load/reuse an OpenCode session id |
| `session_id_key` | `str` | `"opencode_session_id"` | Session slice key for storing session id |

### OpenCodeACPAdapterConfig

Controls adapter-level behavior (not OpenCode's LLM settings).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode_id` | `str \| None` | `None` | ACP `session/set_mode` after session creation |
| `model_id` | `str \| None` | `None` | ACP `session/set_model` (best-effort, unstable in ACP) |
| `require_structured_output_text` | `bool` | `True` | For WINK structured output parsing |
| `quiet_period_ms` | `int` | `100` | Wait after prompt returns to drain trailing updates |
| `emit_thought_chunks` | `bool` | `False` | Whether to include thought chunks in returned text |

> **Note:** OpenCode's ACP agent implements `unstable_setSessionModel`. Treat model
> switching as best-effort and non-fatal.

## Workspace Management

OpenCode runs locally and reads/writes files under its session `cwd`. For production
patterns (multi-tenant, safety, reproducibility), use an isolated workspace directory.

### OpenCodeWorkspaceSection

Reuse the mount/copy logic from `src/weakincentives/adapters/claude_agent_sdk/workspace.py`:

- Accepts `HostMount` tuples, `allowed_host_roots`, and max-bytes budgets
- Materializes a temporary directory with copied files
- Exposes `temp_dir` path for use as `OpenCodeACPClientConfig.cwd`

The existing `ClaudeAgentWorkspaceSection` can be reused directly (it's not
Claude-specific in behavior), or wrap it with a new name to avoid confusion.

**Key types from workspace module:**

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
| `read_text_file(...)` | Operate within workspace root |
| `write_text_file(...)` | Operate within workspace root (if `allow_file_writes=True`) |
| `create_terminal(...)` | Stub unless `allow_terminal=True` |

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

**1. Append output contract** to the rendered markdown before sending.

Generate JSON Schema using `serde.schema()` from `src/weakincentives/serde/schema.py`:

```python
from weakincentives.serde import schema

json_schema = schema(
    rendered.output_type,
    scope=SerdeScope.STRUCTURED_OUTPUT,
)
```

Append a contract block:

````markdown
## Output Format (MANDATORY)
Return ONLY valid JSON in a ```json``` fenced block.

Schema:
```json
{ ... }
````

````

**2. Parse after completion** using WINK's structured output parser:

```python
from weakincentives.prompt.structured_output import parse_structured_output

parsed = parse_structured_output(text, rendered)
````

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
  - fs: `readTextFile=True`, `writeTextFile=True` (if `allow_file_writes=True`)
  - terminal: only if `allow_terminal=True`

Then:

- If `reuse_session=True` and WINK session contains stored OpenCode `session_id`:
  - Attempt `conn.load_session(cwd=cwd, mcp_servers=[...], session_id=...)`
  - Fall back to `conn.new_session(...)` on failure
- Else call `conn.new_session(cwd=cwd, mcp_servers=[...])`

Persist returned `sessionId` if `reuse_session=True`.

### 5. Set Mode/Model (Best-Effort)

If `adapter_config.mode_id` is set:

- Call `conn.set_session_mode(mode_id=..., session_id=...)`
- Ignore `method_not_found` / request errors (non-fatal)

If `adapter_config.model_id` is set:

- Call `conn.set_session_model(model_id=..., session_id=...)`
- Treat failures as non-fatal (ACP marks this unstable)

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

## MCP Tool Bridging (Optional)

OpenCode supports MCP servers provided at ACP `session/new` as `mcpServers`.

### Goal

Expose WINK tools (Python) to OpenCode as MCP tools, allowing OpenCode to call
them during its agent loop.

### Design

1. Implement MCP server process in WINK (stdio):

   - CLI: `python -m weakincentives.mcp.server`
   - Reads/writes MCP JSON-RPC over stdio
   - Registers tools from the rendered prompt's tool set

1. In `OpenCodeACPAdapter.evaluate()`:

   - Build bridged tool registry (reuse semantics from `claude_agent_sdk/_bridge.py`)
   - Start MCP server process
   - Include ACP `McpServerStdio` entry in `mcp_servers`:
     - `name="wink"`
     - `command=sys.executable`
     - `args=["-m", "weakincentives.mcp.server", "--session", <id>, ...]`

1. Tool call telemetry:

   - OpenCode emits ACP `tool_call` / `tool_call_update` for MCP tools
   - Emit `ToolInvoked` as usual
   - For bridged tools, enforce WINK transactional semantics inside MCP server

### Visibility Expansion (Optional)

If bridged WINK tools raise `VisibilityExpansionRequired`, replicate the Claude
adapter's pattern:

- MCP server catches `VisibilityExpansionRequired`
- Returns non-error tool result explaining the expansion
- Signals back to adapter (shared file, env var, socket)
- Adapter raises exception after OpenCode completes, so caller can re-render

## Cancellation and Deadlines

If `effective_deadline` expires while waiting:

- Call `conn.cancel(session_id=...)` (ACP `session/cancel`)
- Kill subprocess if it doesn't exit quickly
- Raise `PromptEvaluationError(phase="request")` or `DeadlineExceededError`

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
        cwd="/abs/path/to/workspace",
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

## Related Specifications

- `specs/ADAPTERS.md` - Provider adapter protocol and lifecycle
- `specs/CLAUDE_AGENT_SDK.md` - Reference adapter with similar architecture
- `specs/PROMPTS.md` - Prompt system and rendering
- `specs/SESSIONS.md` - Session state and events
- `specs/TOOLS.md` - Tool registration and policies

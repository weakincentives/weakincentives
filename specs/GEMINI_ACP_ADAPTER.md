# Gemini CLI ACP Adapter Specification

> **Status:** Proposed.
> **Package:** `src/weakincentives/adapters/gemini_acp/` (planned)
> **Adapter name:** `gemini_acp`
> **Base class:** `ACPAdapter` from `src/weakincentives/adapters/acp/`
> **Gemini entrypoint:** `gemini --experimental-acp`
> **Validated against:** `gemini 0.29.5` with `agent-client-protocol 0.8.0`

## Purpose

`GeminiACPAdapter` is a thin wrapper around the generic `ACPAdapter` that
provides Gemini CLI-specific defaults and quirk handling. The two-layer
architecture separates protocol concerns (generic `acp/` package) from
agent-specific behavior (this package).

Gemini CLI is notable as the **reference ACP implementation** from Google,
one of the primary companies behind the Agent Client Protocol. Its wire
format aligns closely with the `acp` Python SDK's Pydantic models.

See `specs/ACP_ADAPTER.md` for the generic ACP protocol implementation.

## Module Structure

```
src/weakincentives/adapters/gemini_acp/
  __init__.py          # Public exports (re-exports from acp/ + Gemini classes)
  adapter.py           # GeminiACPAdapter(ACPAdapter)
  config.py            # GeminiACPClientConfig, GeminiACPAdapterConfig
```

No ephemeral home module is needed for v1. Gemini does not support a
skill discovery path like OpenCode's `$HOME/.claude/skills/`. If skill
mounting is needed later, a `_ephemeral_home.py` can be added following
the OpenCode pattern.

## Configuration

### GeminiACPClientConfig

Extends `ACPClientConfig` with Gemini defaults:

| Field | Override | Description |
|-------|----------|-------------|
| `agent_bin` | `"gemini"` | Gemini CLI binary |
| `agent_args` | `("--experimental-acp",)` | ACP mode flag |
| `permission_mode` | `"auto"` | Auto-approve (Gemini does not send permission requests) |
| `startup_timeout_s` | `15.0` | Gemini startup is slower than OpenCode |

All other fields inherit from `ACPClientConfig`.

> **Model selection:** Gemini does not support `session/setModel`. The model
> must be passed as a CLI flag at spawn time. When `ACPAdapterConfig.model_id`
> is set, the adapter prepends `--model {model_id}` to `agent_args`. See
> [Model Selection](#model-selection).

### GeminiACPAdapterConfig

Extends `ACPAdapterConfig` with Gemini defaults:

| Field | Override | Description |
|-------|----------|-------------|
| `quiet_period_ms` | `200` | Gemini delivers all updates before prompt response; lower value is safe |
| `emit_thought_chunks` | `True` | Gemini emits rich reasoning by default |
| `approval_mode` | `str \| None` | CLI `--approval-mode` flag: `"default"`, `"auto_edit"`, `"yolo"`, `"plan"` |
| `sandbox` | `False` | Enable OS-level sandboxing (`--sandbox` flag) |
| `sandbox_profile` | `str \| None` | Seatbelt profile name (macOS); set via `SEATBELT_PROFILE` env var |

## Subclass Hook Overrides

### `_adapter_name()`

Returns `"gemini_acp"` (the `GEMINI_ACP_ADAPTER_NAME` constant).

### `_validate_model(model_id, available_models)`

**No-op.** Gemini's `session/new` does not return available models. Model
validation cannot happen at the protocol level. Invalid models are detected
via empty response detection instead.

### `_detect_empty_response(client, prompt_resp)`

Raises `PromptEvaluationError(phase="response")` if:

- Zero `AgentMessageChunk` updates were received, AND
- `prompt_resp.stop_reason` is `None` or missing

This catches invalid model IDs and other silent failures. When Gemini
receives an invalid model via `--model`, it returns immediately with no
content and `stopReason: null`.

### `_agent_spawn_args()`

Constructs CLI arguments dynamically:

1. Start with `("--experimental-acp",)`
1. If `model_id` is set, append `("--model", model_id)`
1. If `approval_mode` is set, append `("--approval-mode", approval_mode)`
1. If `sandbox` is `True`, append `"--sandbox"`

### `_prepare_execution_env()`

When `sandbox_profile` is set, injects the `SEATBELT_PROFILE` environment
variable into the subprocess environment. This selects which macOS seatbelt
profile Gemini uses when `--sandbox` is active.

Returns the base environment (no ephemeral home needed for v1).

## Gemini-Specific Behaviors

### Model Selection

Gemini does not implement `session/setModel` (returns `-32601 Method not found`). Model selection happens exclusively via the `--model` CLI flag at
process spawn time.

| Approach | Status |
|----------|--------|
| `session/setModel` | Not supported (-32601) |
| `--model` CLI flag | Works |
| Default (no flag) | Uses gemini-2.5-pro |

**Observed models:**

| Model ID | Notes |
|----------|-------|
| `gemini-2.5-pro` | Default; most capable |
| `gemini-2.5-flash` | Faster, lower latency |

Model availability depends on auth method and account tier.

### Mode Selection

Gemini does not implement `session/setMode` (returns `-32601`). The
`--approval-mode` CLI flag controls execution behavior:

| Approval Mode | Description |
|---------------|-------------|
| `default` | Prompt for approval on actions |
| `auto_edit` | Auto-approve edit tools |
| `yolo` | Auto-approve all tools |
| `plan` | Read-only mode |

For non-interactive WINK execution, `yolo` is the recommended default.

### Sandboxing

> **Known Issue (v0.29.5):** `--sandbox` and `--experimental-acp` are
> **incompatible**. The sandbox re-launches Gemini via `sandbox-exec`
> (macOS) or a container (Linux), which breaks ACP's stdio pipe protocol.
> All sandbox profiles — including `permissive-open` — cause the ACP
> handshake to time out. The config fields (`sandbox`, `sandbox_profile`)
> are retained for documentation and forward-compatibility, but **cannot
> be used with the ACP adapter** until upstream Gemini CLI fixes the
> interaction. See [Sandbox + ACP Incompatibility](#sandbox--acp-incompatibility).

Gemini CLI supports OS-level sandboxing via the `--sandbox` flag. The
sandbox method depends on the platform:

| Platform | Method | Enforcement |
|----------|--------|-------------|
| macOS | Seatbelt (`sandbox-exec`) | Kernel-level via `.sb` profiles |
| Linux | Docker/Podman container | Container isolation |

#### Sandbox + ACP Incompatibility

When `--sandbox` is active, Gemini's entry point (`gemini.js`) spawns
`sandbox-exec -f <profile>.sb sh -c "SANDBOX=sandbox-exec gemini ..."`.
The child Gemini process detects the `SANDBOX` env var and skips
re-sandboxing. However, the double-process indirection (`sandbox-exec` →
`sh` → `gemini`) breaks the stdio pipe that ACP relies on for JSON-RPC
communication.

**Experimentally verified** (2026-02-22, Gemini v0.29.5):

| Configuration | Result |
|---------------|--------|
| `--experimental-acp` (no sandbox) | PASS — responds correctly |
| `--sandbox --experimental-acp` (`permissive-open`) | FAIL — ACP handshake timeout |
| `--sandbox --experimental-acp` (default profile) | FAIL — ACP handshake timeout |
| `--sandbox --experimental-acp` (`permissive-closed`, 60s timeout) | FAIL — ACP handshake timeout |

**Root cause:** `sandbox-exec` spawns with `stdio: 'inherit'`, but the
pipe ownership passes through the intermediate `sh` process, disrupting
the byte-level framing that JSON-RPC over stdio requires.

**Workaround:** None for ACP mode. For non-ACP interactive usage,
`--sandbox` works as documented below. For ACP-based WINK execution, rely
on `--approval-mode` for agent-level restrictions and `--allowed-tools`
to limit tool access.

#### Seatbelt Profiles (macOS)

When `--sandbox` is active on macOS (non-ACP mode), Gemini uses seatbelt
profiles to restrict file system and network access. The profile is
selected via the `SEATBELT_PROFILE` environment variable. Six built-in
profiles ship with Gemini CLI (validated against v0.29.5):

| Profile | Default policy | File Reads | File Writes | Network Out |
|---------|---------------|------------|-------------|-------------|
| `permissive-open` | `(allow default)` | Unrestricted | Workspace + tmp + cache | Unrestricted |
| `permissive-proxied` | `(allow default)` | Unrestricted | Workspace + tmp + cache | Proxy only (localhost:8877) |
| `permissive-closed` | `(allow default)` | Unrestricted | Workspace + tmp + cache | **Blocked** |
| `restrictive-open` | `(deny default)` | `(allow file-read*)` | Workspace + tmp + cache | Unrestricted |
| `restrictive-proxied` | `(deny default)` | `(allow file-read*)` | Workspace + tmp + cache | Proxy only |
| `restrictive-closed` | `(deny default)` | `(allow file-read*)` | Workspace + tmp + cache | **Blocked** |

**Permissive** profiles: `(allow default)` then selectively deny writes.
All system calls (mach lookups, sysctl, ioctls) are allowed.

**Restrictive** profiles: `(deny default)` then explicitly allowlist only
`file-read*`, `process-exec`, `process-fork`, specific `sysctl-read`
entries, `mach-lookup` for sysmond, and tty ioctl.

All profiles restrict writes to: `TARGET_DIR` (subprocess `cwd`),
`TMP_DIR`, `CACHE_DIR`, `~/.gemini`, `~/.npm`, `~/.cache`,
`~/.gitconfig`, plus up to 5 `--include-directories`.

The default profile is `permissive-open`.

#### Network Restrictions

The `-closed` profiles block all outbound traffic. The `-proxied`
profiles allow outbound only to `localhost:8877` (requires a running
proxy). **Both block Gemini API access**, making them unsuitable for
any Gemini usage (ACP or interactive).

#### WINK Sandbox Mode Mapping

**Not available in ACP mode** due to the incompatibility described above.
The mapping below is retained for reference if the upstream issue is
resolved:

| WINK Mode | Gemini Implementation |
|-----------|----------------------|
| `"read-only"` | `--sandbox` + `--approval-mode plan` (agent-level read-only + OS-level write restriction) |
| `"workspace-write"` | `--sandbox` + `--approval-mode yolo` (OS-level write restriction to cwd) |

#### Additional Writable Directories

The `--include-directories` CLI flag adds up to 5 additional writable
directories to the seatbelt profile. This is not currently exposed in the
adapter config but could be added if needed for multi-mount workspaces.

### `initialized` Notification

Gemini does **not** handle the `initialized` notification. It logs an
error to stderr:

```
Error handling notification { method: 'initialized' } {
  code: -32601, message: '"Method not found": initialized'
}
```

The adapter should still send the notification for protocol compliance.
The error is harmless and does not affect behavior.

### Double Initialize

Gemini accepts duplicate `initialize` calls without error. Unlike
OpenCode, it returns a new result each time. The adapter should not
rely on double-initialize rejection for lifecycle management.

### Session Update Wire Format

Gemini's session update format matches the ACP SDK v0.8.0 Pydantic models.
The SDK correctly deserializes Gemini's updates into typed objects:

| Wire Field | SDK Type | Example |
|------------|----------|---------|
| `sessionUpdate: "agent_message_chunk"` | `AgentMessageChunk` | Text content |
| `sessionUpdate: "agent_thought_chunk"` | `AgentThoughtChunk` | Reasoning text |
| `sessionUpdate: "tool_call"` | `ToolCallStart` | Tool invocation start |
| `sessionUpdate: "tool_call_update"` | `ToolCallProgress` | Tool completion |

The discriminator field is `sessionUpdate` (snake_case), and content is
wrapped in a `content` object. The `ACPClient._track_update()` method uses
`type(update).__name__` which returns the PascalCase SDK class name
regardless of wire format — **no adapter-specific update handling needed.**

### Token Usage

**Not reported.** Gemini's `session/prompt` response does not include a
`usage` field. No `UsageUpdate` session updates are emitted. The adapter's
`extract_token_usage()` will return `None`, and `BudgetTracker` will
receive no data. Token budget enforcement is not available for Gemini.

### Permission Requests

**Not observed.** Gemini does not send `session/request_permission` to the
client. It handles tool approval internally based on the `--approval-mode`
CLI flag. The `ACPClient.request_permission()` implementation is still
needed for protocol compliance but will not be called in practice.

### MCP Server Support

Gemini reports `mcpCapabilities: {http: true, sse: true}`.

| Transport | Supported |
|-----------|-----------|
| HTTP (StreamableHTTP) | Yes |
| SSE | Yes |
| Stdio | **No** |

The WINK `MCPHttpServer` uses `StreamableHTTPServerTransport` over HTTP,
which aligns with Gemini's capabilities. The adapter passes `HttpMcpServer`
configs on `session/new` — same as the generic `ACPAdapter`.

Stdio MCP servers passed to `session/new` are silently ignored.

### Native Tools

Gemini has built-in tools that it uses autonomously:

| Tool | Description |
|------|-------------|
| `google_web_search` | Web search via Google |
| Shell execution | Command execution (observed via `tool_call` updates) |
| File operations | Read/write files in workspace |

Native tool invocations appear as `tool_call` / `tool_call_update`
session updates and are mapped to `ToolInvoked` events via the existing
`dispatch_tool_invoked()` function.

### Auth

Gemini inherits host-level credentials. Three auth methods are reported:

| Auth ID | Method |
|---------|--------|
| `oauth-personal` | Google OAuth (interactive browser login) |
| `gemini-api-key` | `GEMINI_API_KEY` environment variable |
| `vertex-ai` | Vertex AI service account credentials |

For non-interactive WINK execution, `GEMINI_API_KEY` or pre-cached OAuth
credentials are required. The adapter does not implement interactive auth
flows.

## Protocol Details

### ACP Protocol Reference

All protocol details validated against `gemini 0.29.5` with
`agent-client-protocol 0.8.0`. Key findings:

- Wire format includes `"jsonrpc": "2.0"` header (standard JSON-RPC 2.0)
- `PROTOCOL_VERSION` is `1` (integer)
- `session/new` returns only `sessionId` (no models, modes, or config)
- `session/prompt` requires `prompt` as array of `TextContentBlock` objects
- `session/prompt` response contains only `stopReason` (no `usage`)
- `session/setModel`, `session/setMode`, `session/list` all return -32601
- `initialized` notification logs error but does not break protocol
- Gemini exits cleanly (code 0) when stdin is closed

### MCP Server Config Format

```python
from acp.schema import HttpMcpServer, HttpHeader

HttpMcpServer(
    url="http://127.0.0.1:{port}/mcp",
    name="wink-tools",
    headers=[
        HttpHeader(name="Authorization", value="Bearer {token}"),
    ],
    type="http",
)
```

Fields: `url` (str), `name` (str), `headers` (list[HttpHeader]),
`type` (Literal["http"]).

## Usage Examples

### Basic

```python
from weakincentives import Prompt, PromptTemplate, MarkdownSection
from weakincentives.runtime import Session, InProcessDispatcher
from weakincentives.adapters.gemini_acp import (
    GeminiACPAdapter,
    GeminiACPAdapterConfig,
    GeminiACPClientConfig,
)

bus = InProcessDispatcher()
session = Session(dispatcher=bus)

template = PromptTemplate(
    ns="demo",
    key="gemini",
    sections=(
        MarkdownSection(
            title="Task",
            key="task",
            template="List the files in the repo and summarize.",
        ),
    ),
)
prompt = Prompt(template)

adapter = GeminiACPAdapter(
    adapter_config=GeminiACPAdapterConfig(
        model_id="gemini-2.5-flash",
    ),
    client_config=GeminiACPClientConfig(
        cwd="/absolute/path/to/workspace",
        permission_mode="auto",
        allow_file_reads=True,
    ),
)

with prompt.resources:
    resp = adapter.evaluate(prompt, session=session)

print(resp.text)
```

### With Approval Mode

```python
adapter = GeminiACPAdapter(
    adapter_config=GeminiACPAdapterConfig(
        model_id="gemini-2.5-pro",
        approval_mode="yolo",  # Auto-approve all tools
    ),
    client_config=GeminiACPClientConfig(
        cwd="/absolute/path/to/workspace",
    ),
)
```

### With Sandbox

```python
adapter = GeminiACPAdapter(
    adapter_config=GeminiACPAdapterConfig(
        model_id="gemini-2.5-flash",
        approval_mode="yolo",
        sandbox=True,  # Enable OS-level sandboxing
    ),
    client_config=GeminiACPClientConfig(
        cwd="/absolute/path/to/workspace",  # Writes restricted to this dir
    ),
)
```

### With Custom Seatbelt Profile (macOS)

```python
adapter = GeminiACPAdapter(
    adapter_config=GeminiACPAdapterConfig(
        sandbox=True,
        sandbox_profile="strict-open",  # Restrict reads + writes
    ),
    client_config=GeminiACPClientConfig(
        cwd="/absolute/path/to/workspace",
    ),
)
```

## Testing

Tests at `tests/adapters/gemini_acp/`:

| File | Coverage |
|------|----------|
| `test_adapter.py` | Gemini hook overrides, defaults, model/sandbox flag injection, env preparation |
| `test_config.py` | Gemini configuration defaults, sandbox fields, CLI arg construction |
| `conftest.py` | Imports from `tests/adapters/acp/conftest.py` |

## Non-Goals (v1)

- Skill installation (Gemini does not support Claude-compatible skill paths)
- Interactive auth flows (OAuth browser login)
- Token budget enforcement (Gemini does not report usage)
- Mode selection via ACP protocol (use `--approval-mode` flag instead)
- Session list/resume/fork (unimplemented by Gemini)

## Design Decisions

### Why CLI Flags for Model/Mode

Gemini does not implement `session/setModel` or `session/setMode`. These
are fundamental ACP methods that Gemini chose not to support, likely
because the CLI is designed around a single-model, single-mode execution
paradigm. The adapter must translate WINK's runtime model/mode
configuration into spawn-time CLI arguments.

### Why `yolo` Approval Mode

For non-interactive WINK execution, all tool approvals must be automatic.
Gemini's internal approval system is controlled by the `--approval-mode`
flag. The `yolo` mode auto-approves everything, matching WINK's
`permission_mode="auto"` semantics.

### Why No Ephemeral Home

Unlike OpenCode (which discovers skills at `$HOME/.claude/skills/`),
Gemini CLI uses its own extension system (`gemini extensions`). Skill
mounting via ephemeral HOME would have no effect. If skill support is
needed, it should use Gemini's native extension API.

## Related Specifications

- `specs/ACP_ADAPTER.md` — Generic ACP adapter (base class)
- `specs/OPENCODE_ADAPTER.md` — Sibling ACP adapter for OpenCode
- `specs/ADAPTERS.md` — Provider adapter protocol
- `specs/TOOLS.md` — Tool registration and policies

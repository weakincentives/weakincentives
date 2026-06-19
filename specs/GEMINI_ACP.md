# Gemini CLI ACP Adapter Specification

> **Status:** Draft.
> **Package:** `src/weakincentives/adapters/gemini_acp/`
> **Adapter name:** `gemini_acp`
> **Base class:** `ACPAdapter` from `src/weakincentives/adapters/acp/`
> **Gemini CLI entrypoint:** `gemini --experimental-acp`
> **ACP protocol:** v1 (JSON-RPC 2.0 over newline-delimited JSON on stdio)
> **Compatibility tests:** `tests/adapters/acp/test_gemini_compat.py`

## Purpose

`GeminiACPAdapter` is a thin wrapper around the generic `ACPAdapter` that
provides Gemini CLI-specific defaults and quirk handling. The two-layer
architecture separates protocol concerns (generic `acp/` package) from
agent-specific behavior (this package), following the same pattern established
by `OpenCodeACPAdapter`.

See `specs/ACP_ADAPTER.md` for the generic ACP protocol implementation.

| Responsibility | Owner |
|----------------|-------|
| Prompt composition, resource binding, session telemetry | WINK |
| Agentic execution (planning, reasoning, tool calls, code execution) | Gemini CLI |

## Why Gemini CLI

Gemini CLI is Google's open-source agentic coding assistant. It supports the
Agent Client Protocol via the `--experimental-acp` flag, making it compatible
with the generic `ACPAdapter` infrastructure.

Key differentiators from OpenCode:

- **Pre-authenticated:** Uses Google Cloud Application Default Credentials or
  API keys configured externally. No in-protocol authentication flow.
- **Multimodal capabilities:** Advertises `image: true`, `audio: true` in
  `PromptCapabilities`. WINK currently sends text-only prompts.
- **Proper tool call IDs:** Sends non-empty `tool_call_id` values (unlike
  OpenCode which sends empty strings requiring synthetic ID generation).
- **TextContentBlock content:** Sends `TextContentBlock` pydantic models in
  `AgentMessageChunk.content` (not plain strings).
- **Gemini-specific update types:** Emits `UsageUpdate`, `AgentPlanUpdate`,
  `AvailableCommandsUpdate`, `CurrentModeUpdate`, and other notification types
  not present in OpenCode's implementation.

## Requirements

### Runtime Dependencies

1. **Gemini CLI** installed and available on `PATH` (`gemini` binary)
1. **ACP Python SDK:** `agent-client-protocol>=0.8.0`
1. **MCP Python SDK:** `mcp>=1.26.0` (for in-process MCP server)
1. **ASGI server:** `uvicorn>=0.40.0` (for MCP HTTP transport)
1. WINK (`weakincentives`) runtime

### Gemini CLI Installation

```bash
npm install -g @anthropic-ai/gemini-cli
# or
npx @anthropic-ai/gemini-cli
```

### Authentication

Gemini CLI uses Google Cloud credentials configured externally. WINK does not
participate in the authentication flow. The `InitializeResponse` may include
an `auth_methods` field — the adapter ignores it and proceeds directly to
`new_session`.

Credential sources (in order of precedence):

1. `GOOGLE_API_KEY` environment variable
1. `GOOGLE_APPLICATION_CREDENTIALS` environment variable
1. Google Cloud Application Default Credentials (`~/.config/gcloud/`)
1. `~/.gemini/` configuration directory

## Module Structure

```
src/weakincentives/adapters/gemini_acp/
  __init__.py          # Public exports (re-exports from acp/ + Gemini classes)
  adapter.py           # GeminiACPAdapter(ACPAdapter)
  config.py            # GeminiACPClientConfig, GeminiACPAdapterConfig
```

All protocol implementation, client, events, MCP HTTP server, and structured
output handling live in the generic `adapters/acp/` package. This package only
contains configuration defaults and subclass hook overrides.

> **Note:** No ephemeral home directory is needed. Gemini CLI does not use
> `$HOME/.claude/skills/` for skill discovery. Skill mounting is a non-goal
> for v1.

## Configuration

### GeminiACPClientConfig

Extends `ACPClientConfig` with Gemini CLI defaults:

| Field | Override | Description |
|-------|----------|-------------|
| `agent_bin` | `"gemini"` | Gemini CLI binary |
| `agent_args` | `("--experimental-acp",)` | ACP mode flag |
| `permission_mode` | `"auto"` | Auto-approve all permission requests |

All other fields inherit from `ACPClientConfig`.

> **CWD requirement:** Same as generic ACP — must be an absolute path. Gemini
> CLI uses CWD as the workspace root for file operations and code execution.

### GeminiACPAdapterConfig

Extends `ACPAdapterConfig` with Gemini CLI defaults:

| Field | Override | Description |
|-------|----------|-------------|
| `quiet_period_ms` | `500` | Drain window for trailing updates |

> **Model IDs:** Gemini CLI model identifiers follow Google's naming convention
> (e.g., `gemini-2.5-pro`, `gemini-2.5-flash`). The exact list depends on the
> authenticated user's access tier. Model validation should be lenient until
> the available model set is confirmed through live testing.

## Adapter Name

The adapter registers as `"gemini_acp"`. This requires adding a constant to
`src/weakincentives/types/adapter.py`:

```python
GEMINI_ACP_ADAPTER_NAME: Final[AdapterName] = "gemini_acp"
```

## Subclass Hook Overrides

`GeminiACPAdapter` extends `ACPAdapter` and overrides the following hooks:

### `_adapter_name()`

Returns `GEMINI_ACP_ADAPTER_NAME` (`"gemini_acp"`).

### `_validate_model(model_id, available_models)`

Gemini CLI behavior with invalid models is not yet confirmed. Initial
implementation should follow the OpenCode pattern: check `model_id` against the
list returned by `new_session().models.available_models` and raise
`PromptEvaluationError(phase="request")` if not found.

> **Open question:** Does Gemini CLI return a proper error for invalid models,
> or does it silently produce empty output like OpenCode? If it raises a
> protocol error, this hook can be simplified to a no-op. Confirm via live
> testing.

### `_detect_empty_response(client, prompt_resp)`

Same pattern as OpenCode: raise `PromptEvaluationError(phase="response")` if
zero `AgentMessageChunk` updates were received. This catches configuration
errors and internal agent failures not surfaced via `stop_reason`.

### `_prepare_execution_env()`

Delegates to the parent implementation (`_build_env()`). No ephemeral home
directory manipulation is needed since Gemini CLI does not use the Claude skill
discovery path.

If Gemini CLI credentials are stored in `~/.gemini/` or
`~/.config/gcloud/`, the parent environment inheritance (returning `None` from
`_build_env()`) ensures these are available to the subprocess.

## Protocol Compatibility

### Handshake

The handshake follows the standard ACP flow:

1. `conn.initialize(protocol_version=1, client_capabilities=..., client_info=...)`
1. `conn.new_session(cwd=..., mcp_servers=[...])`

**Gemini-specific behavior:** The `InitializeResponse` may include an
`auth_methods` field. The adapter ignores this — Gemini is pre-authenticated
via external credentials. The `_handshake()` method in `ACPAdapter` already
discards the `InitializeResponse` return value (stores in `_`), so no override
is needed.

**Agent capabilities:** Gemini advertises `mcp_capabilities: {http: true, sse: true}`, confirming HTTP MCP server support. The WINK MCP HTTP server
(`type="http"`) is compatible.

### Content Block Types

Gemini sends `TextContentBlock` pydantic models in `AgentMessageChunk.content`,
not plain strings. The generic `_extract_chunk_text()` function handles this
via `getattr(raw, "text", None)`:

```
chunk.content → TextContentBlock(type="text", text="...")
               → getattr(raw, "text", None) → "..."
```

Gemini may also send `ImageContentBlock` objects (multimodal responses). These
lack a `.text` attribute and fall through to `str(raw)`. This is acceptable —
WINK does not process image content, and the string representation provides
debugging visibility.

### Session Update Types

Gemini emits several update types beyond the core set:

| Update Type | Handling | Notes |
|-------------|----------|-------|
| `AgentMessageChunk` | Tracked in `message_chunks` | Primary response text |
| `AgentThoughtChunk` | Tracked in `thought_chunks` | Reasoning (if `emit_thought_chunks`) |
| `ToolCallStart` | Tracked in `tool_call_tracker` | Proper `tool_call_id` values |
| `ToolCallProgress` | Tracked in `tool_call_tracker` | Terminal status updates |
| `UsageUpdate` | Silently ignored by `_track_update()` | Bumps `last_update_time` |
| `AgentPlanUpdate` | Silently ignored | Bumps `last_update_time` |
| `AvailableCommandsUpdate` | Silently ignored | Bumps `last_update_time` |
| `CurrentModeUpdate` | Silently ignored | Bumps `last_update_time` |
| `ConfigOptionUpdate` | Silently ignored | Bumps `last_update_time` |
| `SessionInfoUpdate` | Silently ignored | Bumps `last_update_time` |

All unrecognized update types are handled safely: `session_update()` always
bumps `last_update_time` (ensuring quiet period drain works correctly), and
`_track_update()` dispatches by `type(update).__name__` — unmatched types are
silently dropped.

> **Important:** The `last_update_time` bump is critical. Without it, the quiet
> period drain would exit prematurely if the only updates received were
> Gemini-specific types, potentially missing trailing `AgentMessageChunk`
> updates still in the pipe.

### Tool Call IDs

Gemini sends proper (non-empty) `tool_call_id` values on `ToolCallStart` and
`ToolCallProgress`. The `_resolve_tool_id()` method in `ACPClient` passes
non-empty IDs through without generating synthetic replacements:

```python
if raw_id:           # Non-empty → use as-is
    return raw_id
# Empty → generate _tc_N synthetic ID (OpenCode path)
```

No adapter override needed.

### Token Usage

Gemini's `PromptResponse.usage` follows the standard ACP `Usage` schema:

| ACP `Usage` field | WINK `TokenUsage` field | Notes |
|-------------------|------------------------|-------|
| `input_tokens` | `input_tokens` | Standard mapping |
| `output_tokens` | `output_tokens` | Standard mapping |
| `cached_read_tokens` | `cached_tokens` | Gemini context caching |
| `thought_tokens` | (logged, not mapped) | Extended thinking tokens |

The generic `extract_token_usage()` handles this mapping. No override needed.

> **Open question:** Does Gemini populate `cached_read_tokens` for context
> caching scenarios? If not, it will be `None` — handled gracefully.

### MCP Server Passthrough

Gemini advertises `mcp_capabilities: {http: true}` in its `InitializeResponse`.
The WINK MCP HTTP server is passed as:

```python
HttpMcpServer(
    url="http://127.0.0.1:{port}/mcp",
    name="wink-tools",
    headers=[HttpHeader(name="Authorization", value="Bearer {token}")],
    type="http",
)
```

The `type="http"` discriminator matches the ACP schema union type. Bearer token
authentication is handled by the ASGI app in `_mcp_http.py`.

### Prompt Format

`_send_prompt()` sends prompts as `TextContentBlock` instances:

```python
conn.prompt(
    [TextContentBlock(type="text", text=rendered_text)],
    session_id=acp_session_id,
)
```

This matches the `PromptRequest.prompt` field's expected union type. Gemini
processes `TextContentBlock` natively.

### Permission Requests

Gemini uses the standard `request_permission` callback. The `ACPClient`
implementation handles this based on `permission_mode`:

- `"auto"` → Approve (default for Gemini config)
- `"deny"` → Deny with policy reason
- `"prompt"` → Deny with non-interactive reason

No override needed.

## Gemini-Specific Behaviors

### Authentication Passthrough

Unlike OpenCode (which stores credentials at `~/.local/share/opencode/`),
Gemini CLI uses standard Google Cloud credential paths:

- `GOOGLE_API_KEY` — Environment variable (highest priority)
- `GOOGLE_APPLICATION_CREDENTIALS` — Service account key file path
- `~/.config/gcloud/application_default_credentials.json` — ADC
- `~/.gemini/settings.json` — Gemini CLI-specific config

When `ACPClientConfig.env` is `None` (default), the subprocess inherits the
parent process environment, preserving all credential sources. When custom `env`
is set, it is merged with `os.environ` to maintain credential availability.

### Multimodal Capabilities

Gemini advertises `image: true` and `audio: true` in `PromptCapabilities`.
WINK currently sends text-only prompts (`TextContentBlock`). Multimodal prompt
bridging is a non-goal for v1.

If multimodal support is added later, the adapter would need to:

1. Accept image/audio resources in the WINK prompt
1. Convert to appropriate ACP content block types
1. Include in the `conn.prompt()` call alongside text blocks

### Code Execution

Gemini CLI supports native code execution (Python sandbox). Code execution
tool calls appear as `ToolCallStart` / `ToolCallProgress` updates with titles
like `code_execution`. These are tracked by the generic `ACPClient` tool call
tracker and dispatched as `ToolInvoked` events.

### Session Modes

Gemini may support multiple modes (discoverable via `new_session().modes`).
Mode setting uses the generic `_configure_session()` method with best-effort
semantics — mode errors are logged as warnings via `_handle_mode_error()`.

## Implementation Checklist

### 1. Add Adapter Name Constant

File: `src/weakincentives/types/adapter.py`

```python
GEMINI_ACP_ADAPTER_NAME: Final[AdapterName] = "gemini_acp"
```

Update `__all__` to include the new constant.

### 2. Create Configuration Module

Create adapters/gemini_acp/config.py:

```python
@FrozenDataclass()
class GeminiACPClientConfig(ACPClientConfig):
    agent_bin: str = "gemini"
    agent_args: tuple[str, ...] = ("--experimental-acp",)

@FrozenDataclass()
class GeminiACPAdapterConfig(ACPAdapterConfig):
    pass  # Inherit defaults; override if Gemini needs different quiet_period_ms
```

### 3. Create Adapter Module

Create adapters/gemini_acp/adapter.py:

```python
class GeminiACPAdapter(ACPAdapter):
    def __init__(
        self,
        *,
        adapter_config: GeminiACPAdapterConfig | None = None,
        client_config: GeminiACPClientConfig | None = None,
    ) -> None:
        super().__init__(
            adapter_config=adapter_config or GeminiACPAdapterConfig(),
            client_config=client_config or GeminiACPClientConfig(),
        )

    @override
    def _adapter_name(self) -> AdapterName:
        return GEMINI_ACP_ADAPTER_NAME

    @override
    def _validate_model(self, model_id: str, available_models: list[Any]) -> None:
        if not available_models:
            return
        model_ids = [getattr(m, "model_id", None) for m in available_models]
        if model_id not in model_ids:
            raise PromptEvaluationError(
                message=f"Model '{model_id}' not found. Available: {model_ids}",
                prompt_name="",
                phase="request",
            )

    @override
    def _detect_empty_response(self, client: ACPClient, prompt_resp: Any) -> None:
        if not client.message_chunks:
            raise PromptEvaluationError(
                message=(
                    "Gemini returned an empty response (zero AgentMessageChunks). "
                    "Check model configuration and authentication."
                ),
                prompt_name="",
                phase="response",
            )
```

### 4. Create Package Init

Create adapters/gemini_acp/\_\_init\_\_.py:

```python
from ..acp import ACPClient, ACPSessionState, McpServerConfig
from .adapter import GeminiACPAdapter
from .config import GeminiACPAdapterConfig, GeminiACPClientConfig

__all__ = [
    "ACPClient",
    "ACPSessionState",
    "GeminiACPAdapter",
    "GeminiACPAdapterConfig",
    "GeminiACPClientConfig",
    "McpServerConfig",
]
```

### 5. Create Tests

Create tests/adapters/gemini_acp/test_adapter.py:

- Adapter name returns `"gemini_acp"`
- Default config uses `agent_bin="gemini"`, `agent_args=("--experimental-acp",)`
- Model validation raises for unknown models
- Model validation passes when available_models is empty
- Empty response detection raises `PromptEvaluationError`
- `_prepare_execution_env()` delegates to parent (no ephemeral home)

Create tests/adapters/gemini_acp/test_config.py:

- Default values match Gemini CLI expectations
- Config is frozen
- Custom values accepted

### 6. Update Adapter Registry

File: `src/weakincentives/adapters/__init__.py`

Add `gemini_acp` to the lazy import registry if one exists, or document the
import path in `__all__`.

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
            template="Analyze this repository and suggest improvements.",
        ),
    ),
)
prompt = Prompt(template)

adapter = GeminiACPAdapter(
    adapter_config=GeminiACPAdapterConfig(
        model_id="gemini-2.5-pro",
    ),
    client_config=GeminiACPClientConfig(
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

### Structured Output

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class CodeReview:
    summary: str
    issues: list[str]
    score: int

template = PromptTemplate[CodeReview](
    ns="demo",
    key="review",
    sections=(
        MarkdownSection(
            title="Task",
            key="task",
            template="Review the code changes and provide a structured assessment.",
        ),
    ),
)

prompt = Prompt(template)
with prompt.resources:
    resp = adapter.evaluate(prompt, session=session)

review: CodeReview = resp.output  # Typed structured output
```

### With Custom Environment

```python
adapter = GeminiACPAdapter(
    client_config=GeminiACPClientConfig(
        cwd="/workspace",
        env={"GOOGLE_API_KEY": "your-api-key"},
        startup_timeout_s=30.0,
    ),
)
```

## Compatibility Matrix

Validated protocol behaviors (from `test_gemini_compat.py`):

| Area | Status | Notes |
|------|--------|-------|
| Handshake with `auth_methods` | Compatible | `auth_methods` ignored; proceeds to `new_session` |
| `TextContentBlock` extraction | Compatible | `_extract_chunk_text()` handles `.text` attribute |
| `ImageContentBlock` fallback | Compatible | Falls back to `str()` without crash |
| Unhandled update types | Compatible | Silently ignored; `last_update_time` bumped |
| MCP HTTP server `type="http"` | Compatible | Matches ACP schema discriminator |
| Gemini CLI config defaults | Compatible | `agent_bin="gemini"`, `agent_args=("--experimental-acp",)` |
| Real tool call IDs | Compatible | Non-empty IDs tracked without synthetic generation |
| `cached_read_tokens` mapping | Compatible | Maps to `TokenUsage.cached_tokens` |
| `TextContentBlock` prompt format | Compatible | `_send_prompt()` sends correct type |
| `mcp_capabilities` parsing | Compatible | `{http: true, sse: true}` readable |

## Open Questions

1. **Model validation behavior:** Does Gemini CLI return a protocol error for
   invalid `model_id` values, or does it silently produce empty output? This
   determines whether `_validate_model()` is critical or defensive.

1. **Available models list:** What model IDs does `new_session().models`
   return? Are they Google model names (`gemini-2.5-pro`) or prefixed
   (`google/gemini-2.5-pro`)?

1. **Mode support:** What modes does Gemini CLI expose via `new_session().modes`?
   Does `set_session_mode` work or return an error like OpenCode?

1. **`stop_reason` values:** What `stop_reason` values does Gemini send in
   `PromptResponse`? Standard ACP uses `"end_turn"`, `"max_tokens"`, etc.

1. **Skill discovery:** Does Gemini CLI support a skill/extension discovery
   mechanism? If so, what is the filesystem layout?

1. **Quiet period tuning:** Is 500ms sufficient for Gemini's stdio timing, or
   does it need a larger drain window?

## Non-Goals (v1)

- Multimodal prompt bridging (image/audio inputs)
- Gemini CLI skill/extension installation
- Session fork/resume/list
- Authentication flow management (Gemini uses external credentials)
- Terminal capability support

## Testing

### Existing Compatibility Tests

`tests/adapters/acp/test_gemini_compat.py` validates protocol-level
compatibility at the generic `ACPAdapter` layer:

| Test Class | Coverage |
|------------|----------|
| `TestGeminiHandshakeWithAuthMethods` | Handshake with `auth_methods` present |
| `TestTextContentBlockExtraction` | `TextContentBlock` pydantic model handling |
| `TestImageContentBlockFallback` | Non-text content blocks don't crash |
| `TestUnhandledUpdateTypesDontCrash` | Gemini-specific update types ignored |
| `TestUsageUpdateTimestampsLastUpdate` | Update time bumped for drain logic |
| `TestMcpHttpServerTypeField` | `type="http"` discriminator |
| `TestGeminiConfigDefaults` | Config accepts Gemini arguments |
| `TestRealToolCallIdsTrackedCorrectly` | Non-empty tool call IDs |
| `TestCachedReadTokensMapped` | Token usage field mapping |
| `TestPromptSendsTextContentBlock` | Prompt payload format |
| `TestAgentCapabilitiesMcpHttpCheck` | MCP capabilities parsing |

### Adapter-Specific Tests (to be created)

`tests/adapters/gemini_acp/`:

| File | Coverage |
|------|----------|
| `test_adapter.py` | Hook overrides, defaults, validation |
| `test_config.py` | Configuration defaults and constraints |
| `conftest.py` | Imports from `tests/adapters/acp/conftest.py` |

## Related Specifications

- `specs/ACP_ADAPTER.md` — Generic ACP adapter (base class)
- `specs/OPENCODE_ADAPTER.md` — Sibling adapter (reference implementation)
- `specs/ADAPTERS.md` — Provider adapter protocol
- `specs/TOOLS.md` — Tool registration and policies
- `specs/GUARDRAILS.md` — Tool policies, feedback providers, task completion
- `specs/TRANSCRIPT.md` — Transcript bridge for debug bundles

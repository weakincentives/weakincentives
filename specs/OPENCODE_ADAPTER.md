# OpenCode ACP Adapter Specification

> **Status:** Implemented.
> **Package:** `src/weakincentives/adapters/opencode_acp/`
> **Adapter name:** `opencode_acp`
> **Base class:** `ACPAdapter` from `src/weakincentives/adapters/acp/`
> **OpenCode entrypoint:** `opencode acp`
> **Validated against:** `opencode 1.1.59` with `agent-client-protocol 0.8.0`

## Purpose

`OpenCodeACPAdapter` is a thin wrapper around the generic `ACPAdapter` that
provides OpenCode-specific defaults and quirk handling. The two-layer
architecture separates protocol concerns (generic `acp/` package) from
agent-specific behavior (this package).

See `specs/ACP_ADAPTER.md` for the generic ACP protocol implementation.

## Module Structure

```
src/weakincentives/adapters/opencode_acp/
  __init__.py          # Public exports (re-exports from acp/ + OpenCode classes)
  adapter.py           # OpenCodeACPAdapter(ACPAdapter)
  config.py            # OpenCodeACPClientConfig, OpenCodeACPAdapterConfig
  _ephemeral_home.py   # OpenCodeEphemeralHome (skill installation)
```

All protocol implementation, client, events, MCP HTTP server, and structured
output handling live in the generic `adapters/acp/` package. This package only
contains configuration defaults and subclass hook overrides.

## Configuration

### OpenCodeACPClientConfig

Defined at `src/weakincentives/adapters/opencode_acp/config.py:24`.
Extends `ACPClientConfig` with OpenCode defaults:

| Field | Override | Description |
|-------|----------|-------------|
| `agent_bin` | `"opencode"` | OpenCode CLI binary |
| `agent_args` | `("acp",)` | Standard ACP mode flag |
| `permission_mode` | `"auto"` | Auto-approve all permission requests |

All other fields inherit from `ACPClientConfig`.

### OpenCodeACPAdapterConfig

Defined at `src/weakincentives/adapters/opencode_acp/config.py:34`.
Extends `ACPAdapterConfig` with OpenCode defaults:

| Field | Override | Description |
|-------|----------|-------------|
| `quiet_period_ms` | `500` | Validated minimum for OpenCode's stdio timing |

## Subclass Hook Overrides

`OpenCodeACPAdapter` at `src/weakincentives/adapters/opencode_acp/adapter.py:28`
overrides three hooks from `ACPAdapter`:

### `_adapter_name()`

Returns `"opencode_acp"` (the `OPENCODE_ACP_ADAPTER_NAME` constant).
See `src/weakincentives/adapters/opencode_acp/adapter.py:48`.

### `_validate_model(model_id, available_models)`

At `src/weakincentives/adapters/opencode_acp/adapter.py:52`:

Checks `model_id` against the list returned by `new_session().models`.
If the model is not found, raises `PromptEvaluationError(phase="request")`
with a message listing all available models.

This is OpenCode-specific because `set_session_model` with an invalid model ID
causes OpenCode to silently return an empty response with
`stop_reason="end_turn"` and zero content — no error is raised by the protocol.
The adapter must detect this upfront.

### `_detect_empty_response(client, prompt_resp)`

At `src/weakincentives/adapters/opencode_acp/adapter.py:71`:

Raises `PromptEvaluationError(phase="response")` if zero `AgentMessageChunk`
updates were received. This catches cases where:

- An invalid model was accepted but cannot generate output
- The agent encountered an internal error not surfaced via `stop_reason`

## OpenCode-Specific Behaviors

### Mode Error Tolerance

OpenCode 1.1.59 returns "Internal error" for `set_session_mode`. The generic
`ACPAdapter._handle_mode_error()` logs this as a warning and continues — no
OpenCode-specific override needed. Modes are reported via `new_session().modes`
and via `CurrentModeUpdate` notifications.

### Available Models (OpenCode 1.1.59)

Models observed via `new_session().models.available_models`:

**OpenAI:** `openai/gpt-5.3-codex`, `openai/gpt-5.2-codex`, `openai/gpt-5.2`,
`openai/gpt-5.1-codex-mini`, `openai/gpt-5.1-codex-max`, `openai/gpt-5.1-codex`

**OpenCode Zen:** `opencode/big-pickle`, `opencode/gpt-5-nano`,
`opencode/minimax-m2.5-free`, `opencode/kimi-k2.5-free`

Most models support variants: `low`, `medium`, `high`, `xhigh` (appended as
`model_id/variant`). Model availability depends on auth type and plan.

### Available Modes (OpenCode 1.1.59)

| Mode ID | Description |
|---------|-------------|
| `build` | Default agent. Executes tools based on configured permissions. |
| `plan` | Plan mode. Disallows all edit tools. |

## Usage Examples

### Basic

```python
from weakincentives import Prompt, PromptTemplate, MarkdownSection
from weakincentives.runtime import Session, InProcessDispatcher
from weakincentives.adapters.opencode_acp import (
    OpenCodeACPAdapter,
    OpenCodeACPAdapterConfig,
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
    adapter_config=OpenCodeACPAdapterConfig(
        model_id="openai/gpt-5.1-codex-mini",
    ),
    client_config=OpenCodeACPClientConfig(
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

## Protocol Details

### ACP Protocol Reference

All protocol details were validated against `opencode 1.1.59` with
`agent-client-protocol 0.8.0`. Key findings:

- `spawn_agent_process` communicates over stdio (not HTTP)
- `PROTOCOL_VERSION` is `1` (integer)
- `ClientCapabilities` uses typed Pydantic models, not raw dicts
- `SessionAccumulator.apply()` requires `SessionNotification` wrappers
- `ToolCallStart` uses `title` for tool name, not `name`
- `set_session_mode` returns "Internal error" from OpenCode
- `set_session_model` accepts invalid model IDs without error; prompt returns empty
- `RequestError` is in `acp` top-level, not `acp.schema`

### MCP Server Config Format

The adapter passes the WINK MCP server URL as an `HttpMcpServer`:

```python
from acp.schema import HttpMcpServer

HttpMcpServer(
    url="http://127.0.0.1:{port}/mcp",
    name="wink-tools",
    headers=[],
    type="http",
)
```

Fields: `url` (str), `name` (str), `headers` (list, pass `[]`),
`type` (Literal["http"]).

## Testing

Tests at `tests/adapters/opencode_acp/`:

| File | Coverage |
|------|----------|
| `test_adapter.py` | OpenCode hook overrides, defaults, validation |
| `test_config.py` | OpenCode configuration defaults |
| `conftest.py` | Imports from `tests/adapters/acp/conftest.py` |

## Skill Installation

`OpenCodeEphemeralHome` at `src/weakincentives/adapters/opencode_acp/_ephemeral_home.py`
manages a temporary HOME directory for OpenCode skill discovery. Skills from
`RenderedPrompt.skills` are mounted at `$HOME/.claude/skills/<name>/SKILL.md`,
which is the Claude-compatible global discovery path that OpenCode natively
searches.

The ephemeral home preserves auth data from the real HOME:

- `~/.local/share/opencode/` -- stored API credentials (`auth.json`)
- `~/.aws/` -- AWS credentials for Bedrock provider support

Both copies are best-effort (warn on failure, don't crash) since auth may come
from environment variables instead.

## Guardrails

The OpenCode adapter inherits the full guardrails stack from the generic
`ACPAdapter` base class (see `specs/ACP_ADAPTER.md`):

- **Tool policies**: Enforced in `BridgedTool` before handler execution
- **Feedback providers**: Injected via `post_call_hook` on the MCP tool server,
  collecting guidance after successful tool calls
- **Task completion**: Continuation loop (`_run_prompt_loop`, max 10 rounds)
  re-prompts the agent when the checker reports incomplete with feedback

Implementation: `adapters/acp/_guardrails.py` (shared with all ACP subclasses)

## Non-Goals (v1)

- Full OpenCode config management (providers, keys)
- Session fork/resume/list (ACP supports these; add when needed)
- Authentication flows (OpenCode inherits host-level credentials)

## Related Specifications

- `specs/ACP_ADAPTER.md` — Generic ACP adapter (base class)
- `specs/ADAPTERS.md` — Provider adapter protocol
- `specs/CLAUDE_AGENT_SDK.md` — Reference adapter architecture
- `specs/CODEX_APP_SERVER.md` — Sibling adapter

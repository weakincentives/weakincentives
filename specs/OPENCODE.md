# OpenCode Adapter Specification

> **SDK**: `opencode-sdk>=0.3.0`

## Purpose

`OpenCodeAdapter` evaluates prompts via the OpenCode Python SDK while maintaining
orchestration state in a Session. Provides server lifecycle management, MCP tool
bridging, SSE-based event streaming, and hermetic workspace isolation.

**Implementation:** `src/weakincentives/adapters/opencode/`

## Background

OpenCode is an open-source coding agent with a client/server architecture:

- **OpenCode Server**: HTTP server exposing REST endpoints + SSE events
- **OpenCode TUI/Web**: Clients that connect to the server
- **OpenCode Python SDK**: HTTP client generated from OpenAPI spec

Unlike Claude Agent SDK (which manages a long-running process internally), the
OpenCode Python SDK is purely an HTTP client. WINK must own the server lifecycle.

```
┌─────────────────────┐         ┌──────────────────────┐
│ WINK Adapter        │  HTTP   │ OpenCode Server      │
│ ├─ ServerManager    │────────▶│ ├─ Sessions          │
│ ├─ EventSubscriber  │◀──SSE───│ ├─ Tools (fs, shell) │
│ └─ MCP Server       │         │ └─ MCP Integration   │
└─────────────────────┘         └──────────────────────┘
```

## Requirements

- Python: `pip install 'weakincentives[opencode]'`
- OpenCode CLI: `npm install -g opencode` (or via Bun: `bun add -g opencode`)
- Server health endpoint: `GET /global/health` must return `200`

## Architecture Overview

### Key Differences from Claude Agent SDK

| Aspect | Claude Agent SDK | OpenCode |
|--------|------------------|----------|
| Runtime boundary | SDK-managed process | HTTP server (self-managed) |
| Event mechanism | In-process hooks | SSE stream (`/event`) |
| Tool bridging | SDK MCP helpers | External MCP server |
| Session isolation | Ephemeral HOME | One server per session |
| Permissions | `bypassPermissions` mode | API permission responses |

### Adapter Responsibilities

1. **Server Lifecycle**: Spawn, health-check, and terminate `opencode serve`
2. **Workspace Isolation**: Per-session temp directories and config injection
3. **MCP Tool Bridging**: External MCP server exposing WINK tools
4. **Event Streaming**: SSE subscription for state synchronization
5. **Permission Automation**: Programmatic permission responses
6. **Structured Output**: Tool-based output contract

## Configuration

### OpenCodeClientConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `hostname` | `str` | `"127.0.0.1"` | Server bind address |
| `port` | `int \| None` | `None` | Server port (`None` = auto-assign) |
| `server_password` | `str \| None` | `None` | HTTP Basic Auth password |
| `server_timeout` | `float` | `30.0` | Server startup timeout (seconds) |
| `permission_mode` | `PermissionMode` | `"auto_approve"` | Permission handling mode |
| `max_turns` | `int \| None` | `None` | Maximum conversation turns |
| `suppress_stderr` | `bool` | `True` | Suppress server stderr |
| `isolation` | `OpenCodeIsolationConfig \| None` | `None` | Isolation configuration |

### OpenCodeModelConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `"anthropic/claude-sonnet-4-20250514"` | Model identifier |
| `provider` | `str \| None` | `None` | Provider override (anthropic, openai, etc.) |
| `temperature` | `float \| None` | `None` | Sampling temperature |
| `max_tokens` | `int \| None` | `None` | Maximum output tokens |

### OpenCodeIsolationConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `env` | `Mapping[str, str] \| None` | `None` | Environment variables |
| `config_content` | `str \| None` | `None` | Inline JSON config |
| `config_dir` | `Path \| None` | `None` | Config directory override |
| `cache_dir` | `Path \| None` | `None` | Cache directory override |
| `include_host_env` | `bool` | `False` | Copy non-sensitive env vars |

### PermissionMode

| Mode | Description |
|------|-------------|
| `"auto_approve"` | Automatically approve all permissions (requires isolated workspace) |
| `"auto_approve_with_remember"` | Approve and remember for session |
| `"policy_based"` | Evaluate against WINK policies, deny if no match |
| `"deny_all"` | Deny all permission requests |

## Server Lifecycle

### ServerManager

Manages the OpenCode server subprocess:

```python
@FrozenDataclass()
class ServerManager:
    hostname: str
    port: int
    process: subprocess.Popen[bytes]
    base_url: str

    @classmethod
    def spawn(
        cls,
        *,
        hostname: str = "127.0.0.1",
        port: int | None = None,
        cwd: Path,
        env: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> ServerManager: ...

    def health_check(self) -> ServerHealth: ...
    def terminate(self, timeout: float = 5.0) -> None: ...
```

### Lifecycle Phases

1. **Port Selection**: Allocate free port if not specified
2. **Environment Setup**: Merge isolation env vars, set `OPENCODE_CONFIG_*`
3. **Process Spawn**: Run `opencode serve --hostname X --port Y`
4. **Health Wait**: Poll `GET /global/health` until healthy or timeout
5. **Session Creation**: `POST /session` to create agent session
6. **Execution**: Message loop with SSE event subscription
7. **Cleanup**: Terminate process, remove temp directories

### Health Check Response

```python
@FrozenDataclass()
class ServerHealth:
    healthy: bool
    version: str
    uptime_seconds: float
```

Version compatibility should be validated; SDK/server skew may cause issues.

## Workspace Isolation

### Hermetic Session Pattern

For true isolation, each session requires:

1. **Dedicated server instance** (separate `opencode serve` process)
2. **Isolated workspace directory** (temp dir with mounted files)
3. **Isolated config** (via `OPENCODE_CONFIG_CONTENT` or `OPENCODE_CONFIG_DIR`)
4. **Isolated cache** (via `OPENCODE_CACHE_DIR` to avoid plugin conflicts)

```python
# Create isolated workspace
workspace = OpenCodeWorkspaceSection(
    session=session,
    mounts=(HostMount(host_path="/repo", mount_path="repo"),),
    allowed_host_roots=("/repo",),
)

# Server runs in workspace directory
server = ServerManager.spawn(
    cwd=workspace.temp_dir,
    env={
        "OPENCODE_CONFIG_CONTENT": json.dumps(config),
        "HOME": str(ephemeral_home),
    },
)
```

### Config Injection

OpenCode config can be injected via environment:

| Variable | Description |
|----------|-------------|
| `OPENCODE_CONFIG` | Path to config file |
| `OPENCODE_CONFIG_DIR` | Directory containing config |
| `OPENCODE_CONFIG_CONTENT` | Inline JSON config content |

For hermetic isolation, prefer `OPENCODE_CONFIG_CONTENT` with generated config:

```python
config = {
    "provider": "anthropic",
    "model": "claude-sonnet-4-20250514",
    "mcp": {"wink": {"command": "python", "args": ["-m", "mcp_server"]}},
    "permissions": {"allow": ["read:*", "write:workspace/*"]},
}
```

### OpenCodeWorkspaceSection

Prompt section providing isolated workspace:

```python
@FrozenDataclass()
class OpenCodeWorkspaceSection(Section[None]):
    temp_dir: Path
    mounts: tuple[HostMountPreview, ...]
    server_manager: ServerManager | None

    def resources(self) -> ResourceRegistry:
        return ResourceRegistry.build({Filesystem: self._filesystem})

    def cleanup(self) -> None:
        if self.server_manager:
            self.server_manager.terminate()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
```

## MCP Tool Bridging

### Architecture

Unlike Claude Agent SDK (which provides `create_sdk_mcp_server`), OpenCode
requires an external MCP server. WINK tools are exposed via a standalone MCP
server that OpenCode connects to.

```
┌────────────────┐      JSON-RPC       ┌────────────────┐
│ OpenCode       │────────────────────▶│ WINK MCP Server│
│ (calls tools)  │◀────────────────────│ (tool handlers)│
└────────────────┘                     └────────────────┘
                                              │
                                              ▼
                                       ┌────────────────┐
                                       │ Session        │
                                       │ (state, events)│
                                       └────────────────┘
```

### MCP Server Implementation

```python
class WinkMcpServer:
    """MCP server exposing WINK tools to OpenCode."""

    def __init__(
        self,
        tools: tuple[BridgedTool, ...],
        session: Session,
        prompt: Prompt[Any],
    ) -> None: ...

    def handle_tools_list(self) -> list[ToolSchema]: ...
    def handle_tools_call(self, name: str, arguments: dict[str, Any]) -> ToolResult: ...

    def serve_stdio(self) -> None:
        """Run MCP server over stdio (for subprocess spawning)."""
        ...

    def serve_http(self, port: int) -> None:
        """Run MCP server over HTTP (for remote connection)."""
        ...
```

### BridgedTool (Reused Pattern)

Same transactional wrapper as Claude Agent SDK adapter:

```python
@FrozenDataclass()
class BridgedTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    tool: Tool[Any, Any]

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute with transactional semantics."""
        with tool_transaction(self._session, self._resources, tag=f"tool:{self.name}"):
            result = self._tool.handler(params, context=self._context)
            return self._format_mcp_result(result)
```

### MCP Registration in OpenCode Config

```json
{
  "mcp": {
    "wink": {
      "command": "python",
      "args": ["-m", "weakincentives.adapters.opencode.mcp_server"],
      "env": {
        "WINK_SESSION_ID": "session-123",
        "WINK_SOCKET": "/tmp/wink-session-123.sock"
      }
    }
  }
}
```

Or via dynamic registration:

```python
# POST /mcp to add MCP server at runtime
client.mcp.add(name="wink", config=mcp_config)
```

## Event Streaming

### SSE Event Subscription

OpenCode provides SSE streams for real-time events:

- `GET /event` - Session-scoped events
- `GET /global/event` - Server-wide events

```python
class EventSubscriber:
    """Subscribes to OpenCode SSE events."""

    def __init__(self, base_url: str, session_id: str) -> None: ...

    async def subscribe(self) -> AsyncIterator[OpenCodeEvent]:
        """Yield events from SSE stream."""
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", f"{self.base_url}/event") as response:
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        yield self._parse_event(line[5:])
```

### Event Types

| Event | Description | WINK Mapping |
|-------|-------------|--------------|
| `session.created` | Session initialized | Log |
| `message.created` | New message added | Log |
| `message.updated` | Message content changed | Streaming output |
| `tool.execute.before` | Tool execution starting | Pre-tool snapshot |
| `tool.execute.after` | Tool execution complete | `ToolInvoked` event |
| `permission.requested` | Permission needed | Permission handler |
| `session.idle` | Agent awaiting input | Turn complete |
| `session.compacted` | Context compacted | Log |
| `error` | Error occurred | Error handling |

### State Synchronization via Events

```python
async def run_event_loop(
    subscriber: EventSubscriber,
    session: Session,
    permission_handler: PermissionHandler,
) -> None:
    async for event in subscriber.subscribe():
        match event.type:
            case "tool.execute.before":
                # Snapshot state for potential rollback
                snapshot = create_snapshot(session, resources)
                pending_tools[event.tool_id] = snapshot

            case "tool.execute.after":
                # Dispatch ToolInvoked event
                session.dispatch(ToolInvoked(
                    tool_name=event.tool_name,
                    params=event.params,
                    result=event.result,
                ))
                del pending_tools[event.tool_id]

            case "permission.requested":
                # Handle permission request
                decision = permission_handler.decide(event.permission)
                await client.permissions.respond(
                    session_id=session_id,
                    permission_id=event.permission_id,
                    allowed=decision.allowed,
                    remember=decision.remember,
                )

            case "session.idle":
                # Turn complete, check for task completion
                break
```

## Permission Handling

### Permission API

OpenCode exposes permission handling via REST:

```python
# Respond to permission request
POST /session/:id/permissions/:permissionID
{
    "allowed": true,
    "remember": true
}
```

### PermissionHandler

```python
class PermissionHandler:
    """Handles OpenCode permission requests."""

    def __init__(
        self,
        mode: PermissionMode,
        policies: tuple[ToolPolicy, ...] = (),
    ) -> None: ...

    def decide(self, permission: PermissionRequest) -> PermissionDecision:
        match self.mode:
            case "auto_approve":
                return PermissionDecision(allowed=True, remember=False)

            case "auto_approve_with_remember":
                return PermissionDecision(allowed=True, remember=True)

            case "policy_based":
                # Evaluate against WINK policies
                for policy in self.policies:
                    decision = policy.check_permission(permission)
                    if not decision.allowed:
                        return PermissionDecision(allowed=False)
                return PermissionDecision(allowed=True, remember=True)

            case "deny_all":
                return PermissionDecision(allowed=False, remember=False)
```

### Permission Types

| Permission | Description | Mapping |
|------------|-------------|---------|
| `read:path` | File read access | ReadBeforeWritePolicy state |
| `write:path` | File write access | Policy check |
| `execute:command` | Shell command | Command allowlist |
| `network:domain` | Network access | NetworkPolicy |

## Message API

### Synchronous Messaging

```python
# POST /session/:id/message (blocking)
response = client.session.message(
    session_id=session_id,
    message=MessageInput(
        role="user",
        content=prompt_text,
    ),
)
```

### Asynchronous Messaging

```python
# POST /session/:id/prompt_async (non-blocking)
task = client.session.prompt_async(
    session_id=session_id,
    message=prompt_text,
)

# Poll or use SSE for completion
async for event in subscriber.subscribe():
    if event.type == "session.idle":
        result = await client.session.get(session_id)
        break
```

### Message Loop Pattern

```python
async def execute_prompt(
    client: OpenCodeClient,
    session_id: str,
    prompt: str,
    *,
    deadline: datetime | None = None,
) -> PromptResponse:
    # Start async prompt
    await client.session.prompt_async(session_id, prompt)

    # Subscribe to events
    async for event in EventSubscriber(client.base_url, session_id).subscribe():
        if deadline and datetime.now(UTC) > deadline:
            await client.session.abort(session_id)
            raise DeadlineExceededError()

        match event.type:
            case "session.idle":
                break
            case "error":
                raise OpenCodeError(event.message)

    # Extract result
    session = await client.session.get(session_id)
    return extract_response(session)
```

## Structured Output

### Tool-Based Output Contract

OpenCode doesn't provide Claude-Agent-SDK-style `StructuredOutput` tool natively.
Implement structured output via an MCP tool:

```python
@FrozenDataclass()
class WinkEmitParams:
    output: dict[str, Any]  # JSON-serialized output

def wink_emit_handler(
    params: WinkEmitParams,
    *,
    context: ToolContext,
) -> ToolResult[None]:
    """Emit structured output and signal completion."""
    # Validate against output schema
    try:
        parsed = serde.parse(context.output_type, params.output)
    except Exception as e:
        return ToolResult.error(f"Invalid output format: {e}")

    # Store in session for extraction
    context.session.dispatch(StructuredOutputEmitted(value=parsed))
    return ToolResult.ok(None, message="Output recorded. Task complete.")
```

### Prompt Instructions for Structured Output

Include in prompt template:

```markdown
## Output Format

When you have completed the task, call the `wink_emit` tool with your result:

```json
{
  "output": {
    // Your result matching the expected schema
  }
}
```

Do not provide any other response after calling `wink_emit`.
```

## Events

| Event | When |
|-------|------|
| `PromptRendered` | After render |
| `ToolInvoked` | Each tool call (native + bridged) |
| `PermissionHandled` | Permission request processed |
| `PromptExecuted` | Completion (includes `TokenUsage`) |

## Usage Patterns

### Basic Evaluation

```python
from weakincentives.adapters.opencode import (
    OpenCodeAdapter,
    OpenCodeClientConfig,
    OpenCodeModelConfig,
)

adapter = OpenCodeAdapter(
    model_config=OpenCodeModelConfig(model="anthropic/claude-sonnet-4-20250514"),
    client_config=OpenCodeClientConfig(
        permission_mode="auto_approve",
    ),
)

response = adapter.evaluate(prompt, session=session)
```

### Isolated Workspace

```python
from weakincentives.adapters.opencode import (
    OpenCodeAdapter,
    OpenCodeClientConfig,
    OpenCodeIsolationConfig,
    OpenCodeWorkspaceSection,
)

workspace = OpenCodeWorkspaceSection(
    session=session,
    mounts=(HostMount(host_path="/path/to/repo", mount_path="repo"),),
    allowed_host_roots=("/path/to",),
)

adapter = OpenCodeAdapter(
    client_config=OpenCodeClientConfig(
        isolation=OpenCodeIsolationConfig(
            config_content=json.dumps({
                "provider": "anthropic",
                "permissions": {"allow": ["read:*", "write:workspace/*"]},
            }),
        ),
    ),
)
```

### Policy-Based Permissions

```python
from weakincentives.prompt.policy import ReadBeforeWritePolicy

adapter = OpenCodeAdapter(
    client_config=OpenCodeClientConfig(
        permission_mode="policy_based",
    ),
)

template = PromptTemplate(
    sections=[
        workspace,
        MarkdownSection(
            key="task",
            template="Review and fix the code.",
            policies=[ReadBeforeWritePolicy()],
        ),
    ],
)
```

### Custom MCP Tools

```python
# WINK tools automatically bridged to OpenCode via MCP
template = PromptTemplate(
    sections=[
        MarkdownSection(
            key="tools",
            template="Use the search tool to find relevant code.",
            tools=[semantic_search_tool, run_tests_tool],
        ),
    ],
)

# Tools exposed as MCP tools under "wink" server
adapter.evaluate(Prompt(template), session=session)
```

## Operational Notes

### Network Considerations

- OpenCode server binds to localhost by default
- For remote access, use `--hostname 0.0.0.0` with `server_password`
- Proxy environments may require `NO_PROXY=localhost,127.0.0.1`

### Plugin Caching

OpenCode installs npm plugins to `~/.cache/opencode/node_modules/`. For hermetic
isolation:

- Pre-bake plugin dependencies in container images
- Or set `OPENCODE_CACHE_DIR` to isolated temp directory
- Or disable plugins via config

### Server Version Compatibility

```python
health = server.health_check()
if not is_compatible_version(health.version, SDK_VERSION):
    raise IncompatibleVersionError(
        f"Server {health.version} incompatible with SDK {SDK_VERSION}"
    )
```

### Debugging

- Server logs: Captured stderr if `suppress_stderr=False`
- Event stream: Log all SSE events in debug mode
- Session state: `GET /session/:id` returns full session state
- Message history: `GET /session/:id/messages` for conversation

### Error Recovery

| Error | Recovery |
|-------|----------|
| Server startup timeout | Retry with increased timeout |
| SSE disconnection | Reconnect and resume from last event |
| Permission timeout | Auto-deny and log warning |
| Tool execution failure | Rollback and return error to model |

## Comparison: ACP Alternative

OpenCode also supports ACP (Agent Communication Protocol) via stdio:

```bash
opencode acp
```

ACP provides:
- JSON-RPC over stdio (similar to MCP)
- Process lifecycle more similar to Claude Agent SDK
- No HTTP server management

Consider ACP if:
- HTTP server management is undesirable
- Single-session, single-process model preferred
- Direct process communication preferred

The HTTP adapter is recommended for:
- Multi-session scenarios
- Long-running server processes
- Web/IDE client integration

## Limitations

- **Server lifecycle**: Adapter must manage `opencode serve` subprocess
- **No native structured output**: Requires tool-based workaround
- **Permission latency**: SSE-based permission handling adds latency
- **Plugin isolation**: Plugin cache may conflict across sessions
- **Config complexity**: Multiple config sources require careful isolation
- **Event ordering**: SSE events may arrive out of order under load

## Related Specifications

- `specs/ADAPTERS.md` - Provider adapter protocol
- `specs/CLAUDE_AGENT_SDK.md` - Claude Agent SDK adapter (reference)
- `specs/TOOLS.md` - Tool registration and policies
- `specs/WORKSPACE.md` - Workspace tools and isolation
- `specs/SESSIONS.md` - Session lifecycle and events

# OpenCode HTTP API Adapter Specification

> **SDK**: `opencode-sdk>=0.3.0`

## Purpose

`OpenCodeHttpAdapter` evaluates prompts via the OpenCode HTTP REST API. This
approach requires WINK to manage the OpenCode server lifecycle externally—spawning
`opencode serve`, monitoring health, routing requests, and terminating the process.

**Implementation:** `src/weakincentives/adapters/opencode/http/`

## Background

OpenCode's HTTP API is the primary integration surface for external clients:

- **REST Endpoints**: Session CRUD, message handling, MCP management
- **SSE Events**: Real-time event streaming for tool calls, permissions, updates
- **OpenAPI Spec**: Auto-generated client SDKs via `/doc` endpoint

```
┌────────────────────────┐         ┌────────────────────────────┐
│ WINK OpenCodeHttpAdapter│         │ OpenCode Server            │
│ ├─ ServerManager       │──HTTP──▶│ ├─ /session (CRUD)         │
│ ├─ SessionClient       │         │ ├─ /session/:id/message    │
│ ├─ EventSubscriber     │◀──SSE───│ ├─ /session/:id/permissions│
│ └─ MCP Server          │         │ ├─ /event (SSE)            │
└────────────────────────┘         │ ├─ /mcp (registration)     │
                                   │ └─ /global/health          │
                                   └────────────────────────────┘
```

## Architecture Overview

### Lifecycle Comparison: Claude Agent SDK vs OpenCode HTTP

| Phase | Claude Agent SDK | OpenCode HTTP |
|-------|------------------|---------------|
| **Startup** | SDK spawns Claude Code internally | WINK spawns `opencode serve` subprocess |
| **Health** | SDK handles internally | WINK polls `/global/health` until ready |
| **Connect** | `client.connect(prompt)` | `POST /session` + `POST /session/:id/message` |
| **Events** | In-process hook callbacks | SSE subscription to `/event` |
| **Tools** | SDK MCP helper (`create_sdk_mcp_server`) | External MCP server + config injection |
| **Permissions** | `bypassPermissions` mode | `POST /session/:id/permissions/:permissionID` |
| **Shutdown** | SDK handles via `client.disconnect()` | WINK terminates subprocess |

### Key Architectural Differences

1. **Server Process**: WINK owns full lifecycle (spawn → health → terminate)
2. **Network Boundary**: All communication over HTTP (latency, error handling)
3. **Event Streaming**: SSE requires connection management and reconnection logic
4. **State Sync**: No in-process hooks; must reconstruct from SSE events

## Components

### 1. ServerManager

Manages the `opencode serve` subprocess lifecycle.

```python
@FrozenDataclass()
class ServerManagerConfig:
    """Configuration for OpenCode server process."""
    hostname: str = "127.0.0.1"
    port: int | None = None  # None = auto-assign free port
    cwd: Path | None = None  # Working directory
    password: str | None = None  # HTTP Basic Auth
    startup_timeout: float = 30.0  # Seconds to wait for health
    shutdown_timeout: float = 5.0  # Seconds for graceful shutdown
    health_poll_interval: float = 0.5  # Seconds between health checks
    env: Mapping[str, str] | None = None  # Additional environment


@FrozenDataclass()
class ServerManager:
    """Manages OpenCode server subprocess."""
    config: ServerManagerConfig
    process: subprocess.Popen[bytes]
    base_url: str
    assigned_port: int

    @classmethod
    def spawn(cls, config: ServerManagerConfig) -> ServerManager:
        """Spawn server and wait for health.

        Lifecycle:
        1. Select free port if not specified
        2. Build environment (merge config.env + isolation vars)
        3. Spawn: opencode serve --hostname X --port Y [--password Z]
        4. Poll GET /global/health until healthy or timeout
        5. Return ServerManager instance

        Raises:
            ServerStartupError: If health check times out or process exits
        """
        ...

    def health_check(self) -> ServerHealth:
        """Check server health.

        Returns:
            ServerHealth with version, uptime, status

        Raises:
            ServerUnavailableError: If server is not responding
        """
        ...

    def terminate(self) -> None:
        """Gracefully terminate server.

        Lifecycle:
        1. Send SIGTERM
        2. Wait up to shutdown_timeout
        3. Send SIGKILL if still running
        4. Clean up resources
        """
        ...

    def __enter__(self) -> ServerManager:
        return self

    def __exit__(self, *exc: object) -> None:
        self.terminate()
```

### 2. SessionClient

HTTP client for OpenCode session operations.

```python
@FrozenDataclass()
class SessionClient:
    """HTTP client for OpenCode session management."""
    base_url: str
    auth: tuple[str, str] | None = None  # (username, password)
    timeout: float = 30.0

    def create_session(self, *, cwd: str | None = None) -> SessionInfo:
        """Create new session.

        POST /session
        {
            "cwd": "/path/to/workspace"
        }

        Returns:
            SessionInfo with session_id, created_at
        """
        ...

    def send_message(
        self,
        session_id: str,
        message: str,
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> MessageResponse:
        """Send message and wait for response (synchronous).

        POST /session/:id/message
        {
            "content": "...",
            "system": "...",
            "tools": [...]
        }

        Returns:
            MessageResponse with result, messages, usage
        """
        ...

    def send_prompt_async(
        self,
        session_id: str,
        message: str,
    ) -> AsyncTaskHandle:
        """Send message without waiting (asynchronous).

        POST /session/:id/prompt_async
        {
            "content": "..."
        }

        Returns:
            AsyncTaskHandle with task_id for tracking
        """
        ...

    def abort_session(self, session_id: str) -> None:
        """Abort running session.

        POST /session/:id/abort
        """
        ...

    def respond_to_permission(
        self,
        session_id: str,
        permission_id: str,
        *,
        allowed: bool,
        remember: bool = False,
    ) -> None:
        """Respond to permission request.

        POST /session/:id/permissions/:permissionID
        {
            "allowed": true,
            "remember": false
        }
        """
        ...

    def get_session(self, session_id: str) -> SessionState:
        """Get current session state.

        GET /session/:id
        """
        ...

    def register_mcp_server(
        self,
        name: str,
        config: McpServerConfig,
    ) -> None:
        """Register MCP server dynamically.

        POST /mcp
        {
            "name": "wink",
            "config": {...}
        }
        """
        ...
```

### 3. EventSubscriber

SSE event stream subscription and parsing.

```python
@FrozenDataclass()
class EventSubscriber:
    """Subscribes to OpenCode SSE event stream."""
    base_url: str
    session_id: str | None = None  # None = global events
    reconnect_attempts: int = 3
    reconnect_delay: float = 1.0

    async def subscribe(self) -> AsyncIterator[OpenCodeEvent]:
        """Subscribe to event stream.

        GET /event (session-scoped)
        GET /global/event (server-wide)

        Yields:
            OpenCodeEvent instances parsed from SSE data

        Handles:
            - Automatic reconnection on disconnect
            - Event ID tracking for resume
            - Heartbeat/keepalive handling
        """
        ...


@FrozenDataclass()
class OpenCodeEvent:
    """Parsed SSE event from OpenCode."""
    type: str  # "tool.execute.before", "permission.requested", etc.
    session_id: str | None
    timestamp: datetime
    data: dict[str, Any]
```

### 4. WinkMcpServer

External MCP server exposing WINK tools to OpenCode.

```python
class WinkMcpServer:
    """MCP server for WINK tool bridging.

    Unlike Claude Agent SDK (which provides create_sdk_mcp_server),
    OpenCode requires an external MCP server. Options:

    1. Stdio MCP: Spawned as subprocess by OpenCode
    2. HTTP MCP: Connected via network URL

    Stdio is recommended for single-session isolation.
    """

    def __init__(
        self,
        tools: tuple[BridgedTool, ...],
        session: SessionProtocol,
        prompt: PromptProtocol[Any],
    ) -> None: ...

    # JSON-RPC handlers
    def handle_tools_list(self) -> list[ToolSchema]:
        """Handle tools/list request."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
            }
            for tool in self._tools
        ]

    def handle_tools_call(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle tools/call request with transactional semantics."""
        tool = self._find_tool(name)
        with tool_transaction(
            self._session,
            self._prompt.resources.context,
            tag=f"tool:{name}",
        ) as snapshot:
            result = tool.execute(arguments, snapshot=snapshot)
            return self._format_mcp_result(result)

    def serve_stdio(self) -> NoReturn:
        """Run MCP server over stdio (for subprocess spawning)."""
        # Read JSON-RPC from stdin, write to stdout
        ...

    def serve_http(self, port: int) -> NoReturn:
        """Run MCP server over HTTP."""
        ...
```

### 5. PermissionHandler

Automated permission handling for non-interactive execution.

```python
class PermissionMode(StrEnum):
    """Permission handling strategies."""
    AUTO_APPROVE = "auto_approve"  # Allow all (requires isolation)
    AUTO_APPROVE_REMEMBER = "auto_approve_remember"  # Allow + remember
    POLICY_BASED = "policy_based"  # Evaluate via WINK policies
    DENY_ALL = "deny_all"  # Reject all


@FrozenDataclass()
class PermissionHandler:
    """Handles OpenCode permission requests."""
    mode: PermissionMode
    policies: tuple[ToolPolicy, ...] = ()
    allowed_commands: frozenset[str] = frozenset()
    allowed_paths: frozenset[str] = frozenset()

    def decide(self, event: OpenCodeEvent) -> PermissionDecision:
        """Decide how to respond to permission request.

        Permission types from OpenCode:
        - read:path - File read access
        - write:path - File write access
        - execute:command - Shell command execution
        - network:domain - Network access

        Returns:
            PermissionDecision with allowed, remember flags
        """
        match self.mode:
            case PermissionMode.AUTO_APPROVE:
                return PermissionDecision(allowed=True, remember=False)

            case PermissionMode.AUTO_APPROVE_REMEMBER:
                return PermissionDecision(allowed=True, remember=True)

            case PermissionMode.POLICY_BASED:
                return self._evaluate_policies(event)

            case PermissionMode.DENY_ALL:
                return PermissionDecision(allowed=False, remember=False)

    def _evaluate_policies(self, event: OpenCodeEvent) -> PermissionDecision:
        """Evaluate permission against WINK policies."""
        # Map OpenCode permission to synthetic tool call
        # Check against ReadBeforeWritePolicy, etc.
        ...
```

## Configuration

### OpenCodeHttpClientConfig

```python
@FrozenDataclass()
class OpenCodeHttpClientConfig:
    """Configuration for OpenCode HTTP adapter."""
    # Server management
    hostname: str = "127.0.0.1"
    port: int | None = None
    server_password: str | None = None
    server_timeout: float = 30.0

    # Session behavior
    permission_mode: PermissionMode = PermissionMode.AUTO_APPROVE
    max_turns: int | None = None

    # Isolation
    isolation: OpenCodeIsolationConfig | None = None

    # Debugging
    suppress_stderr: bool = True
    log_sse_events: bool = False
```

### OpenCodeModelConfig

```python
@FrozenDataclass()
class OpenCodeModelConfig:
    """Model configuration for OpenCode."""
    model: str = "anthropic/claude-sonnet-4-20250514"
    provider: str | None = None  # anthropic, openai, bedrock, etc.
    temperature: float | None = None
    max_tokens: int | None = None
```

### OpenCodeIsolationConfig

```python
@FrozenDataclass()
class OpenCodeIsolationConfig:
    """Isolation configuration for hermetic sessions."""
    # Config injection (avoids host ~/.config/opencode)
    config_content: str | None = None  # Inline JSON config
    config_dir: Path | None = None  # Override config directory

    # Environment isolation
    env: Mapping[str, str] | None = None
    include_host_env: bool = False
    cache_dir: Path | None = None  # Isolated plugin cache

    # Network
    no_proxy: str = "localhost,127.0.0.1"
```

## Adapter Implementation

### OpenCodeHttpAdapter

```python
class OpenCodeHttpAdapter[OutputT](ProviderAdapter[OutputT]):
    """Adapter using OpenCode HTTP API."""

    def __init__(
        self,
        *,
        model_config: OpenCodeModelConfig | None = None,
        client_config: OpenCodeHttpClientConfig | None = None,
    ) -> None:
        self._model_config = model_config or OpenCodeModelConfig()
        self._client_config = client_config or OpenCodeHttpClientConfig()

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
    ) -> PromptResponse[OutputT]:
        """Evaluate prompt via OpenCode HTTP API.

        Lifecycle:
        1. Create isolated workspace (temp dir with mounts)
        2. Generate OpenCode config with MCP server registration
        3. Spawn OpenCode server (ServerManager.spawn)
        4. Create session (SessionClient.create_session)
        5. Start SSE subscription (EventSubscriber.subscribe)
        6. Send prompt (SessionClient.send_prompt_async)
        7. Process events until session.idle:
           - Handle tool.execute.* for state sync
           - Handle permission.requested for auto-response
           - Handle message.updated for streaming
        8. Extract result from final session state
        9. Terminate server (ServerManager.terminate)
        10. Clean up workspace

        Error handling:
        - Server startup timeout → ServerStartupError
        - SSE disconnect → Reconnect or abort
        - Permission timeout → Auto-deny + log warning
        - Deadline exceeded → Abort session + terminate
        """
        return run_async(
            self._evaluate_async(
                prompt,
                session=session,
                deadline=deadline,
                budget_tracker=budget_tracker or (
                    BudgetTracker(budget) if budget else None
                ),
                heartbeat=heartbeat,
                run_context=run_context,
            )
        )

    async def _evaluate_async(
        self,
        prompt: Prompt[OutputT],
        *,
        session: SessionProtocol,
        deadline: Deadline | None,
        budget_tracker: BudgetTracker | None,
        heartbeat: Heartbeat | None,
        run_context: RunContext | None,
    ) -> PromptResponse[OutputT]:
        """Async implementation."""
        # 1. Render prompt
        rendered = prompt.render(session=session)

        # 2. Create workspace
        workspace = self._create_workspace(prompt)

        # 3. Generate config with MCP registration
        config = self._generate_config(rendered.tools)

        # 4. Spawn server
        server_config = ServerManagerConfig(
            hostname=self._client_config.hostname,
            port=self._client_config.port,
            cwd=workspace.temp_dir,
            env=self._build_env(config),
            startup_timeout=self._client_config.server_timeout,
        )

        try:
            with ServerManager.spawn(server_config) as server:
                return await self._run_with_server(
                    server=server,
                    prompt=prompt,
                    rendered=rendered,
                    session=session,
                    deadline=deadline,
                    budget_tracker=budget_tracker,
                    heartbeat=heartbeat,
                    run_context=run_context,
                )
        finally:
            workspace.cleanup()

    async def _run_with_server(
        self,
        *,
        server: ServerManager,
        prompt: Prompt[OutputT],
        rendered: RenderedPrompt[OutputT],
        session: SessionProtocol,
        deadline: Deadline | None,
        budget_tracker: BudgetTracker | None,
        heartbeat: Heartbeat | None,
        run_context: RunContext | None,
    ) -> PromptResponse[OutputT]:
        """Run evaluation with active server."""
        client = SessionClient(
            base_url=server.base_url,
            auth=self._get_auth(),
        )

        # Create OpenCode session
        oc_session = client.create_session(
            cwd=str(server.config.cwd),
        )

        # Start event subscription
        subscriber = EventSubscriber(
            base_url=server.base_url,
            session_id=oc_session.session_id,
        )

        # Permission handler
        permission_handler = PermissionHandler(
            mode=self._client_config.permission_mode,
        )

        # Send prompt asynchronously
        client.send_prompt_async(
            session_id=oc_session.session_id,
            message=rendered.text,
        )

        # Process events
        snapshots: dict[str, CompositeSnapshot] = {}

        async for event in subscriber.subscribe():
            # Check deadline
            if deadline and deadline.remaining().total_seconds() <= 0:
                client.abort_session(oc_session.session_id)
                raise DeadlineExceededError()

            # Beat heartbeat
            if heartbeat:
                heartbeat.beat()

            # Handle event
            match event.type:
                case "tool.execute.before":
                    # Snapshot for potential rollback
                    snapshots[event.data["tool_id"]] = create_snapshot(
                        session,
                        prompt.resources.context,
                        tag=f"tool:{event.data['tool_name']}",
                    )

                case "tool.execute.after":
                    # Dispatch ToolInvoked
                    tool_id = event.data["tool_id"]
                    session.dispatcher.dispatch(
                        ToolInvoked(
                            tool_name=event.data["tool_name"],
                            params=event.data.get("params"),
                            result=event.data.get("result"),
                        )
                    )
                    snapshots.pop(tool_id, None)

                case "permission.requested":
                    # Auto-respond to permission
                    decision = permission_handler.decide(event)
                    client.respond_to_permission(
                        session_id=oc_session.session_id,
                        permission_id=event.data["permission_id"],
                        allowed=decision.allowed,
                        remember=decision.remember,
                    )

                case "session.idle":
                    # Agent finished
                    break

                case "error":
                    raise OpenCodeError(event.data.get("message", "Unknown error"))

        # Extract result
        final_state = client.get_session(oc_session.session_id)
        return self._extract_response(final_state, rendered)
```

## Event Mapping

### OpenCode Events → WINK Events

| OpenCode Event | WINK Action |
|----------------|-------------|
| `tool.execute.before` | Create snapshot |
| `tool.execute.after` | Dispatch `ToolInvoked`, clear snapshot |
| `permission.requested` | Call `PermissionHandler.decide`, respond |
| `message.created` | Log |
| `message.updated` | Log (streaming) |
| `session.idle` | Exit event loop |
| `session.compacted` | Log |
| `error` | Raise `OpenCodeError` |

### State Synchronization

Unlike Claude Agent SDK hooks (which are synchronous callbacks in the same
process), HTTP adapter must reconstruct state from SSE events:

```python
async def sync_state_from_events(
    subscriber: EventSubscriber,
    session: SessionProtocol,
    prompt: PromptProtocol[Any],
) -> None:
    """Synchronize WINK session state from SSE events."""
    pending_snapshots: dict[str, CompositeSnapshot] = {}

    async for event in subscriber.subscribe():
        match event.type:
            case "tool.execute.before":
                # Native tool starting - snapshot state
                tool_id = event.data["tool_id"]
                pending_snapshots[tool_id] = create_snapshot(
                    session,
                    prompt.resources.context,
                    tag=f"native:{event.data['tool_name']}",
                )

            case "tool.execute.after":
                # Native tool completed
                tool_id = event.data["tool_id"]
                if not event.data.get("success", True):
                    # Rollback on failure
                    snapshot = pending_snapshots.get(tool_id)
                    if snapshot:
                        restore_snapshot(session, prompt.resources.context, snapshot)
                pending_snapshots.pop(tool_id, None)

                # Dispatch event for session reducers
                session.dispatcher.dispatch(
                    ToolInvoked(
                        tool_name=event.data["tool_name"],
                        params=event.data.get("params"),
                        result=ToolResult(
                            success=event.data.get("success", True),
                            message=event.data.get("message"),
                        ),
                    )
                )
```

## Structured Output

### Tool-Based Contract

OpenCode doesn't provide native structured output. Implement via MCP tool:

```python
@FrozenDataclass()
class WinkEmitParams:
    """Parameters for structured output emission."""
    output: dict[str, Any]


def create_wink_emit_tool(output_type: type[OutputT]) -> Tool[WinkEmitParams, None]:
    """Create tool for structured output emission."""

    def handler(
        params: WinkEmitParams,
        *,
        context: ToolContext,
    ) -> ToolResult[None]:
        # Validate against output schema
        try:
            parsed = serde.parse(output_type, params.output)
        except Exception as e:
            return ToolResult.error(f"Invalid output format: {e}")

        # Store in session for extraction
        context.session.dispatch(StructuredOutputEmitted(value=parsed))
        return ToolResult.ok(None, message="Output recorded. Task complete.")

    return Tool(
        name="wink_emit",
        description="Emit structured output when task is complete",
        handler=handler,
    )
```

### Prompt Template Addition

```markdown
## Output Format

When you have completed the task, you MUST call the `wink_emit` tool with your
result. The output must match this JSON schema:

{output_schema}

Example:
```json
wink_emit({"output": {"status": "success", "files_modified": ["main.py"]}})
```

Do not provide any other response after calling `wink_emit`.
```

## Workspace Isolation

### Per-Session Server Pattern

For hermetic isolation, spawn one server per evaluation:

```python
def create_isolated_session(
    workspace_mounts: tuple[HostMount, ...],
    isolation: OpenCodeIsolationConfig,
) -> tuple[ServerManager, Path]:
    """Create isolated server with workspace."""
    # 1. Create temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="wink-opencode-"))

    # 2. Copy mounted files
    for mount in workspace_mounts:
        copy_mount(mount, temp_dir)

    # 3. Generate inline config
    config_content = json.dumps({
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "mcp": {
            "wink": {
                "command": "python",
                "args": ["-m", "weakincentives.adapters.opencode.mcp_server"],
                "env": {"WINK_SOCKET": str(temp_dir / "wink.sock")},
            }
        },
    })

    # 4. Build environment
    env = {
        "HOME": str(temp_dir / ".home"),
        "OPENCODE_CONFIG_CONTENT": config_content,
        "NO_PROXY": "localhost,127.0.0.1",
        **(isolation.env or {}),
    }

    # 5. Spawn server
    server = ServerManager.spawn(
        ServerManagerConfig(
            cwd=temp_dir,
            env=env,
        )
    )

    return server, temp_dir
```

## Error Handling

### Exception Hierarchy

```python
class OpenCodeError(PromptEvaluationError):
    """Base for OpenCode-specific errors."""
    pass


class ServerStartupError(OpenCodeError):
    """Server failed to start within timeout."""
    exit_code: int | None
    stderr: str | None


class ServerUnavailableError(OpenCodeError):
    """Server is not responding to health checks."""
    base_url: str
    last_error: Exception | None


class SessionError(OpenCodeError):
    """Session-level error."""
    session_id: str
    operation: str


class PermissionDeniedError(OpenCodeError):
    """Permission was denied (policy_based or deny_all mode)."""
    permission_type: str
    resource: str
```

### Retry Logic

```python
async def with_sse_reconnect[T](
    subscriber: EventSubscriber,
    handler: Callable[[AsyncIterator[OpenCodeEvent]], Awaitable[T]],
    *,
    max_attempts: int = 3,
    backoff_base: float = 1.0,
) -> T:
    """Execute handler with SSE reconnection."""
    attempt = 0
    last_event_id: str | None = None

    while attempt < max_attempts:
        try:
            events = subscriber.subscribe(last_event_id=last_event_id)
            return await handler(events)
        except SSEDisconnectedError as e:
            attempt += 1
            last_event_id = e.last_event_id
            if attempt >= max_attempts:
                raise
            await asyncio.sleep(backoff_base * (2 ** attempt))
```

## Implementation Effort Analysis

### Components to Build

| Component | Complexity | LOC Estimate | Notes |
|-----------|------------|--------------|-------|
| `ServerManager` | Medium | ~200 | Subprocess management, health polling |
| `SessionClient` | Low | ~150 | HTTP client wrapper |
| `EventSubscriber` | Medium | ~250 | SSE parsing, reconnection |
| `WinkMcpServer` | High | ~400 | Full MCP server implementation |
| `PermissionHandler` | Low | ~100 | Policy evaluation |
| `OpenCodeHttpAdapter` | High | ~500 | Main orchestration |
| Tests | High | ~800 | Integration tests with mocks |
| **Total** | | ~2400 | |

### Dependencies Required

- `httpx` - Async HTTP client
- `sse-starlette` or custom SSE parsing
- MCP server library (or implement from scratch)

### Risk Factors

1. **SSE Reliability**: Connection drops, event ordering, reconnection
2. **Server Lifecycle**: Process management edge cases (zombies, crashes)
3. **MCP Server**: No SDK helper; must implement full MCP protocol
4. **Latency**: Network overhead for every operation
5. **Testing**: Requires real OpenCode server or extensive mocking

## Operational Notes

### Network Considerations

- Always bind to localhost unless explicitly configured otherwise
- Use `NO_PROXY=localhost,127.0.0.1` to avoid corporate proxy issues
- Consider SSE connection timeouts and keepalives

### Debugging

```python
# Enable verbose logging
client_config = OpenCodeHttpClientConfig(
    log_sse_events=True,
    suppress_stderr=False,
)

# Server logs available at stderr
# SSE events logged at DEBUG level
```

### Performance

- Server startup adds ~2-5s overhead per evaluation
- SSE subscription adds ~100ms latency per event
- Consider server pooling for high-volume scenarios (not recommended for isolation)

## Related Specifications

- `specs/OPENCODE_ACP.md` - Alternative ACP-based adapter
- `specs/ADAPTERS.md` - Provider adapter protocol
- `specs/CLAUDE_AGENT_SDK.md` - Reference implementation
- `specs/TOOLS.md` - Tool bridging patterns

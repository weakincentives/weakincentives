# OpenCode ACP Adapter Specification

> **Protocol**: Agent Client Protocol (ACP) v1
> **SDK**: `agent-client-protocol>=0.10.0`

## Purpose

`OpenCodeAcpAdapter` evaluates prompts via the Agent Client Protocol (ACP) over
stdio. This approach spawns `opencode acp` as a subprocess and communicates via
JSON-RPC 2.0, providing a lifecycle model closely matching Claude Agent SDK.

**Implementation:** `src/weakincentives/adapters/opencode/acp/`

## Background

The Agent Client Protocol (ACP) is an open standard that enables any editor to
connect with any AI coding agent. OpenCode supports ACP natively via the
`opencode acp` command.

### Why ACP is the Recommended Approach

| Aspect | HTTP API | ACP (Recommended) |
|--------|----------|-------------------|
| **Process Model** | HTTP server + client | Subprocess over stdio |
| **Claude SDK Similarity** | Low | High |
| **Lifecycle** | Server spawn → health → terminate | Process spawn → initialize → disconnect |
| **Events** | SSE subscription | JSON-RPC notifications |
| **Tool Bridging** | External MCP server | ACP tool registration |
| **State Sync** | Reconstruct from SSE | Native in protocol |
| **Implementation** | ~2400 LOC | ~1200 LOC |
| **Risk** | High (SSE, server mgmt) | Low (stdio, similar to SDK) |

```
┌────────────────────────┐         ┌────────────────────────────┐
│ WINK OpenCodeAcpAdapter│         │ OpenCode (subprocess)      │
│ ├─ ProcessManager      │──stdio──│ ├─ ACP Protocol Handler    │
│ ├─ AcpClient           │◀──JSON──│ ├─ Session Management      │
│ └─ NotificationHandler │  RPC    │ └─ Tool Execution          │
└────────────────────────┘         └────────────────────────────┘
```

## Architecture Comparison

### Claude Agent SDK vs OpenCode ACP

| Phase | Claude Agent SDK | OpenCode ACP |
|-------|------------------|--------------|
| **Startup** | `ClaudeSDKClient(options)` | `subprocess.Popen(["opencode", "acp"])` |
| **Initialize** | SDK handles internally | `initialize` request with capabilities |
| **Session** | `client.connect(prompt)` | `session/new` + `session/prompt` |
| **Events** | Hook callbacks (PreToolUse, etc.) | `session/update` notifications |
| **Streaming** | `receive_messages()` async iterator | `session/update` with message chunks |
| **Tools** | MCP server via `create_sdk_mcp_server` | Tool schemas in `initialize` response |
| **Shutdown** | `client.disconnect()` | Process termination |

The key insight: **ACP's process model mirrors Claude SDK's architecture**.
Both spawn a subprocess that handles the agent loop internally, with the
caller receiving structured updates.

## ACP Protocol Overview

### JSON-RPC 2.0 Foundation

ACP uses JSON-RPC 2.0 with two message types:

- **Methods**: Request-response pairs (`id` required)
- **Notifications**: One-way messages (no `id`, no response expected)

Communication is bidirectional—both sides can initiate requests.

### Core Methods

| Method | Direction | Purpose |
|--------|-----------|---------|
| `initialize` | Client → Agent | Negotiate capabilities |
| `session/new` | Client → Agent | Create session |
| `session/prompt` | Client → Agent | Send user prompt |
| `session/cancel` | Client → Agent | Cancel running prompt |
| `session/set_mode` | Client → Agent | Change session mode |

### Notifications

| Notification | Direction | Purpose |
|--------------|-----------|---------|
| `session/update` | Agent → Client | Streaming updates |

### Message Flow

```
Client                              Agent (OpenCode)
   │                                     │
   │── initialize ─────────────────────▶│
   │◀─ InitializeResponse ──────────────│
   │                                     │
   │── session/new ────────────────────▶│
   │◀─ NewResponse (session_id) ────────│
   │                                     │
   │── session/prompt ─────────────────▶│
   │◀─ session/update (thought) ────────│ (notifications)
   │◀─ session/update (message) ────────│
   │◀─ session/update (tool_call) ──────│
   │◀─ session/update (message) ────────│
   │◀─ PromptResponse ──────────────────│ (final response)
   │                                     │
   │── session/cancel ─────────────────▶│ (if needed)
```

## Components

### 1. ProcessManager

Manages the `opencode acp` subprocess.

```python
@FrozenDataclass()
class ProcessManagerConfig:
    """Configuration for OpenCode ACP subprocess."""
    cwd: Path | None = None  # Working directory
    env: Mapping[str, str] | None = None  # Environment overrides
    startup_timeout: float = 10.0  # Seconds for initialize
    shutdown_timeout: float = 5.0  # Graceful shutdown


@FrozenDataclass()
class ProcessManager:
    """Manages OpenCode ACP subprocess lifecycle."""
    config: ProcessManagerConfig
    process: subprocess.Popen[bytes]
    stdin: IO[bytes]
    stdout: IO[bytes]

    @classmethod
    def spawn(cls, config: ProcessManagerConfig) -> ProcessManager:
        """Spawn OpenCode ACP subprocess.

        Lifecycle:
        1. Build environment (merge config.env + isolation vars)
        2. Spawn: opencode acp
        3. Return ProcessManager with stdio handles

        No health check needed - initialize request validates readiness.
        """
        env = {
            **os.environ,
            **(config.env or {}),
        }

        process = subprocess.Popen(
            ["opencode", "acp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=config.cwd,
            env=env,
        )

        return cls(
            config=config,
            process=process,
            stdin=process.stdin,
            stdout=process.stdout,
        )

    def send(self, message: dict[str, Any]) -> None:
        """Send JSON-RPC message to subprocess."""
        data = json.dumps(message).encode() + b"\n"
        self.stdin.write(data)
        self.stdin.flush()

    def receive(self, timeout: float | None = None) -> dict[str, Any]:
        """Receive JSON-RPC message from subprocess.

        Handles both responses and notifications.
        """
        ...

    def terminate(self) -> None:
        """Gracefully terminate subprocess."""
        self.process.terminate()
        try:
            self.process.wait(timeout=self.config.shutdown_timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()

    def __enter__(self) -> ProcessManager:
        return self

    def __exit__(self, *exc: object) -> None:
        self.terminate()
```

### 2. AcpClient

JSON-RPC client for ACP protocol.

```python
@FrozenDataclass()
class AcpClient:
    """ACP protocol client."""
    process: ProcessManager
    request_id: int = 0
    _pending_requests: dict[int, asyncio.Future[dict[str, Any]]]

    def initialize(
        self,
        capabilities: ClientCapabilities,
    ) -> InitializeResponse:
        """Initialize ACP connection.

        Request:
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": 1,
                "clientCapabilities": {...}
            }
        }

        Response:
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "protocolVersion": 1,
                "agentCapabilities": {...}
            }
        }
        """
        self.request_id += 1
        self.process.send({
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": 1,
                "clientCapabilities": capabilities.to_dict(),
            },
        })
        response = self.process.receive(
            timeout=self.process.config.startup_timeout
        )
        return InitializeResponse.from_dict(response["result"])

    def session_new(self, cwd: str | None = None) -> NewResponse:
        """Create new session.

        Request:
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": {
                "cwd": "/path/to/workspace"
            }
        }

        Response:
        {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "sessionId": "session-123"
            }
        }
        """
        self.request_id += 1
        params: dict[str, Any] = {}
        if cwd:
            params["cwd"] = cwd

        self.process.send({
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "session/new",
            "params": params,
        })
        response = self.process.receive()
        return NewResponse.from_dict(response["result"])

    def session_prompt(
        self,
        session_id: str,
        prompt: str,
    ) -> AsyncIterator[SessionUpdate | PromptResponse]:
        """Send prompt and stream updates.

        Request:
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session/prompt",
            "params": {
                "sessionId": "session-123",
                "prompt": "Fix the bug in main.py"
            }
        }

        Notifications (streaming):
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "session-123",
                "type": "thought",
                "content": "Let me analyze..."
            }
        }

        Final Response:
        {
            "jsonrpc": "2.0",
            "id": 3,
            "result": {
                "messages": [...],
                "usage": {...}
            }
        }
        """
        ...

    def session_cancel(self, session_id: str) -> None:
        """Cancel running prompt.

        Request:
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "session/cancel",
            "params": {
                "sessionId": "session-123"
            }
        }
        """
        self.request_id += 1
        self.process.send({
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "session/cancel",
            "params": {"sessionId": session_id},
        })
        # Cancel may not have a response if session already complete
        try:
            self.process.receive(timeout=1.0)
        except TimeoutError:
            pass


@FrozenDataclass()
class ClientCapabilities:
    """Capabilities advertised by client."""
    supports_text_prompts: bool = True
    supports_audio_prompts: bool = False
    supports_streaming: bool = True
    available_tools: tuple[ToolSchema, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "textPrompts": self.supports_text_prompts,
            "audioPrompts": self.supports_audio_prompts,
            "streaming": self.supports_streaming,
            "tools": [t.to_dict() for t in self.available_tools],
        }


@FrozenDataclass()
class InitializeResponse:
    """Response from initialize."""
    protocol_version: int
    agent_capabilities: AgentCapabilities

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InitializeResponse:
        return cls(
            protocol_version=data["protocolVersion"],
            agent_capabilities=AgentCapabilities.from_dict(
                data["agentCapabilities"]
            ),
        )


@FrozenDataclass()
class AgentCapabilities:
    """Capabilities advertised by agent."""
    supports_streaming: bool
    supports_cancel: bool
    supported_modes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentCapabilities:
        return cls(
            supports_streaming=data.get("streaming", False),
            supports_cancel=data.get("cancel", False),
            supported_modes=tuple(data.get("modes", [])),
        )
```

### 3. SessionUpdate Types

```python
class UpdateType(StrEnum):
    """Types of session updates."""
    THOUGHT = "thought"  # Agent's internal reasoning
    MESSAGE = "message"  # Content for user
    TOOL_CALL = "tool_call"  # Tool invocation
    TOOL_RESULT = "tool_result"  # Tool completion
    PLAN = "plan"  # Execution plan update
    MODE_CHANGE = "mode_change"  # Session mode changed
    PERMISSION = "permission"  # Permission request


@FrozenDataclass()
class SessionUpdate:
    """Update notification from agent."""
    session_id: str
    update_type: UpdateType
    content: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionUpdate:
        return cls(
            session_id=data["sessionId"],
            update_type=UpdateType(data["type"]),
            content=data.get("content", {}),
        )


@FrozenDataclass()
class ToolCallUpdate:
    """Tool call notification."""
    tool_id: str
    tool_name: str
    arguments: dict[str, Any]
    status: str  # "pending", "running", "complete", "failed"
    result: Any | None = None

    @classmethod
    def from_content(cls, content: dict[str, Any]) -> ToolCallUpdate:
        return cls(
            tool_id=content["toolId"],
            tool_name=content["toolName"],
            arguments=content.get("arguments", {}),
            status=content.get("status", "pending"),
            result=content.get("result"),
        )
```

### 4. NotificationHandler

Processes streaming notifications and maps to WINK events.

```python
class NotificationHandler:
    """Handles session/update notifications."""

    def __init__(
        self,
        session: SessionProtocol,
        prompt: PromptProtocol[Any],
        heartbeat: Heartbeat | None = None,
    ) -> None:
        self._session = session
        self._prompt = prompt
        self._heartbeat = heartbeat
        self._pending_tools: dict[str, CompositeSnapshot] = {}

    def handle(self, update: SessionUpdate) -> None:
        """Process session update notification.

        Maps ACP updates to WINK patterns:
        - tool_call (pending) → Create snapshot
        - tool_call (complete) → Dispatch ToolInvoked
        - tool_call (failed) → Restore snapshot, dispatch error
        - permission → Handle via policy
        """
        if self._heartbeat:
            self._heartbeat.beat()

        match update.update_type:
            case UpdateType.TOOL_CALL:
                self._handle_tool_call(ToolCallUpdate.from_content(update.content))

            case UpdateType.THOUGHT:
                logger.debug(
                    "opencode.acp.thought",
                    context={"content": update.content.get("text", "")[:200]},
                )

            case UpdateType.MESSAGE:
                logger.debug(
                    "opencode.acp.message",
                    context={"content": update.content.get("text", "")[:200]},
                )

            case UpdateType.PERMISSION:
                self._handle_permission(update.content)

    def _handle_tool_call(self, tool_call: ToolCallUpdate) -> None:
        """Handle tool call lifecycle."""
        match tool_call.status:
            case "pending" | "running":
                # Create snapshot for potential rollback
                if tool_call.tool_id not in self._pending_tools:
                    self._pending_tools[tool_call.tool_id] = create_snapshot(
                        self._session,
                        self._prompt.resources.context,
                        tag=f"tool:{tool_call.tool_name}",
                    )

            case "complete":
                # Success - dispatch event
                self._session.dispatcher.dispatch(
                    ToolInvoked(
                        tool_name=tool_call.tool_name,
                        params=tool_call.arguments,
                        result=ToolResult.ok(
                            tool_call.result,
                            message=str(tool_call.result),
                        ),
                    )
                )
                self._pending_tools.pop(tool_call.tool_id, None)

            case "failed":
                # Failure - restore snapshot
                snapshot = self._pending_tools.pop(tool_call.tool_id, None)
                if snapshot:
                    restore_snapshot(
                        self._session,
                        self._prompt.resources.context,
                        snapshot,
                    )

                self._session.dispatcher.dispatch(
                    ToolInvoked(
                        tool_name=tool_call.tool_name,
                        params=tool_call.arguments,
                        result=ToolResult.error(
                            str(tool_call.result) if tool_call.result else "Failed"
                        ),
                    )
                )

    def _handle_permission(self, content: dict[str, Any]) -> None:
        """Handle permission request."""
        # In ACP, permissions may be handled differently
        # For now, log and continue (agent typically handles internally)
        logger.warning(
            "opencode.acp.permission_request",
            context={"permission": content},
        )
```

## Configuration

### OpenCodeAcpClientConfig

```python
@FrozenDataclass()
class OpenCodeAcpClientConfig:
    """Configuration for OpenCode ACP adapter."""
    # Process configuration
    cwd: Path | None = None
    startup_timeout: float = 10.0
    shutdown_timeout: float = 5.0

    # Isolation
    isolation: OpenCodeIsolationConfig | None = None

    # Debugging
    suppress_stderr: bool = True
    log_notifications: bool = False
```

### OpenCodeModelConfig

```python
@FrozenDataclass()
class OpenCodeModelConfig:
    """Model configuration for OpenCode."""
    model: str = "anthropic/claude-sonnet-4-20250514"
    provider: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
```

### OpenCodeIsolationConfig

```python
@FrozenDataclass()
class OpenCodeIsolationConfig:
    """Isolation for hermetic sessions."""
    config_content: str | None = None
    config_dir: Path | None = None
    env: Mapping[str, str] | None = None
    include_host_env: bool = False
    cache_dir: Path | None = None
```

## Adapter Implementation

### OpenCodeAcpAdapter

```python
class OpenCodeAcpAdapter[OutputT](ProviderAdapter[OutputT]):
    """Adapter using OpenCode ACP protocol.

    This adapter provides the closest architectural match to Claude Agent SDK:
    - Subprocess lifecycle management
    - Streaming updates via notifications
    - Tool bridging via capability negotiation
    - Transactional state synchronization
    """

    def __init__(
        self,
        *,
        model_config: OpenCodeModelConfig | None = None,
        client_config: OpenCodeAcpClientConfig | None = None,
    ) -> None:
        self._model_config = model_config or OpenCodeModelConfig()
        self._client_config = client_config or OpenCodeAcpClientConfig()

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
        """Evaluate prompt via OpenCode ACP.

        Lifecycle (mirrors Claude Agent SDK):
        1. Create isolated workspace (temp dir with mounts)
        2. Spawn OpenCode ACP subprocess
        3. Send initialize request with capabilities
        4. Create session (session/new)
        5. Send prompt (session/prompt)
        6. Process session/update notifications:
           - Map tool_call updates to WINK snapshots/events
           - Forward thought/message updates for streaming
        7. Receive final PromptResponse
        8. Terminate subprocess
        9. Clean up workspace
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

        session.dispatcher.dispatch(
            PromptRendered(
                prompt_ns=prompt.ns,
                prompt_key=prompt.key,
                prompt_name=prompt.name,
                adapter=OPENCODE_ACP_ADAPTER_NAME,
                session_id=getattr(session, "session_id", None),
                render_inputs=(),
                rendered_prompt=rendered.text,
                created_at=datetime.now(UTC),
                descriptor=None,
                run_context=run_context,
            )
        )

        # 2. Create workspace
        workspace = self._create_workspace(prompt)

        # 3. Build process config
        process_config = ProcessManagerConfig(
            cwd=workspace.temp_dir if workspace else self._client_config.cwd,
            env=self._build_env(),
            startup_timeout=self._client_config.startup_timeout,
            shutdown_timeout=self._client_config.shutdown_timeout,
        )

        try:
            with ProcessManager.spawn(process_config) as process:
                return await self._run_with_process(
                    process=process,
                    prompt=prompt,
                    rendered=rendered,
                    session=session,
                    deadline=deadline,
                    budget_tracker=budget_tracker,
                    heartbeat=heartbeat,
                    run_context=run_context,
                )
        finally:
            if workspace:
                workspace.cleanup()

    async def _run_with_process(
        self,
        *,
        process: ProcessManager,
        prompt: Prompt[OutputT],
        rendered: RenderedPrompt[OutputT],
        session: SessionProtocol,
        deadline: Deadline | None,
        budget_tracker: BudgetTracker | None,
        heartbeat: Heartbeat | None,
        run_context: RunContext | None,
    ) -> PromptResponse[OutputT]:
        """Run evaluation with active ACP process."""
        start_time = datetime.now(UTC)

        # Create ACP client
        client = AcpClient(process=process)

        # Initialize with capabilities
        tool_schemas = self._build_tool_schemas(rendered.tools)
        capabilities = ClientCapabilities(
            supports_streaming=True,
            available_tools=tool_schemas,
        )

        logger.debug(
            "opencode.acp.initializing",
            context={"tool_count": len(tool_schemas)},
        )

        init_response = client.initialize(capabilities)

        logger.debug(
            "opencode.acp.initialized",
            context={
                "protocol_version": init_response.protocol_version,
                "supports_streaming": init_response.agent_capabilities.supports_streaming,
            },
        )

        # Create session
        session_response = client.session_new(
            cwd=str(process.config.cwd) if process.config.cwd else None,
        )

        logger.debug(
            "opencode.acp.session_created",
            context={"session_id": session_response.session_id},
        )

        # Create notification handler
        notification_handler = NotificationHandler(
            session=session,
            prompt=prompt,
            heartbeat=heartbeat,
        )

        # Send prompt and process updates
        result_messages: list[Any] = []
        usage: TokenUsage | None = None

        try:
            async for update in client.session_prompt(
                session_id=session_response.session_id,
                prompt=rendered.text,
            ):
                # Check deadline
                if deadline and deadline.remaining().total_seconds() <= 0:
                    client.session_cancel(session_response.session_id)
                    raise DeadlineExceededError()

                # Check budget
                if budget_tracker:
                    try:
                        budget_tracker.check()
                    except Exception:
                        client.session_cancel(session_response.session_id)
                        raise

                if isinstance(update, SessionUpdate):
                    notification_handler.handle(update)
                elif isinstance(update, PromptResponse):
                    # Final response
                    result_messages = update.messages
                    usage = update.usage
                    break

        except Exception:
            # Attempt to cancel on error
            with contextlib.suppress(Exception):
                client.session_cancel(session_response.session_id)
            raise

        end_time = datetime.now(UTC)
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        # Extract result
        result_text, output = self._extract_result(result_messages, rendered)

        # Record token usage
        if budget_tracker and usage:
            budget_tracker.record_cumulative(
                prompt.name or f"{prompt.ns}:{prompt.key}",
                usage,
            )

        response = PromptResponse(
            prompt_name=prompt.name or f"{prompt.ns}:{prompt.key}",
            text=result_text,
            output=output,
        )

        session.dispatcher.dispatch(
            PromptExecuted(
                prompt_name=response.prompt_name,
                adapter=OPENCODE_ACP_ADAPTER_NAME,
                result=response,
                session_id=getattr(session, "session_id", None),
                created_at=datetime.now(UTC),
                usage=usage,
                run_context=run_context,
            )
        )

        logger.info(
            "opencode.acp.complete",
            context={
                "prompt_name": response.prompt_name,
                "duration_ms": duration_ms,
                "input_tokens": usage.input_tokens if usage else None,
                "output_tokens": usage.output_tokens if usage else None,
            },
        )

        return response

    def _build_tool_schemas(
        self,
        tools: tuple[Tool[Any, Any], ...],
    ) -> tuple[ToolSchema, ...]:
        """Convert WINK tools to ACP tool schemas.

        ACP supports tool registration during initialize,
        eliminating the need for external MCP server.
        """
        return tuple(
            ToolSchema(
                name=tool.name,
                description=tool.description,
                input_schema=schema(tool.handler),
            )
            for tool in tools
        )

    def _build_env(self) -> dict[str, str]:
        """Build environment for subprocess."""
        isolation = self._client_config.isolation

        env: dict[str, str] = {}

        if isolation:
            if isolation.include_host_env:
                env.update(os.environ)

            if isolation.env:
                env.update(isolation.env)

            if isolation.config_content:
                env["OPENCODE_CONFIG_CONTENT"] = isolation.config_content

            if isolation.config_dir:
                env["OPENCODE_CONFIG_DIR"] = str(isolation.config_dir)

            if isolation.cache_dir:
                env["OPENCODE_CACHE_DIR"] = str(isolation.cache_dir)

        # Always set NO_PROXY for local communication
        env.setdefault("NO_PROXY", "localhost,127.0.0.1")

        return env
```

## Tool Bridging

### Native ACP Tool Support

Unlike HTTP API (which requires external MCP server), ACP supports tool
registration during initialization:

```python
# Tools registered in ClientCapabilities
capabilities = ClientCapabilities(
    available_tools=(
        ToolSchema(
            name="search_code",
            description="Search codebase for patterns",
            input_schema={"type": "object", "properties": {...}},
        ),
        ToolSchema(
            name="run_tests",
            description="Run test suite",
            input_schema={"type": "object", "properties": {...}},
        ),
    ),
)

# Agent can invoke these tools via session/update notifications
# WINK handles execution and returns results
```

### Tool Call Flow

```
Agent                         WINK
  │                            │
  │── session/update ─────────▶│ (tool_call, status=pending)
  │                            │ → Create snapshot
  │                            │ → Execute tool handler
  │                            │
  │◀─ tool_result ─────────────│ (via ACP mechanism)
  │                            │
  │── session/update ─────────▶│ (tool_call, status=complete)
  │                            │ → Dispatch ToolInvoked
  │                            │ → Clear snapshot
```

### BridgedTool for ACP

```python
@FrozenDataclass()
class AcpBridgedTool:
    """Tool wrapper for ACP execution."""
    tool: Tool[Any, Any]
    session: SessionProtocol
    prompt: PromptProtocol[Any]

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        tool_id: str,
    ) -> dict[str, Any]:
        """Execute tool with transactional semantics."""
        # Parse arguments
        params = serde.parse(
            self.tool.handler.__annotations__.get("params", dict),
            arguments,
            extra="forbid",
        )

        # Create context
        context = ToolContext(
            prompt=self.prompt,
            rendered_prompt=None,
            adapter=None,
            session=self.session,
        )

        # Execute with transaction
        with tool_transaction(
            self.session,
            self.prompt.resources.context,
            tag=f"tool:{self.tool.name}",
        ):
            result = self.tool.handler(params, context=context)

        # Format for ACP
        return {
            "success": result.success,
            "content": result.message,
            "value": serde.dump(result.value) if result.value else None,
        }
```

## Structured Output

### ACP-Native Approach

ACP may support structured output schemas in the initialize response.
Check agent capabilities:

```python
if init_response.agent_capabilities.supports_structured_output:
    # Use native structured output
    output_schema = schema(rendered.output_type)
    # Pass to session/prompt params
```

### Fallback: Tool-Based

If not supported natively, use same pattern as HTTP adapter:

```python
# Add wink_emit tool to capabilities
capabilities = ClientCapabilities(
    available_tools=(
        *tool_schemas,
        ToolSchema(
            name="wink_emit",
            description="Emit structured output when task is complete",
            input_schema=schema(WinkEmitParams),
        ),
    ),
)
```

## Implementation Effort Analysis

### Components to Build

| Component | Complexity | LOC Estimate | Notes |
|-----------|------------|--------------|-------|
| `ProcessManager` | Low | ~100 | Simple subprocess management |
| `AcpClient` | Medium | ~300 | JSON-RPC protocol handling |
| `NotificationHandler` | Medium | ~200 | Update processing, state sync |
| `OpenCodeAcpAdapter` | Medium | ~400 | Main orchestration |
| Protocol types | Low | ~150 | Dataclasses for messages |
| Tests | Medium | ~400 | Simpler than HTTP |
| **Total** | | ~1550 | |

### Dependencies

- Standard library only (`subprocess`, `json`)
- Optional: `agent-client-protocol` Python SDK for type definitions

### Comparison with HTTP API

| Aspect | HTTP API | ACP |
|--------|----------|-----|
| LOC estimate | ~2400 | ~1550 |
| External dependencies | httpx, sse | None |
| Server management | Complex | None |
| Event handling | SSE parsing + reconnect | stdio JSON-RPC |
| Tool bridging | External MCP server | Native capabilities |
| Risk level | High | Low |

### Risk Factors

1. **Protocol Evolution**: ACP is newer; may have breaking changes
2. **Tool Execution**: Need to verify bidirectional tool call support
3. **Permissions**: May differ from HTTP API model
4. **OpenCode ACP Support**: Verify feature parity with HTTP API

## Recommendation

**ACP is the recommended approach for WINK integration:**

1. **Architectural Alignment**: Process model matches Claude Agent SDK
2. **Simpler Implementation**: ~35% less code than HTTP API
3. **Fewer Dependencies**: No HTTP client or SSE library needed
4. **Native Tool Support**: No external MCP server required
5. **Lower Risk**: Stdio communication is simpler than HTTP+SSE

The primary caveat is verifying OpenCode's ACP implementation supports all
required features (tool execution, structured output, etc.).

## Related Specifications

- `specs/OPENCODE_HTTP_API.md` - Alternative HTTP-based adapter
- `specs/ADAPTERS.md` - Provider adapter protocol
- `specs/CLAUDE_AGENT_SDK.md` - Reference implementation
- `specs/TOOLS.md` - Tool bridging patterns

## External References

- [Agent Client Protocol](https://agentclientprotocol.com/)
- [ACP GitHub Repository](https://github.com/agentclientprotocol/agent-client-protocol)
- [ACP Python SDK](https://github.com/agentclientprotocol/python-sdk)

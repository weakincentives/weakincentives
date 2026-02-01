# OpenCode Adapter Specification

> **Protocol**: Agent Client Protocol (ACP) v1
> **Reference**: `src/weakincentives/adapters/claude_agent_sdk/`

## Purpose

`OpenCodeAdapter` evaluates prompts via the Agent Client Protocol (ACP) by
spawning `opencode acp` as a subprocess and communicating via JSON-RPC 2.0
over stdio. This approach mirrors the Claude Agent SDK adapter architecture.

**Implementation:** `src/weakincentives/adapters/opencode/`

## Background

OpenCode is an open-source coding agent supporting the Agent Client Protocol
(ACP)—an open standard for AI agent integration developed by Zed, JetBrains,
and the broader community. ACP standardizes communication between editors and
coding agents via JSON-RPC over stdio.

The ACP approach was chosen over OpenCode's HTTP API because:

1. **Architectural alignment** with Claude Agent SDK (subprocess model)
2. **Simpler implementation** (~40% less code)
3. **No external dependencies** (stdlib only)
4. **Native tool support** (no MCP server required)

## Feature Comparison Matrix

### Claude Agent SDK vs OpenCode ACP

| Feature | Claude Agent SDK | OpenCode ACP | Notes |
|---------|------------------|--------------|-------|
| **Process Model** | | | |
| Subprocess management | SDK internal | WINK owns | ACP requires explicit spawn |
| Process communication | SDK methods | JSON-RPC stdio | Same conceptual pattern |
| Graceful shutdown | `client.disconnect()` | Process termination | ACP uses SIGTERM |
| **Session Lifecycle** | | | |
| Initialization | `ClaudeSDKClient(options)` | `initialize` request | Capability negotiation |
| Session creation | `client.connect(prompt)` | `session/new` request | Returns session ID |
| Prompt execution | `receive_messages()` | `session/prompt` + notifications | Streaming in both |
| Cancellation | N/A (deadline-based) | `session/cancel` request | ACP has explicit cancel |
| **Event Streaming** | | | |
| Mechanism | In-process hooks | JSON-RPC notifications | Hooks vs `session/update` |
| Tool start | `PreToolUse` hook | `tool_call` (status=pending) | Same semantic |
| Tool complete | `PostToolUse` hook | `tool_call` (status=complete) | Same semantic |
| Message streaming | Hook callbacks | `message` notifications | Same semantic |
| **Tool Bridging** | | | |
| Registration | `create_sdk_mcp_server()` | `initialize` capabilities | ACP is simpler |
| Invocation | MCP protocol | ACP tool calls | Native in ACP |
| Transactional | `tool_transaction()` | `tool_transaction()` | Same pattern |
| **State Synchronization** | | | |
| Snapshot on tool start | `HookContext._tracker` | `NotificationHandler` | Same pattern |
| Restore on failure | `restore_snapshot()` | `restore_snapshot()` | Same function |
| Event dispatch | `session.dispatcher` | `session.dispatcher` | Same interface |
| **Isolation** | | | |
| Ephemeral HOME | `EphemeralHome` class | Environment injection | Same goal |
| Config isolation | `setting_sources=[]` | `OPENCODE_CONFIG_CONTENT` | Different mechanism |
| Workspace | `ClaudeAgentWorkspaceSection` | `OpenCodeWorkspaceSection` | Same pattern |
| **Structured Output** | | | |
| Native support | SDK `output_format` | None | Must use tool |
| Tool-based fallback | N/A | `structured_output` tool | Required for OpenCode |
| Schema validation | `serde.parse()` | `serde.parse()` | Same function |
| **Budget & Deadline** | | | |
| Token tracking | `BudgetTracker` | `BudgetTracker` | Same class |
| Deadline enforcement | Pre-round check | Pre-round check | Same pattern |
| Cancellation on exceed | Implicit (loop exit) | `session/cancel` | ACP is explicit |
| **Permissions** | | | |
| Mode | `bypassPermissions` | Config-based | Different mechanism |
| Runtime handling | N/A | Via notifications | ACP may need handler |

### Implementation Mapping

| Claude Agent SDK Component | OpenCode ACP Equivalent |
|----------------------------|------------------------|
| `ClaudeAgentSDKAdapter` | `OpenCodeAdapter` |
| `ClaudeSDKClient` | `AcpClient` |
| `ClaudeAgentOptions` | `ClientCapabilities` |
| `HookContext` | `NotificationHandler` context |
| `create_pre_tool_use_hook()` | `_handle_tool_pending()` |
| `create_post_tool_use_hook()` | `_handle_tool_complete()` |
| `create_bridged_tools()` | `_build_tool_schemas()` |
| `create_mcp_server()` | N/A (native in ACP) |
| `EphemeralHome` | `_build_env()` |
| `_run_sdk_query()` | `_run_acp_session()` |
| `_extract_result()` | `_extract_result()` |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        OpenCodeAdapter                          │
├─────────────────────────────────────────────────────────────────┤
│  evaluate()                                                     │
│    │                                                            │
│    ├─► render prompt                                            │
│    ├─► create workspace (temp dir + mounts)                     │
│    ├─► spawn opencode acp subprocess                            │
│    ├─► initialize (capability negotiation)                      │
│    ├─► session/new (create session)                             │
│    ├─► session/prompt (send prompt)                             │
│    │     │                                                      │
│    │     └─► process session/update notifications               │
│    │           ├─► tool_call pending  → create snapshot         │
│    │           ├─► tool_call complete → dispatch ToolInvoked    │
│    │           ├─► tool_call failed   → restore snapshot        │
│    │           ├─► message            → log/stream              │
│    │           └─► thought            → log                     │
│    │                                                            │
│    ├─► receive PromptResponse                                   │
│    ├─► extract result + dispatch PromptExecuted                 │
│    └─► terminate subprocess + cleanup workspace                 │
└─────────────────────────────────────────────────────────────────┘
```

## ACP Protocol Details

### JSON-RPC 2.0 Foundation

ACP uses JSON-RPC 2.0 with newline-delimited messages over stdio:

```
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{...}}\n
{"jsonrpc":"2.0","id":1,"result":{...}}\n
{"jsonrpc":"2.0","method":"session/update","params":{...}}\n
```

Two message types:
- **Requests**: Have `id`, expect response
- **Notifications**: No `id`, no response expected

### Protocol Flow

```
WINK                                    OpenCode (subprocess)
 │                                           │
 │── initialize ────────────────────────────▶│
 │   {protocolVersion, clientCapabilities}   │
 │◀─ InitializeResponse ─────────────────────│
 │   {protocolVersion, agentCapabilities}    │
 │                                           │
 │── session/new ───────────────────────────▶│
 │   {cwd}                                   │
 │◀─ NewResponse ────────────────────────────│
 │   {sessionId}                             │
 │                                           │
 │── session/prompt ────────────────────────▶│
 │   {sessionId, prompt}                     │
 │                                           │
 │◀─ session/update (thought) ───────────────│  ←─┐
 │◀─ session/update (tool_call pending) ─────│    │ notifications
 │◀─ session/update (tool_call complete) ────│    │ (no id)
 │◀─ session/update (message) ───────────────│  ←─┘
 │                                           │
 │◀─ PromptResponse ─────────────────────────│  ←── has id
 │   {messages, usage}                       │
 │                                           │
 │── [terminate process] ───────────────────▶│
```

### Message Schemas

#### Initialize Request

```python
{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": 1,
        "clientCapabilities": {
            "textPrompts": True,
            "streaming": True,
            "tools": [
                {
                    "name": "structured_output",
                    "description": "Emit structured output",
                    "inputSchema": {"type": "object", ...}
                }
            ]
        }
    }
}
```

#### Session Update Notification

```python
# Tool call starting
{
    "jsonrpc": "2.0",
    "method": "session/update",
    "params": {
        "sessionId": "session-123",
        "type": "tool_call",
        "content": {
            "toolId": "call-456",
            "toolName": "Read",
            "arguments": {"file_path": "/src/main.py"},
            "status": "pending"
        }
    }
}

# Tool call complete
{
    "jsonrpc": "2.0",
    "method": "session/update",
    "params": {
        "sessionId": "session-123",
        "type": "tool_call",
        "content": {
            "toolId": "call-456",
            "toolName": "Read",
            "status": "complete",
            "result": "file contents..."
        }
    }
}
```

## Components

### 1. ProcessManager

Manages the `opencode acp` subprocess lifecycle.

**Reference:** Similar to how `ClaudeSDKClient` manages the underlying process,
but WINK owns the lifecycle explicitly.

```python
@FrozenDataclass()
class ProcessManager:
    """Manages OpenCode ACP subprocess.

    Lifecycle mirrors what ClaudeSDKClient does internally:
    - Spawn process with configured environment
    - Provide stdio handles for communication
    - Terminate gracefully on completion
    """
    process: subprocess.Popen[bytes]
    stdin: IO[bytes]
    stdout: IO[bytes]
    stderr: IO[bytes]
    _stderr_buffer: list[str]

    @classmethod
    def spawn(
        cls,
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ProcessManager:
        """Spawn opencode acp subprocess.

        Args:
            cwd: Working directory (like ClaudeAgentOptions.cwd)
            env: Environment overrides (like EphemeralHome.get_env())

        Returns:
            ProcessManager ready for communication
        """
        merged_env = {**os.environ, **(env or {})}

        process = subprocess.Popen(
            ["opencode", "acp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=merged_env,
        )

        return cls(
            process=process,
            stdin=cast(IO[bytes], process.stdin),
            stdout=cast(IO[bytes], process.stdout),
            stderr=cast(IO[bytes], process.stderr),
            _stderr_buffer=[],
        )

    def send(self, message: dict[str, Any]) -> None:
        """Send JSON-RPC message."""
        line = json.dumps(message, separators=(",", ":")) + "\n"
        self.stdin.write(line.encode())
        self.stdin.flush()

    def receive(self, timeout: float | None = None) -> dict[str, Any]:
        """Receive JSON-RPC message.

        Handles interleaved notifications while waiting for response.
        Returns first message with matching id, queues notifications.
        """
        # Implementation handles timeout via select/poll
        line = self.stdout.readline()
        if not line:
            raise ProcessTerminatedError()
        return json.loads(line.decode())

    def receive_until_response(
        self,
        request_id: int,
        *,
        timeout: float | None = None,
        on_notification: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Receive messages until response with matching id.

        Yields notifications to callback while waiting.
        This is the key pattern for handling streaming updates.
        """
        deadline = time.monotonic() + timeout if timeout else None

        while True:
            remaining = (deadline - time.monotonic()) if deadline else None
            if remaining is not None and remaining <= 0:
                raise TimeoutError()

            message = self.receive(timeout=remaining)

            # Notification (no id) - forward to handler
            if "id" not in message:
                if on_notification:
                    on_notification(message)
                continue

            # Response - check if it's ours
            if message.get("id") == request_id:
                return message

            # Response for different request - shouldn't happen in our usage
            logger.warning(
                "opencode.acp.unexpected_response",
                context={"expected_id": request_id, "got_id": message.get("id")},
            )

    def terminate(self, timeout: float = 5.0) -> int:
        """Terminate subprocess gracefully.

        Returns:
            Exit code
        """
        self.process.terminate()
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            return self.process.wait()

    def __enter__(self) -> ProcessManager:
        return self

    def __exit__(self, *exc: object) -> None:
        self.terminate()
```

### 2. AcpClient

JSON-RPC client implementing ACP protocol methods.

**Reference:** Equivalent to `ClaudeSDKClient` in the Claude Agent SDK adapter.

```python
@FrozenDataclass()
class AcpClient:
    """ACP protocol client.

    Mirrors ClaudeSDKClient interface:
    - initialize() ↔ ClaudeSDKClient(options)
    - session_new() ↔ client.connect()
    - session_prompt() ↔ client.receive_messages()
    - session_cancel() ↔ N/A (deadline-based in SDK)
    """
    process: ProcessManager
    _next_id: int = 0
    _agent_capabilities: AgentCapabilities | None = None

    def _send_request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float | None = None,
        on_notification: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Send request and wait for response."""
        self._next_id += 1
        request_id = self._next_id

        self.process.send({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        })

        response = self.process.receive_until_response(
            request_id,
            timeout=timeout,
            on_notification=on_notification,
        )

        if "error" in response:
            raise AcpError.from_response(response["error"])

        return response["result"]

    def initialize(
        self,
        capabilities: ClientCapabilities,
        *,
        timeout: float = 10.0,
    ) -> AgentCapabilities:
        """Initialize ACP connection.

        Equivalent to ClaudeSDKClient(options) initialization.
        Negotiates capabilities between WINK and OpenCode.

        Args:
            capabilities: What WINK supports (tools, streaming, etc.)
            timeout: Max time to wait for response

        Returns:
            AgentCapabilities describing what OpenCode supports
        """
        result = self._send_request(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": capabilities.to_dict(),
            },
            timeout=timeout,
        )

        self._agent_capabilities = AgentCapabilities.from_dict(
            result["agentCapabilities"]
        )
        return self._agent_capabilities

    def session_new(
        self,
        *,
        cwd: str | None = None,
        timeout: float = 10.0,
    ) -> str:
        """Create new session.

        Equivalent to client.connect(prompt) session creation.

        Returns:
            Session ID for subsequent requests
        """
        params: dict[str, Any] = {}
        if cwd:
            params["cwd"] = cwd

        result = self._send_request("session/new", params, timeout=timeout)
        return result["sessionId"]

    def session_prompt(
        self,
        session_id: str,
        prompt: str,
        *,
        on_update: Callable[[SessionUpdate], None],
        timeout: float | None = None,
    ) -> PromptResult:
        """Send prompt and stream updates.

        Equivalent to async for message in client.receive_messages().

        The on_update callback receives SessionUpdate notifications
        while waiting for the final PromptResponse. This mirrors
        how SDK hooks receive events during execution.

        Args:
            session_id: From session_new()
            prompt: User prompt text
            on_update: Callback for streaming updates (like SDK hooks)
            timeout: Max time for entire prompt execution

        Returns:
            PromptResult with messages and usage
        """
        def handle_notification(msg: dict[str, Any]) -> None:
            if msg.get("method") == "session/update":
                update = SessionUpdate.from_dict(msg["params"])
                on_update(update)

        result = self._send_request(
            "session/prompt",
            {"sessionId": session_id, "prompt": prompt},
            timeout=timeout,
            on_notification=handle_notification,
        )

        return PromptResult.from_dict(result)

    def session_cancel(self, session_id: str) -> None:
        """Cancel running prompt.

        No equivalent in Claude Agent SDK (uses deadline instead).
        Call this when deadline exceeded or budget exhausted.
        """
        try:
            self._send_request(
                "session/cancel",
                {"sessionId": session_id},
                timeout=5.0,
            )
        except (TimeoutError, AcpError):
            # Cancel may fail if already complete
            pass
```

### 3. NotificationHandler

Processes `session/update` notifications and synchronizes WINK state.

**Reference:** Equivalent to hook functions in `_hooks.py`:
- `create_pre_tool_use_hook()` → `_handle_tool_pending()`
- `create_post_tool_use_hook()` → `_handle_tool_complete()`

```python
class NotificationHandler:
    """Handles session/update notifications.

    Mirrors HookContext + hook functions from Claude Agent SDK adapter.
    Key responsibilities:
    - Snapshot state before tool execution (like PreToolUse hook)
    - Restore state on tool failure (like PostToolUse hook with error)
    - Dispatch ToolInvoked events (like PostToolUse hook success)
    - Track cumulative statistics
    """

    def __init__(
        self,
        *,
        session: SessionProtocol,
        prompt: PromptProtocol[Any],
        adapter_name: AdapterName,
        prompt_name: str,
        deadline: Deadline | None = None,
        budget_tracker: BudgetTracker | None = None,
        heartbeat: Heartbeat | None = None,
        run_context: RunContext | None = None,
    ) -> None:
        # Same fields as HookContext
        self._session = session
        self._prompt = prompt
        self._adapter_name = adapter_name
        self._prompt_name = prompt_name
        self._deadline = deadline
        self._budget_tracker = budget_tracker
        self._heartbeat = heartbeat
        self._run_context = run_context

        # Pending tool snapshots (like PendingToolTracker)
        self._pending_tools: dict[str, CompositeSnapshot] = {}

        # Statistics (like HookStats)
        self._stats = ExecutionStats()

    def handle(self, update: SessionUpdate) -> None:
        """Process session update notification.

        Routes to type-specific handlers like SDK hook dispatch.
        """
        # Beat heartbeat on every update (like hooks do)
        if self._heartbeat:
            self._heartbeat.beat()

        match update.update_type:
            case UpdateType.TOOL_CALL:
                self._handle_tool_call(update.content)
            case UpdateType.MESSAGE:
                self._handle_message(update.content)
            case UpdateType.THOUGHT:
                self._handle_thought(update.content)
            case UpdateType.PERMISSION:
                self._handle_permission(update.content)
            case _:
                logger.debug(
                    "opencode.acp.unhandled_update",
                    context={"type": update.update_type},
                )

    def _handle_tool_call(self, content: dict[str, Any]) -> None:
        """Handle tool call lifecycle.

        Mirrors PreToolUse/PostToolUse hook logic from _hooks.py.
        """
        tool_id = content["toolId"]
        tool_name = content["toolName"]
        status = content.get("status", "pending")

        match status:
            case "pending" | "running":
                # PreToolUse equivalent: create snapshot
                # See _hooks.py:create_pre_tool_use_hook()
                if tool_id not in self._pending_tools:
                    self._pending_tools[tool_id] = create_snapshot(
                        self._session,
                        self._prompt.resources.context,
                        tag=f"tool:{tool_name}",
                    )
                    logger.debug(
                        "opencode.acp.tool_pending",
                        context={"tool_id": tool_id, "tool_name": tool_name},
                    )

            case "complete":
                # PostToolUse equivalent (success): dispatch event
                # See _hooks.py:create_post_tool_use_hook()
                self._pending_tools.pop(tool_id, None)
                self._stats.tool_count += 1

                self._session.dispatcher.dispatch(
                    ToolInvoked(
                        tool_name=tool_name,
                        params=content.get("arguments"),
                        result=ToolResult.ok(
                            content.get("result"),
                            message=str(content.get("result", ""))[:200],
                        ),
                        adapter=self._adapter_name,
                        prompt_name=self._prompt_name,
                        created_at=datetime.now(UTC),
                        run_context=self._run_context,
                    )
                )

                logger.debug(
                    "opencode.acp.tool_complete",
                    context={"tool_id": tool_id, "tool_name": tool_name},
                )

            case "failed":
                # PostToolUse equivalent (failure): restore snapshot
                # See _hooks.py:create_post_tool_use_hook() error path
                snapshot = self._pending_tools.pop(tool_id, None)
                if snapshot:
                    restore_snapshot(
                        self._session,
                        self._prompt.resources.context,
                        snapshot,
                    )
                    logger.debug(
                        "opencode.acp.tool_failed_restored",
                        context={"tool_id": tool_id, "tool_name": tool_name},
                    )

                self._session.dispatcher.dispatch(
                    ToolInvoked(
                        tool_name=tool_name,
                        params=content.get("arguments"),
                        result=ToolResult.error(
                            str(content.get("error", "Tool execution failed"))
                        ),
                        adapter=self._adapter_name,
                        prompt_name=self._prompt_name,
                        created_at=datetime.now(UTC),
                        run_context=self._run_context,
                    )
                )

    def _handle_message(self, content: dict[str, Any]) -> None:
        """Handle message update (streaming output)."""
        logger.debug(
            "opencode.acp.message",
            context={"preview": str(content.get("text", ""))[:200]},
        )

    def _handle_thought(self, content: dict[str, Any]) -> None:
        """Handle thought update (agent reasoning)."""
        logger.debug(
            "opencode.acp.thought",
            context={"preview": str(content.get("text", ""))[:200]},
        )

    def _handle_permission(self, content: dict[str, Any]) -> None:
        """Handle permission request.

        OpenCode may request permissions during execution.
        Log warning - permissions should be pre-configured.
        """
        logger.warning(
            "opencode.acp.permission_request",
            context={"permission": content},
        )

    @property
    def stats(self) -> ExecutionStats:
        """Execution statistics (like HookContext.stats)."""
        return self._stats


@FrozenDataclass()
class ExecutionStats:
    """Execution statistics.

    Mirrors HookStats from _hooks.py.
    """
    tool_count: int = 0
    turn_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
```

### 4. Tool Schema Builder

Converts WINK tools to ACP capability format.

**Reference:** Similar to `create_bridged_tools()` but simpler—ACP doesn't
need MCP wrapper, just schema in capabilities.

```python
def build_tool_schemas(
    tools: tuple[Tool[Any, Any], ...],
) -> list[dict[str, Any]]:
    """Convert WINK tools to ACP tool schemas.

    Unlike Claude Agent SDK (which needs create_mcp_server()),
    ACP registers tools directly in initialize capabilities.

    Reference: _bridge.py:create_bridged_tools() for schema generation
    """
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": schema(
                # Get params type from handler signature
                tool.handler.__annotations__.get(
                    list(tool.handler.__annotations__.keys())[0],
                    dict,
                )
            ),
        }
        for tool in tools
    ]


def create_structured_output_tool(output_type: type[T]) -> Tool[StructuredOutputParams, None]:
    """Create structured output emission tool.

    Claude Agent SDK has native output_format support.
    OpenCode requires tool-based structured output.

    This tool:
    1. Validates output against schema
    2. Stores in session for extraction
    3. Signals task completion
    """

    @FrozenDataclass()
    class StructuredOutputParams:
        output: dict[str, Any]

    def handler(
        params: StructuredOutputParams,
        *,
        context: ToolContext,
    ) -> ToolResult[None]:
        # Validate against output schema (like SDK's output_format validation)
        try:
            parsed = serde.parse(output_type, params.output, extra="ignore")
        except Exception as e:
            return ToolResult.error(f"Invalid output format: {e}")

        # Store for extraction (dispatch event for session slice)
        context.session.dispatcher.dispatch(
            StructuredOutputEmitted(value=parsed)
        )

        return ToolResult.ok(None, message="Output recorded. Task complete.")

    return Tool(
        name="structured_output",
        description=(
            "Emit structured output when task is complete. "
            "Call this exactly once with your final result."
        ),
        handler=handler,
    )
```

## Configuration

### OpenCodeClientConfig

```python
@FrozenDataclass()
class OpenCodeClientConfig:
    """Configuration for OpenCode ACP adapter.

    Mirrors ClaudeAgentSDKClientConfig structure.
    """
    # Working directory (like ClaudeAgentSDKClientConfig.cwd)
    cwd: Path | None = None

    # Timeouts
    initialize_timeout: float = 10.0
    prompt_timeout: float | None = None  # None = use deadline

    # Isolation (generates environment like EphemeralHome)
    isolation: OpenCodeIsolationConfig | None = None

    # Debugging (like suppress_stderr in SDK config)
    suppress_stderr: bool = True
    log_notifications: bool = False

    # Task completion (like task_completion_checker in SDK)
    task_completion_checker: TaskCompletionChecker | None = None
```

### OpenCodeModelConfig

```python
@FrozenDataclass()
class OpenCodeModelConfig:
    """Model configuration for OpenCode.

    Mirrors ClaudeAgentSDKModelConfig.
    """
    model: str = "anthropic/claude-sonnet-4-20250514"
    provider: str | None = None  # anthropic, openai, bedrock
    temperature: float | None = None
    max_tokens: int | None = None
```

### OpenCodeIsolationConfig

```python
@FrozenDataclass()
class OpenCodeIsolationConfig:
    """Isolation configuration.

    Equivalent to IsolationConfig + EphemeralHome from SDK adapter.
    """
    # Config injection (like setting_sources=[] in SDK)
    config_content: str | None = None  # Inline JSON via OPENCODE_CONFIG_CONTENT
    config_dir: Path | None = None  # Override via OPENCODE_CONFIG_DIR

    # Environment (like IsolationConfig.env)
    env: Mapping[str, str] | None = None
    include_host_env: bool = False

    # Cache isolation (prevents plugin conflicts)
    cache_dir: Path | None = None  # Override via OPENCODE_CACHE_DIR

    def build_env(self, workspace_path: Path | None = None) -> dict[str, str]:
        """Build environment dict.

        Equivalent to EphemeralHome.get_env().
        """
        env: dict[str, str] = {}

        if self.include_host_env:
            # Copy non-sensitive vars (like IsolationConfig.include_host_env)
            for key, value in os.environ.items():
                if not any(s in key.upper() for s in ("KEY", "SECRET", "TOKEN", "PASSWORD")):
                    env[key] = value

        if self.env:
            env.update(self.env)

        # Config isolation
        if self.config_content:
            env["OPENCODE_CONFIG_CONTENT"] = self.config_content

        if self.config_dir:
            env["OPENCODE_CONFIG_DIR"] = str(self.config_dir)

        if self.cache_dir:
            env["OPENCODE_CACHE_DIR"] = str(self.cache_dir)

        # Always bypass proxy for local communication
        env.setdefault("NO_PROXY", "localhost,127.0.0.1")

        return env
```

## Adapter Implementation

### OpenCodeAdapter

```python
OPENCODE_ADAPTER_NAME: AdapterName = "opencode"


class OpenCodeAdapter[OutputT](ProviderAdapter[OutputT]):
    """Adapter using OpenCode ACP protocol.

    Implementation mirrors ClaudeAgentSDKAdapter structure:
    - evaluate() → _evaluate_async() pattern
    - Workspace creation if no filesystem bound
    - Resource context management
    - Subprocess lifecycle with cleanup in finally
    """

    def __init__(
        self,
        *,
        model_config: OpenCodeModelConfig | None = None,
        client_config: OpenCodeClientConfig | None = None,
    ) -> None:
        self._model_config = model_config or OpenCodeModelConfig()
        self._client_config = client_config or OpenCodeClientConfig()
        self._stderr_buffer: list[str] = []

        logger.debug(
            "opencode.adapter.init",
            context={
                "model": self._model_config.model,
                "cwd": str(self._client_config.cwd) if self._client_config.cwd else None,
            },
        )

    @override
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

        Mirrors ClaudeAgentSDKAdapter.evaluate() structure.
        """
        if budget and not budget_tracker:
            budget_tracker = BudgetTracker(budget)

        effective_deadline = deadline or (budget.deadline if budget else None)

        prompt_name = prompt.name or f"{prompt.ns}:{prompt.key}"

        # Deadline check (like SDK adapter)
        if effective_deadline and effective_deadline.remaining().total_seconds() <= 0:
            raise PromptEvaluationError(
                message="Deadline expired before execution",
                prompt_name=prompt_name,
                phase="request",
            )

        return run_async(
            self._evaluate_async(
                prompt,
                session=session,
                deadline=effective_deadline,
                budget_tracker=budget_tracker,
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
        """Async implementation.

        Mirrors ClaudeAgentSDKAdapter._evaluate_async() structure:
        1. Render prompt
        2. Dispatch PromptRendered
        3. Create workspace if needed
        4. Enter resource context
        5. Run with subprocess
        6. Cleanup
        """
        self._stderr_buffer.clear()

        # 1. Render prompt
        rendered = prompt.render(session=session)
        prompt_text = rendered.text
        prompt_name = prompt.name or f"{prompt.ns}:{prompt.key}"

        logger.debug(
            "opencode.evaluate.rendered",
            context={
                "prompt_name": prompt_name,
                "tool_count": len(rendered.tools),
                "tool_names": [t.name for t in rendered.tools],
            },
        )

        # 2. Dispatch PromptRendered (like SDK adapter)
        session.dispatcher.dispatch(
            PromptRendered(
                prompt_ns=prompt.ns,
                prompt_key=prompt.key,
                prompt_name=prompt.name,
                adapter=OPENCODE_ADAPTER_NAME,
                session_id=getattr(session, "session_id", None),
                render_inputs=(),
                rendered_prompt=prompt_text,
                created_at=datetime.now(UTC),
                descriptor=None,
                run_context=run_context,
            )
        )

        # 3. Create workspace if no filesystem bound (like SDK adapter)
        temp_workspace_dir: str | None = None
        effective_cwd = self._client_config.cwd

        if prompt.filesystem() is None:
            if effective_cwd is None:
                temp_workspace_dir = tempfile.mkdtemp(prefix="wink-opencode-")
                effective_cwd = Path(temp_workspace_dir)

            filesystem = HostFilesystem(_root=str(effective_cwd))
            prompt = prompt.bind(resources={Filesystem: filesystem})

        try:
            # 4. Enter resource context (like SDK adapter)
            with prompt.resources:
                return await self._run_with_context(
                    prompt=prompt,
                    prompt_name=prompt_name,
                    prompt_text=prompt_text,
                    rendered=rendered,
                    session=session,
                    effective_cwd=effective_cwd,
                    deadline=deadline,
                    budget_tracker=budget_tracker,
                    heartbeat=heartbeat,
                    run_context=run_context,
                )
        finally:
            # 6. Cleanup (like SDK adapter)
            if temp_workspace_dir:
                shutil.rmtree(temp_workspace_dir, ignore_errors=True)

    async def _run_with_context(
        self,
        *,
        prompt: Prompt[OutputT],
        prompt_name: str,
        prompt_text: str,
        rendered: RenderedPrompt[OutputT],
        session: SessionProtocol,
        effective_cwd: Path | None,
        deadline: Deadline | None,
        budget_tracker: BudgetTracker | None,
        heartbeat: Heartbeat | None,
        run_context: RunContext | None,
    ) -> PromptResponse[OutputT]:
        """Run within resource context.

        Mirrors _run_with_prompt_context() in SDK adapter.
        """
        start_time = datetime.now(UTC)

        # Build environment (like EphemeralHome)
        isolation = self._client_config.isolation or OpenCodeIsolationConfig()
        env = isolation.build_env(workspace_path=effective_cwd)

        # Add model config to environment
        if self._model_config.model:
            env["OPENCODE_MODEL"] = self._model_config.model
        if self._model_config.provider:
            env["OPENCODE_PROVIDER"] = self._model_config.provider

        logger.debug(
            "opencode.run_context.env",
            context={
                "cwd": str(effective_cwd) if effective_cwd else None,
                "env_keys": [k for k in env if "KEY" not in k.upper()],
            },
        )

        # Build tool schemas (simpler than create_mcp_server)
        tools = list(rendered.tools)

        # Add structured_output tool for structured output if output type specified
        if rendered.output_type and rendered.output_type is not type(None):
            output_tool = create_structured_output_tool(rendered.output_type)
            tools.append(output_tool)

        tool_schemas = build_tool_schemas(tuple(tools))

        # Spawn subprocess
        try:
            with ProcessManager.spawn(cwd=effective_cwd, env=env) as process:
                return await self._run_acp_session(
                    process=process,
                    prompt=prompt,
                    prompt_name=prompt_name,
                    prompt_text=prompt_text,
                    rendered=rendered,
                    session=session,
                    tool_schemas=tool_schemas,
                    effective_cwd=effective_cwd,
                    deadline=deadline,
                    budget_tracker=budget_tracker,
                    heartbeat=heartbeat,
                    run_context=run_context,
                    start_time=start_time,
                )
        except Exception as error:
            # Normalize errors (like normalize_sdk_error)
            captured_stderr = "\n".join(self._stderr_buffer) if self._stderr_buffer else None
            logger.debug(
                "opencode.run_context.error",
                context={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "stderr": captured_stderr,
                },
            )
            raise PromptEvaluationError(
                message=str(error),
                prompt_name=prompt_name,
                phase="request",
            ) from error

    async def _run_acp_session(
        self,
        *,
        process: ProcessManager,
        prompt: Prompt[OutputT],
        prompt_name: str,
        prompt_text: str,
        rendered: RenderedPrompt[OutputT],
        session: SessionProtocol,
        tool_schemas: list[dict[str, Any]],
        effective_cwd: Path | None,
        deadline: Deadline | None,
        budget_tracker: BudgetTracker | None,
        heartbeat: Heartbeat | None,
        run_context: RunContext | None,
        start_time: datetime,
    ) -> PromptResponse[OutputT]:
        """Run ACP session.

        Mirrors _run_sdk_query() in SDK adapter:
        1. Initialize with capabilities
        2. Create session
        3. Send prompt and process updates
        4. Extract result
        """
        client = AcpClient(process=process)

        # 1. Initialize (like ClaudeSDKClient construction)
        capabilities = ClientCapabilities(
            supports_text_prompts=True,
            supports_streaming=True,
            tools=tool_schemas,
        )

        logger.debug(
            "opencode.acp.initializing",
            context={"tool_count": len(tool_schemas)},
        )

        agent_caps = client.initialize(
            capabilities,
            timeout=self._client_config.initialize_timeout,
        )

        logger.debug(
            "opencode.acp.initialized",
            context={
                "supports_streaming": agent_caps.supports_streaming,
                "supports_cancel": agent_caps.supports_cancel,
            },
        )

        # 2. Create session (like client.connect)
        session_id = client.session_new(
            cwd=str(effective_cwd) if effective_cwd else None,
            timeout=self._client_config.initialize_timeout,
        )

        logger.info(
            "opencode.acp.session_created",
            context={"session_id": session_id, "prompt_name": prompt_name},
        )

        # 3. Create notification handler (like HookContext)
        handler = NotificationHandler(
            session=session,
            prompt=cast("PromptProtocol[object]", prompt),
            adapter_name=OPENCODE_ADAPTER_NAME,
            prompt_name=prompt_name,
            deadline=deadline,
            budget_tracker=budget_tracker,
            heartbeat=heartbeat,
            run_context=run_context,
        )

        # 4. Send prompt and process updates
        try:
            # Check constraints before prompt (like SDK's pre-round checks)
            self._check_constraints(deadline, budget_tracker, prompt_name)

            result = client.session_prompt(
                session_id,
                prompt_text,
                on_update=handler.handle,
                timeout=self._compute_timeout(deadline),
            )

        except (TimeoutError, DeadlineExceededError):
            # Cancel on timeout (SDK uses deadline, ACP has explicit cancel)
            if agent_caps.supports_cancel:
                client.session_cancel(session_id)
            raise DeadlineExceededError() from None

        except Exception:
            # Cancel on error
            if agent_caps.supports_cancel:
                with contextlib.suppress(Exception):
                    client.session_cancel(session_id)
            raise

        # 5. Extract result (like _extract_result)
        end_time = datetime.now(UTC)
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        result_text, output = self._extract_result(result, rendered, session)

        # Record usage (like SDK adapter)
        usage = TokenUsage(
            input_tokens=result.usage.get("input_tokens") if result.usage else None,
            output_tokens=result.usage.get("output_tokens") if result.usage else None,
        )

        if budget_tracker and (usage.input_tokens or usage.output_tokens):
            budget_tracker.record_cumulative(prompt_name, usage)

        # Build response
        response = PromptResponse(
            prompt_name=prompt_name,
            text=result_text,
            output=output,
        )

        # Dispatch PromptExecuted (like SDK adapter)
        session.dispatcher.dispatch(
            PromptExecuted(
                prompt_name=prompt_name,
                adapter=OPENCODE_ADAPTER_NAME,
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
                "prompt_name": prompt_name,
                "duration_ms": duration_ms,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "tool_count": handler.stats.tool_count,
            },
        )

        return response

    def _check_constraints(
        self,
        deadline: Deadline | None,
        budget_tracker: BudgetTracker | None,
        prompt_name: str,
    ) -> None:
        """Check deadline and budget constraints.

        Like SDK adapter's pre-round checks in _run_sdk_query.
        """
        if deadline and deadline.remaining().total_seconds() <= 0:
            raise DeadlineExceededError()

        if budget_tracker:
            budget_tracker.check()  # Raises if exhausted

    def _compute_timeout(self, deadline: Deadline | None) -> float | None:
        """Compute timeout from deadline."""
        if deadline:
            remaining = deadline.remaining().total_seconds()
            return max(0.1, remaining)  # Minimum 100ms
        return self._client_config.prompt_timeout

    def _extract_result(
        self,
        result: PromptResult,
        rendered: RenderedPrompt[OutputT],
        session: SessionProtocol,
    ) -> tuple[str | None, OutputT | None]:
        """Extract text and structured output.

        Like SDK adapter's _extract_result + _try_parse_structured_output.
        """
        # Get text from last message
        result_text: str | None = None
        if result.messages:
            last_msg = result.messages[-1]
            if isinstance(last_msg, dict):
                result_text = last_msg.get("content") or last_msg.get("text")

        # Get structured output from session (set by structured_output tool)
        output: OutputT | None = None
        try:
            output_slice = session[StructuredOutputState]
            if output_slice.latest():
                output = cast(OutputT, output_slice.latest().value)
        except (KeyError, AttributeError):
            pass

        return result_text, output
```

## Workspace Section

### OpenCodeWorkspaceSection

```python
@FrozenDataclass()
class OpenCodeWorkspaceSection(Section[None]):
    """Prompt section providing isolated workspace.

    Mirrors ClaudeAgentWorkspaceSection from SDK adapter.
    """
    temp_dir: Path
    mounts: tuple[HostMountPreview, ...]
    _filesystem: HostFilesystem

    @classmethod
    def create(
        cls,
        *,
        session: SessionProtocol,
        mounts: tuple[HostMount, ...] = (),
        allowed_host_roots: tuple[str, ...] = (),
    ) -> OpenCodeWorkspaceSection:
        """Create workspace with mounted files.

        Like ClaudeAgentWorkspaceSection.__init__.
        """
        temp_dir = Path(tempfile.mkdtemp(prefix="wink-opencode-"))

        previews: list[HostMountPreview] = []
        for mount in mounts:
            preview = _copy_mount(mount, temp_dir, allowed_host_roots)
            previews.append(preview)

        filesystem = HostFilesystem(_root=str(temp_dir))

        return cls(
            temp_dir=temp_dir,
            mounts=tuple(previews),
            _filesystem=filesystem,
        )

    def resources(self) -> ResourceRegistry:
        """Contribute filesystem resource.

        Like ClaudeAgentWorkspaceSection.resources().
        """
        return ResourceRegistry.build({Filesystem: self._filesystem})

    def cleanup(self) -> None:
        """Remove temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # Section protocol
    @property
    def key(self) -> str:
        return "opencode.workspace"

    @property
    def title(self) -> str:
        return "Workspace"

    def render(self, context: RenderContext) -> str:
        mount_list = "\n".join(f"- {m.mount_path}" for m in self.mounts)
        return f"Working directory: {self.temp_dir}\n\nMounted:\n{mount_list}"
```

## Structured Output

### Session State Slice

```python
@FrozenDataclass()
class StructuredOutputState:
    """Session slice for structured output.

    Receives StructuredOutputEmitted events from structured_output tool.
    """
    value: Any | None = None

    @reducer(on=StructuredOutputEmitted)
    def on_emit(self, event: StructuredOutputEmitted) -> SliceOp[StructuredOutputState]:
        return Replace(StructuredOutputState(value=event.value))


@FrozenDataclass()
class StructuredOutputEmitted:
    """Event dispatched when structured_output tool called."""
    value: Any
```

### Prompt Template Addition

When using structured output, add instructions:

```python
STRUCTURED_OUTPUT_INSTRUCTIONS = """
## Output Format

When you have completed the task, call the `structured_output` tool with your result:

```json
structured_output({{"output": <your result matching the schema>}})
```

The output must conform to this schema:
{schema}

IMPORTANT: Call structured_output exactly once when the task is complete.
Do not provide additional responses after calling structured_output.
"""
```

## Error Handling

### Exception Hierarchy

```python
class OpenCodeError(PromptEvaluationError):
    """Base for OpenCode-specific errors."""
    pass


class ProcessTerminatedError(OpenCodeError):
    """Subprocess terminated unexpectedly."""
    exit_code: int | None
    stderr: str | None


class AcpError(OpenCodeError):
    """ACP protocol error."""
    code: int
    message: str
    data: Any | None

    @classmethod
    def from_response(cls, error: dict[str, Any]) -> AcpError:
        return cls(
            code=error.get("code", -1),
            message=error.get("message", "Unknown error"),
            data=error.get("data"),
        )


class InitializeError(AcpError):
    """Failed to initialize ACP connection."""
    pass
```

## Usage Examples

### Basic Evaluation

```python
from weakincentives.adapters.opencode import (
    OpenCodeAdapter,
    OpenCodeClientConfig,
    OpenCodeModelConfig,
)

adapter = OpenCodeAdapter(
    model_config=OpenCodeModelConfig(
        model="anthropic/claude-sonnet-4-20250514",
    ),
    client_config=OpenCodeClientConfig(
        cwd=Path("/path/to/project"),
    ),
)

response = adapter.evaluate(prompt, session=session)
```

### With Isolation

```python
from weakincentives.adapters.opencode import (
    OpenCodeAdapter,
    OpenCodeClientConfig,
    OpenCodeIsolationConfig,
)

adapter = OpenCodeAdapter(
    client_config=OpenCodeClientConfig(
        isolation=OpenCodeIsolationConfig(
            config_content=json.dumps({
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
            }),
            cache_dir=Path("/tmp/opencode-cache"),
        ),
    ),
)
```

### With Workspace Section

```python
from weakincentives.adapters.opencode import (
    OpenCodeAdapter,
    OpenCodeWorkspaceSection,
)
from weakincentives.adapters.claude_agent_sdk.workspace import HostMount

workspace = OpenCodeWorkspaceSection.create(
    session=session,
    mounts=(
        HostMount(host_path="/path/to/repo", mount_path="repo"),
    ),
    allowed_host_roots=("/path/to",),
)

template = PromptTemplate(
    sections=(
        workspace,
        MarkdownSection(
            key="task",
            template="Review and fix the code in the repo directory.",
        ),
    ),
)

try:
    response = adapter.evaluate(Prompt(template), session=session)
finally:
    workspace.cleanup()
```

### With Structured Output

```python
@FrozenDataclass()
class CodeReviewResult:
    issues_found: int
    files_modified: list[str]
    summary: str

template = PromptTemplate[CodeReviewResult](
    ns="review",
    key="code",
    sections=(
        MarkdownSection(
            key="task",
            template="Review the code and fix any issues.",
        ),
    ),
)

# structured_output tool added automatically when output_type specified
response = adapter.evaluate(Prompt(template), session=session)
print(response.output)  # CodeReviewResult instance
```

## Implementation Checklist

### Phase 1: Core Components

- [ ] `ProcessManager` - subprocess spawn/terminate
- [ ] `AcpClient` - JSON-RPC protocol implementation
- [ ] `NotificationHandler` - update processing and state sync
- [ ] `OpenCodeAdapter` - main adapter class
- [ ] Protocol dataclasses (`ClientCapabilities`, `SessionUpdate`, etc.)

### Phase 2: Integration

- [ ] `build_tool_schemas()` - tool schema generation
- [ ] `create_structured_output_tool()` - structured output tool
- [ ] `StructuredOutputState` - session slice
- [ ] `OpenCodeIsolationConfig.build_env()` - environment building

### Phase 3: Workspace

- [ ] `OpenCodeWorkspaceSection` - workspace section
- [ ] Reuse `HostMount` from Claude Agent SDK adapter
- [ ] Mount copying with glob filtering

### Phase 4: Testing

- [ ] Unit tests for `ProcessManager`
- [ ] Unit tests for `AcpClient` with mock subprocess
- [ ] Unit tests for `NotificationHandler`
- [ ] Integration tests with real OpenCode (optional)

## Related Specifications

- `specs/ADAPTERS.md` - Provider adapter protocol
- `specs/CLAUDE_AGENT_SDK.md` - Reference implementation
- `specs/TOOLS.md` - Tool registration and policies
- `specs/SESSIONS.md` - Session lifecycle and events
- `specs/WORKSPACE.md` - Workspace tools

## External References

- [Agent Client Protocol](https://agentclientprotocol.com/)
- [ACP GitHub Repository](https://github.com/agentclientprotocol/agent-client-protocol)
- [OpenCode Documentation](https://opencode.ai/docs/)
- [OpenCode GitHub](https://github.com/anomalyco/opencode)

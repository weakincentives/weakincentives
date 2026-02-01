# ACP Adapter Specification

> **Protocol**: Agent Client Protocol (ACP) v1
> **Reference**: `src/weakincentives/adapters/claude_agent_sdk/`

## Purpose

`AcpAdapter` evaluates prompts via the Agent Client Protocol (ACP)—an open
standard for AI agent integration. WINK spawns any ACP-compliant agent as a
subprocess and communicates via JSON-RPC 2.0 over stdio.

**Implementation:** `src/weakincentives/adapters/acp/`

## What is ACP?

The Agent Client Protocol (ACP) is an open standard developed by Zed, JetBrains,
and the broader community that standardizes communication between clients and
AI coding agents. ACP is to coding agents what LSP is to language servers.

### Key Characteristics

- **JSON-RPC 2.0** over stdio (newline-delimited)
- **Subprocess model**: Client spawns agent, communicates via pipes
- **Capability negotiation**: Agent and client declare supported features
- **Streaming updates**: Real-time notifications during execution
- **Tool support**: Client can provide tools for agent to invoke

### Supported Agents

Any ACP-compliant agent works with this adapter:

| Agent | Command | Status |
|-------|---------|--------|
| **OpenCode** | `opencode acp` | Primary implementation |
| Claude Code | `claude acp` | Planned |
| Codex CLI | `codex acp` | Potential |
| Gemini CLI | `gemini acp` | Potential |
| goose | `goose acp` | Potential |

OpenCode is the initial reference implementation. Other agents can be added
by specifying the appropriate command in `AcpAgentConfig`.

## Tool Exposure via ACP

**This is a critical architectural difference from Claude Agent SDK.**

### How Tools Work in ACP

Tools are exposed to ACP agents through **capability negotiation** during
the `initialize` handshake. WINK declares available tools in `clientCapabilities`,
and the agent can invoke them during execution.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TOOL EXPOSURE FLOW                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. WINK renders prompt → extracts tools from sections                   │
│                                                                          │
│  2. WINK builds tool schemas for ACP:                                    │
│     tools = [                                                            │
│       {"name": "search_code", "description": "...", "inputSchema": {...}}│
│       {"name": "run_tests", "description": "...", "inputSchema": {...}} │
│       {"name": "structured_output", "description": "...", ...}           │
│     ]                                                                    │
│                                                                          │
│  3. WINK sends initialize request with tools in capabilities:            │
│     {                                                                    │
│       "method": "initialize",                                            │
│       "params": {                                                        │
│         "clientCapabilities": {                                          │
│           "tools": [...]  ◄── WINK tools declared here                   │
│         }                                                                │
│       }                                                                  │
│     }                                                                    │
│                                                                          │
│  4. Agent acknowledges and can now invoke any declared tool              │
│                                                                          │
│  5. During session/prompt, agent invokes tools via session/update:       │
│     {                                                                    │
│       "method": "session/update",                                        │
│       "params": {                                                        │
│         "type": "tool_call",                                             │
│         "content": {                                                     │
│           "toolName": "search_code",  ◄── Agent calls WINK tool          │
│           "arguments": {"query": "..."},                                 │
│           "status": "pending"                                            │
│         }                                                                │
│       }                                                                  │
│     }                                                                    │
│                                                                          │
│  6. WINK executes tool with transactional semantics                      │
│                                                                          │
│  7. Result returned to agent, continues execution                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Comparison: ACP vs Claude Agent SDK Tool Bridging

| Aspect | Claude Agent SDK | ACP |
|--------|------------------|-----|
| **Declaration** | MCP server via `create_sdk_mcp_server()` | `initialize` capabilities |
| **Transport** | MCP protocol (separate channel) | Native ACP messages |
| **Complexity** | Requires MCP server setup | Direct in protocol |
| **Tool Schemas** | Same JSON Schema format | Same JSON Schema format |
| **Transactional** | `tool_transaction()` wrapper | `tool_transaction()` wrapper |

**Key insight**: ACP is simpler because tools are first-class protocol citizens,
not bridged through a separate MCP server.

### Tool Schema Format

```python
# WINK tool definition
@FrozenDataclass()
class SearchParams:
    query: str
    max_results: int = 10

search_tool = Tool(
    name="search_code",
    description="Search codebase for patterns",
    handler=search_handler,
)

# Converted to ACP schema
{
    "name": "search_code",
    "description": "Search codebase for patterns",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "default": 10}
        },
        "required": ["query"]
    }
}
```

### Tool Invocation Lifecycle

```
Agent                           WINK (NotificationHandler)
  │                                   │
  │── session/update ────────────────▶│  tool_call, status=pending
  │   (tool invocation request)       │
  │                                   │  1. Create snapshot
  │                                   │  2. Execute handler
  │                                   │  3. On success: dispatch ToolInvoked
  │                                   │  4. On failure: restore snapshot
  │                                   │
  │◀─ tool result ────────────────────│  (mechanism varies by agent)
  │                                   │
  │── session/update ────────────────▶│  tool_call, status=complete
  │   (tool completion)               │
```

## Feature Comparison Matrix

### Claude Agent SDK vs ACP Adapter

| Feature | Claude Agent SDK | ACP Adapter | Notes |
|---------|------------------|-------------|-------|
| **Process Model** | | | |
| Subprocess management | SDK internal | WINK owns | ACP requires explicit spawn |
| Process communication | SDK methods | JSON-RPC stdio | Same conceptual pattern |
| Graceful shutdown | `client.disconnect()` | Process termination | ACP uses SIGTERM |
| **Session Lifecycle** | | | |
| Initialization | `ClaudeSDKClient(options)` | `initialize` request | Capability negotiation |
| Session creation | `client.connect(prompt)` | `session/new` request | Returns session ID |
| Prompt execution | `receive_messages()` | `session/prompt` + notifications | Streaming in both |
| Cancellation | N/A (deadline-based) | `session/cancel` request | ACP has explicit cancel |
| **Tool Bridging** | | | |
| Registration | `create_sdk_mcp_server()` | `initialize` capabilities | **ACP is simpler** |
| Invocation | MCP protocol | Native ACP messages | No separate channel |
| Transactional | `tool_transaction()` | `tool_transaction()` | Same pattern |
| **Event Streaming** | | | |
| Mechanism | In-process hooks | JSON-RPC notifications | Hooks vs `session/update` |
| Tool start | `PreToolUse` hook | `tool_call` (status=pending) | Same semantic |
| Tool complete | `PostToolUse` hook | `tool_call` (status=complete) | Same semantic |
| Message streaming | Hook callbacks | `message` notifications | Same semantic |
| **State Synchronization** | | | |
| Snapshot on tool start | `HookContext._tracker` | `NotificationHandler` | Same pattern |
| Restore on failure | `restore_snapshot()` | `restore_snapshot()` | Same function |
| Event dispatch | `session.dispatcher` | `session.dispatcher` | Same interface |
| **Isolation** | | | |
| Ephemeral HOME | `EphemeralHome` class | Environment injection | Same goal |
| Config isolation | `setting_sources=[]` | Agent-specific env vars | Different mechanism |
| Workspace | `ClaudeAgentWorkspaceSection` | `AcpWorkspaceSection` | Same pattern |
| **Structured Output** | | | |
| Native support | SDK `output_format` | None | Must use tool |
| Tool-based fallback | N/A | `structured_output` tool | Required for ACP |
| Schema validation | `serde.parse()` | `serde.parse()` | Same function |
| **Budget & Deadline** | | | |
| Token tracking | `BudgetTracker` | `BudgetTracker` | Same class |
| Deadline enforcement | Pre-round check | Pre-round check | Same pattern |
| Cancellation on exceed | Implicit (loop exit) | `session/cancel` | ACP is explicit |

### Implementation Mapping

| Claude Agent SDK Component | ACP Adapter Equivalent |
|----------------------------|------------------------|
| `ClaudeAgentSDKAdapter` | `AcpAdapter` |
| `ClaudeSDKClient` | `AcpClient` |
| `ClaudeAgentOptions` | `ClientCapabilities` |
| `HookContext` | `NotificationHandler` |
| `create_pre_tool_use_hook()` | `_handle_tool_pending()` |
| `create_post_tool_use_hook()` | `_handle_tool_complete()` |
| `create_bridged_tools()` | `build_tool_schemas()` |
| `create_mcp_server()` | N/A (native in ACP) |
| `EphemeralHome` | `AcpIsolationConfig.build_env()` |
| `_run_sdk_query()` | `_run_acp_session()` |
| `_extract_result()` | `_extract_result()` |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          AcpAdapter                              │
├─────────────────────────────────────────────────────────────────┤
│  evaluate()                                                     │
│    │                                                            │
│    ├─► render prompt → extract tools                            │
│    ├─► build_tool_schemas() → ACP tool format                   │
│    ├─► create workspace (temp dir + mounts)                     │
│    ├─► spawn agent subprocess (e.g., opencode acp)              │
│    ├─► initialize (send tools in clientCapabilities)            │
│    ├─► session/new (create session)                             │
│    ├─► session/prompt (send prompt)                             │
│    │     │                                                      │
│    │     └─► process session/update notifications               │
│    │           ├─► tool_call pending  → snapshot + execute      │
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
WINK                                    ACP Agent (subprocess)
 │                                           │
 │── initialize ────────────────────────────▶│
 │   {protocolVersion, clientCapabilities    │
 │    with tools array}                      │
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

### Initialize with Tools

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
                    "name": "search_code",
                    "description": "Search codebase for patterns",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "structured_output",
                    "description": "Emit structured output when task is complete",
                    "inputSchema": {"type": "object", ...}
                }
            ]
        }
    }
}
```

### Tool Call Notification

```python
# Agent requests tool execution
{
    "jsonrpc": "2.0",
    "method": "session/update",
    "params": {
        "sessionId": "session-123",
        "type": "tool_call",
        "content": {
            "toolId": "call-456",
            "toolName": "search_code",
            "arguments": {"query": "def authenticate"},
            "status": "pending"
        }
    }
}

# After WINK executes the tool
{
    "jsonrpc": "2.0",
    "method": "session/update",
    "params": {
        "sessionId": "session-123",
        "type": "tool_call",
        "content": {
            "toolId": "call-456",
            "toolName": "search_code",
            "status": "complete",
            "result": ["auth.py:42", "login.py:15"]
        }
    }
}
```

## Configuration

### AcpAgentConfig

```python
@FrozenDataclass()
class AcpAgentConfig:
    """Configuration for a specific ACP agent.

    This defines how to spawn and configure a particular agent.
    """
    # Command to spawn the agent
    command: tuple[str, ...] = ("opencode", "acp")

    # Agent-specific environment variables
    env_prefix: str = "OPENCODE"  # e.g., OPENCODE_MODEL, OPENCODE_PROVIDER

    # Config injection mechanism
    config_env_var: str | None = "OPENCODE_CONFIG_CONTENT"  # Inline JSON config


# Pre-configured agents
OPENCODE_AGENT = AcpAgentConfig(
    command=("opencode", "acp"),
    env_prefix="OPENCODE",
    config_env_var="OPENCODE_CONFIG_CONTENT",
)

# Future agents
CLAUDE_CODE_AGENT = AcpAgentConfig(
    command=("claude", "acp"),
    env_prefix="CLAUDE",
    config_env_var="CLAUDE_CONFIG_CONTENT",
)
```

### AcpClientConfig

```python
@FrozenDataclass()
class AcpClientConfig:
    """Configuration for ACP adapter.

    Mirrors ClaudeAgentSDKClientConfig structure.
    """
    # Which agent to use
    agent: AcpAgentConfig = OPENCODE_AGENT

    # Working directory
    cwd: Path | None = None

    # Timeouts
    initialize_timeout: float = 10.0
    prompt_timeout: float | None = None  # None = use deadline

    # Isolation
    isolation: AcpIsolationConfig | None = None

    # Debugging
    suppress_stderr: bool = True
    log_notifications: bool = False

    # Task completion
    task_completion_checker: TaskCompletionChecker | None = None
```

### AcpModelConfig

```python
@FrozenDataclass()
class AcpModelConfig:
    """Model configuration for ACP agents."""
    model: str = "anthropic/claude-sonnet-4-20250514"
    provider: str | None = None  # anthropic, openai, bedrock
    temperature: float | None = None
    max_tokens: int | None = None
```

### AcpIsolationConfig

```python
@FrozenDataclass()
class AcpIsolationConfig:
    """Isolation configuration.

    Equivalent to IsolationConfig + EphemeralHome from SDK adapter.
    """
    # Config injection (agent-specific)
    config_content: str | None = None  # Inline JSON config
    config_dir: Path | None = None  # Config directory override

    # Environment
    env: Mapping[str, str] | None = None
    include_host_env: bool = False

    # Cache isolation
    cache_dir: Path | None = None

    def build_env(
        self,
        agent: AcpAgentConfig,
        model_config: AcpModelConfig,
        workspace_path: Path | None = None,
    ) -> dict[str, str]:
        """Build environment dict for agent subprocess."""
        env: dict[str, str] = {}

        if self.include_host_env:
            for key, value in os.environ.items():
                if not any(s in key.upper() for s in ("KEY", "SECRET", "TOKEN", "PASSWORD")):
                    env[key] = value

        if self.env:
            env.update(self.env)

        # Agent-specific config injection
        prefix = agent.env_prefix
        if self.config_content and agent.config_env_var:
            env[agent.config_env_var] = self.config_content

        if self.config_dir:
            env[f"{prefix}_CONFIG_DIR"] = str(self.config_dir)

        if self.cache_dir:
            env[f"{prefix}_CACHE_DIR"] = str(self.cache_dir)

        # Model configuration
        if model_config.model:
            env[f"{prefix}_MODEL"] = model_config.model
        if model_config.provider:
            env[f"{prefix}_PROVIDER"] = model_config.provider

        env.setdefault("NO_PROXY", "localhost,127.0.0.1")
        return env
```

## Components

### 1. ProcessManager

Manages any ACP agent subprocess lifecycle.

```python
@FrozenDataclass()
class ProcessManager:
    """Manages ACP agent subprocess.

    Works with any ACP-compliant agent by accepting the command
    to spawn from AcpAgentConfig.
    """
    process: subprocess.Popen[bytes]
    stdin: IO[bytes]
    stdout: IO[bytes]
    stderr: IO[bytes]
    _stderr_buffer: list[str]

    @classmethod
    def spawn(
        cls,
        command: tuple[str, ...],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ProcessManager:
        """Spawn ACP agent subprocess.

        Args:
            command: Agent command (e.g., ("opencode", "acp"))
            cwd: Working directory
            env: Environment overrides

        Returns:
            ProcessManager ready for communication
        """
        merged_env = {**os.environ, **(env or {})}

        process = subprocess.Popen(
            command,
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
        """Receive JSON-RPC message."""
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
        """Receive messages until response with matching id."""
        deadline = time.monotonic() + timeout if timeout else None

        while True:
            remaining = (deadline - time.monotonic()) if deadline else None
            if remaining is not None and remaining <= 0:
                raise TimeoutError()

            message = self.receive(timeout=remaining)

            if "id" not in message:
                if on_notification:
                    on_notification(message)
                continue

            if message.get("id") == request_id:
                return message

    def terminate(self, timeout: float = 5.0) -> int:
        """Terminate subprocess gracefully."""
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

JSON-RPC client implementing ACP protocol.

```python
@FrozenDataclass()
class AcpClient:
    """ACP protocol client.

    Works with any ACP-compliant agent.
    """
    process: ProcessManager
    _next_id: int = 0
    _agent_capabilities: AgentCapabilities | None = None

    def initialize(
        self,
        capabilities: ClientCapabilities,
        *,
        timeout: float = 10.0,
    ) -> AgentCapabilities:
        """Initialize ACP connection with tools."""
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

    def session_new(self, *, cwd: str | None = None, timeout: float = 10.0) -> str:
        """Create new session."""
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
        """Send prompt and stream updates."""
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
        """Cancel running prompt."""
        try:
            self._send_request("session/cancel", {"sessionId": session_id}, timeout=5.0)
        except (TimeoutError, AcpError):
            pass

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
```

### 3. Tool Schema Builder

Converts WINK tools to ACP capability format.

```python
def build_tool_schemas(tools: tuple[Tool[Any, Any], ...]) -> list[dict[str, Any]]:
    """Convert WINK tools to ACP tool schemas.

    This is the key function for exposing WINK tools to ACP agents.
    Tools are registered in clientCapabilities during initialize.
    """
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": schema(
                tool.handler.__annotations__.get(
                    list(tool.handler.__annotations__.keys())[0],
                    dict,
                )
            ),
        }
        for tool in tools
    ]


def create_structured_output_tool(output_type: type[T]) -> Tool[StructuredOutputParams, None]:
    """Create structured output tool.

    ACP agents don't have native structured output like Claude SDK.
    This tool provides equivalent functionality.
    """

    @FrozenDataclass()
    class StructuredOutputParams:
        output: dict[str, Any]

    def handler(
        params: StructuredOutputParams,
        *,
        context: ToolContext,
    ) -> ToolResult[None]:
        try:
            parsed = serde.parse(output_type, params.output, extra="ignore")
        except Exception as e:
            return ToolResult.error(f"Invalid output format: {e}")

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

### 4. NotificationHandler

Processes `session/update` notifications and synchronizes WINK state.

```python
class NotificationHandler:
    """Handles session/update notifications.

    Key responsibility: Execute WINK tools when agent invokes them
    via tool_call notifications.
    """

    def __init__(
        self,
        *,
        session: SessionProtocol,
        prompt: PromptProtocol[Any],
        tools: dict[str, Tool[Any, Any]],  # Tool lookup for execution
        adapter_name: AdapterName,
        prompt_name: str,
        deadline: Deadline | None = None,
        budget_tracker: BudgetTracker | None = None,
        heartbeat: Heartbeat | None = None,
        run_context: RunContext | None = None,
    ) -> None:
        self._session = session
        self._prompt = prompt
        self._tools = tools  # name -> Tool for invocation
        self._adapter_name = adapter_name
        self._prompt_name = prompt_name
        self._deadline = deadline
        self._budget_tracker = budget_tracker
        self._heartbeat = heartbeat
        self._run_context = run_context
        self._pending_tools: dict[str, CompositeSnapshot] = {}
        self._stats = ExecutionStats()

    def handle(self, update: SessionUpdate) -> None:
        """Process session update notification."""
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

    def _handle_tool_call(self, content: dict[str, Any]) -> None:
        """Handle tool call lifecycle.

        When agent invokes a WINK tool:
        1. Create snapshot (pending)
        2. Execute tool handler
        3. Dispatch ToolInvoked (complete)
        4. Or restore snapshot (failed)
        """
        tool_id = content["toolId"]
        tool_name = content["toolName"]
        status = content.get("status", "pending")

        match status:
            case "pending" | "running":
                if tool_id not in self._pending_tools:
                    # Create snapshot before tool execution
                    self._pending_tools[tool_id] = create_snapshot(
                        self._session,
                        self._prompt.resources.context,
                        tag=f"tool:{tool_name}",
                    )

                    # Execute WINK tool if it's one of ours
                    if tool_name in self._tools:
                        self._execute_tool(
                            tool_id,
                            tool_name,
                            content.get("arguments", {}),
                        )

            case "complete":
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

            case "failed":
                snapshot = self._pending_tools.pop(tool_id, None)
                if snapshot:
                    restore_snapshot(
                        self._session,
                        self._prompt.resources.context,
                        snapshot,
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

    def _execute_tool(
        self,
        tool_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        """Execute a WINK tool."""
        tool = self._tools[tool_name]

        # Parse arguments
        params_type = tool.handler.__annotations__.get(
            list(tool.handler.__annotations__.keys())[0],
            dict,
        )
        params = serde.parse(params_type, arguments, extra="ignore")

        # Create context
        context = ToolContext(
            prompt=self._prompt,
            session=self._session,
        )

        # Execute with transaction
        with tool_transaction(
            self._session,
            self._prompt.resources.context,
            tag=f"tool:{tool_name}",
        ):
            result = tool.handler(params, context=context)
            # Result is communicated back to agent via protocol

    def _handle_message(self, content: dict[str, Any]) -> None:
        logger.debug("acp.message", context={"preview": str(content.get("text", ""))[:200]})

    def _handle_thought(self, content: dict[str, Any]) -> None:
        logger.debug("acp.thought", context={"preview": str(content.get("text", ""))[:200]})

    def _handle_permission(self, content: dict[str, Any]) -> None:
        logger.warning("acp.permission_request", context={"permission": content})

    @property
    def stats(self) -> ExecutionStats:
        return self._stats
```

## Adapter Implementation

### AcpAdapter

```python
ACP_ADAPTER_NAME: AdapterName = "acp"


class AcpAdapter[OutputT](ProviderAdapter[OutputT]):
    """Adapter using Agent Client Protocol (ACP).

    Works with any ACP-compliant agent. OpenCode is the default.
    """

    def __init__(
        self,
        *,
        model_config: AcpModelConfig | None = None,
        client_config: AcpClientConfig | None = None,
    ) -> None:
        self._model_config = model_config or AcpModelConfig()
        self._client_config = client_config or AcpClientConfig()

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
        """Evaluate prompt via ACP agent."""
        # ... (same structure as before, but using ACP naming)
        return run_async(self._evaluate_async(...))

    async def _run_acp_session(self, ...) -> PromptResponse[OutputT]:
        """Run ACP session with tool exposure."""
        # Build tool schemas for ACP
        tools = list(rendered.tools)
        if rendered.output_type and rendered.output_type is not type(None):
            tools.append(create_structured_output_tool(rendered.output_type))

        tool_schemas = build_tool_schemas(tuple(tools))
        tool_lookup = {t.name: t for t in tools}

        # Initialize with tools in capabilities
        capabilities = ClientCapabilities(
            supports_text_prompts=True,
            supports_streaming=True,
            tools=tool_schemas,  # ◄── Tools declared here
        )

        agent_caps = client.initialize(capabilities, timeout=...)

        # Create handler with tool lookup for execution
        handler = NotificationHandler(
            session=session,
            prompt=prompt,
            tools=tool_lookup,  # ◄── Tools available for execution
            ...
        )

        # Send prompt - agent can now invoke declared tools
        result = client.session_prompt(session_id, prompt_text, on_update=handler.handle)
        ...
```

## Usage Examples

### Basic Usage with OpenCode

```python
from weakincentives.adapters.acp import AcpAdapter, AcpClientConfig

adapter = AcpAdapter()  # Uses OpenCode by default
response = adapter.evaluate(prompt, session=session)
```

### With Custom Agent

```python
from weakincentives.adapters.acp import (
    AcpAdapter,
    AcpClientConfig,
    AcpAgentConfig,
)

# Configure a different ACP agent
custom_agent = AcpAgentConfig(
    command=("my-agent", "acp"),
    env_prefix="MY_AGENT",
    config_env_var="MY_AGENT_CONFIG",
)

adapter = AcpAdapter(
    client_config=AcpClientConfig(agent=custom_agent),
)
```

### With Custom Tools

```python
# Define custom tool
@FrozenDataclass()
class SearchParams:
    query: str

def search_handler(params: SearchParams, *, context: ToolContext) -> ToolResult[list[str]]:
    results = do_search(params.query)
    return ToolResult.ok(results)

search_tool = Tool(
    name="search_code",
    description="Search codebase",
    handler=search_handler,
)

# Include in prompt
template = PromptTemplate(
    sections=(
        MarkdownSection(
            key="task",
            template="Find authentication code",
            tools=(search_tool,),  # ◄── Tool exposed to ACP agent
        ),
    ),
)

# Agent can now invoke search_code
response = adapter.evaluate(Prompt(template), session=session)
```

### With Structured Output

```python
@FrozenDataclass()
class ReviewResult:
    issues: list[str]
    fixed: bool

template = PromptTemplate[ReviewResult](
    sections=(
        MarkdownSection(key="task", template="Review code"),
    ),
)

# structured_output tool added automatically
response = adapter.evaluate(Prompt(template), session=session)
print(response.output)  # ReviewResult instance
```

## Implementation Checklist

### Phase 1: Core Protocol

- [ ] `ProcessManager` - subprocess spawn/terminate
- [ ] `AcpClient` - JSON-RPC protocol
- [ ] `build_tool_schemas()` - **tool exposure**
- [ ] Protocol dataclasses

### Phase 2: Tool Execution

- [ ] `NotificationHandler` - with tool execution
- [ ] `create_structured_output_tool()`
- [ ] `StructuredOutputState` slice

### Phase 3: Adapter

- [ ] `AcpAdapter` - main adapter
- [ ] `AcpAgentConfig` - agent configuration
- [ ] `AcpIsolationConfig` - environment building

### Phase 4: Workspace

- [ ] `AcpWorkspaceSection`
- [ ] Reuse `HostMount` patterns

## Related Specifications

- `specs/ADAPTERS.md` - Provider adapter protocol
- `specs/CLAUDE_AGENT_SDK.md` - Reference implementation
- `specs/TOOLS.md` - Tool registration and policies

## External References

- [Agent Client Protocol](https://agentclientprotocol.com/)
- [ACP GitHub Repository](https://github.com/agentclientprotocol/agent-client-protocol)
- [OpenCode Documentation](https://opencode.ai/docs/)

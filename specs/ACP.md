# ACP Adapter Specification

> **Protocol**: Agent Client Protocol (ACP) v1
> **SDK**: `agent-client-protocol` Python package
> **Reference**: `src/weakincentives/adapters/claude_agent_sdk/`

## Purpose

`AcpAdapter` evaluates prompts via the Agent Client Protocol (ACP)—an open
standard for AI agent integration. WINK uses the official ACP Python SDK
to communicate with any ACP-compliant agent.

**Implementation:** `src/weakincentives/adapters/acp/`

**Dependency:** `agent-client-protocol` (PyPI)

## What is ACP?

The Agent Client Protocol (ACP) is an open standard developed by Zed, JetBrains,
and the broader community that standardizes communication between clients and
AI coding agents. ACP is to coding agents what LSP is to language servers.

### Key Characteristics

- **JSON-RPC 2.0** over stdio (newline-delimited)
- **Subprocess model**: Client spawns agent, communicates via pipes
- **Bidirectional requests**: Both client and agent can send requests
- **Capability negotiation**: Agent and client declare supported features
- **MCP integration**: Tools provided via MCP servers passed at session creation
- **Streaming updates**: Real-time notifications during execution

### Using the Official SDK

The official ACP Python SDK (`agent-client-protocol`) provides:

- **`acp.schema`**: Generated Pydantic models tracking every ACP release
- **`acp.helpers`**: Content blocks, tool calls, and session updates
- **`acp.contrib`**: Session accumulators, tool call trackers, permission brokers
- **Asyncio transports**: Stdio JSON-RPC plumbing and lifecycle helpers

```python
# Installation
# uv add agent-client-protocol

from acp.client import AcpClient
from acp.schema import (
    InitializeParams,
    SessionNewParams,
    SessionPromptParams,
    ContentBlock,
    McpServerConfig,
)
```

### Supported Agents

Any ACP-compliant agent works with this adapter:

| Agent | Command | Status |
| ------------ | -------------- | ---------------------- |
| **OpenCode** | `opencode acp` | Primary implementation |
| Claude Code | `claude acp` | Planned |
| Codex CLI | `codex acp` | Potential |
| Gemini CLI | `gemini acp` | Potential |
| goose | `goose acp` | Potential |

OpenCode is the initial reference implementation. Other agents can be added
by specifying the appropriate command in `AcpProviderConfig`.

______________________________________________________________________

## Tool Exposure via MCP

**Critical**: ACP exposes tools through MCP servers, not through client
capabilities. This matches the Claude Agent SDK pattern where WINK already
runs an MCP server for tool bridging.

### How Tools Work in ACP

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TOOL EXPOSURE FLOW                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. WINK renders prompt → extracts tools from sections                  │
│                                                                         │
│  2. WINK starts MCP server exposing tools:                              │
│     - Uses existing `create_sdk_mcp_server()` infrastructure            │
│     - Server listens on stdio or SSE (agent-dependent)                  │
│                                                                         │
│  3. WINK sends session/new with mcpServers:                             │
│     {                                                                   │
│       "method": "session/new",                                          │
│       "params": {                                                       │
│         "cwd": "/absolute/path/to/workspace",                           │
│         "mcpServers": [                                                 │
│           {                                                             │
│             "name": "wink-tools",                                       │
│             "transport": { "type": "stdio", "command": "..." }          │
│           }                                                             │
│         ]                                                               │
│       }                                                                 │
│     }                                                                   │
│                                                                         │
│  4. Agent connects to MCP server, discovers available tools             │
│                                                                         │
│  5. During prompt, agent calls tools via MCP protocol                   │
│     WINK's MCP server executes with transactional semantics             │
│                                                                         │
│  6. Agent reports tool calls via session/update notifications           │
│     (WINK observes but doesn't execute - already handled via MCP)       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### MCP Server Integration

WINK reuses the existing MCP infrastructure from Claude Agent SDK:

```python
from weakincentives.adapters.claude_agent_sdk import create_sdk_mcp_server

def create_wink_mcp_server(
    tools: tuple[Tool[Any, Any], ...],
    session: SessionProtocol,
    prompt: PromptProtocol[Any],
) -> McpServer:
    """Create MCP server exposing WINK tools.

    Same infrastructure as Claude Agent SDK adapter.
    """
    return create_sdk_mcp_server(
        tools=tools,
        session=session,
        prompt=prompt,
        # Transactional execution happens inside MCP handlers
    )
```

### Comparison: ACP vs Claude Agent SDK Tool Bridging

| Aspect | Claude Agent SDK | ACP Adapter |
| ------------------ | ---------------------------- | -------------------------------- |
| **Declaration** | MCP server config in options | MCP server in `session/new` |
| **Transport** | MCP protocol | MCP protocol (same) |
| **Server Lifecycle** | Per-session | Per-session |
| **Tool Schemas** | JSON Schema via MCP | JSON Schema via MCP |
| **Transactional** | `tool_transaction()` in MCP | `tool_transaction()` in MCP |
| **Agent observation** | SDK hooks | `session/update` notifications |

**Key insight**: Both adapters use MCP for tool execution. The difference is
how the MCP server is connected to the agent.

______________________________________________________________________

## ACP Protocol Details

### Message Shapes (Per ACP Specification)

ACP defines specific message shapes that must be followed exactly.

#### Initialize Request

```python
{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "clientCapabilities": {
            "fs": False,       # Whether client implements fs/* methods
            "terminal": False  # Whether client implements terminal/* methods
        },
        "clientInfo": {
            "name": "wink",
            "version": "1.0.0"
        }
    }
}
```

#### Initialize Response

```python
{
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "protocolVersion": "2024-11-05",
        "agentCapabilities": {
            "promptCapabilities": {
                "image": False,
                "audio": False,
                "embeddedContext": False
            },
            "mcp": {
                "http": True,
                "sse": True
            }
        },
        "agentInfo": {
            "name": "opencode",
            "version": "0.1.0"
        },
        "authMethods": []
    }
}
```

#### Session New Request

```python
{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "session/new",
    "params": {
        "cwd": "/absolute/path/to/workspace",  # Required, must be absolute
        "mcpServers": [
            {
                "name": "wink-tools",
                "transport": {
                    "type": "stdio",
                    "command": "python",
                    "args": ["-m", "wink.mcp_server", "--session-id", "..."]
                }
            }
        ]
    }
}
```

#### Session Prompt Request

```python
{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "session/prompt",
    "params": {
        "sessionId": "session-123",
        "prompt": [  # ContentBlock[], NOT a string
            {"type": "text", "text": "Review the code in src/main.py"}
        ]
    }
}
```

#### Session Prompt Response

```python
{
    "jsonrpc": "2.0",
    "id": 3,
    "result": {
        "stopReason": "end_turn"  # Just stop reason, content comes via updates
    }
}
```

#### Session Update Notification (from agent)

```python
{
    "jsonrpc": "2.0",
    "method": "session/update",
    "params": {
        "sessionId": "session-123",
        "update": {
            "sessionUpdate": "tool_call",  # Discriminator field
            "toolCallId": "call-456",
            "name": "search_code",
            "arguments": {"query": "def authenticate"},
            "status": "pending"  # pending | in_progress | completed | failed
        }
    }
}

# Agent message chunk
{
    "jsonrpc": "2.0",
    "method": "session/update",
    "params": {
        "sessionId": "session-123",
        "update": {
            "sessionUpdate": "agent_message_chunk",
            "text": "I found several matches..."
        }
    }
}

# Tool call update (status change)
{
    "jsonrpc": "2.0",
    "method": "session/update",
    "params": {
        "sessionId": "session-123",
        "update": {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "call-456",
            "status": "completed",
            "result": "Found 3 matches"
        }
    }
}
```

#### Session Cancel Notification

**Important**: Cancel is a notification, not a request. No response expected.

```python
{
    "jsonrpc": "2.0",
    "method": "session/cancel",  # No "id" field
    "params": {
        "sessionId": "session-123"
    }
}
```

#### Permission Request (from agent)

The agent can request permission via a JSON-RPC **request** (has `id`):

```python
# Agent sends:
{
    "jsonrpc": "2.0",
    "id": 100,
    "method": "session/request_permission",
    "params": {
        "sessionId": "session-123",
        "permission": {
            "type": "file_edit",
            "path": "/workspace/src/main.py"
        }
    }
}

# Client must respond:
{
    "jsonrpc": "2.0",
    "id": 100,
    "result": {
        "outcome": "allow"  # allow | deny | cancelled
    }
}
```

______________________________________________________________________

## Using the ACP Python SDK

The official SDK handles all bidirectional JSON-RPC complexity:

### Client Setup

```python
from acp.client import AcpClient
from acp.schema import ClientCapabilities, ClientInfo
from acp.contrib import PermissionBroker, SessionAccumulator

async def create_acp_client(
    command: tuple[str, ...],
    *,
    cwd: Path,
    auto_allow_permissions: bool = True,
) -> AcpClient:
    """Create ACP client using official SDK.

    The SDK handles:
    - Bidirectional message routing
    - Permission request/response
    - Session update accumulation
    - Tool call tracking
    """
    # Permission broker handles permission requests from agent
    permission_broker = PermissionBroker(
        auto_allow=auto_allow_permissions,
    )

    # Create client with subprocess transport
    client = await AcpClient.spawn(
        command=command,
        cwd=cwd,
        permission_handler=permission_broker,
    )

    return client
```

### Session Lifecycle with SDK

```python
from acp.schema import (
    InitializeParams,
    SessionNewParams,
    SessionPromptParams,
    ContentBlock,
    McpServerConfig,
    McpServerTransport,
)
from acp.contrib import SessionAccumulator

async def run_session(
    client: AcpClient,
    prompt_text: str,
    mcp_server: McpServer,
    *,
    cwd: Path,
) -> SessionResult:
    """Run ACP session using official SDK."""

    # Initialize (SDK provides typed params)
    await client.initialize(
        InitializeParams(
            protocol_version="2024-11-05",
            client_capabilities=ClientCapabilities(
                fs=False,
                terminal=False,
            ),
            client_info=ClientInfo(name="wink", version="1.0.0"),
        )
    )

    # Create session with MCP server
    session = await client.session_new(
        SessionNewParams(
            cwd=str(cwd.absolute()),
            mcp_servers=[
                McpServerConfig(
                    name="wink-tools",
                    transport=mcp_server.transport_config(),
                )
            ],
        )
    )

    # Use SDK's accumulator for transcript building
    accumulator = SessionAccumulator()

    # Send prompt with streaming updates
    result = await client.session_prompt(
        SessionPromptParams(
            session_id=session.session_id,
            prompt=[ContentBlock(type="text", text=prompt_text)],
        ),
        on_update=accumulator.handle_update,
    )

    return SessionResult(
        transcript=accumulator.get_transcript(),
        tool_calls=accumulator.get_tool_calls(),
        stop_reason=result.stop_reason,
    )
```

### Permission Handling

The SDK provides `PermissionBroker` for handling agent permission requests:

```python
from acp.contrib import PermissionBroker

# Auto-allow all permissions
broker = PermissionBroker(auto_allow=True)

# Custom permission logic
class WinkPermissionBroker(PermissionBroker):
    """WINK-specific permission handling."""

    def __init__(self, allowed_paths: list[Path]):
        self._allowed_paths = allowed_paths

    async def handle_permission(self, permission: Permission) -> PermissionOutcome:
        if permission.type == "file_edit":
            path = Path(permission.path)
            if any(path.is_relative_to(p) for p in self._allowed_paths):
                return PermissionOutcome.ALLOW
            return PermissionOutcome.DENY

        # Default: allow
        return PermissionOutcome.ALLOW
```

### Cancel and Cleanup

```python
# SDK handles cancel as notification (not request)
await client.session_cancel(session_id)

# Per ACP spec, continue accepting updates after cancel
# SDK's PermissionBroker automatically responds with "cancelled"
# to pending permission requests after cancel is sent
```

______________________________________________________________________

## Client Capabilities

ACP client capabilities are **not** a tools array. They declare which
client-implemented methods the agent can call.

```python
@FrozenDataclass()
class AcpClientCapabilities:
    """Client capabilities for ACP.

    These declare which client methods the agent can call.
    """
    # Whether client implements fs/* methods (read, write, list, etc.)
    fs: bool = False

    # Whether client implements terminal/* methods (run commands)
    terminal: bool = False

    # Optional metadata for extensions
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "fs": self.fs,
            "terminal": self.terminal,
        }
        if self.meta:
            result["_meta"] = self.meta
        return result
```

### When to Enable fs/terminal

| Capability | Enable When | WINK Implication |
| ----------- | -------------------------------------------- | ----------------------------------- |
| `fs: true` | WINK wants to be the file system executor | Must implement fs/\* handlers |
| `fs: false` | Agent handles its own file operations | Agent edits files directly |
| `terminal: true` | WINK wants to execute shell commands | Must implement terminal/\* handlers |
| `terminal: false` | Agent handles its own commands | Agent runs commands directly |

For WINK's transactional semantics, consider:

- Set `fs: false, terminal: false` and let agent execute directly
- Provide WINK tools via MCP that wrap operations with transactions
- Use sandboxed workspace for isolation

______________________________________________________________________

## Session Update Handling

### Update Types (per ACP schema)

```python
class SessionUpdateType(Enum):
    """ACP session update types."""
    TOOL_CALL = "tool_call"
    TOOL_CALL_UPDATE = "tool_call_update"
    AGENT_MESSAGE_CHUNK = "agent_message_chunk"
    AGENT_THOUGHT_CHUNK = "agent_thought_chunk"
    # Additional types as defined by ACP


@FrozenDataclass()
class SessionUpdate:
    """Parsed session update notification."""
    session_id: str
    update_type: SessionUpdateType
    content: dict[str, Any]

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> SessionUpdate:
        update = params["update"]
        update_type = SessionUpdateType(update["sessionUpdate"])
        return cls(
            session_id=params["sessionId"],
            update_type=update_type,
            content=update,
        )
```

### Notification Handler

```python
class NotificationHandler:
    """Handles session/update notifications.

    Builds transcript from agent messages and observes tool activity.
    Does NOT execute tools - that happens via MCP.
    """

    def __init__(
        self,
        *,
        session: SessionProtocol,
        heartbeat: Heartbeat | None = None,
    ):
        self._session = session
        self._heartbeat = heartbeat
        self._transcript: list[str] = []
        self._tool_calls: dict[str, ToolCallRecord] = {}
        self._stats = ExecutionStats()

    def handle(self, update: SessionUpdate) -> None:
        """Process session update notification."""
        if self._heartbeat:
            self._heartbeat.beat()

        match update.update_type:
            case SessionUpdateType.AGENT_MESSAGE_CHUNK:
                self._handle_message_chunk(update.content)
            case SessionUpdateType.AGENT_THOUGHT_CHUNK:
                self._handle_thought_chunk(update.content)
            case SessionUpdateType.TOOL_CALL:
                self._handle_tool_call(update.content)
            case SessionUpdateType.TOOL_CALL_UPDATE:
                self._handle_tool_call_update(update.content)

    def _handle_message_chunk(self, content: dict[str, Any]) -> None:
        """Accumulate agent message text."""
        text = content.get("text", "")
        self._transcript.append(text)

    def _handle_thought_chunk(self, content: dict[str, Any]) -> None:
        """Log agent thought (internal reasoning)."""
        text = content.get("text", "")
        logger.debug("acp.thought", preview=text[:200])

    def _handle_tool_call(self, content: dict[str, Any]) -> None:
        """Observe tool call start (agent executes via MCP)."""
        tool_call_id = content["toolCallId"]
        self._tool_calls[tool_call_id] = ToolCallRecord(
            id=tool_call_id,
            name=content["name"],
            arguments=content.get("arguments", {}),
            status=content.get("status", "pending"),
        )

    def _handle_tool_call_update(self, content: dict[str, Any]) -> None:
        """Observe tool call status change."""
        tool_call_id = content["toolCallId"]
        if tool_call_id in self._tool_calls:
            record = self._tool_calls[tool_call_id]
            record.status = content.get("status", record.status)
            if content.get("status") == "completed":
                record.result = content.get("result")
                self._stats.tool_count += 1

    def get_transcript(self) -> str:
        """Get accumulated agent message text."""
        return "".join(self._transcript)

    @property
    def stats(self) -> ExecutionStats:
        return self._stats
```

______________________________________________________________________

## Structured Output via MCP Tool

Since WINK tools are exposed via MCP, the `structured_output` tool is also
an MCP tool that the agent calls.

```python
def create_structured_output_mcp_tool(
    output_type: type[T],
    session: SessionProtocol,
) -> McpTool:
    """Create MCP tool for structured output.

    Agent calls this via MCP when task is complete.
    """

    async def handler(arguments: dict[str, Any]) -> str:
        # Check for duplicate calls
        existing = session[StructuredOutputState].latest()
        if existing and existing.outputs:
            return "Error: structured_output already called."

        # Parse and validate
        try:
            parsed = serde.parse(output_type, arguments["output"], extra="ignore")
        except Exception as e:
            return f"Error: Invalid output format: {e}"

        # Record the output
        session.dispatcher.dispatch(
            StructuredOutputEmitted(value=parsed)
        )

        return "Output recorded. Task complete."

    return McpTool(
        name="structured_output",
        description=(
            "Emit structured output when task is complete. "
            "Call EXACTLY ONCE with your final result."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "output": {"type": "object"}
            },
            "required": ["output"]
        },
        handler=handler,
    )
```

______________________________________________________________________

## Error Taxonomy

```python
class AcpErrorCategory(Enum):
    """Error categories for retry/recovery decisions."""
    TRANSIENT = "transient"      # Network, timeout → retry
    AGENT = "agent"              # Invalid response → abort
    PROTOCOL = "protocol"        # Version/shape mismatch → fail fast
    CANCELLED = "cancelled"      # User/deadline cancellation


@FrozenDataclass()
class AcpError(Exception):
    """Structured ACP error with category."""
    code: int
    message: str
    category: AcpErrorCategory
    data: dict[str, Any] | None = None

    @classmethod
    def from_response(cls, error: dict[str, Any]) -> AcpError:
        """Parse JSON-RPC error into typed AcpError."""
        code = error.get("code", -32000)
        message = error.get("message", "Unknown error")

        category = cls._categorize(code, message)
        return cls(code=code, message=message, category=category)

    @staticmethod
    def _categorize(code: int, message: str) -> AcpErrorCategory:
        if code == -32600:  # Invalid request
            return AcpErrorCategory.PROTOCOL
        if code == -32601:  # Method not found
            return AcpErrorCategory.PROTOCOL
        if code == -32602:  # Invalid params
            return AcpErrorCategory.PROTOCOL
        if code == -32603:  # Internal error
            return AcpErrorCategory.TRANSIENT
        if "cancel" in message.lower():
            return AcpErrorCategory.CANCELLED
        return AcpErrorCategory.AGENT
```

______________________________________________________________________

## Deadline Integration

### DeadlineMonitor

```python
class DeadlineMonitor:
    """Background task that sends cancel notification on deadline expiry."""

    def __init__(
        self,
        router: AcpMessageRouter,
        session_id: str,
        deadline: Deadline,
    ):
        self._router = router
        self._session_id = session_id
        self._deadline = deadline
        self._cancelled = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._cancelled.set()

    def _monitor(self) -> None:
        while not self._cancelled.wait(timeout=1.0):
            if self._deadline.is_expired():
                # Send cancel NOTIFICATION (not request)
                self._router.send_notification(
                    "session/cancel",
                    {"sessionId": self._session_id},
                )
                return
```

______________________________________________________________________

## Configuration

### AcpProviderConfig

```python
@FrozenDataclass()
class AcpProviderConfig:
    """Configuration for a specific ACP agent provider."""
    command: tuple[str, ...] = ("opencode", "acp")
    env_prefix: str = "OPENCODE"
    config_env_var: str | None = "OPENCODE_CONFIG_CONTENT"


OPENCODE_PROVIDER = AcpProviderConfig(
    command=("opencode", "acp"),
    env_prefix="OPENCODE",
    config_env_var="OPENCODE_CONFIG_CONTENT",
)

CLAUDE_CODE_PROVIDER = AcpProviderConfig(
    command=("claude", "acp"),
    env_prefix="CLAUDE",
    config_env_var="CLAUDE_CONFIG_CONTENT",
)
```

### AcpAdapterConfig

```python
@FrozenDataclass()
class AcpAdapterConfig:
    """Configuration for ACP adapter."""
    provider: AcpProviderConfig = OPENCODE_PROVIDER
    cwd: Path | None = None
    initialize_timeout: float = 10.0
    prompt_timeout: float | None = None

    # Client capabilities
    enable_fs: bool = False      # Implement fs/* methods
    enable_terminal: bool = False  # Implement terminal/* methods

    # Permission handling
    auto_allow_permissions: bool = True

    # MCP server for WINK tools
    mcp_server_command: tuple[str, ...] | None = None

    # Isolation
    isolation: AcpIsolationConfig | None = None
```

______________________________________________________________________

## Adapter Implementation

### AcpAdapter (Using Official SDK)

```python
from acp.client import AcpClient
from acp.schema import (
    InitializeParams,
    SessionNewParams,
    SessionPromptParams,
    ContentBlock,
    McpServerConfig,
    ClientCapabilities,
    ClientInfo,
)
from acp.contrib import PermissionBroker, SessionAccumulator


class AcpAdapter[OutputT](ProviderAdapter[OutputT]):
    """Adapter using Agent Client Protocol (ACP).

    Uses the official ACP Python SDK for protocol handling.
    """

    def __init__(
        self,
        *,
        model_config: AcpModelConfig | None = None,
        adapter_config: AcpAdapterConfig | None = None,
    ) -> None:
        self._model_config = model_config or AcpModelConfig()
        self._adapter_config = adapter_config or AcpAdapterConfig()

    async def _run_acp_session(
        self,
        prompt: Prompt[OutputT],
        rendered: RenderedPrompt[OutputT],
        *,
        session: SessionProtocol,
        deadline: Deadline | None,
        heartbeat: Heartbeat | None,
    ) -> PromptResponse[OutputT]:
        """Run ACP session with MCP tool exposure."""
        config = self._adapter_config

        # Start MCP server for WINK tools
        tools = list(rendered.tools)
        if rendered.output_type and rendered.output_type is not type(None):
            tools.append(create_structured_output_mcp_tool(
                rendered.output_type, session
            ))

        mcp_server = create_wink_mcp_server(
            tools=tuple(tools),
            session=session,
            prompt=prompt,
        )

        async with mcp_server:
            # Create ACP client using official SDK
            client = await AcpClient.spawn(
                command=config.provider.command,
                cwd=config.cwd,
                permission_handler=PermissionBroker(
                    auto_allow=config.auto_allow_permissions
                ),
            )

            try:
                # Initialize with SDK's typed params
                await client.initialize(
                    InitializeParams(
                        protocol_version="2024-11-05",
                        client_capabilities=ClientCapabilities(
                            fs=config.enable_fs,
                            terminal=config.enable_terminal,
                        ),
                        client_info=ClientInfo(name="wink", version="1.0.0"),
                    )
                )

                # Create session with MCP server
                acp_session = await client.session_new(
                    SessionNewParams(
                        cwd=str(config.cwd.absolute()),
                        mcp_servers=[
                            McpServerConfig(
                                name="wink-tools",
                                transport=mcp_server.transport_config(),
                            )
                        ],
                    )
                )

                # Use SDK's accumulator for transcript
                accumulator = SessionAccumulator()

                def on_update(update):
                    accumulator.handle_update(update)
                    if heartbeat:
                        heartbeat.beat()

                # Start deadline monitor
                monitor = None
                if deadline:
                    monitor = DeadlineMonitor(client, acp_session.session_id, deadline)
                    monitor.start()

                try:
                    # Send prompt with SDK (handles ContentBlock[] format)
                    result = await client.session_prompt(
                        SessionPromptParams(
                            session_id=acp_session.session_id,
                            prompt=[ContentBlock(type="text", text=rendered.text)],
                        ),
                        on_update=on_update,
                    )
                finally:
                    if monitor:
                        monitor.stop()

                # Build response from accumulator
                return PromptResponse(
                    output=self._extract_output(session, rendered.output_type),
                    text=accumulator.get_transcript(),
                    stop_reason=result.stop_reason,
                )
            finally:
                await client.close()
```

______________________________________________________________________

## Feature Comparison Matrix

### Claude Agent SDK vs ACP Adapter

| Feature | Claude Agent SDK | ACP Adapter |
| ------------------------- | ------------------------------- | ------------------------------------ |
| **Process Model** | | |
| Subprocess management | SDK internal | WINK owns |
| Process communication | SDK methods | JSON-RPC stdio |
| **Tool Exposure** | | |
| Mechanism | MCP server in SDK options | MCP server in `session/new` |
| Same MCP infrastructure | Yes | Yes |
| **Bidirectional RPC** | | |
| Permission requests | SDK handles internally | Client must respond |
| fs/terminal methods | SDK handles internally | Client implements if enabled |
| **Session Lifecycle** | | |
| Initialization | SDK constructor | `initialize` request |
| Session creation | SDK connect | `session/new` with mcpServers |
| Prompt format | SDK prompt | ContentBlock[] array |
| Cancellation | SDK internal | `session/cancel` notification |
| **Output Building** | | |
| Text accumulation | SDK provides | Build from agent_message_chunk |
| Stop reason | SDK provides | From prompt response |

______________________________________________________________________

## Implementation Checklist

### Phase 1: SDK Integration

- [ ] Add `agent-client-protocol` dependency
- [ ] Configure SDK client with WINK's permission policy
- [ ] Integrate SDK's `SessionAccumulator` for transcript building

### Phase 2: MCP Integration

- [ ] Reuse `create_sdk_mcp_server()` for tool exposure
- [ ] Configure MCP server transport for `session/new`
- [ ] `structured_output` as MCP tool

### Phase 3: Session Management

- [ ] `DeadlineMonitor` with SDK's cancel notification
- [ ] Heartbeat integration with SDK's update callbacks
- [ ] Error handling with SDK's typed exceptions

### Phase 4: Adapter

- [ ] `AcpAdapter` with SDK client lifecycle
- [ ] `AcpProviderConfig` for different agents
- [ ] Output extraction from session state

### Phase 5: Testing

- [ ] Integration tests with OpenCode
- [ ] Mock using SDK's test utilities (if available)
- [ ] Verify MCP tool execution through full flow

______________________________________________________________________

## Related Specifications

- `specs/ADAPTERS.md` - Provider adapter protocol
- `specs/CLAUDE_AGENT_SDK.md` - Reference implementation (MCP patterns)
- `specs/TOOLS.md` - Tool registration and policies

## External References

### ACP Python SDK

- [ACP Python SDK Repository](https://github.com/agentclientprotocol/python-sdk)
- PyPI: `agent-client-protocol`

### ACP Protocol Specification

- [Agent Client Protocol - Overview](https://agentclientprotocol.com/protocol/overview)
- [Agent Client Protocol - Initialization](https://agentclientprotocol.com/protocol/initialization)
- [Agent Client Protocol - Session Setup](https://agentclientprotocol.com/protocol/session-setup)
- [Agent Client Protocol - Prompt Turn](https://agentclientprotocol.com/protocol/prompt-turn)
- [Agent Client Protocol - Tool Calls](https://agentclientprotocol.com/protocol/tool-calls)
- [Agent Client Protocol - Schema](https://agentclientprotocol.com/protocol/schema)

### Agent Documentation

- [OpenCode Documentation](https://opencode.ai/docs/)

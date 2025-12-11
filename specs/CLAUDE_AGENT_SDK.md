# Claude Agent SDK Adapter Specification

## Purpose

The Claude Agent SDK adapter enables weakincentives prompts to leverage Claude's
full agentic capabilities through the official `claude-agent-sdk` Python package.
Unlike other adapters that use the shared `InnerLoop` for tool dispatch, this
adapter delegates tool execution entirely to the SDK, preserving Claude Code's
native tool handling, permission model, and multi-turn reasoning.

## Guiding Principles

- **Full SDK power**: Embrace the SDK's agentic loop rather than wrapping it in
  weakincentives abstractions. The SDK handles tool execution, retries, and
  conversation state.
- **Dynamic tool bridging**: Translate weakincentives `Tool` definitions to SDK
  MCP tools at runtime, enabling native execution with full type safety.
- **Structured output via tools**: Use a dedicated `respond` tool pattern for
  typed outputs, since the SDK lacks native JSON schema enforcement.
- **Observable integration**: Publish weakincentives events at key boundaries
  while respecting that internal SDK operations are opaque.
- **Graceful async bridging**: Handle the SDK's async-only API without blocking
  event loops or breaking nested execution.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      ClaudeAgentSDKAdapter                              │
├─────────────────────────────────────────────────────────────────────────┤
│  evaluate()                                                             │
│    │                                                                    │
│    ├── Render prompt → RenderedPrompt                                   │
│    │                                                                    │
│    ├── Build system prompt from rendered sections                       │
│    │                                                                    │
│    ├── Create ephemeral MCP server from Tool definitions                │
│    │     └── Each Tool[ParamsT, ResultT] → @tool decorated handler      │
│    │                                                                    │
│    ├── If structured output requested:                                  │
│    │     └── Add `respond` tool with output schema                      │
│    │                                                                    │
│    ├── Invoke SDK (query or ClaudeSDKClient)                            │
│    │     └── SDK manages full agentic loop internally                   │
│    │                                                                    │
│    ├── Collect ResultMessage with usage/cost                            │
│    │                                                                    │
│    └── Return PromptResponse[OutputT]                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Difference from Other Adapters:**

| Aspect | OpenAI/LiteLLM | Claude Agent SDK |
|--------|----------------|------------------|
| Tool loop | `InnerLoop` in weakincentives | SDK internal loop |
| Tool execution | `ToolExecutor` | SDK via MCP bridge |
| Multi-turn | Explicit message threading | SDK session state |
| Streaming | Not exposed | `AsyncIterator[Message]` |
| Structured output | JSON schema response format | `respond` tool pattern |

## SDK Fundamentals

### Installation

```bash
pip install claude-agent-sdk
```

Requires Claude Code CLI installed on the system.

### Core APIs

The SDK provides two interaction patterns:

**Stateless (`query`)**: Fresh session per call, returns `AsyncIterator[Message]`.

```python
async for message in query("prompt", options=options):
    ...
```

**Stateful (`ClaudeSDKClient`)**: Maintains conversation across multiple exchanges.

```python
async with ClaudeSDKClient(options) as client:
    await client.query("first")
    async for msg in client.receive_response():
        ...
    await client.query("follow-up")  # Remembers context
```

### Message Types

```python
UserMessage       # User input
AssistantMessage  # Claude response with content blocks
SystemMessage     # System metadata
ResultMessage     # Final result with duration, cost, usage, session_id
```

### Content Blocks

```python
TextBlock         # Text responses
ThinkingBlock     # Internal reasoning (when available)
ToolUseBlock      # Tool invocation requests
ToolResultBlock   # Tool execution results
```

## Configuration

### ClaudeAgentSDKClientConfig

```python
@FrozenDataclass()
class ClaudeAgentSDKClientConfig:
    """Client-level configuration for Claude Agent SDK."""

    permission_mode: PermissionMode = "bypassPermissions"
    cwd: str | None = None
    add_dirs: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    setting_sources: tuple[SettingSource, ...] = ()
    sandbox: SandboxSettings | None = None


PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions"]
SettingSource = Literal["user", "project", "local"]
```

**Fields:**

| Field | Default | Description |
|-------|---------|-------------|
| `permission_mode` | `"bypassPermissions"` | Tool permission handling |
| `cwd` | `None` | Working directory for SDK operations |
| `add_dirs` | `()` | Additional accessible directories |
| `env` | `None` | Environment variables passed to SDK |
| `setting_sources` | `()` | Config file sources (empty = isolated) |
| `sandbox` | `None` | Sandboxing configuration |

### ClaudeAgentSDKModelConfig

```python
@FrozenDataclass()
class ClaudeAgentSDKModelConfig(LLMConfig):
    """Model-level configuration for Claude Agent SDK."""

    model: str = "claude-sonnet-4-20250514"
```

Inherits `temperature`, `max_tokens`, `top_p` from `LLMConfig`. Note that SDK
support for these parameters may vary.

### SandboxSettings

```python
@FrozenDataclass()
class SandboxSettings:
    """Sandboxing configuration for SDK execution."""

    exclude_commands: tuple[str, ...] = ()
    allow_local_binding: bool = False
    allow_unix_sockets: bool = False
```

## Adapter Implementation

### Constructor

```python
class ClaudeAgentSDKAdapter(ProviderAdapter[OutputT]):
    """Adapter leveraging Claude Agent SDK's full agentic capabilities."""

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-20250514",
        client_config: ClaudeAgentSDKClientConfig | None = None,
        model_config: ClaudeAgentSDKModelConfig | None = None,
        stateful: bool = False,
    ) -> None:
        """Initialize the Claude Agent SDK adapter.

        Args:
            model: Claude model identifier.
            client_config: SDK client configuration.
            model_config: Model parameter configuration.
            stateful: Use ClaudeSDKClient for multi-turn (default: query).
        """
```

### evaluate Method

```python
def evaluate(
    self,
    prompt: Prompt[OutputT],
    *,
    session: SessionProtocol,
    deadline: Deadline | None = None,
    visibility_overrides: Mapping[SectionPath, SectionVisibility] | None = None,
    budget: Budget | None = None,
    budget_tracker: BudgetTracker | None = None,
) -> PromptResponse[OutputT]:
    """Evaluate the prompt using Claude Agent SDK.

    Unlike other adapters, this bypasses InnerLoop entirely. The SDK
    manages its own agentic loop, executing tools via an ephemeral
    MCP server bridge.
    """
```

**Execution Flow:**

1. Render prompt with visibility overrides
2. Build system prompt from rendered markdown
3. Create MCP tool bridge from `RenderedPrompt.tools`
4. Add `respond` tool if structured output declared
5. Configure `ClaudeAgentOptions`
6. Execute via `query()` or `ClaudeSDKClient`
7. Collect response and extract structured output
8. Publish `PromptExecuted` event
9. Return `PromptResponse[OutputT]`

## Tool Bridging

### MCP Server Bridge

The adapter dynamically creates an MCP server exposing weakincentives tools:

```python
def _create_mcp_bridge(
    self,
    tools: tuple[Tool[Any, Any], ...],
    context: ToolContext,
) -> MCPServerConfig:
    """Create ephemeral MCP server from weakincentives tools."""

    server = create_sdk_mcp_server()

    for tool in tools:
        handler = self._wrap_tool_handler(tool, context)
        server.register(
            name=tool.name,
            description=tool.description,
            schema=self._tool_to_input_schema(tool),
            handler=handler,
        )

    return server.config
```

### Handler Wrapping

Each weakincentives tool handler is wrapped to match SDK expectations:

```python
def _wrap_tool_handler(
    self,
    tool: Tool[ParamsT, ResultT],
    context: ToolContext,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Wrap weakincentives tool handler for SDK execution."""

    async def sdk_handler(args: dict[str, Any]) -> dict[str, Any]:
        # Parse arguments to dataclass
        params = parse(tool.params_type, args, extra="forbid")

        # Execute original handler
        result = tool.handler(params, context=context)

        # Publish ToolInvoked event
        session.bus.publish(ToolInvoked(
            tool_name=tool.name,
            params=args,
            result=result,
            success=result.success,
        ))

        # Format for SDK
        return {
            "content": [{"type": "text", "text": result.message}],
            "isError": not result.success,
        }

    return sdk_handler
```

### Input Schema Generation

```python
def _tool_to_input_schema(
    self,
    tool: Tool[ParamsT, ResultT],
) -> dict[str, Any]:
    """Convert weakincentives tool params to SDK input schema."""

    if tool.params_type is type(None):
        return {}

    return schema(tool.params_type, mode="serialization")
```

### Tool Naming Convention

SDK tools registered via MCP use the naming pattern `mcp__<server>__<tool>`.
The adapter configures `allowed_tools` accordingly:

```python
mcp_tool_names = [f"mcp__wink__{tool.name}" for tool in tools]
options = ClaudeAgentOptions(
    allowed_tools=mcp_tool_names + ["respond"],
    mcp_servers={"wink": mcp_bridge},
)
```

## Structured Output

### The `respond` Tool Pattern

Since the SDK lacks native JSON schema enforcement, structured outputs use a
dedicated tool that the model must call to provide its final answer:

```python
def _create_respond_tool(
    self,
    output_type: type[OutputT],
) -> ToolDefinition:
    """Create respond tool for structured output."""

    output_schema = schema(output_type, mode="serialization")

    return ToolDefinition(
        name="respond",
        description="Provide your final structured response. You MUST use this tool to answer.",
        input_schema=output_schema,
    )
```

### System Prompt Augmentation

When structured output is declared, the system prompt includes explicit
instructions:

```python
STRUCTURED_OUTPUT_INSTRUCTION = """
## Response Format

You MUST provide your final answer by calling the `respond` tool with a JSON
object matching the required schema. Do not provide your answer as plain text.

Required output schema:
```json
{schema}
```
"""
```

### Output Extraction

```python
def _extract_structured_output(
    self,
    messages: list[Message],
    output_type: type[OutputT],
) -> OutputT | None:
    """Extract structured output from respond tool call."""

    for message in reversed(messages):
        if not isinstance(message, AssistantMessage):
            continue
        for block in message.content:
            if isinstance(block, ToolUseBlock) and block.name == "respond":
                return parse(output_type, block.input, extra="ignore")

    return None
```

## Async/Sync Bridge

### Primary Strategy

The adapter uses `asyncio.run()` for synchronous execution:

```python
def evaluate(self, prompt, *, session, **kwargs) -> PromptResponse[OutputT]:
    return asyncio.run(self._evaluate_async(prompt, session=session, **kwargs))
```

### Nested Event Loop Handling

For environments with existing event loops (e.g., Jupyter), the adapter
detects and handles appropriately:

```python
def evaluate(self, prompt, *, session, **kwargs) -> PromptResponse[OutputT]:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop - safe to use asyncio.run()
        return asyncio.run(self._evaluate_async(...))

    # Running loop exists - use nest_asyncio or thread delegation
    if _NEST_ASYNCIO_AVAILABLE:
        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.run(self._evaluate_async(...))

    # Fallback: run in thread pool
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(asyncio.run, self._evaluate_async(...))
        return future.result()
```

## Stateful Sessions

### ClaudeSDKClient Mode

When `stateful=True`, the adapter maintains conversation context:

```python
class ClaudeAgentSDKAdapter:
    def __init__(self, *, stateful: bool = False, ...):
        self._stateful = stateful
        self._client: ClaudeSDKClient | None = None
        self._session_id: str | None = None

    async def _evaluate_async(self, prompt, *, session, **kwargs):
        if self._stateful:
            return await self._evaluate_stateful(prompt, session=session, **kwargs)
        return await self._evaluate_stateless(prompt, session=session, **kwargs)

    async def _evaluate_stateful(self, prompt, *, session, **kwargs):
        if self._client is None:
            self._client = ClaudeSDKClient(self._build_options())
            await self._client.__aenter__()

        await self._client.query(user_prompt)
        messages = []
        async for msg in self._client.receive_response():
            messages.append(msg)
            if isinstance(msg, ResultMessage):
                self._session_id = msg.session_id
        ...
```

### Session Lifecycle

```python
def reset_session(self) -> None:
    """Reset the stateful session, starting fresh."""
    if self._client is not None:
        asyncio.run(self._client.__aexit__(None, None, None))
        self._client = None
        self._session_id = None

def fork_session(self) -> str | None:
    """Fork current session for branching conversations."""
    return self._session_id
```

## Error Handling

### SDK Exception Mapping

```python
def _normalize_sdk_error(
    self,
    error: Exception,
    prompt_name: str,
) -> PromptEvaluationError:
    """Convert SDK exceptions to weakincentives error types."""

    if isinstance(error, CLINotFoundError):
        return PromptEvaluationError(
            message="Claude Code CLI not found. Install with: npm install -g @anthropic/claude-code",
            prompt_name=prompt_name,
            phase="request",
        )

    if isinstance(error, CLIConnectionError):
        return ThrottleError(
            message=str(error),
            prompt_name=prompt_name,
            phase="request",
            details=ThrottleDetails(
                kind=ThrottleKind.TIMEOUT,
                retry_after=None,
                attempts=1,
                retry_safe=True,
                provider_payload=None,
            ),
        )

    if isinstance(error, ProcessError):
        return PromptEvaluationError(
            message=f"Claude Code process failed: {error.stderr}",
            prompt_name=prompt_name,
            phase="request",
            provider_payload={"exit_code": error.exit_code, "stderr": error.stderr},
        )

    if isinstance(error, CLIJSONDecodeError):
        return PromptEvaluationError(
            message=f"Failed to parse SDK response: {error}",
            prompt_name=prompt_name,
            phase="response",
        )

    return PromptEvaluationError(
        message=str(error),
        prompt_name=prompt_name,
        phase="request",
    )
```

### Tool Execution Errors

Tool failures within the SDK are handled natively. The MCP bridge returns
error results that Claude can reason about:

```python
async def sdk_handler(args: dict[str, Any]) -> dict[str, Any]:
    try:
        params = parse(tool.params_type, args, extra="forbid")
        result = tool.handler(params, context=context)
    except ValidationError as e:
        return {
            "content": [{"type": "text", "text": f"Invalid parameters: {e}"}],
            "isError": True,
        }
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Tool error: {e}"}],
            "isError": True,
        }
    ...
```

## Budget and Deadline Integration

### Budget Tracking

The SDK's `ResultMessage` provides usage and cost information:

```python
async def _evaluate_async(self, prompt, *, session, budget_tracker, **kwargs):
    ...
    for msg in messages:
        if isinstance(msg, ResultMessage):
            if budget_tracker and msg.usage:
                budget_tracker.record(
                    input_tokens=msg.usage.input_tokens,
                    output_tokens=msg.usage.output_tokens,
                )
            usage = msg.usage
            cost = msg.total_cost
    ...
```

### Deadline Enforcement

Deadlines are checked before SDK invocation. The SDK itself does not support
mid-execution cancellation, so deadline violations during execution raise
after completion:

```python
async def _evaluate_async(self, prompt, *, deadline, **kwargs):
    if deadline and deadline.is_expired():
        raise DeadlineExceededError(
            message="Deadline expired before SDK invocation",
            prompt_name=prompt.name,
        )

    # SDK execution (no cancellation support)
    messages = await self._invoke_sdk(...)

    if deadline and deadline.is_expired():
        raise DeadlineExceededError(
            message="Deadline expired during SDK execution",
            prompt_name=prompt.name,
        )
```

## Telemetry

### Events Published

| Event | When | Payload |
|-------|------|---------|
| `PromptRendered` | After prompt render | Text, tools, metadata |
| `ToolInvoked` | Each tool execution | Name, params, result |
| `PromptExecuted` | After SDK completion | Response, tokens, timing |

### Logging

```python
# SDK invocation
logger.info(
    "claude_agent_sdk.invoke",
    extra={
        "event": "sdk.invoke.start",
        "prompt_name": prompt.name,
        "model": self._model,
        "stateful": self._stateful,
        "tool_count": len(tools),
    },
)

# SDK completion
logger.info(
    "claude_agent_sdk.complete",
    extra={
        "event": "sdk.invoke.complete",
        "prompt_name": prompt.name,
        "duration_ms": result.duration_ms,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_cost": cost,
        "session_id": session_id,
    },
)
```

## File Structure

```
src/weakincentives/adapters/
├── claude_agent_sdk/
│   ├── __init__.py           # Public exports
│   ├── adapter.py            # ClaudeAgentSDKAdapter
│   ├── config.py             # Configuration dataclasses
│   ├── _bridge.py            # MCP tool bridge
│   ├── _respond_tool.py      # Structured output tool
│   ├── _async_utils.py       # Async/sync bridging
│   └── _errors.py            # Error normalization
```

## Usage Examples

### Basic Evaluation

```python
from weakincentives import Prompt, MarkdownSection
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
)

adapter = ClaudeAgentSDKAdapter(
    model="claude-sonnet-4-20250514",
    client_config=ClaudeAgentSDKClientConfig(
        permission_mode="bypassPermissions",
    ),
)

prompt = Prompt[Answer](
    ns="demo",
    key="basic",
    sections=[
        MarkdownSection(
            title="Task",
            key="task",
            template="Answer: $question",
        ),
    ],
)

response = adapter.evaluate(prompt, session=session)
print(response.output)
```

### With Tools

```python
from weakincentives import Tool, ToolResult, ToolContext

@dataclass(slots=True, frozen=True)
class SearchParams:
    query: str

@dataclass(slots=True, frozen=True)
class SearchResult:
    title: str
    url: str

def search_handler(
    params: SearchParams,
    *,
    context: ToolContext,
) -> ToolResult[SearchResult]:
    # Implementation
    return ToolResult(
        message=f"Found result for: {params.query}",
        value=SearchResult(title="Example", url="https://example.com"),
    )

search_tool = Tool[SearchParams, SearchResult](
    name="search",
    description="Search for information",
    handler=search_handler,
)

prompt = Prompt[Answer](
    ns="demo",
    key="with-tools",
    sections=[
        MarkdownSection(
            title="Task",
            key="task",
            template="Research: $topic",
            tools=(search_tool,),
        ),
    ],
)

# Tools are bridged to SDK via MCP and executed natively
response = adapter.evaluate(prompt, session=session)
```

### Stateful Conversations

```python
adapter = ClaudeAgentSDKAdapter(
    model="claude-sonnet-4-20250514",
    stateful=True,
)

# First turn
response1 = adapter.evaluate(prompt1, session=session)

# Second turn - Claude remembers context
response2 = adapter.evaluate(prompt2, session=session)

# Reset when needed
adapter.reset_session()
```

### Workspace Access

```python
config = ClaudeAgentSDKClientConfig(
    cwd="/path/to/project",
    add_dirs=("/path/to/libs", "/path/to/data"),
    permission_mode="acceptEdits",
)

adapter = ClaudeAgentSDKAdapter(
    model="claude-sonnet-4-20250514",
    client_config=config,
)

# Claude can read/write files in configured directories
response = adapter.evaluate(code_review_prompt, session=session)
```

## Limitations

- **CLI dependency**: Requires Claude Code CLI installed on the system
- **No streaming in evaluate()**: Results collected after completion; streaming
  requires direct SDK usage
- **Deadline granularity**: Cannot cancel mid-execution; deadlines checked at
  boundaries
- **Structured output reliability**: Tool-based pattern less strict than JSON
  schema enforcement
- **Platform support**: SDK availability may vary by platform

## Testing

### Unit Tests

- Mock SDK responses for message type coverage
- Test MCP bridge tool registration
- Verify structured output extraction
- Test error normalization for all exception types
- Validate async/sync bridge behavior

### Integration Tests

- Require Claude Code CLI installation
- Test full tool round-trip via MCP bridge
- Verify stateful session continuity
- Test budget tracking accuracy
- Validate event publication

### Fixtures

- `tests/fixtures/claude_agent_sdk/` contains sample message sequences
- `tests/helpers/claude_agent_sdk.py` provides mock client

## Dependencies

```toml
[project.optional-dependencies]
claude-agent-sdk = [
    "claude-agent-sdk>=0.1.0",
]
```

Optional for async loop handling:

```toml
[project.optional-dependencies]
claude-agent-sdk-jupyter = [
    "claude-agent-sdk>=0.1.0",
    "nest-asyncio>=1.5.0",
]
```

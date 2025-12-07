# LangGraph Adapter Specification

## Purpose

This specification describes how to integrate LangGraph into the WINK adapter
ecosystem. LangGraph is a library for building stateful, multi-actor LLM
applications modeled as graphs. The adapter bridges WINK's prompt abstraction
with LangGraph's graph-based execution model, enabling users to leverage
LangGraph's built-in agent patterns (ReAct, supervisor, swarm) while
maintaining compatibility with WINK's session management, event bus, and
tooling infrastructure.

**Target Version**: LangGraph 0.2.x+ (pre-1.0, with awareness that 1.0 releases
October 2025)

## Guiding Principles

- **Graph-first execution**: Preserve LangGraph's native graph execution model
  rather than forcing a request/response pattern.
- **Minimal wrapping**: Delegate to LangGraph's prebuilt components
  (`create_react_agent`, `ToolNode`) where possible.
- **Session integration**: Map LangGraph checkpointing to WINK session snapshots
  when appropriate.
- **Tool compatibility**: Translate WINK tools to LangGraph's tool format
  without losing handler semantics.

## LangGraph Overview

LangGraph models agent workflows using three primitives:

| Primitive | Description |
|-----------|-------------|
| **State** | Shared data structure (TypedDict or Pydantic) representing the application snapshot |
| **Node** | Function that receives state, performs computation, returns updated state |
| **Edge** | Function or constant determining which node executes next |

Graphs must be compiled via `.compile()` before execution. Compilation
validates structure and enables runtime features like checkpointing.

### Key API Surface

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState, add_messages
from langgraph.prebuilt import create_react_agent, ToolNode

# StateGraph is the primary abstraction
graph = StateGraph(MyState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
graph.add_edge("tools", "agent")

compiled = graph.compile()
result = compiled.invoke({"messages": [...]})
```

### MessagesState

LangGraph provides `MessagesState` for chat-based workflows:

```python
from langgraph.graph.message import MessagesState

class AgentState(MessagesState):
    # Inherits `messages: Annotated[list[AnyMessage], add_messages]`
    custom_field: str
```

The `add_messages` reducer appends new messages rather than replacing.

### Prebuilt Agents

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model="anthropic:claude-3-5-sonnet",
    tools=[my_tool],
    prompt="System instructions here",
)

# invoke() for single response
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})

# stream() for incremental output
for event in agent.stream({"messages": [...]}):
    print(event)
```

## Adapter Design

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      LangGraphAdapter                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐   ┌──────────────┐   ┌───────────────────┐    │
│  │   Prompt    │──▶│  Tool        │──▶│  StateGraph or    │    │
│  │   Render    │   │  Translation │   │  create_react_    │    │
│  └─────────────┘   └──────────────┘   │  agent            │    │
│                                        └───────────────────┘    │
│                                                 │               │
│                                                 ▼               │
│                                        ┌───────────────────┐    │
│                                        │  Graph Execution  │    │
│                                        │  (invoke/stream)  │    │
│                                        └───────────────────┘    │
│                                                 │               │
│                                                 ▼               │
│  ┌─────────────┐   ┌──────────────┐   ┌───────────────────┐    │
│  │  Response   │◀──│  Output      │◀──│  Event Bus        │    │
│  │  Assembly   │   │  Parsing     │   │  Integration      │    │
│  └─────────────┘   └──────────────┘   └───────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Configuration

```python
@FrozenDataclass()
class LangGraphClientConfig:
    """Configuration for LangGraph execution.

    Attributes:
        recursion_limit: Maximum graph steps before termination. Defaults to 25.
        interrupt_before: Node names to pause before (for human-in-the-loop).
        interrupt_after: Node names to pause after.
    """

    recursion_limit: int = 25
    interrupt_before: tuple[str, ...] | None = None
    interrupt_after: tuple[str, ...] | None = None

    def to_invoke_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"recursion_limit": self.recursion_limit}
        if self.interrupt_before:
            kwargs["interrupt_before"] = list(self.interrupt_before)
        if self.interrupt_after:
            kwargs["interrupt_after"] = list(self.interrupt_after)
        return kwargs


@FrozenDataclass()
class LangGraphModelConfig(LLMConfig):
    """LangGraph-specific model configuration.

    Extends LLMConfig with parameters for model binding.

    Attributes:
        model_provider: Provider prefix (e.g., "anthropic", "openai").
            When set, model identifier becomes "{provider}:{model}".
        parallel_tool_calls: Whether to allow parallel tool execution.
    """

    model_provider: str | None = None
    parallel_tool_calls: bool = True
```

### Adapter Protocol Implementation

```python
class LangGraphAdapter(ProviderAdapter[Any]):
    """Adapter that evaluates prompts via LangGraph's graph execution.

    This adapter supports two modes:

    1. **Prebuilt mode**: Uses `create_react_agent` for standard ReAct patterns.
       Suitable for most use cases where WINK prompts map cleanly to agent tasks.

    2. **Custom graph mode**: Accepts a pre-compiled StateGraph for advanced
       workflows requiring custom control flow.

    Args:
        model: Model identifier. With model_provider set, becomes
            "{provider}:{model}" format expected by LangGraph.
        client_config: Graph execution configuration.
        model_config: Model binding parameters.
        tool_choice: Tool selection mode. LangGraph always uses "auto" internally
            but this affects how tools are bound.
        graph: Pre-compiled StateGraph. When provided, the adapter delegates
            directly to this graph instead of using create_react_agent.
        checkpointer: Optional LangGraph checkpointer for state persistence.
    """

    def __init__(
        self,
        *,
        model: str,
        client_config: LangGraphClientConfig | None = None,
        model_config: LangGraphModelConfig | None = None,
        tool_choice: ToolChoice = "auto",
        graph: CompiledStateGraph | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        ...

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
        ...
```

## Tool Translation

WINK tools must be translated to LangGraph's expected format.

### From WINK Tool to LangGraph Tool

```python
def wink_tool_to_langgraph(
    tool: Tool[SupportsDataclassOrNone, SupportsToolResult],
    *,
    session: SessionProtocol,
    prompt: Prompt[Any],
    adapter: ProviderAdapter[Any],
    deadline: Deadline | None,
    budget_tracker: BudgetTracker | None,
) -> Callable[..., str]:
    """Convert a WINK tool to a LangGraph-compatible callable.

    LangGraph tools are functions that:
    - Accept keyword arguments matching the tool's parameter schema
    - Return a string (or object that will be stringified)
    - Can be decorated with @tool for metadata

    The wrapper preserves WINK's handler semantics:
    - ToolContext is constructed and passed to the handler
    - ToolResult is serialized to string for LangGraph
    - Events are published to the session's event bus
    """

    @tool(name=tool.name, description=tool.description)
    def wrapper(**kwargs: Any) -> str:
        # Parse arguments to dataclass
        if tool.params_type is type(None):
            params = None
        else:
            params = parse(tool.params_type, kwargs, extra="forbid")

        # Build context
        context = ToolContext(
            prompt=prompt,
            rendered_prompt=None,  # Not available in LangGraph flow
            adapter=adapter,
            session=session,
            deadline=deadline,
            budget_tracker=budget_tracker,
        )

        # Execute handler
        result = tool.handler(params, context=context)

        # Publish event
        session.event_bus.publish(
            ToolInvoked(
                prompt_name=prompt.name,
                adapter="langgraph",
                name=tool.name,
                params=params,
                result=result,
                ...
            )
        )

        # Serialize for LangGraph
        return serialize_tool_message(result)

    return wrapper
```

### Tool Binding

Tools are bound to the LLM using LangGraph's `bind_tools`:

```python
from langchain_core.language_models import BaseChatModel

def bind_wink_tools(
    llm: BaseChatModel,
    tools: Sequence[Tool[SupportsDataclassOrNone, SupportsToolResult]],
    *,
    context: ToolExecutionContext,
) -> BaseChatModel:
    """Bind WINK tools to a LangChain chat model for LangGraph execution."""

    langgraph_tools = [
        wink_tool_to_langgraph(tool, **context.as_dict())
        for tool in tools
    ]
    return llm.bind_tools(langgraph_tools)
```

## Structured Output

LangGraph supports structured output via `with_structured_output`:

```python
from langchain_core.pydantic_v1 import BaseModel

class ResponseSchema(BaseModel):
    answer: str
    confidence: float

llm_structured = llm.with_structured_output(ResponseSchema)
```

### Integration with WINK Prompts

When a prompt declares structured output (`output_type` and `container`):

1. **Dataclass to Pydantic**: Convert the WINK dataclass schema to a Pydantic
   model for LangGraph's `with_structured_output`.

2. **Response extraction**: Parse the structured response from LangGraph's
   output and convert back to the WINK dataclass.

```python
def build_pydantic_schema(
    dataclass_type: type[SupportsDataclass],
) -> type[BaseModel]:
    """Convert a WINK dataclass to a Pydantic model for LangGraph."""
    from pydantic import create_model
    from weakincentives.serde import schema

    json_schema = schema(dataclass_type)
    # Use Pydantic's schema-to-model conversion
    ...


def parse_langgraph_output(
    output: BaseModel | dict[str, Any],
    target_type: type[OutputT],
) -> OutputT:
    """Convert LangGraph structured output to WINK dataclass."""
    if isinstance(output, dict):
        return parse(target_type, output)
    return parse(target_type, output.model_dump())
```

## Execution Flow

### Prebuilt Agent Mode

```python
def _execute_prebuilt(
    self,
    rendered: RenderedPrompt[OutputT],
    *,
    session: SessionProtocol,
    deadline: Deadline | None,
) -> PromptResponse[OutputT]:
    """Execute using create_react_agent."""

    # Build model identifier
    model_id = self._model
    if self._model_config and self._model_config.model_provider:
        model_id = f"{self._model_config.model_provider}:{self._model}"

    # Translate tools
    langgraph_tools = [
        wink_tool_to_langgraph(tool, session=session, ...)
        for tool in rendered.tools
    ]

    # Create agent
    agent = create_react_agent(
        model=model_id,
        tools=langgraph_tools,
        prompt=rendered.text,  # System prompt
    )

    # Prepare input
    input_messages = [{"role": "user", "content": rendered.text}]

    # Execute with deadline awareness
    invoke_kwargs = self._client_config.to_invoke_kwargs() if self._client_config else {}
    if deadline:
        # LangGraph doesn't have native deadline support
        # We rely on recursion_limit and check between iterations
        ...

    try:
        result = agent.invoke(
            {"messages": input_messages},
            **invoke_kwargs,
        )
    except Exception as error:
        raise self._normalize_error(error)

    # Extract response
    return self._parse_result(result, rendered)
```

### Custom Graph Mode

```python
def _execute_custom_graph(
    self,
    rendered: RenderedPrompt[OutputT],
    *,
    session: SessionProtocol,
) -> PromptResponse[OutputT]:
    """Execute using a pre-compiled StateGraph."""

    # Inject WINK tools into graph state if needed
    input_state = {
        "messages": [{"role": "user", "content": rendered.text}],
    }

    result = self._graph.invoke(
        input_state,
        config={"configurable": {"thread_id": session.session_id}},
    )

    return self._parse_result(result, rendered)
```

## Error Handling

### LangGraph Exceptions

LangGraph raises several exception types that must be normalized:

| LangGraph Exception | WINK Mapping |
|---------------------|--------------|
| `GraphRecursionError` | `PromptEvaluationError` with phase="request" |
| `InvalidUpdateError` | `PromptEvaluationError` with phase="tool" |
| `NodeInterrupt` | Not an error; indicates human-in-the-loop pause |
| Underlying LLM errors | Delegate to provider-specific normalization |

```python
def _normalize_langgraph_error(
    error: Exception,
    *,
    prompt_name: str,
) -> PromptEvaluationError:
    """Convert LangGraph exceptions to WINK error types."""

    error_name = error.__class__.__name__

    if "recursion" in error_name.lower():
        return PromptEvaluationError(
            f"Graph execution exceeded recursion limit: {error}",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
        )

    if "invalid" in error_name.lower() and "update" in error_name.lower():
        return PromptEvaluationError(
            f"Invalid state update during graph execution: {error}",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_TOOL,
        )

    # Check for wrapped LLM errors (rate limits, etc.)
    cause = error.__cause__
    if cause is not None:
        throttle = _detect_throttle_error(cause)
        if throttle:
            return throttle

    return PromptEvaluationError(
        str(error) or "LangGraph execution failed.",
        prompt_name=prompt_name,
        phase=PROMPT_EVALUATION_PHASE_REQUEST,
    )
```

### Throttle Detection

LangGraph wraps underlying LLM client errors. Throttle detection must unwrap:

```python
def _detect_throttle_error(error: Exception) -> ThrottleError | None:
    """Detect rate limit errors from wrapped LLM clients."""

    message = str(error).lower()
    class_name = error.__class__.__name__.lower()

    # Common patterns across providers
    if any(term in message for term in ["rate limit", "ratelimit", "429"]):
        return ThrottleError(
            str(error),
            prompt_name="",  # Filled by caller
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            details=throttle_details(kind="rate_limit"),
        )

    if "quota" in message or "insufficient" in message:
        return ThrottleError(
            str(error),
            prompt_name="",
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            details=throttle_details(kind="quota_exhausted"),
        )

    if "timeout" in class_name:
        return ThrottleError(
            str(error),
            prompt_name="",
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            details=throttle_details(kind="timeout"),
        )

    return None
```

## Event Bus Integration

The adapter publishes standard WINK events:

### PromptRendered

Published before graph execution begins:

```python
session.event_bus.publish(
    PromptRendered(
        prompt_ns=prompt.ns,
        prompt_key=prompt.key,
        prompt_name=prompt.name,
        adapter=LANGGRAPH_ADAPTER_NAME,
        session_id=session.session_id,
        render_inputs=prompt.params,
        rendered_prompt=rendered.text,
        ...
    )
)
```

### ToolInvoked

Published by each tool wrapper during graph execution (see Tool Translation).

### PromptExecuted

Published after graph execution completes:

```python
session.event_bus.publish(
    PromptExecuted(
        prompt_name=prompt.name,
        adapter=LANGGRAPH_ADAPTER_NAME,
        result=response,
        session_id=session.session_id,
        usage=self._extract_usage(result),  # If available
        ...
    )
)
```

## Budget Tracking

LangGraph doesn't expose token usage directly. Budget tracking requires:

1. **Callback instrumentation**: Use LangChain callbacks to capture token counts
   from the underlying LLM.

2. **Cumulative tracking**: Record usage after each LLM call within the graph.

```python
from langchain_core.callbacks import BaseCallbackHandler

class BudgetTrackingCallback(BaseCallbackHandler):
    """Callback that tracks token usage for budget enforcement."""

    def __init__(
        self,
        tracker: BudgetTracker,
        evaluation_id: str,
    ) -> None:
        self.tracker = tracker
        self.evaluation_id = evaluation_id

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        usage = response.llm_output.get("token_usage") if response.llm_output else None
        if usage:
            self.tracker.record_cumulative(
                self.evaluation_id,
                TokenUsage(
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                ),
            )
            self.tracker.check()  # Raises BudgetExceededError
```

## Checkpointing and Session State

LangGraph's checkpointing can optionally synchronize with WINK sessions:

```python
from langgraph.checkpoint.base import BaseCheckpointSaver

class SessionCheckpointer(BaseCheckpointSaver):
    """Checkpointer that persists to WINK session state."""

    def __init__(self, session: SessionProtocol) -> None:
        self.session = session

    def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> None:
        # Store checkpoint in session
        self.session.mutate(LangGraphCheckpoint).dispatch(
            StoreCheckpoint(checkpoint=checkpoint)
        )

    def get(self, config: RunnableConfig) -> Checkpoint | None:
        # Retrieve from session
        return self.session.query(LangGraphCheckpoint).latest()
```

**Note**: This is optional. Most use cases should use LangGraph's built-in
checkpointers (memory, SQLite, PostgreSQL) rather than coupling to WINK sessions.

## Module Structure

```
src/weakincentives/adapters/
├── langgraph.py          # Main adapter implementation
├── _langgraph_tools.py   # Tool translation utilities
├── _langgraph_output.py  # Structured output handling
└── config.py             # Add LangGraphClientConfig, LangGraphModelConfig
```

## Dependencies

The adapter requires `langgraph` as an optional dependency:

```toml
# pyproject.toml
[project.optional-dependencies]
langgraph = ["langgraph>=0.2.0,<1.0"]
```

Import guard pattern:

```python
_ERROR_MESSAGE: Final[str] = (
    "LangGraph support requires the optional 'langgraph' dependency. "
    "Install it with `uv sync --extra langgraph` or `pip install weakincentives[langgraph]`."
)

def _load_langgraph() -> ModuleType:
    try:
        return import_module("langgraph")
    except ModuleNotFoundError as exc:
        raise RuntimeError(_ERROR_MESSAGE) from exc
```

## Usage Examples

### Basic ReAct Agent

```python
from weakincentives.adapters.langgraph import (
    LangGraphAdapter,
    LangGraphClientConfig,
    LangGraphModelConfig,
)

adapter = LangGraphAdapter(
    model="claude-3-5-sonnet",
    model_config=LangGraphModelConfig(
        model_provider="anthropic",
        temperature=0.7,
    ),
    client_config=LangGraphClientConfig(
        recursion_limit=50,
    ),
)

response = adapter.evaluate(
    prompt,
    session=session,
    deadline=deadline,
)
```

### Custom Graph

```python
from langgraph.graph import StateGraph, START, END

# Define custom graph
graph = StateGraph(MyState)
graph.add_node("research", research_node)
graph.add_node("synthesize", synthesize_node)
graph.add_edge(START, "research")
graph.add_edge("research", "synthesize")
graph.add_edge("synthesize", END)

compiled = graph.compile()

adapter = LangGraphAdapter(
    model="gpt-4o",
    graph=compiled,
)
```

## Testing

### Unit Tests

- Mock LangGraph's `create_react_agent` and `StateGraph.invoke`
- Verify tool translation preserves handler semantics
- Test structured output parsing for valid and malformed payloads
- Confirm event publication at each stage

### Integration Tests

```python
@pytest.mark.integration
def test_langgraph_react_agent(openai_api_key: str) -> None:
    """Verify end-to-end ReAct execution."""

    adapter = LangGraphAdapter(
        model="gpt-4o-mini",
        model_config=LangGraphModelConfig(model_provider="openai"),
    )

    prompt = Prompt(
        ns="test",
        key="langgraph",
        name="test_prompt",
        sections=[
            MarkdownSection(
                title="Task",
                template="Calculate 2 + 2 using the calculator tool.",
                key="task",
                tools=(calculator_tool,),
            ),
        ],
    )

    bus = InProcessEventBus()
    session = Session(bus=bus)

    response = adapter.evaluate(prompt, session=session)

    assert response.text is not None
    assert "4" in response.text
```

### Fixtures

Add to `tests/helpers/adapters.py`:

```python
class MockLangGraphAgent:
    """Mock for LangGraph's create_react_agent result."""

    def __init__(self, responses: Sequence[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.call_count = 0

    def invoke(self, input: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response
```

## Limitations and Caveats

1. **Streaming**: The initial implementation uses `invoke()`. Streaming support
   via `stream()` is deferred to a future iteration.

2. **Human-in-the-loop**: `interrupt_before`/`interrupt_after` are exposed in
   config but require additional orchestration to handle pauses.

3. **Token tracking**: Depends on LangChain callback instrumentation; accuracy
   varies by underlying provider.

4. **Deadline enforcement**: LangGraph lacks native deadline support. The
   adapter checks remaining time between graph steps but cannot interrupt
   mid-node execution.

5. **State isolation**: Each `evaluate()` call creates a fresh graph state.
   Cross-call state persistence requires explicit checkpointer configuration.

## Future Work

- Streaming support via `stream()` and `astream()`
- Async adapter variant (`AsyncLangGraphAdapter`)
- Supervisor and swarm patterns
- LangGraph Cloud integration for remote execution
- MCP adapter integration when LangGraph 1.0 stabilizes

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [How to force tool-calling agent to structure output](https://langchain-ai.github.io/langgraph/how-tos/react-agent-structured-output/)
- [ToolNode Reference](https://langchain-ai.github.io/langgraphjs/reference/classes/langgraph_prebuilt.ToolNode.html)

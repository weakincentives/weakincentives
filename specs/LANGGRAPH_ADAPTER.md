# LangGraph Adapter Specification

## Purpose

This specification describes how to integrate LangGraph into the WINK adapter
ecosystem. LangGraph is a library for building stateful, multi-actor LLM
applications modeled as graphs. The adapter bridges WINK's prompt abstraction
with LangGraph's graph-based execution model, enabling users to leverage
LangGraph's agent patterns while maintaining compatibility with WINK's session
management, event bus, and tooling infrastructure.

**Target Version**: LangGraph 1.0.x (requires Python 3.10+)

**Release Context**: LangGraph 1.0 was released October 2025 as the first stable
major release in the durable agent framework space. It shipped with zero breaking
changes from 0.2.x and is production-tested at companies like Uber, LinkedIn,
and Klarna.

## Guiding Principles

- **Graph-first execution**: Preserve LangGraph's native graph execution model
  rather than forcing a request/response pattern.
- **Minimal wrapping**: Delegate to LangGraph's `create_agent` and `ToolNode`
  where possible.
- **Durable state**: Leverage LangGraph 1.0's built-in persistence and
  checkpoint recovery.
- **Session integration**: Map LangGraph checkpointing to WINK session snapshots
  when appropriate.
- **Tool compatibility**: Translate WINK tools to LangGraph's tool format
  without losing handler semantics.

## LangGraph 1.0 Overview

LangGraph models agent workflows using three primitives:

| Primitive | Description |
|-----------|-------------|
| **State** | Shared data structure (**TypedDict only** in 1.0) representing the application snapshot |
| **Node** | Function that receives state, performs computation, returns updated state |
| **Edge** | Function or constant determining which node executes next |

Graphs must be compiled via `.compile()` before execution. Compilation
validates structure and enables runtime features like checkpointing.

### Key 1.0 Features

- **Durable State**: Agent execution state persists automatically. Server
  restarts or workflow interruptions resume exactly where they left off.
- **Built-in Persistence**: Save and resume workflows at any point without
  custom database logic. Enables multi-day approval processes and background
  jobs.
- **Human-in-the-Loop**: First-class API support for pausing execution for
  human review, modification, or approval.
- **Middleware System**: Customization hooks for human approval, summarization,
  PII redaction, and other cross-cutting concerns.

### Key API Surface

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState, add_messages
from langgraph.prebuilt import ToolNode
from langchain.agents import create_agent  # New in 1.0

# StateGraph remains the primary abstraction
graph = StateGraph(MyState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
graph.add_edge("tools", "agent")

compiled = graph.compile()
result = compiled.invoke({"messages": [...]})
```

### State Schema Requirements

**Important**: As of LangGraph 1.0, custom state schemas **must** be `TypedDict`
types. Pydantic models and dataclasses are no longer supported for state
definitions.

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

# Correct: TypedDict
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    custom_field: str

# Incorrect: Pydantic/dataclass no longer supported for state
# class AgentState(BaseModel): ...  # Will fail
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

### Agent Creation (1.0 Pattern)

The `create_react_agent` function from `langgraph.prebuilt` is **deprecated**.
Use `create_agent` from `langchain.agents`:

```python
from langchain.agents import create_agent

agent = create_agent(
    model,
    tools=[my_tool],
    system_prompt="System instructions here",
)

# invoke() for single response
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})

# stream() for incremental output
for event in agent.stream({"messages": [...]}):
    print(event)
```

### Deprecated Components

| Deprecated (langgraph.prebuilt) | Replacement (langchain.agents) |
|---------------------------------|--------------------------------|
| `create_react_agent` | `create_agent` |
| `AgentState` | `AgentState` |
| `AgentStatePydantic` | Removed (use TypedDict) |
| `AgentStateWithStructuredResponse` | `AgentState` |
| `HumanInterruptConfig` | `middleware.human_in_the_loop.InterruptOnConfig` |
| `ValidationNode` | Auto-validation via `create_agent` |
| `MessageGraph` | `StateGraph` with messages key |

## Adapter Design

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      LangGraphAdapter                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐   ┌──────────────┐   ┌───────────────────┐    │
│  │   Prompt    │──▶│  Tool        │──▶│  StateGraph or    │    │
│  │   Render    │   │  Translation │   │  create_agent     │    │
│  └─────────────┘   └──────────────┘   └───────────────────┘    │
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
        thread_id: Optional thread identifier for checkpoint persistence.
    """

    recursion_limit: int = 25
    interrupt_before: tuple[str, ...] | None = None
    interrupt_after: tuple[str, ...] | None = None
    thread_id: str | None = None

    def to_invoke_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"recursion_limit": self.recursion_limit}
        if self.interrupt_before:
            kwargs["interrupt_before"] = list(self.interrupt_before)
        if self.interrupt_after:
            kwargs["interrupt_after"] = list(self.interrupt_after)
        return kwargs

    def to_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {}
        if self.thread_id:
            config["configurable"] = {"thread_id": self.thread_id}
        return config


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

    1. **Agent mode**: Uses `create_agent` for standard agent patterns.
       Suitable for most use cases where WINK prompts map cleanly to agent tasks.

    2. **Custom graph mode**: Accepts a pre-compiled StateGraph for advanced
       workflows requiring custom control flow.

    Args:
        model: Model identifier. With model_provider set, becomes
            "{provider}:{model}" format expected by LangGraph.
        client_config: Graph execution configuration.
        model_config: Model binding parameters.
        tool_choice: Tool selection mode. LangGraph uses "auto" internally
            but this affects how tools are bound.
        graph: Pre-compiled StateGraph. When provided, the adapter delegates
            directly to this graph instead of using create_agent.
        checkpointer: Optional LangGraph checkpointer for state persistence.
            LangGraph 1.0 includes built-in checkpointers for memory, SQLite,
            and PostgreSQL.
        middleware: Optional list of middleware for the agent. Supports
            human-in-the-loop, summarization, and custom hooks.
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
        middleware: Sequence[Middleware] | None = None,
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
from pydantic import BaseModel

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

### Agent Mode (create_agent)

```python
def _execute_agent(
    self,
    rendered: RenderedPrompt[OutputT],
    *,
    session: SessionProtocol,
    deadline: Deadline | None,
) -> PromptResponse[OutputT]:
    """Execute using create_agent from langchain.agents."""

    from langchain.agents import create_agent

    # Build model identifier
    model_id = self._model
    if self._model_config and self._model_config.model_provider:
        model_id = f"{self._model_config.model_provider}:{self._model}"

    # Translate tools
    langgraph_tools = [
        wink_tool_to_langgraph(tool, session=session, ...)
        for tool in rendered.tools
    ]

    # Create agent with middleware if configured
    agent_kwargs: dict[str, Any] = {
        "model": model_id,
        "tools": langgraph_tools,
        "system_prompt": rendered.text,
    }
    if self._middleware:
        agent_kwargs["middleware"] = list(self._middleware)

    agent = create_agent(**agent_kwargs)

    # Prepare input
    input_messages = {"messages": [{"role": "user", "content": rendered.text}]}

    # Build invoke config
    invoke_kwargs = self._client_config.to_invoke_kwargs() if self._client_config else {}
    config = self._client_config.to_config() if self._client_config else {}

    # Add checkpointer if configured
    if self._checkpointer:
        config["checkpointer"] = self._checkpointer

    try:
        result = agent.invoke(input_messages, config=config, **invoke_kwargs)
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

    config = self._client_config.to_config() if self._client_config else {}
    if not config.get("configurable", {}).get("thread_id"):
        config.setdefault("configurable", {})["thread_id"] = session.session_id

    result = self._graph.invoke(input_state, config=config)

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

LangGraph 1.0's built-in persistence simplifies checkpoint management:

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver

# In-memory (development)
checkpointer = MemorySaver()

# SQLite (single-node production)
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# PostgreSQL (distributed production)
checkpointer = PostgresSaver.from_conn_string(DATABASE_URL)

adapter = LangGraphAdapter(
    model="gpt-4o",
    checkpointer=checkpointer,
    client_config=LangGraphClientConfig(thread_id="user-123-session-456"),
)
```

### Optional WINK Session Integration

For workflows requiring WINK session-backed persistence:

```python
from langgraph.checkpoint.base import BaseCheckpointSaver

class SessionCheckpointer(BaseCheckpointSaver):
    """Checkpointer that persists to WINK session state."""

    def __init__(self, session: SessionProtocol) -> None:
        self.session = session

    def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> None:
        self.session.mutate(LangGraphCheckpoint).dispatch(
            StoreCheckpoint(checkpoint=checkpoint)
        )

    def get(self, config: RunnableConfig) -> Checkpoint | None:
        return self.session.query(LangGraphCheckpoint).latest()
```

**Recommendation**: Use LangGraph's built-in checkpointers for most use cases.
WINK session integration is only needed when checkpoint data must live alongside
other session state.

## Human-in-the-Loop

LangGraph 1.0 provides first-class human-in-the-loop support:

```python
from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig

adapter = LangGraphAdapter(
    model="gpt-4o",
    middleware=[
        InterruptOnConfig(
            interrupt_before=["dangerous_action"],
            interrupt_after=["review_step"],
        ),
    ],
    checkpointer=MemorySaver(),  # Required for resume
)

# First invocation pauses at interrupt point
result = adapter.evaluate(prompt, session=session)
if result.interrupted:
    # Human reviews and approves
    result = adapter.resume(session=session, approval=True)
```

## LangSmith Integration

LangSmith provides observability, prompt management, and evaluation capabilities
that integrate naturally with both LangGraph and WINK. This section describes
how to implement a LangSmith-compatible prompt overrides store and enable
end-to-end tracing.

### Environment Configuration

```python
import os

# Enable LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your-api-key"
os.environ["LANGSMITH_PROJECT"] = "my-wink-project"

# Optional: Custom endpoint
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
```

When these environment variables are set, all LangGraph executions automatically
send traces to LangSmith without additional configuration.

### LangSmith Prompt Hub Overrides Store

The `LangSmithPromptOverridesStore` implements WINK's `PromptOverridesStore`
protocol using LangSmith's Prompt Hub as the backing store. This enables:

- **Centralized prompt management**: Edit prompts in LangSmith's Playground
- **Version control**: Each push creates a commit hash for rollback
- **Team collaboration**: Share prompts across organization members
- **A/B testing**: Use tags to manage production vs experimental variants

```python
from langsmith import Client
from weakincentives.prompt.overrides import (
    PromptOverridesStore,
    PromptOverride,
    SectionOverride,
    ToolOverride,
)
from weakincentives.prompt.rendering import PromptDescriptor

@FrozenDataclass()
class LangSmithStoreConfig:
    """Configuration for LangSmith prompt store.

    Attributes:
        api_key: LangSmith API key. Falls back to LANGSMITH_API_KEY env var.
        api_url: LangSmith API endpoint. Falls back to LANGSMITH_ENDPOINT.
        organization: Organization name for prompt namespacing.
        default_tag: Default version tag. Use "latest" for development,
            commit hashes for production stability.
    """

    api_key: str | None = None
    api_url: str | None = None
    organization: str | None = None
    default_tag: str = "latest"


class LangSmithPromptOverridesStore(PromptOverridesStore):
    """Prompt overrides store backed by LangSmith Prompt Hub.

    This store maps WINK's prompt override model to LangSmith's prompt
    versioning system:

    - WINK namespace/key → LangSmith repo name: "{org}/{ns}--{prompt_key}"
    - WINK tag → LangSmith tag or commit hash
    - WINK section overrides → LangSmith prompt template variables
    - WINK content hashes → Stored in prompt metadata for validation

    Example LangSmith prompt structure:
        Repo: "myorg/webapp-agents--welcome"
        Template: "{system_section}\n\n{task_section}"
        Metadata: {"wink_hashes": {"system": "abc123...", "task": "def456..."}}
    """

    def __init__(self, config: LangSmithStoreConfig | None = None) -> None:
        self._config = config or LangSmithStoreConfig()
        self._client = Client(
            api_key=self._config.api_key,
            api_url=self._config.api_url,
        )

    def _repo_name(self, ns: str, prompt_key: str) -> str:
        """Build LangSmith repo name from WINK identifiers."""
        prefix = f"{self._config.organization}/" if self._config.organization else ""
        # Replace path separators with double-dash for flat namespace
        safe_ns = ns.replace("/", "--")
        return f"{prefix}{safe_ns}--{prompt_key}"

    def resolve(
        self,
        descriptor: PromptDescriptor,
        tag: str = "latest",
    ) -> PromptOverride | None:
        """Pull prompt override from LangSmith Hub.

        Args:
            descriptor: WINK prompt descriptor with section/tool hashes.
            tag: Version tag or commit hash. Use "latest" for most recent,
                "prod" for production, or a commit hash for pinned versions.

        Returns:
            PromptOverride if found and hashes match, None otherwise.
        """
        repo_name = self._repo_name(descriptor.ns, descriptor.key)
        prompt_id = f"{repo_name}:{tag}" if tag != "latest" else repo_name

        try:
            prompt = self._client.pull_prompt(prompt_id)
        except Exception:
            return None  # Prompt not found in Hub

        # Extract metadata for hash validation
        metadata = getattr(prompt, "metadata", {}) or {}
        wink_hashes = metadata.get("wink_hashes", {})

        # Build section overrides from prompt template variables
        sections: dict[tuple[str, ...], SectionOverride] = {}
        input_variables = getattr(prompt, "input_variables", []) or []

        for section_desc in descriptor.sections:
            section_key = section_desc.path[-1] if section_desc.path else ""
            var_name = f"{section_key}_section"

            if var_name in input_variables:
                stored_hash = wink_hashes.get(section_key)
                if stored_hash and stored_hash != section_desc.content_hash:
                    # Hash mismatch - section template changed, skip override
                    continue

                # Extract section content from prompt template
                template_content = self._extract_section(prompt, var_name)
                if template_content:
                    sections[section_desc.path] = SectionOverride(
                        expected_hash=section_desc.content_hash,
                        body=template_content,
                    )

        # Build tool overrides from metadata
        tool_overrides: dict[str, ToolOverride] = {}
        tool_metadata = metadata.get("wink_tools", {})

        for tool_desc in descriptor.tools:
            tool_data = tool_metadata.get(tool_desc.name)
            if tool_data:
                stored_hash = tool_data.get("contract_hash")
                if stored_hash and stored_hash != tool_desc.contract_hash:
                    continue  # Contract changed, skip override

                tool_overrides[tool_desc.name] = ToolOverride(
                    name=tool_desc.name,
                    expected_contract_hash=tool_desc.contract_hash,
                    description=tool_data.get("description"),
                    param_descriptions=tool_data.get("param_descriptions", {}),
                )

        if not sections and not tool_overrides:
            return None

        return PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=tag,
            sections=sections,
            tool_overrides=tool_overrides,
        )

    def upsert(
        self,
        descriptor: PromptDescriptor,
        override: PromptOverride,
    ) -> PromptOverride:
        """Push prompt override to LangSmith Hub.

        Creates or updates a prompt in the Hub with WINK metadata for
        hash validation on subsequent pulls.
        """
        from langchain_core.prompts import ChatPromptTemplate

        repo_name = self._repo_name(descriptor.ns, descriptor.key)

        # Build template with section variables
        template_parts: list[str] = []
        input_variables: list[str] = []
        wink_hashes: dict[str, str] = {}

        for section_desc in descriptor.sections:
            section_key = section_desc.path[-1] if section_desc.path else ""
            var_name = f"{section_key}_section"
            template_parts.append(f"{{{var_name}}}")
            input_variables.append(var_name)
            wink_hashes[section_key] = section_desc.content_hash

        template = "\n\n".join(template_parts)
        prompt = ChatPromptTemplate.from_template(template)

        # Store WINK metadata
        metadata = {
            "wink_hashes": wink_hashes,
            "wink_ns": descriptor.ns,
            "wink_key": descriptor.key,
            "wink_tools": {
                tool.name: {
                    "contract_hash": tool.contract_hash,
                    "description": override.tool_overrides.get(tool.name, ToolOverride(
                        name=tool.name,
                        expected_contract_hash=tool.contract_hash,
                    )).description,
                    "param_descriptions": override.tool_overrides.get(
                        tool.name, ToolOverride(
                            name=tool.name,
                            expected_contract_hash=tool.contract_hash,
                        )
                    ).param_descriptions,
                }
                for tool in descriptor.tools
            },
        }

        # Push to Hub with tag
        self._client.push_prompt(
            repo_name,
            object=prompt,
            tags=[override.tag] if override.tag != "latest" else None,
        )

        return override

    def delete(self, *, ns: str, prompt_key: str, tag: str) -> None:
        """Delete is not supported by LangSmith Hub.

        LangSmith maintains version history; use tags to deprecate versions.
        """
        raise NotImplementedError(
            "LangSmith Hub does not support deletion. "
            "Remove the tag or archive the prompt in the LangSmith UI."
        )

    def seed(
        self,
        prompt: PromptLike,
        *,
        tag: str = "latest",
    ) -> PromptOverride:
        """Initialize a Hub prompt from an existing WINK prompt."""
        descriptor = descriptor_for_prompt(prompt)
        override = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=tag,
            sections={},
            tool_overrides={},
        )
        return self.upsert(descriptor, override)

    @staticmethod
    def _extract_section(prompt: object, var_name: str) -> str | None:
        """Extract section content from a LangSmith prompt template."""
        # Implementation depends on prompt structure
        # May need to parse template or access partial variables
        template = getattr(prompt, "template", None)
        if template and f"{{{var_name}}}" in template:
            # Section is a variable - return None to use default
            return None
        partial_variables = getattr(prompt, "partial_variables", {}) or {}
        return partial_variables.get(var_name)
```

### Version Pinning for Production

For production deployments, pin prompt versions using commit hashes:

```python
# Development: always use latest
store = LangSmithPromptOverridesStore(
    LangSmithStoreConfig(default_tag="latest")
)

# Staging: use a named tag
store = LangSmithPromptOverridesStore(
    LangSmithStoreConfig(default_tag="staging")
)

# Production: pin to commit hash
store = LangSmithPromptOverridesStore(
    LangSmithStoreConfig(default_tag="a1b2c3d4")  # Specific commit
)

# Or use the "prod" tag that you update deliberately
store = LangSmithPromptOverridesStore(
    LangSmithStoreConfig(default_tag="prod")
)
```

### Tracing Integration

The adapter automatically integrates with LangSmith tracing when environment
variables are configured. For custom tracing, use the `@traceable` decorator:

```python
from langsmith import traceable

@traceable(run_type="chain", name="WINK Agent Execution")
def execute_with_tracing(
    adapter: LangGraphAdapter,
    prompt: Prompt[OutputT],
    session: SessionProtocol,
) -> PromptResponse[OutputT]:
    """Execute adapter with custom LangSmith trace metadata."""
    return adapter.evaluate(prompt, session=session)
```

### Callback-Based Tracing

For more control, inject LangSmith callbacks directly:

```python
from langsmith.run_helpers import get_current_run_tree

class LangSmithTracingCallback(BaseCallbackHandler):
    """Callback that enriches LangSmith traces with WINK metadata."""

    def __init__(self, prompt_name: str, session_id: str | None) -> None:
        self.prompt_name = prompt_name
        self.session_id = session_id

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.extra["wink_prompt"] = self.prompt_name
            run_tree.extra["wink_session"] = self.session_id

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.extra["wink_tool_input"] = input_str
```

### Adapter Configuration with LangSmith

```python
from weakincentives.adapters.langgraph import (
    LangGraphAdapter,
    LangGraphClientConfig,
    LangSmithStoreConfig,
    LangSmithPromptOverridesStore,
)

# Configure LangSmith store
langsmith_store = LangSmithPromptOverridesStore(
    LangSmithStoreConfig(
        organization="myorg",
        default_tag="prod",
    )
)

# Create adapter with LangSmith integration
adapter = LangGraphAdapter(
    model="gpt-4o",
    model_config=LangGraphModelConfig(model_provider="openai"),
)

# Evaluate with LangSmith-managed overrides
response = adapter.evaluate(
    prompt,
    session=session,
    overrides_store=langsmith_store,
    overrides_tag="prod",
)
```

### OpenTelemetry Integration

LangSmith supports OpenTelemetry for cross-service tracing:

```python
from opentelemetry import trace
from langsmith.wrappers import wrap_openai

# Wrap LLM client for OTel traces
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("wink_agent_execution"):
    response = adapter.evaluate(prompt, session=session)
```

### Prompt Hub Workflow

**Development Workflow**:

1. Author prompts in WINK codebase (source of truth)
2. Seed to LangSmith Hub: `store.seed(prompt, tag="dev")`
3. Iterate in LangSmith Playground
4. Pull changes back: `store.resolve(descriptor, tag="dev")`

**Production Workflow**:

1. Test with `staging` tag
2. Promote to `prod` tag in LangSmith UI
3. Application pulls `prod` tag for stable execution
4. Rollback by reverting tag to previous commit hash

### LangSmith Store Module Structure

```
src/weakincentives/integrations/
├── langsmith/
│   ├── __init__.py
│   ├── store.py          # LangSmithPromptOverridesStore
│   ├── tracing.py        # Tracing callbacks and decorators
│   └── config.py         # LangSmithStoreConfig
```

### Dependencies for LangSmith

```toml
# pyproject.toml
[project.optional-dependencies]
langsmith = [
    "langsmith>=0.1.0",
]
langgraph = [
    "langgraph>=1.0.0,<2.0",
    "langchain>=1.0.0,<2.0",
    "langchain-core>=1.0.0,<2.0",
    "langsmith>=0.1.0",  # Include for full integration
]
```

## Module Structure

```
src/weakincentives/adapters/
├── langgraph.py          # Main adapter implementation
├── _langgraph_tools.py   # Tool translation utilities
├── _langgraph_output.py  # Structured output handling
└── config.py             # Add LangGraphClientConfig, LangGraphModelConfig

src/weakincentives/integrations/
└── langsmith/
    ├── __init__.py
    ├── store.py          # LangSmithPromptOverridesStore
    ├── tracing.py        # Tracing utilities
    └── config.py         # LangSmithStoreConfig
```

## Dependencies

The adapter requires `langgraph` and `langchain` as optional dependencies:

```toml
# pyproject.toml
[project.optional-dependencies]
langgraph = [
    "langgraph>=1.0.0,<2.0",
    "langchain>=1.0.0,<2.0",
    "langchain-core>=1.0.0,<2.0",
]
```

**Python Version**: LangGraph 1.0 requires Python 3.10+ (Python 3.9 support was
dropped following its October 2025 end-of-life).

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

### Basic Agent

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

### With Persistence

```python
from langgraph.checkpoint.sqlite import SqliteSaver

adapter = LangGraphAdapter(
    model="gpt-4o",
    model_config=LangGraphModelConfig(model_provider="openai"),
    checkpointer=SqliteSaver.from_conn_string("agent_state.db"),
    client_config=LangGraphClientConfig(
        thread_id=f"user-{user_id}-session-{session_id}",
    ),
)
```

### Custom Graph

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class MyState(TypedDict):
    messages: Annotated[list, add_messages]
    research_notes: str

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

- Mock LangGraph's `create_agent` and `StateGraph.invoke`
- Verify tool translation preserves handler semantics
- Test structured output parsing for valid and malformed payloads
- Confirm event publication at each stage
- Test checkpoint save/restore cycles

### Integration Tests

```python
@pytest.mark.integration
def test_langgraph_agent(openai_api_key: str) -> None:
    """Verify end-to-end agent execution."""

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
    """Mock for LangGraph's create_agent result."""

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

2. **Token tracking**: Depends on LangChain callback instrumentation; accuracy
   varies by underlying provider.

3. **Deadline enforcement**: LangGraph lacks native deadline support. The
   adapter checks remaining time between graph steps but cannot interrupt
   mid-node execution.

4. **State schema**: LangGraph 1.0 requires TypedDict for state schemas.
   Pydantic models and dataclasses are not supported.

## Future Work

- Streaming support via `stream()` and `astream()`
- Async adapter variant (`AsyncLangGraphAdapter`)
- Supervisor and swarm patterns
- LangGraph Cloud integration for remote execution
- MCP adapter integration

## References

### LangGraph

- [LangGraph Documentation](https://docs.langchain.com/)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [LangGraph v1 Migration Guide](https://docs.langchain.com/oss/python/migrate/langgraph-v1)
- [LangChain and LangGraph 1.0 Announcement](https://blog.langchain.com/langchain-langgraph-1dot0/)
- [How to force tool-calling agent to structure output](https://langchain-ai.github.io/langgraph/how-tos/react-agent-structured-output/)

### LangSmith

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [Manage Prompts Programmatically](https://docs.langchain.com/langsmith/manage-prompts)
- [Trace with LangGraph](https://docs.smith.langchain.com/observability/how_to_guides/trace_with_langgraph)
- [LangChain Hub Announcement](https://blog.langchain.com/langchain-prompt-hub/)
- [End-to-End OpenTelemetry Support](https://blog.langchain.com/end-to-end-opentelemetry-langsmith/)

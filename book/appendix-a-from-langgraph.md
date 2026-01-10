# Appendix A: Coming from LangGraph or LangChain?

If you've built agents with LangGraph, LangChain, or similar frameworks, here's a quick orientation.

## Different philosophy, different primitives

LangGraph centers on **graphs**: nodes are functions, edges are transitions, state flows through the graph. You model agent behavior as explicit control flow. LangChain centers on **chains**: composable sequences of calls to LLMs, tools, and retrievers.

WINK centers on **the prompt itself**. There's no graph. There's no chain. The prompt—a tree of typed sections—_is_ your agent. The model decides what to do next based on what's in the prompt. Tools, instructions, and state all live in that tree.

## Concept mapping

Here's how core concepts translate from LangGraph/LangChain to WINK:

| LangGraph/LangChain | WINK | Notes |
| --- | --- | --- |
| **Graph / Chain** | `PromptTemplate` | Tree of sections vs. graph/chain |
| **Node / Tool** | `Tool` + handler function | Similar in both |
| **State / Memory** | `Session` | Typed slices + reducers vs. dict/class |
| **Router / Conditional edge** | `enabled()` predicate on sections | Declarative enablement |
| **Checkpointing** | `session.snapshot()` / `session.restore()` | Immutable snapshots |
| **LangSmith tracing** | Session events + debug UI | Event-driven telemetry |

## What's familiar

- **Tools are functions with typed params and results.** You'll recognize this pattern. Tools take typed input and return structured output.
- **State management exists.** Sessions use an event-driven pattern: state is immutable, and changes flow through pure functions called "reducers."
- **Provider abstraction exists.** Adapters swap between OpenAI, LiteLLM, Claude—just like LangChain's model abstractions.

## What's different

### No explicit routing

You don't define edges. The model reads the prompt and decides which tools to call. Sections can be conditionally enabled, but there's no "if tool X returns Y, go to node Z."

**LangGraph example:**

```python
# LangGraph: explicit routing
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("action", tool_node)
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)
```

**WINK equivalent:**

```python
# WINK: model decides routing
template = PromptTemplate[Output](
    ns="agent",
    key="workflow",
    sections=[
        instructions_section,
        tool_section,  # Model decides whether to use tools
    ],
)
```

### Prompt and tools are co-located

In LangChain, you define tools in one place and prompts in another. In WINK, the section that explains "use this tool for searching" is the same section that registers the tool. They can't drift apart.

**LangChain example:**

```python
# LangChain: separate definitions
@tool
def search(query: str) -> str:
    """Search the web."""
    return web_search(query)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the search tool when needed."),
    ("human", "{input}"),
])

agent = create_tool_calling_agent(llm, [search], prompt)
```

**WINK equivalent:**

```python
# WINK: co-located
def search_handler(params: SearchParams, *, context: ToolContext) -> ToolResult[str]:
    return ToolResult.ok(web_search(params.query))

search_tool = Tool(
    name="search",
    description="Search the web",
    handler=search_handler,
)

search_section = MarkdownSection(
    title="Search",
    key="search",
    template="Use the search tool when you need current information.",
    tools=(search_tool,),
)
```

### Deterministic by default

Prompt rendering is pure. State transitions flow through reducers. Side effects are confined to tool handlers. You can snapshot the entire state at any point and restore it later.

**What this means:**

- Same inputs always produce same prompts
- You can test prompt rendering without hitting APIs
- State changes are auditable through event logs
- Time-travel debugging is straightforward

### No async (yet)

Adapters are synchronous. This simplifies debugging at the cost of throughput. Async may come later.

**Implications:**

- Easier to reason about execution flow
- No async/await ceremony
- May need threading for concurrency
- Not ideal for high-throughput streaming

## When to use WINK instead of LangGraph

Choose WINK when:

- **You want the prompt to be the source of truth**, not a graph definition
- **You're building single-agent workflows** where the model handles most routing
- **You value determinism, testability, and auditability** over flexibility
- **You're tired of prompt text and tool definitions drifting apart**
- **You need to version and review exact prompt text** (compliance, debugging)

## When to stick with LangGraph

Stick with LangGraph when:

- **You need explicit multi-step workflows** with complex branching logic
- **You're building multi-agent systems** with explicit handoffs between agents
- **You need async streaming** throughout your pipeline
- **You have workflows that don't fit well into prompt-based routing**

## Can you use both?

Yes! WINK for prompt/tool/state management, LangGraph for higher-level orchestration:

```python
# Use WINK for individual agents
wink_agent = MainLoop(prompt=agent_prompt, adapter=adapter)

# Use LangGraph for multi-agent coordination
workflow = StateGraph(AgentState)
workflow.add_node("research", lambda state: wink_research_agent.execute(...))
workflow.add_node("write", lambda state: wink_writer_agent.execute(...))
workflow.add_edge("research", "write")
```

## Migration example

Here's a simple LangGraph agent translated to WINK:

**LangGraph version:**

```python
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Calculate a math expression."""
    return str(eval(expression))

def agent_node(state):
    # Call model with tools
    result = model.invoke(state["messages"])
    return {"messages": [result]}

def tool_node(state):
    # Execute tool calls
    tools = {"calculator": calculator}
    results = []
    for tool_call in state["messages"][-1].tool_calls:
        result = tools[tool_call["name"]].invoke(tool_call["args"])
        results.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
    return {"messages": results}

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
```

**WINK version:**

```python
from weakincentives.prompt import Prompt, PromptTemplate, MarkdownSection, Tool, ToolResult
from dataclasses import dataclass

@dataclass(frozen=True)
class CalculateParams:
    expression: str

def calculator_handler(params: CalculateParams, *, context: ToolContext) -> ToolResult[str]:
    try:
        result = str(eval(params.expression))
        return ToolResult.ok(result, message=f"Calculated: {result}")
    except Exception as e:
        return ToolResult.error(f"Error: {e}")

calculator_tool = Tool(
    name="calculator",
    description="Calculate a math expression",
    handler=calculator_handler,
)

template = PromptTemplate[str](
    ns="math",
    key="calculator",
    sections=[
        MarkdownSection(
            title="Instructions",
            key="instructions",
            template="Calculate the result of the given expression: ${expression}",
            tools=(calculator_tool,),
        ),
    ],
)

# Use it
prompt = Prompt(template)
response = adapter.evaluate(prompt.bind(CalculateParams(expression="2 + 2")))
print(response.output)  # "4"
```

Key differences:
- No explicit graph—model decides whether to call the tool
- Tool and instructions co-located in the same section
- Typed parameters with dataclasses
- Explicit success/error handling with `ToolResult`

## Summary

| Aspect | LangGraph/LangChain | WINK |
| --- | --- | --- |
| **Core abstraction** | Graph/Chain | Prompt tree |
| **Routing** | Explicit edges | Model-driven |
| **Prompt management** | Separate from tools | Co-located with tools |
| **State** | Mutable dict/class | Immutable slices + reducers |
| **Determinism** | Optional | Default |
| **Async** | First-class | Not yet supported |
| **Best for** | Multi-agent, complex workflows | Single-agent, prompt-centric |

Both frameworks are valuable. Choose based on whether your agent's behavior should be defined by **explicit control flow** (LangGraph) or by **what the prompt contains** (WINK).

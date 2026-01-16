# Coming from LangGraph or LangChain?

If you've built agents with LangGraph, LangChain, or similar frameworks, here's
a quick orientation.

## Different Philosophy, Different Primitives

**LangGraph** centers on **graphs**: nodes are functions, edges are transitions,
state flows through the graph. You model agent behavior as explicit control
flow.

**LangChain** centers on **chains**: composable sequences of calls to LLMs,
tools, and retrievers.

**WINK** centers on **the prompt itself**. There's no graph. There's no chain.
The prompt—a tree of typed sections—*is* your agent. The model decides what to
do next based on what's in the prompt. Tools, instructions, and state all live
in that tree.

## Concept Mapping

| LangGraph/LangChain | WINK |
| --- | --- |
| Graph / Chain | `PromptTemplate` (tree of sections) |
| Node / Tool | `Tool` + handler function |
| State / Memory | `Session` (typed slices + reducers) |
| Router / Conditional edge | `enabled()` predicate on sections |
| Checkpointing | `session.snapshot()` / `session.restore()` |
| LangSmith tracing | Session events + debug UI |

## What's Familiar

- **Tools are functions with typed params and results.** You'll recognize this
  pattern.

- **State management exists.** Sessions use an event-driven pattern: state is
  immutable, and changes flow through pure functions called "reducers."

- **Provider abstraction exists.** Adapters swap between OpenAI, LiteLLM, and
  Claude.

## What's Different

**No explicit routing.** You don't define edges. The model reads the prompt and
decides which tools to call. Sections can be conditionally enabled, but there's
no "if tool X returns Y, go to node Z."

**Prompt and tools are co-located.** In LangChain, you define tools in one place
and prompts in another. In WINK, the section that explains "use this tool for
searching" is the same section that registers the tool. They can't drift apart.

**Deterministic by default.** Prompt rendering is pure. State transitions flow
through reducers. Side effects are confined to tool handlers. You can snapshot
the entire state at any point and restore it later.

**No async (yet).** Adapters are synchronous. This simplifies debugging at the
cost of throughput.

## When to Use WINK Instead of LangGraph

- You want the prompt to be the source of truth, not a graph definition.
- You're building single-agent workflows where the model handles most routing.
- You value determinism, testability, and auditability over flexibility.
- You're tired of prompt text and tool definitions drifting apart.

## When to Stick with LangGraph

- You need explicit multi-step workflows with complex branching logic.
- You're building multi-agent systems with explicit handoffs.
- You need async streaming throughout.

## Using Both

You can use both: WINK for prompt/tool/state management, LangGraph for
higher-level orchestration. WINK doesn't try to own your application
architecture.

## Migration Example

**LangGraph style:**

```python
# Define tools separately
@tool
def search(query: str) -> str:
    return f"Results for {query}"

# Define graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge("agent", "tools")
graph.add_conditional_edges("tools", should_continue)
```

**WINK style:**

```python
# Tool and instructions together
search_tool = Tool[SearchParams, SearchResult](
    name="search",
    description="Search for information",
    handler=search_handler,
)

section = MarkdownSection(
    title="Search",
    key="search",
    template="Use search to find information.",
    tools=(search_tool,),
)

# No explicit routing - model decides based on prompt
template = PromptTemplate(
    ns="agent",
    key="main",
    sections=(section,),
)
```

The model sees the instructions and tools together and decides when to use them.
No routing graph needed.

## Next Steps

- [Philosophy](philosophy.md): Understand the WINK approach
- [Quickstart](quickstart.md): Get something running
- [Tools](tools.md): Learn how tools work in WINK

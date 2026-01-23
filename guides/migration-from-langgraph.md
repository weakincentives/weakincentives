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

This isn't just a different API—it's a bet that orchestration complexity is
shrinking. Frontier models increasingly handle planning, reasoning, and
self-correction in a single context window. What required elaborate routing
graphs yesterday often works in one well-structured prompt today.

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

## WINK Strengths

### Type Safety as a First Principle

Everything is typed: params, tool calls, tool results, structured outputs,
session state. Type mismatches surface at construction time, not when the model
is mid-response. Pyright strict mode is enforced throughout.

```python nocheck
# LangGraph: schemas often defined separately, drift possible
# WINK: types are the contract

@dataclass(frozen=True)
class SearchParams:
    query: Annotated[str, {"min_length": 1, "max_length": 500}]
    limit: Annotated[int, {"ge": 1, "le": 100}] = 10

@dataclass(frozen=True)
class SearchResult:
    matches: tuple[Match, ...]
    total_count: int
```

Constraints are validated at parse time. The model sees a JSON Schema generated
from these types. When the model returns invalid data, WINK catches it before
your code runs.

### Testable Without a Model

Most agent testing happens without calling an LLM:

- **Prompt rendering tests**: Assert on exact prompt text, deterministically
- **Tool handler tests**: Call handlers directly with fake contexts
- **Reducer tests**: Pure functions—given state and event, expect output
- **Session state tests**: Verify state evolution through dispatched events

```python nocheck
# Test prompt rendering without any LLM
def test_prompt_includes_context():
    rendered = prompt.bind(params).render(session=session)
    assert "Search the repository" in rendered.text
    assert "grep" in [t.name for t in rendered.tools]

# Test tool handlers in isolation
def test_search_handler():
    result = search_handler(SearchParams(query="test"), context=fake_context)
    assert result.success
```

LangGraph tests often require mocking the entire graph or running real model
calls. WINK's determinism makes unit testing straightforward.

### First-Class Debugging

WINK captures everything:

- **Session events**: Every prompt render, tool invocation, and state change
- **Debug bundles**: Self-contained zip archives with full execution state
- **Debug UI**: Visual timeline of prompts, tools, and state evolution
- **Structured logging**: Machine-parseable JSON with event taxonomy

```bash
# Start the debug UI
wink debug ./debug_bundles/

# See exactly what was sent, what tools ran, how state evolved
```

When something goes wrong, you can inspect the exact prompt that was sent, the
tool results the model saw, and every state transition. No guesswork.

### Zero-Dependency Serialization

No Pydantic required. WINK's `serde` module handles dataclass serialization with
stdlib types and `typing.Annotated` for constraints:

```python nocheck
from weakincentives.serde import parse, dump, schema

# Parse with validation
user = parse(User, {"name": "Ada", "age": "39"})  # Coerces age to int

# Dump to JSON-compatible dict
payload = dump(user)

# Generate JSON Schema for model tool calls
user_schema = schema(User)
```

This keeps your dependency tree minimal and your serialization predictable.

### Immutable State with Pure Reducers

Session state is never mutated directly. Changes flow through events and pure
reducer functions:

```python nocheck
@reducer(on=AddStep)
def add_step(state: tuple[Plan, ...], event: AddStep) -> SliceOp[Plan]:
    current = state[-1] if state else Plan(steps=())
    return Append(Plan(steps=current.steps + (event.step,)))
```

This makes state transitions:

- **Predictable**: Same events always produce the same state
- **Inspectable**: Query the full history of any slice
- **Replayable**: Snapshot and restore at any point

### Safe Prompt Iteration

Prompt overrides are validated against hashes. When you change a section in
code, existing overrides stop applying until you explicitly update them:

```python nocheck
# Overrides are tied to specific section versions
# No silent drift between "tested" and "running"
override = SectionOverride(
    section_key="search.instructions",
    content_hash="abc123...",  # Must match current section
    text="Updated instructions...",
)
```

A/B test prompts in production without deploys, with confidence that you're
modifying what you think you're modifying.

### Progressive Disclosure

Control context size by defaulting to summaries and expanding on demand:

```python nocheck
section = MarkdownSection(
    title="Detailed Context",
    key="context",
    template="...(long context)...",
    summary="Context available. Use expand_context tool for details.",
    visibility=Visibility.COLLAPSED,
)
```

The model requests what it needs instead of everything being stuffed into the
prompt upfront. This keeps token counts manageable and models focused.

### Design by Contract

Public APIs use preconditions and postconditions:

```python nocheck
from weakincentives.dbc import require, ensure

@require(lambda query: len(query) > 0, "query must not be empty")
@ensure(lambda result: result.success or result.message, "failed results need messages")
def search_handler(params: SearchParams, *, context: ToolContext) -> ToolResult[SearchResult]:
    ...
```

Contracts make invariants explicit. When a contract fails, you know exactly
which assumption was violated.

## When to Use WINK Instead of LangGraph

**You want the prompt to be the source of truth.** Instead of modeling behavior
in a graph, you describe it in structured sections. The model reads the prompt
and makes decisions. This works well when the model is capable enough to handle
routing—which modern models increasingly are.

**You value testability.** If you want to unit test your agent logic without
spinning up mocks or making real API calls, WINK's deterministic design makes
this natural. Prompt rendering is pure. Reducers are pure. Tool handlers can be
tested in isolation.

**You want auditability.** Every prompt sent, every tool called, every state
change—captured in an event log you can query, snapshot, and replay. Debug
bundles give you a complete picture of what happened after the fact.

**You need type safety.** If you've been bitten by schema drift, runtime type
errors, or models returning unexpected shapes, WINK's strict typing catches
these issues early.

**You're building domain-specific tools.** WINK's colocation pattern means your
tool definitions and instructions stay together. When you update one, the other
is right there. They can't silently diverge.

**You want to iterate on prompts safely.** Hash-validated overrides let you A/B
test prompt changes without code deploys, with guarantees that you're modifying
the version you think you're modifying.

## When to Stick with LangGraph

- You need explicit multi-step workflows with complex branching logic that the
  model shouldn't control.
- You're building multi-agent systems with explicit handoffs between specialized
  agents.
- You need async streaming throughout the stack.
- You prefer graph-based visualization and debugging tools.

## Using Both

You can use both: WINK for prompt/tool/state management, LangGraph for
higher-level orchestration. WINK doesn't try to own your application
architecture.

A common pattern: use WINK for the pieces that benefit from determinism—prompt
composition, tool contracts, state snapshots—and let LangGraph or another
orchestrator handle multi-agent coordination when you need it. WINK is a library,
not a framework that demands you go all-in.

## Migration Patterns

### Converting Tools

LangGraph tools are often decorated functions. Convert to WINK's typed pattern:

```python nocheck
# Before: LangGraph
@tool
def lookup_user(user_id: str) -> dict:
    """Look up a user by ID."""
    return {"name": "Ada", "email": "ada@example.com"}

# After: WINK
@dataclass(frozen=True)
class LookupUserParams:
    user_id: Annotated[str, {"pattern": r"^usr_[a-z0-9]+$"}]

@dataclass(frozen=True)
class UserInfo:
    name: str
    email: str

def lookup_user_handler(params: LookupUserParams, *, context: ToolContext) -> ToolResult[UserInfo]:
    user = db.get(params.user_id)
    if not user:
        return ToolResult.error(f"User {params.user_id} not found")
    return ToolResult.ok(UserInfo(name=user.name, email=user.email))

lookup_user_tool = Tool[LookupUserParams, UserInfo](
    name="lookup_user",
    description="Look up a user by their ID",
    handler=lookup_user_handler,
)
```

The WINK version has:

- Typed, validated params (regex ensures valid user ID format)
- Typed results (not just `dict`)
- Explicit error handling via `ToolResult`
- Testable handler function

### Converting State

LangGraph uses TypedDict state. Convert to WINK sessions:

```python nocheck
# Before: LangGraph
class AgentState(TypedDict):
    messages: list[dict]
    current_step: str

def update_step(state: AgentState) -> AgentState:
    state["current_step"] = "analyzing"
    return state

# After: WINK
@dataclass(frozen=True)
class WorkflowState:
    current_step: str
    messages: tuple[Message, ...]

@dataclass(frozen=True)
class StepChanged:
    new_step: str

@reducer(on=StepChanged)
def step_changed(state: tuple[WorkflowState, ...], event: StepChanged) -> SliceOp[WorkflowState]:
    current = state[-1]
    return Replace(WorkflowState(current_step=event.new_step, messages=current.messages))

# Usage
session = Session()
session.install(WorkflowState, initial=lambda: WorkflowState(current_step="started", messages=()))
session.dispatch(StepChanged(new_step="analyzing"))

# Query state
current = session[WorkflowState].latest()
history = session[WorkflowState].all()  # Full history available
```

The WINK version:

- Immutable state (frozen dataclass)
- Pure reducer functions
- Full history available for inspection
- State changes are explicit events

### Converting Conditional Edges

Replace graph routing with section visibility or prompt structure:

```python nocheck
# Before: LangGraph conditional routing
def should_search(state: AgentState) -> str:
    if needs_search(state):
        return "search_node"
    return "respond_node"

graph.add_conditional_edges("analyze", should_search)

# After: WINK - let the model decide, or use conditional sections
search_section = MarkdownSection(
    title="Search",
    key="search",
    template="If you need more information, use the search tool.",
    tools=(search_tool,),
    enabled=lambda ctx: not ctx.session[SearchState].latest().complete,
)
```

Often, you don't need the conditional at all—just describe when to use the tool
in the instructions, and let the model decide.

## Migration Example

**LangGraph style:**

```python nocheck
# Tools defined separately from prompts
@tool
def search(query: str) -> str:
    return f"Results for {query}"

# State is a TypedDict
class AgentState(TypedDict):
    messages: list
    results: list

# Graph defines control flow
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge("agent", "tools")
graph.add_conditional_edges("tools", should_continue)

# Prompts live somewhere else, can drift from tools
SYSTEM_PROMPT = "You are a search assistant..."
```

**WINK style:**

```python nocheck
from dataclasses import dataclass
from typing import Annotated
from weakincentives.prompt import Tool, ToolResult, ToolContext, MarkdownSection, PromptTemplate
from weakincentives.runtime import Session

# Typed params with constraints
@dataclass(frozen=True)
class SearchParams:
    query: Annotated[str, {"min_length": 1, "max_length": 500}]
    limit: Annotated[int, {"ge": 1, "le": 100}] = 10

@dataclass(frozen=True)
class SearchResult:
    matches: tuple[str, ...]
    total: int

# Handler is a regular function, easily testable
def search_handler(params: SearchParams, *, context: ToolContext) -> ToolResult[SearchResult]:
    results = do_search(params.query, params.limit)
    return ToolResult.ok(SearchResult(matches=results, total=len(results)))

# Tool definition
search_tool = Tool[SearchParams, SearchResult](
    name="search",
    description="Search the knowledge base",
    handler=search_handler,
)

# Tool and instructions co-located—they can't drift apart
search_section = MarkdownSection(
    title="Search",
    key="search",
    template="""
    Use the search tool to find relevant information.
    - Search before answering factual questions
    - Use specific, targeted queries
    """,
    tools=(search_tool,),
)

# Prompt is a typed, composable tree
template = PromptTemplate[str](
    ns="assistant",
    key="main",
    sections=(search_section,),
)

# Session captures all state and events
session = Session()

# Render is deterministic—same inputs, same output
rendered = template.prompt().bind(params).render(session=session)
print(rendered.text)  # Inspect exactly what's sent
```

The model sees instructions and tools together. No routing graph—the model
decides when to search based on the prompt. Test the handler without a model.
Snapshot the session at any point. Query the event log to see what happened.

## Side-by-Side Comparison

| Aspect | LangGraph | WINK |
| --- | --- | --- |
| Control flow | Graph edges | Model decides from prompt |
| Tool definitions | Separate from prompts | Co-located with instructions |
| State mutations | Direct dict updates | Immutable, event-driven reducers |
| Testing | Often requires mocks/integration | Unit test handlers and reducers directly |
| Debugging | Step through graph | Debug UI, event logs, snapshots |
| Type safety | Optional schemas | Strict typing, validated at parse time |
| Serialization | Pydantic or custom | Built-in serde, no dependencies |
| Prompt iteration | Code changes | Hash-validated overrides |

## Next Steps

**Getting started:**

- [Quickstart](quickstart.md): Get a working agent running
- [Philosophy](philosophy.md): Understand the "weak incentives" approach

**Core concepts:**

- [Prompts](prompts.md): Typed, testable prompts as first-class objects
- [Tools](tools.md): Define tool contracts and handlers
- [Sessions](sessions.md): Immutable state with event-driven reducers
- [Serialization](serialization.md): Built-in serde without Pydantic

**Testing and debugging:**

- [Testing](testing.md): Unit test prompts, tools, and reducers without a model
- [Debugging](debugging.md): Debug UI, bundles, and structured logging

**Production:**

- [Adapters](adapters.md): Connect to OpenAI, LiteLLM, or Claude
- [Evaluation](evaluation.md): Systematic testing with datasets

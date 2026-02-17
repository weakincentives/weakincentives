# Coming from LangGraph or LangChain?

If you've built agents with LangGraph, LangChain, or similar frameworks, here's
a quick orientation—and a different way of thinking about agent design.

## The Execution Harness Has Already Won

Before discussing WINK, consider the landscape. Tools like **Claude Code**,
**Codex**, and **OpenCode** are high-level execution harnesses. They own the
agentic loop: planning, tool execution, retries, context management,
sandboxing. They're battle-tested, maintained by well-resourced teams, and
improving rapidly.

When you build a LangGraph graph, you're competing with these harnesses. You're
writing your own orchestration—your own routing, your own retry logic, your own
tool execution pipeline—on top of a model that's increasingly capable of
handling all of that itself. Every edge in your graph is a decision you're
making *instead of* the model or its native harness.

**WINK takes a different approach: don't compete with the harness. Compose with
it.**

The Claude Agent SDK adapter, for example, delegates execution entirely to
Claude Code's native runtime. WINK doesn't run tools—Claude Code does. WINK
doesn't manage retries—the harness does. WINK doesn't plan—the model does.

What WINK owns is the layer that harnesses *don't*: prompt composition, tool
contracts, session state, and domain knowledge encoding. This is the layer you
actually differentiate on.

## Policies, Not Workflows

This leads to the core design philosophy: **policies over workflows**.

**Workflows** prescribe exactly what happens: "First do A. If condition X, do B,
otherwise do C. Then always do D." You encode behavior as a state machine.

**Policies** describe what's allowed and what matters: "Here are your tools.
Here are the constraints. Here's what success looks like. Figure it out."

This distinction matters because frontier models are increasingly capable of
planning, reasoning, and self-correction. When you over-specify control flow,
you're fighting the model's capabilities rather than leveraging them. And when
you're running inside a capable harness like Claude Code or Codex, you're also
fighting the harness's orchestration.

Consider a code review agent:

```python nocheck
# Workflow approach (LangGraph)
# You must anticipate every path—and you're reimplementing
# what Claude Code already does natively
graph.add_node("fetch_diff", fetch_diff)
graph.add_node("analyze_security", analyze_security)
graph.add_node("analyze_style", analyze_style)
graph.add_node("check_tests", check_tests)
graph.add_node("synthesize", synthesize)
graph.add_conditional_edges("fetch_diff", route_by_file_type)
graph.add_conditional_edges("analyze_security", route_by_severity)
# ... 50 more edges for edge cases you'll discover in production

# Policy approach (WINK)
# Describe what matters. Let the model and harness handle execution.
review_section = MarkdownSection(
    title="Review Policy",
    key="review",
    template="""
    Review this pull request. Consider:
    - Security implications (injection, auth, data exposure)
    - Code style and maintainability
    - Test coverage for changed code paths
    - Performance implications for hot paths

    Use the available tools to fetch context as needed.
    Flag blocking issues separately from suggestions.
    """,
    tools=(search_codebase, read_file, check_coverage),
)
```

The workflow approach requires you to anticipate every branching condition *and*
reimplement tool execution that the harness already provides. The policy
approach tells the model what matters and trusts the model and its harness to
figure out the sequencing.

## Why This Matters Now

Graph-based orchestration made sense when models needed heavy scaffolding. Early
LLMs struggled with multi-step reasoning, lost track of goals, and needed
explicit guidance at every turn.

Three things have changed:

**Models got better at planning.** Modern models can plan multi-step approaches,
revise when they hit obstacles, recognize when they need more information, and
self-correct when initial approaches fail.

**Execution harnesses matured.** Claude Code, Codex, and OpenCode handle tool
execution, sandboxing, retries, and context management. This infrastructure was
hard to build and is now available off-the-shelf.

**The bottleneck moved.** The hard problem isn't orchestration anymore—it's
context engineering. What information does the model need? How should it be
structured? What tools expose your domain capabilities? This is where leverage
lives, and it's exactly where WINK focuses.

The default should flip: **err on the side of giving the agent more degrees of
freedom, and only constrain where you have evidence that constraints help.**

## Where WINK Fits

```
┌─────────────────────────────────────────────┐
│  Execution Harness (Claude Code / Codex)    │
│  Planning, tool execution, retries, sandbox │
│                                             │
│  ┌───────────────────────────────────────┐  │
│  │  WINK                                 │  │
│  │  Prompts, tools, state, contracts     │  │
│  │                                       │  │
│  │  ┌─────────────────────────────────┐  │  │
│  │  │  Your Domain                    │  │  │
│  │  │  Business logic, data, rules    │  │  │
│  │  └─────────────────────────────────┘  │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

WINK sits between the harness and your domain. It provides:

1. **Prompts**: A tree of typed sections that describe capabilities, context,
   and constraints. The harness renders this and hands it to the model.

1. **Tools**: Typed functions with validated params and results. The harness
   calls these via MCP bridging. Your side effects and validation stay in Python.

1. **Sessions**: Immutable state that captures everything—what was sent, what
   tools ran, what changed. Full auditability regardless of which harness runs
   the loop.

```python nocheck
# WINK defines the policy and tools
search_section = MarkdownSection(
    title="Search Capabilities",
    key="search",
    template="""
    You have access to semantic and keyword search over the codebase.
    Prefer targeted queries over broad ones. Refine if initial results
    aren't helpful.
    """,
    tools=(semantic_search, keyword_search),
)

# The harness (Claude Code) handles execution
adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        max_turns=20,
        max_budget_usd=2.0,
        isolation=IsolationConfig(
            network_policy=NetworkPolicy.no_network(),
        ),
    ),
)
```

WINK defines *what* the agent knows and *what* it can do. The harness decides
*how* and *when* to act on that.

## Constraints Where They Matter

Policies don't mean "anything goes." You still constrain—but you constrain at
boundaries, not at every decision point.

**Constrain at tool boundaries:**

```python nocheck
@dataclass(frozen=True)
class FileWriteParams:
    path: Annotated[str, {"pattern": r"^src/.*\.py$"}]  # Only src/*.py
    content: str

def file_write_handler(params: FileWriteParams, *, context: ToolContext) -> ToolResult[None]:
    if ".." in params.path:
        return ToolResult.error("Path traversal not allowed")
    # ...
```

The model has freedom to decide *when* to write files and *what* to write. The
constraint ("only Python files in src/") is enforced at the boundary.

**Constrain with conditional sections:**

```python nocheck
dangerous_ops_section = MarkdownSection(
    title="Destructive Operations",
    key="dangerous",
    template="You may delete files and reset git state if explicitly requested.",
    tools=(delete_file, git_reset),
    enabled=lambda ctx: ctx.session[Permissions].latest().allow_destructive,
)
```

The section only appears when permissions allow. The model doesn't see tools it
can't use. The policy adapts to context.

**Constrain with design by contract:**

```python nocheck
@require(lambda params: len(params.query) >= 3, "query too short")
@ensure(lambda result: len(result.value.matches) <= 100, "too many results")
def search_handler(params: SearchParams, *, context: ToolContext) -> ToolResult[SearchResult]:
    ...
```

Preconditions and postconditions make invariants explicit. When violated, you
know exactly which assumption broke.

## What LangGraph Gives You That You Lose

Be clear-eyed about what you're trading:

**Explicit control flow visualization.** Graph-based systems can be visualized
as flowcharts. WINK's control flow is implicit in the prompt—harder to draw on
a whiteboard, though session events let you reconstruct what actually happened.

**Deterministic routing.** If your business requires "always do X before Y,"
a graph edge guarantees it. Policy-based approaches rely on the model following
instructions, which is reliable but not deterministic.

**Ecosystem.** LangGraph has LangSmith, LangServe, and a large community. WINK
is alpha software with a smaller user base.

These tradeoffs are real. But consider: if you're deploying inside Claude Code
or Codex, you're already getting execution, sandboxing, and observability from
the harness. You don't need LangGraph to provide those.

## When Workflows Still Make Sense

Use explicit workflow control when:

**Hard sequencing requirements exist.** If step B literally cannot run until
step A completes (not "shouldn't"—"cannot"), encode that.

**Compliance requires proving the path.** Some domains require demonstrating
that specific steps happened in specific order.

**The model consistently fails at planning.** If you've tried policy-based
approaches and the model reliably makes bad decisions for your specific task,
add structure. But try the simpler approach first.

**You're orchestrating across harness boundaries.** When different harness
instances need to coordinate, explicit handoffs may work better than shared
context.

## Concept Mapping

| LangGraph | WINK | Notes |
| --- | --- | --- |
| Graph | `PromptTemplate` | Tree of sections, not edges |
| Node | Tool handler | Executed by the harness |
| Edge | (implicit) | Model/harness decides sequencing |
| Conditional edge | `enabled()` predicate | Section appears or doesn't |
| State | `Session` | Immutable, event-driven |
| Checkpointing | `snapshot()`/`restore()` | Full state capture |
| LangSmith | Debug UI + events | Built-in, no external service |
| Tool executor | Harness (Claude Code) | WINK doesn't execute tools in SDK mode |

## WINK Strengths

### Type Safety Throughout

Everything is typed: tool params, tool results, session state, structured
outputs. Mismatches surface at construction time, not mid-execution.

```python nocheck
@dataclass(frozen=True)
class SearchParams:
    query: Annotated[str, {"min_length": 1, "max_length": 500}]
    limit: Annotated[int, {"ge": 1, "le": 100}] = 10

@dataclass(frozen=True)
class SearchResult:
    matches: tuple[Match, ...]
    total_count: int
```

Constraints validate at parse time. The model sees a JSON Schema generated from
these types. Invalid model output is caught before your code runs.

### Testable Without Models

Most testing happens without LLM calls:

```python nocheck
# Test prompt rendering deterministically
def test_prompt_includes_search_instructions():
    rendered = prompt.bind(params).render(session=session)
    assert "semantic search" in rendered.text.lower()

# Test tool handlers directly
def test_search_returns_results():
    result = search_handler(SearchParams(query="auth"), context=fake_context)
    assert result.success

# Test reducers as pure functions
def test_add_finding_appends():
    state = (Finding(text="first"),)
    event = AddFinding(text="second")
    op = finding_reducer(state, event)
    assert isinstance(op, Append)
```

No mocking graphs or integration tests pretending to be unit tests. Determinism
makes testing straightforward.

### First-Class Debugging

Every prompt, tool call, and state change is captured:

```bash
wink debug ./debug_bundles/
```

See exactly what was sent, what tools ran, how state evolved. Full artifacts,
not just logs.

### Zero-Dependency Serialization

No Pydantic required. WINK's `serde` module uses stdlib dataclasses:

```python nocheck
from weakincentives.serde import parse, dump, schema

user = parse(User, {"name": "Ada", "age": "39"})  # Coerces types
payload = dump(user)  # JSON-compatible dict
user_schema = schema(User)  # JSON Schema for tools
```

### Immutable State with Pure Reducers

State changes flow through events and pure functions:

```python nocheck
@reducer(on=AddFinding)
def add_finding(state: tuple[Finding, ...], event: AddFinding) -> SliceOp[Finding]:
    return Append(Finding(text=event.text, severity=event.severity))
```

Same events always produce the same state. Full history available. Snapshot and
restore at any point.

### Safe Prompt Iteration

Overrides are hash-validated. When you change a section in code, existing
overrides stop applying until explicitly updated:

```python nocheck
override = SectionOverride(
    section_key="search.instructions",
    content_hash="abc123...",
    text="Updated instructions...",
)
```

A/B test prompts without deploys, with confidence you're modifying what you
think you're modifying.

### Progressive Disclosure

Control context size by defaulting to summaries:

```python nocheck
section = MarkdownSection(
    title="Codebase Context",
    key="context",
    template="...(detailed context)...",
    summary="Codebase context available. Use expand_context for details.",
    visibility=Visibility.COLLAPSED,
)
```

The model requests what it needs instead of everything being stuffed into every
prompt.

## Making the Transition

1. **Identify what the harness already provides.** If you're deploying on Claude
   Code or Codex, list the capabilities your LangGraph graph reimplements:
   tool execution, retries, sandboxing, context management. These can go.

1. **Extract your domain tools.** The durable value in your LangGraph agents is
   the tool implementations—the code that talks to your database, your APIs,
   your business logic. Convert these to WINK's typed pattern.

1. **Separate true constraints from guidance.** Which graph edges represent hard
   requirements ("must happen before") vs. guidance ("usually do this first")?
   Keep the former as explicit instructions. Convert the latter to policy.

1. **Write policy, not procedure.** Describe what the agent should accomplish
   and what tools are available. Let the model and harness figure out sequencing.

1. **Add structure only when needed.** If the model consistently makes poor
   decisions about ordering, add conditional sections or explicit instructions.
   But start permissive and tighten based on evidence.

## Migration Example

**LangGraph—you own the loop:**

```python nocheck
@tool
def search(query: str) -> str:
    return f"Results for {query}"

class AgentState(TypedDict):
    messages: list
    search_done: bool
    analysis_done: bool

graph = StateGraph(AgentState)
graph.add_node("search", search_node)
graph.add_node("analyze", analyze_node)
graph.add_node("respond", respond_node)
graph.add_edge(START, "search")
graph.add_conditional_edges("search", lambda s: "analyze" if s["search_done"] else "search")
graph.add_conditional_edges("analyze", lambda s: "respond" if s["analysis_done"] else "analyze")
```

**WINK—the harness owns the loop, you own the knowledge:**

```python nocheck
@dataclass(frozen=True)
class SearchParams:
    query: Annotated[str, {"min_length": 1}]

def search_handler(params: SearchParams, *, context: ToolContext) -> ToolResult[SearchResult]:
    results = do_search(params.query)
    return ToolResult.ok(SearchResult(matches=results))

search_section = MarkdownSection(
    title="Search",
    key="search",
    template="""
    Search the codebase when you need to understand existing implementations
    or find relevant context. Refine queries if initial results aren't helpful.
    """,
    tools=(Tool(name="search", handler=search_handler, description="Search codebase"),),
)

analysis_section = MarkdownSection(
    title="Analysis",
    key="analysis",
    template="""
    After gathering context, analyze the code for:
    - Correctness issues
    - Security concerns
    - Style violations

    Distinguish blocking issues from suggestions.
    """,
)

template = PromptTemplate[str](
    ns="reviewer",
    key="main",
    sections=(search_section, analysis_section),
)

# Claude Code handles execution—you handle knowledge
adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        max_turns=20,
        isolation=IsolationConfig(
            network_policy=NetworkPolicy.no_network(),
        ),
    ),
)
```

## Side-by-Side

| Aspect | LangGraph | WINK + Harness |
| --- | --- | --- |
| Who runs the loop | Your graph | Harness (Claude Code, Codex) |
| Control flow | Explicit edges | Model decides from prompt |
| Tool execution | Your code | Harness + MCP bridge |
| Sandboxing | Your responsibility | Harness provides |
| Adding capabilities | New nodes + edges | New section + tools |
| Handling edge cases | More conditional edges | Model adapts |
| Testing | Mock graph execution | Test components directly |
| Constraints | Edges that block paths | Tool validation + contracts |

## Next Steps

**Understand the philosophy:**

- [Philosophy](philosophy.md): The "weak incentives" approach in depth

**Learn the primitives:**

- [Prompts](prompts.md): Typed, composable prompt trees
- [Tools](tools.md): Typed tool contracts and handlers
- [Sessions](sessions.md): Immutable state with event-driven reducers

**Integration with harnesses:**

- [Claude Agent SDK](claude-agent-sdk.md): Production integration with Claude
  Code
- [Skills Authoring](skills-authoring.md): Extend harness capabilities

**Build and test:**

- [Quickstart](quickstart.md): Get a working agent running
- [Testing](testing.md): Test without models
- [Debugging](debugging.md): Debug UI and event inspection

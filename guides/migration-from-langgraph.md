# Coming from LangGraph or LangChain?

If you've built agents with LangGraph, LangChain, or similar frameworks, here's
a quick orientation—and a different way of thinking about agent design.

## Policies, Not Workflows

The most important difference between WINK and graph-based frameworks isn't the
API. It's the underlying philosophy: **policies over workflows**.

**Workflows** prescribe exactly what happens: "First do A. If condition X, do B,
otherwise do C. Then always do D." You encode behavior as a state machine.

**Policies** describe what's allowed and what matters: "Here are your tools.
Here are the constraints. Here's what success looks like. Figure it out."

This distinction matters because frontier models are increasingly capable of
planning, reasoning, and self-correction. When you over-specify control flow,
you're often fighting the model's capabilities rather than leveraging them.

Consider a code review agent:

```python nocheck
# Workflow approach (LangGraph)
# You must anticipate every path
graph.add_node("fetch_diff", fetch_diff)
graph.add_node("analyze_security", analyze_security)
graph.add_node("analyze_style", analyze_style)
graph.add_node("check_tests", check_tests)
graph.add_node("synthesize", synthesize)
graph.add_conditional_edges("fetch_diff", route_by_file_type)
graph.add_conditional_edges("analyze_security", route_by_severity)
# ... 50 more edges

# Policy approach (WINK)
# Describe what matters, let the model plan
"""
Review this pull request. Consider:
- Security implications (injection, auth, data exposure)
- Code style and maintainability
- Test coverage for changed code paths
- Performance implications for hot paths

Use the available tools to fetch context as needed.
Flag blocking issues separately from suggestions.
"""
```

The workflow approach requires you to anticipate every branching condition. The
policy approach tells the model what matters and trusts it to figure out the
sequencing. When the model encounters an unexpected situation—a file type you
didn't anticipate, a security issue that requires additional context—it can
adapt. A workflow would need another edge.

## Why This Matters Now

Graph-based orchestration made sense when models needed heavy scaffolding. Early
LLMs struggled with multi-step reasoning, lost track of goals, and needed
explicit guidance at every turn.

That's changing. Modern models can:

- Plan multi-step approaches and revise them when they hit obstacles
- Recognize when they need more information and request it
- Self-correct when initial approaches don't work
- Handle ambiguity without explicit disambiguation steps

This doesn't mean workflows are never useful. It means the default should flip:
**err on the side of giving the agent more degrees of freedom, and only
constrain where you have evidence that constraints help.**

## WINK's Approach

WINK operationalizes policies through three primitives:

1. **Prompts**: A tree of typed sections that describe capabilities, context,
   and constraints. The model reads this and decides what to do.

1. **Tools**: Typed functions the model can call. The prompt explains when and
   why to use them. The model decides if and when to invoke them.

1. **Sessions**: Immutable state that captures everything—what was sent, what
   tools ran, what changed. No hidden state, full auditability.

There's no graph. The prompt *is* the agent. The model's context window is the
execution environment.

```python nocheck
# The prompt describes policy, not procedure
search_section = MarkdownSection(
    title="Search Capabilities",
    key="search",
    template="""
    You have access to semantic and keyword search over the codebase.

    Use search when you need to:
    - Find implementations of specific functionality
    - Locate usages of a function or type
    - Understand how existing code handles similar cases

    Prefer targeted queries over broad ones. If initial results aren't
    helpful, refine your query rather than giving up.
    """,
    tools=(semantic_search, keyword_search),
)
```

The model decides when to search, what to search for, and how to interpret
results. You've described the policy ("use search when you need context, prefer
targeted queries"), not the workflow ("first search for X, then if results
contain Y, search for Z").

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
constraint ("only Python files in src/") is enforced at the boundary. You don't
need a graph edge that says "if writing file, check if it's in src/".

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
can't use. No workflow edge needed—the policy adapts to context.

**Constrain with design by contract:**

```python nocheck
@require(lambda params: len(params.query) >= 3, "query too short")
@ensure(lambda result: len(result.value.matches) <= 100, "too many results")
def search_handler(params: SearchParams, *, context: ToolContext) -> ToolResult[SearchResult]:
    ...
```

Preconditions and postconditions make invariants explicit. When violated, you
know exactly what assumption broke. No defensive graph edges required.

## When Workflows Still Make Sense

Policies aren't universally better. Use explicit workflow control when:

**Hard sequencing requirements exist.** If step B literally cannot run until
step A completes (not "shouldn't"—"cannot"), encode that. Example: you must
create a database migration before running it.

**Compliance requires auditability of the path.** Some domains require proving
that specific steps happened in specific order. A workflow graph provides that
proof directly.

**The model consistently fails at planning.** If you've tried policy-based
approaches and the model reliably makes bad sequencing decisions for your
specific task, add structure. But try the simpler approach first.

**You're orchestrating multiple agents.** When different models or agent
instances need to coordinate, explicit handoffs often work better than hoping
they'll figure it out from shared context.

## Concept Mapping

If you're coming from LangGraph, here's how concepts translate:

| LangGraph | WINK | Notes |
| --- | --- | --- |
| Graph | `PromptTemplate` | Tree of sections, not edges |
| Node | Tool handler | Called when model decides to |
| Edge | (implicit) | Model decides sequencing |
| Conditional edge | `enabled()` predicate | Section appears or doesn't |
| State | `Session` | Immutable, event-driven |
| Checkpointing | `snapshot()`/`restore()` | Full state capture |
| LangSmith | Debug UI + events | Built-in, no external service |

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
# Test prompt rendering
def test_prompt_includes_search_instructions():
    rendered = prompt.bind(params).render(session=session)
    assert "semantic search" in rendered.text.lower()

# Test tool handlers directly
def test_search_returns_results():
    result = search_handler(SearchParams(query="auth"), context=fake_context)
    assert result.success
    assert len(result.value.matches) > 0

# Test reducers as pure functions
def test_add_finding_appends():
    state = (Finding(text="first"),)
    event = AddFinding(text="second")
    op = finding_reducer(state, event)
    assert isinstance(op, Append)
```

No mocking graphs. No integration tests pretending to be unit tests. Determinism
makes testing straightforward.

### First-Class Debugging

Every prompt, tool call, and state change is captured:

```bash
wink debug ./debug_bundles/
```

See exactly what was sent, what tools ran, how state evolved. When something
goes wrong, you have the full picture—not just logs, but the actual artifacts.

### Zero-Dependency Serialization

No Pydantic required. WINK's `serde` module uses stdlib dataclasses:

```python nocheck
from weakincentives.serde import parse, dump, schema

user = parse(User, {"name": "Ada", "age": "39"})  # Coerces types
payload = dump(user)  # JSON-compatible dict
user_schema = schema(User)  # JSON Schema for tools
```

Minimal dependencies, predictable behavior.

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
    content_hash="abc123...",  # Must match current section
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

## Migration Example

**LangGraph style—workflow as graph:**

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

**WINK style—policy as prompt:**

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
```

The WINK version describes what matters. The model decides when to search, how
many times, and when it has enough context to analyze. No edges to maintain.

## Side-by-Side Comparison

| Aspect | LangGraph (Workflow) | WINK (Policy) |
| --- | --- | --- |
| Control flow | Explicit graph edges | Model decides from prompt |
| Sequencing | Predefined paths | Emergent from context |
| Adding capabilities | New nodes + edges | New section + tools |
| Handling edge cases | More conditional edges | Model adapts |
| Testing | Mock graph execution | Test components directly |
| Debugging | Step through graph | Inspect prompts + events |
| Constraints | Edges that block paths | Tool validation + contracts |

## Making the Transition

1. **Start with your tools.** Convert LangGraph tools to WINK's typed pattern
   with explicit params and results. This works regardless of orchestration.

1. **Identify true constraints.** Which edges in your graph represent actual
   requirements ("must happen before") vs. guidance ("usually do this first")?
   Keep the former, convert the latter to prompt instructions.

1. **Write policy, not procedure.** Describe what the agent should accomplish
   and what tools are available. Let it figure out sequencing.

1. **Add structure only when needed.** If the model consistently makes poor
   decisions about ordering, add conditional sections or explicit instructions.
   But start permissive.

1. **Test the components.** Verify tool handlers work correctly. Verify prompts
   render as expected. Verify reducers produce correct state. These tests don't
   need the model.

## Next Steps

**Understand the philosophy:**

- [Philosophy](philosophy.md): The "weak incentives" approach in depth

**Learn the primitives:**

- [Prompts](prompts.md): Typed, composable prompt trees
- [Tools](tools.md): Typed tool contracts and handlers
- [Sessions](sessions.md): Immutable state with event-driven reducers

**Build and test:**

- [Quickstart](quickstart.md): Get a working agent running
- [Testing](testing.md): Test without models
- [Debugging](debugging.md): Debug UI and event inspection

**Production:**

- [Adapters](adapters.md): Connect to OpenAI, LiteLLM, or Claude
- [Evaluation](evaluation.md): Systematic agent evaluation

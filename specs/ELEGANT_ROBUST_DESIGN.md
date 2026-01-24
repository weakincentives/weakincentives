# Elegant Robust Design: A Vision for WINK

> *"Perfection is achieved, not when there is nothing more to add, but when there
> is nothing left to take away."* — Antoine de Saint-Exupéry

This document analyzes WINK through the lens of Japanese design philosophy
(sophisticated simplicity) and Toyota production principles (industrial
durability). The goal: identify gaps, anticipate objections, and chart future
development that embodies both qualities.

______________________________________________________________________

## Design Philosophy Mapping

### Japanese Aesthetics → Software Principles

| Concept | Meaning | WINK Application |
|---------|---------|------------------|
| **Kanso** | Simplicity, elimination of clutter | One responsibility per layer; sections bundle related concerns |
| **Fukinsei** | Asymmetry, irregularity | Policies over rigid workflows; embrace model reasoning |
| **Shibui** | Subtle, unobtrusive beauty | APIs that feel obvious in hindsight |
| **Shizen** | Naturalness without pretense | Types as documentation; code that reads like intent |
| **Datsuzoku** | Freedom from convention | Prompts-as-agents vs string templates |
| **Seijaku** | Tranquility, silence | Pure functions; no hidden side effects |
| **Ma** | Negative space, pause | Progressive disclosure; show only what matters now |

### Toyota Production System → Software Reliability

| Principle | Meaning | WINK Application |
|-----------|---------|------------------|
| **Jidoka** | Stop when defects detected | Design-by-contract always enabled; fail fast |
| **Poka-yoke** | Error-proofing | Type system prevents invalid states |
| **Andon** | Visual problem signaling | Debug bundles; session query tools |
| **Genchi Genbutsu** | Go see for yourself | Deterministic rendering; inspect actual prompts |
| **Kaizen** | Continuous improvement | Hash-validated overrides; safe iteration |
| **Heijunka** | Level production | Budgets and deadlines prevent runaway costs |
| **Nemawashi** | Building consensus | Policies declare intent; agent finds path |

______________________________________________________________________

## Part I: Gaps in the Current Design

### Gap 1: The Async Void

**Current state:** Synchronous everywhere. Thread-safe via RLock, but no async
providers, no async reducers, no concurrent tool execution.

**Why it matters:** Production agents often need I/O parallelism—fetching from
multiple APIs, parallel file operations, streaming responses. The GIL limits
threading benefits; async is the Python answer.

**The elegant solution:**

```python
# Dual-mode protocol: sync by default, async when needed
class ProviderAdapter(Protocol[OutputT]):
    def evaluate(self, ...) -> PromptResponse[OutputT]: ...

    # Optional async variant
    async def aevaluate(self, ...) -> PromptResponse[OutputT]:
        return self.evaluate(...)  # Default: wrap sync
```

Add `AsyncSession` that wraps reducers in async context. Tool handlers gain
optional `async def` signature. This follows the httpx pattern: sync API for
simplicity, async for performance—same interface.

**Toyota parallel:** Just-in-time delivery requires coordination without
blocking the line. Async enables this for I/O-bound agents.

______________________________________________________________________

### Gap 2: Missing Observability Primitives

**Current state:** Debug bundles capture post-mortem snapshots. RunContext
provides correlation IDs. But no live observability during execution.

**Missing pieces:**

- No OpenTelemetry integration (traces, spans, metrics)
- No structured logging with context propagation
- No live dashboard for long-running agents
- Token usage tracked but not exposed as metrics

**The elegant solution:**

```python
@dataclass(frozen=True)
class Span:
    """Lightweight span for tracing without vendor lock-in."""
    name: str
    trace_id: str
    parent_id: str | None
    start_time: datetime
    attributes: Mapping[str, str | int | float | bool]

# Auto-instrumentation via decorators
@traced("tool.execute")
def execute_tool(tool: Tool, params: P, context: ToolContext) -> ToolResult[R]:
    ...
```

Provide a `TracingProvider` protocol with `NoopTracer` default. Users plug in
OpenTelemetry, Datadog, or custom backends. Spans auto-created for:

- Prompt renders
- Provider calls
- Tool executions
- Reducer dispatches

**Japanese parallel (Andon):** The factory floor has boards showing status at a
glance. Agents need the same: where are we, what's happening, is anything stuck?

______________________________________________________________________

### Gap 3: No Graceful Degradation Path

**Current state:** Tools either succeed or fail (with rollback). No partial
success, no degraded operation modes.

**Why it matters:** Real systems face flaky APIs, rate limits, transient
failures. "All or nothing" is clean but brittle.

**The elegant solution:**

```python
@dataclass(frozen=True)
class ToolResult(Generic[T]):
    success: bool
    value: T | None
    message: str
    degraded: bool = False  # New: partial success flag
    fallback_used: bool = False  # New: indicate reduced capability

    @staticmethod
    def degraded(value: T, message: str) -> "ToolResult[T]":
        """Partial success—result usable but not ideal."""
        return ToolResult(True, value, message, degraded=True)
```

Policies can check `degraded` to decide if retries are warranted. LLM sees
"result obtained but may be incomplete" vs hard failure.

**Toyota parallel (Jidoka):** Stop on defects, but distinguish between "stop the
line" defects and "flag for inspection" issues. Not all problems are equal.

______________________________________________________________________

### Gap 4: Resource Graph Opacity

**Current state:** Resources resolve lazily. Cycle detection works. But no
visibility into the dependency graph.

**Missing:**

- No way to visualize resource dependencies
- Errors don't suggest available bindings
- No eager validation of entire graph before execution

**The elegant solution:**

```python
class ResourceRegistry:
    def validate_graph(self) -> GraphValidation:
        """Eagerly resolve entire graph, return validation result."""
        ...

    def render_graph(self) -> str:
        """ASCII art of dependency graph for debugging."""
        ...

    def suggest_bindings(self, protocol: type[T]) -> Sequence[type]:
        """Return protocols that could satisfy this dependency."""
        ...
```

Add `wink resources visualize` CLI command. Print graph on startup in debug
mode.

**Japanese parallel (Shizen):** A well-designed tool reveals its structure
naturally. The dependency graph is structure—make it visible.

______________________________________________________________________

### Gap 5: No Memory/Retrieval Integration

**Current state:** Sessions track execution state. No long-term memory, no
vector retrieval, no conversation history beyond current session.

**Why it matters:** Background agents need context across sessions. "Remember
what we discussed yesterday" is a common requirement.

**The elegant solution (deliberate minimalism):**

```python
class MemoryProvider(Protocol):
    """Narrow protocol for retrieval—not a full RAG system."""

    def store(self, key: str, content: str, metadata: Mapping[str, str]) -> None:
        """Store content with metadata for later retrieval."""
        ...

    def retrieve(self, query: str, limit: int = 5) -> Sequence[MemoryEntry]:
        """Retrieve relevant entries by semantic similarity."""
        ...

    def forget(self, key: str) -> None:
        """Remove specific memory entry."""
        ...
```

WINK provides the protocol and a simple file-backed implementation. Users bring
their own vector store. Retrieval is a section that injects context—following
"prompts as agents" philosophy.

**Japanese parallel (Ma):** Don't fill every space. Provide the opening for
memory; let users decide what belongs there.

______________________________________________________________________

### Gap 6: Formal Verification Remains Aspirational

**Current state:** TLA+ specs exist for mailbox and slices. Framework for
embedding specs. But adoption limited, no CI validation, counterexamples don't
translate to Python.

**The gap:** Beautiful infrastructure, underutilized. Like having a precision
manufacturing facility and using it for rough carpentry.

**The elegant solution:**

1. **CI integration:** `make formal-check` runs TLC on all embedded specs
1. **Coverage tracking:** Which state machines are specified vs not?
1. **Counterexample translation:** When TLC finds violation, generate Python
   test case that reproduces it
1. **Gradual adoption:** Start with critical paths (session state, tool
   rollback), expand over time

```python
@formal_spec("""
---- MODULE ToolRollback ----
VARIABLES session_state, fs_state, tool_result

Rollback ==
  tool_result = "failure" =>
    /\ session_state' = session_state_before
    /\ fs_state' = fs_state_before
====
""")
def execute_tool_with_rollback(...):
    ...
```

**Toyota parallel:** Toyota doesn't just have quality systems—they use them
relentlessly. Formal verification should be exercised, not displayed.

______________________________________________________________________

### Gap 7: Error Taxonomy is Thin

**Current state:** 8 public exceptions. Adapter exceptions scattered. No
hierarchy or categorization.

**Why it matters:** Error handling is interface design. Callers need to know:

- Is this retryable?
- Is this a bug in my code or external failure?
- What context helps debugging?

**The elegant solution:**

```python
class WinkError(Exception):
    """Base for all WINK errors."""
    retryable: bool = False

class ConfigurationError(WinkError):
    """Errors in setup—fix your code."""

class ExecutionError(WinkError):
    """Errors during runtime—may be transient."""
    retryable: bool = True

class ProviderError(ExecutionError):
    """Errors from LLM providers."""
    provider: str
    status_code: int | None

class ContractViolation(WinkError):
    """Design-by-contract failures."""
    contract_type: Literal["require", "ensure", "invariant"]
    expression: str
```

Clear hierarchy. `retryable` flag enables generic retry logic. Provider errors
carry context for debugging.

**Japanese parallel (Shibui):** Errors should be as carefully designed as happy
paths. Subtle, useful, never surprising.

______________________________________________________________________

### Gap 8: Testing Helpers are Scattered

**Current state:** Mock adapters created ad-hoc per test file. `FilesystemValidationSuite` exists but pattern not replicated elsewhere.

**The elegant solution:**

```python
# tests/helpers/testing.py
class MockAdapter(ProviderAdapter[T]):
    """Configurable mock for testing prompt flows."""

    def __init__(
        self,
        responses: Sequence[PromptResponse[T]] | Callable[..., PromptResponse[T]],
        tool_results: Mapping[str, ToolResult[Any]] | None = None,
    ): ...

    @property
    def calls(self) -> Sequence[RenderedPrompt]:
        """All prompts this adapter received."""
        ...

class SessionFixture:
    """Pre-configured session with common slices."""

    @classmethod
    def with_plan(cls, steps: Sequence[str]) -> Session:
        ...

class PromptAssertion:
    """Fluent assertions for prompt content."""

    def contains_tool(self, name: str) -> Self: ...
    def has_section(self, key: str) -> Self: ...
    def matches_snapshot(self, path: Path) -> Self: ...
```

Centralized, composable, documented. Tests become declarative.

**Toyota parallel:** Standardized work. Every test uses the same well-designed
fixtures, reducing variation and defects.

______________________________________________________________________

## Part II: Anticipated Objections

### Objection 1: "This is over-engineered for simple use cases"

**The criticism:** All this machinery (sections, reducers, policies, contracts)
is overkill for a chatbot or simple automation.

**The response:**

This is a valid tension. WINK optimizes for **unattended background agents**
where correctness matters and debugging is hard. For a simple CLI chatbot,
LangChain or raw API calls are fine.

**The design response:**

```python
# Simple mode for simple needs
from weakincentives import quick

response = quick.prompt(
    "Summarize this document",
    tools=[quick.read_file],  # Pre-built tools
    model="claude-3-5-sonnet",
)
```

A `quick` module with sensible defaults. No sections, no sessions, no policies
unless you want them. Graduate to full API when complexity warrants.

**Japanese parallel (Fukinsei):** Asymmetry is intentional. Simple things should
be simple; complex things should be possible.

______________________________________________________________________

### Objection 2: "Design-by-contract always enabled is too aggressive"

**The criticism:** Contracts in production add overhead and may fail on edge
cases the tests didn't cover.

**The response:**

This is the Toyota philosophy made explicit: **stop when defects are detected**.
A contract violation in production means your invariants are wrong or your code
is wrong. Either way, you want to know immediately, not after corrupted state
propagates.

**The design mitigations already in place:**

- `dbc_suspended()` context for known edge cases
- Contracts are predicates, not complex assertions—low overhead
- Clear error messages identify exact violation

**What we could add:**

- Contract violation telemetry (count, location, frequency)
- "Warn mode" for gradual adoption in legacy code
- Performance profiling to identify expensive contracts

______________________________________________________________________

### Objection 3: "Pure reducers don't scale to complex state"

**The criticism:** Redux-style state works for UI, but agent state can be
complex—nested objects, graphs, large histories.

**The response:**

Valid concern. Mitigations:

1. **Structural sharing:** Immutable tuples reuse unchanged portions
1. **Slice partitioning:** Each concern in its own slice, composed at read time
1. **JSONL backing:** Large histories stream to disk, not held in memory
1. **Projection queries:** `session[T].where(predicate)` avoids full scans

**What we could add:**

- Lazy slices that only load relevant portions
- Indexed queries for common access patterns
- Automatic archival of old entries

**Toyota parallel (Heijunka):** Level the load. Don't hold everything in memory;
stream what you can, index what you query.

______________________________________________________________________

### Objection 4: "Transactional rollback doesn't work for external effects"

**The criticism:** You can roll back session state and filesystem, but what
about API calls, database writes, sent emails?

**The response:**

Correct—WINK doesn't solve distributed transactions. This is intentional scope
limitation.

**The design philosophy:**

- Tools should be idempotent where possible
- External effects should be in dedicated "commit" tools (explicit side-effect
  boundary)
- Policies can require confirmation before irreversible actions

**What we could add:**

- `@idempotent` decorator with automatic deduplication
- Saga pattern support for multi-step external operations
- "Dry run" mode that captures intended effects without executing

```python
@tool
@idempotent(key=lambda p: f"send_email:{p.recipient}:{p.subject}")
def send_email(params: EmailParams, *, context: ToolContext) -> ToolResult[str]:
    """Idempotent email send—same key won't re-send."""
    ...
```

______________________________________________________________________

### Objection 5: "Progressive disclosure adds complexity for uncertain benefit"

**The criticism:** Managing visibility states, expansion tools, multiple render
passes—is the context savings worth the cognitive overhead?

**The response:**

For small prompts, no. For large agents with many tools and extensive
context, absolutely.

**The data point:** Claude's context is 200K tokens. A complex agent with full
workspace access easily hits 50K+ tokens in context. Progressive disclosure
keeps initial prompts focused (5-10K), expanding only when the model indicates
need.

**What we could add:**

- Metrics showing expansion patterns (which sections expand most?)
- Auto-tuning that adjusts initial visibility based on task type
- A/B testing visibility strategies via experiments

______________________________________________________________________

### Objection 6: "The learning curve is too steep"

**The criticism:** 44 specs, 27 guides, design-by-contract, reducers,
policies—there's a lot to learn.

**The response:**

Fair. The documentation is comprehensive but potentially overwhelming.

**The design response:**

1. **Learning paths:** Already have beginner → production → advanced tracks
1. **Interactive tutorial:** Add `wink tutorial` CLI that walks through concepts
1. **Starter templates:** `wink new --template=simple-agent` scaffolds minimal
   project
1. **Error messages as teachers:** When you hit an error, message includes link
   to relevant guide

```bash
$ wink new my-agent --template=code-reviewer
Creating agent from template: code-reviewer
  ✓ src/my_agent/prompt.py (PromptTemplate with sections)
  ✓ src/my_agent/tools.py (Tool definitions)
  ✓ tests/test_prompt.py (Prompt rendering tests)
  ✓ Makefile (check, test, format targets)

Next steps:
  cd my-agent
  make check  # Verify everything works
  wink tutorial  # Interactive walkthrough
```

**Japanese parallel (Omotenashi):** Anticipate needs. A new user shouldn't have
to read 44 specs to get started.

______________________________________________________________________

### Objection 7: "Why not just use LangGraph/CrewAI/AutoGen?"

**The criticism:** Established frameworks have larger communities, more
integrations, battle-tested code.

**The response:**

Different optimization targets:

| Framework | Optimizes For |
|-----------|---------------|
| LangGraph | Graph workflows, complex multi-agent choreography |
| CrewAI | Role-based multi-agent collaboration |
| AutoGen | Conversational multi-agent patterns |
| **WINK** | Single-agent correctness, determinism, portability |

WINK is not trying to be a workflow engine. It's trying to be the best way to
define a single agent's behavior such that:

- You can test it deterministically
- You can port it across providers
- You can inspect exactly what happened
- You can iterate safely

Use WINK for the agent definition, use something else for orchestration if
needed.

______________________________________________________________________

## Part III: Future Development Areas

### Area 1: The Skill Ecosystem

**Vision:** Skills as the npm of agent capabilities.

**Current state:** Skills exist (SKILL.md format, mounting, validation). But no
discovery, no sharing, no versioning.

**The elegant future:**

```bash
$ wink skill search "code review"
  code-review-python (v2.1.0) - Python code review with style checking
  code-review-security (v1.3.0) - Security-focused code analysis
  code-review-docs (v0.9.0) - Documentation coverage checking

$ wink skill add code-review-python
Added code-review-python@2.1.0 to wink.toml

$ wink skill update
Updating skills...
  code-review-python: 2.1.0 → 2.2.0 (security patch)
```

Skills as versioned packages. Registry (centralized or git-based). Dependency
resolution. Security scanning.

**Toyota parallel:** Supplier ecosystem. Toyota doesn't make every part—they
have trusted suppliers with quality standards. Skills are agent suppliers.

______________________________________________________________________

### Area 2: Agent Composition Patterns

**Vision:** Compose agents without complex orchestration frameworks.

**Current state:** Single agent focus. MainLoop handles one agent.

**The elegant future:**

```python
# Pipeline: output of one feeds input of next
pipeline = Pipeline([
    planning_agent,      # Breaks task into steps
    execution_agent,     # Executes each step
    verification_agent,  # Checks results
])

# Ensemble: multiple agents, best answer wins
ensemble = Ensemble(
    agents=[fast_agent, thorough_agent],
    selector=lambda results: max(results, key=lambda r: r.confidence),
)

# Handoff: agent decides when to delegate
@tool
def handoff_to_specialist(params: HandoffParams, *, context: ToolContext):
    """Let current agent hand off to a specialist."""
    return context.spawn(specialist_agents[params.domain])
```

Composition primitives, not a full orchestration language. Keep it simple.

**Japanese parallel (Kanso):** Simplicity. Don't build a workflow engine—build
composable pieces that fit together naturally.

______________________________________________________________________

### Area 3: Prompt Optimization Loop

**Vision:** Systematic prompt improvement with measurable outcomes.

**Current state:** Experiments provide A/B testing. Evals framework exists. But
no closed-loop optimization.

**The elegant future:**

```python
optimizer = PromptOptimizer(
    baseline=current_prompt,
    eval_suite=code_review_evals,
    search_space=SearchSpace(
        instruction_variants=["detailed", "concise", "examples"],
        tool_descriptions=["minimal", "verbose"],
        section_order=permutations(["context", "task", "constraints"]),
    ),
)

# Run optimization
result = optimizer.optimize(budget=Budget(max_evals=100))
print(f"Best variant: {result.best_config}")
print(f"Improvement: {result.baseline_score} → {result.best_score}")

# Apply winning variant
optimized_prompt = result.apply_to(current_prompt)
```

Bayesian optimization over prompt space. Automatic experiment tracking.
Statistical significance testing.

______________________________________________________________________

### Area 4: Native Streaming Support

**Vision:** Stream tokens, tool calls, and state updates in real-time.

**Current state:** Batch responses only. No streaming.

**The elegant future:**

```python
async for event in adapter.stream(prompt, session):
    match event:
        case TokenEvent(token):
            print(token, end="", flush=True)
        case ToolCallStart(tool_name):
            print(f"\n[Calling {tool_name}...]")
        case ToolCallComplete(tool_name, result):
            print(f"[{tool_name} complete]")
        case StateUpdate(slice_type, op):
            # Real-time state visibility
            ...
```

Event-driven streaming. Tool calls interleaved with tokens. State updates
visible as they happen.

**Toyota parallel (Andon):** See problems as they happen, not after the shift
ends.

______________________________________________________________________

### Area 5: Cost Intelligence

**Vision:** Understand and optimize agent costs automatically.

**Current state:** Budget tracking exists. Token counts captured.

**The elegant future:**

```python
cost_analyzer = CostAnalyzer(session)

# Where did tokens go?
breakdown = cost_analyzer.token_breakdown()
# → context: 45%, tool_results: 30%, instructions: 15%, output: 10%

# What's expensive?
hotspots = cost_analyzer.identify_hotspots()
# → "workspace_context section uses 12K tokens; consider summarization"

# Automatic suggestions
suggestions = cost_analyzer.suggest_optimizations()
# → "Enable progressive disclosure for file_browser section"
# → "Cache expensive tool results with TTL"
```

Cost as first-class metric. Automatic optimization suggestions. Budget
forecasting.

______________________________________________________________________

### Area 6: Semantic Versioning for Prompts

**Vision:** Prompts have versions with clear compatibility guarantees.

**Current state:** Hash-based override validation. No semantic versioning.

**The elegant future:**

```python
@prompt_template(
    ns="acme.code-review",
    version="2.1.0",  # semver
    compatible_with=">=2.0.0",  # backward compatibility
)
class CodeReviewPrompt(PromptTemplate[ReviewResult]):
    ...

# Version checking
if not is_compatible(saved_session.prompt_version, current_prompt.version):
    raise IncompatibleVersionError(
        f"Session was created with {saved_session.prompt_version}, "
        f"but current prompt is {current_prompt.version}"
    )
```

Version prompts like APIs. Breaking changes get major bumps. Sessions validate
version compatibility.

______________________________________________________________________

### Area 7: Multi-Modal Native Support

**Vision:** Images, audio, video as first-class prompt content.

**Current state:** Text-focused. No explicit multi-modal support.

**The elegant future:**

```python
class ImageSection(Section):
    """Section that renders images in prompt."""

    def render(self, context: RenderContext) -> SectionContent:
        return SectionContent(
            images=[
                Image(path=self.image_path, detail="high"),
            ],
            text=f"Analyze the image above and {self.instruction}",
        )

# In tools
@tool
def capture_screenshot(
    params: ScreenshotParams,
    *,
    context: ToolContext
) -> ToolResult[Image]:
    """Capture screenshot and return as image for analysis."""
    ...
```

Images, audio, video in sections. Multi-modal tool results. Provider adapters
handle encoding.

______________________________________________________________________

## Part IV: The Aesthetic Vision

### What "Elegant" Means for WINK

**1. APIs that disappear**

The best API is one you forget you're using. You think in terms of your domain
(prompts, tools, agents), not in terms of the framework.

```python
# This reads like English, not framework code
prompt = (
    Prompt(code_review_template)
    .bind(files=changed_files, standards=team_standards)
    .with_deadline(minutes=5)
)

with prompt.resources:
    result = adapter.evaluate(prompt, session)
```

**2. Errors that teach**

Every error message should help you fix the problem:

```
ContractViolation: @require failed in `withdraw(amount)`
  Expression: amount >= 0
  Actual: amount = -50

  This precondition ensures withdrawal amounts are non-negative.
  If you need to handle refunds, use `credit(amount)` instead.

  See: https://wink.dev/guides/contracts#require
```

**3. Defaults that work**

Sensible defaults mean most users never configure anything:

- Design-by-contract: enabled (safety first)
- Deadline: 5 minutes (prevent runaway)
- Budget: model-appropriate limits
- Tools: sandboxed by default

**4. Escape hatches that exist**

When defaults don't fit, overrides are clean:

```python
with dbc_suspended():  # Temporarily disable contracts
    ...

prompt.with_deadline(None)  # No deadline
prompt.with_budget(unlimited=True)  # No budget limit
```

______________________________________________________________________

### What "Robust" Means for WINK

**1. Fail fast, fail loud**

Contracts catch bugs at the source. Type errors caught at definition time.
Invalid states unrepresentable.

**2. Fail safe**

Transactional tools roll back on failure. Resources clean up automatically.
Deadlines prevent infinite loops.

**3. Fail observable**

Debug bundles capture everything. Session queries find specific events. Traces
connect distributed operations.

**4. Fail recoverable**

Snapshots enable restore. JSONL slices survive crashes. Idempotent tools enable
retry.

______________________________________________________________________

## Conclusion: The Path Forward

WINK already embodies many qualities of elegant, robust design:

- **Type safety** prevents invalid states
- **Contracts** catch bugs early
- **Pure reducers** ensure predictable state
- **Transactional tools** enable safe recovery
- **Deterministic rendering** enables testing

To achieve Toyota-level durability with Japanese elegance, focus on:

1. **Complete the observability story** (tracing, metrics, live dashboards)
1. **Build the skill ecosystem** (discovery, versioning, quality gates)
1. **Add async without complexity** (dual-mode APIs, same mental model)
1. **Make formal verification real** (CI integration, counterexample translation)
1. **Lower the learning curve** (tutorials, templates, teaching errors)

The philosophy is sound. The architecture is clean. The execution continues.

> *"The details are not the details. They make the design."* — Charles Eames

______________________________________________________________________

## Appendix: Implementation Priorities

### Phase 1: Foundation Strengthening (High Impact, Lower Effort)

1. **Error taxonomy** - Clear hierarchy, retryable flag, helpful messages
1. **Testing helpers** - Centralized mocks, fixtures, assertions
1. **Resource graph visualization** - CLI command, debug output
1. **Graceful degradation** - `degraded` flag on ToolResult

### Phase 2: Observability (High Impact, Medium Effort)

1. **Tracing protocol** - OpenTelemetry-compatible, pluggable backend
1. **Structured logging** - Context propagation, standard fields
1. **Cost analyzer** - Token breakdown, optimization suggestions
1. **Live metrics** - Prometheus/StatsD export

### Phase 3: Ecosystem (Medium Impact, Higher Effort)

1. **Skill registry** - Discovery, versioning, security scanning
1. **`wink new` templates** - Scaffolding for common patterns
1. **Interactive tutorial** - `wink tutorial` CLI
1. **Prompt versioning** - Semver, compatibility checking

### Phase 4: Advanced Features (Variable Impact, Higher Effort)

1. **Async support** - Dual-mode adapters, async session
1. **Streaming** - Token events, tool call events
1. **Composition primitives** - Pipeline, Ensemble, Handoff
1. **Formal verification CI** - TLC integration, counterexample translation

______________________________________________________________________

*Document version: 1.0*
*Created: 2026-01-24*
*Philosophy: Elegant simplicity, industrial durability*

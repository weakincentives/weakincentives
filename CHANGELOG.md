# Changelog

Release highlights for weakincentives.

## Unreleased

### Evaluation Framework

New `weakincentives.evals` module provides a minimal evaluation framework built
on MainLoop for testing agent outputs:

```python
from weakincentives.evals import (
    EvalLoop, EvalLoopCompleted, Sample, exact_match, load_jsonl
)

# Build dataset programmatically
dataset = tuple(
    Sample(id=str(i), input=f"What is {a}+{b}?", expected=str(a+b))
    for i, (a, b) in enumerate([(1, 1), (2, 3)])
)

# Or load from JSONL
dataset = load_jsonl(Path("tests/fixtures/qa.jsonl"), str, str)

# Create EvalLoop with MainLoop, evaluator, and bus
eval_loop = EvalLoop(loop=main_loop, evaluator=exact_match, bus=bus)

# Direct execution
report = eval_loop.execute(dataset)
print(f"Pass rate: {report.pass_rate:.1%}")

# Or event-driven usage (for cluster deployment)
bus.subscribe(EvalLoopCompleted, handle_completed)
bus.publish(EvalLoopRequest(dataset=dataset))
```

Features:

- **Core types**: `Sample`, `Score`, `EvalResult`, `EvalReport`
- **Event-driven orchestration**: `EvalLoop` mirrors `MainLoop` for cluster deployment
- **Request/response events**: `EvalLoopRequest`, `EvalLoopCompleted`, `EvalLoopFailed`
- **Built-in evaluators**: `exact_match`, `contains`, `all_of`, `any_of`
- **Custom evaluators**: `within_tolerance`, `json_subset`
- **JSONL loading**: `load_jsonl` for loading datasets from files
- **LLM-as-judge**: `llm_judge` factory for subjective criteria evaluation
- **Progress events**: `SampleEvaluated` events published to EventBus

### First-Class Resource Injection

You can now pass custom resources into `adapter.evaluate()` and
`MainLoop.execute()`, making tools cleaner, more testable, and portable.

```python
from weakincentives.prompt.tool import ResourceRegistry
from myapp.http import HTTPClient

# Build resource registry with your dependencies
http_client = HTTPClient(base_url="https://api.example.com")
resources = ResourceRegistry.build({HTTPClient: http_client})

# Pass to adapter - merged with workspace resources (e.g., filesystem)
response = adapter.evaluate(prompt, session=session, resources=resources)

# Or at the MainLoop level
config = MainLoopConfig(resources=resources)
loop = MyLoop(adapter=adapter, bus=bus, config=config)
response, session = loop.execute(request)
```

**Key features:**

- `ResourceRegistry.merge()` combines registries with the second taking
  precedence on conflicts
- User-provided resources override workspace defaults (e.g., custom filesystem)
- Tool handlers access resources via `context.resources.get(ResourceType)`
- `MainLoopConfig`, `MainLoopRequest`, and `MainLoop.execute()` all accept
  `resources` parameter
- All adapters (OpenAI, LiteLLM, Claude Agent SDK) support resource injection

## v0.16.0 - 2025-12-21

### Transactional Tool Execution

Tool calls in WINK are now **transactional by default**. When a tool fails or
is aborted, both session state and filesystem changes are automatically rolled
back to their pre-invocation state. No more partial failures corrupting your
agent's working memory or leaving orphaned files on disk.

```python
from weakincentives.runtime import ExecutionState

state = ExecutionState(session=session, resources=resources)

# Tool execution is atomic - on failure, state is restored automatically
result = runner.execute(tool_call, context=context)
# If result.success is False, session and filesystem are unchanged
```

This release introduces several new primitives that work together to provide
these guarantees:

- **ExecutionState**: Unified root for all mutable runtime state (session +
  resources). Provides `snapshot()` and `restore()` for coordinated state
  capture across all snapshotable resources.

- **CompositeSnapshot**: Captures session slices and all snapshotable resources
  (e.g., filesystem) in a single point-in-time snapshot.

- **Snapshotable protocol**: Runtime-checkable protocol for objects supporting
  `snapshot(tag=...)` and `restore(snapshot)` operations. Both `Session` and
  `SnapshotableFilesystem` satisfy this protocol.

- **SlicePolicy enum**: Classifies session slices for selective rollback:

  - `STATE`: Working state restored on tool failure (Plan, VisibilityOverrides)
  - `LOG`: Append-only records preserved during restore (ToolInvoked events)

- **ToolRunner**: Shared tool execution with identical transaction semantics
  across all adapters (OpenAI, LiteLLM, Claude Agent SDK). Automatic snapshot
  before execution and restore on failure.

- **Filesystem snapshots**: `HostFilesystem` uses git-based copy-on-write for
  efficient snapshots. `InMemoryFilesystem` uses Python structural sharing.
  Both implement `SnapshotableFilesystem` for coordinated rollback.

New error classes for execution state operations:

- `ExecutionStateError`: Base for execution state errors
- `SnapshotMismatchError`: Snapshot incompatible with current state
- `RestoreFailedError`: Failed to restore from snapshot

### Added

- **`read_section` tool for summarized sections.** Models can now retrieve the
  full markdown content of a summarized section without permanently changing
  visibility state. This is a read-only operation—the section remains
  summarized in subsequent turns. Sections with tools still use `open_sections`
  for expansion; sections without tools use the new `read_section` for
  token-efficient peek access.

- **Workspace digest summary.** `WorkspaceDigest` now includes a `summary`
  field with a short 1-paragraph overview. `WorkspaceDigestSection` defaults
  to `SUMMARY` visibility, enabling token-efficient prompts while preserving
  full digest access via `read_section`.

- **WINK_GUIDE.md.** Comprehensive guide for engineers building deterministic,
  typed, safe background agents. Covers philosophy, quickstart, prompts, tools,
  sessions, adapters, orchestration, progressive disclosure, prompt overrides,
  workspace tools, debugging, testing, and API reference.

- **100% type coverage enforcement.** `make check` now includes
  `pyright --verifytypes` to ensure all exported symbols have complete type
  annotations. Type coverage is enforced at 100% (2417+ symbols with known
  types).

- **`is_dataclass_instance` helper** exported from `weakincentives.types` for
  consistent dataclass instance checking with proper `TypeGuard` narrowing.

### Changed

- **Internal refactoring.** The contrib tools modules (filesystem, asteval,
  podman, vfs) have been decomposed into smaller, focused modules. The adapter
  `shared.py` has been split into focused components. These are internal
  changes with no public API impact.

## v0.15.0 - 2025-12-17

### Added

- **ResourceRegistry for typed runtime resources.** `ToolContext` now exposes a
  `resources` property containing a typed registry for runtime services. This
  provides clean extensibility for future resources (HTTPClient, KVStore,
  ArtifactStore, Clock, Tracer) without bloating the core dataclass:

  ```python
  # Access via typed registry
  fs = context.resources.get(Filesystem)
  budget = context.resources.get(BudgetTracker)

  # Sugar property for common resources
  context.filesystem  # shorthand for context.resources.get(Filesystem)
  ```

- **Filesystem protocol with pluggable backends.** New `Filesystem` protocol
  abstracts file operations with three implementations:

  - `HostFilesystem`: Direct host access with optional root jail
  - `InMemoryFilesystem`: Pure in-memory for testing
  - `PodmanFilesystem`: Container-isolated operations

- **Automatic workspace injection.** `ToolContext.for_workspace()` factory
  creates contexts with filesystem and budget tracker pre-configured from
  workspace settings.

### Changed

- **ToolContext uses ResourceRegistry internally.** The `filesystem` and
  `budget_tracker` properties now delegate to the resource registry. This is
  backwards-compatible—existing code continues to work.

## v0.14.0 - 2025-12-15

### Claude Agent SDK Adapter

A new adapter enables running WINK prompts through Claude Code's agentic
runtime with hook-based session synchronization and hermetic isolation
options.

```python
from weakincentives.adapters.claude_agent_sdk import ClaudeAgentSDKAdapter
from weakincentives.runtime import Session

session = Session()
adapter = ClaudeAgentSDKAdapter(model="claude-sonnet-4-5-20250929")
response = adapter.evaluate(prompt, session=session)
```

Key capabilities:

- **Session synchronization**: Hook-based state sync keeps your WINK `Session`
  as the source of truth while Claude Code executes tools.
- **MCP tool bridging**: WINK tools with handlers are exposed to Claude Code
  via MCP so the SDK can call them.
- **Workspace materialization**: Materialize host paths into a temporary
  workspace for SDK access via `ClaudeAgentWorkspaceSection`.
- **Isolation**: Run without touching the host's `~/.claude` config using
  `ClaudeAgentSDKIsolation`.

See `specs/CLAUDE_AGENT_SDK.md` for full details.

### LiteLLM Adapter

A new adapter supports 100+ LLM providers through LiteLLM:

```python
from weakincentives.adapters.litellm import LiteLLMAdapter

adapter = LiteLLMAdapter(model="anthropic/claude-sonnet-4-20250514")
response = adapter.evaluate(prompt, session=session)
```

### Prompt Registry: List & Duplicate

```python
from weakincentives.prompt.registry import PromptRegistry

# List all registered prompts
for fqn in registry.list():
    print(fqn)  # "namespace/key"

# Duplicate a prompt with modifications
registry.duplicate(
    "code-reviewer/main",
    "code-reviewer-v2/main",
    override={"sections": [...]},
)
```

### Custom Visibility Layers

Define project-specific visibility layers beyond the built-in user/developer/debug:

```python
from weakincentives.prompt import VisibilityOverrides

overrides = VisibilityOverrides(
    custom_layers={
        "internal": 50,   # Between user (0) and developer (100)
        "audit": 200,     # Above debug (150)
    }
)
```

### MainLoop: Event-Driven Agent Orchestration

New `MainLoop` abstract base class standardizes the agent execution loop with
event-driven architecture:

```python
from weakincentives.runtime import MainLoop

class CodeReviewLoop(MainLoop[PRInput, ReviewOutput]):
    def create_prompt(self, request: PRInput) -> Prompt[ReviewOutput]:
        return Prompt(self.template).bind(PRParams(pr=request))

    def create_session(self) -> Session:
        return Session(bus=self._bus)

loop = CodeReviewLoop(adapter=adapter, bus=event_bus)
response, session = loop.execute(pr_input)
```

Key features:

- **Request/Response events**: `MainLoopRequest` and `MainLoopResponse` for
  observability
- **Automatic event publishing**: All requests and responses flow through the
  event bus
- **Session lifecycle**: Each execution creates a fresh session

## v0.13.0 - 2025-01-10

### Visibility: Token Budgets & Progressive Disclosure

Override token budgets and visibility per prompt invocation:

```python
from weakincentives.prompt import VisibilityOverrides, VisibilityLevel

overrides = VisibilityOverrides(
    token_budget=4000,
    section_overrides={"debug-info": VisibilityLevel.HIDDEN},
)
prompt = prompt.with_overrides(overrides)
```

### DbC: Design-by-Contract Decorators

New `weakincentives.dbc` module with contract decorators:

```python
from weakincentives.dbc import require, ensure, invariant

@require(lambda x: x > 0, "x must be positive")
@ensure(lambda result: result >= 0, "result must be non-negative")
def compute(x: int) -> int:
    return x * 2
```

# Chapter 18: API Reference

This is a curated reference of the APIs you'll touch most often. For complete details, read module docstrings and the specs.

## 18.1 Top-level exports

Import from `weakincentives` when you want the "90% API":

**Budgets/time:**

- `Deadline`, `DeadlineExceededError`
- `Budget`, `BudgetTracker`, `BudgetExceededError`

**Prompt primitives:**

- `PromptTemplate`, `Prompt`, `RenderedPrompt`
- `Section`, `MarkdownSection`
- `Tool`, `ToolContext`, `ToolResult`, `ResourceRegistry`
- `SectionVisibility`
- `parse_structured_output`, `OutputParseError`

**Runtime primitives:**

- `Session`, `InProcessDispatcher`
- `MainLoop`, `MainLoopConfig` and loop events (`MainLoopRequest`, `MainLoopCompleted`, `MainLoopFailed`)
- Reducer helpers (`append_all`, `replace_latest`, `upsert_by`, ...)
- Logging helpers (`configure_logging`, `get_logger`)

**Errors:**

- `WinkError`, `ToolValidationError`, snapshot/restore errors

## 18.2 weakincentives.prompt

```python
PromptTemplate[OutputT](ns, key, name=None, sections=..., allow_extra_keys=False)
Prompt(template, overrides_store=None, overrides_tag="latest")
    .bind(*params)
    .render(session=None)
    .find_section(SectionType)

MarkdownSection(title, key, template, summary=None, visibility=..., tools=..., policies=...)
Tool(name, description, handler, examples=...)
ToolResult.ok(value, message="OK")      # success case
ToolResult.error(message)               # failure case
```

**Tool policies:**

- `ToolPolicy`: Protocol for tool invocation constraints
- `SequentialDependencyPolicy(dependencies)`: Enforce tool ordering
- `ReadBeforeWritePolicy()`: Prevent overwrites without reading first

**Progressive disclosure:**

- `VisibilityExpansionRequired`

For detailed explanations of prompt composition and overrides, see:
- [Chapter 2: Prompts](02-prompts.md)
- [Chapter 3: Tools](03-tools.md)
- [Chapter 15: Prompt Optimization](15-prompt-optimization.md)

## 18.3 weakincentives.runtime

```python
Session(bus, tags=None, parent=None)
SessionView(session)                    # Read-only wrapper for reducer contexts
session[Type].all() / latest() / where()
session.dispatch(event)                 # All mutations go through dispatch

# Convenience methods (dispatch events internally)
session[Type].seed(value)               # → InitializeSlice
session[Type].clear()                   # → ClearSlice

# Visibility overrides (in runtime.session)
VisibilityOverrides, SetVisibilityOverride, ClearVisibilityOverride
session.snapshot(include_all=False)
session.restore(snapshot, preserve_logs=True)

# MainLoop configuration
MainLoopConfig(deadline=..., budget=..., resources=...)
MainLoop.execute(request, deadline=..., budget=..., resources=...)
```

**Slice storage (in runtime.session):**

- `SliceView[T]`: Read-only protocol for reducer input
- `Slice[T]`: Mutable protocol for storage operations
- `SliceOp`: Algebraic type (`Append | Extend | Replace | Clear`)
- `InitializeSlice[T]`, `ClearSlice[T]`: System events for slice mutations
- `MemorySlice` / `JsonlSlice`: In-memory and JSONL backends

**Reducers:**

- `append_all`, `replace_latest`, `replace_latest_by`, `upsert_by`
- Reducers receive `SliceView[S]` and return `SliceOp[S]`

**Event bus:**

- `InProcessDispatcher`
- Telemetry events (`PromptRendered`, `ToolInvoked`, `PromptExecuted`, `TokenUsage`)

**Lifecycle management:**

- `Runnable`: Protocol for loops with graceful shutdown (`run()`, `shutdown()`, `running`, `heartbeat`)
- `ShutdownCoordinator.install()`: Singleton for SIGTERM/SIGINT handling
- `LoopGroup(loops, health_port=..., watchdog_threshold=...)`: Run multiple loops with coordinated shutdown, health endpoints, and watchdog monitoring
- `Heartbeat`: Thread-safe timestamp tracker for worker liveness
- `Watchdog`: Daemon thread that monitors heartbeats and terminates on stall
- `HealthServer`: Minimal HTTP server for `/health/live` and `/health/ready`
- `wait_until(predicate, timeout=...)`: Poll predicate with timeout

For detailed explanations of sessions and state management, see:
- [Chapter 4: Sessions](04-sessions.md)
- [Chapter 7: MainLoop](07-main-loop.md)
- [Chapter 12: Lifecycle Management](12-lifecycle.md)

## 18.4 weakincentives.adapters

```python
ProviderAdapter.evaluate(prompt, session=..., deadline=..., budget=..., budget_tracker=...)
PromptResponse(prompt_name, text, output)
PromptEvaluationError
```

**Configs:**

- `OpenAIClientConfig`, `OpenAIModelConfig`
- `LiteLLMClientConfig`, `LiteLLMModelConfig`

**Resources (weakincentives.resources):**

- `Binding[T](protocol, provider, scope=Scope.SINGLETON, eager=False)`
- `Binding.instance(protocol, value)` - bind pre-constructed instance
- `Scope` enum: `SINGLETON`, `TOOL_CALL`, `PROTOTYPE`
- `ResourceRegistry.of(*bindings)` - build registry from bindings
- `ResourceRegistry.merge(base, override)` - combine registries (override wins)
- `ResourceRegistry.conflicts(other)` - return protocols bound in both
- `ResourceRegistry.open()` - context manager for resource lifecycle
- `ScopedResourceContext` - resolution context with lifecycle management
- `Closeable`, `PostConstruct` - lifecycle protocols
- `CircularDependencyError`, `DuplicateBindingError`, `ProviderError`, `UnboundResourceError` - dependency injection errors

**Throttling:**

- `ThrottlePolicy`, `new_throttle_policy`, `ThrottleError`

For detailed explanations of adapters and resources, see:
- [Chapter 5: Provider Adapters](05-adapters.md)
- [Chapter 6: Resource Registry](06-resources.md)

## 18.5 weakincentives.contrib.tools

**Planning:**

- `PlanningToolsSection(session, strategy=..., accepts_overrides=False)`

**Workspace:**

- `VfsToolsSection(session, config=VfsConfig(...), accepts_overrides=False)`
- `HostMount(host_path, mount_path=None, include_glob=(), exclude_glob=())`
- `WorkspaceDigestSection(session, title="Workspace Digest", key="workspace-digest")`

**Sandboxes:**

- `AstevalSection(session, accepts_overrides=False)`
- `PodmanSandboxSection(session, config=PodmanSandboxConfig(...))` (extra)

For detailed explanations of tools, see:
- [Chapter 8: Workspace Tools](08-workspace.md)
- [Chapter 9: Planning Tools](09-planning.md)

## 18.6 weakincentives.optimizers

- `PromptOptimizer` protocol and `BasePromptOptimizer`
- `OptimizationContext`, `OptimizationResult`

**Contrib:**

- `WorkspaceDigestOptimizer`

For detailed explanations of optimization, see:
- [Chapter 15: Prompt Optimization](15-prompt-optimization.md)

## 18.7 weakincentives.serde

Dataclass serialization utilities (no Pydantic required):

```python
from weakincentives.serde import dump, parse, schema, clone

# Serialize a dataclass to a JSON-compatible dict
data = dump(my_dataclass)

# Parse a dict back into a dataclass (with validation)
obj = parse(MyDataclass, {"field": "value"})

# Generate JSON schema for a dataclass
json_schema = schema(MyDataclass)

# Deep copy a frozen dataclass
copy = clone(my_dataclass)
```

**Key behaviors:**

- `parse()` validates required fields and rejects unknown keys by default
- Nested dataclasses are recursively parsed
- `tuple`, `frozenset`, and other immutable collections are handled
- `schema()` produces OpenAI-compatible JSON schemas for structured output

For detailed explanations, see:
- [Chapter 2: Prompts](02-prompts.md) (structured output section)

## 18.8 weakincentives.evals

**Core types:**

```python
Sample[InputT, ExpectedT](id, input, expected)
Dataset[InputT, ExpectedT](samples)
Dataset.load(path, input_type, expected_type)

Score(value, passed, reason="")
EvalResult(sample_id, score, latency_ms, error=None)
EvalReport(results)
    .pass_rate / .mean_score / .mean_latency_ms / .failed_samples()
```

**Evaluators:**

```python
exact_match(output, expected) -> Score
contains(output, expected) -> Score
all_of(*evaluators) -> Evaluator
any_of(*evaluators) -> Evaluator
llm_judge(adapter, criterion) -> Evaluator
adapt(evaluator) -> SessionEvaluator  # Convert standard to session-aware
```

**Session evaluators:**

```text
tool_called(name) -> SessionEvaluator
tool_not_called(name) -> SessionEvaluator
tool_call_count(name, min_count, max_count) -> SessionEvaluator
all_tools_succeeded() -> SessionEvaluator
token_usage_under(max_tokens) -> SessionEvaluator
slice_contains(T, predicate) -> SessionEvaluator
```

**Loop and helpers:**

```text
EvalLoop(loop, evaluator, requests)
    .run(max_iterations=None)

submit_dataset(dataset, requests)
collect_results(results, expected_count, timeout_seconds=300)
```

For detailed explanations of evaluation, see:
- [Chapter 10: Evaluation Framework](10-evals.md)

## 18.9 weakincentives.skills

Agent Skills specification support (following https://agentskills.io):

```python
from weakincentives.skills import (
    Skill,
    SkillMount,
    SkillConfig,
    validate_skill,
    validate_skill_name,
    resolve_skill_name,
    MAX_SKILL_FILE_BYTES,      # 1 MiB
    MAX_SKILL_TOTAL_BYTES,     # 10 MiB
)

# Mount skills for Claude Agent SDK isolation
config = SkillConfig(
    skills=(
        SkillMount(source=Path("./skills/code-review")),
        SkillMount(source=Path("./skills/testing"), enabled=False),
    ),
    validate_on_mount=True,  # Default: validate before copying
)

# Validation functions
validate_skill(Path("./skills/my-skill"))  # Raises SkillValidationError
name = resolve_skill_name(mount)           # Derive name from path
```

**Errors:**

- `SkillError` (base), `SkillValidationError`, `SkillNotFoundError`, `SkillMountError`

For detailed explanations of skills, see:
- [Chapter 11: Agent Skills](11-skills.md)

## 18.10 weakincentives.filesystem

Filesystem protocol and implementations:

```python
from weakincentives.filesystem import (
    Filesystem,                # Protocol for file operations
    SnapshotableFilesystem,    # Extended protocol with snapshot/restore
    HostFilesystem,            # Host filesystem with git-based snapshots
)
# InMemoryFilesystem is in contrib:
# from weakincentives.contrib.tools import InMemoryFilesystem

# Binary operations (new in v0.19.0)
content = fs.read_bytes("image.png", offset=0, limit=1024)
fs.write_bytes("output.bin", b"\x00\x01\x02", mode="overwrite")

# Text operations
result = fs.read("config.json")
fs.write("output.txt", "content", mode="overwrite")

# Search operations
matches = fs.glob("**/*.py")
grep_result = fs.grep("TODO", path="src/", glob="*.py")
```

**Key behaviors:**

- `read_bytes()` supports offset and limit for partial reads
- `write_bytes()` supports "overwrite" and "append" modes
- `read()` raises `ValueError` with actionable message for binary content
- `grep()` silently skips non-UTF-8 files
- UTF-8 paths are now allowed (ASCII-only restriction removed)

For detailed explanations, see:
- [Chapter 8: Workspace Tools](08-workspace.md)

## 18.11 CLI

Installed via the `wink` extra:

```bash
pip install "weakincentives[wink]"
```

**Commands:**

```bash
# Start the debug UI server
wink debug <snapshot_path> [options]

# Access bundled documentation
wink docs --guide       # Print WINK_GUIDE.md
wink docs --reference   # Print llms.md (API reference)
wink docs --specs       # Print all spec files concatenated
wink docs --changelog   # Print CHANGELOG.md
```

**Debug options:**

| Option | Default | Description |
| ------------------- | ----------- | -------------------------------------- |
| `--host` | `127.0.0.1` | Host interface to bind |
| `--port` | `8000` | Port to bind |
| `--open-browser` | `true` | Open browser automatically |
| `--no-open-browser` | - | Disable auto-open |
| `--log-level` | `INFO` | Log verbosity (DEBUG, INFO, etc.) |
| `--json-logs` | `true` | Emit structured JSON logs |
| `--no-json-logs` | - | Emit plain text logs |

**Exit codes:**

- `0`: Success
- `2`: Invalid input (missing file, parse error)
- `3`: Server failed to start

For detailed explanations of debugging, see:
- [Chapter 13: Debugging](13-debugging.md)

---

## Where to go deeper

- **Prompts**: [specs/PROMPTS.md](../specs/PROMPTS.md)
- **Tools**: [specs/TOOLS.md](../specs/TOOLS.md)
- **Tool Policies**: [specs/TOOL_POLICIES.md](../specs/TOOL_POLICIES.md)
- **Sessions**: [specs/SESSIONS.md](../specs/SESSIONS.md)
- **MainLoop**: [specs/MAIN_LOOP.md](../specs/MAIN_LOOP.md)
- **Evals**: [specs/EVALS.md](../specs/EVALS.md)
- **Health & Lifecycle**: [specs/HEALTH.md](../specs/HEALTH.md)
- **Resources**: [specs/RESOURCE_REGISTRY.md](../specs/RESOURCE_REGISTRY.md)
- **Skills**: [specs/SKILLS.md](../specs/SKILLS.md)
- **Filesystem**: [specs/FILESYSTEM.md](../specs/FILESYSTEM.md)
- **Workspace**: [specs/WORKSPACE.md](../specs/WORKSPACE.md)
- **Overrides & optimization**: [specs/PROMPT_OPTIMIZATION.md](../specs/PROMPT_OPTIMIZATION.md)
- **Exhaustiveness checking**: [specs/EXHAUSTIVENESS.md](../specs/EXHAUSTIVENESS.md)
- **Formal verification**: [specs/FORMAL_VERIFICATION.md](../specs/FORMAL_VERIFICATION.md) (embedding TLA+ specs in Python)
- **Code review example**: [guides/code-review-agent.md](../guides/code-review-agent.md)
- **Contributor guide**: [AGENTS.md](../AGENTS.md)

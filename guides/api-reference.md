# API Reference

This is a curated reference of the APIs you'll touch most often. For complete
details, read module docstrings and the specs.

## Top-Level Exports

Import from `weakincentives` when you want the "90% API":

**Budgets/time:**

- `Deadline`, `DeadlineExceededError`
- `Budget`, `BudgetTracker`, `BudgetExceededError`

**Prompt primitives:**

- `Prompt`, `MarkdownSection`
- `Tool`, `ToolContext`, `ToolHandler`, `ToolResult`
- `parse_structured_output`

**Adapters:**

- `PromptResponse`

**Skills:**

- `Skill`, `SkillConfig`, `SkillMount`
- `SkillError`, `SkillValidationError`, `SkillNotFoundError`, `SkillMountError`

**Logging:**

- `StructuredLogger`, `configure_logging`, `get_logger`

**Types:**

- `JSONValue`, `SupportsDataclass`

**Dataclasses:**

- `FrozenDataclass`

**Errors:**

- `WinkError`, `ToolValidationError`

For everything else (e.g., `PromptTemplate`, `RenderedPrompt`, `Section`, `SectionVisibility`, `OutputParseError`, `Session`, `InProcessDispatcher`, `AgentLoop`, `AgentLoopConfig`, reducer helpers), import from the relevant subpackage: `weakincentives.prompt`, `weakincentives.runtime`, etc.

## weakincentives.prompt

```python nocheck
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

## weakincentives.runtime

```python nocheck
Session(dispatcher, tags=None, parent=None)
SessionView(session)                    # Read-only wrapper
session[Type].all() / latest() / where()
session.dispatch(event)                 # All mutations go through dispatch

# Convenience methods
session[Type].seed(value)               # → InitializeSlice
session[Type].clear()                   # → ClearSlice

# Snapshots
session.snapshot(include_all=False)
session.restore(snapshot, preserve_logs=True)

# AgentLoop
AgentLoopConfig(budget=..., resources=...)
AgentLoop.execute(request, deadline=..., budget=..., resources=...)
```

**Slice storage:**

- `SliceView[T]`: Read-only protocol for reducer input
- `SliceOp`: Algebraic type (`Append | Extend | Replace | Clear`)
- `MemorySlice` / `JsonlSlice`: In-memory and JSONL backends

**Reducers:**

- `append_all`, `replace_latest`, `replace_latest_by`, `upsert_by`
- Reducers receive `SliceView[S]` and return `SliceOp[S]`

**Lifecycle management:**

- `Runnable`: Protocol for loops
- `ShutdownCoordinator.install()`: Signal handling
- `LoopGroup(loops, health_port=..., watchdog_threshold=...)`

## weakincentives.adapters

```python nocheck
ProviderAdapter.evaluate(prompt, session=..., deadline=..., budget=...)
PromptResponse(prompt_name, text, output)
```

**Configs:**

- `LLMConfig` (base model parameters)

**Resources:**

- `Binding[T](protocol, provider, scope=Scope.SINGLETON)`
- `Binding.instance(protocol, value)`
- `Scope` enum: `SINGLETON`, `TOOL_CALL`, `PROTOTYPE`
- `ResourceRegistry.of(*bindings)`

## weakincentives.contrib.tools

**Workspace Digest:**

- `WorkspaceDigestSection(session)` — Renders cached workspace summary
- `set_workspace_digest(session, digest)` — Store digest in session state

## weakincentives.adapters.claude_agent_sdk (Workspace)

**Workspace sections:**

- `ClaudeAgentWorkspaceSection(session, mounts)` — Workspace with file access
- `HostMount(host_path, include_glob=(), exclude_glob=(), max_bytes=None)`

## weakincentives.evals

**Core types:**

```python nocheck
Sample[InputT, ExpectedT](id, input, expected)
Dataset[InputT, ExpectedT](samples)
Dataset.load(path, input_type, expected_type)

Score(value, passed, reason="")
EvalResult(sample_id, score, latency_ms, error=None)
EvalReport(results)
    .pass_rate / .mean_score / .mean_latency_ms / .failed_samples()
```

**Evaluators:**

```python nocheck
exact_match(output, expected) -> Score
contains(output, expected) -> Score
all_of(*evaluators) -> Evaluator
any_of(*evaluators) -> Evaluator
llm_judge(adapter, criterion) -> Evaluator
adapt(evaluator) -> SessionEvaluator
```

**Session evaluators:**

- `tool_called(name)`
- `tool_not_called(name)`
- `tool_call_count(name, min_count, max_count)`
- `all_tools_succeeded()`
- `token_usage_under(max_tokens)`
- `slice_contains(T, predicate)`

**Loop:**

```text
EvalLoop(loop, evaluator, requests).run(max_iterations=None)
submit_dataset(dataset, requests)
collect_results(results, expected_count, timeout_seconds=300)
```

## weakincentives.serde

Dataclass serialization utilities:

```python nocheck
from weakincentives.serde import dump, parse, schema, clone

data = dump(my_dataclass)                    # Serialize to dict
obj = parse(MyDataclass, {"field": "value"}) # Parse with validation
json_schema = schema(MyDataclass)            # Generate JSON schema
copy = clone(my_dataclass)                   # Deep copy frozen dataclass
```

## weakincentives.filesystem

```python nocheck
from weakincentives.filesystem import (
    Filesystem,                # Protocol for file operations
    SnapshotableFilesystem,    # Extended protocol with snapshot/restore
    HostFilesystem,            # Host filesystem with git-based snapshots
)

# Binary operations
content = fs.read_bytes("image.png", offset=0, limit=1024)
fs.write_bytes("output.bin", b"\x00\x01\x02", mode="overwrite")

# Text operations
result = fs.read("config.json")
fs.write("output.txt", "content", mode="overwrite")

# Search operations
matches = fs.glob("**/*.py")
grep_result = fs.grep("TODO", path="src/", glob="*.py")
```

## weakincentives.skills

```python nocheck
from weakincentives.skills import (
    SkillMount,
    SkillConfig,
    validate_skill,
)

config = SkillConfig(
    skills=(
        SkillMount(source=Path("./skills/code-review")),
    ),
    validate_on_mount=True,
)
```

## CLI

```bash
pip install "weakincentives[wink]"

# Start the debug UI server
wink debug <snapshot_path> [options]

# Access bundled documentation
wink docs --guide       # Print guide
wink docs --reference   # Print API reference
wink docs --specs       # Print all specs
wink docs --changelog   # Print changelog
```

## Where to Go Deeper

- **Prompts**: [specs/PROMPTS.md](../specs/PROMPTS.md)
- **Tools**: [specs/TOOLS.md](../specs/TOOLS.md)
- **Sessions**: [specs/SESSIONS.md](../specs/SESSIONS.md)
- **AgentLoop**: [specs/AGENT_LOOP.md](../specs/AGENT_LOOP.md)
- **Evals**: [specs/EVALS.md](../specs/EVALS.md)
- **Health & Lifecycle**: [specs/HEALTH.md](../specs/HEALTH.md)
- **Resources**: [specs/RESOURCE_REGISTRY.md](../specs/RESOURCE_REGISTRY.md)
- **Full API reference**: [llms.md](../llms.md)

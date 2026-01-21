# Changelog

Release highlights for weakincentives.

## Unreleased

> **Commits reviewed:** 2026-01-17 (bdba902) through 2026-01-20 (e0defaf)

### TL;DR

This release delivers **AWS Bedrock authentication** for the Claude Agent SDK,
a completely **redesigned debug bundle system** with automatic per-request
capture, **progressive disclosure** support for the Claude Agent SDK adapter,
**controllable time dependencies** for deterministic testing, and a **unified
verification toolbox** that replaces scattered build scripts. Debug bundles now
include comprehensive metrics, session snapshots, and filesystem archives in
atomic zip files. The codebase received significant internal cleanup including
path normalization consolidation, large file decomposition, and actionable
error messages throughout the toolchain.

### AWS Bedrock Support

`IsolationConfig` now supports AWS Bedrock authentication by inheriting
credentials from the host environment. When `api_key` is `None` (the default),
authentication is inherited transparently—this works with both Anthropic API
(`ANTHROPIC_API_KEY`) and AWS Bedrock (`CLAUDE_CODE_USE_BEDROCK=1` + AWS
credentials).

```python
# Inherit host auth (works with Bedrock or Anthropic API)
adapter = ClaudeAgentSDKAdapter(
    model="us.anthropic.claude-sonnet-4-20250514-v1:0",
    client_config=ClaudeAgentSDKClientConfig(
        isolation=IsolationConfig(),  # Inherits host auth
    ),
)

# Docker: specify where AWS config is mounted
isolation = IsolationConfig(aws_config_path="/mnt/aws")
```

New `IsolationConfig` fields:

- `aws_config_path`: Path to AWS config directory for Docker containers where
  credentials are mounted at non-standard paths

Factory methods for explicit intent and fail-fast validation:

- `IsolationConfig.inherit_host_auth()`: Inherit auth, fail if none configured
- `IsolationConfig.with_api_key(key)`: Use explicit API key
- `IsolationConfig.for_bedrock()`: Require Bedrock, fail if not configured

Model ID helpers for unified model selection (defaults to Sonnet 4.5):

- `get_default_model()`: Returns model ID in the appropriate format for auth mode
- `to_bedrock_model_id(name)`: Convert Anthropic model name to Bedrock ID
- `to_anthropic_model_name(id)`: Convert Bedrock ID to Anthropic model name

New exports: `AuthMode`, `BedrockConfig`, `AwsConfigResolution`,
`IsolationAuthError`, `DEFAULT_MODEL`, `DEFAULT_BEDROCK_MODEL`,
`get_supported_bedrock_models()`.

### Redesigned Debug Bundle System

The debug bundle system has been completely redesigned around a unified
`BundleWriter` API that replaces the previous fragmented utilities
(`collect_all_logs`, `dump_session`, `archive_filesystem`).

```python
from weakincentives.debug import BundleConfig, CaptureMode
from weakincentives.runtime import MainLoopConfig

config = MainLoopConfig(
    debug_bundle=BundleConfig(
        target="./debug_bundles/",
        mode=CaptureMode.STANDARD,
    ),
)
# Bundles are now created automatically per-request
```

**Key improvements:**

- **Automatic capture**: MainLoop generates debug bundles automatically when
  configured, including session state before/after execution, logs, metrics,
  and filesystem snapshots
- **Atomic zip creation**: Bundles are written to temporary files first, then
  atomically renamed to prevent partial/corrupted archives on crashes
- **Comprehensive metrics**: New `metrics.json` includes timing data
  (`started_at`, `ended_at`, `duration_ms`), token usage (input, output,
  cached, total, prompt count), and budget consumption when tracking is enabled
- **Per-request override**: `MainLoopRequest.debug_bundle` allows overriding
  the config-level bundle settings for individual requests
- **Prompt metadata**: Bundles include prompt namespace, key, and adapter name
- **Visibility overrides**: `prompt_overrides.json` captures visibility changes
  made during execution
- **Simplified UI**: Debug viewer streamlined with unified JSON tree renderer,
  removed over-engineered command palette and complex filtering

**Removed** (replaced by BundleWriter):
- `collect_all_logs(path)` → Use `BundleWriter.capture_logs()`
- `dump_session(session, target)` → Use `BundleWriter.write_session_after()`
- `archive_filesystem(fs, target)` → Use `BundleWriter.write_filesystem()`
- `max_log_lines` parameter removed from `BundleConfig` (bundles collect all
  logs without limits)

`MainLoopResult` now includes `bundle_path: Path | None` for accessing the
created bundle.

### Progressive Disclosure for Claude Agent SDK

The Claude Agent SDK adapter now supports progressive disclosure via the
`open_sections` tool. Models can request section expansion, and the adapter
handles the cross-context exception propagation transparently.

**How it works:** Since the MCP bridge runs in a different execution context
than the adapter, `VisibilityExpansionRequired` exceptions cannot propagate
directly. A new `VisibilityExpansionSignal` captures exceptions in tool
handlers and re-raises them after the SDK query completes, enabling the
standard progressive disclosure retry loop.

**Improved validation:** The `open_sections` tool is now more lenient—
requesting already-expanded sections no longer fails (they are silently
skipped). Errors only occur when ALL requested sections are already expanded.

### Controllable Time Dependencies

A new clock abstraction enables deterministic testing of time-dependent code
without real delays or monkeypatching.

```python
from weakincentives import FakeClock, SYSTEM_CLOCK, Clock

# Production code uses SYSTEM_CLOCK (the default)
deadline = Deadline(budget.deadline, clock=SYSTEM_CLOCK)

# Test code uses FakeClock for instant, deterministic time control
fake = FakeClock()
fake.advance(seconds=300)  # Instant time travel
```

**New protocols:**

- `MonotonicClock`: Elapsed time measurement (timeouts, rate limiting)
- `WallClock`: UTC timestamps (deadlines, event recording)
- `Sleeper`: Delay operations
- `Clock`: Unified protocol combining all three

**Implementations:**

- `SystemClock` / `SYSTEM_CLOCK`: Production implementation using system time
- `FakeClock`: Test implementation with `advance()`, `set_monotonic()`,
  `set_wall()` methods

All time-dependent components now accept optional `clock` parameters with
sensible defaults, including `Deadline`, `Heartbeat`, `LeaseExtender`,
`wait_until()`, and `InMemoryMailbox`.

See `specs/CLOCK.md` for the complete specification.

### Unified Verification Toolbox

The scattered build scripts have been consolidated into a unified, extensible
`toolchain/` framework with a single entry point.

```bash
# Run all checks
python check.py

# Run specific checks
python check.py lint test typecheck

# List available checks
python check.py --list

# Machine-readable output for CI
python check.py --json
```

**Key features:**

- **Single entry point**: `check.py` replaces 10+ scattered scripts in `build/`
  and `scripts/`
- **Extensible protocol**: Simple `Checker` interface for adding new checks
- **Structured diagnostics**: `Location` (file:line:column) + `Diagnostic` +
  `CheckResult` types for IDE-clickable output
- **Multiple formatters**: `ConsoleFormatter` (colored), `JSONFormatter`
  (machine-readable), `QuietFormatter` (failures only)
- **Unified parsers**: Extract structured diagnostics from ruff, pyright,
  pytest, mdformat, bandit, deptry, pip-audit output

**Actionable error messages**: All toolchain failures now provide structured,
multi-line diagnostics explaining what went wrong, how to reproduce, and how to
fix. Pytest failures include traceback context from `--tb=short` output.

See `specs/VERIFICATION_TOOLBOX.md` for the complete specification.

### Fixed

- **Frozenset serialization**: Session snapshots now correctly serialize
  `frozenset` fields (previously caused "not JSON serializable" errors)
- **Tool example warnings**: Non-dataclass tool example values (primitives,
  sequences) no longer trigger spurious warnings during serialization
- **RunContext population**: Events dispatched by the Claude Agent SDK adapter
  now include `session_id` for proper event correlation and distributed tracing
- **Manifest checksum circularity**: `manifest.json` is no longer included in
  its own checksums dictionary, eliminating the circular dependency

### Internal

- **Path normalization standardized**: Consolidated duplicate path normalization
  logic from three locations into `weakincentives.filesystem._path` module with
  `normalize_path_string()`, `validate_path()`, and `strip_mount_point()`
- **Large file decomposition**: Split `main_loop.py` (~728→540 lines) and
  `session.py` (~954→700 lines) into focused modules: `main_loop_types.py`,
  `message_handlers.py`, `session_cloning.py`, `session_dispatch.py`,
  `session_telemetry.py`
- **Documentation consolidation**: AGENTS.md, GEMINI.md, and WARP.md now
  redirect to CLAUDE.md as the single source of truth for AI assistant
  guidelines, with a prominent "Definition of Done" requirement
- **Test improvements**: Removed unnecessary `time.sleep()` calls using
  controllable clock injection; added `--timeout=10` to pytest for hang
  prevention
- **CI improvements**: Integration tests now skip automatically when API keys
  are missing (removed Makefile gatekeeper); Claude code review workflow hides
  previous reviews before posting new ones

## v0.20.0 - 2026-01-16

This release focuses on production observability and agent reliability.
**Experiments** enable systematic A/B testing of prompt variants with integrated
evaluation reporting. **RunContext** provides distributed tracing with
correlation IDs flowing through the entire request lifecycle. **Feedback
Providers** give unattended agents soft course-correction signals, while **Task
Completion Checking** ensures agents finish all assigned work before stopping.
**LeaseExtender** prevents message timeout during long operations, and new
**Debug Utilities** simplify post-mortem analysis.

### Experiments for A/B Testing

A new **Experiment** concept enables systematic evaluation of agent behavior
variants. An experiment bundles a prompt overrides tag with feature flags,
allowing coordinated changes to both prompt content and runtime behavior for
A/B testing, optimization runs, and controlled rollouts.

```python
from weakincentives.evals import Experiment, BASELINE, submit_experiments

treatment = Experiment(
    name="concise-prompts",
    overrides_tag="v2-concise",
    flags={"max_response_tokens": 500},
    owner="alice@example.com",
    description="Test shorter, more direct prompt phrasing",
)

# Submit dataset under both experiments for comparison
submit_experiments(dataset, [BASELINE, treatment], requests)

# Compare results
comparison = report.compare_experiments("baseline", "concise-prompts")
print(f"Delta: {comparison.pass_rate_delta:+.1%}")
```

Key features:

- **Experiment**: Immutable bundle of name, overrides tag, feature flags, and
  metadata
- **Request-level binding**: Experiments flow through `MainLoopRequest.experiment`
  and `EvalRequest.experiment`
- **EvalReport extensions**: `by_experiment()`, `pass_rate_by_experiment()`,
  `compare_experiments()` for result analysis
- **BASELINE/CONTROL sentinels**: Pre-defined experiments for common patterns

See `specs/EXPERIMENTS.md` for the full specification.

### RunContext for Distributed Tracing

`RunContext` provides immutable execution metadata that flows through the system
from MainLoop to tool handlers and telemetry events, enabling distributed
tracing, request correlation, and debugging.

```python
from weakincentives.runtime import RunContext, MainLoopRequest

ctx = RunContext(
    worker_id="worker-42",
    trace_id="abc-123",
    span_id="xyz-789",
)

request = MainLoopRequest(request=MyRequest(...), run_context=ctx)
```

Key features:

- **Correlation identifiers**: `run_id` (per-execution), `request_id` (stable
  across retries), `session_id`
- **Retry tracking**: `attempt` field from message delivery count
- **OpenTelemetry integration**: `trace_id` and `span_id` pass through unchanged
- **ToolContext access**: Tool handlers access via `context.run_context`
- **Structured logging**: `to_log_context()` for logger binding
- **Logger binding helper**: `bind_run_context(logger, ctx)` consistently binds
  all context fields to structured loggers throughout the request lifecycle

All adapters and telemetry events (`PromptRendered`, `ToolInvoked`,
`PromptExecuted`) now include `run_context`. See `specs/RUN_CONTEXT.md`.

### Feedback Providers for Agent Progress Assessment

A new **Feedback Provider** system enables ongoing progress assessment for
unattended agents. Unlike tool policies that gate individual calls, feedback
providers analyze patterns over time and inject contextual guidance for soft
course-correction.

```python
from weakincentives.prompt import (
    DeadlineFeedback,
    FeedbackProviderConfig,
    FeedbackTrigger,
    PromptTemplate,
)

template = PromptTemplate[OutputType](
    ns="my-agent",
    key="main",
    feedback_providers=(
        FeedbackProviderConfig(
            provider=DeadlineFeedback(warning_threshold_seconds=120),
            trigger=FeedbackTrigger(every_n_seconds=30),
        ),
    ),
)
```

Key components:

- **FeedbackProvider**: Protocol for producing feedback based on session state
- **FeedbackTrigger**: Conditions (every N calls or every N seconds) for when
  providers run
- **DeadlineFeedback**: Built-in provider that warns as deadlines approach

See `specs/FEEDBACK_PROVIDERS.md` for the full specification.

### Task Completion Checking

Task completion checking verifies that an agent has completed all assigned tasks
before allowing it to stop or produce final output.

```python
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    PlanBasedChecker,
)
from weakincentives.contrib.tools.planning import Plan

adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        task_completion_checker=PlanBasedChecker(plan_type=Plan),
    ),
)
```

Built-in checkers:

- `PlanBasedChecker`: Verifies all plan steps have `status == "done"`
- `CompositeChecker`: Combines multiple checkers with configurable logic

See `specs/TASK_COMPLETION.md` for the full specification.

### Automatic Message Lease Extension

`LeaseExtender` prevents message visibility timeout during long-running request
processing by extending the lease whenever a heartbeat occurs. This ties lease
extension to proof-of-work: if the worker is actively processing (beating), the
lease extends; if stuck (no beats), the lease expires naturally.

```python
from weakincentives.runtime import MainLoopConfig, LeaseExtenderConfig

config = MainLoopConfig(
    lease_extender=LeaseExtenderConfig(
        interval=60.0,   # Rate-limit to once per minute
        extension=300,   # Extend by 5 minutes on each beat
    ),
)
```

Key features:

- **Heartbeat-based**: Extension piggybacks on tool execution beats
- **Fail-safe**: Stuck workers let leases expire (correct behavior)
- **EvalLoop support**: Both MainLoop and EvalLoop support automatic extension
- **Claude Agent SDK**: Native tools trigger beats via hook system

See `specs/LEASE_EXTENDER.md` for the full specification.

### Adapter-Specific Exception Classes

Provider adapters now raise typed exceptions for better error handling:

```python
from weakincentives.adapters import OpenAIError, LiteLLMError, ClaudeAgentSDKError

try:
    response = adapter.evaluate(prompt, session=session)
except OpenAIError as e:
    print(f"OpenAI failed: {e.message}, status: {e.status_code}")
```

All exceptions inherit from `PromptEvaluationError` and include `message`,
`status_code` (if applicable), and `original_error`.

### Debug Bundles

**Note:** The standalone functions `collect_all_logs`, `archive_filesystem`, and
`dump_session` have been replaced by the unified `BundleWriter` API. Use
`BundleConfig` with `MainLoopConfig` for automatic bundle creation per-request.

```python
from weakincentives.debug import BundleConfig
from weakincentives.runtime import MainLoopConfig

config = MainLoopConfig(
    debug_bundle=BundleConfig(
        target="./debug_bundles/",
    ),
)
```

For manual bundle creation, use `BundleWriter` directly. See
`specs/DEBUG_BUNDLE.md` for the full specification.

### RedisMailbox: Default TTL for Redis Keys

RedisMailbox now applies a default 3-day TTL to all Redis keys, preventing
orphaned data from accumulating indefinitely. TTL is refreshed on every
operation, so active queues stay alive indefinitely.

```python
from weakincentives.contrib.mailbox import RedisMailbox, DEFAULT_TTL_SECONDS

mailbox = RedisMailbox(name="events", client=redis_client)  # 3-day default
mailbox = RedisMailbox(name="events", client=redis_client, default_ttl=86400)  # 1 day
mailbox = RedisMailbox(name="events", client=redis_client, default_ttl=0)  # Disabled
```

### Revamped `wink docs` CLI for AI Agent Exploration

The `wink docs` command has been completely redesigned with a subcommand
structure optimized for AI coding agents exploring documentation efficiently.

**New subcommands:**

- `wink docs list [specs|guides]` — List available documents with descriptions
- `wink docs search PATTERN` — Search documentation with context snippets
- `wink docs toc {spec,guide} NAME` — Preview document structure (headings only)
- `wink docs read {reference,changelog,example,spec,guide} [NAME]` — Read documents

**Key features:**

- **Progressive disclosure**: List → Search → TOC → Read workflow minimizes
  context usage
- **Search with context**: Case-insensitive substring matching with `--context`,
  `--max-results`, and `--regex` options
- **Table of contents**: Preview document headings before loading full content
- **Document descriptions**: List output includes descriptions for each spec/guide

See `specs/WINK_DOCS.md` for the full specification.

### Documentation

**Guide documentation reorganization:** Broke the monolithic WINK_GUIDE.md into
focused, standalone guides in `guides/` designed for human consumption. Each
guide uses a narrative style explaining design decisions and building the
correct mental model for agent development.

New guides include: `philosophy.md`, `quickstart.md`, `prompts.md`, `tools.md`,
`sessions.md`, `adapters.md`, `claude-agent-sdk.md`, `orchestration.md`,
`evaluation.md`, `lifecycle.md`, `progressive-disclosure.md`,
`prompt-overrides.md`, `workspace-tools.md`, `debugging.md`, `testing.md`,
`code-quality.md`, `recipes.md`, `troubleshooting.md`, `api-reference.md`,
`migration-from-langgraph.md`, `migration-from-dspy.md`,
`formal-verification.md`, and `code-review-agent.md`.

**Specification additions:**

- Added `specs/EXPERIMENTS.md` covering experiment configuration for A/B testing
- Added `specs/RUN_CONTEXT.md` covering execution metadata and distributed tracing
- Added `specs/LEASE_EXTENDER.md` covering automatic message visibility extension
- Added `specs/TASK_COMPLETION.md` covering task completion checking patterns
- Added `specs/DLQ.md` covering dead letter queue configuration
- Added `specs/POLICIES_OVER_WORKFLOWS.md` documenting declarative policies philosophy
- Added `specs/WINK_DOCS.md` covering the redesigned docs CLI
- Added `IDENTITY.md` with WINK project description
- Rewrote `llms.md` as comprehensive technical guide for PyPI

### Breaking Changes

**Removed Notification from Claude Agent SDK public API:** The `Notification`
and `NotificationSource` types have been removed from the public API.

**PromptOverridesStore API refactored:** Now uses `PromptDescriptor` for
identifying prompts instead of separate namespace/key parameters.

**Mailbox reply_to parameter refactored:** Now accepts a `Mailbox` instance
instead of a string identifier for type-safe routing.

### Internal

- Renamed `Dispatcher` references to `ControlDispatcher` for clarity
- Improved library modularity with import validation (`make check-core-imports`)
- Refactored specs for conciseness and clarity, focusing on design over implementation
- Consolidated and reviewed specification documents
- Fixed documentation broken links and incorrect references
- Dependency upgrades: aiohttp, anyio, certifi, filelock, huggingface-hub,
  hypothesis, sse-starlette, textual, tokenizers, typer-slim

## v0.19.0 - 2026-01-07

### Tool Policies: Declarative Constraints for Safe Tool Invocation

A new **Tool Policy** system enables declarative constraints on tool invocation
sequences, preventing unsafe operations without hardcoding logic into handlers.
Policies are declared at prompt and section levels, track state in session
slices, and survive snapshot/restore cycles.

```python
from weakincentives.prompt import SequentialDependencyPolicy, ReadBeforeWritePolicy

# Enforce deployment pipeline: lint → build → test → deploy
policy = SequentialDependencyPolicy(
    dependencies={"deploy": frozenset({"test", "build"})}
)

# Prevent file overwrites without reading first
read_first = ReadBeforeWritePolicy(mount_point="/workspace")
```

**Built-in policies:**

- `SequentialDependencyPolicy`: Enforce unconditional tool ordering (e.g.,
  CI/CD pipelines)
- `ReadBeforeWritePolicy`: Prevent overwriting files without reading them first;
  new files can be created freely

**Integration:** `VFSToolsSection` and `PodmanSandboxSection` now have
`ReadBeforeWritePolicy` enabled by default. See `specs/TOOLS.md#tool-policies`.

### Exhaustiveness Checking for Union Types

Match statements on union types now include exhaustiveness sentinels that
pyright validates at type-check time. When a new variant is added to a union,
missing handlers are caught immediately—no runtime surprises.

```python
from typing import assert_never

match op:
    case Append(item=item):
        slice_instance.append(item)
    case Extend(items=items):
        slice_instance.extend(items)
    case _ as unreachable:
        assert_never(unreachable)  # pyright catches missing cases
```

Applied to `SliceOp` handling throughout the session runtime. See
`specs/DBC.md`.

### Skill Validation at Mount Time

Skills now undergo comprehensive validation when mounted, catching configuration
errors early rather than at runtime:

- **Directory skills**: Validates SKILL.md exists with correct frontmatter
- **File skills**: Enforces `.md` extension and 1 MiB size limit
- **Frontmatter validation**: Checks required fields (`name`, `description`)
  and optional fields against the Agent Skills specification
- **Lazy YAML loading**: pyyaml imported only when validation runs, keeping
  the base package lightweight

**Validation functions:** `validate_skill()`, `validate_skill_name()`,
`resolve_skill_name()`.

**Constants:** `MAX_SKILL_FILE_BYTES` (1 MiB), `MAX_SKILL_TOTAL_BYTES` (10 MiB).

### Comprehensive Prompt Override System

The prompt overrides system now supports **all text literals** affecting agent
performance—not just section bodies:

**New override types:**

- `TaskExampleOverride`: Override entire task examples (objective, outcome, steps)
- `TaskStepOverride`: Modify individual steps within examples
- `ToolExampleOverride`: Override tool example descriptions, inputs, and outputs

**Enhanced `SectionOverride`:** Now includes optional `summary` and `visibility`
fields for fine-grained prompt optimization.

**Performance improvements:**

- TOCTOU-safe atomic operations via internal `_resolve_unlocked()` /
  `_upsert_unlocked()` helpers
- Efficient hash-based staleness detection
- Conditional field serialization to minimize file sizes

### Binary Read/Write Support for Filesystem

The `Filesystem` protocol now supports binary file operations alongside text,
enabling proper handling of images, compiled binaries, archives, and exact-copy
operations without encoding overhead.

**New methods:**

- `read_bytes(path, *, offset=0, limit=None)` → `ReadBytesResult`
- `write_bytes(path, content, *, mode="overwrite", create_parents=True)` →
  `WriteResult`

**Behavior changes:**

- `InMemoryFilesystem` now stores files as bytes internally
- `read()` raises `ValueError` with actionable message for binary content
- `grep()` silently skips non-UTF-8 files
- UTF-8 paths now allowed (ASCII-only restriction removed)

### Skills as Core Library Concept

Skills promoted from Claude Agent SDK adapter to `weakincentives.skills` module,
following the [Agent Skills specification](https://agentskills.io):

```python
from weakincentives.skills import Skill, SkillMount, SkillConfig

config = SkillConfig(
    mounts=(SkillMount(name="code-review", path="/path/to/skill"),),
    validate_on_mount=True,
)
```

**Error hierarchy:** `SkillError` → `SkillValidationError`, `SkillNotFoundError`,
`SkillMountError`.

### Enhanced Debug Logging for Claude Agent SDK

Comprehensive DEBUG-level logging coverage for better observability:

- **`HookStats` dataclass**: Tracks cumulative execution metrics (tool count,
  turn count, subagent count, context compactions, token usage including
  thinking tokens)
- **Timing metrics**: `elapsed_ms` and `hook_duration_ms` in all hook events
- **Constraint tracking**: Budget consumption and deadline remaining in
  pre-tool-use events
- **Lifecycle events**: Turn boundaries, subagent start/stop, context window
  compaction with utilization percentage
- **Error logging**: Comprehensive error classification with stderr capture

### Architecture Improvements

- **HostFilesystem moved to core**: Now in `weakincentives.filesystem` alongside
  the protocol. Backward-compatible re-export from `contrib.tools` retained.
- **Workspace protocols promoted**: `ToolSuiteSection` and `WorkspaceSection`
  moved from contrib to `prompt.protocols`. New `make check-core-imports`
  validates core never imports from contrib.

### Fixed

- **Logging formatter KeyError**: Text formatter now gracefully handles missing
  `event` and `context` fields from external libraries
- **Circular import in contrib**: Replaced eager imports with lazy `__getattr__`
  loading

### Breaking Changes

**Prompt overrides API:**

```python
# Before
store.set_section_override(prompt, tag="latest", path=("section",), body="...")

# After
from weakincentives.prompt import SectionOverride
from weakincentives.prompt.overrides import PromptDescriptor

descriptor = PromptDescriptor.from_prompt(prompt)
store.store(descriptor, SectionOverride(path=("section",), expected_hash="...", body="..."), tag="latest")
```

**Override file format upgraded from v1 to v2.** Existing override files in
`.weakincentives/overrides/` should be regenerated with `store.seed(prompt)`.

### Internal

- Removed mutation testing infrastructure (mutmut)
- CI/CD: Consolidated 4 static analysis jobs → 1, added concurrency control,
  standardized on actions/checkout@v5, Python 3.12 → 3.14 in verify.yml
- Dependency upgrades: aiohttp, anyio, certifi, filelock, huggingface-hub,
  hypothesis, sse-starlette, textual (6.x → 7.x), tokenizers, typer-slim

## v0.18.0 - 2026-01-05

### Breaking Changes

**Prompt resource lifecycle via `prompt.resources`:** The context manager pattern
changed from `with prompt:` to `with prompt.resources:`. The `Prompt` class no
longer implements `__enter__`/`__exit__` directly.

```python
# Before
with prompt:
    fs = prompt.resources.get(Filesystem)

# After
with prompt.resources:
    fs = prompt.resources.get(Filesystem)
```

**Removed `ExecutionState`:** Resources now bind directly to `Prompt` instead of
a separate `ExecutionState`. Use `with prompt.resources:` to manage lifecycle and
`adapter.evaluate(prompt, session=session)` for evaluation.

**Removed `mount_point` from Filesystem protocol:** Mount point handling moved to
`FilesystemToolHandlers`. Remove the property from custom filesystem implementations.

### Lifecycle Management

`LoopGroup` runs multiple `MainLoop` or `EvalLoop` instances with coordinated
shutdown, optional health endpoints (`/health/live`, `/health/ready`), and watchdog
monitoring for Kubernetes deployments.

```python
from weakincentives.runtime import LoopGroup

group = LoopGroup(
    loops=[main_loop, eval_loop],
    health_port=8080,
    watchdog_threshold=720.0,
)
group.run()  # Blocks until SIGTERM/SIGINT
```

### Resource Registry with Dependency Injection

`ResourceRegistry` supports provider-based lazy construction with scoped lifecycles
(`SINGLETON`, `TOOL_CALL`, `PROTOTYPE`) and cycle detection.

```python
from weakincentives.resources import Binding, ResourceRegistry, Scope

registry = ResourceRegistry.of(
    Binding(Config, lambda r: Config.from_env()),
    Binding(HTTPClient, lambda r: HTTPClient(r.get(Config).url)),
    Binding(Tracer, lambda r: Tracer(), scope=Scope.TOOL_CALL),
)
```

### Session Evaluators

Evaluate agent behavior—not just output—with session-aware evaluators for tool
usage patterns, token budgets, and state invariants.

```python
from weakincentives.evals import tool_called, all_tools_succeeded, all_of

evaluator = all_of(
    exact_match,
    tool_called("search"),
    all_tools_succeeded(),
)
```

Built-in evaluators: `tool_called`, `tool_not_called`, `tool_call_count`,
`all_tools_succeeded`, `token_usage_under`, `slice_contains`.

### TLA+ Specification Embedding

Co-locate formal TLA+ specifications with Python code using the `@formal_spec`
decorator. Extract and verify with TLC model checker via `weakincentives.formal`.

### Mailbox Reply-to Routing

Messages support `reply_to` routing, allowing workers to derive response
destinations from incoming messages instead of requiring fixed response mailboxes.

### Claude Agent SDK

- **Hermetic by default:** The adapter now uses isolated `~/.claude` configuration
  to prevent interference from host settings.
- **Skill mounting:** Mount custom skills into the hermetic environment via
  `SkillConfig` and `SkillMount` in `IsolationConfig`.

### Other Improvements

- **`ToolResult.ok()` / `ToolResult.error()`:** Convenience constructors reduce
  boilerplate in tool handlers.
- **`wink docs --changelog`:** Access release history without repository access.
- **Mailbox timeout validation:** Rejects invalid `visibility_timeout` values.
- **Fixed:** Parameterless tool execution crash, Redis mailbox thread leak, and
  `decode_responses=True` compatibility.

## v0.17.0 - 2025-12-25

### Breaking: Reducer Signature Changes

Reducers now receive `SliceView[S]` instead of `tuple[S, ...]` and return
`SliceOp[S]` instead of `tuple[S, ...]`:

```python
from weakincentives.runtime.session import SliceView, Replace, Append

# Before
def my_reducer(state: tuple[Plan, ...], event: AddStep) -> tuple[Plan, ...]:
    return (*state, Plan(steps=(event.step,)))

# After
def my_reducer(state: SliceView[Plan], event: AddStep) -> Append[Plan]:
    return Append(Plan(steps=(event.step,)))
```

The `@reducer` decorator now requires methods to return `Replace[T]` instead
of `T`:

```python
@dataclass(frozen=True)
class AgentPlan:
    steps: tuple[str, ...] = ()

    @reducer(on=AddStep)
    def add_step(self, event: AddStep) -> Replace["AgentPlan"]:
        return Replace(replace(self, steps=(*self.steps, event.step)))
```

### Breaking: Session Slice Observers Removed

The slice observer API (`session.observe()`, `SliceObserver`, `Subscription`)
has been removed. This reactive notification system added complexity without
providing value—real applications simply query session state when needed using
`session[Type].latest()` rather than subscribing to changes.

**Removed:**

- `session.observe(SliceType, callback)` method
- `SliceObserver` type alias
- `Subscription` class and `unsubscribe()` method
- Observer notification after reducer execution

**Migration:** Replace observer callbacks with direct state queries:

```python
# Before (removed)
def on_change(old, new): ...
subscription = session.observe(Plan, on_change)

# After
plan = session[Plan].latest()  # Query when needed
```

### Mailbox Message Queue Abstraction

A new `Mailbox` protocol provides SQS-compatible semantics for durable,
at-least-once message delivery between processes. Unlike the pub/sub
`Dispatcher`, Mailbox delivers messages point-to-point with visibility timeout
and explicit acknowledgment.

```python
from weakincentives.runtime.mailbox import InMemoryMailbox, Message

# Create a typed mailbox
mailbox: Mailbox[WorkRequest] = InMemoryMailbox()

# Send a message
message_id = mailbox.send(WorkRequest(task="analyze"))

# Receive with visibility timeout
messages = mailbox.receive(visibility_timeout=30, wait_time_seconds=5)
for msg in messages:
    process(msg.body)
    msg.acknowledge()  # Remove from queue
```

**Implementations:**

- `InMemoryMailbox`: Single-process queues for testing and development
- `RedisMailbox`: Distributed queues using Redis lists and sorted sets for
  visibility timeout management

**Key features:**

- Point-to-point delivery with acknowledgment semantics
- Visibility timeout for in-flight message protection (redelivery on failure)
- Long polling support for efficient message retrieval
- Message metadata including delivery count for dead-letter handling
- Bidirectional communication via request/result mailbox pairs

### Evaluation Framework

A minimal evaluation framework built on MainLoop for testing agent behavior.
MainLoop handles orchestration; evals add datasets and scoring.

```python
from weakincentives.evals import Dataset, EvalLoop, exact_match

# Load a dataset
dataset = Dataset.load(Path("qa.jsonl"), str, str)

# Run evaluation
loop = EvalLoop(main_loop=my_loop, evaluator=exact_match)
report = loop.execute(dataset)

print(f"Accuracy: {report.accuracy:.2%}")
```

**Core types:**

- `Sample[InputT, ExpectedT]`: Single evaluation case with input and expected
  output
- `Dataset[InputT, ExpectedT]`: Immutable collection of samples with JSONL
  loading
- `Score`: Result from evaluating a single sample (0.0–1.0 with optional
  metadata)
- `EvalResult`: Pairs a sample with its output and score
- `EvalReport`: Aggregated metrics across all samples

**Built-in evaluators:**

- `exact_match`: Strict equality comparison
- `contains`: Substring matching with `all_of`/`any_of` combinators
- `llm_judge`: LLM-as-Judge with categorical ratings using a judge template

### First-Class Resource Injection

Pass custom resources into `adapter.evaluate()` and `MainLoop.execute()`,
making tools cleaner, more testable, and portable.

```python
from weakincentives.resources import ResourceRegistry
from myapp.http import HTTPClient

# Build resource registry with your dependencies
http_client = HTTPClient(base_url="https://api.example.com")
resources = ResourceRegistry.build({HTTPClient: http_client})

# Pass to adapter - merged with workspace resources (e.g., filesystem)
response = adapter.evaluate(prompt, session=session, resources=resources)

# Or at the MainLoop level
config = MainLoopConfig(resources=resources)
loop = MyLoop(adapter=adapter, dispatcher=dispatcher, config=config)
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

### Unified Dispatch API

All session mutations now flow through a single `session.dispatch(event)` method,
providing a consistent, auditable mutation interface. Convenience methods on slice
accessors dispatch events internally rather than mutating state directly.

```python
from weakincentives.runtime import Session, InProcessDispatcher
from weakincentives.runtime.session import InitializeSlice, ClearSlice

dispatcher = InProcessDispatcher()
session = Session(dispatcher=dispatcher)

# All mutations go through dispatch
session.dispatch(AddStep(step="Research"))

# Convenience methods dispatch events internally
session[Plan].seed(initial_plan)    # → dispatches InitializeSlice
session[Plan].clear()               # → dispatches ClearSlice

# Direct system event dispatch (equivalent to methods above)
session.dispatch(InitializeSlice(Plan, (initial_plan,)))
session.dispatch(ClearSlice(Plan))
```

**New system events:**

- `InitializeSlice[T]`: Replace all values in a slice
- `ClearSlice[T]`: Remove all values from a slice

### Slice Storage Backends

Protocol-based slice storage backends enable swapping between in-memory tuples
and persistent JSONL files without changing Session semantics.

**New protocols and types:**

- `SliceView[T]`: Lazy read-only protocol for reducer input
- `Slice[T]`: Mutable protocol for storage operations
- `SliceFactory`: Creates slices by type
- `SliceOp` (algebraic type): `Append[T] | Extend[T] | Replace[T] | Clear`

**Implementations:**

- `MemorySlice` / `MemorySliceView`: In-memory tuple-backed storage (default)
- `JsonlSlice` / `JsonlSliceView`: JSONL file-backed persistence with locking
- `SliceFactoryConfig`: Policy-based factory selection (`STATE` vs `LOG`)

### SessionView for Read-Only Access

`SessionView` provides a read-only wrapper around `Session` for safe, immutable
access to session state. Reducer contexts now receive `SessionView` to prevent
accidental mutations during reducer execution.

```python
from weakincentives.runtime.session import SessionView

# SessionView exposes query operations only
view = SessionView(session)
latest = view[Plan].latest()       # ✓ Query allowed
all_plans = view[Plan].all()       # ✓ Query allowed
view[Plan].seed(plan)              # ✗ AttributeError - no mutation methods
```

**SessionView exposes:**

- Query operations: `all()`, `latest()`, `where()` via `ReadOnlySliceAccessor`
- Event dispatch: `dispatch()` for broadcasting events
- Snapshot: `snapshot()` for capturing state
- Properties: `dispatcher`, `parent`, `children`, `tags`

**SessionView omits:**

- Session methods: `reset()`, `restore()`, `install()`
- Slice accessor methods: `seed()`, `clear()`, `append()`, `register()`

### Formal Verification for Redis Mailbox

The Redis mailbox algorithms are now formally verified using TLA+ model checking
and Hypothesis stateful property-based testing. Key invariants verified:
message state exclusivity, receipt handle freshness, no message loss, delivery
count monotonicity, and visibility timeout. See `specs/VERIFICATION.md` for
the complete verification framework.

### Added

- **`MainLoop.execute()` direct execution.** MainLoop now supports direct
  execution without mailbox routing, making it easier to run single evaluations
  or integrate with external orchestrators.

- **`wink docs` CLI command.** Exposes bundled documentation: `--reference`
  prints llms.md (API reference), `--guide` prints WINK_GUIDE.md, and `--specs`
  prints all spec files concatenated. Uses `importlib.resources` for reliable
  access when the package is installed outside the repository.

- **DSPy migration guide.** WINK_GUIDE.md now includes a comprehensive migration
  guide for users coming from DSPy, covering philosophy differences, concept
  mapping, and step-by-step migration paths with code examples.

- **Thread-safe ExecutionState.** `ExecutionState` now uses an `RLock` to guard
  all accesses to pending tool executions and snapshot/restore flows. Concurrent
  tool executions from multiple threads are now safe.

- **Generic filesystem validation suite.** A reusable test suite validates
  filesystem implementations against the protocol contract.

### Changed

- **MainLoop initialization refactored.** The two abstract methods
  `create_session()` and `create_prompt(request)` have been replaced with a
  single `prepare(request)` method returning `(prompt, session)`. A new
  optional `finalize(prompt, session)` hook is called after successful
  evaluation for cleanup or post-processing.

- **Filesystem protocol moved to core.** The `Filesystem` and
  `SnapshotableFilesystem` protocols now live in `weakincentives.filesystem`.
  Implementations (`InMemoryFilesystem`, `HostFilesystem`) remain in contrib.
  Backward-compatible re-exports are provided.

- **ResourceRegistry simplified.** Now implemented as a dataclass instead of a
  custom class, reducing complexity while preserving the same API.

### Fixed

- **Workspace cleanup removes git directories.** Temporary workspaces now
  properly clean up `.git` directories on teardown.

- **Snapshot errors propagated correctly.** `HostFilesystem.snapshot()` now
  raises `SnapshotError` when git commit fails instead of silently returning a
  stale commit hash. This ensures transactional rollback works correctly.

### Internal

- Decomposed `shared.py`, `utilities.py`, VFS module, asteval tool suite, and
  Podman tool file into smaller, focused modules.
- Moved `visibility_overrides` to break circular dependency between modules.
- Added integration test reference validation during build.
- Identified PEP 695 adoption gaps for future migration.
- Added code quality approach documentation.

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
  fs = context.filesystem  # equivalent
  ```

  `BudgetTracker` is now automatically registered in the `ResourceRegistry`
  during adapter evaluation, making token/time budget tracking available to
  all tool handlers.

- Local link checker in `make markdown-check` validates that Markdown links to
  local files point to existing targets. Fenced code blocks and inline code
  are skipped to avoid false positives.

### Fixed

- **Documentation: Standardized Session API examples.** All docs and specs now
  use the canonical `session[SliceType]` indexing API. Removed outdated
  references to the deprecated `session.query(T)` and `session.mutate(T)` APIs
  that were documented in the previous changelog entry but not updated in
  example code.

### Changed

- **Section base class now provides default `render()` implementation.** Custom
  sections can override `render_body()` instead of reimplementing heading
  composition. New helper methods:

  - `format_heading(depth, number, path)` - consistent markdown heading formatting
  - `render_body(params, *, visibility, path, session)` - override for custom body
  - `render_tool_examples()` - renders tool examples block

  The default `render()` combines heading + body + tool examples. Existing
  sections that override `render()` continue to work unchanged.

### Breaking: Session Slice Accessors + Explicit Broadcast

Session's query/mutation builders have been replaced by slice accessors via
indexing (`session[SliceType]`), and event dispatch is now explicit via
`session.broadcast(event)`.

**Removed:**

- `session.query(T)` / `QueryBuilder`
- `session.mutate(T)` / `MutationBuilder`
- `session.mutate()` / `GlobalMutationBuilder`
- `session.select_all(T)`

**New API:**

```python
# Query
latest = session[Plan].latest()
all_plans = session[Plan].all()
active = session[Plan].where(lambda p: p.active)

# Direct mutations (bypass reducers)
session[Plan].seed(initial_plan)
session[Plan].clear()

# Event dispatch (broadcast by event type)
session.broadcast(AddStep(step="Research"))

# Session-wide operations
session.reset()
session.restore(snapshot)
```

This also removes the footgun where `session.mutate(Plan).dispatch(e)` looked
slice-scoped but actually routed by `type(e)` (broadcast semantics).

**Migration:**

- `session.query(T).latest()` → `session[T].latest()`
- `session.query(T).all()` → `session[T].all()`
- `session.mutate(T).seed(x)` → `session[T].seed(x)`
- `session.mutate(T).clear(...)` → `session[T].clear(...)`
- `session.mutate(T).dispatch(e)` → `session.broadcast(e)`
- `session.mutate().reset()` → `session.reset()`
- `session.mutate().rollback(s)` → `session.restore(s)`

### Breaking: Ledger Semantics for Default Reducers

The default reducer now uses ledger semantics with `append_all`, which always
appends unconditionally. The previous `append` reducer (which deduped by
equality) has been replaced.

All event types default to `append_all` when no reducers are registered.

**Migration:**

- `append` → `append_all` (import path: `weakincentives.runtime.session`)

```python
from weakincentives.runtime.session import append_all

# Explicitly register ledger semantics (now also the default)
session[ToolInvoked].register(ToolInvoked, append_all)
```

### Added: Declarative State Slices

State slices can now be defined declaratively with reducers co-located as
methods on the dataclass itself:

```python
from dataclasses import dataclass, replace
from weakincentives.runtime.session import reducer

@dataclass(frozen=True)
class AddStep:
    step: str

@dataclass(frozen=True)
class AgentPlan:
    steps: tuple[str, ...] = ()
    current_step: int = 0

    @reducer(on=AddStep)
    def add_step(self, event: AddStep) -> "AgentPlan":
        return replace(self, steps=(*self.steps, event.step))

session.install(AgentPlan, initial=AgentPlan)
session.broadcast(AddStep(step="Research"))
session[AgentPlan].latest()
```

### Breaking: Filesystem Protocol + ToolContext Filesystem

Workspace file tools are now backed by a `Filesystem` protocol instead of a
`VirtualFileSystem` session snapshot.

- New `weakincentives.contrib.tools.filesystem` module with `Filesystem`,
  `InMemoryFilesystem`, and `HostFilesystem`.
- `ToolContext` now includes `filesystem: Filesystem | None` for handlers.
- `Prompt.filesystem()` returns the filesystem exposed by any section
  implementing `WorkspaceSection`.
- `VirtualFileSystem` has been removed; VFS tools now operate through a backend.

### Breaking: Prompt Params Type Naming

- `Section.param_type` has been removed; use `Section.params_type`.
- `PromptTemplate.param_types` and `RegistrySnapshot.param_types` have been
  renamed to `params_types`.

### Breaking: Typing Exports Consolidated Under `weakincentives.types`

`SupportsDataclass`, `SupportsDataclassOrNone`, and `SupportsToolResult` are no
longer exported from `weakincentives.prompt`; import them from
`weakincentives.types` (they remain exported at the package root).

### Claude Agent SDK Adapter: Additional Configuration

The Claude Agent SDK adapter now supports additional configuration options via
`ClaudeAgentSDKClientConfig` and `ClaudeAgentSDKModelConfig` (e.g. `max_turns`,
`max_budget_usd`, `betas`, and `max_thinking_tokens`).

### Validation: SUMMARY Requires a Summary Template

Requesting `SectionVisibility.SUMMARY` for a section without a `summary` template
now raises `PromptValidationError`.

### Documentation: Split Guides from Specs

Documentation has been reorganized to separate user-facing how-to material from
design specifications:

- New `guides/` folder for quickstarts, patterns, recipes, and best practices
- `specs/` now exclusively contains design contracts, invariants, and semantics
- Moved `CODE_REVIEWER_EXAMPLE.md` → `guides/code-review-agent.md`

### Typing & Internals

- `PromptTemplateProtocol` no longer advertises a `render()` method (rendering is
  done via `Prompt.render()`).
- `ProviderAdapterProtocol.evaluate()` no longer includes a `dispatcher` parameter
  (telemetry is published via `session.dispatcher`).
- `_ToolExecutionContext` is now internal (previous `ToolExecutionContext` name
  removed).

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
  `IsolationConfig` + `EphemeralHome`.
- **Network policy (tools only)**: Restrict tool egress with
  `NetworkPolicy.no_network()` / `NetworkPolicy.with_domains(...)`.

### Session-Managed Visibility Overrides

Visibility overrides are now managed through session state using Redux-style
events:

```python
from weakincentives.prompt import SectionVisibility
from weakincentives.runtime.session import (
    Session,
    SetVisibilityOverride,
    VisibilityOverrides,
)

session = Session()

session[VisibilityOverrides].apply(
    SetVisibilityOverride(path=("section",), visibility=SectionVisibility.FULL)
)
response = adapter.evaluate(prompt, session=session)
```

Visibility selectors and enabled predicates now accept an optional `session`
parameter for dynamic decisions based on session state.

Sessions automatically register visibility reducers—no manual setup required.

### Planning Tools

- Planning tool handlers now return the latest `Plan` snapshot after mutations
  (`planning_setup_plan`, `planning_add_step`, `planning_update_step`).

### API & Reliability

- `DeadlineExceededError` and `ToolValidationError` are now exported at the
  package root (`from weakincentives import DeadlineExceededError`).
- `@pure` contract enforcement is now thread-safe and no longer interferes
  with file I/O or logging in other threads during concurrent execution.

### Breaking Changes

- **Visibility overrides moved to Session state**: The `visibility_overrides`
  parameter has been removed from all adapter `evaluate()` methods. Use the
  `VisibilityOverrides` session state slice instead (see above).
- **Tools relocated to `weakincentives.contrib.tools`**: Planning, VFS,
  asteval, Podman, and workspace digest utilities now live in the `contrib`
  package.
- **Optimizer relocated to `weakincentives.contrib.optimizers`**:
  `WorkspaceDigestOptimizer` moved to contrib.
- **MainLoop return value**: `MainLoop.execute()` now returns
  `(response, session)` instead of just `response`.
- **MainLoop config**: `MainLoopConfig.parse_output` has been removed.
- **Reducer/event payloads**: Reducers now receive the event dataclass
  directly (no `.value` wrapper). `ToolInvoked` and `PromptExecuted` no longer
  expose a `.value` field.

### Architecture

The library is now organized as "core primitives" + "batteries for specific
agent styles":

- **Core** (`weakincentives.*`): Prompt composition, sessions, adapters,
  serde, design-by-contract—the minimal essential abstractions
- **Contrib** (`weakincentives.contrib.*`): Planning, VFS, Podman, asteval,
  workspace digest optimizer—optional domain-specific utilities

### Wink Debugger

- Slice list now displays class names instead of full module paths for
  improved readability.

### Documentation

- `specs/` is pruned to match what is implemented (unimplemented/aspirational
  specs removed).

## v0.13.0 - 2025-12-07

### MainLoop Orchestration

- Added `MainLoop[UserRequestT, OutputT]` abstract orchestrator that
  standardizes agent workflow execution: receive request, build prompt,
  evaluate, handle visibility expansion, publish result. Implementations
  define only the domain-specific factories via `create_prompt()` and
  `create_session()`.
- Added event types for request/response routing: `MainLoopRequest[T]`,
  `MainLoopCompleted[T]`, and `MainLoopFailed`.
- Added `MainLoopConfig` for configuring default deadline and budget
  constraints.
- The MainLoop automatically handles `VisibilityExpansionRequired` exceptions,
  accumulating visibility overrides and retrying evaluation with a shared
  `BudgetTracker` across retries.

### Budget Abstraction

- Added `Budget` resource envelope combining time and token limits
  (`deadline`, `max_total_tokens`, `max_input_tokens`, `max_output_tokens`).
  At least one limit must be set.
- Added `BudgetTracker` for thread-safe cumulative token tracking across
  multiple evaluations against a Budget.
- Added `BudgetExceededError` exception raised when any budget dimension is
  breached, with typed `BudgetExceededDimension` indicating which limit was
  hit.

### Session Runtime

- Added `session.query(T)` unified selector API that consolidates all session
  state queries into a fluent interface with `.latest()`, `.where(predicate)`,
  and `.all()` methods.
- Added `session.mutate()` fluent API as the counterpart to `session.query()`.
  This provides a unified interface for all session state mutations:
  - `session.mutate(T).seed(values)` - Initialize/replace slice values
  - `session.mutate(T).clear(predicate?)` - Remove items from a slice
  - `session.mutate(T).dispatch(event)` - Event-driven mutation through
    reducers
  - `session.mutate(T).append(value)` - Shorthand for dispatch with default
    reducer
  - `session.mutate(T).register(E, reducer)` - Register reducer for event type
  - `session.mutate().reset()` - Clear all slices
  - `session.mutate().restore(snapshot)` - Restore from snapshot

### Prompts & Templates

- Added generic `PromptTemplate[OutputT]` and `Prompt[OutputT]` abstractions
  that derive structured-output schema (object vs array, allow-extra-keys) and
  support parameterized binding.
- `PromptTemplate` is now immutable using `FrozenDataclass` with cached
  descriptor computation.
- Sections now include their path in rendered output for traceability.
- Added `SectionVisibility` enum (`FULL`, `SUMMARY`) with callable visibility
  selectors that can compute visibility from bound parameters.
- Added `summary` property to sections for use with `SUMMARY` visibility mode.
- Removed `parse_output` flag from `PromptTemplate` - parsing is now always
  performed by adapters.
- Removed `ResponseFormatSection` requirement - structured output instructions
  are now handled directly by adapters.

### Adapters

- Renamed `ConversationRunner` to `InnerLoop` with improved abstraction and
  maintainability.
- Added typed configuration objects: `LLMConfig`, `OpenAIClientConfig`,
  `OpenAIModelConfig`, `LiteLLMClientConfig`, `LiteLLMModelConfig`.
- Enabled parallel tool calls in OpenAI adapter via `parallel_tool_calls`
  config option.
- Simplified adapter method signatures to use `session` parameter only,
  removing separate `dispatcher` parameter since session now owns its event dispatcher.
- Added native web search tool integration spec for provider-executed tools.

### Error Handling

- Introduced `WinkError` as the root exception class for all library
  exceptions, allowing callers to catch all weakincentives errors with a
  single handler. Existing exceptions now inherit from `WinkError` while
  maintaining backward compatibility with their original base types
  (`ValueError`, `RuntimeError`).

### Tools

- Added `Tool.wrap` static helper that constructs `Tool` instances from
  handler annotations and docstrings, simplifying tool definition for common
  cases.
- Tools now support `None` for parameter and result types, including schema
  generation and prompt rendering for handlers that accept no input or return
  no structured output.

### Events & Telemetry

- Added `TokenUsage` tracking to `PromptExecuted` and `ToolInvoked` events,
  exposing prompt and completion token counts from provider responses.
- Added `unsubscribe(event_type, handler)` method to the `EventBus` protocol
  and `InProcessEventBus` implementation, allowing handlers to be removed
  after registration. The method returns `True` if the handler was found and
  removed, `False` otherwise.

### Dataclasses

- Added `FrozenDataclass` decorator that provides pre-init normalization
  hooks, `copy()` and `asdict()` helpers, and mapping utilities for immutable
  dataclass patterns. Exported from the package root.

### Wink Debugger

- Polished `wink debug` web interface with enhanced UX including keyboard
  navigation, improved layout, and better state visualization.
- Enhanced snapshot viewer with collapsible tree navigation, search filtering,
  and download capabilities.

### Tools & Sandboxes

- Podman sandbox containers now start with networking disabled
  (`network_mode=none`), and an integration test verifies they cannot reach
  external hosts.
- **Breaking**: Removed all subagent and prompt delegation code including
  `SubagentsSection`, `dispatch_subagents`, `DelegationPrompt`, and related
  composition helpers. This approach is not the right fit from a state and
  context management perspective.

### Documentation & Specs

- Added `MAIN_LOOP.md` spec documenting MainLoop orchestration and visibility
  override handling.
- Added `HOSTED_TOOLS.md` spec for provider-executed tools (web search, code
  interpreter).
- Added `TASK_SECTION.md` spec for task dataclass patterns in prompts.
- Added `WINK_OVERRIDES.md` spec documenting the `wink overrides` CLI command.
- Simplified and consolidated specification documents for clarity.

### Examples

- Refactored `code_reviewer_example.py` to use the new `MainLoop` abstraction,
  demonstrating the recommended pattern for agent workflow implementation.
- Code reviewer REPL now creates a fresh five-minute default deadline per
  request so long-running interactive sessions continue working without manual
  deadline overrides.

### Quality & Infrastructure

- Parallelized CI jobs for faster feedback.
- Added Vulture for dead code detection.
- Removed backward compatibility code and import shims.

## v0.12.0 - 2025-11-30

### Wink Debugger & Snapshots

- Snapshot viewer now renders markdown-rich values with a toggle between
  rendered HTML and raw markdown, refreshed styling, and separate state/event
  panels plus refresh controls.
- Snapshots persist session tags and parent/child relationships, surfacing
  tags in the debug UI and filtering out stray `.jsonl` bundles when listing
  snapshots.
- Debug app defaults and pagination were tightened, shared snapshot slice
  payloads are reused across the UI, and snapshot path handling is documented
  so missing or renamed files stay discoverable.

### Sessions

- Session locking was simplified and invariant/`skip_invariant` typing
  tightened to keep reducer mutations predictable and thread-safe.

### Prompts & Events

- Prompt render/execution events include the prompt descriptor, clone methods
  return `Self` for stricter typing, and `OptimizationScope` is now a
  `StrEnum` (update any comparisons against the enum values).
- Prompt rendering events tolerate payloads that omit the `value` field to
  keep logging resilient to adapter differences.
- Removed the prompt chapter abstraction; prompts now compose sections only
  and chapter-focused specs and tests have been removed.

### Adapters & API Surface

- Provider responses now flow through shared dataclasses, OpenAI response
  format handling is simplified, and adapter `__all__` exports are sorted for
  deterministic imports.
- Public API exports were clarified, legacy import shims and the package-level
  `__getattr__` fallback were removed, and schema/digest helpers were trimmed
  to reduce surface area.

### Tools, VFS & Sandboxes

- Host mount traversal now uses `Path.walk` for more reliable discovery in VFS
  and Podman-backed workspaces.
- Tool validation tightened with explicit `ToolExample` support and new
  examples covering VFS, planning, ASTEval, and Podman tools.
- Tool execution flow now relies on focused helpers for argument parsing,
  deadline checks, handler invocation, and result logging to reduce complexity
  while preserving structured events.
- Podman tool configuration helpers were refactored and workspace digest keys
  simplified to keep sandboxed runs and tests aligned.

### Serde & Schema

- Dataclass serde imports were consolidated into authoritative modules and
  shared schema helpers/constants were cleaned up to avoid duplication.

### Quality & Testing

- Added mutation-testing support and broader Ruff quality checks to catch
  issues earlier.
- Added unit coverage for tool execution success cases, validation failures,
  deadline expirations, and unexpected handler exceptions.

### Documentation & Specs

- AGENTS guidance now calls out the alpha/stability status explicitly, and
  throttling specs were aligned with the current implementation.

## v0.11.0 - 2025-11-23

### Wink Debugger & Snapshot Explorer

- Added a `wink debug` FastAPI UI for browsing session snapshot JSON files
  with reload/download actions, copy-to-clipboard, and a searchable tree
  viewer.
- Improved the viewer with keyword search, expand/collapse controls, compact
  array rendering, newline preservation, scrollable truncation boxes, and
  consistent spacing/padding so large payloads stay readable.

### Snapshot Persistence & Session State

- Session snapshots now log writes, skip empty payloads, and include runtime
  events while tolerating unknown slice types during replay.
- The code review example persists snapshots by default; snapshot writes are
  isolated in tests, and workspace digest optimization clones ASTEval tools to
  keep digest renders consistent.

### Adapters & Throttling

- Migrated the OpenAI adapter to the Responses API, updating tool negotiation,
  structured outputs, and retry behavior to match the new endpoint and tests.
- Added a shared throttling policy with jittered backoff and structured
  `ThrottleError`s across OpenAI and LiteLLM adapters, plus
  conversation-runner retry coverage.

### Documentation

- Added an OpenAI Responses API migration guide, a snapshot explorer
  walkthrough, and refreshed AGENTS/session hierarchy guidance.
- Clarified specs for throttling, deadlines, planning strategies/tools,
  prompts, overrides, adapters, logging, structured output, VFS, Podman
  sandboxing, and related design rationales.

## v0.10.0 - 2025-11-22

### Public API & Imports

- Curated exports now live in `api.py` modules across the root package and
  major subsystems; `__init__` files lazy-load those surfaces to keep import
  paths stable while trimming broad re-exports.

### Prompt Authoring & Overrides

- Rendered prompts now include hierarchical numbering (with punctuation) on
  section headings and descriptors so multi-section outputs remain readable
  and traceable.
- Prompts and sections have explicit clone contracts that rebind to new
  sessions and event dispatchers, keeping reusable prompt trees isolated.
- The prompt overrides store protocol is unified and re-exported through the
  versioned overrides module so custom stores and adapters target the same
  interface.
- Structured output metadata now flows through the shared prompt protocols and
  rendered prompts, simplifying adapters that need container/dataclass details
  without provider-specific shims.

### Tools & Execution

- Provider tool calls run through a shared `tool_execution` context manager
  that standardizes argument parsing, deadline enforcement, logging, and
  publish/ rollback handling; executor wrappers stay thin and fix prior
  mutable default edge cases.
- Added deadline coverage and clearer telemetry around tool execution to
  surface timeouts consistently.
- Introduced workspace digest helpers (`WorkspaceDigest`,
  `WorkspaceDigestSection`) so sessions can persist and render cached
  workspace summaries across runs.

### Session & Concurrency

- `Session` now owns an in-process event dispatcher by default, removing boilerplate
  for common setups while preserving the ability to inject a custom dispatcher.
- Added a reusable locking helper around session mutations to guard reducers
  and state updates without sprinkling ad hoc locks.

## v0.9.0 - 2025-11-17

### Podman Sandbox

- Added `PodmanSandboxSection` (renamed from `PodmanToolsSection`) as a
  Podman-backed workspace that mirrors the VFS contract, exposes
  `shell_execute`, `ls`/`read_file`/`write_file`/`rm`, and publishes
  `PodmanWorkspace` slices so reducers can inspect container state. The
  section sits behind the new `weakincentives[podman]` optional extra, is
  re-exported from `weakincentives.tools`, and ships with a spec plus pytest
  marker for Podman-only suites.
- `evaluate_python` is now available inside the Podman sandbox with the same
  return schema as ASTEval but without the 2,000-character cap, and file
  writes now run through `podman cp` to keep binary-safe edits consistent with
  the VFS.
- The `code_reviewer_example.py` prompt auto-detects local Podman connections,
  mounts repositories into the sandbox, and falls back to the in-memory VFS
  when Podman is unavailable so the workflow keeps functioning in both modes.

### Tooling & Runtime

- `PlanningToolsSection`, `VfsToolsSection`, and `AstevalSection` now require
  a live `Session` at construction time, register their reducers immediately,
  and verify that tool handlers execute within the same `Session`/event dispatcher.
  Update custom prompts to pass the session you plan to route through
  `ToolContext`.
- Section registration reinstates strict validation for tool entries and tool
  generics: `Tool[...]` now enforces dataclass result types (or sequences of
  dataclasses), handler annotations must match the declared generics, and
  `ToolContext` exposes typed prompt/adapter protocols so invalid overrides
  fail fast during initialization.

### Prompt Runtime & Overrides

- `Prompt.render` accepts any overrides store implementing the new
  `PromptOverridesStoreProtocol`, and we now export `PromptProtocol`,
  `ProviderAdapterProtocol`, and `RenderedPromptProtocol` for adapters and
  tool authors that need structural typing without importing the concrete
  prompt classes.
- `Prompt` exposes a `structured_output` descriptor (and `RenderedPrompt`
  mirrors it) so runtimes can inspect the resolved dataclass/container
  contract directly instead of juggling separate `output_type`, `container`,
  and `allow_extra_keys` attributes.

### Serde, Logging & Session State

- Introduced canonical JSON typing helpers under `weakincentives.types` (also
  re-exported from the package root) and rewired structured logging to enforce
  JSON-friendly payloads, nested adapter support, and context-preserving
  `StructuredLogger.bind()` calls.
- Dataclass serialization/deserialization now keeps set outputs deterministic,
  respects `exclude_none` inside nested collections, and emits clearer errors
  when schema/constraint validators fail, while session snapshots gained typed
  slice aliases plus an `event_bus` accessor on `Session` for reducer helpers.

## v0.8.0 - 2025-11-15

### Prompt Runtime & Overrides

- `Prompt.render` now accepts override stores and tags directly, removing the
  helper functions by plumbing the parameters through provider adapters and
  the code review CLI so tagged overrides stay consistent end-to-end.
- The code reviewer example gained typed request/response dataclasses, a
  centralized `initialize_code_reviewer_runtime`, plan snapshot rendering, and
  event-bus driven prompt/tool logging so multi-turn runs stay deterministic
  and easy to trace.

### Built-in Tools

- The virtual filesystem suite was rewritten to match the DeepAgents contract:
  new `VfsPath`/`VirtualFileSystem` dataclasses, ASCII/UTF-8 guards,
  list/glob/grep/edit helpers, host mount previews, and refreshed
  exports/specs/tests keep VFS sessions deterministic.
- The ASTEval tool always runs multi-line inputs, dropping the expression-mode
  toggle while refreshing the section template, serde fixtures, and tests to
  reflect the simplified contract.
- Added pytest-powered audits that enforce documentation, slots, tuple
  defaults, and precise typing on every built-in tool dataclass.

### Runtime & Infrastructure

- `Deadline.remaining` now rejects naive datetime overrides and
  `configure_logging` differentiates explicit log levels from environment
  defaults to keep adapters' logging choices intact.
- Raised dependency minimums, added `pytest-rerunfailures` to the dev stack,
  refreshed CI workflow pins, and updated planning/serde tests for the
  stricter type checking.
- Thread-safety docs/tests were rewritten around the shared session plus
  overrides store story, retiring the legacy Sunfish session scaffolding and
  clarifying reducer expectations.

### Documentation

- `AGENTS.md` now mirrors the current repository layout, strict TDD workflow,
  and DbC expectations so contributors have a single source of truth for the
  process.
- The VFS and ASTEval specs were refreshed to describe the expanded file
  tooling and simplified execution contract, keeping the docs aligned with the
  new surfaces.

## v0.7.0 - 2025-11-13

### Sessions & Contracts

- Introduced `Session.reset()` to clear accumulated slices without removing
  reducer registrations, making long-lived interactive flows easier to manage.
- Added a design-by-contract module that exposes `require`, `ensure`,
  `invariant`, and `pure` decorators along with runtime toggles and a pytest
  plugin so projects can opt into contract enforcement when debugging.
- Wrapped the session container with invariants that validate UUID metadata
  and timezone-aware timestamps, wiring the DbC utilities into the public
  export surface to keep runtime guarantees consistent across adapters.

### Prompt Authoring

- Parameterless prompts and sections now accept zero-argument `enabled`
  callables, keeping declarative gating logic concise.
- Centralized structured output payload parsing into shared helpers used by
  the prompt runtime and adapters to keep response handling consistent.

### Events & Telemetry

- Replaced the `PromptStarted` lifecycle event with `PromptRendered`, removed
  wrapper dataclasses, and now emit the rendered prompt plus adapter metadata
  directly to reducers and subscribers.
- Assigned stable UUID identifiers to prompt and tool events while enforcing
  timezone-aware session creation for richer telemetry.

### Tooling & Quality

- Added the `pytest-randomly` plugin to shake out order-dependent test
  assumptions during development.

## v0.6.0 - 2025-11-05

### Prompt & Overrides

- Rebuilt the local prompt overrides store with structured logging, strict
  slug/tag validation, file-level locks, and atomic writes, and taught
  sections and tools to declare an `accepts_overrides` flag so override
  pipelines target only opted-in surfaces.

### Tool Runtime

- Introduced the typed `ToolContext` passed to every handler (rendered prompt,
  adapter, session, and bus) and updated planning, VFS, and ASTEval sections
  to pull session state from the context while validating handler signatures.
- Added configurable planning strategy templates and tighter reducer wiring
  for plan updates, aligning built-in planning tools with the new context and
  override controls.
- Made the ASTEval integration an optional extra and removed the signal-based
  timeout shim while keeping the tool quiet by default.

### Session & Adapters

- Hardened session concurrency with RLock-protected reducers, snapshot
  restores, and new thread-safety regression tests/specs while the event dispatcher
  now emits structured logs for publish failures.
- Centralized adapter protocols and the conversation runner, enforcing that
  adapters always supply a session and event dispatcher before executing prompts and
  improving tool invocation error reporting.

### Logging & Telemetry

- Added a structured logging facility used across sessions, event dispatchers, and
  prompt overrides, alongside dedicated unit tests and README guidance for
  configuring INFO-level output in the code review example.

### Documentation, Examples & Tooling

- Replaced the multi-file code review demo with an updated
  `code_reviewer_example.py` that mounts repositories, tracks tool calls, and
  emits structured logs, removing the legacy example modules/tests.
- Expanded the specs portfolio with new documents for the CLI, logging schema,
  tool context, planning strategies, prompt composition, and thread safety,
  plus refreshed README sections.
- Added a `make demo` target, tightened the Bandit compatibility shim, and
  refreshed dependency locks.

## v0.5.0 - 2025-11-02

### Session & State

- Added session snapshot capture and rollback APIs to persist dataclass
  slices, with new helpers and regression coverage.

### Prompt Overrides

- Introduced `LocalPromptOverridesStore` for persisting prompt overrides on
  disk with strict validation and a README walkthrough.
- Renamed the prompt overrides protocol and exports to `PromptOverridesStore`
  so runtime and specs share consistent terminology.

### Tool Execution & Adapters

- Centralized adapter tool execution through a shared helper, removing
  redundant aliases and unifying reducer rollback handling.
- Tool handlers now emit structured JSON responses (including a `success`
  flag) and adapters treat failures as non-fatal session events.

### Events & Telemetry

- Event dispatchers now return a `PublishResult` summary capturing handler failures
  and expose `raise_if_errors` for aggregated exceptions.

### Tooling & Quality

- Enabled Pyright strict mode and tightened type contracts across adapters,
  tool serialization, and session snapshot plumbing.

### Documentation

- Added specs covering session snapshots, local prompt overrides, and tool
  error handling.
- Expanded ASTEval guidance with tool invocation examples and refreshed README
  tutorials with spec links and symbol search tooling.

## v0.4.0 - 2025-11-01

### Evaluation Tools

- Added `AstevalSection` to expose an `evaluate_python` tool that runs inside
  the sandbox, bridges the virtual filesystem, captures stdout/stderr,
  templates writes, and enforces timeouts.
- Declared `asteval>=1.0.6` as a runtime dependency and documented the
  synchronous handler contract in the ASTEval spec.

### Virtual Filesystem

- Extended the VFS to accept UTF-8 content for writes and host mounts,
  refreshed prompt guidance, and mounted the sunfish README to demonstrate
  multibyte data.

### Examples

- Added a code review agent example that exercises the VFS helpers safely and
  surfaces tool call history through the console session scaffold.
- Wired the ASTEval section into the code review prompt example so agents can
  invoke the `evaluate_python` tool during reviews.

### Typing & Tests

- Expanded type annotations across prompts, adapters, and examples, removed
  Ruff annotation ignores, and broadened the pytest suite to cover new
  behaviors and VFS regression cases while updating coverage configuration.

### Documentation

- Replaced the README quickstart with a step-by-step code review tutorial that
  contrasts Weak Incentives with LangGraph and DSPy.
- Expanded the ASTEval specification with the section entry point, full-VFS
  access guidance, and updated timeout expectations.
- Removed the legacy `docs/` pages now superseded by `specs/`.

## v0.3.0 - 2025-11-01

### Prompt & Rendering

- Renamed and reorganized the prompt authoring primitives (`MarkdownSection`,
  `SectionNode`, `Tool`, `ToolResult`, `parse_structured_output`, …) under the
  consolidated `weakincentives.prompt` surface.
- Prompts now require namespaces and explicit section keys so overrides line
  up with rendered content and structured response formats.
- Added tool-aware prompt version metadata and the `PromptVersionStore`
  override workflow to track section edits and tool changes across revisions.

### Session & State

- Introduced the `Session` container with typed reducers/selectors that
  capture prompt outputs and tool payloads directly from emitted events.
- Added helper reducers (`append`, `replace_latest`, `upsert_by`) and
  selectors (`select_latest`, `select_where`) to simplify downstream state
  management.

### Built-in Tools

- Shipped the planning tool suite (`PlanningToolsSection` plus typed plan
  dataclasses) for creating, updating, and tracking multi-step execution plans
  inside a session.
- Added the virtual filesystem tool suite (`VfsToolsSection`) with host-mount
  materialization, ASCII write limits, and reducers that maintain a versioned
  snapshot.

### Events & Telemetry

- Implemented the event dispatcher with `ToolInvoked` and `PromptExecuted` payloads
  and wired adapters/examples to publish them for sessions or external
  observers.

### Adapters

- Added a LiteLLM adapter behind the `litellm` extra with tool execution
  parity and structured output parsing.
- Updated the OpenAI adapter to emit native JSON schema response formats,
  tighten `tool_choice` handling, avoid echoing tool payloads, and surface
  richer telemetry.

### Examples

- Rebuilt the OpenAI and LiteLLM demos as shared CLI entry points powered by
  the new code review agent scaffold, complete with planning and virtual
  filesystem sections.

### Tooling & Packaging

- Lowered the supported Python baseline to 3.12 (the repository now pins 3.14
  for development) and curated package exports to match the reorganized
  modules.
- Added OpenAI integration tests and stabilized the tool execution loop used
  by the adapters.
- Raised the optional `litellm` extra to require the latest upstream release.

### Documentation

- Documented the planning and virtual filesystem tool suites, optional
  provider extras, and updated installation guidance.
- Refreshed the README and supporting docs to highlight the new prompt
  workflow, adapters, and development tooling expectations.

## v0.2.0 - 2025-10-29

### Highlights

- Launched the prompt composition system with typed `Prompt`, `Section`, and
  `TextSection` building blocks, structured rendering, and placeholder
  validation backed by comprehensive tests.
- Added tool orchestration primitives including the `Tool` dataclass, shared
  dataclass handling, duplicate detection, and prompt-level aggregation
  utilities.
- Delivered stdlib-only dataclass serde helpers (`parse`, `dump`, `clone`,
  `schema`) for lightweight validation and JSON serialization.

### Integrations

- Introduced an optional OpenAI adapter behind the `openai` extra that builds
  configured clients and provides friendly guidance when the dependency is
  missing.

### Developer Experience

- Tightened the quality gate with quiet wrappers for Ruff, Ty, pytest (100%
  coverage), Bandit, Deptry, and pip-audit, all wired through `make check`.
- Adopted Hatch VCS versioning, refreshed `pyproject.toml` metadata, and
  standardized automation scripts for releases.

### Documentation

- Replaced `WARP.md` with a comprehensive `AGENTS.md` handbook describing
  workflows, TDD guidance, and integration expectations.
- Added prompt and tool specifications under `specs/` and refreshed the README
  to highlight the new primitives and developer tooling.

## v0.1.0 - 2025-10-22

Initial repository bootstrap with the package scaffold, testing and linting
toolchain, CI configuration, and contributor documentation.

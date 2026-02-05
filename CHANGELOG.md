# Changelog

Release highlights for weakincentives.

## Unreleased

*Commits reviewed: 2026-01-29 (be065a9) through 2026-02-03 (642048e)*

### TL;DR

This release focuses on **developer experience and type safety improvements**.
**Faster local testing** via pytest-testmon runs only tests affected by your
changes while CI still enforces 100% coverage. The **Claude Agent SDK adapter**
now enables **transcript collection by default** and emits **RenderedTools
events** for better observability of tool schemas during prompt evaluation.
**Feedback messages** switch to **XML-style tags** (`<feedback>`, `<blocker>`)
for clearer LLM parsing. **BundleConfig** replaces the simple `debug_bundle_dir`
path, enabling **external storage handlers** for cloud uploads. The **Session**
class is **refactored into focused subsystems** (SliceStore, ReducerRegistry,
SessionSnapshotter) for better maintainability. **Type safety** improves with
**TypedDict definitions** for all LLM API payloads and a **generic Snapshotable
protocol**. The **debug web UI** gains **comprehensive JavaScript testing** with
77 unit tests via Bun. **CI is parallelized** into 6 test groups for faster
feedback. A new **OpenCode ACP Adapter specification** documents delegating
execution to OpenCode via the Agent Client Protocol. **Documentation** is
reorganized with the **quickstart guide rewritten** to use a starter project and
**guardrails specs consolidated** into a unified document.

---

### Breaking Changes

#### EvalLoopConfig.debug_bundle_dir → debug_bundle

The `debug_bundle_dir: Path | None` field is replaced with `debug_bundle:
BundleConfig | None`, enabling advanced bundle management features like storage
handlers and conditional creation.

**Migration:**
```python
# Old ❌
config = EvalLoopConfig(debug_bundle_dir=Path("/bundles"))

# New ✅
from weakincentives.debug.bundle import BundleConfig
config = EvalLoopConfig(debug_bundle=BundleConfig(target=Path("/bundles")))
```

#### Transcript Collection Now Enabled by Default

The Claude Agent SDK adapter now enables transcript collection by default for
better observability. To restore the previous behavior:

```python
config = ClaudeAgentSDKClientConfig(transcript_collection=None)
```

---

### New Features

#### Intelligent Test Selection with pytest-testmon

Local test runs now only execute tests affected by your changes using
coverage-based selection:

- First run builds `.testmondata` coverage database
- Subsequent runs skip unaffected tests automatically
- CI still validates 100% coverage on all tests
- No configuration required—automatically detected via `CI` environment variable

```bash
make test      # Local: runs only affected tests
CI=true make test  # CI: runs full suite with 100% coverage
```

#### Tool Schema Rendering and Event Correlation

The Claude Agent SDK adapter now emits `RenderedTools` events alongside
`PromptRendered` events, providing visibility into which tools are available
for each prompt evaluation:

- Tool schemas (name, description, parameters) extracted and included in events
- Events correlated via `render_event_id` for tracking
- Both events share `session_id` and `created_at` for consistency

#### XML-Style Feedback Formatting

Feedback messages now use semantic XML tags for clearer LLM parsing:

```xml
<feedback provider='DeadlineFeedback'>
You have 30 seconds remaining.
</feedback>

<blocker>
You have 2 incomplete tasks. Please complete them before producing output.
</blocker>
```

The `<blocker>` tag explicitly marks blocking issues for agent comprehension.

#### External Storage Handler for Bundles

Upload bundles to external storage (S3, GCS, etc.) after creation:

```python
config = EvalLoopConfig(
    debug_bundle=BundleConfig(
        target=Path("/bundles"),
        storage_handler=S3StorageHandler(bucket="my-bucket"),
        enabled=True,
    )
)
```

Storage handler failures are logged but don't fail evaluations.

#### OpenCode ACP Adapter Specification

New specification (`specs/OPENCODE_ACP_ADAPTER.md`) documents `OpenCodeACPAdapter`
for delegating agentic execution to OpenCode via the Agent Client Protocol while
retaining WINK's prompt composition and session telemetry. Features:

- MCP server bridging via Claude Agent SDK infrastructure
- Structured output via dedicated MCP tool (no text parsing)
- Workspace isolation with file boundary enforcement
- Session persistence and reuse with validation

---

### Improvements

#### Generic Snapshotable Protocol

The `Snapshotable` protocol now accepts a type parameter for better type safety:

```python
# Before
class MyResource(Snapshotable):
    def snapshot(self, *, tag: str | None = None) -> Any: ...
    def restore(self, snapshot: Any) -> None: ...

# After
class MyResource(Snapshotable[MySnapshot]):
    def snapshot(self, *, tag: str | None = None) -> MySnapshot: ...
    def restore(self, snapshot: MySnapshot) -> None: ...
```

#### TypedDict Definitions for LLM API Payloads

New `_api_types.py` module provides type-safe definitions for all LLM provider
payloads, replacing generic `dict[str, Any]` annotations:

- `MessageDict`, `AssistantMessageDict`, `ToolMessageDict` for messages
- `ToolSpecDict`, `FunctionSpecDict` for tool specifications
- `LLMRequestParams` for generation parameters
- `ProviderPayload`, `ToolArguments` for decoded payloads

Enables static type checking and IDE autocompletion across all adapters.

#### Session Refactored into Specialized Subsystems

Session's internal state management extracted into three focused components:

- **SliceStore**: Manages slice storage and policy-based factory selection
- **ReducerRegistry**: Routes events to registered reducers
- **SessionSnapshotter**: Handles snapshot creation and restoration

Session remains a thin facade coordinating these subsystems. No public API
changes; internal structure is cleaner and more testable.

#### Type Safety in Serialization

Replaced blanket pyright suppression in `serde/dump.py` with explicit `cast()`
calls, improving code clarity without changing behavior.

---

### Documentation

#### Quickstart Guide Redesigned

The quickstart guide now prioritizes hands-on learning:

- Clone the [WINK starter project](https://github.com/weakincentives/starter)
- Run a working agent immediately with `make` commands
- Learn concepts through a concrete "secret trivia game" example
- Added development workflow commands and `wink` CLI debugging
- Preserved pip installation as alternative for starting from scratch

#### CLAUDE.md Enhanced with Architectural Guidance

Added comprehensive "Core Philosophy" and "Guiding Principles" sections:

- **The prompt is the agent**: Prompts bundle instructions and tools together
- **Event-driven state**: Immutable state via pure reducers
- **Definition vs Harness**: Clear boundary between agent definition and runtime
- **Policies over Workflows**: Declarative constraints over prescriptive steps
- **Transactional Tools**: Atomic tool calls with automatic rollback
- Added Key Specs reference table for design documents

#### Guardrails Specification Consolidated

Three separate specs (FEEDBACK_PROVIDERS.md, TASK_COMPLETION.md, tool policies
from TOOLS.md) merged into unified `specs/GUARDRAILS.md`:

- Tool policies: Gate tool invocations (hard block)
- Feedback providers: Soft guidance over time (advisory)
- Task completion: Verify goals before stopping

---

### CI / Infrastructure

#### Parallelized Test Execution

Tests now run in 6 parallel groups for faster CI feedback:

- `test-group-1`: Adapters (~220 tests)
- `test-group-2`: CLI + Contrib (~150 tests)
- `test-group-3`: Evals + Serde (~280 tests)
- `test-group-4`: Prompt + Prompts (~400 tests)
- `test-group-5`: Runtime (~290 tests)
- `test-group-6`: Tools + Root (~540 tests)

Coverage combined after all groups complete with unified 100% enforcement.
New `make test-parallel` target for local testing that mirrors CI behavior.

#### Simplified Change Detection

Removed 12 module-specific change detection outputs. Heavy tests (formal
verification, property tests) now gated by single `run_heavy_tests` flag.
Standard tests run on every PR for consistent coverage.

#### JavaScript Testing Infrastructure

Added Bun test runner for debug web UI with 77 unit tests:

- Extracted 12 pure utility functions to `lib.js` for testability
- 100% coverage requirement for pure functions
- Integrated with `make test` and `make check` targets
- Biome cognitive complexity threshold lowered to 8

#### Shared CI Setup Action

New `.github/actions/setup-env` composite action centralizes:

- Python installation via uv
- Dependency synchronization with frozen resolution
- Bun and Biome caching for faster runs

#### Skip CI for Docs-Only Changes

Static analysis and test jobs now skip for documentation-only PRs while
`docs-check` provides feedback path.

---

### Dependencies

#### Major Updates

| Package | Old | New | Notes |
|---------|-----|-----|-------|
| redis | 5.0.0 | 7.1.0 | Major version bump |
| fastapi | 0.115.0 | 0.128.0 | 13 minor versions |
| uvicorn | 0.30.0 | 0.40.0 | 10 minor versions |
| openai | 2.8.0 | 2.16.0 | 8 minor versions |
| claude-agent-sdk | 0.1.15 | 0.1.27 | 12 patch releases |
| pip | 25.3 | 26.0 | Major version bump |

#### Other Updates

- litellm: 1.79.3 → 1.81.6
- pyjwt: 2.10.1 → 2.11.0
- tqdm: 4.67.1 → 4.67.2
- pyyaml: 6.0.0 → 6.0.3
- asteval: 1.0.7 → 1.0.8
- podman: 5.6.0 → 5.7.0
- pytest-testmon: Added (≥2.1.0)
- Development tools: ruff, pyright, pytest, hypothesis, bandit, pip-audit updated

---

### Internal

#### Test Infrastructure Improvements

- Consolidated mock exception classes into `tests/adapters/claude_agent_sdk/error_mocks.py`
- Centralized CLI test helpers (FakeLogger, FakeContextManager) into `tests/cli/helpers.py`
- Enhanced `tests/helpers/__init__.py` with comprehensive documentation
- Refactored `wink query` tests into focused modules (helpers, database, CLI)

#### Bug Fixes

- JavaScript tests now properly executed in CI test job (Bun was only installed
  in static-analysis job)
- Improved test coverage for git remote credential redaction

---

## v0.24.0 — 2026-01-30

*Commits reviewed: 2026-01-25 (e5a00a4) through 2026-01-29 (a1c5996)*

### TL;DR

This release introduces **TranscriptCollector**, a hook-driven system replacing
the old log aggregator for real-time Claude Agent SDK transcript collection with
automatic sub-agent discovery. The **debug UI receives major upgrades**: a new
**Transcript tab** with filtering/search, an **Environment tab** displaying system
and runtime context, **virtual scrolling** for large lists, a **zoom modal** for
detailed entry inspection with keyboard navigation, and **image file support** in
the file viewer. **AgentLoop** (renamed from MainLoop) gains a **transforming
`finalize()` hook** that can modify outputs post-execution. **Scoped field
visibility** lets you hide dataclass fields from LLM structured outputs while
keeping them for post-processing. **Bundle lifecycle management** adds retention
policies (max count/age/size) and external storage handler support for cloud
uploads. The **Task view is removed**—request/response data now lives in session
state. Frontend code now enforces **Biome linting**. The codebase modernizes to
**PEP 695 type syntax** and gains **comprehensive docstrings** across all 26
public modules.

---

### Breaking Changes

#### MainLoop Renamed to AgentLoop

The `MainLoop` class and all related types have been renamed to `AgentLoop` for
clarity. This affects all imports and type annotations.

**Migration:**
```python
# Old ❌
from weakincentives.runtime import MainLoop, MainLoopConfig, MainLoopRequest, MainLoopResult

# New ✅
from weakincentives.runtime import AgentLoop, AgentLoopConfig, AgentLoopRequest, AgentLoopResult
```

All related files renamed: `main_loop.py` → `agent_loop.py`,
`main_loop_types.py` → `agent_loop_types.py`. Spec file renamed:
`MAIN_LOOP.md` → `AGENT_LOOP.md`.

#### AgentLoop.finalize() Signature Change

The `finalize()` method now receives the parsed output and returns a (possibly
transformed) output. Subclasses overriding this method must update their signature.

**Old signature:**
```python
def finalize(self, prompt: Prompt[OutputT], session: Session) -> None:
```

**New signature:**
```python
def finalize(
    self,
    prompt: Prompt[OutputT],
    session: Session,
    output: OutputT | None,
) -> OutputT | None:
```

**Migration:** Add the `output` parameter to your override and return it (or a
transformed version).

#### Debug UI Task View Removed

The dedicated Task view tab and its API endpoints (`/api/request/input`,
`/api/request/output`) have been removed. Request and response data is now
captured in session state and viewable through the Sessions tab instead.

- Keyboard shortcuts reduced from 6 tabs to 5 tabs
- Filesystem view moved from key `5` to key `4`

---

### New Features

#### TranscriptCollector System

Replaces `ClaudeLogAggregator` with a hook-driven transcript collection system
that provides real-time collection from Claude Agent SDK sessions.

**Key capabilities:**
- Uses SDK hooks (`SubagentStart`, `SubagentStop`) for immediate transcript path
  discovery instead of directory polling
- Automatic sub-agent transcript discovery and tailing
- Emits structured DEBUG logs with full context (`prompt_name`, `source`,
  `entry_type`, `raw_json`)
- File rotation detection via inode tracking
- Configurable poll intervals and max read bytes

**Configuration:**
```python
config = ClaudeAgentSDKClientConfig(
    transcript_collection=TranscriptCollectorConfig(
        poll_interval=0.25,
        subagent_discovery_interval=1.0,
        emit_raw_json=True,
    )
)
```

#### Transcript Analysis Views

New `wink query` database schema (v5) with normalized transcript data:

**New `transcript` table** with columns: `timestamp`, `prompt_name`,
`transcript_source`, `sequence_number`, `entry_type`, `role`, `content`,
`tool_name`, `tool_use_id`, `raw_json`, `parsed`

**Four pre-built SQL views:**
- `transcript_flow` — Conversation flow with truncated previews
- `transcript_tools` — Tool calls paired with their results
- `transcript_thinking` — Thinking block analysis with length metrics
- `transcript_agents` — Subagent metrics and hierarchy

**New `/api/transcript` endpoint** with filtering by source, entry type, and
full-text search.

**New Transcript tab** in `wink debug` with:
- Filter chips for sources (main vs. subagents) and entry types
- Full-text search on message content
- Drilldown into raw JSON

#### Environment Data Tables

New Environment tab in `wink debug` displaying captured runtime context:

**Six new database tables:**
- `env_system` — OS, kernel, architecture, CPU, memory, hostname
- `env_python` — Version, implementation, executable, virtualenv status
- `env_git` — Commit SHA, branch, dirty status, remotes, tags
- `env_container` — Runtime type, container ID, image, containerized flag
- `env_vars` — Filtered environment variables
- `environment` — Flat key-value table with prefixed naming

**New `/api/environment` endpoint** returning structured JSON.

#### Virtual Scrolling

Efficient windowed rendering for logs and transcript lists in `wink debug`:

- Renders only visible items plus configurable buffer (default 10 items)
- IntersectionObserver-based infinite scroll
- ResizeObserver for responsive layout updates
- Automatic garbage collection of off-screen DOM elements
- Falls back gracefully to traditional rendering when needed

#### Zoom Modal for Entry Inspection

Click any transcript entry to open a full-screen modal with:

- Two-panel layout: formatted content (left) and metadata/JSON tree (right)
- Tool call + result pairs displayed side-by-side
- Collapsible/expandable JSON tree with syntax highlighting
- Keyboard navigation: `J`/`K` or arrows for prev/next, `Escape` to close
- Copy JSON button for exporting entry data

Transcript list entries now display in compact mode (60px max-height with
fade-out) with automatic tool call + result pairing.

#### Bundle Retention Policy

Automatic cleanup of old debug bundles based on configurable limits:

```python
config = BundleConfig(
    target=Path("./debug/"),
    retention=BundleRetentionPolicy(
        max_bundles=10,           # Keep at most N bundles
        max_age_seconds=86400,    # Delete bundles older than 24 hours
        max_total_bytes=500_000_000,  # Keep under 500MB total
    ),
)
```

All limits are optional; when multiple are set, the most restrictive wins.
Retention errors are logged as warnings but never fail the request.

#### External Storage Handler

Protocol for uploading bundles to external storage after creation:

```python
class S3StorageHandler:
    bucket: str
    prefix: str = "debug-bundles/"

    def store_bundle(self, bundle_path: Path, manifest: BundleManifest) -> None:
        key = f"{self.prefix}{manifest.bundle_id}.zip"
        s3_client.upload_file(str(bundle_path), self.bucket, key)

config = BundleConfig(
    target=Path("./debug/"),
    storage_handler=S3StorageHandler(bucket="my-bucket"),
)
```

Retention is applied before storage handler invocation, so only surviving
bundles are uploaded.

#### Scoped Field Visibility

Hide dataclass fields from LLM structured outputs while keeping them available
for post-processing:

```python
from weakincentives.serde import HiddenInStructuredOutput

@dataclass
class AnalysisResult:
    summary: str           # LLM generates this
    confidence: float      # LLM generates this

    # Hidden from LLM — populated in finalize()
    processing_time_ms: Annotated[int, HiddenInStructuredOutput()] = 0
    model_version: Annotated[str, HiddenInStructuredOutput()] = ""
```

- `schema(..., scope=SerdeScope.STRUCTURED_OUTPUT)` excludes hidden fields
- `parse(..., scope=SerdeScope.STRUCTURED_OUTPUT)` skips hidden fields, uses defaults
- `dump()` always serializes all fields (unchanged)

The response parser and structured output modules automatically use the
`STRUCTURED_OUTPUT` scope.

#### Image File Support in Debug Viewer

The filesystem tab in `wink debug` now displays image files inline instead of
showing "binary file cannot be displayed":

- Supports PNG, JPG, JPEG, GIF, WebP, SVG, ICO, and BMP formats
- Case-insensitive extension matching
- Responsive sizing (max 70% viewport height)
- Base64 encoding with MIME type whitelisting for security

---

### Improvements

#### Output Transformation in finalize()

`AgentLoop.finalize()` can now transform the model output before returning:

```python
class MyLoop(AgentLoop[Input, Output]):
    def finalize(self, prompt, session, output):
        if output is not None:
            return replace(output, timestamp=datetime.now(UTC))
        return output
```

#### Increased File Write Size Limit

The maximum file write size limit has been increased from 48KB to 32MB across
all filesystem operations and VFS tools. This enables agents to write larger
files such as database dumps, generated images, or bundled outputs without
hitting size constraints.

#### Session State for Request/Response

Request and response data is now captured in session state via three new event
types: `LoopRequestState`, `LoopRawResponse`, `LoopFinalResponse`. This enables
viewing the data through the Sessions tab and standard session inspection APIs.

---

### Internal Changes

#### Claude Agent SDK Adapter Refactored to Use ClaudeSDKClient

The Claude Agent SDK adapter now uses `ClaudeSDKClient` directly instead of the
high-level `sdk.query()` interface. This provides direct control over session
lifecycle management with explicit `connect()`/`disconnect()` semantics, better
error handling and resource cleanup, and lays the foundation for future
enhancements like session reuse and multi-turn conversations.

#### PEP 695 Type Syntax

Type aliases modernized to Python 3.12+ `type` statement syntax:

```python
# Old
ContractCallable = Callable[..., bool | tuple[bool, str]]

# New
type ContractCallable = Callable[..., bool | tuple[bool, str]]
```

Affected modules: `dbc`, `prompt/_visibility.py`, `prompt/errors.py`,
`runtime/session/_types.py`, `resources/binding.py`. Also removed unnecessary
`builtins` module usage in `filesystem_memory.py`.

#### Frontend Linting with Biome

Added Biome configuration for JavaScript linting/formatting:

- New `package.json`, `biome.json` configuration files
- `make biome` and `make biome-fix` targets
- `make check` now includes Biome validation

Rules enforce complexity limits (max cognitive complexity 25), correctness
(unused variables as errors), performance warnings, and security checks
(forbids dangerously setting innerHTML).

#### Dependency Updates

- `mcp` 1.25.0 → 1.26.0 (minor)
- `openai` 2.15.0 → 2.16.0 (minor)
- `claude-agent-sdk` 0.1.22 → 0.1.25 (patch)
- `cryptography` 46.0.3 → 46.0.4 (security patch)
- `coverage`, `huggingface-hub`, `hypothesis`, `litellm`, `multidict`,
  `python-multipart`, `rich` patch updates
- Removed `grpcio` as transitive dependency
- `actions/checkout` v5 → v6 in CI workflows

#### Test Infrastructure

- Improved TOCTOU test reliability for bundle retention on filesystems with
  aggressive inode reuse (e.g., tmpfs)
- Added missing test case for bundle deletion without identity tracking

#### Dependabot Configuration

Added `.github/dependabot.yml` for automated dependency updates:

- Python packages: weekly on Mondays, grouped by category (dev, adapters, web,
  contrib), minor/patch only
- GitHub Actions: weekly, minor/patch only
- Both assigned to `weakincentives/maintainers` team

---

### Documentation

#### Comprehensive Module Docstrings

All 26 public `__init__.py` modules now include NumPy/SciPy-style docstrings
(+5,000 lines) with:

- Module overview and use cases
- Organized class/function descriptions by category
- Practical runnable examples
- Cross-references to related modules

#### Specification Updates

- **TRANSCRIPT_COLLECTION.md** — New spec for TranscriptCollector architecture
- **AGENT_LOOP.md** — Updated for finalize() signature change
- **DATACLASSES.md** — New section on scoped field visibility
- **DEBUG_BUNDLE.md** — Bundle naming and retention policy documentation
- **CLAUDE_AGENT_SDK.md** — Default model changed from Opus 4.5 to Sonnet 4.5
- Various API clarifications across EVALS.md, MAILBOX.md, SESSIONS.md,
  TOOLS.md, RESOURCE_REGISTRY.md, VERIFICATION.md

#### README

Added DeepWiki badge linking to external documentation.

## v0.23.0 - 2026-01-25

*Commits reviewed: 2026-01-23 (5274d9d) through 2026-01-24 (6f62164)*

### TL;DR

This release introduces **debug bundle support for EvalLoop**, enabling full
observability into evaluation runs with session state, logs, and eval-specific
metadata. The **serialization system** undergoes a major overhaul: polymorphic
type embedding (`__type__`) is replaced with **generic alias syntax** at parse
time (`parse(Wrapper[Data], data)`), and **AST-based type resolution** now
handles complex nested generics and `Literal` types. **Design-by-Contract is now
always enabled**—contracts run in production, not just tests. A new
**`MailboxWorker` base class** extracts common mailbox-driven processing
infrastructure, reducing code duplication between AgentLoop and EvalLoop. The
**`Experiment` class moves to package root** to resolve circular imports. Bug
fixes address **RedisMailbox generic type deserialization** and prevent **data
inconsistency when bundle finalization fails**. **Section subclasses now
automatically infer `_params_type`** from generic base classes, eliminating
manual type assignment. New documentation includes a comprehensive **Query
guide** for SQL-based bundle analysis.

---

### Breaking Changes

#### Design-by-Contract Always Enabled

**Design-by-Contract is now always enabled by default.** DbC checks are enforced
in all contexts (tests and production) and cannot be globally disabled. This
ensures contracts catch bugs early in production rather than only during testing.
The implementation now uses `ContextVar` for thread-safe and async-safe suspension
tracking.

- Removed `enable_dbc()` and `disable_dbc()` global functions
- Removed `WEAKINCENTIVES_DBC` environment variable
- Renamed `dbc_enabled()` to `dbc_suspended()` for clarity
- Added `dbc_suspended()` context manager for temporarily disabling checks in
  performance-sensitive code paths

**Migration:**
- Remove calls to `enable_dbc()` and `disable_dbc()` (no longer needed)
- Replace `with dbc_enabled(False):` with `with dbc_suspended():`
- Replace `with dbc_enabled():` or `with dbc_enabled(True):` with nothing (DbC
  is now always active)

#### Serialization API Overhaul

**Polymorphic type embedding removed.** The `__type__` field approach is replaced
with explicit type specification at parse time using generic alias syntax.

**Removed parameters from `dump()`:**
- `include_dataclass_type: bool` — no longer embeds type metadata
- `type_key: str` — no longer customizable

**Removed parameters from `parse()`:**
- `allow_dataclass_type: bool` — types must now be specified upfront
- `type_key: str` — no longer reads embedded type metadata
- `cls` parameter is now mandatory (was optional)

**Migration:**
- Replace `dump(obj, include_dataclass_type=True)` with `dump(obj)` and store
  type information separately (e.g., in a database column or message envelope)
- Replace `parse(None, data, allow_dataclass_type=True)` with
  `parse(KnownType, data)` where the type is determined from context
- For generic dataclasses, use generic alias syntax: `parse(Wrapper[Data], data)`

#### Bundle API Rename

**`BundleWriter.write_eval()` renamed to `write_metadata(name, data)`** for
better separation of concerns. The bundle layer is now generic, not eval-specific.

**Migration:**
- Replace `ctx.write_eval(eval_info)` with `ctx.write_metadata("eval", eval_info)`

#### Experiment Class Relocated

**`Experiment` moved from `evals/_experiment.py` to `experiment.py`** (package
root) to resolve circular import issues.

**Migration:**
- Replace `from weakincentives.evals._experiment import Experiment` with
  `from weakincentives.experiment import Experiment`
- The public API `from weakincentives.evals import Experiment` continues to work

---

### New Features

#### Debug Bundle Support for EvalLoop

EvalLoop now supports debug bundle creation, providing full observability into
evaluation runs.

**Configuration:**
```python
config = EvalLoopConfig(debug_bundle_dir=Path("/tmp/eval-bundles"))
```

**Bundle contents:**
- Session state before/after execution
- Application logs during execution
- Request input (sample and experiment)
- Response output from AgentLoop
- `eval.json` metadata: `sample_id`, `experiment_name`, `score`, `latency_ms`,
  optional `error`
- Environment information

**New fields:**
- `EvalLoopConfig.debug_bundle_dir: Path | None` — enables bundling when set
- `EvalResult.bundle_path: Path | None` — path to created bundle
- `EvalResult.experiment_name: str | None` — experiment identifier

#### MailboxWorker Base Class

A new abstract base class `MailboxWorker[RequestT, ResponseT]` extracts common
mailbox-driven processing infrastructure:

- Message polling with configurable iterations/timeouts
- Automatic lease extension via `LeaseExtender`
- Graceful shutdown with timeout
- Context manager protocol

Both `AgentLoop` and `EvalLoop` now extend this base class, eliminating ~500+
lines of duplicate code.

#### AgentLoop Bundle API

New `AgentLoop.execute_with_bundle()` context manager enables bundled execution
with custom metadata injection:

```python
with agent_loop.execute_with_bundle(request, bundle_target=path) as ctx:
    # Access ctx.response, ctx.session, ctx.latency_ms
    ctx.write_metadata("custom", {"key": "value"})
```

#### Generic Alias Serialization

Full support for generic dataclass serialization using type alias syntax:

```python
@dataclass
class Wrapper[T]:
    value: T

# Parse with concrete type
data = {"value": {"name": "test"}}
result = parse(Wrapper[MyData], data)  # T resolved to MyData
```

Supports nested generics: `parse(Outer[Inner[int]], data)`

#### AST-Based Type Resolution

Complex generic type annotations are now resolved using AST parsing instead of
simple string matching. This enables proper handling of:

- Nested generics: `Container[Inner[T]]`
- Union types: `str | int`
- Literal types: `Literal["foo", 1, True, -1]`
- Forward references with postponed evaluation (`from __future__ import annotations`)

**Security improvement:** Type resolution no longer uses `eval()`. Only safe,
well-known types from `builtins` and `typing` modules are resolved.

#### Automatic Section Type Inference

Section subclasses now automatically infer `_params_type` from generic base
classes. When creating `class MySection(MarkdownSection[MyParams])`, the params
type is propagated automatically—no manual `_params_type` assignment needed.

```python
@dataclass
class MyParams:
    value: str

# _params_type is automatically inferred - no explicit setting required
class MySection(MarkdownSection[MyParams]):
    pass
```

This works through `__init_subclass__` which propagates `_params_type` from
specialized base classes created by `__class_getitem__`.

---

### Bug Fixes

#### RedisMailbox Generic Type Deserialization

Fixed `RedisMailbox._deserialize()` to properly handle generic type aliases like
`AgentLoopRequest[T]`. The old `hasattr(type, "__dataclass_fields__")` check
failed for generic aliases; now uses `get_origin()` and `is_dataclass()` for
correct detection.

#### Circular Import Resolution

Moved `Experiment` class to package root to break circular import chain:
`runtime → evals → prompt → resources → runtime`. This fixes deserialization of
`EvalRequest` and `AgentLoopRequest` where nested `Experiment` objects remained
as dicts instead of proper instances.

#### Bundle Finalization Error Handling

Fixed data inconsistency when bundle finalization fails after successful
execution. The system now tracks execution state and reuses existing results
instead of re-executing the sample, preventing inconsistent data from multiple
executions.

---

### Documentation

#### Query Guide

New comprehensive guide (`guides/query.md`) for SQL-based debug bundle analysis:
- `wink query` CLI usage with `--schema`, `--table`, `--export-jsonl` options
- Complete database schema reference (core tables, optional tables, dynamic
  slice tables)
- Pre-built SQL views: `tool_timeline`, `native_tool_calls`, `error_summary`
- Common query patterns for errors, tool performance, session state, token usage
- Real-world debugging scenarios with example queries

#### README Onboarding Update

Refreshed onboarding guidance with a "hands-on first" approach:
- Points new users to the starter repository for immediate experimentation
- Repositions guides as supplementary resource for improvement
- Acknowledges learning-by-doing preference

#### Spec and Guide Updates

- `specs/DBC.md` — Updated for always-on enforcement model
- `specs/DEBUG_BUNDLE.md` — Added EvalLoop integration and `eval.json` schema
- `specs/EVALS.md` — Added `EvalLoopConfig`, `experiment_name`, `bundle_path`
- `specs/EXPERIMENTS.md` — Updated implementation location
- `specs/DATACLASSES.md` — Removed `type_key`, added generic alias examples
- `guides/debugging.md` — Added EvalLoop bundle example
- `guides/evaluation.md` — New "Debug Bundles for Evaluations" section
- `guides/serialization.md` — Replaced polymorphic section with generic alias
  documentation

---

### Dependencies

Updated to latest versions:
- `claude-agent-sdk`: 0.1.21 → 0.1.22
- `huggingface-hub`: 1.3.2 → 1.3.3
- `hypothesis`: 6.150.2 → 6.150.3
- `rich`: 14.2.0 → 14.3.0
- `ruff`: 0.14.13 → 0.14.14 (sdist 25% smaller)

---

### Tests

#### Comprehensive Loop Type Serialization Tests

New test suite (`tests/serde/test_loop_serde.py`) with 50+ test cases covering:
- `Score`, `Sample`, `Experiment` round-trip serialization
- `EvalRequest`, `EvalResult` with generic types and error handling
- `AgentLoopConfig`, `AgentLoopRequest`, `AgentLoopResult` with complex nested types
- `RunContext`, `Budget`, `Deadline` serialization
- Clone operations and type coercion from JSON
- Edge cases: unspecialized generics, missing fields, invalid data

#### EvalLoop Bundle Tests

8 new test cases for EvalLoop debug bundle functionality:
- Basic bundle creation and directory structure
- Failed evaluation and None output handling
- Session-aware evaluator support
- Fallback behavior and re-execution prevention

#### Generic Type Resolution Tests

18 new tests for AST-based type resolution:
- Simple names, namespace handling, subscripted generics
- Builtin generics, union types, literal types
- Error handling and edge cases

## v0.22.0 - 2026-01-23

*Commits reviewed: 2026-01-20 (065e76c) through 2026-01-23 (633507f)*

### TL;DR

This release delivers significant **debug UI performance improvements** through
SQLite-backed caching shared with `wink query`, enabling instant startup when
databases are pre-built and powerful log filtering. The **query command** gains
**SQL views** for common analysis patterns, **sequence number tracking** for
native tool range queries, and **JSONL export** for `jq`-based workflows. **Tool
schema auditing** captures available tools at each prompt render for debugging
and analysis. All logs now carry **correlation IDs** from `RunContext` for
unified distributed tracing. **Environment capture** is now wired into the main
loop for automatic reproducibility envelopes. New **user guides** cover skills
authoring, dependency injection, and serialization. The **toolchain** gains
enhanced error reporting with precise import tracking and better diagnostics.

### Debug UI SQLite Caching

The `wink debug` web interface now shares the SQLite database infrastructure
with `wink query`, eliminating code duplication and dramatically improving
performance. When a cached database exists from a previous `wink query` run, the
debug UI starts instantly without re-parsing bundle contents.

**Key improvements:**

- **Unified caching**: Both `wink debug` and `wink query` use the same `.sqlite`
  cache file, avoiding redundant parsing
- **Thread-safe access**: Database operations use locking and
  `check_same_thread=False` for FastAPI compatibility
- **SQL-powered pagination**: Filtering and pagination handled by SQLite
  LIMIT/OFFSET instead of in-memory Python lists
- **Enhanced log filtering**: New filters for `level`, `logger`, `event`,
  `exclude_logger`, `exclude_event`, and full-text `search`
- **Filter facets API**: Returns counts per logger/event/level for UI
  autocomplete
- **Directory mode**: `wink debug /path/to/dir` auto-selects the newest bundle

### Query Command Enhancements

The `wink query` command gains SQL views, sequence number tracking, and raw JSONL
export for advanced bundle analysis workflows.

**SQL Views** provide pre-built queries for common analysis patterns:

- **`tool_timeline`**: Tool calls ordered by timestamp with command extraction
- **`native_tool_calls`**: Claude Code native tools from log_aggregator events
- **`error_summary`**: Errors with truncated traceback for quick debugging

**Sequence Number Column** adds a `seq` column to the logs table that extracts
`sequence_number` from `log_aggregator.log_line` events, enabling range queries
on native tool executions:

```sql
SELECT * FROM logs WHERE seq BETWEEN 100 AND 200
```

**JSONL Export** bypasses the SQL layer for power users who prefer `jq`
processing:

```bash
wink query bundle.zip --export-jsonl        # Export logs/app.jsonl
wink query bundle.zip --export-jsonl=session  # Export session/after.jsonl
```

**Additional improvements:**

- **Schema hints**: Enhanced `--schema` output with `json_extraction` patterns
  and `common_queries` examples
- **No truncation mode**: `--no-truncate` flag disables column truncation in
  ASCII table output

### Tool Schema Auditing

A new `RenderedTools` session slice captures the complete set of available tools
and their JSON Schema definitions at each prompt render, enabling debugging and
analysis of tool availability.

```python
from weakincentives.runtime.session import RenderedTools, ToolSchema

# Query tools available during prompt renders
for record in session[RenderedTools].all():
    print(record.tool_names)       # Tuple of tool names
    print(record.tool_count)       # Number of tools
    schema = record.get_tool("my_tool")  # Get specific ToolSchema
```

Each `RenderedTools` record includes a `render_event_id` that matches the
corresponding `PromptRendered` event, enabling precise correlation of "what
tools were available when this prompt was rendered."

### Correlation IDs for Distributed Tracing

All provider adapters (OpenAI, LiteLLM) and tool executors now bind `run_id`
from `RunContext` to their loggers, enabling unified traceability across the
entire execution lifecycle. Logs from prompt calls, provider calls, tool calls,
and filesystem changes can now be joined into a coherent timeline for debugging.

**Pattern improvements:**

- Bound logger created once and reused throughout request handling
- Removed redundant context fields (`tool_name`, `call_id`, `prompt_name`) that
  are already carried by the bound logger
- Consistent correlation ID handling across all adapter modules

### Environment Capture Integration

The `BundleWriter.write_environment()` method, which captures comprehensive
reproducibility envelopes, is now automatically called by AgentLoop during debug
bundle finalization. Previously, this method existed but was never invoked.

Debug bundles now automatically include:

- System information (OS, architecture, CPU, memory)
- Python runtime details (version, virtualenv, packages)
- Git state (commit, branch, uncommitted changes)
- Environment variables (filtered and redacted)
- Command-line invocation context

### Native Tool Tracking in Debug Database

Native Claude Agent SDK tools (Bash, Read, Write, etc.) now appear in the
`tool_calls` debug table alongside MCP-bridged WINK tools. The fix corrects an
event naming pattern that prevented native tool events from matching the query
filter.

### Documentation

**New guides:**

- **`guides/skills-authoring.md`**: Comprehensive guide for creating skills
  following the Agent Skills specification, covering SKILL.md format,
  frontmatter fields, mounting configuration, validation, and troubleshooting

- **`guides/resources.md`**: Complete reference for the dependency injection
  system including `Binding`, `Scope` (SINGLETON, TOOL_CALL, PROTOTYPE),
  lifecycle protocols (Closeable, PostConstruct, Snapshotable), testing
  patterns, and best practices

- **`guides/serialization.md`**: Full documentation of the `weakincentives.serde`
  module covering `parse`/`dump`/`clone`/`schema` functions, type coercion,
  validation constraints, custom validators, field aliases, extra field
  handling, polymorphic serialization, and JSON Schema generation

**Specification updates:**

- `specs/DEBUG_BUNDLE.md`: Aligned with implementation—removed unimplemented
  capture modes and CLI commands, added environment capture documentation,
  updated API routes to match actual endpoints
- `specs/WINK_QUERY.md`: Updated implementation status to reference
  `src/weakincentives/cli/query.py`
- Debug CLI documentation corrected to show actual `--host`, `--port`,
  `--no-open-browser` options instead of non-existent subcommands

### Toolchain Improvements

**Enhanced error reporting:**

- **Tool-specific prefixes**: Type checker diagnostics now show `[pyright]` or
  `[ty]` prefixes for clear attribution
- **Precise import tracking**: Architecture violations show the exact import
  statement causing the violation, not just the module name
- **Location ranges**: `Location` class supports `end_line` and `end_column`
  for range-based error reporting
- **Coverage details**: Uncovered files now listed with line numbers (up to 10
  files)
- **Modern format support**: Coverage parser handles branch coverage columns;
  mdformat parser handles both old and new output formats
- **Actionable hints**: Truncated output shows "Run: python check.py X -v"
  guidance

### Internal

- Added `# pragma: no cover` annotations for platform-specific and
  version-specific code paths that cannot be tested in a single environment
- Simplified GitHub Actions code review workflow: marker-based comment detection
  (`<!-- claude-review -->`), focused review scope (bugs/security/design only),
  reduced allowed tools
- Dependency upgrades: bandit 1.9.3, claude-agent-sdk 0.1.21, huggingface-hub
  1.3.2, litellm 1.81.1, packaging 26.0, podman 5.7.0, pycparser 3.0, pyparsing
  3.3.2, regex 2026.1.15, ruff 0.14.13, sse-starlette 3.2.0

## v0.21.0 - 2026-01-20

### TL;DR

This release introduces a comprehensive **debug bundle system** for post-mortem
analysis with SQL-based exploration, environment capture, and automatic
per-request collection. The Claude Agent SDK adapter gains **progressive
disclosure** support. New **controllable time dependencies** enable
deterministic testing without real delays. A **unified verification toolbox**
consolidates scattered build scripts into a single extensible framework.

### Debug Bundle System

A new debug bundle system captures comprehensive execution state for post-mortem
analysis. Bundles are self-contained zip archives that AgentLoop generates
automatically per-request, enabling reliable debugging without manual
instrumentation.

```python
from weakincentives.debug import BundleConfig
from weakincentives.runtime import AgentLoopConfig

config = AgentLoopConfig(
    debug_bundle=BundleConfig(target="./debug_bundles/"),
)
# Bundles created automatically for each request
```

**Bundle contents:**

- **Session state**: Snapshots before and after execution (`session/`)
- **Logs**: Complete DEBUG-level logs from the request (`logs/`)
- **Metrics**: Timing, token usage, and budget consumption (`metrics.json`)
- **Request I/O**: Input parameters and output response (`request/`)
- **Filesystem**: Workspace snapshot within size limits (`filesystem/`)
- **Environment**: Reproducibility envelope for issue reproduction
  (`environment/`)

**Reproducibility envelope** captures execution context with security-conscious
redaction:

- System info (OS, architecture, CPU, memory)
- Python runtime (version, virtualenv detection, packages via `pip freeze`)
- Git state (commit, branch, dirty status, remotes with credential redaction)
- Container detection (Docker/Podman/Kubernetes)
- Filtered environment variables with sensitive value redaction

**SQL-based exploration** via `wink query` enables familiar querying of bundle
contents:

```bash
wink query bundle.zip --schema
wink query bundle.zip "SELECT tool_name, success FROM tool_calls"
wink query bundle.zip "SELECT * FROM logs WHERE level = 'ERROR'" --table
```

Auto-generates typed SQLite tables from bundle artifacts (`manifest`, `logs`,
`tool_calls`, `errors`, `session_slices`, `files`, `metrics`). Session state
types become queryable tables (e.g., `slice_agentplan`). Caches database
alongside bundle for fast repeated queries.

**Additional features:**

- Atomic zip creation prevents partial archives on crashes
- Per-request override via `AgentLoopRequest.debug_bundle`
- `AgentLoopResult.bundle_path` provides access to created bundle

See `specs/DEBUG_BUNDLE.md` and `specs/WINK_QUERY.md` for specifications.

### Progressive Disclosure for Claude Agent SDK

The Claude Agent SDK adapter now supports progressive disclosure via the
`open_sections` tool, enabling models to request section expansion with
automatic cross-context exception handling.

Since the MCP bridge runs in a different execution context than the adapter,
`VisibilityExpansionRequired` exceptions cannot propagate directly. A new
`VisibilityExpansionSignal` captures these exceptions in tool handlers and
re-raises them after the SDK query completes, enabling the standard retry loop.

The `open_sections` tool validation is lenient—already-expanded sections are
silently skipped rather than causing errors.

### Controllable Time Dependencies

New clock abstractions enable deterministic testing of time-dependent code
without real delays or monkeypatching.

```python
from weakincentives import FakeClock, SYSTEM_CLOCK

# Production: uses system time (the default)
deadline = Deadline(budget.deadline, clock=SYSTEM_CLOCK)

# Testing: instant, deterministic time control
fake = FakeClock()
fake.advance(seconds=300)  # No actual delay
```

**Protocols:** `MonotonicClock` (elapsed time), `WallClock` (UTC timestamps),
`Sleeper` (delays), and unified `Clock` combining all three.

**Implementations:** `SystemClock`/`SYSTEM_CLOCK` for production;
`FakeClock` with `advance()`, `set_monotonic()`, `set_wall()` for testing.

All time-dependent components accept optional `clock` parameters: `Deadline`,
`Heartbeat`, `LeaseExtender`, `wait_until()`, `InMemoryMailbox`.

See `specs/CLOCK.md` for the complete specification.

### Unified Verification Toolbox

Build verification consolidated into an extensible `toolchain/` framework with
a single entry point replacing 10+ scattered scripts.

```bash
python check.py                    # Run all checks
python check.py lint test          # Run specific checks
python check.py --list             # List available checks
python check.py --json             # Machine-readable output
```

**Features:** Extensible `Checker` protocol; structured diagnostics with
`Location` (file:line:column) for IDE-clickable output; multiple formatters
(console, JSON, quiet); actionable error messages with reproduction steps and
fix guidance.

See `specs/VERIFICATION_TOOLBOX.md` for the complete specification.

### Fixed

- **Frozenset serialization**: Session snapshots now correctly serialize
  `frozenset` fields
- **Tool example warnings**: Non-dataclass tool example values no longer
  trigger spurious warnings during serialization
- **RunContext population**: Claude Agent SDK adapter events now include
  `session_id` for proper correlation
- **Sandbox settings**: `allowUnsandboxedCommands=False` now explicitly written
  to settings.json
- **Toolchain coverage parser**: Now correctly handles branch coverage columns
  in pytest output format
- **Toolchain mdformat parser**: Now correctly extracts file paths from modern
  mdformat error format
- **Toolchain multi-import statements**: Architecture checker now shows the
  specific import causing a violation, not all imports on the same line

### Internal

- Consolidated path normalization into `weakincentives.filesystem._path`
- Split `agent_loop.py` and `session.py` into focused modules
- Unified AI assistant guidelines in CLAUDE.md
- Replaced `time.sleep()` calls with controllable clock injection in tests
- Integration tests auto-skip when API keys missing
- Claude code review workflow hides previous reviews before posting new ones

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
- **Request-level binding**: Experiments flow through `AgentLoopRequest.experiment`
  and `EvalRequest.experiment`
- **EvalReport extensions**: `by_experiment()`, `pass_rate_by_experiment()`,
  `compare_experiments()` for result analysis
- **BASELINE/CONTROL sentinels**: Pre-defined experiments for common patterns

See `specs/EXPERIMENTS.md` for the full specification.

### RunContext for Distributed Tracing

`RunContext` provides immutable execution metadata that flows through the system
from AgentLoop to tool handlers and telemetry events, enabling distributed
tracing, request correlation, and debugging.

```python
from weakincentives.runtime import RunContext, AgentLoopRequest

ctx = RunContext(
    worker_id="worker-42",
    trace_id="abc-123",
    span_id="xyz-789",
)

request = AgentLoopRequest(request=MyRequest(...), run_context=ctx)
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
from weakincentives.runtime import AgentLoopConfig, LeaseExtenderConfig

config = AgentLoopConfig(
    lease_extender=LeaseExtenderConfig(
        interval=60.0,   # Rate-limit to once per minute
        extension=300,   # Extend by 5 minutes on each beat
    ),
)
```

Key features:

- **Heartbeat-based**: Extension piggybacks on tool execution beats
- **Fail-safe**: Stuck workers let leases expire (correct behavior)
- **EvalLoop support**: Both AgentLoop and EvalLoop support automatic extension
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
`BundleConfig` with `AgentLoopConfig` for automatic bundle creation per-request.

```python
from weakincentives.debug import BundleConfig
from weakincentives.runtime import AgentLoopConfig

config = AgentLoopConfig(
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


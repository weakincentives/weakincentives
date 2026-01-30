# Changelog

Release highlights for weakincentives.

## Unreleased

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

`LoopGroup` runs multiple `AgentLoop` or `EvalLoop` instances with coordinated
shutdown, optional health endpoints (`/health/live`, `/health/ready`), and watchdog
monitoring for Kubernetes deployments.

```python
from weakincentives.runtime import LoopGroup

group = LoopGroup(
    loops=[agent_loop, eval_loop],
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

A minimal evaluation framework built on AgentLoop for testing agent behavior.
AgentLoop handles orchestration; evals add datasets and scoring.

```python
from weakincentives.evals import Dataset, EvalLoop, exact_match

# Load a dataset
dataset = Dataset.load(Path("qa.jsonl"), str, str)

# Run evaluation
loop = EvalLoop(agent_loop=my_loop, evaluator=exact_match)
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

Pass custom resources into `adapter.evaluate()` and `AgentLoop.execute()`,
making tools cleaner, more testable, and portable.

```python
from weakincentives.resources import ResourceRegistry
from myapp.http import HTTPClient

# Build resource registry with your dependencies
http_client = HTTPClient(base_url="https://api.example.com")
resources = ResourceRegistry.build({HTTPClient: http_client})

# Pass to adapter - merged with workspace resources (e.g., filesystem)
response = adapter.evaluate(prompt, session=session, resources=resources)

# Or at the AgentLoop level
config = AgentLoopConfig(resources=resources)
loop = MyLoop(adapter=adapter, dispatcher=dispatcher, config=config)
response, session = loop.execute(request)
```

**Key features:**

- `ResourceRegistry.merge()` combines registries with the second taking
  precedence on conflicts
- User-provided resources override workspace defaults (e.g., custom filesystem)
- Tool handlers access resources via `context.resources.get(ResourceType)`
- `AgentLoopConfig`, `AgentLoopRequest`, and `AgentLoop.execute()` all accept
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

- **`AgentLoop.execute()` direct execution.** AgentLoop now supports direct
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

- **AgentLoop initialization refactored.** The two abstract methods
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
- **AgentLoop return value**: `AgentLoop.execute()` now returns
  `(response, session)` instead of just `response`.
- **AgentLoop config**: `AgentLoopConfig.parse_output` has been removed.
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

### AgentLoop Orchestration

- Added `AgentLoop[UserRequestT, OutputT]` abstract orchestrator that
  standardizes agent workflow execution: receive request, build prompt,
  evaluate, handle visibility expansion, publish result. Implementations
  define only the domain-specific factories via `create_prompt()` and
  `create_session()`.
- Added event types for request/response routing: `AgentLoopRequest[T]`,
  `AgentLoopCompleted[T]`, and `AgentLoopFailed`.
- Added `AgentLoopConfig` for configuring default deadline and budget
  constraints.
- The AgentLoop automatically handles `VisibilityExpansionRequired` exceptions,
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

- Added `AGENT_LOOP.md` spec documenting AgentLoop orchestration and visibility
  override handling.
- Added `HOSTED_TOOLS.md` spec for provider-executed tools (web search, code
  interpreter).
- Added `TASK_SECTION.md` spec for task dataclass patterns in prompts.
- Added `WINK_OVERRIDES.md` spec documenting the `wink overrides` CLI command.
- Simplified and consolidated specification documents for clarity.

### Examples

- Refactored `code_reviewer_example.py` to use the new `AgentLoop` abstraction,
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

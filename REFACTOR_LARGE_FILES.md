# Large File Refactoring Analysis

Analysis of all files exceeding ~600 lines in `src/` and `tests/`, with
concrete split/refactor proposals for each. Ordered by size (largest first).

---

## Source Files

### `src/weakincentives/cli/query.py` (2,043 lines)

This module conflates six distinct responsibilities into a single file: JSON/SQL
type conversion utilities, a 256-line transcript extraction pipeline that
recursively parses nested log structures, session snapshot processing, the
1,201-line `QueryDatabase` god-class (which owns connection lifecycle, 19 table
builders, 8 SQL view definitions, schema introspection, and query execution),
output formatting, and file-based cache management. Refactor by extracting the
transcript extraction functions into a `_transcript.py` module, the session
processing helpers into `_session_processing.py`, the view SQL definitions into
a data-driven registry (reducing the 165-line `_build_views()` method to ~30
lines), and the cache/path resolution into `_cache.py`; then decompose
`QueryDatabase` itself so that table-building is delegated to a `TableFactory`
or set of per-domain builder functions (environment tables alone are 624 lines),
leaving the class focused on connection management and query execution.

### `src/weakincentives/adapters/claude_agent_sdk/adapter.py` (1,772 lines)

The `ClaudeAgentSDKAdapter` class is a 1,506-line monolith that orchestrates
prompt evaluation end-to-end: initialization, async lifecycle management (prompt
rendering, filesystem binding, resource contexts), SDK options introspection and
version-compatible filtering, hook configuration assembly, deadline/budget
constraint enforcement, multi-turn response stream handling, the 202-line
`_run_sdk_query()` main loop, structured output parsing, and post-execution
validation. Refactor by extracting the module-level message content extraction
and schema normalization utilities (253 lines) into `_message_utils.py` and
`_schema.py` respectively, the options building pipeline (152 lines across 4
methods) into an `_options_builder.py` module, the stream message handling with
deadline enforcement (106 lines) into `_stream.py`, and the result extraction
and validation logic (306 lines) into `_result.py`; this would reduce the
adapter class to ~800 lines focused purely on orchestration.

### `src/weakincentives/debug/bundle.py` (1,531 lines)

This file houses both the write path (`BundleWriter`, 791 lines) and the read
path (`DebugBundle`, 271 lines) for debug bundles, plus log collection
infrastructure, 8 metadata dataclasses, utility functions, and a 205-line
retention policy enforcement subsystem buried inside `BundleWriter`. Refactor by
separating the reader (`DebugBundle`) into `_reader.py`, extracting the
retention policy logic (age/count/size limit enforcement with TOCTOU protection)
into a dedicated `_retention.py` module, moving the metadata dataclasses
(`BundleManifest`, `CaptureInfo`, `PromptInfo`, etc.) into `_manifest.py`, and
pulling the log collector (already annotated as "inlined") into
`_log_collector.py`; this leaves `BundleWriter` at ~400 lines focused on
artifact streaming and ZIP finalization.

### `src/weakincentives/adapters/claude_agent_sdk/isolation.py` (1,239 lines)

The `EphemeralHome` class (498 lines) is a god-object that handles temp
directory creation, `settings.json` generation, AWS config copying, skill file
mounting, subprocess environment variable assembly with priority-based
overrides, and cleanup. Refactor by extracting an `_environment_builder.py`
module for the 6 `_apply_*` methods that compose the subprocess environment dict
(120 lines), a `_settings_generator.py` for the settings.json construction
pipeline (117 lines including sandbox and auth configuration), and an
`_aws_isolation.py` for AWS config resolution and copying (90 lines); also move
the 8 authentication detection helper functions and `BedrockConfig` (135 lines)
into an `_auth_detection.py` module, and the skill copying logic into
`_skill_isolation.py` (87 lines), leaving `EphemeralHome` as a thin
orchestrator (~100 lines) that delegates to these focused components.

### `src/weakincentives/adapters/claude_agent_sdk/_hooks.py` (1,178 lines)

Seven hook factory functions (`create_pre_tool_use_hook`,
`create_post_tool_use_hook`, `create_stop_hook`,
`create_task_completion_stop_hook`, `create_subagent_stop_hook`,
`create_pre_compact_hook`, `create_user_prompt_submit_hook`) are packed into one
file alongside the `HookContext`/`HookStats`/`HookConstraints` data classes, 40
lines of constraint-checking utilities, 293 lines of post-tool-use processing
helpers, and the 84-line `safe_hook_wrapper`. Refactor by grouping hooks by
lifecycle phase: extract constraint checking and pre-tool-use logic into
`_constraint_hooks.py`, the post-tool-use processing pipeline (parsing, event
dispatch, transaction handling, feedback collection) into
`_tool_transaction_hooks.py`, task completion and stop hooks into
`_completion_hooks.py`, and the error-wrapping infrastructure into
`_hook_safety.py`; keep `HookContext` and data classes in the current file as
the shared foundation.

### `src/weakincentives/contrib/mailbox/_redis.py` (1,181 lines)

The `RedisMailbox` class (835 lines) is appropriately cohesive for a queue
implementation, but 286 lines are consumed by an embedded `@formal_spec` TLA+
model definition and 171 lines are Lua script string constants. Refactor by
extracting the 8 Lua scripts into a `_lua_scripts.py` module (making them
independently reviewable and testable for correctness), moving the formal
specification into a standalone `.tla` file or `_formal_spec.py` that can be
verified separately, and extracting the serialization/deserialization pair (30
lines) into `_serde.py` if pluggable serialization (e.g., MessagePack) is ever
desired; the background reaper thread (25 lines) could also become a pluggable
strategy, but this is lower priority since the class's core send/receive/ack
loop is already well-structured.

### `src/weakincentives/serde/parse.py` (1,054 lines)

This module implements a full recursive-descent type parser with cleanly layered
internals but everything in one file: primitive coercers (52 lines), union/
Optional handling (55 lines), literal validation (44 lines), sequence and
mapping coercion (113 lines), an AST-based string annotation resolver class (85
lines), generic TypeVar resolution utilities (160 lines), field discovery with
case-insensitive key matching and alias resolution (108 lines), and the central
`_coerce_to_type()` dispatcher. Refactor by extracting the type resolution
engine (`_ASTResolver`, `_resolve_generic_string_type`, `_get_field_types`,
TypeVar mapping) into `_type_resolver.py` (~245 lines), the individual coercion
handlers into `_coercers.py` (~370 lines), and the field processing pipeline
(key finding, alias resolution, extra fields, validation hooks) into
`_field_processor.py` (~136 lines), leaving `parse.py` as a thin dispatcher
that imports and orchestrates these components.

### `src/weakincentives/runtime/agent_loop.py` (1,045 lines)

The `AgentLoop` class mixes four concerns: direct synchronous execution (170
lines including the `execute_with_bundle` context manager), mailbox message
processing with bundled/unbundled paths (360 lines with heavy conditional
branching), debug bundle artifact writing and metrics collection (200 lines),
and core evaluation with retry logic (135 lines). Refactor by extracting the
debug bundling concern (artifact writing, metrics collection, visibility
override formatting) into a `_bundle_collector.py` that the loop delegates to,
and the mailbox message handling (`_process_message`,
`_handle_message_with_bundle`, `_handle_message_without_bundle`) into a
`_message_handler.py`; this would let the core `AgentLoop` focus on the
`prepare` -> `_resolve_settings` -> `_evaluate_with_retries` -> `finalize`
pipeline, reducing it to ~400 lines.

### `src/weakincentives/cli/debug_app.py` (1,040 lines)

This FastAPI debug UI module mixes data access (`BundleStore`, 484 lines), HTTP
handlers (`_DebugAppHandlers`, 203 lines), SQL filter construction (124 lines),
markdown rendering utilities (34 lines), and server lifecycle management (46
lines). Refactor by extracting the SQL filter building functions
(`_build_log_filters`, `_build_transcript_filters`, row parsers) into a
`_filters.py` module, the markdown detection and rendering into `_markdown.py`,
and the server lifecycle (`build_debug_app`, `run_debug_server`,
`_open_browser`) into `_server.py`; `BundleStore` could also benefit from
splitting its 102-line `get_environment()` method and the 8 environment
sub-table queries into a dedicated environment accessor, though this is lower
priority since the store's internal method grouping is already reasonable.

### `src/weakincentives/adapters/codex_app_server/adapter.py` (938 lines)

The `CodexAppServerAdapter` (784 lines) handles protocol orchestration, message
streaming, notification processing, tool invocation, deadline enforcement, and
output parsing all within a single class. Refactor by extracting the message
processing pipeline (`_consume_messages`, `_process_notification`,
`_apply_notification`, `_handle_item_completed`, `_handle_turn_completed`) into
a `_message_processor.py` module (~223 lines), the tool execution handler
(`_handle_server_request`, `_handle_tool_call`) into `_tool_handler.py` (~71
lines), the deadline watchdog pattern into `_deadline.py` (reusable for other
adapters), and the 4 pure schema transformation helpers into `_schemas.py` (~85
lines); this leaves the adapter class focused on the protocol state machine
(`_execute_protocol` with its 5 sequential phases) and high-level orchestration.

### `src/weakincentives/prompt/overrides/validation.py` (852 lines)

This file handles six responsibilities for prompt override validation:
section normalization and loading, tool validation and normalization, task
example parsing (146 lines of nested step/output handling), JSON serialization
(226 lines), write-time strict validation, and seeding from live prompts.
Refactor by splitting along the section/tool/task-example axis: extract section
operations (normalization, parsing, loading) into `_section_ops.py`, tool
operations (validation, normalization, loading) into `_tool_ops.py`, task
example and step override handling into `_task_example_ops.py`, and the
serialization functions into `_serialization.py`; the seeding functions (62
lines) and shared utilities can remain in the current file or move to a
`_seeding.py` module.

### `src/weakincentives/prompt/registry.py` (812 lines)

The `PromptRegistry` class (209 lines) handles mutable section registration and
snapshot generation, while `RegistrySnapshot` (90 lines) provides the frozen
read-only view; however, 352 lines are consumed by 7 structural validators and
a 215-line task example validation subsystem that checks step tool names and
input/output type coherence. Refactor by extracting all `_validate_*` functions
into a `_validation.py` module and the task example validation into
`_task_example_validation.py` (it's already a cohesive block at lines 584-799),
moving `RegistrySnapshot` and index precomputation into `_snapshot.py`, and
leaving `PromptRegistry` in the main file focused on registration and snapshot
creation.

### `src/weakincentives/filesystem/_host.py` (784 lines)

`HostFilesystem` is a single 710-line dataclass implementing sandboxed file
I/O, search, and git-based snapshots. Refactor by extracting the snapshot/
restore subsystem (`_ensure_git`, `snapshot`, `restore`, `cleanup` -- 174 lines)
into `_snapshot.py` since it's the most self-contained boundary, the search
operations (`glob`, `grep`, `_grep_file` -- 98 lines) into `_search.py`, and
optionally the write operations (`write`, `write_bytes` -- 107 lines) into
`_write.py`; keep path resolution, `_run_git`, and read operations as the
foundation layer, reducing the main class to ~400 lines.

### `src/weakincentives/cli/wink.py` (786 lines)

_Not analyzed in detail -- likely CLI entrypoint with Click/Typer command
definitions that could be split by command group (debug, query, docs, etc.)._

### `src/weakincentives/dbc/__init__.py` (762 lines)

This design-by-contract module packages four decorator implementations plus
shared infrastructure in a single `__init__.py`: `require` (26 lines), `ensure`
(41 lines), `invariant` with 6 supporting functions (106 lines), and `pure`
with its I/O interception, argument snapshotting, and patching system (165
lines). Refactor by extracting the `pure` decorator and its infrastructure
(global guards, patch activation, snapshot utilities) into `_pure.py` since it's
the most complex and self-contained piece, the invariant system into
`_invariant.py`, and the shared contract evaluation logic (65 lines) into
`_evaluation.py`; keep `require`, `ensure`, and state management in
`__init__.py` as the lightweight public surface.

### `src/weakincentives/debug/environment.py` (713 lines)

This environment capture module has 9 frozen dataclasses for metadata, plus
capture functions grouped by domain: memory detection (40 lines, platform-
specific), system/Python info (27 lines), package listing (30 lines),
environment variables with allowlist/redaction (25 lines), git operations (76
lines), untracked file handling (69 lines), and container detection (70 lines).
Refactor by extracting the git operations and untracked file handling (145 lines
combined) into `_git_capture.py` since they form the largest cohesive subsystem,
and the container detection heuristics (cgroup parsing, Kubernetes markers) into
`_container_detection.py`; the remaining platform-specific memory queries could
join system/Python capture in a `_platform.py` module, keeping the dataclasses
and the `capture_environment()` orchestrator in the main file.

### `src/weakincentives/contrib/tools/filesystem_memory.py` (681 lines)

`InMemoryFilesystem` (573 lines) implements the full `Filesystem` protocol
in-memory with parallel read/write methods, directory listing, glob/grep search,
and snapshot/restore via structural sharing. Refactor by extracting the search
operations (`glob`, `grep` -- 67 lines) and the directory listing logic
(`_collect_file_entries`, `_collect_explicit_dir_entries`, `list` -- 65 lines)
into mixin classes or standalone helper functions, and the snapshot/restore
mechanism (56 lines) into `_snapshot.py`; however, since all methods operate on
the same `_files`/`_directories` internal state, the benefit is moderate -- the
better refactoring here is to extract the write operations (119 lines) and read
operations (102 lines) into clearly documented internal method groups and
consider whether `HostFilesystem` and `InMemoryFilesystem` could share more
validation logic through a base class or shared utility module.

### `src/weakincentives/prompt/tool.py` (658 lines)

The `Tool` class (324 lines) conflates type resolution, 7 validation methods
(131 lines), example validation (87 lines), result annotation normalization, and
handler wrapping/inference (110 lines) all in `__post_init__` and classmethods.
Refactor by extracting the validation methods into `_tool_validation.py`,
example handling (`ToolExample` + validation) into `_tool_examples.py`, the
handler introspection and wrapping factory (`wrap` and its 5 supporting static
methods) into `_tool_factory.py`, and the generic type resolution logic into
`_tool_types.py`; this would reduce `Tool` from 324 to ~100 lines focused on
the dataclass fields and `__post_init__` orchestration.

### `src/weakincentives/prompt/feedback.py` (639 lines)

This module packages feedback data models (`Observation`, `Feedback` -- 67
lines), the `FeedbackContext` state accessor (145 lines), the `FeedbackProvider`
protocol (59 lines), trigger configuration and evaluation (`FeedbackTrigger`,
`_should_trigger` -- 95 lines), and the orchestration functions
(`run_feedback_providers`, `collect_feedback` -- 95 lines). Refactor by
extracting `FeedbackContext` into `_context.py` (it's the largest component and
independently testable), the trigger types and `_should_trigger()` evaluation
into `_triggers.py` (orthogonal concern), and the runner/orchestration into
`_runner.py`; keep the data models and protocol in the main file as the public
API surface.

### `src/weakincentives/adapters/_shared/_bridge.py` (616 lines)

`BridgedTool` (292 lines) handles tool execution orchestration with
transactional semantics, while `MCPToolExecutionState` (82 lines) manages
hook-to-execution correlation, and three factory functions build bridged tools,
async handlers, and MCP servers. Refactor by extracting `MCPToolExecutionState`
into `_mcp_tool_state.py` (self-contained thread-safe correlation), the factory
functions (`create_bridged_tools`, `make_async_handler`, `create_mcp_server` --
151 lines) into `_factories.py`, and optionally separating the result formatting
and event dispatch methods from `BridgedTool` into a `_result_formatter.py`;
this would reduce the bridge file to `BridgedTool`'s core execution logic (~200
lines).

### `src/weakincentives/runtime/transactions.py` (584 lines)

`CompositeSnapshot` dominates at 229 lines, with `from_json()` alone being 147
lines of deserialization with comprehensive validation. The rest is well-
factored: `PendingToolTracker` (118 lines), `create_snapshot`/
`restore_snapshot` factory functions (72 lines), and the `tool_transaction`
context manager (40 lines). Refactor by extracting `CompositeSnapshot`
serialization (`to_json`/`from_json` -- 204 lines) into `_snapshot_serde.py`,
which would isolate the heaviest validation logic and make the main file focused
on the runtime transaction semantics; the tracker and context manager are
already well-encapsulated and can remain together.

### `src/weakincentives/prompt/progressive_disclosure.py` (535 lines)

_Not analyzed in detail -- likely progressive section visibility logic that
could split visibility calculation from rendering._

### `src/weakincentives/runtime/session/snapshots.py` (519 lines)

_Not analyzed in detail -- likely session snapshot logic that could split
serialization from snapshot management._

### `src/weakincentives/prompt/overrides/versioning.py` (515 lines)

_Not analyzed in detail -- likely version tracking that could split version
computation from storage._

### `src/weakincentives/runtime/lifecycle.py` (504 lines)

_Not analyzed in detail -- likely lifecycle management that could split shutdown
coordination from health monitoring._

### `src/weakincentives/evals/_loop.py` (503 lines)

_Not analyzed in detail -- likely evaluation loop that could split dataset
submission from result collection._

### `src/weakincentives/dataclasses/__init__.py` (502 lines)

_Not analyzed in detail -- likely frozen dataclass infrastructure._

### `src/weakincentives/prompt/overrides/local_store.py` (498 lines)

_Not analyzed in detail -- likely file-based override storage._

### `src/weakincentives/adapters/claude_agent_sdk/workspace.py` (492 lines)

_Not analyzed in detail -- likely workspace management for the SDK adapter._

### `src/weakincentives/evals/_types.py` (452 lines)

_Not analyzed in detail -- likely evaluation type definitions._

### `src/weakincentives/adapters/codex_app_server/workspace.py` (450 lines)

_Not analyzed in detail -- likely workspace management for Codex adapter._

### `src/weakincentives/prompt/rendering.py` (440 lines)

_Not analyzed in detail -- likely prompt rendering logic._

### `src/weakincentives/evals/__init__.py` (438 lines)

_Not analyzed in detail -- likely evaluation public API surface._

### `src/weakincentives/serde/schema.py` (414 lines)

_Not analyzed in detail -- likely JSON schema generation._

### `src/weakincentives/skills/_validation.py` (418 lines)

_Not analyzed in detail -- likely skill validation logic._

---

## Test Files

### `tests/adapters/claude_agent_sdk/test_adapter.py` (3,056 lines)

This file packs 17 test classes covering 8 unrelated concerns: adapter
initialization, core evaluation and event publishing, SDK option configuration,
isolation/sandboxing (407 lines), message content extraction (395 lines), task
completion verification (355 lines), multi-turn continuation edge cases (420
lines), and schema utility functions. Split into 6-8 files by concern:
`test_adapter_init.py` (init + evaluate), `test_adapter_options.py` (SDK
config/errors/budgets), `test_adapter_isolation.py` (sandboxing/permissions),
`test_adapter_messages.py` (content extraction), `test_adapter_completion.py`
(task completion verification), `test_adapter_multiturn.py` (continuation/
deadline edge cases), and `test_adapter_schema.py` (normalization/introspection);
move shared fixtures and mocks into a `conftest.py`.

### `tests/runtime/test_agent_loop.py` (2,196 lines)

Contains ~62 tests covering 6 distinct architectural concerns: config/request/
result dataclasses, core execution loop, resource injection and budget
propagation, execution context and tracing, debug bundle integration (18 tests,
500+ lines), and prompt cleanup lifecycle. Split by concern into
`test_agent_loop_core.py` (config + execution + mailbox), `test_agent_loop_
resources.py` (injection + budget), `test_agent_loop_bundles.py` (all 18 debug
bundling tests), and `test_agent_loop_cleanup.py` (lifecycle + retry caps);
extract shared helpers (`_MockAdapter`, `_TestLoop`, dataclasses) into
`conftest.py`.

### `tests/cli/test_wink_query_database.py` (2,189 lines)

24 test classes organized into 6 domains: core database functionality, transcript
processing (446-line single test class), edge cases and error handling (9
classes), file and session handling, schema and views, and environment tables.
Split into `test_query_core.py` (basic DB ops), `test_query_transcript.py`
(the large transcript extraction tests), `test_query_edge_cases.py` (error
handling), `test_query_files.py` (file/session/export), `test_query_views.py`
(schema + views), and `test_query_environment.py` (environment tables); the
shared fixture module `_query_fixtures.py` already exists.

### `tests/adapters/claude_agent_sdk/test_hooks.py` (2,044 lines)

16 test classes with ~95 tests covering hook context setup, pre-tool-use
permission checks, post-tool-use event dispatch (439 lines), stop and
completion hooks, safe wrapper error handling, transactional semantics, tool
error detection, and composite checker composition. Split by hook lifecycle
phase: `test_hooks_pre_tool.py` (pre-tool-use + transactional snapshots),
`test_hooks_post_tool.py` (post-tool-use + typed parsing + transactions),
`test_hooks_completion.py` (stop hooks + task completion checkers + composite),
`test_hooks_utilities.py` (safe wrapper + error detection + compact), and move
shared mocks/fixtures into `conftest.py`.

### `tests/debug/test_bundle.py` (1,985 lines)

21 test classes testing bundle configuration, core writing (14 tests), bundle
loading (13 tests), exception resilience (17 tests for all write methods),
filesystem archiving, integrity verification, retention policy integration (13
tests + 8 nested directory tests), and storage handler integration. Split into
`test_bundle_config.py` (config + manifest + helpers), `test_bundle_writer.py`
(writing + exception handling), `test_bundle_reader.py` (loading + integrity),
`test_bundle_retention.py` (all retention + nested directory tests), and
`test_bundle_storage.py` (storage handler + config integration).

### `tests/cli/test_wink_debug_app.py` (1,873 lines)

All standalone test functions (no classes) covering 8 concerns: bundle store
loading, API routes for slices/meta, log filtering (8 tests), transcript API,
file content serving (6 tests with multi-format detection), environment endpoint
(5 tests), server startup, and bundle lifecycle. Split by API resource:
`test_debug_app_store.py` (loading/switching/lifecycle), `test_debug_app_
logs.py` (log endpoints + facets + filters), `test_debug_app_files.py` (file
content type detection), `test_debug_app_environment.py` (environment endpoint
variants), `test_debug_app_server.py` (uvicorn/browser), and keep remaining
small endpoints in a core file; extract `_create_test_bundle` and
`_create_minimal_bundle` helpers into `conftest.py`.

### `tests/adapters/codex_app_server/test_adapter.py` (1,809 lines)

30 test classes across 8 domains: schema strictification (174 lines), adapter
initialization, RPC setup, notification handling (109 lines), tool execution,
deadline management, CWD resolution, structured output, and end-to-end
evaluation (264-line class with 7 sub-tests). Split into `test_codex_schema.py`
(OpenAI strict schema transformation), `test_codex_protocol.py` (init + auth +
thread + turn), `test_codex_messages.py` (notifications + terminal errors),
`test_codex_tools.py` (tool calls + approvals), `test_codex_deadline.py`
(watchdog + timeout bounding), and `test_codex_evaluate.py` (end-to-end +
structured output + budget tracking).

### `tests/evals/test_loop.py` (1,728 lines)

37 tests split across core EvalLoop behavior, dataset/experiment submission,
result collection, mailbox error edge cases (10 tests, 425 lines), DLQ
integration (7 tests, 306 lines), and debug bundle tests (10 tests, 500 lines).
Split into `test_eval_loop_core.py` (basic functionality + helpers + e2e),
`test_eval_loop_errors.py` (mailbox error handling + edge cases),
`test_eval_loop_dlq.py` (DLQ policies and retry logic), and
`test_eval_loop_bundles.py` (debug bundle creation and capture).

### `tests/adapters/claude_agent_sdk/test_isolation.py` (1,682 lines)

29 test classes covering configuration dataclasses and factories (393 lines),
EphemeralHome core lifecycle and settings generation (355 lines), skill
infrastructure (466 lines), and advanced auth/AWS features (398 lines). Split
into `test_isolation_config.py` (all IsolationConfig, NetworkPolicy,
SandboxConfig, factory methods), `test_ephemeral_home_core.py` (lifecycle,
settings.json, environment variables), `test_ephemeral_home_skills.py` (skill
validation, copying, mounting, deduplication), and `test_ephemeral_home_auth.py`
(host auth inheritance, AWS config copying, helper functions).

### `tests/runtime/test_lifecycle.py` (1,620 lines)

10 test groups spanning `wait_until` utility, `ShutdownCoordinator` singleton
(130 lines), `LoopGroup` orchestration (79 lines), `AgentLoop` shutdown
semantics (392 lines with complex nested mock classes), `Runnable` protocol
compliance, signal handlers, `EvalLoop` shutdown (241 lines), `HealthServer`
HTTP endpoints (153 lines), health server integration (104 lines), and watchdog
monitoring (120 lines). Split into `test_shutdown_coordinator.py` (singleton +
signals), `test_loop_group.py` (parallel orchestration + shutdown propagation),
`test_agent_loop_shutdown.py` (the 392-line shutdown semantics with nested
classes), `test_eval_loop_shutdown.py` (evaluation loop lifecycle),
`test_health_server.py` (HTTP endpoints + readiness integration), and
`test_watchdog.py` (heartbeat monitoring).

### `tests/prompts/test_task_examples.py` (1,556 lines)

_Not analyzed in detail -- likely task example validation tests that could split
by validation concern (step validation, output type checking, tool reference
checking)._

### `tests/adapters/claude_agent_sdk/test_bridge.py` (1,534 lines)

_Not analyzed in detail -- likely bridge tool tests that could split into tool
execution, MCP state management, and factory function tests._

### `tests/prompt/test_feedback.py` (1,431 lines)

_Not analyzed in detail -- likely feedback provider tests that could split by
trigger evaluation, context queries, and orchestration._

### `tests/runtime/test_mailbox.py` (1,308 lines)

_Not analyzed in detail -- likely mailbox protocol tests that could split by
send/receive/ack operations._

### `tests/prompts/overrides/test_local_prompt_overrides_store.py` (1,295 lines)

_Not analyzed in detail -- likely override store tests that could split by
CRUD operations._

### `tests/resources/test_resources.py` (1,228 lines)

_Not analyzed in detail -- likely resource container tests that could split by
scope (singleton, prototype, tool_call) and lifecycle._

### `tests/debug/test_environment.py` (1,193 lines)

_Not analyzed in detail -- likely environment capture tests that could split by
subsystem (git, container, memory, packages)._

### `tests/serde/test_parse_and_dump.py` (1,035 lines)

_Not analyzed in detail -- likely serde round-trip tests that could split by
type category (primitives, collections, dataclasses, generics)._

### `tests/prompts/test_progressive_disclosure.py` (1,029 lines)

_Not analyzed in detail -- likely progressive disclosure tests that could split
by visibility strategy._

### `tests/runtime/test_dlq.py` (1,023 lines)

_Not analyzed in detail -- likely DLQ tests that could split by policy type._

### `tests/serde/test_loop_serde.py` (1,011 lines)

_Not analyzed in detail -- likely loop serialization tests._

### `tests/helpers/filesystem.py` (805 lines)

_Not analyzed in detail -- likely shared test filesystem helpers that could be
better organized into focused utility modules._

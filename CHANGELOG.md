# Changelog

Release highlights for weakincentives.

## Unreleased

*Commits reviewed: 2026-02-07 (b47a2736) through 2026-02-13 (a8da20c4)*

### TL;DR

WINK gains a full **ACP adapter** with **OpenCode integration** (~2,100 lines of
adapter code, ~4,200 lines of tests), enabling agents to run against any
ACP-compatible CLI (e.g., OpenCode) with MCP-over-HTTP tool bridging,
consolidated transcript emission, and structured output via an MCP tool. The
adapter stack is two-layer: a generic `ACPAdapter` handles the protocol flow
(initialize, new_session, prompt dispatch, drain quiet period), while
`OpenCodeACPAdapter` adds model validation, empty-response detection, and
OpenCode-specific quirks.

The **unified transcript system** introduced earlier this cycle is now
production-complete — a shared `TranscriptEmitter` in the runtime layer and
adapter-specific bridges (`CodexTranscriptBridge`, `ACPTranscriptBridge`) enable
all three adapters (Claude SDK, Codex App Server, ACP) to emit structurally
identical transcript entries using 9 canonical entry types. The Claude adapter
**splits mixed content blocks** (assistant messages containing `tool_use` blocks
become separate `assistant_message` + `tool_use` entries). Debug bundles gain a
dedicated `transcript.jsonl` artifact, and `wink query` schema is upgraded to v9.

**Adapter workspace sections are consolidated** into a single provider-agnostic
`WorkspaceSection` in the prompt layer, replacing `ClaudeAgentWorkspaceSection`
and `CodexWorkspaceSection`. **Skills move from adapter config to prompt
sections** — `SkillConfig` is deleted and `SkillMount` instances attach directly
to sections via `Section(skills=(...))`. The `deadline` field is **removed from
`AgentLoopConfig`**. The **serde API is simplified** by removing
`case_insensitive`, `alias_generator`, `aliases` parameters, `extra="allow"`
mode, and AST-based type resolution. The `@pure` decorator is **removed from
all production code**.

A massive **modular decomposition** splits 15+ large source files and 5+
monolithic test files into focused, single-responsibility modules — the Codex
adapter, Claude SDK adapter, serde, prompt registry, query database, debug
bundle, and CLI are all restructured. A new **Adapter Compatibility Kit (ACK)
specification** defines a unified integration testing framework across all
adapters. A new **code length checker** enforces 120-line method and 720-line
file limits. The **WINK presentation slides** are added as a Marp-based deck
with a GitHub Actions build workflow.

---

### Fixed

- **Transcript collector no longer loses transcripts when files appear late.**
  When the Claude Agent SDK fires a hook before the `.jsonl` file exists on
  disk, `TranscriptCollector` now records the source in a `_pending_tailers`
  dict and retries on each poll cycle (`_poll_once()`). Once the file appears,
  the tailer activates and tailing proceeds normally. The initial warning is
  logged only once per source to prevent log spam.
  (`_transcript_collector.py`: `_start_tailer()`, `_poll_once()`)

- **Codex `agentMessage` no longer contaminates tool metrics in `wink query`.**
  The `transcript_tools` SQL view previously matched `entry_type = 'assistant'`,
  catching all assistant messages. Now it matches
  `entry_type IN ('assistant_message', 'tool_use')` with proper `tool_name`
  filtering. (`query.py`: schema v9)

- **Bridged WINK tool events now appear in Codex transcripts.** The Codex
  adapter's `_handle_tool_call()` now emits `tool_use` and `tool_result`
  transcript entries via `CodexTranscriptBridge` for WINK-bridged tools, which
  were previously invisible. (`codex_app_server/adapter.py`)

- **Entry type override bug in query database.** `_extract_transcript_details_tuple()`
  previously overrode the canonical entry type with `parsed.get("type")`,
  leaking SDK-native types into the query database. Now uses the canonical
  `entry_type` directly. (`query.py`)

---

### Breaking Changes

#### Removed `deadline` from `AgentLoopConfig`

A deadline is a specific point in time. Setting one at config-construction time
means it can expire before execution even starts — a long-lived `AgentLoop`
reusing the same config would eventually have every request fail with a stale
deadline. Deadlines belong on per-request objects.

**Migration:**
```python
# Old ❌
config = AgentLoopConfig(
    deadline=Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5)),
)

# New ✅ — pass deadline per-request
request = AgentLoopRequest(
    request=my_request,
    deadline=Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5)),
)

# Or via execute()
loop.execute(my_request, deadline=Deadline(...))
```

#### Skills Attached at Section Level Instead of Adapter Config

`SkillConfig` is deleted. `IsolationConfig.skills` is removed along with the
`skills` parameter on all `IsolationConfig` factory methods
(`inherit_host_auth`, `explicit_api_key`, `require_anthropic_env`, `bedrock`).
`SkillMount.enabled` is removed — use section visibility instead.

Skills are now attached to sections and collected during prompt rendering,
following the same rules as tools: skills on `SUMMARY`-visibility sections are
not collected; sections with skills participate in progressive disclosure via
`open_sections`.

**Migration:**
```python
# Old ❌
adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        isolation=IsolationConfig(
            skills=SkillConfig(
                skills=(SkillMount(Path("./skills/review")),)
            ),
        ),
    ),
)

# New ✅ — attach skills to sections
section = MarkdownSection(
    title="Review",
    key="review",
    template="Review the code.",
    skills=(SkillMount(Path("./skills/review")),),
)
# Skills are automatically collected during rendering and mounted by the adapter
```

**New section-level infrastructure:**
- `Section.__init__` accepts `skills: Sequence[object] | None`
- `MarkdownSection.clone()` preserves skills
- `RenderedPrompt.skills` exposes collected skills
- `PromptRegistry` tracks skill names, detects duplicates, and computes
  `subtree_has_skills` index
- `EphemeralHome.mount_skills()` is now public with single-call enforcement

#### Consolidated `WorkspaceSection` Replaces Per-Adapter Implementations

`ClaudeAgentWorkspaceSection` (492 lines, deleted) and `CodexWorkspaceSection`
(450 lines, deleted) are replaced by a single provider-agnostic
`WorkspaceSection` in `weakincentives.prompt.workspace`. `HostMount`,
`HostMountPreview`, `WorkspaceBudgetExceededError`, `WorkspaceSecurityError`,
and `compute_workspace_fingerprint` also move to the prompt layer. The old
protocol `WorkspaceSection` in `prompt/protocols.py` is renamed to
`WorkspaceSectionProtocol`.

**Migration:**
```python
# Old ❌
from weakincentives.adapters.claude_agent_sdk import ClaudeAgentWorkspaceSection
from weakincentives.adapters.codex_app_server import CodexWorkspaceSection

# New ✅
from weakincentives.prompt import WorkspaceSection
```

#### Serde API Simplification

Removed rarely-used parameters from `parse()`, `dump()`, and `schema()` to
reduce API surface and simplify the implementation. Field-level aliases via
`field(metadata={"alias": "..."})` remain supported.

**Removed from `parse()`:**
- `case_insensitive` — case-insensitive key matching
- `alias_generator` — callable to transform field names
- `aliases` — dict of field-to-alias mappings

**Removed from `dump()`:**
- `alias_generator` — callable to transform field names

**Removed from `schema()`:**
- `alias_generator` — callable to transform property names

**Removed `extra="allow"` mode** — only `"ignore"` (default) and `"forbid"`
remain. The `"allow"` mode attached extra fields as instance attributes, which
conflicted with slotted dataclasses and was rarely used.

**Removed AST-based type resolution** — the AST resolver for complex generic
type annotations is removed in favor of the simpler generic alias approach
(`parse(Wrapper[Data], data)`).

**Migration:**
```python
# Field-level aliases still work ✅
@dataclass
class User:
    user_id: str = field(metadata={"alias": "id"})

user = parse(User, {"id": "abc123"})

# Removed ❌
parse(User, data, case_insensitive=True)
parse(User, data, alias_generator=camel_case)
parse(User, data, aliases={"user_id": "uid"})
parse(User, data, extra="allow")
dump(user, alias_generator=camel_case)
```

#### Unified Transcript Format Across Adapters

The transcript system switches from adapter-specific formats to a canonical
schema. All log events, context field names, and entry types change:

**Log events renamed** (`transcript.collector.*` → `transcript.*`):
- `transcript.collector.entry` → `transcript.entry`
- `transcript.collector.start` → `transcript.start`
- `transcript.collector.stop` → `transcript.stop`
- `transcript.collector.path_discovered` → `transcript.path_discovered`
- `transcript.collector.subagent_discovered` → `transcript.subagent_discovered`
- `transcript.collector.error` → `transcript.error`

**Log context fields renamed:**
- `transcript_source` → `source`
- `raw_json` → `raw`
- `parsed` → `detail` (Claude SDK entries wrapped under `detail.sdk_entry`)

**Entry types canonicalized:**
- `user` → `user_message`
- `assistant` → `assistant_message`
- `summary` → `system_event`
- `system` → `system_event`
- `unparsed` → removed (entries are always parsed now)
- `invalid` → `unknown`

**Config field renamed:** `TranscriptCollectorConfig.emit_raw_json` →
`TranscriptCollectorConfig.emit_raw`. `TranscriptCollectorConfig.parse_entries`
removed entirely.

**Message splitting:** Claude SDK assistant messages containing `tool_use`
blocks are now split into separate `assistant_message` + `tool_use` entries.
User messages containing `tool_result` blocks are similarly split. Entry counts
will differ from previous behavior.

**Query database:** Schema bumped to v9. Existing cached databases will be
rebuilt automatically.

#### Removed `@pure` Decorator from Production Code

The `@pure` decorator has been removed from all 28 production functions across
`cli/query.py` (18 functions), `runtime/session/` (8 functions), and public
API listings. The decorator infrastructure remains in the `dbc` module but is
no longer applied or exported. Related purity tests are deleted (159 lines
removed from `test_dbc_contracts.py`).

---

### New Features

#### ACP Adapter with OpenCode Integration

New two-layer ACP adapter stack (`src/weakincentives/adapters/acp/`, ~1,900
lines; `src/weakincentives/adapters/opencode_acp/`, ~150 lines) for running
WINK agents against any ACP-compatible CLI.

**`ACPAdapter`** (`adapter.py`, 737 lines): Core adapter implementing the full
ACP protocol flow — CWD resolution, MCP server lifecycle, ACP handshake
(`initialize` + `new_session`), session configuration, prompt dispatch, drain
quiet period, text extraction, and structured output resolution. Provides
subclass hooks: `_validate_model()`, `_handle_mode_error()`,
`_detect_empty_response()`.

**`ACPClient`** (`client.py`, 264 lines): Implements `acp.interfaces.Client`
protocol. Handles `session_update()` notifications (tracking
`AgentMessageChunk`, `AgentThoughtChunk`, `ToolCallStart`, `ToolCallProgress`),
permission requests, and filesystem access with workspace-boundary enforcement.
Generates synthetic monotonic tool IDs (`_tc_1`, `_tc_2`, ...) for agents that
send empty `tool_call_id` values.

**`MCPHttpServer`** (`_mcp_http.py`, 295 lines): In-process HTTP server
exposing WINK tools to ACP agents via `StreamableHTTPServerTransport` with
per-instance bearer token authentication. Runs uvicorn in a daemon thread.

**`ACPTranscriptBridge`** (`_transcript.py`, 186 lines): Buffers streaming
deltas and emits consolidated transcript entries on type change or session end.
Reduces ~850 raw token-level entries to ~70 consolidated entries.

**`StructuredOutputTool`** (`_structured_output.py`, 162 lines): MCP-compatible
tool that captures structured output. Discovered via `tools/list`, validates
against JSON schema. Thread-safe capture for cross-thread access.

**`OpenCodeACPAdapter`** (`opencode_acp/adapter.py`, 82 lines): Thin subclass
adding model validation against `available_models`, empty-response detection
(raises on zero `AgentMessageChunk` updates), and adapter name override.

**New optional dependency:** `weakincentives[acp]` requires
`agent-client-protocol>=0.8.0`, `mcp>=1.26.0`, `uvicorn>=0.40.0`.

**New adapter name constants:** `ACP_ADAPTER_NAME`, `OPENCODE_ACP_ADAPTER_NAME`.

#### Unified Transcript System (`TranscriptEmitter` + Adapter Bridges)

A shared transcript runtime (`src/weakincentives/runtime/transcript.py`, 283
lines) provides adapter-agnostic transcript emission with 9 canonical entry
types: `user_message`, `assistant_message`, `tool_use`, `tool_result`,
`thinking`, `system_event`, `token_usage`, `error`, `unknown`.

- **`TranscriptEmitter`**: Thread-safe emitter with sequence numbering,
  timestamp capture, per-source counters, and exception suppression. Emits
  `transcript.entry` DEBUG logs with a full envelope (`prompt_name`, `adapter`,
  `entry_type`, `sequence_number`, `source`, `timestamp`, `session_id`,
  `detail`, `raw`). `start()`/`stop()` methods emit summary statistics.
- **`TranscriptEntry`**: Frozen dataclass representing a single canonical
  transcript entry.
- **`TranscriptSummary`**: Aggregate statistics (total entries, entries by
  type/source, timestamps).
- **`reconstruct_transcript()`**: Converts log records back to typed
  `TranscriptEntry` objects for post-hoc analysis.

**Codex App Server integration** via `CodexTranscriptBridge`
(`adapters/codex_app_server/_transcript.py`, 186 lines):
- Maps Codex JSON-RPC notifications to canonical types
- Suppresses streaming deltas (`item/agentMessage/delta`, etc.)
- Handles `turn/started`, `item/started`, `item/completed`,
  `item/reasoning/completed`, `thread/tokenUsage/updated`, `turn/completed`
- Emits `tool_use`/`tool_result` for WINK-bridged tool calls

**ACP integration** via `ACPTranscriptBridge`
(`adapters/acp/_transcript.py`, 186 lines):
- Buffers `AgentMessageChunk` and `AgentThoughtChunk` streaming deltas
- Consolidates `ToolCallProgress` updates per `tool_call_id`
- Emits consolidated `assistant_message`, `thinking`, `tool_use`, `tool_result`
  entries on type change or session end

**Claude SDK message splitting** — mixed content blocks are split into
separate transcript entries for structural consistency with Codex:
- `_emit_assistant_split()`: text blocks → `assistant_message`, each tool_use
  block → `tool_use`
- `_emit_user_tool_result_split()`: non-tool blocks → `user_message`, each
  tool_result block → `tool_result`

**New configuration fields:**
- `CodexAppServerClientConfig.transcript: bool = True`
- `CodexAppServerClientConfig.transcript_emit_raw: bool = True`

#### Debug Bundle Transcript Artifact

`BundleWriter._extract_transcript()` scans `app.jsonl` for
`event == "transcript.entry"` records and writes them to a separate
`transcript.jsonl` artifact in the bundle. `DebugBundle.transcript` property
provides typed access. The `wink query` CLI loads transcripts from this artifact
first, falling back to log scanning.

---

### Improvements

#### Section-Level Skill Registration with Duplicate Detection

`PromptRegistry` now tracks skill names across sections with duplicate
detection. A `PromptValidationError` is raised if two sections declare skills
with the same name. The registry precomputes a `subtree_has_skills` index for
O(1) lookups during rendering, and skills participate in progressive disclosure
alongside tools.

#### Expanded `wink query` Extraction Pipeline

New extraction functions handle the unified transcript format:
- `_apply_split_block_details()` — extracts `tool_name`/`tool_use_id` from
  split content blocks
- `_apply_notification_item_details()` — extracts tool metadata from Codex
  notification items
- `_apply_bridged_tool_details()` — extracts tool metadata from WINK bridged
  tool events
- `tool_call_id` added as fallback key alongside `id` and `tool_use_id` for ACP
  adapter compatibility
- Two-phase transcript loading: artifact-first, log-fallback

#### Debug UI Improvements

- `wink debug` index page now returns `HTMLResponse` with `Cache-Control:
  no-cache` header. HTTP middleware `_no_cache_static()` added to prevent stale
  static asset caching.

#### Integration Test Timeout Increase

Integration test timeout bumped from 180s to 300s for `integration-tests`,
`redis-tests`, `redis-standalone-tests`, and `redis-cluster-tests` in the
Makefile.

#### Modernized Type Hints and Error Handling

- `evals/_helpers.py`: `submit_dataset` and `submit_experiments` migrated from
  `TypeVar` declarations to PEP 695 generic syntax (`def func[InputT, ...]`)
- `_ephemeral_home.py`: Bare `assert api_key is not None` replaced with
  explicit `ValueError` raise
- `dataclasses/__init__.py`: Redundant `cast(Callable[[type[T]], type[T]], ...)`
  simplified to direct `dataclass()` call

---

### Refactoring

#### Large-Scale Modular Decomposition

15+ large source files were decomposed into focused, single-responsibility
private modules. All public APIs remain unchanged — imports continue to work
through package `__init__.py` re-exports. In total, ~30 new private modules
were created. Key decompositions:

**Claude Agent SDK adapter** (`adapters/claude_agent_sdk/`):
- `adapter.py` (1,774 → 695 lines): extracted `_message_extraction.py`,
  `_schema_normalization.py`, `_result_extraction.py`, `_sdk_execution.py`,
  `_sdk_options.py`
- `isolation.py` (1,249 → 551 lines): extracted `_ephemeral_home.py`,
  `_model_utils.py`
- `_transcript_collector.py`: parsing logic extracted to
  `_transcript_parser.py`
- `_hooks.py`: extracted `_hook_context.py` (HookStats, HookContext) and
  `_hook_tools.py` (constraint checking, tool processing)

**Codex App Server adapter** (`adapters/codex_app_server/`):
- `adapter.py`: extracted `_protocol.py` (JSON-RPC execution flow),
  `_response.py` (response building), `_schema.py` (tool spec conversion)

**Debug system** (`debug/`):
- `bundle.py` (1,591 → 459 lines): extracted `_bundle_reader.py`,
  `_bundle_writer.py`, `_bundle_retention.py`
- `environment.py`: git operations extracted to `_git.py`

**CLI** (`cli/`):
- `query.py` (2,204 → 996 lines): extracted `_query_helpers.py`,
  `_query_tables.py`, `_query_transcript.py`, then further into
  `_query_builders.py`, `_query_environment.py`, `_query_formatters.py`.
  `build_all_tables()` orchestration function added.
- `wink.py`: doc subcommand logic extracted to `_docs.py`
- `debug_app.py` (736 → 34 lines): extracted `_bundle_store.py`

**Prompt system** (`prompt/`):
- `registry.py`: extracted `_validators.py` (section/tool/skill validation
  functions, registry invariant callbacks) and `_indices.py`
  (`build_registry_indices()`)
- `overrides/validation.py` (853 lines): split into `_section_overrides.py`,
  `_tool_overrides.py`, `_task_example_overrides.py`

**Serde** (`serde/`):
- `parse.py`: extracted `_coercers.py` (all type coercion functions) and
  `_generics.py` (generic type resolution, TypeVar mapping)

**Formal verification** (`formal/`):
- `__init__.py`: extracted `_metadata.py` (data types) and `_codegen.py`
  (TLA+ generation)

**Filesystem** (`filesystem/`):
- `_host.py`: git operations extracted to `_git_ops.py`

**Runtime** (`runtime/`):
- `agent_loop.py`: bundle metrics/artifacts extracted to
  `_agent_loop_bundle.py`

**Contrib** (`contrib/mailbox/`):
- `_redis.py`: Lua scripts extracted to `_lua_scripts.py`, TLA+ spec to
  `_redis_spec.py`

All backward-compatibility `__getattr__` shims and `TYPE_CHECKING` re-export
aliases removed — imports updated to reference defining modules directly.

#### Test Suite Reorganization

Five monolithic test files decomposed into focused test modules:

- `test_hooks.py` (2,044 lines) → 8 focused modules in `tests/adapters/claude_agent_sdk/hooks/`
- `test_bundle.py` (2,248 lines) → 6 modules by concern (config, writer, loader, retention, filesystem, storage)
- `test_wink_query_database.py` (2,259 lines) → 5 modules (core, transcript, logs, views, files)
- `test_adapter.py` for Claude SDK (3,153 lines) → 11 modules in `tests/adapters/claude_agent_sdk/`
- `test_agent_loop.py` (2,207 lines) → 10 modules in `tests/runtime/agent_loop/`

All restructured tests preserve existing assertions with no functional changes.

---

### Documentation

#### Unified Transcript Specification

New `specs/TRANSCRIPT.md` (486 lines) replaces the Claude-specific
`specs/TRANSCRIPT_COLLECTION.md` (409 lines, deleted). The new spec covers:

- Common envelope schema with 9 canonical entry types
- Adapter mapping tables for Claude Agent SDK, Codex App Server, and ACP
- `TranscriptEmitter`, `TranscriptEntry`, `TranscriptSummary` class specs
- `CodexTranscriptBridge` and `ACPTranscriptBridge` specifications
- `reconstruct_transcript()` function spec
- Debug bundle `transcript.jsonl` integration
- Seven formal invariants (envelope completeness, sequence monotonicity, type
  vocabulary, non-blocking emission, adapter labeling, source consistency,
  timestamp ordering)
- Configuration changes and migration guide

#### ACP Adapter Specifications

- `specs/ACP_ADAPTER.md` (430 lines): Generic ACP adapter protocol flow, MCP
  HTTP bridging, session lifecycle, transcript consolidation, structured output,
  tool event dispatch
- `specs/OPENCODE_ADAPTER.md` (250 lines): OpenCode-specific model validation,
  empty response detection, quirk handling, config defaults
- `specs/OPENCODE_ACP_ADAPTER.md` trimmed to overview pointing at the two new
  specs

#### Adapter Compatibility Kit (ACK) Specification

New `specs/ACK.md` (931 lines) defines a unified integration testing framework
across all provider adapters:

- `AdapterFixture` protocol with capability-based test gating
- Three test tiers: basic evaluation, observability, advanced behavior
- Shared scenario builders and assertion helpers
- Transcript-centric validation with envelope completeness and sequence
  monotonicity invariants
- Four-phase migration plan from per-adapter tests to unified ACK suite
- Concrete fixture examples for Claude SDK, Codex, ACP, and OpenCode adapters

#### Debugging Guide Expansion

`guides/debugging.md` expanded from 237 to 407 lines (+170 lines):

- Debug UI examples updated from `.jsonl` to `.zip` bundle format
- Capability list expanded from 4 to 8 categories (session state, logs,
  transcripts, tool calls, files, environment, metrics, errors)
- New bundle management subsection (auto-detection, listing, reloading)
- Complete `wink query` inline reference: CLI usage, 7 core tables, 5
  environment tables, 7 pre-built views, 7 example SQL queries
- New "Instructing Coding Agents to Use wink query" section with prompt
  templates, key views, investigation queries, and a 5-step agent workflow

#### Query Guide Updates

`guides/query.md` expanded with:
- New "Environment Tables" section documenting 6 `env_*` tables with 5 example
  queries
- Four new transcript views added to the views reference table
  (`transcript_flow`, `transcript_tools`, `transcript_thinking`,
  `transcript_agents`)

#### README: Execution Harnesses Documentation

`README.md` restructured from a Claude-SDK-only section to a multi-harness
layout:

- New `## Execution Harnesses` parent section with introductory paragraph
- `### Claude Agent SDK` subsection with specific native tool names (Read,
  Write, Edit, Glob, Grep, Bash) and install command
- New `### Codex App Server` subsection with feature list (native tools,
  dynamic tools, structured output, no extra deps), install command, and
  complete Python usage example

#### Codex App Server Spec Update

`specs/CODEX_APP_SERVER.md` updated to reflect modular architecture: documents
new module locations (`_protocol.py`, `_response.py`, `_schema.py`,
`_transcript.py`, `_events.py`).

#### Serde Documentation Updated for API Simplification

- `specs/DATACLASSES.md` — Removed `case_insensitive`, `alias_generator`,
  `aliases` parameters from `parse()` table; removed `alias_generator` from
  `dump()` table; updated `extra` to document only `"ignore"` and `"forbid"`
  modes; added `alias` to constraints table for field-level alias support
- `guides/serialization.md` — Removed "Alias Generator", "Explicit Aliases
  Mapping", and "Case-Insensitive Parsing" sections; removed `extra="allow"`
  mode and examples; removed `alias_generator` from `dump()` and `schema()`
  option tables

#### WINK Presentation Slides

New `wink-presentation.md` (472 lines) — Marp-based presentation covering WINK
architecture, design philosophy, and usage patterns. Published via
`.github/workflows/presentation.yml` GitHub Actions workflow.

---

### Developer Experience

#### Code Length Checker

New `CodeLengthChecker` in `toolchain/checkers/code_length.py` integrated into
`make check`. Uses AST-based analysis to enforce:

- Maximum file length: 720 lines
- Maximum function/method length: 120 lines

Known violations tracked in `toolchain/checkers/code_length_baseline.txt` (79
entries). New violations fail the check; existing ones warn.

Six methods refactored to satisfy the 120-line limit:
`FormalSpec.to_tla`, `QueryDatabase._build_views`, `CompositeSnapshot.from_json`,
`model_check`, and two test helpers.

---

### Dependencies

- `redis`: 7.1.0 → 7.1.1
- `claude-agent-sdk`: 0.1.27 → 0.1.35
- `fastapi`: 0.128.0 → 0.129.0
- `ruff`: 0.15.0 → 0.15.1
- `ty`: 0.0.14 → 0.0.16
- `hypothesis`: 6.151.4 → 6.151.6
- New optional: `agent-client-protocol>=0.8.0`, `mcp>=1.26.0` (via `[acp]` extra)
- Lint rule: `ANN201` replaced with `RUF069` in per-file test ignores

---

## v0.25.0 — 2026-02-06

### TL;DR

WINK adds a **Codex App Server adapter** (~10,000 lines) for running agents in
OpenAI's Codex sandboxed environment via JSON-RPC over stdio, with shared MCP
tool bridging extracted into a reusable `_shared` adapter module. Simultaneously,
WINK **standardizes on agentic harness integrations** by removing the OpenAI and
LiteLLM adapters (~15,600 lines) and all harness-provided tool sections (VFS,
Podman, planning, Asteval, optimizers, examples — ~19,000 lines). A new
**prompt lifecycle** adds `Section.cleanup()` and `Prompt.cleanup()` with
automatic invocation by AgentLoop, plus a visibility expansion retry limit.

The **feedback system** gains **concurrent multi-provider evaluation**,
**file-based trigger activation** (`FileCreatedTrigger`), and **XML-style
semantic tags** (`<feedback>`, `<blocker>`). The **Claude adapter** adds
**`call_id` correlation for MCP-bridged tools**, **`RenderedTools` event
emission**, **SDK-native exception types**, and replaces `max_thinking_tokens`
with **adaptive `reasoning` effort levels**. **Transcript collection is now
enabled by default**, and the **transcript tab hides** when bundles lack
transcript data.

The **debug web UI** undergoes a **full modular decomposition** — `app.js`
broken into a state store and 7 view modules — with new **markdown rendering**,
**infinite scroll** in the zoom modal, and extracted reusable components. The
**default model updates to Claude Opus 4.6**.

Developer experience improves with **pytest-testmon**, **AutoFormatChecker**
(auto-fix locally, check-only in CI), **parallelized CI** across 6 test groups,
and **pre-commit hooks enforcing full CI suites**. `BundleConfig` replaces
`debug_bundle_dir`. The **Session** is refactored into **SliceStore**,
**ReducerRegistry**, and **SessionSnapshotter**. New specs cover **Codex App
Server** and **OpenCode ACP** integrations.

---

### Fixed

- Hardened Codex App Server workspace copying to skip symlink files when
  `follow_symlinks=False`, preventing escaped symlink artifacts from entering
  the temp workspace.
- Fixed a Codex client request race that could hang indefinitely when the
  subprocess exited between pending-registration and response wait.
- Updated `AgentLoop` debug bundle adapter labeling to emit canonical
  `"codex_app_server"` for `CodexAppServerAdapter` instances.
- Removed duplicate `prompt.cleanup()` invocation risk in the bundled execution
  error path via `prompt_cleaned_up` guard.
- Added `_MAX_VISIBILITY_RETRIES = 10` guard in `AgentLoop._evaluate_with_retries()`
  to prevent infinite visibility expansion loops.
- Corrected Bedrock model ID for Opus 4.6 — dropped `:0` suffix
  (`us.anthropic.claude-opus-4-6-v1:0` → `us.anthropic.claude-opus-4-6-v1`).
- JavaScript tests now properly executed in CI test job (Bun was previously
  only installed in static-analysis job; `45292cb1` fixes `be065a9e`).

---

### Breaking Changes

#### Removed `deadline` from `AgentLoopConfig`

A deadline is a specific point in time. Setting one at config-construction time
means it can expire before execution even starts. Deadlines belong on
per-request objects (`AgentLoopRequest`, `Budget`, or `execute(deadline=...)`).

**Migration:**
```python
# Old ❌
config = AgentLoopConfig(
    deadline=Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5)),
)

# New ✅ — pass deadline per-request
request = AgentLoopRequest(
    request=my_request,
    deadline=Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5)),
)

# Or via execute()
loop.execute(my_request, deadline=Deadline(...))
```

#### Removed OpenAI and LiteLLM Adapters

WINK now focuses exclusively on agentic harness integrations. The `OpenAIAdapter`
and `LiteLLMAdapter` have been removed along with all supporting infrastructure:
`InnerLoop`, `ToolExecutor`, `ResponseParser`, `ProviderResponse`,
`DeadlineUtils`, token usage tracking, and rendering utilities — totaling
~15,600 lines across 57 files.

**Removed:**
- `weakincentives.adapters.openai` module (911 lines)
- `weakincentives.adapters.litellm` module (487 lines)
- `weakincentives.adapters.inner_loop` module (617 lines)
- `weakincentives.adapters.tool_executor` module (877 lines)
- `weakincentives.adapters.response_parser`, `rendering`, `provider_response`,
  `deadline_utils`, `_tool_messages`, `utilities`, `token_usage` modules
- `OpenAIClientConfig`, `OpenAIModelConfig` config classes
- `LiteLLMClientConfig`, `LiteLLMModelConfig` config classes
- `OPENAI_ADAPTER_NAME`, `LITELLM_ADAPTER_NAME` constants
- Optional dependencies: `weakincentives[openai]`, `weakincentives[litellm]`

**Migration:**
```python
# Old ❌
from weakincentives.adapters.openai import OpenAIAdapter
adapter = OpenAIAdapter(model="gpt-4o")

# New ✅
from weakincentives.adapters.claude_agent_sdk import ClaudeAgentSDKAdapter
adapter = ClaudeAgentSDKAdapter()
```

#### Removed Harness-Provided Tool Sections

All tool sections that duplicate capabilities now provided by execution harness
runtimes have been removed (~19,000 lines across 94 files):

**Removed modules:**
- `weakincentives.contrib.tools.vfs` — VFS filesystem tools (855 lines)
- `weakincentives.contrib.tools.vfs_mounts` — VFS mount configuration (355 lines)
- `weakincentives.contrib.tools.vfs_types` — VFS type definitions (792 lines)
- `weakincentives.contrib.tools.planning` — Planning tools (658 lines)
- `weakincentives.contrib.tools.podman` — Podman sandbox tools (1,336 lines)
- `weakincentives.contrib.tools.podman_connection` — Podman connection management
- `weakincentives.contrib.tools.podman_eval` — Podman evaluation tools
- `weakincentives.contrib.tools.asteval` — Asteval math evaluation (1,099 lines)
- `weakincentives.contrib.tools._context` — Tool context module
- `weakincentives.optimizers` — Entire optimizer package (~652 lines)
- `weakincentives.examples` — Entire examples package (5 scripts, ~778 lines)

**Migration:** Use `WorkspaceSection` (from `weakincentives.prompt`)
with host mounts instead of VFS tools. Use SDK native Bash tool instead of
Asteval.

#### EvalLoopConfig.debug_bundle_dir → debug_bundle

The `debug_bundle_dir: Path | None` field is replaced with `debug_bundle:
BundleConfig | None`, adding an `enabled` flag, `storage_handler` callback,
and `retention` policy.

**Migration:**
```python
# Old ❌
config = EvalLoopConfig(debug_bundle_dir=Path("/bundles"))

# New ✅
from weakincentives.debug.bundle import BundleConfig
config = EvalLoopConfig(debug_bundle=BundleConfig(target=Path("/bundles")))
```

#### Transcript Collection Now Enabled by Default

The Claude Agent SDK adapter's `transcript_collection` default changed from
`None` (disabled) to `TranscriptCollectorConfig()` (enabled). The adapter uses
`contextlib.nullcontext()` as a fallback when explicitly set to `None`. To
restore the previous behavior:

```python
config = ClaudeAgentSDKClientConfig(transcript_collection=None)
```

#### Default Model Changed to Claude Opus 4.6

The default model throughout the codebase changed from `claude-sonnet-4-5-20250929`
to `claude-opus-4-6`. This affects `DEFAULT_MODEL`, `DEFAULT_BEDROCK_MODEL`,
`ClaudeAgentSDKAdapter.__init__`, and `ClaudeAgentSDKModelConfig.model`. A new
Bedrock mapping `"claude-opus-4-6" → "us.anthropic.claude-opus-4-6-v1"` was added.

#### Replaced max_thinking_tokens with Reasoning Effort Levels

`ClaudeAgentSDKModelConfig.max_thinking_tokens: int | None` replaced with
`reasoning: ReasoningEffort | None` where `ReasoningEffort = Literal["low",
"medium", "high", "max"]`. Default changed from disabled (`None`) to `"high"`.

**Migration:**
```python
# Old ❌
config = ClaudeAgentSDKModelConfig(max_thinking_tokens=16000)

# New ✅
config = ClaudeAgentSDKModelConfig(reasoning="max")
```

#### Removed Unsupported SDK Hook Types

`create_subagent_start_hook` and `create_notification_hook` removed from the
Claude Agent SDK adapter hooks. Custom `PostToolUseInput` dataclass and
`HookCallback`/`AsyncHookCallback` type aliases replaced with SDK-native types
(`PostToolUseHookInput`, `PreToolUseHookInput`, `StopHookInput`,
`SyncHookJSONOutput`, etc.). Added `HookConstraints` dataclass to group
optional deadline/budget/heartbeat/run_context parameters.

#### Feedback Providers Now Run Concurrently

The feedback evaluation model changed from "first match wins" to "all matching
providers run." Each provider maintains **independent trigger state** via
scoped methods `last_feedback_for_provider()` and
`tool_calls_since_last_feedback_for_provider()` on `FeedbackContext`. All
matching feedback blocks are collected and combined with `"\n\n".join()`.

---

### New Features

#### Codex App Server Adapter

Full adapter implementation (`src/weakincentives/adapters/codex_app_server/`,
~2,000 lines) for running WINK agents in OpenAI's Codex sandboxed environment:

- **`CodexAppServerAdapter`**: JSON-RPC 2.0 over stdio, notification-based
  transcript collection, structured output with OpenAI-strict schema
  transformation, tool bridging via MCP
- **`CodexClient`**: Async JSON-RPC client with request/response correlation,
  notification routing, and subprocess lifecycle management
- **`WorkspaceSection`**: Host mount support with symlink-safe copying
  and file filtering by glob/size (now in `weakincentives.prompt`)
- **`CodexAppServerModelConfig`/`CodexAppServerClientConfig`**: Configuration
  with defaults (model `"gpt-5.3-codex"`, approval policy, sandbox toggles)
- Event mapping from Codex `item/completed` notifications to WINK `ToolInvoked`
  and `TokenUsage` events

The code reviewer example now supports `--adapter codex` for Codex-based
reviews alongside Claude.

#### Shared Adapter Module

MCP tool bridging logic extracted from the Claude SDK adapter into
`src/weakincentives/adapters/_shared/` (~750 lines), shared between Claude SDK
and Codex adapters:

- `_bridge.py`: MCP tool bridging with `BridgedTool` execution
- `_async_utils.py`: Shared async utilities
- `_visibility_signal.py`: Visibility expansion signal handling

#### Prompt Lifecycle: cleanup()

New `Section.cleanup()` and `Prompt.cleanup()` methods enable sections to
release resources (e.g., temporary directories) after execution. `AgentLoop`
automatically calls `prompt.cleanup()` in all execution paths (normal, bundled,
error) with a `prompt_cleaned_up` guard preventing double invocation.

#### call_id Correlation for MCP-Bridged Tools

New `MCPToolExecutionState` provides thread-safe correlation between SDK
`PreToolUse` hook events and MCP-bridged tool executions using a bounded deque
with MD5-based parameter matching. `ToolInvoked` events now include `call_id`
for MCP-bridged tools, enabling end-to-end tool execution tracing.

#### FileCreatedTrigger for File-Based Feedback Activation

New `FileCreatedTrigger` fires once when a specified file appears on the
filesystem. Includes `FileCreatedTriggerState` session slice tracking fired
triggers via `frozenset[str]`. Also adds `StaticFeedbackProvider` — a simple
provider returning a fixed message:

```python
FeedbackProviderConfig(
    provider=StaticFeedbackProvider(feedback="Plan file detected."),
    trigger=FeedbackTrigger(on_file_created=FileCreatedTrigger(filename="plan.md")),
)
```

#### Tool Schema Rendering and Event Correlation

The Claude Agent SDK adapter now emits `RenderedTools` events alongside
`PromptRendered` events, correlated via shared `render_event_id` (UUID4).
Tool schemas (name, description, JSON Schema parameters) extracted via
`tool_to_spec()`. Dispatch failures logged without aborting evaluation.

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

#### Intelligent Test Selection with pytest-testmon

Local test runs now only execute tests affected by your changes:

- First run builds `.testmondata` coverage database
- Subsequent runs skip unaffected tests automatically
- CI still validates 100% coverage on all tests
- Detected via `CI` environment variable

```bash
make test          # Local: runs only affected tests
CI=true make test  # CI: runs full suite with 100% coverage
```

#### AutoFormatChecker with Dual-Mode Operation

New `AutoFormatChecker` class in the toolchain:

- **Locally**: Runs formatters with auto-fix, reports which files were changed
- **In CI** (detected via `GITHUB_ACTIONS`/`CI`): Runs in check-only mode

Supports JSON output parsing (ruff) and text-based `FileListParser` callbacks
(mdformat). The markdown checker now uses `AutoFormatChecker` with a regex
parser. Info-severity diagnostics (cyan) display even when checks pass.

#### Markdown File Support in Debug Viewer

The debug app file browser and transcript viewer now render markdown files:

- `.md` files rendered via `_markdown.render()` to HTML
- Toggle buttons for "Rendered" vs "Raw" views
- Transcript entries with markdown content get a `content_html` field
- Invalid UTF-8 falls back to binary display

#### Infinite Scroll in Transcript Zoom Modal

Navigating past the last loaded entry automatically fetches more entries:

- `zoomNext()` converted to async with `zoomNextWithLoad()` helper
- Retry counter (`MAX_TRANSCRIPT_LOAD_RETRIES = 3`) prevents infinite loops
- Key events suppressed while a load is pending

#### Transcript Tab Visibility Toggle

The transcript tab in the debug viewer automatically hides when the bundle
contains no transcript data. Keyboard shortcut numbers renumber dynamically
to skip hidden tabs.

---

### Improvements

#### SDK-Native Exception Types for Error Handling

Refactored Claude Agent SDK adapter error handling to use `isinstance()` checks
against SDK types (`CLINotFoundError`, `CLIConnectionError`, `ProcessError`,
`CLIJSONDecodeError`) instead of string-based comparisons. Decomposed monolithic
`normalize_sdk_error()` into focused handler functions. Test error mocks now
inherit from actual SDK exception types.

#### Generic Snapshotable Protocol

`Snapshotable` now accepts a type parameter (`Snapshotable[SnapshotT]`),
replacing `Any` annotations on `snapshot()` and `restore()`.

#### TypedDict Definitions for LLM API Payloads

New `_api_types.py` module (269 lines) provides type-safe definitions for all
LLM provider payloads, replacing `dict[str, Any]` across seven adapter modules.

#### Session Refactored into Specialized Subsystems

Session internals extracted into three focused components:

- **SliceStore**: Thread-safe typed slice instances with policy-based factories
- **ReducerRegistry**: Event-type-to-reducer mapping with multi-reducer support
- **SessionSnapshotter**: Snapshot creation/restoration with validation

Session becomes a thin facade. The `_locked_method` decorator removed in favor
of explicit lock management.

#### Type Safety in Serialization

Replaced blanket pyright suppression in `serde/dump.py` (7 disabled rules) with
three targeted `cast()` calls.

#### Bun Test Integration into Unified Toolchain

JavaScript test execution moved into the Python toolchain. New
`parse_bun_test()` parser extracts failures, stack traces, and pass/fail counts
into `Diagnostic` objects. Graceful Bun-not-installed handling.

#### Pre-Commit Hooks Enforce Full CI Test Suite

The pre-commit hook now runs `CI=true make check`, forcing the full test suite
before every commit to prevent CI failures from testmon-skipped tests.

#### Suppressed Build Output Noise

- `--quiet` flag added to all `uv run` commands in Makefile
- `--no-header` added to local pytest runs
- Biome check suppresses stdout on success
- Bandit AST patching warnings suppressed
- Removed `--preview` flag from ruff format JSON check

---

### Documentation

#### Quickstart Guide Redesigned

The quickstart guide now leads with cloning a
[starter project](https://github.com/weakincentives/starter), running a
"secret trivia game" agent immediately, then explaining concepts through that
concrete example.

#### CLAUDE.md Enhanced with Architectural Guidance

Added "Core Philosophy" and "Guiding Principles" sections (prompt-is-agent,
event-driven state, definition-vs-harness, policies-over-workflows,
transactional tools). Added Key Specs reference table and mandatory Git hooks
documentation. `AGENTS.md`/`GEMINI.md`/`WARP.md` converted to symlinks.

#### Guardrails Specification Consolidated

Three separate specs (`FEEDBACK_PROVIDERS.md`, `TASK_COMPLETION.md`, tool
policies from `TOOLS.md`) merged into unified `specs/GUARDRAILS.md` (439 lines)
with an overview table showing how all three mechanisms complement each other.

#### OpenCode ACP Adapter Specification

New specification (`specs/OPENCODE_ACP_ADAPTER.md`, 549 lines) documents
`OpenCodeACPAdapter` for delegating agentic execution to OpenCode via ACP.

#### Codex App Server Specification

New specification (`specs/CODEX_APP_SERVER.md`) documents the Codex App Server
adapter for running WINK agents via JSON-RPC 2.0 over stdio. The spec was
rewritten after 20 protocol probes against `codex-cli 0.98.0`, fixing 18+
discrepancies:

- Replaced MCP tool bridging with Codex **native dynamic tools** protocol
- Corrected `ApprovalPolicy` values, `SandboxMode` enum, `ReasoningEffort`
  values, and `agentMessage/delta` field path
- Structured output uses native `outputSchema` on `turn/start`
- Documents dual notification system (v2 `item/*` vs v1 `codex/event/*`)

#### ROADMAP Refocused on Integrations

Replaced "Tracing & Observability" → "ACP Integration", added "Codex App Server
Integration", "Session Durability" → "Checkpointer Mechanism", "Named Entities"
→ "Basic Metrics".

---

### CI / Infrastructure

#### Parallelized Test Execution

Tests now run in 6 parallel matrix groups with coverage combined in a dedicated
`coverage` job enforcing the 100% threshold:

- `test-group-1`: Adapters
- `test-group-2`: CLI + Contrib
- `test-group-3`: Evals + Serde
- `test-group-4`: Prompt + Prompts
- `test-group-5`: Runtime
- `test-group-6`: Tools + Root + Misc

New `make test-parallel` and `make test-group-N` targets for local use.

#### Simplified Change Detection

Removed 12 module-specific change detection outputs. Heavy tests gated by
single `run_heavy_tests` flag.

#### JavaScript Testing Infrastructure

Bun test runner for debug web UI with 77+ unit tests. Pure utility functions
extracted to `lib.js`. Biome cognitive complexity threshold lowered from 25 to
8. Integrated into `make test` and `make check`.

#### Shared CI Setup Action

New `.github/actions/setup-env` composite action centralizes Python/uv/Bun
setup across 4 workflow files (ci, integration-tests, release).

#### Skip CI for Docs-Only Changes

Static analysis and test jobs skip for documentation-only PRs.

#### Claude Code Review Configuration

PR code review workflow updated to use Claude Opus 4.6 at maximum effort level.

---

### Debug Web UI

#### Full Modular Decomposition

The monolithic `app.js` (~2,700 lines) decomposed into modular architecture:

- **`store.js`**: Centralized mutable state via `createInitialState()`
- **`views/environment-view.js`**: System/Python/Git/container info
- **`views/filesystem-view.js`**: File browser with image preview and markdown
- **`views/keyboard-shortcuts.js`**: Global keyboard handler with vim-style j/k
- **`views/logs-view.js`**: Log display with faceted search and virtual scrolling
- **`views/sessions-view.js`**: Slice browser with expandable tree
- **`views/transcript-view.js`**: Transcript with source/type chip filtering
- **`views/zoom-modal.js`**: Detail zoom with JSON tree rendering

`app.js` reduced to ~163 lines of bootstrap/wiring code.

#### Extracted Reusable Components

`VirtualScroller`, `createFilterChip`/`createActiveFilter`, and
`renderZoomJsonTree` extracted into `components/` directory as ES modules.

---

### Dependencies

#### Major Updates

| Package | Old | New | Notes |
|---------|-----|-----|-------|
| redis | 5.0.0 | 7.1.0 | Major version bump |
| fastapi | 0.115.0 | 0.128.0 | 13 minor versions |
| uvicorn | 0.30.0 | 0.40.0 | 10 minor versions |
| claude-agent-sdk | 0.1.15 | 0.1.27 | 12 patch releases |
| pip | 25.3 | 26.0 | Major version bump |
| ruff | 0.14.5 | 0.15.0 | Major dev tool bump |
| ty | 0.0.1a35 | 0.0.14 | Alpha → stable |
| hypothesis | 6.100.0 | 6.151.4 | 51 minor versions |
| hatchling | 1.27.0 | 1.28.0 | Minor bump |

#### Other Updates

- pyjwt: 2.10.1 → 2.11.0
- tqdm: 4.67.1 → 4.67.2
- pyyaml: 6.0.0 → 6.0.3
- pytest-testmon: Added (≥2.1.0)
- pyright: 1.1.407 → 1.1.408
- bandit: 1.8.6 → 1.9.3
- pytest: 9.0.1 → 9.0.2
- pip-audit: 2.9.0 → 2.10.0

#### Removed Dependencies

- `openai` — No longer needed (adapter removed)
- `litellm` — No longer needed (adapter removed)
- `asteval` — No longer needed (tool section removed)
- `podman` — No longer needed (tool section removed)
- `aiohttp` — No longer needed (removed with adapters)

---

### Internal

#### Test Infrastructure Improvements

- Consolidated mock SDK exception classes into centralized
  `tests/adapters/claude_agent_sdk/error_mocks.py` (inheriting from real SDK types)
- Centralized CLI test helpers (`FakeLogger`, `FakeContextManager`) into
  `tests/cli/helpers.py`
- Refactored `wink query` tests into 4 focused modules with shared fixtures
- Split monolithic `test_dataclass_serde.py` (2,115 lines) into 4 modules
  with shared `_fixtures.py`
- Added comprehensive Codex App Server adapter tests (~3,600 lines) and
  integration tests (~2,400 lines)
- Enhanced `tests/helpers/__init__.py` with comprehensive usage guide

#### Other

- `code_reviewer_example.py` rewritten around `AgentLoop` + `InMemoryMailbox` +
  `InProcessDispatcher` with `--adapter {claude,codex}` flag
- Adapter derives `cwd` from workspace section's `HostFilesystem.root` when
  not explicitly set
- `CODEX_APP_SERVER_ADAPTER_NAME` constant added to public API
- `scratch/` added to `.gitignore`
- `.claude/settings.json` added for project configuration
- Simplified Makefile demo targets (`make demo-claude`, `make demo-codex`)
- Older changelog entries (v0.19.0 and earlier) pruned from CHANGELOG.md
- `pyproject.toml`: Added `--timeout=10 --timeout-method=thread` to default
  pytest addopts; `integration-tests` added to `norecursedirs`

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

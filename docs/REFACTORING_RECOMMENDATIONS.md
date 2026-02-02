# Refactoring Recommendations for Large Files

This document provides refactoring recommendations for files exceeding 1000 lines
(source) or 1000 lines (tests). Each recommendation focuses on achieving strong
encapsulation and quality abstractions while improving maintainability.

---

## Source Files (>900 lines)

### `src/weakincentives/cli/query.py` (2043 lines)

This file combines SQL schema generation, query execution, result formatting, and
CLI command handling into a single module. Extract the SQL schema definitions and
DDL statements into a dedicated `_schema.py` module, move the various formatters
(JSON, table, CSV) into a `_formatters.py` module implementing a `Formatter`
protocol, and isolate the query builder logic (predicate parsing, SQL construction)
into a `_query_builder.py` module with a clean `QueryBuilder` class that composes
predicates declaratively rather than through string manipulation.

### `src/weakincentives/debug/bundle.py` (1531 lines)

The bundle module conflates two distinct responsibilities: writing debug bundles
(`BundleWriter`) and reading them (`DebugBundle`). Split into `_writer.py` containing
`BundleWriter` and its internal helpers (artifact collection, SQLite insertion, JSON
serialization), and `_reader.py` containing `DebugBundle` with its query and iteration
logic; additionally, extract the SQLite table definitions and migration logic into a
shared `_schema.py` module that both reader and writer import, ensuring schema changes
remain synchronized.

### `src/weakincentives/adapters/claude_agent_sdk/adapter.py` (1381 lines)

This adapter interleaves SDK process management, MCP server setup, result parsing,
and error handling. Extract the subprocess/runner management into `_runner.py`
implementing a `SDKRunner` protocol, move MCP tool bridging logic into `_mcp.py`
with a `MCPBridge` class, and isolate the result parsing and normalization functions
into `_result_parser.py`; the main adapter should then compose these components
through dependency injection, making each piece independently testable and allowing
different runner implementations (subprocess, Docker, mock).

### `src/weakincentives/contrib/tools/podman.py` (1336 lines)

The Podman module bundles container lifecycle management, filesystem operations,
shell execution, and tool registration into one file. Factor out container lifecycle
(`_container.py` with `ContainerManager`), filesystem tools (`_filesystem.py` with
read/write/glob operations), shell execution (`_shell.py` with `ShellExecutor`), and
keep the main module as a thin facade that registers tools by composing these
components; this separation enables unit testing filesystem operations without
spinning up containers and allows swapping container runtimes.

### `src/weakincentives/adapters/claude_agent_sdk/_hooks.py` (1240 lines)

This module defines nine distinct hook factory functions with significant duplication
in input parsing and response construction. Group related hooks into focused modules:
`_tool_hooks.py` (pre/post tool use with shared `_ParsedToolData` extraction),
`_lifecycle_hooks.py` (stop, subagent start/stop, notification), and `_context_hooks.py`
(user prompt submit, pre-compact); define a common `HookResponse` builder protocol
that standardizes response construction and error handling across all hooks.

### `src/weakincentives/adapters/claude_agent_sdk/isolation.py` (1215 lines)

The isolation module mixes configuration dataclasses, environment detection logic,
ephemeral home creation, and skill mounting. Extract configuration types into
`_config.py` (keeping `IsolationConfig`, `SandboxConfig`, `BedrockConfig`,
`NetworkPolicy`), move AWS/Bedrock credential detection into `_auth.py` with a
`CredentialDetector` that returns a discriminated union, and isolate
`EphemeralHome` construction into `_home.py`; this allows testing credential
detection logic independently and makes the complex settings.json generation
more accessible for modifications.

### `src/weakincentives/contrib/mailbox/_redis.py` (1181 lines)

This module combines Lua script definitions, the formal TLA+ specification decorator,
and the `RedisMailbox` implementation. Extract Lua scripts into `_scripts.py` as
named constants with documentation, move the formal specification into a separate
`_formal_spec.py` that can be imported conditionally (keeping core runtime lean),
and retain the core `RedisMailbox` class with clear method groupings (send/receive,
acknowledge/nack/extend, lifecycle); consider a `_operations.py` module for the
atomic operation wrappers if method count grows.

### `src/weakincentives/contrib/tools/asteval.py` (1099 lines)

The asteval module combines AST validation, safe execution sandbox setup, result
serialization, and tool registration. Separate into `_validator.py` (AST node
whitelist, security checks, forbidden pattern detection), `_sandbox.py` (namespace
setup, safe builtins, execution context), and `_serializer.py` (result formatting,
truncation, type coercion for tool output); the main module becomes a thin tool
registration layer that composes these components, enabling security audits focused
on the validator without wading through serialization code.

### `src/weakincentives/serde/parse.py` (1054 lines)

The parse module implements type coercion for unions, literals, primitives,
sequences, mappings, and dataclasses in a single file with the `_ASTResolver`.
Extract type coercion functions into `_coercers.py` organized by type category
(primitives, containers, composites), move `_ASTResolver` into `_ast_resolver.py`
as it's a standalone utility for string annotation resolution, and keep the main
`parse()` function as the public entry point that dispatches to coercers; this
makes adding new type support (e.g., attrs classes) a localized change.

### `src/weakincentives/runtime/agent_loop.py` (1022 lines)

The agent loop mixes abstract orchestration, bundle artifact writing, metrics
collection, and retry logic. Extract bundle writing into `_bundle_writer.py`
(used by `execute_with_bundle`), move metrics/stats collection into `_metrics.py`
with a `LoopMetrics` class, and isolate the visibility timeout retry logic into
`_retry.py` with a `RetryStrategy` protocol; the core `AgentLoop` class should
focus solely on the prepare→evaluate→finalize lifecycle, delegating cross-cutting
concerns to composed components.

### `src/weakincentives/adapters/openai.py` (911 lines)

While under 1000 lines, this adapter would benefit from the same treatment as the
Claude adapter. Extract input/output message normalization into `_messages.py`,
move response parsing (`_choice_from_response`, `_extract_all_tool_calls`) into
`_response_parser.py`, and isolate throttle error detection into `_throttle.py`;
this mirrors the inner_loop module pattern and enables sharing throttle detection
logic with the LiteLLM adapter.

---

## Test Files (>1000 lines)

### `tests/cli/test_wink_query.py` (3330 lines)

This test file covers query parsing, SQL generation, formatting, and CLI integration
in one module. Split by test category: `test_query_parsing.py` (predicate parsing,
filter construction), `test_sql_generation.py` (DDL, query building), `test_formatters.py`
(JSON, table, CSV output), and `test_query_cli.py` (command-line integration); share
fixtures through a `conftest.py` in the cli test directory that provides pre-built
bundles and common query patterns.

### `tests/adapters/claude_agent_sdk/test_adapter.py` (2469 lines)

The adapter tests span SDK invocation, result parsing, error handling, and MCP
bridging. Organize into `test_adapter_invocation.py` (subprocess launching, timeout
handling), `test_adapter_parsing.py` (response normalization, tool call extraction),
`test_adapter_errors.py` (error classification, retry behavior), and
`test_adapter_mcp.py` (MCP tool bridging); extract common mock SDK responses into
fixtures in `conftest.py` to reduce duplication across test files.

### `tests/adapters/test_openai_adapter.py` (2257 lines)

Tests cover message normalization, response parsing, throttle handling, and
end-to-end evaluation. Split into `test_openai_messages.py` (input/output
normalization), `test_openai_responses.py` (choice extraction, tool call parsing),
`test_openai_throttle.py` (rate limit detection, retry-after parsing), and
`test_openai_evaluation.py` (full prompt evaluation); use parametrized fixtures
for the various OpenAI response shapes (text, tool calls, structured output).

### `tests/serde/test_dataclass_serde.py` (2115 lines)

Serde tests cover primitives, containers, unions, generics, and edge cases in a
monolithic file. Split by type category: `test_serde_primitives.py` (int, str,
bool, None), `test_serde_containers.py` (list, dict, tuple, set),
`test_serde_composites.py` (dataclass, nested, optional, union), and
`test_serde_generics.py` (TypeVar, Generic aliases); this organization mirrors
the recommended source split and makes test-driven type support additions easier.

### `tests/adapters/test_litellm_adapter.py` (2012 lines)

LiteLLM adapter tests mirror the OpenAI adapter pattern. Apply the same split:
`test_litellm_messages.py`, `test_litellm_responses.py`, `test_litellm_throttle.py`,
and `test_litellm_evaluation.py`; additionally extract common adapter test utilities
(mock response builders, assertion helpers) into `tests/adapters/conftest.py` to
share between OpenAI and LiteLLM test suites.

### `tests/debug/test_bundle.py` (1982 lines)

Bundle tests cover writing, reading, querying, and format validation. Split into
`test_bundle_writer.py` (artifact collection, SQLite insertion),
`test_bundle_reader.py` (query execution, iteration), and `test_bundle_schema.py`
(table definitions, migrations); share test bundles and fixtures through the debug
test directory's `conftest.py`.

### `tests/runtime/test_agent_loop.py` (1942 lines)

Agent loop tests span lifecycle management, message handling, retry behavior, and
bundle integration. Organize into `test_agent_loop_lifecycle.py` (prepare/finalize),
`test_agent_loop_messages.py` (message processing, visibility extension),
`test_agent_loop_retry.py` (timeout handling, redelivery), and
`test_agent_loop_bundle.py` (artifact writing); use a common `FakeMailbox` fixture
from `conftest.py`.

### `tests/adapters/test_shared_components.py` (1775 lines)

This file tests utilities shared across adapters (tool serialization, message
formatting). Split by component: `test_tool_messages.py` (serialization/
deserialization), `test_tool_utilities.py` (argument parsing, choice handling),
and `test_response_parser.py` (JSON schema building); these smaller files will
be easier to locate when modifying specific shared utilities.

### `tests/evals/test_loop.py` (1728 lines)

Evaluation loop tests cover orchestration, scoring, and result aggregation.
Split into `test_eval_orchestration.py` (loop execution, parallelism),
`test_eval_scoring.py` (metric calculation, aggregation), and
`test_eval_results.py` (output formatting, persistence); this mirrors the
evaluation module structure.

### `tests/cli/test_wink_debug_app.py` (1689 lines)

Debug CLI tests span multiple subcommands. Split by command:
`test_debug_bundle_cmd.py`, `test_debug_query_cmd.py`, `test_debug_export_cmd.py`;
share CLI runner fixtures through `tests/cli/conftest.py`.

### `tests/adapters/claude_agent_sdk/test_isolation.py` (1667 lines)

Isolation tests cover config parsing, environment detection, and ephemeral home
creation. Split into `test_isolation_config.py` (dataclass validation),
`test_isolation_auth.py` (credential detection), and `test_isolation_home.py`
(temp directory, settings.json generation).

### `tests/runtime/test_lifecycle.py` (1620 lines)

Lifecycle tests cover session creation, event dispatch, and shutdown. Organize
by phase: `test_lifecycle_init.py` (session creation, resource binding),
`test_lifecycle_events.py` (dispatch, reducers), and `test_lifecycle_shutdown.py`
(cleanup, error handling).

### `tests/prompts/test_task_examples.py` (1556 lines)

Task example tests exercise various prompt templates. Split by prompt category
or keep as integration tests in a dedicated `integration/` subdirectory; consider
using pytest marks to separate fast unit tests from slower template rendering tests.

### `tests/tools/test_vfs_tools.py` (1534 lines)

VFS tool tests cover read, write, glob, and permission handling. Split by
operation: `test_vfs_read.py`, `test_vfs_write.py`, `test_vfs_glob.py`,
`test_vfs_permissions.py`; share the `FakeFilesystem` fixture through
`tests/tools/conftest.py`.

### `tests/adapters/claude_agent_sdk/test_hooks.py` (1472 lines)

Hook tests span all nine hook types. Split by hook category (mirroring source):
`test_tool_hooks.py` (pre/post tool use), `test_lifecycle_hooks.py` (stop,
subagent), and `test_context_hooks.py` (user prompt, compact).

### `tests/adapters/claude_agent_sdk/test_bridge.py` (1358 lines)

Bridge tests cover MCP tool registration and invocation. Split into
`test_bridge_registration.py` (tool discovery, schema generation) and
`test_bridge_invocation.py` (call routing, result handling).

### `tests/runtime/test_mailbox.py` (1308 lines)

Mailbox tests cover the abstract protocol and various implementations. Split
into `test_mailbox_protocol.py` (interface compliance), `test_mailbox_memory.py`
(in-memory implementation), and `test_mailbox_redis.py` (Redis-specific behavior);
use pytest marks to skip Redis tests when no server is available.

### `tests/prompts/overrides/test_local_prompt_overrides_store.py` (1295 lines)

Override store tests cover loading, caching, and merging. Split by operation:
`test_overrides_loading.py`, `test_overrides_caching.py`, and
`test_overrides_merging.py`.

### `tests/resources/test_resources.py` (1228 lines)

Resource tests cover binding, scoping, and lifecycle. Split by concern:
`test_resource_binding.py` (registration, resolution), `test_resource_scopes.py`
(singleton, prototype, tool-call), and `test_resource_lifecycle.py` (PostConstruct,
Closeable).

### `tests/tools/test_podman_filesystem.py` (1187 lines)

Podman filesystem tests are already focused but large. Consider splitting by
operation type: `test_podman_fs_read.py`, `test_podman_fs_write.py`,
`test_podman_fs_glob.py`; alternatively, use pytest marks to enable running
filesystem tests in isolation from shell tests.

### `tests/debug/test_environment.py` (1166 lines)

Environment tests cover detection and configuration. Split into
`test_env_detection.py` (platform, runtime detection) and `test_env_config.py`
(variable handling, defaults).

### `tests/tools/test_asteval_tool.py` (1122 lines)

Asteval tests cover validation, execution, and result handling. Split by phase:
`test_asteval_validation.py` (AST checks, forbidden patterns),
`test_asteval_execution.py` (sandbox, namespace), and `test_asteval_results.py`
(serialization, truncation).

### `tests/prompts/test_progressive_disclosure.py` (1029 lines)

Progressive disclosure tests cover section visibility and rendering. Split by
concern: `test_disclosure_visibility.py` (condition evaluation) and
`test_disclosure_rendering.py` (section ordering, merging).

### `tests/runtime/test_dlq.py` (1023 lines)

DLQ tests cover routing, retention, and replay. Split by operation:
`test_dlq_routing.py` (failure classification), `test_dlq_retention.py`
(storage, expiry), and `test_dlq_replay.py` (reprocessing).

### `tests/serde/test_loop_serde.py` (1011 lines)

Loop serde tests cover event serialization for debugging. Split by event type
or combine with the main serde tests if the patterns are similar.

---

## Summary

The refactoring recommendations above follow these principles:

1. **Single Responsibility**: Each module should have one reason to change
2. **Protocol-Based Composition**: Define narrow protocols, compose implementations
3. **Test Mirroring**: Test file structure should mirror source file structure
4. **Shared Fixtures**: Extract common test utilities into `conftest.py` files
5. **Incremental Extraction**: Start with the most cohesive piece, extract one module at a time

Priority order for source files (by impact × complexity ratio):
1. `_hooks.py` - High duplication, clear groupings
2. `isolation.py` - Clear separation points, enables testing
3. `parse.py` - Logical type categories, additive changes
4. `bundle.py` - Reader/writer split is obvious
5. `query.py` - Largest file, multiple clear components

Priority order for test files:
1. `test_wink_query.py` - Largest, clear categories
2. `test_dataclass_serde.py` - Mirrors source split recommendation
3. Adapter tests - Share refactoring pattern across adapters

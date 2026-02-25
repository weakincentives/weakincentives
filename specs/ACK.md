# Adapter Compatibility Kit (ACK) Specification

## Purpose

The **Adapter Compatibility Kit** (ACK) is a unified suite of integration tests
that validates any `ProviderAdapter` implementation against the behavioral
contract defined by the WINK framework. A passing ACK run certifies that an
adapter correctly implements prompt evaluation, tool bridging, event emission,
transcript logging, error handling, and transactional semantics — regardless of
the underlying provider.

Today, integration tests are duplicated per adapter. The Claude SDK and Codex
App Server each have their own copies of greeting tests, tool invocation tests,
progressive disclosure tests, transactional rollback tests, and eval tests.
These copies drift over time: one adapter gains a new scenario that the other
never receives. ACK replaces this with a single, parameterized test suite that
every adapter must pass.

**Implementation:**

- ACK suite: `integration-tests/ack/`
- Adapter fixtures: `integration-tests/ack/adapters/`
- Shared scenarios: `integration-tests/ack/scenarios/`

## Principles

- **One suite, many adapters.** The same test scenarios execute against every
  adapter. Adapter-specific setup lives in fixture modules, not in tests.
- **Contract-first.** Tests verify the `ProviderAdapter` contract (events,
  responses, errors, transcripts) — not provider-specific internals.
- **Transcript as oracle.** Transcript entries provide a chronological,
  adapter-agnostic record of what happened. ACK asserts against transcript
  structure, entry types, and ordering — the strongest evidence that an adapter
  works correctly end-to-end.
- **Clean break.** ACK replaces all per-adapter integration test files in a
  single step. No coexistence period, no gradual porting.
- **Capability gating over phasing.** New or immature adapters declare
  conservative capabilities; tests they cannot pass yet are skipped, not
  missing. This replaces incremental migration phases with a static suite
  where each adapter simply declares what it supports.
- **Skip, don't fail.** When an adapter's provider is unavailable (no API key,
  CLI not on PATH), the entire adapter is skipped — never a spurious failure.

## Relationship to Existing Tests

### Current State

Integration tests live in `integration-tests/` as flat files:

| Test File | Adapter | Scenario |
|-----------|---------|----------|
| `test_claude_agent_sdk_adapter_integration.py` | Claude SDK | Greeting, tool invocation, structured output, events, deadlines, isolation |
| `test_codex_app_server_adapter_integration.py` | Codex | Greeting, tool invocation, structured output, events, workspace |
| `test_progressive_disclosure_integration.py` | Claude SDK | Two-level hierarchy expansion, leaf tool access |
| `test_codex_progressive_disclosure_integration.py` | Codex | Same scenarios, copy-pasted with adapter swap |
| `test_transactional_tools_integration.py` | Claude SDK | Rollback on failure, sequential operations, debug bundle verification |
| `test_codex_transactional_tools_integration.py` | Codex | Same scenarios, copy-pasted with adapter swap |
| `test_evals_math_integration.py` | Claude SDK | Math evaluation benchmarks |
| `test_codex_evals_math_integration.py` | Codex | Same evals, different adapter |
| `test_codex_isolation_integration.py` | Codex | Workspace isolation, host mounts, env forwarding |
| `test_codex_sandbox_integration.py` | Codex | Sandbox policy verification |
| `test_codex_network_policy_integration.py` | Codex | Network restrictions |

**Problems:** Duplication, coverage gaps, drift when new scenarios are added
to one adapter but not the other, and no transcript validation.

### Migration

ACK replaces all per-adapter integration test files in a single changeset.
Every scenario is ported to ACK or absorbed into adapter-specific scenario files.
The old flat files are deleted (see [What Gets Deleted](#what-gets-deleted)).

## Architecture

### Directory Layout

```
integration-tests/
├── conftest.py                         # Shared hooks (unchanged)
├── test_redis_mailbox_integration.py   # Infrastructure (not adapter-specific)
├── test_package_structure.py           # Module structure verification
├── ack/
│   ├── conftest.py                     # ACK-wide fixtures, adapter parameterization
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── _protocol.py                # AdapterFixture protocol
│   │   ├── claude_agent_sdk.py         # Claude SDK fixture
│   │   ├── codex_app_server.py         # Codex fixture
│   │   ├── acp.py                      # Generic ACP fixture
│   │   └── opencode_acp.py            # OpenCode ACP fixture
│   └── scenarios/
│       ├── __init__.py                 # Shared prompt builders
│       ├── _transcript_helpers.py      # Transcript assertion helpers
│       ├── _event_helpers.py           # Event assertion helpers
│       ├── test_basic_evaluation.py    # Tier 1: text response, prompt name
│       ├── test_tool_invocation.py     # Tier 1: bridged tool execution
│       ├── test_structured_output.py   # Tier 1: typed output parsing
│       ├── test_event_emission.py      # Tier 2: PromptRendered, ToolInvoked, PromptExecuted
│       ├── test_transcript.py          # Tier 2: transcript entry types, ordering, envelope
│       ├── test_progressive_disclosure.py  # Tier 3: section expansion, leaf tool access
│       ├── test_transactional_tools.py # Tier 3: rollback on failure, session + filesystem
│       ├── test_error_handling.py      # Tier 3: deadline expiry, budget exhaustion, invalid input
│       ├── test_tool_policies.py      # Tier 4: tool policy enforcement
│       ├── test_feedback_providers.py # Tier 4: feedback provider delivery
│       ├── test_task_completion.py    # Tier 4: task completion checking
│       ├── test_native_tools.py        # Adapter-specific: native tool ToolInvoked events
│       ├── test_workspace_isolation.py # Adapter-specific: host mounts, env forwarding
│       ├── test_sandbox_policy.py      # Adapter-specific: sandbox enforcement
│       ├── test_network_policy.py      # Adapter-specific: network restrictions
│       └── test_skill_installation.py  # Adapter-specific: skill mounting
```

### AdapterFixture Protocol

Every adapter provides a fixture module implementing the `AdapterFixture`
protocol at `integration-tests/ack/adapters/_protocol.py`. This is the only
adapter-specific code in ACK. Each fixture implements:

- `adapter_name` — canonical name (e.g., `"claude_agent_sdk"`)
- `capabilities: AdapterCapabilities` — declared capability set
- `is_available()` — checks credentials/CLI presence
- `create_adapter(tmp_path)` — creates a configured adapter instance
- `create_adapter_with_sandbox(tmp_path, *, sandbox_mode)` — adapter with sandbox
- `create_adapter_with_env(tmp_path, *, env)` — adapter with custom env vars
- `create_session()` — session configured for integration testing
- `get_model()` — model name (respects env var overrides)

Fixture implementations at `integration-tests/ack/adapters/claude_agent_sdk.py`,
`codex_app_server.py`, `acp.py`, and `opencode_acp.py`.

### AdapterCapabilities

Declared by each fixture; drives capability gating for all tests:

| Capability | Tier | Description |
|-----------|------|-------------|
| `text_response` | 1 | Basic text evaluation |
| `tool_invocation` | 1 | Bridged tool execution |
| `structured_output` | 1 | Typed output parsing |
| `event_emission` | 2 | PromptRendered/ToolInvoked/PromptExecuted events |
| `transcript` | 2 | Transcript entry types, ordering, envelope |
| `rendered_tools_event` | 2 | RenderedTools event emission |
| `progressive_disclosure` | 3 | Section expansion, leaf tool access |
| `transactional_tools` | 3 | Rollback on failure |
| `deadline_enforcement` | 3 | Deadline expiry behavior |
| `budget_enforcement` | 3 | Budget exhaustion behavior |
| `tool_policies` | 4 | Tool policy enforcement |
| `feedback_providers` | 4 | Feedback provider delivery |
| `task_completion` | 4 | Task completion checking |
| `native_tools` | Adapter-specific | Native tool ToolInvoked events |
| `workspace_isolation` | Adapter-specific | Host mounts, env forwarding |
| `network_policy` | Adapter-specific | Network restrictions |
| `sandbox_policy` | Adapter-specific | Sandbox enforcement |
| `skill_installation` | Adapter-specific | Skill mounting |

All capabilities default to `True` for Tiers 1-3 and `False` for Tier 4 and
adapter-specific scenarios, so new adapters can start conservative and expand.

### Parameterization

`integration-tests/ack/conftest.py` discovers all adapter fixtures, filters by
availability, and provides them as pytest parameters so every scenario file runs
once per available adapter. Tests declare required capabilities via
`@pytest.mark.ack_capability(capability)` markers; tests are skipped when the
current adapter does not support the named capability.

## Test Tiers

### Tier 1: Basic Evaluation

The minimum bar for any adapter. Verifies prompt evaluation, bridged tool
invocation, and structured output parsing.

```
test_returns_text_response
    Given a simple greeting prompt
    When adapter.evaluate() is called
    Then response.text is non-empty

test_bridged_tool_is_called
    Given a prompt with a single uppercase_text tool
    And the prompt instructs the model to call the tool
    When adapter.evaluate() is called
    Then at least one ToolInvoked event is emitted for "uppercase_text"

test_structured_output_parsing
    Given a PromptTemplate[ReviewAnalysis] with summary and sentiment fields
    When adapter.evaluate() is called
    Then response.output is a ReviewAnalysis instance
```

### Tier 2: Observability

Verifies that adapters emit correct telemetry events and transcript entries.

```
test_prompt_rendered_event
    Given any prompt
    When adapter.evaluate() is called
    Then exactly one PromptRendered event is emitted
    And event.prompt_name matches the prompt

test_transcript_contains_user_message
    Given a prompt with known text
    When adapter.evaluate() is called
    Then at least one transcript entry has entry_type == "user_message"

test_transcript_tool_use_before_tool_result
    Given a prompt that invokes a tool
    When adapter.evaluate() is called
    Then a "tool_use" entry appears before the corresponding "tool_result" entry

test_transcript_envelope_completeness
    For every transcript entry emitted during evaluation:
    Then entry has prompt_name, adapter, entry_type, sequence_number, source, timestamp

test_transcript_sequence_monotonicity
    Given all transcript entries for a single source
    Then sequence_numbers are strictly increasing with no gaps
```

### Tier 3: Advanced Behavior

Exercises transactional semantics, multi-turn loops, and resource enforcement.

```
test_tool_failure_rolls_back_filesystem
    Given a workspace with a write_and_fail tool
    When the tool writes a file then returns failure
    Then the file does not exist after evaluation

test_tool_failure_rolls_back_session_state
    Given a session with a ToolOperationLog slice
    When write_and_fail dispatches an event then fails
    Then the operation is not present in the session

test_deadline_enforcement
    Given a prompt with a deadline in the near past
    When adapter.evaluate() is called
    Then a PromptEvaluationError is raised
```

### Adapter-Specific Scenarios

Tests for capabilities only some adapters support. Skipped when the capability
flag is `False`.

```
test_sandbox_restricts_writes_outside_workspace  [requires: sandbox_policy]
test_network_denied_by_default                   [requires: network_policy]
test_workspace_section_mounts_host_files         [requires: workspace_isolation]
test_skill_knowledge_available                   [requires: skill_installation]
test_native_tool_emits_tool_invoked             [requires: native_tools]
```

## Shared Helpers

- **Prompt builders** at `integration-tests/ack/scenarios/__init__.py`: shared
  factories for greeting, tool, structured output, progressive disclosure, and
  transactional prompts.
- **Transcript helpers** at `integration-tests/ack/scenarios/_transcript_helpers.py`:
  `collect_transcript_entries`, `assert_envelope_complete`,
  `assert_sequence_monotonic`, `assert_entry_order`.
- **Event helpers** at `integration-tests/ack/scenarios/_event_helpers.py`:
  `capture_events`, `assert_prompt_rendered`, `assert_tool_invoked`.

## Transcript Validation Strategy

Transcript checks are the most valuable part of ACK. They verify the
adapter-agnostic log record that downstream tools (debug bundles, `wink query`,
log viewers) depend on.

### What ACK Validates

| Property | Assertion | Why |
|----------|-----------|-----|
| Envelope completeness | All required keys present | Consumers depend on schema |
| Canonical entry types | `entry_type` in known set | Forward compatibility via `unknown` |
| Sequence monotonicity | Strictly increasing per source | Ordering guarantees |
| Adapter label | `adapter` matches fixture name | Attribution correctness |
| Causal ordering | `tool_use` before `tool_result` | Conversation coherence |
| Start/stop bracketing | `transcript.start` before entries, `transcript.stop` after | Lifecycle completeness |
| Bundle inclusion | `transcript.jsonl` in debug bundle | Offline debugging |
| Summary statistics | `transcript.stop` contains entry counts | Operational visibility |

### What ACK Does NOT Validate

- **Content correctness** of model responses (that's evals, not ACK).
- **Exact transcript length** (adapters may emit different numbers of entries).
- **`detail` payload structure** (adapter-specific, opaque to ACK).
- **Subagent transcript entries** (only Claude SDK produces these).

## Running ACK

```bash
# Run ACK for all available adapters
uv run pytest integration-tests/ack/ -v --timeout=120

# Run ACK for a specific adapter
uv run pytest integration-tests/ack/ -v -k "claude_agent_sdk"

# Run a specific tier
uv run pytest integration-tests/ack/scenarios/test_basic_evaluation.py -v
```

ACK tests are marked `@pytest.mark.integration` and excluded from `make check`
(unit tests only). CI runs ACK as a separate job per adapter matrix with
provider credentials injected.

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `CLAUDE_AGENT_SDK_TEST_MODEL` | Claude SDK model override | `get_default_model()` (Opus 4.6) |
| `CODEX_APP_SERVER_TEST_MODEL` | Codex model override | `gpt-5.3-codex` |
| `OPENCODE_ACP_TEST_MODEL` | OpenCode model override | `gpt-5.3` |
| `ACK_TIMEOUT` | Per-test timeout seconds | `120` |

## Invariants

1. **Fixture isolation.** Each test receives a fresh `tmp_path` and session.
   No state leaks between tests or between adapter parameterizations.

1. **Skip semantics.** An unavailable adapter produces `SKIPPED`, never
   `FAILED`. A capability the adapter does not declare produces `SKIPPED`.

1. **Adapter-agnostic assertions.** No test in `scenarios/` imports from
   any specific adapter package. Tests use only `ProviderAdapter`, session
   events, and transcript entries.

1. **Prompt determinism.** ACK prompts use explicit instructions, constrained
   output schemas, and short expected responses to minimize non-determinism.

1. **Transcript contract.** Every adapter that declares
   `capabilities.transcript = True` must emit transcript entries conforming to
   `specs/TRANSCRIPT.md`. ACK enforces the common envelope, canonical types,
   sequence ordering, and lifecycle events.

1. **Event contract.** Every adapter must emit `PromptRendered` before
   evaluation, `ToolInvoked` for each tool call, and `PromptExecuted` after
   completion.

1. **Transactional contract.** Adapters that declare
   `capabilities.transactional_tools = True` must restore session and
   filesystem state to the pre-call snapshot when a tool returns failure.

## Adding a New Adapter to ACK

1. Create a fixture module in `integration-tests/ack/adapters/` implementing
   `AdapterFixture`. Declare capabilities honestly — set `False` for anything
   the adapter does not yet support.

1. Register the fixture in `integration-tests/ack/conftest.py` by adding it
   to `_ALL_FIXTURES`.

1. Run the full suite. Fix failures — these are almost always real adapter bugs.

1. Expand capabilities over time by flipping flags from `False` to `True`.

## What Gets Deleted

| Deleted | Replaced By |
|---------|-------------|
| `test_claude_agent_sdk_adapter_integration.py` | `ack/scenarios/test_basic_evaluation.py` + `test_tool_invocation.py` + `test_structured_output.py` + `test_event_emission.py` + `test_transcript.py` |
| `test_codex_app_server_adapter_integration.py` | Same ACK scenarios |
| `test_progressive_disclosure_integration.py` | `ack/scenarios/test_progressive_disclosure.py` |
| `test_codex_progressive_disclosure_integration.py` | Same ACK scenario |
| `test_transactional_tools_integration.py` | `ack/scenarios/test_transactional_tools.py` |
| `test_codex_transactional_tools_integration.py` | Same ACK scenario |
| `test_codex_isolation_integration.py` | `ack/scenarios/test_workspace_isolation.py` |
| `test_codex_sandbox_integration.py` | `ack/scenarios/test_sandbox_policy.py` |
| `test_codex_network_policy_integration.py` | `ack/scenarios/test_network_policy.py` |
| `test_evals_math_integration.py` | `ack/scenarios/test_basic_evaluation.py` |
| `test_codex_evals_math_integration.py` | Same ACK scenario |

**Not deleted:** `test_redis_mailbox_integration.py`, `test_package_structure.py`,
`conftest.py`, `redis_utils.py` — infrastructure and shared utilities.

## Related Specifications

- `specs/ADAPTERS.md` — Provider adapter protocol and contract
- `specs/CLAUDE_AGENT_SDK.md` — Claude SDK adapter
- `specs/CODEX_APP_SERVER.md` — Codex App Server adapter
- `specs/ACP_ADAPTER.md` — Generic ACP adapter
- `specs/OPENCODE_ACP_ADAPTER.md` — OpenCode ACP adapter
- `specs/TRANSCRIPT.md` — Transcript entry schema and emission
- `specs/TOOLS.md` — Tool specification and transactional semantics
- `specs/SESSIONS.md` — Session events and snapshots
- `specs/TESTING.md` — Testing standards
- `specs/DEBUG_BUNDLE.md` — Debug bundle format

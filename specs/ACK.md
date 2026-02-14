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
- **Incremental adoption.** Adapters opt in to ACK tiers. A new adapter can
  pass Tier 1 (basic evaluation) before tackling Tier 3 (transactional tools).
  Existing per-adapter tests remain until their ACK equivalent is proven stable.
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

**Problems:**

1. **Duplication.** Progressive disclosure, transactional tools, and evals are
   nearly identical between Claude and Codex — differing only in adapter
   construction and skip conditions.
1. **Coverage gaps.** Claude SDK has isolation tests that Codex lacks (and vice
   versa). No ACP/OpenCode integration tests exist at all.
1. **Drift.** When a new scenario is added to one adapter's tests, the other
   adapter often doesn't receive it.
1. **No transcript validation.** Existing tests check events
   (`PromptRendered`, `ToolInvoked`, `PromptExecuted`) but not transcript
   entries, missing an entire dimension of adapter correctness.

### Migration Path

ACK does not replace everything at once. The migration proceeds in phases:

1. **Phase 1: ACK scaffold + Tier 1 tests.** Create the fixture/scenario
   structure. Port greeting, tool invocation, and structured output tests.
   Existing per-adapter tests remain untouched.

1. **Phase 2: Tier 2 + Tier 3 tests.** Port event emission, progressive
   disclosure, and transactional tool tests. Add transcript validation.
   Per-adapter tests that have ACK equivalents are deleted.

1. **Phase 3: Adapter-specific tiers.** Consolidate isolation, sandbox, and
   network policy tests into adapter-specific ACK tiers. Delete remaining
   per-adapter files.

1. **Phase 4: ACP/OpenCode onboarding.** New adapters implement ACK fixtures
   and run the full suite from day one.

At completion, `integration-tests/` contains only:

- `ack/` — the unified suite
- `test_redis_mailbox_integration.py` — non-adapter infrastructure test
- `test_package_structure.py` — module structure verification
- `conftest.py` — shared configuration

## Architecture

### Directory Layout

```
integration-tests/
├── conftest.py                         # Shared hooks (unchanged)
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
│       ├── __init__.py
│       ├── test_basic_evaluation.py    # Tier 1: text response, prompt name
│       ├── test_tool_invocation.py     # Tier 1: bridged tool execution
│       ├── test_structured_output.py   # Tier 1: typed output parsing
│       ├── test_event_emission.py      # Tier 2: PromptRendered, ToolInvoked, PromptExecuted
│       ├── test_transcript.py          # Tier 2: transcript entry types, ordering, envelope
│       ├── test_progressive_disclosure.py  # Tier 3: section expansion, leaf tool access
│       ├── test_transactional_tools.py # Tier 3: rollback on failure, session + filesystem
│       └── test_error_handling.py      # Tier 3: deadline expiry, budget exhaustion, invalid input
```

### AdapterFixture Protocol

Every adapter provides a fixture module that implements the `AdapterFixture`
protocol. This is the only adapter-specific code in ACK.

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from weakincentives.adapters.core import ProviderAdapter
from weakincentives.runtime.session import Session


@dataclass(slots=True, frozen=True)
class AdapterCapabilities:
    """Declares which ACK tiers this adapter supports.

    Adapters that do not support a capability are skipped for tests
    that require it. This enables incremental onboarding.
    """

    # Tier 1: Basic evaluation
    text_response: bool = True
    tool_invocation: bool = True
    structured_output: bool = True

    # Tier 2: Observability
    event_emission: bool = True
    transcript: bool = True

    # Tier 3: Advanced
    progressive_disclosure: bool = True
    transactional_tools: bool = True
    deadline_enforcement: bool = True
    budget_enforcement: bool = True

    # Adapter-specific
    native_tools: bool = False      # Adapter has its own tools (Codex commands, Claude Bash/Read)
    workspace_isolation: bool = False
    network_policy: bool = False
    sandbox_policy: bool = False


@runtime_checkable
class AdapterFixture(Protocol):
    """Protocol that each adapter's fixture module must implement."""

    @property
    def adapter_name(self) -> str:
        """Canonical adapter name (e.g., 'claude_agent_sdk', 'codex_app_server')."""
        ...

    @property
    def capabilities(self) -> AdapterCapabilities:
        """Declare which ACK tiers this adapter supports."""
        ...

    def is_available(self) -> bool:
        """Check if the adapter's provider is available (credentials, CLI, etc.).

        Returns False to skip all tests for this adapter.
        """
        ...

    def create_adapter(self, tmp_path: Path) -> ProviderAdapter:
        """Create a configured adapter instance for testing.

        The adapter MUST use tmp_path as its working directory to
        prevent side effects on the host filesystem.
        """
        ...

    def create_session(self) -> Session:
        """Create a session configured for integration testing."""
        ...

    def get_model(self) -> str:
        """Return the model name to use for tests.

        Should respect environment variable overrides for CI flexibility.
        """
        ...
```

### Parameterization

The ACK `conftest.py` discovers all adapter fixtures, filters by availability,
and provides them as pytest parameters:

```python
import pytest

from .adapters.claude_agent_sdk import ClaudeAgentSDKFixture
from .adapters.codex_app_server import CodexAppServerFixture
from .adapters.acp import ACPFixture
from .adapters.opencode_acp import OpenCodeACPFixture

_ALL_FIXTURES = [
    ClaudeAgentSDKFixture(),
    CodexAppServerFixture(),
    ACPFixture(),
    OpenCodeACPFixture(),
]


def _available_fixtures() -> list[AdapterFixture]:
    return [f for f in _ALL_FIXTURES if f.is_available()]


@pytest.fixture(params=_available_fixtures(), ids=lambda f: f.adapter_name)
def adapter_fixture(request: pytest.FixtureRequest) -> AdapterFixture:
    """Parameterized fixture providing each available adapter."""
    return request.param


@pytest.fixture
def adapter(adapter_fixture: AdapterFixture, tmp_path: Path) -> ProviderAdapter:
    """Create an adapter instance from the current fixture."""
    return adapter_fixture.create_adapter(tmp_path)


@pytest.fixture
def session(adapter_fixture: AdapterFixture) -> Session:
    """Create a session from the current fixture."""
    return adapter_fixture.create_session()
```

### Capability Gating

Tests declare required capabilities via markers. Tests are skipped when the
current adapter does not support the required capability:

```python
import pytest

def requires_capability(capability: str):
    """Skip test if the adapter does not support the named capability."""
    return pytest.mark.ack_capability(capability)


# In conftest.py:
def pytest_runtest_setup(item: pytest.Item) -> None:
    for marker in item.iter_markers("ack_capability"):
        capability = marker.args[0]
        fixture = item.funcargs.get("adapter_fixture")
        if fixture and not getattr(fixture.capabilities, capability, False):
            pytest.skip(
                f"{fixture.adapter_name} does not support {capability}"
            )
```

## Test Tiers

### Tier 1: Basic Evaluation

The minimum bar for any adapter. These tests verify that the adapter can
evaluate prompts, invoke bridged tools, and parse structured output.

#### `test_basic_evaluation.py`

```
test_returns_text_response
    Given a simple greeting prompt
    When adapter.evaluate() is called
    Then response.text is non-empty
    And response.prompt_name matches the prompt

test_prompt_name_propagation
    Given a prompt with name "ack_greeting"
    When adapter.evaluate() is called
    Then response.prompt_name == "ack_greeting"
```

#### `test_tool_invocation.py`

```
test_bridged_tool_is_called
    Given a prompt with a single uppercase_text tool
    And the prompt instructs the model to call the tool
    When adapter.evaluate() is called
    Then at least one ToolInvoked event is emitted for "uppercase_text"

test_tool_result_is_correct
    Given a prompt instructing uppercase_text(text="hello")
    When the tool is invoked
    Then the tool receives params.text == "hello"
    And the tool returns TransformResult(text="HELLO")
```

#### `test_structured_output.py`

```
test_structured_output_parsing
    Given a PromptTemplate[ReviewAnalysis] with summary and sentiment fields
    When adapter.evaluate() is called
    Then response.output is a ReviewAnalysis instance
    And response.output.summary is non-empty
    And response.output.sentiment is non-empty

test_structured_output_type_fidelity
    Given a PromptTemplate[T] for a frozen dataclass T
    When adapter.evaluate() is called
    Then type(response.output) is T
```

### Tier 2: Observability

These tests verify that adapters emit the correct telemetry events and
transcript entries, enabling debugging and auditing.

#### `test_event_emission.py`

```
test_prompt_rendered_event
    Given any prompt
    When adapter.evaluate() is called
    Then exactly one PromptRendered event is emitted
    And event.prompt_name matches the prompt
    And event.adapter matches adapter_fixture.adapter_name
    And event.rendered_prompt contains the prompt text

test_prompt_executed_event
    Given any prompt
    When adapter.evaluate() is called
    Then exactly one PromptExecuted event is emitted
    And event.prompt_name matches the prompt
    And event.adapter matches adapter_fixture.adapter_name

test_tool_invoked_event
    Given a prompt with a tool
    When the tool is called by the model
    Then a ToolInvoked event is emitted
    And event.name == tool_name
    And event.adapter matches adapter_fixture.adapter_name
    And event.params contains the tool input
    And event.result contains the tool output

test_rendered_tools_event
    Given a prompt with tools
    When adapter.evaluate() is called
    Then a RenderedTools event is emitted
    And event.tools contains schemas for all declared tools
    And event.render_event_id correlates with the PromptRendered event
```

#### `test_transcript.py`

Transcript tests are the distinguishing feature of ACK. They verify that
adapters produce a correct chronological record via the `TranscriptEmitter`
system defined in `specs/TRANSCRIPT.md`.

```
test_transcript_contains_user_message
    Given a prompt with known text
    When adapter.evaluate() is called
    Then at least one transcript entry has entry_type == "user_message"
    And the entry's detail or raw contains the prompt text

test_transcript_contains_assistant_message
    Given a prompt that produces a text response
    When adapter.evaluate() is called
    Then at least one transcript entry has entry_type == "assistant_message"

test_transcript_tool_use_before_tool_result
    Given a prompt that invokes a tool
    When adapter.evaluate() is called
    Then a "tool_use" entry appears before the corresponding "tool_result" entry
    And both share the same source

test_transcript_envelope_completeness
    For every transcript entry emitted during evaluation:
    Then entry has prompt_name (non-empty string)
    And entry has adapter (matches adapter_fixture.adapter_name)
    And entry has entry_type (one of the canonical types)
    And entry has sequence_number (non-negative integer)
    And entry has source (non-empty string)
    And entry has timestamp (valid ISO-8601)

test_transcript_sequence_monotonicity
    Given all transcript entries for a single source
    Then sequence_numbers are strictly increasing with no gaps

test_transcript_canonical_types_only
    For every transcript entry:
    Then entry_type is one of: user_message, assistant_message, tool_use,
         tool_result, thinking, system_event, token_usage, error, unknown

test_transcript_adapter_label
    For every transcript entry:
    Then entry.adapter == adapter_fixture.adapter_name

test_transcript_start_stop_events
    When adapter.evaluate() is called
    Then a transcript.start log event is emitted before any transcript.entry
    And a transcript.stop log event is emitted after all transcript.entry events
    And transcript.stop contains summary statistics (entry counts)

test_transcript_in_debug_bundle
    Given a debug bundle captured during evaluation
    Then bundle contains transcript.jsonl
    And transcript.jsonl contains exactly the transcript entries from the evaluation
    And entries are ordered by sequence_number
```

### Tier 3: Advanced Behavior

These tests exercise complex adapter interactions that stress transactional
semantics, multi-turn loops, and resource enforcement.

#### `test_progressive_disclosure.py`

```
test_two_level_hierarchy_expansion
    Given a prompt with a two-level summarized section hierarchy
    And a leaf section containing a verify_result tool
    When adapter.evaluate() is called in an expansion loop
    Then at least one VisibilityExpansionRequired is raised
    And the model eventually calls the leaf tool
    And session VisibilityOverrides contain FULL for both parent and child

test_direct_leaf_expansion
    Given the same hierarchy
    When the model requests expansion of parent and child together
    Then the framework handles it correctly
    And the leaf section's tool becomes available
```

#### `test_transactional_tools.py`

```
test_tool_failure_rolls_back_filesystem
    Given a workspace with a write_and_fail tool
    When the tool writes a file then returns failure
    Then the file does not exist after evaluation
    And any file written by a prior successful tool still exists

test_tool_failure_rolls_back_session_state
    Given a session with a ToolOperationLog slice
    When write_and_fail dispatches a RecordToolOperation then fails
    Then the operation is not present in the session
    And operations from prior successful tools are present

test_sequential_operations_isolation
    Given three sequential tool calls: success, failure, success
    Then the first and third operations persist
    And the second operation is rolled back
    And filesystem and session state are consistent

test_rollback_verified_in_debug_bundle
    Given a debug bundle captured after mixed success/failure tool calls
    Then bundle filesystem matches expected state
    And bundle session snapshot matches expected state
```

#### `test_error_handling.py`

```
test_deadline_enforcement
    Given a prompt with a deadline in the near past
    When adapter.evaluate() is called
    Then a PromptEvaluationError is raised
    And error.phase == "request" or "budget"

test_invalid_tool_params_returns_error
    Given a tool that expects {text: str}
    When the model sends {text: 123}
    Then the tool returns an error result (not an exception)
    And evaluation continues (adapter does not abort)
```

## Shared Scenarios

To avoid duplication within ACK itself, test scenarios are built from shared
prompt builders and assertion helpers.

### Prompt Builders

```python
# integration-tests/ack/scenarios/__init__.py

def build_greeting_prompt(ns: str) -> PromptTemplate[object]:
    """Build a simple greeting prompt for basic evaluation tests."""
    ...

def build_tool_prompt(ns: str) -> tuple[PromptTemplate[object], Tool]:
    """Build a prompt with an uppercase_text tool."""
    ...

def build_structured_prompt(ns: str) -> PromptTemplate[ReviewAnalysis]:
    """Build a prompt expecting structured ReviewAnalysis output."""
    ...

def build_progressive_disclosure_prompt(ns: str) -> tuple[PromptTemplate[object], Tool]:
    """Build a two-level hierarchy prompt with a verify_result leaf tool."""
    ...

def build_transactional_prompt(ns: str) -> tuple[PromptTemplate[object], Tool, Tool, type]:
    """Build a prompt with write_and_succeed and write_and_fail tools."""
    ...
```

### Transcript Assertion Helpers

```python
# integration-tests/ack/scenarios/_transcript_helpers.py

def collect_transcript_entries(caplog) -> list[dict]:
    """Extract transcript entries from pytest log capture."""
    ...

def assert_envelope_complete(entry: dict) -> None:
    """Assert all required envelope keys are present and valid."""
    ...

def assert_sequence_monotonic(entries: list[dict], source: str) -> None:
    """Assert sequence numbers are strictly increasing for a source."""
    ...

def assert_entry_order(entries: list[dict], *expected_types: str) -> None:
    """Assert that entry_types appear in the given order (subsequence match)."""
    ...

def assert_tool_use_before_result(entries: list[dict], tool_name: str) -> None:
    """Assert tool_use entry precedes tool_result for the named tool."""
    ...
```

### Event Assertion Helpers

```python
# integration-tests/ack/scenarios/_event_helpers.py

def capture_events(session: Session, *event_types: type) -> dict[type, list]:
    """Subscribe to event types and return a dict of captured events."""
    ...

def assert_prompt_rendered(events: list, adapter_name: str, prompt_name: str) -> None:
    """Assert exactly one PromptRendered with correct fields."""
    ...

def assert_prompt_executed(events: list, adapter_name: str, prompt_name: str) -> None:
    """Assert exactly one PromptExecuted with correct fields."""
    ...

def assert_tool_invoked(events: list, tool_name: str, adapter_name: str) -> None:
    """Assert at least one ToolInvoked for the named tool."""
    ...
```

## Adapter Fixture Examples

### Claude Agent SDK Fixture

```python
class ClaudeAgentSDKFixture:
    adapter_name = "claude_agent_sdk"

    capabilities = AdapterCapabilities(
        native_tools=True,
        workspace_isolation=True,
        network_policy=True,
        sandbox_policy=True,
    )

    def is_available(self) -> bool:
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError:
            return False
        return _is_bedrock_mode() or "ANTHROPIC_API_KEY" in os.environ

    def create_adapter(self, tmp_path: Path) -> ClaudeAgentSDKAdapter:
        return ClaudeAgentSDKAdapter(
            model=self.get_model(),
            client_config=ClaudeAgentSDKClientConfig(
                permission_mode="bypassPermissions",
                cwd=str(tmp_path),
            ),
        )

    def create_session(self) -> Session:
        return Session(tags={"suite": "ack"})

    def get_model(self) -> str:
        return os.environ.get("CLAUDE_AGENT_SDK_TEST_MODEL", get_default_model())
```

### Codex App Server Fixture

```python
class CodexAppServerFixture:
    adapter_name = "codex_app_server"

    capabilities = AdapterCapabilities(
        native_tools=True,
        workspace_isolation=True,
        sandbox_policy=True,
    )

    def is_available(self) -> bool:
        return shutil.which("codex") is not None

    def create_adapter(self, tmp_path: Path) -> CodexAppServerAdapter:
        return CodexAppServerAdapter(
            model_config=CodexAppServerModelConfig(model=self.get_model()),
            client_config=CodexAppServerClientConfig(
                cwd=str(tmp_path),
                approval_policy="never",
            ),
        )

    def create_session(self) -> Session:
        return Session(tags={"suite": "ack"})

    def get_model(self) -> str:
        return os.environ.get("CODEX_APP_SERVER_TEST_MODEL", "gpt-5.3-codex")
```

### ACP / OpenCode Fixture

```python
class OpenCodeACPFixture:
    adapter_name = "opencode_acp"

    capabilities = AdapterCapabilities(
        native_tools=False,
        workspace_isolation=False,
        network_policy=False,
        sandbox_policy=False,
        # OpenCode supports these but may need tuning:
        progressive_disclosure=True,
        transactional_tools=True,
    )

    def is_available(self) -> bool:
        return shutil.which("opencode") is not None

    def create_adapter(self, tmp_path: Path) -> OpenCodeACPAdapter:
        return OpenCodeACPAdapter(
            adapter_config=OpenCodeACPAdapterConfig(model=self.get_model()),
            client_config=OpenCodeACPClientConfig(cwd=str(tmp_path)),
        )

    def create_session(self) -> Session:
        return Session(tags={"suite": "ack"})

    def get_model(self) -> str:
        return os.environ.get("OPENCODE_ACP_TEST_MODEL", "gpt-5.3")
```

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
- **Exact transcript length** (adapters may emit different numbers of entries
  for the same logical operation).
- **`detail` payload structure** (adapter-specific, opaque to ACK).
- **`raw` field content** (depends on `emit_raw` configuration).
- **Subagent transcript entries** (only Claude SDK produces these; validated
  in adapter-specific tests, not in the shared ACK suite).

## Running ACK

### Local Execution

```bash
# Run ACK for all available adapters
uv run pytest integration-tests/ack/ -v --timeout=120

# Run ACK for a specific adapter
uv run pytest integration-tests/ack/ -v -k "claude_agent_sdk"

# Run a specific tier
uv run pytest integration-tests/ack/scenarios/test_basic_evaluation.py -v

# Run with transcript debug logging visible
uv run pytest integration-tests/ack/ -v --log-level=DEBUG -k "test_transcript"
```

### CI Configuration

ACK tests are marked `@pytest.mark.integration` and excluded from `make check`
(which runs unit tests only). CI pipelines run ACK as a separate job with
provider credentials available:

```yaml
ack:
  strategy:
    matrix:
      adapter: [claude_agent_sdk, codex_app_server, opencode_acp]
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  steps:
    - run: uv run pytest integration-tests/ack/ -v -k "${{ matrix.adapter }}"
```

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `CLAUDE_AGENT_SDK_TEST_MODEL` | Claude SDK model override | `get_default_model()` (Sonnet 4.5) |
| `CODEX_APP_SERVER_TEST_MODEL` | Codex model override | `gpt-5.3-codex` |
| `OPENCODE_ACP_TEST_MODEL` | OpenCode model override | `gpt-5.3` |
| `ACK_TIMEOUT` | Per-test timeout seconds | `120` |
| `ACK_ADAPTERS` | Comma-separated adapter filter | all available |

## Invariants

1. **Fixture isolation.** Each test receives a fresh `tmp_path` and session.
   No state leaks between tests or between adapter parameterizations.

1. **Skip semantics.** An unavailable adapter produces `SKIPPED`, never
   `FAILED`. A capability the adapter does not declare produces `SKIPPED`.

1. **Adapter-agnostic assertions.** No test in `scenarios/` imports from
   `weakincentives.adapters.claude_agent_sdk`, `.codex_app_server`, or any
   specific adapter package. Tests use only `ProviderAdapter`, session events,
   and transcript entries.

1. **Prompt determinism.** ACK prompts are designed to minimize model
   non-determinism. They use explicit instructions ("call this tool exactly
   once"), constrained output schemas, and short expected responses.

1. **Transcript contract.** Every adapter that declares
   `capabilities.transcript = True` must emit transcript entries conforming to
   `specs/TRANSCRIPT.md`. ACK enforces the common envelope, canonical types,
   sequence ordering, and lifecycle events.

1. **Event contract.** Every adapter must emit `PromptRendered` before
   evaluation, `ToolInvoked` for each tool call, and `PromptExecuted` after
   completion. These are verified against the `ProviderAdapter` contract in
   `specs/ADAPTERS.md`.

1. **Transactional contract.** Adapters that declare
   `capabilities.transactional_tools = True` must restore session and
   filesystem state to the pre-call snapshot when a tool returns
   `ToolResult(success=False)`.

## Adding a New Adapter to ACK

To onboard a new adapter:

1. **Create a fixture module** in `integration-tests/ack/adapters/` implementing
   `AdapterFixture`. Start with conservative capabilities (disable tiers you
   haven't tested manually).

1. **Register the fixture** in `integration-tests/ack/conftest.py` by adding it
   to `_ALL_FIXTURES`.

1. **Run Tier 1** (`test_basic_evaluation.py`, `test_tool_invocation.py`,
   `test_structured_output.py`). Fix any failures — these are almost always
   real adapter bugs, not test issues.

1. **Enable Tier 2** by setting `event_emission=True` and `transcript=True` in
   capabilities. Run `test_event_emission.py` and `test_transcript.py`.

1. **Enable Tier 3** capabilities incrementally. Each capability unlocks
   additional test scenarios.

1. **Add adapter-specific tests** (if needed) as separate files in
   `integration-tests/ack/scenarios/` with an `_<adapter>` suffix, guarded by
   the appropriate capability marker.

## What Gets Deleted (After Full Migration)

| Deleted | Replaced By |
|---------|-------------|
| `test_claude_agent_sdk_adapter_integration.py` | `ack/scenarios/test_basic_evaluation.py` + `test_tool_invocation.py` + `test_structured_output.py` + `test_event_emission.py` |
| `test_codex_app_server_adapter_integration.py` | Same ACK scenarios |
| `test_progressive_disclosure_integration.py` | `ack/scenarios/test_progressive_disclosure.py` |
| `test_codex_progressive_disclosure_integration.py` | Same ACK scenario |
| `test_transactional_tools_integration.py` | `ack/scenarios/test_transactional_tools.py` |
| `test_codex_transactional_tools_integration.py` | Same ACK scenario |
| `test_evals_math_integration.py` | Not ACK — evals remain separate |
| `test_codex_evals_math_integration.py` | Not ACK — evals remain separate |

**Not deleted:**

- `test_redis_mailbox_integration.py` — infrastructure, not adapter
- `test_package_structure.py` — module structure, not adapter
- `test_codex_isolation_integration.py` — adapter-specific, becomes ACK adapter-specific tier
- `test_codex_sandbox_integration.py` — adapter-specific
- `test_codex_network_policy_integration.py` — adapter-specific
- Eval tests — separate concern; evals have their own framework

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

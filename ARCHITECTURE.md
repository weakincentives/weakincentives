# Architecture Decision Records

This document captures key architectural decisions in WINK (Weak Incentives) and
the rationale behind them. These ADRs explain *why* the library is built the way
it is, providing context for contributors and maintainers.

## ADR-001: Protocols Over Abstract Base Classes

**Status**: Accepted

**Context**: WINK needs contracts for adapters, filesystems, optimizers, and
other extension points. Python offers two primary mechanisms: Abstract Base
Classes (ABC) with `@abstractmethod`, and typing `Protocol` classes for
structural subtyping.

**Decision**: Use `Protocol` classes as the primary abstraction mechanism.
Reserve ABCs only for extension points requiring a shared implementation base.

**Rationale**:

1. **Structural typing**: Protocols accept any object with the right
   methods/properties without explicit inheritance. Third-party classes work
   automatically:

   ```python
   # Protocol accepts any OpenAI-compatible client
   class _OpenAIProtocol(Protocol):
       responses: _ResponsesAPI

   # Works with real OpenAI client, mocks, or compatible alternatives
   adapter = OpenAIAdapter(client=my_client)
   ```

2. **Decoupling**: Implementations don't inherit from WINK types, avoiding tight
   coupling and diamond inheritance problems.

3. **Multiple protocol satisfaction**: A single class can implement multiple
   protocols without multiple inheritance complexity.

4. **Testing simplicity**: Test doubles don't need inheritance chains—any object
   satisfying the protocol works.

5. **Forward compatibility**: External libraries can satisfy protocols without
   modification or wrapper classes.

**ABC usage**: `ProviderAdapter` and `Section` remain ABCs because:

- They provide shared implementation (throttling, rendering logic)
- Extension authors benefit from base class structure
- The inheritance contract is explicit and intentional

**Locations**:

- `adapters/_provider_protocols.py` - Provider type contracts
- `runtime/session/_types.py` - Reducer and context protocols
- `runtime/events/_types.py` - Dispatcher protocol
- `filesystem/_protocol.py` - Filesystem protocol
- `optimizers/_protocol.py` - Optimizer protocol

**Consequences**:

- Cannot rely on `isinstance()` checks without `@runtime_checkable`
- Type errors surface at static analysis time rather than registration
- Contributors must understand structural typing patterns

---

## ADR-002: Redux-Style Sessions with Immutable Event Ledgers

**Status**: Accepted

**Context**: Agent orchestration requires tracking state across tool calls,
model responses, and nested sub-sessions. State must support snapshots,
rollbacks, and deterministic replay for debugging and testing.

**Decision**: Implement sessions as immutable event ledgers with pure reducer
functions, inspired by Redux architecture.

**Rationale**:

1. **Deterministic replay**: Given an initial state and event sequence, the
   final state is always reproducible:

   ```python
   # Replay produces identical state
   session.restore(snapshot)
   for event in events:
       session.dispatch(event)
   assert session[Plan].latest() == expected
   ```

2. **Safe snapshots**: Immutable tuples can be captured without deep copying:

   ```python
   snapshot = session.snapshot()
   # Subsequent dispatches don't affect snapshot
   session.dispatch(AddStep(step="new"))
   session.restore(snapshot)  # Rollback to captured state
   ```

3. **Auditability**: The event ledger provides a complete history:

   ```python
   all_events = session[ToolInvoked].all()  # Every tool call recorded
   ```

4. **Transactional tool execution**: Tool handlers can preview changes and
   rollback on failure without corrupting session state.

5. **Pure reducers enable composition**: Reducers are simple functions that
   transform tuples, making them testable in isolation:

   ```python
   @reducer(on=AddStep)
   def add_step(self, event: AddStep) -> "AgentPlan":
       return replace(self, steps=(*self.steps, event.step))
   ```

**Design**:

```
┌─────────────────────────────────────────────────────────┐
│                      Session                            │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Slice[Plan]: (Plan₁, Plan₂, Plan₃, ...)         │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Slice[ToolInvoked]: (Event₁, Event₂, ...)        │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  dispatch(event) → reducer(slice, event) → new_slice   │
└─────────────────────────────────────────────────────────┘
```

**Built-in reducers**:

| Reducer | Semantics | Use Case |
|---------|-----------|----------|
| `append_all` | Append every event | Event ledgers, audit logs |
| `replace_latest` | Keep only newest | Configuration, latest result |
| `upsert_by(key)` | Replace by key | Keyed entities |
| `replace_latest_by(key)` | Latest per key | Per-entity latest |

**Locations**:

- `runtime/session/session.py` - Session container
- `runtime/session/state_slice.py` - Typed slice storage
- `runtime/session/reducers.py` - Built-in and declarative reducers
- `runtime/events/` - Event bus and dispatch

**Consequences**:

- State grows unbounded without explicit eviction (use `replace_latest`)
- Reducers run synchronously; keep them lightweight
- Non-dataclass payloads require special handling

---

## ADR-003: Design-by-Contract for Internal Safety

**Status**: Accepted

**Context**: WINK has complex invariants around session state, prompt rendering,
and tool execution. Runtime assertions help during development but add overhead
in production.

**Decision**: Implement a zero-cost DbC framework with `@require`, `@ensure`,
`@invariant`, and `@pure` decorators that activate only during testing.

**Rationale**:

1. **Zero production overhead**: Decorators are no-ops unless
   `WEAKINCENTIVES_DBC=1` or running under pytest:

   ```python
   @require(lambda x: x > 0)  # No-op in production
   def compute(x: int) -> int:
       ...
   ```

2. **Self-documenting contracts**: Decorators make preconditions and
   postconditions explicit in the code itself:

   ```python
   @require(lambda amount: amount >= 0, "amount must be non-negative")
   @ensure(lambda result: result.balance >= 0, "balance must stay positive")
   def withdraw(account: Account, amount: int) -> Account:
       ...
   ```

3. **Class invariants**: `@invariant` wraps all public methods to validate state
   consistency:

   ```python
   @invariant(
       _session_id_is_well_formed,
       _created_at_has_tz,
       _created_at_is_utc,
   )
   class Session:
       ...
   ```

4. **Purity enforcement**: `@pure` detects side effects during testing by
   patching `open()`, `Path.write_*`, and logging:

   ```python
   @pure
   def serialize(data: Data) -> str:
       # Raises if function writes files or logs
       ...
   ```

5. **Test-time regression detection**: Contract violations surface as
   `AssertionError` with clear diagnostics during CI runs.

**Contract types**:

| Decorator | Checks | Timing |
|-----------|--------|--------|
| `@require` | Preconditions on arguments | Before call |
| `@ensure` | Postconditions on result/exception | After call |
| `@invariant` | Class-level consistency | Before and after public methods |
| `@pure` | No side effects or mutations | During call |

**Activation**:

```python
# Environment variable
WEAKINCENTIVES_DBC=1 python script.py

# Programmatic toggle
from weakincentives.dbc import enable_dbc, dbc_enabled

with dbc_enabled():
    # Contracts enforced in this scope
    ...
```

**Locations**:

- `dbc/__init__.py` - Decorator implementations
- `tests/plugins/dbc.py` - Pytest plugin for automatic activation
- `specs/DBC.md` - Full specification

**Consequences**:

- Internal-only; not exported to public API
- Requires discipline to add meaningful contracts
- Side-effect detection is best-effort (patches known call sites)

---

## ADR-004: Dataclass-First Data Modeling

**Status**: Accepted

**Context**: WINK needs serializable, type-safe data structures for events,
configuration, and tool parameters. Options include Pydantic, attrs, or stdlib
dataclasses.

**Decision**: Use frozen `@dataclass` with `slots=True` as the primary data
modeling pattern. Provide custom serde utilities instead of Pydantic.

**Rationale**:

1. **No external dependency**: Standard library dataclasses work everywhere
   without version conflicts.

2. **Immutability by default**: `frozen=True` prevents accidental mutations:

   ```python
   @dataclass(slots=True, frozen=True)
   class MyConfig:
       name: str
       count: int = 0
   ```

3. **Slots for memory efficiency**: `slots=True` eliminates `__dict__` overhead
   and prevents attribute typos.

4. **Type annotations as source of truth**: Pyright strict mode validates
   without runtime schema duplication.

5. **Predictable serialization**: Custom serde handles dataclasses, enums, and
   common types without magic:

   ```python
   from weakincentives.serde import serialize, parse

   json_dict = serialize(my_config)
   restored = parse(MyConfig, json_dict)
   ```

6. **DbC integration**: Dataclass fields work naturally with contract
   decorators.

**Patterns**:

```python
# Event types
@dataclass(frozen=True, slots=True)
class AddStep:
    step: str

# Configuration
@FrozenDataclass()  # Convenience wrapper
class LLMConfig:
    temperature: float | None = None
    max_tokens: int | None = None

# State slices (with declarative reducers)
@dataclass(frozen=True)
class AgentPlan:
    steps: tuple[str, ...]

    @reducer(on=AddStep)
    def add_step(self, event: AddStep) -> "AgentPlan":
        return replace(self, steps=(*self.steps, event.step))
```

**Locations**:

- `dataclasses/` - FrozenDataclass utilities
- `serde/` - Serialization/deserialization
- `types/dataclass.py` - Type protocols

**Consequences**:

- No automatic validation (use DbC for critical checks)
- No automatic migration (schema changes require explicit handling)
- Nested mutable defaults require `field(default_factory=...)`

---

## ADR-005: Provider-Agnostic Adapter Architecture

**Status**: Accepted

**Context**: WINK must support multiple LLM providers (OpenAI, Anthropic via
LiteLLM, Claude Agent SDK) without coupling orchestration logic to any specific
API.

**Decision**: Define a minimal `ProviderAdapter` ABC with protocol-based type
contracts for provider responses. Adapters encapsulate all provider-specific
logic.

**Rationale**:

1. **Single evaluation interface**: All adapters implement `evaluate()`:

   ```python
   class ProviderAdapter(ABC):
       @abstractmethod
       def evaluate(
           self,
           prompt: Prompt[OutputT],
           *,
           session: SessionProtocol,
           deadline: Deadline | None = None,
           budget: Budget | None = None,
           ...
       ) -> PromptResponse[OutputT]: ...
   ```

2. **Protocol-based response types**: Provider responses are consumed via
   protocols, allowing any compatible response object:

   ```python
   class ProviderCompletionResponse(Protocol):
       @property
       def choices(self) -> Sequence[ProviderChoice]: ...
   ```

3. **Shared inner loop**: Common logic (tool execution, throttling, event
   emission) lives in `run_inner_loop()`, not duplicated per adapter.

4. **Dynamic SDK imports**: Provider SDKs are imported lazily with helpful
   error messages:

   ```python
   try:
       from openai import OpenAI
   except ImportError as e:
       raise RuntimeError(
           "Install openai: pip install weakincentives[openai]"
       ) from e
   ```

5. **Configuration via frozen dataclasses**: Type-safe, serializable config:

   ```python
   adapter = OpenAIAdapter(
       model="gpt-4o",
       client_config=OpenAIClientConfig(timeout=30.0),
       model_config=OpenAIModelConfig(temperature=0.7),
   )
   ```

**Adapter implementations**:

| Adapter | Provider | Notes |
|---------|----------|-------|
| `OpenAIAdapter` | OpenAI Responses API | Native structured output |
| `LiteLLMAdapter` | 100+ providers via LiteLLM | Text-based structured output |
| `ClaudeAgentSDKAdapter` | Claude Agent SDK | MCP tool bridging |

**Locations**:

- `adapters/core.py` - Base ABC and response types
- `adapters/_provider_protocols.py` - Provider type contracts
- `adapters/openai.py` - OpenAI implementation
- `adapters/litellm.py` - LiteLLM implementation
- `adapters/claude_agent_sdk/` - Claude Agent SDK implementation

**Consequences**:

- Each new provider requires a new adapter class
- Provider-specific features may not map cleanly to the common interface
- Testing requires mocking at the adapter or protocol level

---

## ADR-006: Prompt Composition via Sections

**Status**: Accepted

**Context**: Prompts need to be modular, reusable, and dynamically composable
based on runtime conditions. Simple string concatenation doesn't scale.

**Decision**: Structure prompts as trees of `Section` objects with declarative
visibility, tools, and template rendering.

**Rationale**:

1. **Composable structure**: Sections nest to form prompt trees:

   ```python
   prompt = Prompt[OutputType](
       ns="my-namespace",
       key="my-prompt",
       sections=[
           MarkdownSection(title="Instructions", ...),
           MarkdownSection(
               title="Context",
               children=[
                   MarkdownSection(title="History", ...),
                   MarkdownSection(title="Current", ...),
               ],
           ),
       ],
   )
   ```

2. **Progressive disclosure**: Sections can render as summaries until the model
   requests expansion:

   ```python
   section = MarkdownSection(
       title="Details",
       visibility=SectionVisibility.SUMMARY,
       summary="Details available on request",
   )
   ```

3. **Tool co-location**: Tools attach to relevant sections:

   ```python
   section = MarkdownSection(
       title="Search",
       tools=(search_tool, filter_tool),
   )
   ```

4. **Template-based rendering**: Simple `$placeholder` substitution:

   ```python
   section = MarkdownSection(
       template="Process $item_count items",
       key="instructions",
   )
   ```

5. **Override system**: Production prompts can be tuned without code changes
   via `PromptOverride` persistence.

**Section types**:

| Type | Purpose |
|------|---------|
| `MarkdownSection` | Template-based text rendering |
| `DataSection` | Dataclass-based structured content |
| `WorkspaceSection` | VFS/workspace context injection |

**Locations**:

- `prompt/section.py` - Base Section ABC
- `prompt/markdown.py` - MarkdownSection implementation
- `prompt/rendering.py` - PromptRenderer orchestration
- `prompt/_visibility.py` - Visibility selectors
- `prompt/overrides/` - Override persistence

**Consequences**:

- Learning curve for section composition
- Deep nesting can make debugging harder
- Visibility logic adds complexity to rendering

---

## Summary

| ADR | Decision | Key Benefit |
|-----|----------|-------------|
| 001 | Protocols over ABCs | Structural typing, decoupling |
| 002 | Redux-style sessions | Deterministic replay, safe rollback |
| 003 | Zero-cost DbC | Self-documenting contracts, test-time safety |
| 004 | Dataclass-first | No dependencies, type-safe serde |
| 005 | Provider-agnostic adapters | Multi-provider support, shared logic |
| 006 | Section-based prompts | Composable, progressive disclosure |

These decisions work together to create a library that is:

- **Deterministic**: Pure reducers, immutable state, reproducible execution
- **Extensible**: Protocols and ABCs for clean extension points
- **Safe**: DbC catches regressions, snapshots enable rollback
- **Provider-agnostic**: Swap providers without changing orchestration
- **Composable**: Build complex prompts from simple, reusable sections

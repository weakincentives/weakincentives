# Extreme Automated Verification Strategies

This document outlines comprehensive testing strategies for achieving virtually
bug-free code in the weakincentives codebase. Strategies are organized by module
with specific, contextually relevant recommendations.

______________________________________________________________________

## Table of Contents

1. [Executive Summary](#executive-summary)
1. [Current Verification Infrastructure](#current-verification-infrastructure)
1. [Module-Specific Strategies](#module-specific-strategies)
   - [serde/ - Serialization](#serde---serialization)
   - [dbc/ - Design by Contract](#dbc---design-by-contract)
   - [runtime/session/ - State Management](#runtimesession---state-management)
   - [prompt/ - Tool Definitions](#prompt---tool-definitions)
   - [resources/ - Dependency Injection](#resources---dependency-injection)
   - [contrib/mailbox/ - Message Queue](#contribmailbox---message-queue)
   - [adapters/ - LLM Integrations](#adapters---llm-integrations)
   - [formal/ - TLA+ Specifications](#formal---tla-specifications)
1. [Cross-Cutting Verification](#cross-cutting-verification)
1. [Implementation Priority](#implementation-priority)

______________________________________________________________________

## Executive Summary

The codebase already has strong verification infrastructure:

| Current State | Coverage |
|--------------|----------|
| Code coverage | 100% mandatory |
| Type checking | Pyright strict + ty |
| Property testing | Hypothesis (3+ modules) |
| Formal verification | TLA+ model checking |
| Design by contract | `@require`, `@ensure`, `@invariant`, `@pure` |
| Concurrency testing | Thread stress tests |

**To achieve virtually bug-free code, we recommend adding:**

1. **Metamorphic testing** for serde/prompt rendering
1. **Fault injection** for resource lifecycle
1. **Linearizability checking** for concurrent operations
1. **Mutation testing** to verify test quality
1. **Exhaustive boundary testing** via property-based generation
1. **Reference model comparison** for all stateful components

______________________________________________________________________

## Current Verification Infrastructure

```
make check          # All checks (format, lint, typecheck, security, tests)
make verify-formal  # TLA+ model checking (~30s)
make property-tests # Hypothesis property-based tests
make stress-tests   # Concurrent stress tests
make verify-all     # Combined: TLA+ + Hypothesis + stress
```

______________________________________________________________________

## Module-Specific Strategies

### serde/ - Serialization

**Current:** Hypothesis round-trip tests (`test_dataclass_hypothesis.py`)

**Recommended Additions:**

#### 1. Metamorphic Testing for Constraint Validation

Verify that transformations preserve validity/invalidity:

```python
from hypothesis import given, strategies as st, assume

@given(st.integers(), st.integers(min_value=1))
def test_ge_constraint_metamorphic(value: int, offset: int) -> None:
    """If value passes ge=N, then value+offset also passes."""
    @dataclass
    class Sample:
        x: Annotated[int, {"ge": 0}]

    if value >= 0:
        # Valid input stays valid when increased
        assert parse(Sample, {"x": value})
        assert parse(Sample, {"x": value + offset})
```

#### 2. Exhaustive Boundary Testing

```python
@given(st.data())
def test_boundary_conditions_exhaustive(data: st.DataObject) -> None:
    """Test all boundary conditions for numeric constraints."""
    ge_bound = data.draw(st.integers(min_value=-1000, max_value=1000))
    le_bound = data.draw(st.integers(min_value=ge_bound, max_value=ge_bound + 100))

    @dataclass
    class BoundedInt:
        value: Annotated[int, {"ge": ge_bound, "le": le_bound}]

    # Test exact boundaries
    assert parse(BoundedInt, {"value": ge_bound})
    assert parse(BoundedInt, {"value": le_bound})

    # Test violations
    with pytest.raises(ValueError):
        parse(BoundedInt, {"value": ge_bound - 1})
    with pytest.raises(ValueError):
        parse(BoundedInt, {"value": le_bound + 1})
```

#### 3. JSON Schema Consistency

```python
@given(record_strategy())
def test_schema_validates_generated_instances(record: NestedRecord) -> None:
    """Generated JSON Schema accepts all valid instances."""
    import jsonschema
    json_schema = schema(NestedRecord)
    payload = dump(record)
    jsonschema.validate(payload, json_schema)  # Must not raise
```

#### 4. Cross-Version Serialization Compatibility

```python
def test_forward_compatibility() -> None:
    """Old serialized data can be deserialized by new code."""
    # Store known serializations as fixtures
    legacy_data = {"name": "test", "age": 30}  # v1 schema
    result = parse(UserV2, legacy_data, extra="ignore")
    assert result.name == "test"
```

______________________________________________________________________

### dbc/ - Design by Contract

**Current:** Basic contract enforcement tests

**Recommended Additions:**

#### 1. Contract Composition Testing

Test that stacked decorators maintain proper evaluation order:

```python
@given(st.integers(), st.integers())
def test_require_ensure_composition(a: int, b: int) -> None:
    """Preconditions checked before postconditions."""
    call_order = []

    def pre(x: int, y: int) -> bool:
        call_order.append("pre")
        return x > 0

    def post(x: int, y: int, *, result: int) -> bool:
        call_order.append("post")
        return result > x

    @require(pre)
    @ensure(post)
    def add_positive(x: int, y: int) -> int:
        return x + y + 1

    assume(a > 0)
    add_positive(a, b)
    assert call_order == ["pre", "post"]
```

#### 2. Async Contract Support Testing

```python
@pytest.mark.asyncio
async def test_contracts_work_with_async() -> None:
    """Contracts work correctly with async functions."""
    @require(lambda x: x > 0)
    @ensure(lambda x, result: result > x)
    async def async_increment(x: int) -> int:
        await asyncio.sleep(0.001)
        return x + 1

    result = await async_increment(5)
    assert result == 6
```

#### 3. Pure Function Mutation Detection

```python
@given(st.lists(st.integers()))
def test_pure_detects_in_place_sort(items: list[int]) -> None:
    """@pure catches in-place mutations."""
    @pure
    def bad_sort(xs: list[int]) -> list[int]:
        xs.sort()  # Mutation!
        return xs

    with pytest.raises(AssertionError, match="mutation"):
        bad_sort(items.copy())
```

#### 4. Invariant Inheritance Testing

```python
def test_invariant_inherited_by_subclass() -> None:
    """Subclasses inherit invariant checks."""
    def positive_balance(self: object) -> bool:
        return getattr(self, "balance", 0) >= 0

    @invariant(positive_balance)
    class Account:
        def __init__(self, initial: int) -> None:
            self.balance = initial

    class SavingsAccount(Account):
        def add_interest(self) -> None:
            self.balance = int(self.balance * 1.05)

    # Subclass should still have invariant
    acct = SavingsAccount(100)
    acct.add_interest()
    # This should trigger invariant check
```

______________________________________________________________________

### runtime/session/ - State Management

**Current:** Dispatch tests, snapshot tests

**Recommended Additions:**

#### 1. Reducer Commutativity Testing

For independent events, order shouldn't matter:

```python
@given(st.permutations(["event_a", "event_b", "event_c"]))
def test_independent_events_commute(event_order: list[str]) -> None:
    """Independent events produce same final state regardless of order."""
    session = create_test_session()

    events = {
        "event_a": EventA(data="a"),
        "event_b": EventB(data="b"),
        "event_c": EventC(data="c"),
    }

    for name in event_order:
        session.dispatch(events[name])

    # Same final state regardless of order
    assert session[EventA].latest() == EventA(data="a")
    assert session[EventB].latest() == EventB(data="b")
    assert session[EventC].latest() == EventC(data="c")
```

#### 2. Snapshot Idempotency

```python
@given(event_sequence_strategy())
def test_snapshot_restore_idempotent(events: list[SessionEvent]) -> None:
    """Snapshot → Restore → Snapshot produces identical snapshot."""
    session = create_test_session()
    for event in events:
        session.dispatch(event)

    snapshot1 = session.snapshot()
    restored = Session.from_snapshot(snapshot1)
    snapshot2 = restored.snapshot()

    assert snapshot1 == snapshot2
```

#### 3. Linearizability Testing with Reference Model

```python
class SessionReferenceModel:
    """Simple dict-based reference implementation."""
    def __init__(self) -> None:
        self.slices: dict[type, list[Any]] = {}

    def dispatch(self, event: SessionEvent) -> None:
        self.slices.setdefault(type(event), []).append(event)

    def latest(self, event_type: type) -> SessionEvent | None:
        items = self.slices.get(event_type, [])
        return items[-1] if items else None

@given(event_sequence_strategy())
def test_session_matches_reference(events: list[SessionEvent]) -> None:
    """Session behavior matches simple reference model."""
    session = create_test_session()
    reference = SessionReferenceModel()

    for event in events:
        session.dispatch(event)
        reference.dispatch(event)

    for event_type in {type(e) for e in events}:
        assert session[event_type].latest() == reference.latest(event_type)
```

#### 4. Concurrent Dispatch Linearizability

```python
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant

class SessionLinearizabilityTest(RuleBasedStateMachine):
    """Verify concurrent dispatches linearize correctly."""

    def __init__(self) -> None:
        super().__init__()
        self.session = create_thread_safe_session()
        self.expected_count = 0

    @rule()
    def dispatch_event(self) -> None:
        self.session.dispatch(CounterEvent())
        self.expected_count += 1

    @invariant()
    def count_matches(self) -> None:
        actual = len(self.session[CounterEvent].all())
        assert actual == self.expected_count

TestSessionLinearizability = SessionLinearizabilityTest.TestCase
```

______________________________________________________________________

### prompt/ - Tool Definitions

**Current:** Tool validation tests, rendering tests

**Recommended Additions:**

#### 1. Tool Name Regex Exhaustive Testing

```python
import string
from hypothesis import given, strategies as st

VALID_CHARS = string.ascii_lowercase + string.digits + "_-"

@given(st.text(alphabet=VALID_CHARS, min_size=1, max_size=64))
def test_valid_tool_names_accepted(name: str) -> None:
    """All names matching pattern are accepted."""
    @dataclass
    class Params:
        x: int

    tool = Tool[Params, None](
        name=name,
        description="Test tool",
        handler=lambda p, context: ToolResult.ok(None),
    )
    assert tool.name == name

@given(st.text(min_size=1, max_size=100))
def test_invalid_tool_names_rejected(name: str) -> None:
    """Names not matching pattern are rejected."""
    assume(not re.match(r"^[a-z0-9_-]{1,64}$", name))

    with pytest.raises(PromptValidationError):
        Tool[Params, None](name=name, description="Test", handler=None)
```

#### 2. Tool Result Serialization Round-Trip

```python
@given(tool_result_strategy())
def test_tool_result_serializable(result: ToolResult[Any]) -> None:
    """All ToolResults can round-trip through JSON."""
    payload = dump(result)
    restored = parse(type(result), payload)
    assert restored == result
```

#### 3. Rendering Determinism

```python
@given(section_configuration_strategy())
def test_prompt_rendering_deterministic(config: SectionConfig) -> None:
    """Same config always produces same rendered output."""
    prompt = build_prompt(config)
    render1 = prompt.render()
    render2 = prompt.render()
    assert render1 == render2
```

#### 4. Tool Handler Exception Safety

```python
@given(st.sampled_from([ValueError, RuntimeError, TypeError, KeyError]))
def test_tool_handler_exceptions_become_errors(exc_type: type) -> None:
    """All handler exceptions convert to ToolResult.error()."""
    def failing_handler(p: Params, *, context: ToolContext) -> ToolResult[None]:
        raise exc_type("Simulated failure")

    result = execute_tool_safely(failing_handler, Params(x=1), context)
    assert result.is_error
    assert exc_type.__name__ in result.error_message
```

______________________________________________________________________

### resources/ - Dependency Injection

**Current:** Basic lifecycle tests

**Recommended Additions:**

#### 1. Resource Lifecycle State Machine

```python
from hypothesis.stateful import RuleBasedStateMachine, rule, precondition

class ResourceLifecycleTest(RuleBasedStateMachine):
    """Model resource lifecycle transitions."""

    def __init__(self) -> None:
        super().__init__()
        self.registry = ResourceRegistry()
        self.context: ScopedResourceContext | None = None
        self.resources_created: set[str] = set()

    @rule()
    def register_binding(self) -> None:
        name = f"resource_{len(self.resources_created)}"
        self.registry = self.registry.with_binding(
            Binding(Protocol, lambda r: TestResource(name))
        )
        self.resources_created.add(name)

    @precondition(lambda self: self.context is None)
    @rule()
    def enter_context(self) -> None:
        self.context = self.registry.create_context()
        self.context.__enter__()

    @precondition(lambda self: self.context is not None)
    @rule()
    def exit_context(self) -> None:
        self.context.__exit__(None, None, None)
        self.context = None

    @precondition(lambda self: self.context is not None)
    @rule()
    def get_resource(self) -> None:
        # Should not raise
        self.context.get(Protocol)

TestResourceLifecycle = ResourceLifecycleTest.TestCase
```

#### 2. Scope Isolation Testing

```python
@given(st.lists(st.sampled_from(["SINGLETON", "TOOL_CALL", "PROTOTYPE"])))
def test_scope_isolation(scopes: list[str]) -> None:
    """Resources in different scopes don't interfere."""
    registry = ResourceRegistry()

    for i, scope in enumerate(scopes):
        registry = registry.with_binding(
            Binding(f"Resource{i}", lambda r, i=i: Resource(i), scope=Scope[scope])
        )

    with registry.create_context() as ctx:
        for i in range(len(scopes)):
            resource = ctx.get(f"Resource{i}")
            assert resource.id == i
```

#### 3. Circular Dependency Detection

```python
def test_circular_dependency_detected_at_registration() -> None:
    """Circular dependencies raise at registration time, not runtime."""
    registry = ResourceRegistry()

    # A depends on B, B depends on A
    registry = registry.with_binding(
        Binding(ProtocolA, lambda r: ServiceA(r.get(ProtocolB)))
    )

    with pytest.raises(CircularDependencyError):
        registry = registry.with_binding(
            Binding(ProtocolB, lambda r: ServiceB(r.get(ProtocolA)))
        )
```

______________________________________________________________________

### contrib/mailbox/ - Message Queue

**Current:** Excellent coverage with stateful property tests and TLA+ formal verification

**Recommended Additions:**

#### 1. Network Partition Simulation

```python
@pytest.mark.parametrize("partition_duration", [0.1, 0.5, 1.0])
def test_survives_network_partition(
    mailbox: RedisMailbox[Any],
    partition_duration: float,
) -> None:
    """Messages survive simulated network partitions."""
    mailbox.send("before_partition")

    # Simulate partition by pausing Redis
    with redis_pause(duration=partition_duration):
        # Operations should timeout gracefully
        with pytest.raises(TimeoutError):
            mailbox.receive(visibility_timeout=0.1)

    # After partition heals, message still available
    msgs = mailbox.receive(visibility_timeout=30)
    assert len(msgs) == 1
    assert msgs[0].body == "before_partition"
```

#### 2. Clock Skew Tolerance

```python
@given(st.floats(min_value=-5.0, max_value=5.0))
def test_visibility_timeout_handles_clock_skew(skew_seconds: float) -> None:
    """Visibility timeout works with clock skew."""
    with mocked_time(skew=skew_seconds):
        mailbox = RedisMailbox(name="test", client=redis_client)
        mailbox.send("test")

        msgs = mailbox.receive(visibility_timeout=10)
        assert len(msgs) == 1

        # Even with skew, message shouldn't be immediately requeued
        msgs2 = mailbox.receive(visibility_timeout=10, wait_time_seconds=0)
        assert len(msgs2) == 0
```

#### 3. Chaos Engineering: Random Failures

```python
class ChaoticRedisClient:
    """Redis client that randomly fails operations."""

    def __init__(self, client: Redis, failure_rate: float = 0.1) -> None:
        self._client = client
        self._failure_rate = failure_rate

    def __getattr__(self, name: str) -> Any:
        original = getattr(self._client, name)
        if random.random() < self._failure_rate:
            raise ConnectionError("Chaos monkey!")
        return original

@given(st.lists(st.text(min_size=1), min_size=10, max_size=50))
@settings(max_examples=20, deadline=None)
def test_eventual_consistency_under_chaos(messages: list[str]) -> None:
    """All messages eventually delivered under chaotic conditions."""
    chaos_client = ChaoticRedisClient(redis_client, failure_rate=0.1)
    mailbox = RedisMailbox(name="chaos-test", client=chaos_client)

    sent = set()
    for msg in messages:
        for _ in range(3):  # Retry on failure
            try:
                msg_id = mailbox.send(msg)
                sent.add(msg_id)
                break
            except ConnectionError:
                continue

    received = set()
    for _ in range(len(sent) * 5):  # More attempts due to failures
        try:
            msgs = mailbox.receive(visibility_timeout=60)
            for m in msgs:
                received.add(m.id)
                m.acknowledge()
        except ConnectionError:
            continue
        if received == sent:
            break

    assert received == sent, f"Lost messages: {sent - received}"
```

______________________________________________________________________

### adapters/ - LLM Integrations

**Current:** Mock adapter tests

**Recommended Additions:**

#### 1. Response Parser Fuzzing

```python
@given(st.binary(min_size=0, max_size=10000))
def test_response_parser_handles_malformed_input(data: bytes) -> None:
    """Parser never crashes on malformed input."""
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return  # Skip invalid UTF-8

    try:
        result = parse_llm_response(text)
        assert result is not None or result is None  # Just don't crash
    except MalformedResponseError:
        pass  # Expected for invalid input
    # No other exceptions should escape
```

#### 2. Token Budget Exhaustion

```python
@given(st.integers(min_value=1, max_value=100000))
def test_token_budget_never_exceeded(max_tokens: int) -> None:
    """Adapter respects token budget limits."""
    budget = BudgetTracker(max_tokens=max_tokens)
    adapter = MockAdapter(budget=budget)

    while not budget.exhausted:
        response = adapter.complete("Test prompt")
        assert budget.used_tokens <= max_tokens
```

#### 3. Rate Limit Backoff Verification

```python
def test_rate_limit_exponential_backoff() -> None:
    """Rate limiting uses proper exponential backoff."""
    delays = []

    with mock.patch("time.sleep", side_effect=lambda d: delays.append(d)):
        adapter = RateLimitedAdapter(base_delay=1.0, max_retries=5)
        adapter._simulate_rate_limits(count=5)

    # Verify exponential growth
    for i in range(1, len(delays)):
        assert delays[i] >= delays[i-1] * 1.5, "Backoff not exponential"
```

______________________________________________________________________

### formal/ - TLA+ Specifications

**Current:** Extraction and model checking tests

**Recommended Additions:**

#### 1. Specification Consistency with Implementation

```python
def test_spec_actions_match_implementation_methods() -> None:
    """Every TLA+ action has a corresponding Python method."""
    spec = extract_spec(RedisMailbox)

    impl_methods = {m for m in dir(RedisMailbox) if not m.startswith("_")}
    spec_actions = {a.name.lower() for a in spec.actions}

    # All spec actions should have implementation
    missing = spec_actions - impl_methods
    assert not missing, f"Spec actions without implementation: {missing}"
```

#### 2. Invariant Traceability

```python
def test_invariants_have_test_coverage() -> None:
    """Every TLA+ invariant has corresponding Python test."""
    spec = extract_spec(RedisMailbox)

    # Find all test functions in invariant test file
    import tests.contrib.mailbox.test_redis_mailbox_invariants as inv_tests
    test_names = {name.lower() for name in dir(inv_tests) if name.startswith("Test")}

    for inv in spec.invariants:
        # INV-1 should have TestMessageStateExclusivity or similar
        inv_pattern = inv.name.lower().replace("_", "")
        matching = [t for t in test_names if inv_pattern in t.replace("_", "")]
        assert matching, f"No test for invariant {inv.id}: {inv.name}"
```

#### 3. Bounded Model Checking with Varying Bounds

```python
@pytest.mark.parametrize("max_messages,max_consumers", [
    (2, 2),
    (3, 3),
    (5, 2),
    (2, 5),
])
def test_invariants_hold_at_various_bounds(
    max_messages: int,
    max_consumers: int,
    tmp_path: Path,
) -> None:
    """Invariants hold for different state space bounds."""
    spec, _, _, result = extract_and_verify(
        RedisMailbox,
        output_dir=tmp_path,
        model_check_enabled=True,
        tlc_config={
            "constants": {
                "MaxMessages": max_messages,
                "NumConsumers": max_consumers,
            }
        },
    )
    assert result.passed
```

______________________________________________________________________

## Cross-Cutting Verification

### 1. Mutation Testing

Use `mutmut` to verify test quality:

```bash
# Install
uv add --dev mutmut

# Run mutation testing on critical modules
uv run mutmut run --paths-to-mutate src/weakincentives/serde/
uv run mutmut run --paths-to-mutate src/weakincentives/dbc/
uv run mutmut run --paths-to-mutate src/weakincentives/runtime/session/
```

Add to Makefile:

```makefile
# Mutation testing (slow, run periodically)
mutation-test:
    @uv run --all-extras mutmut run \
        --paths-to-mutate src/weakincentives/serde/ \
        --tests-dir tests/serde/ \
        --runner "pytest -x -q"
    @uv run mutmut results
```

Target: **>80% mutation score** (mutations caught/total mutations)

### 2. Fault Injection Framework

```python
# tests/helpers/fault_injection.py

from contextlib import contextmanager
from typing import Iterator
from unittest import mock

@contextmanager
def inject_fault(
    target: str,
    exception: type[Exception] = RuntimeError,
    probability: float = 1.0,
) -> Iterator[None]:
    """Inject faults into specific code paths."""
    original = None

    def faulty(*args, **kwargs):
        if random.random() < probability:
            raise exception("Injected fault")
        return original(*args, **kwargs)

    with mock.patch(target, side_effect=faulty):
        yield

# Usage in tests
def test_recovers_from_disk_failure() -> None:
    with inject_fault("builtins.open", IOError, probability=0.3):
        # Code should handle intermittent disk failures
        session = Session.from_file("session.json")
```

### 3. Contract-Annotated Test Generation

Automatically generate tests from contracts:

```python
# toolchain/contract_test_generator.py

def generate_tests_from_contracts(module: types.ModuleType) -> str:
    """Generate Hypothesis tests from @require/@ensure contracts."""
    tests = []

    for name, func in inspect.getmembers(module, inspect.isfunction):
        contracts = getattr(func, "__contracts__", [])
        if contracts:
            tests.append(generate_property_test(func, contracts))

    return "\n\n".join(tests)
```

### 4. Comprehensive Dataclass Verification Plugin

Extend the existing `dataclass_serde.py` plugin:

```python
# tests/plugins/dataclass_contracts.py

def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-add tests for all dataclasses in session snapshots."""
    for cls in discover_session_dataclasses():
        # Test: slots=True
        # Test: frozen=True (where applicable)
        # Test: all fields have type annotations
        # Test: no mutable defaults
        # Test: serialization round-trip
        # Test: hash stability (for frozen)
        pass
```

______________________________________________________________________

## Implementation Priority

### Phase 1: High Impact, Low Effort (Week 1)

| Strategy | Module | Effort | Impact |
|----------|--------|--------|--------|
| Metamorphic testing | serde | Low | High |
| Boundary testing | serde | Low | High |
| Reducer commutativity | session | Medium | High |
| Tool name exhaustive | prompt | Low | Medium |

### Phase 2: Critical Safety (Week 2)

| Strategy | Module | Effort | Impact |
|----------|--------|--------|--------|
| Linearizability testing | session | High | Critical |
| Network partition simulation | mailbox | Medium | High |
| Fault injection framework | cross-cutting | Medium | High |
| Resource lifecycle state machine | resources | Medium | High |

### Phase 3: Quality Assurance (Week 3)

| Strategy | Module | Effort | Impact |
|----------|--------|--------|--------|
| Mutation testing setup | cross-cutting | Low | High |
| Spec-implementation consistency | formal | Medium | High |
| Chaos engineering | mailbox | High | Medium |
| Response parser fuzzing | adapters | Medium | Medium |

### Phase 4: Continuous Improvement (Ongoing)

| Strategy | Module | Effort | Impact |
|----------|--------|--------|--------|
| Contract test generation | cross-cutting | High | Medium |
| Invariant traceability | formal | Low | Medium |
| Clock skew tolerance | mailbox | Medium | Medium |

______________________________________________________________________

## Appendix: New Makefile Targets

```makefile
# Phase 1
verify-metamorphic:
    @uv run --all-extras pytest tests/ -k "metamorphic" --no-cov -v

verify-boundaries:
    @uv run --all-extras pytest tests/ -k "boundary" --no-cov -v

# Phase 2
verify-linearizable:
    @uv run --all-extras pytest tests/ -k "lineariz" --no-cov -v

verify-fault-injection:
    @uv run --all-extras pytest tests/ -k "fault_injection" --no-cov -v

# Phase 3
mutation-test:
    @uv run --all-extras mutmut run \
        --paths-to-mutate src/weakincentives/serde/,src/weakincentives/dbc/ \
        --tests-dir tests/ \
        --runner "pytest -x -q"

# Combined
verify-extreme:
    @$(MAKE) verify-formal
    @$(MAKE) property-tests
    @$(MAKE) stress-tests
    @$(MAKE) verify-metamorphic
    @$(MAKE) verify-boundaries
    @$(MAKE) verify-linearizable
    @echo "All extreme verification passed"
```

______________________________________________________________________

## Conclusion

By implementing these strategies systematically, the codebase will achieve:

1. **100% mutation score** on critical paths (serde, dbc, session)
1. **Formal guarantees** via TLA+ for all stateful components
1. **Exhaustive boundary coverage** via Hypothesis
1. **Linearizability proof** for concurrent operations
1. **Chaos resilience** through fault injection
1. **Contract completeness** via automated test generation

The combination of existing infrastructure (100% coverage, strict typing, property tests,
formal specs) with these additions creates a verification pipeline that catches virtually
all classes of bugs before they reach production.

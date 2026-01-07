# Property-Based Testing

## Purpose

Property-based testing with Hypothesis generates thousands of random inputs to
verify that invariants hold universally, catching edge cases that humans
systematically miss: empty inputs, maximum values, unicode edge cases, specific
bit patterns, and adversarial combinations. Unlike example-based tests that
verify "these 5 inputs work," property tests verify "for all valid inputs, these
properties hold."

This catches off-by-one errors, integer overflow, incorrect boundary handling,
and assumption violations that only manifest with specific inputs. The impact is
highest for pure functions and data transformations where you can state
properties like "serialization round-trips," "sorting is idempotent," or
"merging is associative."

For a library enforcing 100% coverage, property tests provide deeper coverage by
testing the **space of inputs** rather than specific points.

## Guiding Principles

- **Properties over examples**: State what must always be true, not what happens
  for specific inputs. Example tests document behavior; property tests verify
  correctness.
- **Shrinking is essential**: When a test fails, Hypothesis automatically finds
  the minimal failing example. Never disable shrinking.
- **Strategies encode domain knowledge**: Custom strategies capture valid input
  shapes, constraints, and relationships. Invest in reusable strategies.
- **Stateful tests for protocols**: Use `RuleBasedStateMachine` to verify
  multi-step interactions maintain invariants across operation sequences.
- **Reproducibility via database**: Hypothesis stores failing examples in
  `.hypothesis/`. Commit this directory to preserve regression coverage.
- **Health checks matter**: Suppress health checks deliberately. A slow strategy
  or too-large data indicates design problems.
- **Complement, don't replace**: Property tests supplement example-based tests.
  Use examples for documentation, edge case pinning, and error message testing.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Property Test Structure                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  Strategy   │───▶│  Property   │───▶│  Assertion  │───▶│  Shrinking  │  │
│  │  Library    │    │  Function   │    │  Failure    │    │  Minimal    │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│        │                  │                                     │          │
│        ▼                  ▼                                     ▼          │
│  ┌─────────────┐    ┌─────────────┐                      ┌─────────────┐  │
│  │ st.builds() │    │ @given(...) │                      │ .hypothesis/│  │
│  │ st.from_    │    │ @settings() │                      │  database   │  │
│  │   type()    │    │ @example()  │                      │  (commit!)  │  │
│  └─────────────┘    └─────────────┘                      └─────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         Stateful Testing Structure                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     RuleBasedStateMachine                           │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  @initialize       Set up initial state                             │   │
│  │  @rule(...)        Operations that modify state                     │   │
│  │  @invariant        Properties checked after every rule              │   │
│  │  @precondition     Guards preventing invalid operation sequences    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐               │
│  │  init  │─▶│ rule_1 │─▶│ rule_2 │─▶│ rule_1 │─▶│  ...   │               │
│  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘               │
│       │           │           │           │           │                    │
│       └───────────┴───────────┴───────────┴───────────┘                    │
│                           │                                                 │
│                    @invariant checked                                       │
│                    after each step                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Test Organization

```
tests/
├── property/                    # Hypothesis property tests
│   ├── strategies/              # Reusable strategy library
│   │   ├── __init__.py          # Public strategy exports
│   │   ├── primitives.py        # JSON-safe primitives, constrained strings
│   │   ├── dataclasses.py       # FrozenDataclass construction strategies
│   │   ├── serde.py             # Valid serializable structures
│   │   ├── session.py           # Events, slices, snapshots
│   │   └── prompt.py            # Sections, tools, prompts
│   ├── serde/                   # Serialization round-trip tests
│   ├── dataclasses/             # Immutability and update tests
│   ├── session/                 # State transition tests
│   ├── prompt/                  # Composition and normalization tests
│   └── stateful/                # RuleBasedStateMachine tests
└── ...
```

## Core Concepts

### Strategy Composition

Strategies generate random values satisfying constraints. Compose them to build
complex valid inputs:

```python
from hypothesis import strategies as st
from hypothesis import given, settings, assume
from dataclasses import dataclass

# Primitive strategies with constraints
valid_identifiers = st.from_regex(r"[a-z][a-z0-9_]{0,30}", fullmatch=True)
positive_ints = st.integers(min_value=1, max_value=10_000)
json_primitives = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=100),
)

# Recursive JSON structure
json_values = st.recursive(
    json_primitives,
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(max_size=20), children, max_size=5),
    ),
    max_leaves=20,
)

# Dataclass construction with st.builds
@dataclass(frozen=True, slots=True)
class Config:
    name: str
    count: int
    tags: tuple[str, ...]

config_strategy = st.builds(
    Config,
    name=valid_identifiers,
    count=positive_ints,
    tags=st.tuples(valid_identifiers, valid_identifiers),
)
```

### Property Categories

| Category | Description | Example Properties |
|----------|-------------|-------------------|
| Round-trip | Encode then decode preserves value | `parse(dump(x)) == x` |
| Idempotent | Applying operation twice equals once | `normalize(normalize(k)) == normalize(k)` |
| Invariant | Property holds before and after | `len(slice) >= 0` after any operation |
| Commutative | Order doesn't matter | `merge(a, b) == merge(b, a)` |
| Associative | Grouping doesn't matter | `(a + b) + c == a + (b + c)` |
| Monotonic | Value only increases/decreases | `version_after >= version_before` |
| Identity | Neutral element exists | `merge(x, empty) == x` |
| Inverse | Operation can be undone | `restore(snapshot(s)) == s` |

### Writing Property Tests

```python
from hypothesis import given, settings, assume, example
from hypothesis import strategies as st

from weakincentives.serde import dump, parse

@given(config_strategy)
def test_serde_roundtrip(config: Config) -> None:
    """Serialization round-trip preserves value."""
    serialized = dump(config)
    restored = parse(Config, serialized)
    assert restored == config

@given(st.text())
@example("")                    # Pin edge case
@example("  spaces  ")          # Pin edge case
@example("UPPER_case_123")      # Pin edge case
def test_normalize_idempotent(key: str) -> None:
    """Normalization is idempotent."""
    once = normalize_component_key(key)
    twice = normalize_component_key(once)
    assert once == twice

@given(st.lists(st.integers()))
def test_sort_idempotent(items: list[int]) -> None:
    """Sorting twice equals sorting once."""
    once = sorted(items)
    twice = sorted(once)
    assert once == twice

@given(st.dictionaries(st.text(), json_values))
def test_dump_produces_json_safe(data: dict[str, object]) -> None:
    """Dumped output contains only JSON-safe types."""
    result = dump(data)
    # Should not raise
    json.dumps(result)
```

### Stateful Testing

Use `RuleBasedStateMachine` for protocol verification:

```python
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    rule,
    precondition,
    Bundle,
    consumes,
)
from hypothesis import strategies as st

class SessionStateMachine(RuleBasedStateMachine):
    """Verify session state transitions maintain invariants."""

    items = Bundle("items")

    @initialize()
    def init_session(self) -> None:
        self.session = Session(bus=InProcessDispatcher())
        self.model: list[Item] = []  # Shadow state for verification

    @rule(target=items, item=item_strategy)
    def append_item(self, item: Item) -> Item:
        self.session.dispatch(AppendItem(item))
        self.model.append(item)
        return item

    @rule(item=consumes(items))
    def remove_item(self, item: Item) -> None:
        self.session.dispatch(RemoveItem(item.id))
        self.model = [i for i in self.model if i.id != item.id]

    @precondition(lambda self: len(self.model) > 0)
    @rule()
    def clear_all(self) -> None:
        self.session[Item].clear()
        self.model.clear()

    @invariant()
    def items_match_model(self) -> None:
        actual = list(self.session[Item].all())
        assert actual == self.model

    @invariant()
    def count_non_negative(self) -> None:
        assert len(list(self.session[Item].all())) >= 0


TestSessionState = SessionStateMachine.TestCase
```

## WINK-Specific Strategies

### Serde Strategies

```python
# tests/property/strategies/serde.py
from hypothesis import strategies as st
from typing import Any
from datetime import datetime, timezone
from uuid import UUID
from decimal import Decimal
from pathlib import Path

# JSON-safe primitives (what dump() produces)
json_safe_primitives = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=100),
)

# Types requiring special serialization
serializable_primitives = st.one_of(
    json_safe_primitives,
    st.uuids(),
    st.decimals(allow_nan=False, allow_infinity=False),
    st.datetimes(timezones=st.just(timezone.utc)),
    st.from_type(Path),
)

# Constrained strings matching common patterns
identifier_strategy = st.from_regex(r"[a-z][a-z0-9_]{0,30}", fullmatch=True)
namespace_strategy = st.from_regex(r"[a-z][a-z0-9-]{0,20}", fullmatch=True)
version_strategy = st.from_regex(r"\d+\.\d+\.\d+", fullmatch=True)
```

### Session Strategies

```python
# tests/property/strategies/session.py
from hypothesis import strategies as st
from weakincentives.runtime.session import SliceView, Append, Replace, Clear, Extend

def slice_op_strategy(item_strategy: st.SearchStrategy) -> st.SearchStrategy:
    """Generate valid SliceOp instances."""
    return st.one_of(
        st.builds(Append, item=item_strategy),
        st.builds(Replace, items=st.tuples(item_strategy)),
        st.builds(Extend, items=st.lists(item_strategy, min_size=1, max_size=5)),
        st.just(Clear()),
    )

def event_sequence_strategy(
    event_strategies: list[st.SearchStrategy],
) -> st.SearchStrategy:
    """Generate valid event sequences for session dispatch."""
    return st.lists(
        st.one_of(*event_strategies),
        min_size=1,
        max_size=20,
    )
```

### Dataclass Strategies

```python
# tests/property/strategies/dataclasses.py
from hypothesis import strategies as st
from dataclasses import fields, MISSING
from typing import get_type_hints, Any

def dataclass_strategy(cls: type) -> st.SearchStrategy:
    """
    Build strategy for any frozen dataclass.

    Recursively generates strategies for nested dataclasses.
    Respects field defaults and factories.
    """
    hints = get_type_hints(cls)
    kwargs = {}

    for field in fields(cls):
        if not field.init:
            continue

        field_type = hints[field.name]

        # Use default if available (50% of time for optional feel)
        if field.default is not MISSING:
            kwargs[field.name] = st.one_of(
                st.just(field.default),
                strategy_for_type(field_type),
            )
        elif field.default_factory is not MISSING:
            kwargs[field.name] = st.one_of(
                st.builds(field.default_factory),
                strategy_for_type(field_type),
            )
        else:
            kwargs[field.name] = strategy_for_type(field_type)

    return st.builds(cls, **kwargs)
```

## Property Test Patterns for WINK Modules

### Serde Round-Trip

```python
@given(dataclass_strategy(MyConfig))
def test_parse_dump_roundtrip(instance: MyConfig) -> None:
    """parse(dump(x)) == x for all valid instances."""
    serialized = dump(instance)
    restored = parse(MyConfig, serialized)
    assert restored == instance

@given(dataclass_strategy(MyConfig))
def test_dump_produces_json_safe(instance: MyConfig) -> None:
    """Serialized output contains only JSON primitives."""
    result = dump(instance)
    # json.dumps succeeds without custom encoder
    json_str = json.dumps(result)
    # Round-trip through JSON preserves structure
    assert json.loads(json_str) == result
```

### FrozenDataclass Operations

```python
@given(dataclass_strategy(Config), st.data())
def test_update_preserves_unchanged_fields(config: Config, data: st.DataObject) -> None:
    """update() preserves fields not explicitly changed."""
    # Pick one field to change
    field_name = data.draw(st.sampled_from([f.name for f in fields(Config) if f.init]))
    field_type = get_type_hints(Config)[field_name]
    new_value = data.draw(strategy_for_type(field_type))

    updated = config.update(**{field_name: new_value})

    for field in fields(Config):
        if field.name == field_name:
            assert getattr(updated, field_name) == new_value
        else:
            assert getattr(updated, field.name) == getattr(config, field.name)

@given(dataclass_strategy(Config))
def test_update_idempotent(config: Config) -> None:
    """Updating with same values produces equal result."""
    kwargs = {f.name: getattr(config, f.name) for f in fields(Config) if f.init}
    updated = config.update(**kwargs)
    assert updated == config
```

### Session Snapshot Round-Trip

```python
@given(event_sequence_strategy([add_item_strategy, remove_item_strategy]))
def test_snapshot_restore_roundtrip(events: list[Event]) -> None:
    """snapshot() → restore() → snapshot() produces identical state."""
    session = Session(bus=InProcessDispatcher())
    session[Item].seed(Item(name="initial"))

    for event in events:
        session.dispatch(event)

    snapshot1 = session.snapshot()

    # Create fresh session and restore
    session2 = Session(bus=InProcessDispatcher())
    session2.restore(snapshot1)

    snapshot2 = session2.snapshot()

    assert snapshot1 == snapshot2
```

### SliceOp Composition

```python
@given(
    st.lists(item_strategy, min_size=0, max_size=10),
    st.lists(item_strategy, min_size=1, max_size=5),
)
def test_extend_equals_sequential_appends(
    initial: list[Item], to_add: list[Item]
) -> None:
    """Extend([a, b, c]) produces same result as Append(a), Append(b), Append(c)."""
    # Via Extend
    session1 = Session(bus=InProcessDispatcher())
    session1[Item].seed(*initial)
    session1.dispatch(ExtendItems(to_add))

    # Via sequential Appends
    session2 = Session(bus=InProcessDispatcher())
    session2[Item].seed(*initial)
    for item in to_add:
        session2.dispatch(AppendItem(item))

    assert list(session1[Item].all()) == list(session2[Item].all())
```

### Prompt Key Normalization

```python
@given(st.text(max_size=100))
def test_normalize_idempotent(key: str) -> None:
    """Normalizing twice equals normalizing once."""
    once = normalize_component_key(key)
    twice = normalize_component_key(once)
    assert once == twice

@given(st.text(max_size=100))
def test_normalize_deterministic(key: str) -> None:
    """Same input always produces same output."""
    result1 = normalize_component_key(key)
    result2 = normalize_component_key(key)
    assert result1 == result2
```

## Settings and Configuration

### Default Settings

```python
# conftest.py or tests/property/conftest.py
from hypothesis import settings, Verbosity, Phase

# Register profile for CI (more examples, longer deadline)
settings.register_profile(
    "ci",
    max_examples=1000,
    deadline=None,  # Disable deadline in CI
    print_blob=True,
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
)

# Register profile for local development (faster feedback)
settings.register_profile(
    "dev",
    max_examples=100,
    deadline=1000,  # 1 second per example
    print_blob=True,
)

# Load profile from environment
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
```

### Per-Test Settings

```python
from hypothesis import settings, HealthCheck

@settings(
    max_examples=500,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=5000,  # 5 seconds for complex operations
)
@given(large_dataclass_strategy)
def test_complex_operation(instance: LargeDataclass) -> None:
    ...
```

### Health Check Guidelines

| Health Check | When to Suppress | Indicates |
|--------------|-----------------|-----------|
| `too_slow` | Strategy genuinely slow | Consider simplifying strategy |
| `data_too_large` | Large structures needed | Consider max_size limits |
| `filter_too_much` | Complex constraints | Rewrite strategy to generate valid directly |
| `large_base_example` | Intentional large examples | Document why needed |

Never suppress without documenting reason:

```python
@settings(
    # Suppress: Redis operations have network latency
    suppress_health_check=[HealthCheck.too_slow],
)
```

## CI Integration

### pytest Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
markers = [
    "property: Property-based tests (may be slow)",
]

[tool.hypothesis]
# Default profile settings
default_settings = { max_examples = 100, deadline = 1000 }
```

### Running Property Tests

```bash
# Run only property tests
uv run pytest tests/property/ -v

# Run with CI profile (more examples)
HYPOTHESIS_PROFILE=ci uv run pytest tests/property/ -v

# Run with verbose output showing generated examples
uv run pytest tests/property/ -v --hypothesis-verbosity=verbose

# Run specific property test file
uv run pytest tests/property/serde/test_roundtrip.py -v
```

### Coverage Integration

Property tests contribute to line coverage but may not catch all branches in a
single run. The 100% coverage requirement applies to the combined test suite:

```toml
[tool.coverage.run]
source = ["src/weakincentives"]
branch = true
# Property tests included in coverage
omit = []
```

### Mutation Testing

Property tests are particularly effective at catching surviving mutants because
they explore the input space more thoroughly:

```bash
# Run mutation testing including property tests
make mutation-check
```

Mutation thresholds remain:

- session: 90%
- serde: 85%
- dbc: 85%

## Database Management

### Hypothesis Database

Hypothesis stores failing examples in `.hypothesis/examples/`:

```
.hypothesis/
├── examples/          # Failing example database
│   └── ...           # Binary files with shrunk examples
└── unicode_data/     # Cached unicode categories
```

**Commit `.hypothesis/` to version control.** This preserves regression coverage
—once Hypothesis finds a failing input, it will always test that input first.

### gitignore

```gitignore
# Do NOT ignore .hypothesis - we want failing examples preserved
# .hypothesis/
```

## Writing Guidelines

### Do

- **State properties, not examples**: "for all valid inputs, X holds"
- **Use `assume()` sparingly**: Prefer strategies that generate valid inputs
- **Pin edge cases with `@example()`**: Empty strings, zero, max values
- **Name tests after the property**: `test_serde_roundtrip`, `test_normalize_idempotent`
- **Build reusable strategies**: Add to `tests/property/strategies/`
- **Document suppressed health checks**: Explain why suppression is necessary

### Don't

- **Don't disable shrinking**: Minimal examples are essential for debugging
- **Don't use `@settings(max_examples=1)`**: That's just an example test
- **Don't ignore `Flaky` errors**: They indicate real non-determinism
- **Don't hardcode seeds**: Use database for reproducibility
- **Don't test implementation details**: Test observable behavior properties

### Property Test Checklist

Before submitting property tests:

- [ ] Property name describes the invariant being tested
- [ ] Strategy generates diverse, valid inputs
- [ ] Edge cases pinned with `@example()` decorators
- [ ] Health check suppressions documented
- [ ] Strategies added to shared library if reusable
- [ ] Test runs under 30 seconds locally with dev profile
- [ ] Failing examples committed to `.hypothesis/`

## Module Coverage Targets

| Module | Property Category | Example Properties | Priority |
|--------|------------------|-------------------|----------|
| `serde/parse.py` | Round-trip | `parse(dump(x)) == x` | High |
| `serde/dump.py` | JSON safety | Output is JSON-serializable | High |
| `serde/schema.py` | Consistency | Schema matches type structure | Medium |
| `dataclasses/` | Immutability | `update()` preserves unchanged | High |
| `runtime/session/` | State machine | Operations preserve invariants | High |
| `runtime/session/snapshots.py` | Round-trip | `restore(snapshot(s)) == s` | High |
| `prompt/_normalization.py` | Idempotent | `f(f(x)) == f(x)` | Medium |
| `contrib/mailbox/` | Protocol | Message states exclusive | High |

## References

- [Hypothesis documentation](https://hypothesis.readthedocs.io/)
- [Hypothesis for the scientific stack](https://hypothesis.readthedocs.io/en/latest/numpy.html)
- [Stateful testing](https://hypothesis.readthedocs.io/en/latest/stateful.html)
- [Choosing properties](https://fsharpforfunandprofit.com/posts/property-based-testing-2/)
- `specs/TESTING.md` — Coverage requirements and test organization
- `specs/VERIFICATION.md` — Redis mailbox property tests example

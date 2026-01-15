# Formal Verification Specification

Embedded TLA+ specifications with bounded model checking.

**Source:** `src/weakincentives/formal/`

## Overview

- **Prevents spec-code drift**: Specs live with implementation
- **Bounded model checking**: TLC exhaustively verifies correctness
- **CI integration**: `make verify-formal`

## @formal_spec Decorator

```python
from weakincentives.formal import formal_spec, StateVar, Action, Invariant

@formal_spec(
    module="Counter",
    state_vars=[StateVar("count", "Nat")],
    actions=[
        Action(name="Increment", preconditions=("count < MaxValue",), updates={"count": "count + 1"}),
        Action(name="Decrement", preconditions=("count > 0",), updates={"count": "count - 1"}),
    ],
    invariants=[
        Invariant(id="INV-1", name="NonNegative", predicate="count >= 0"),
    ],
    constants={"MaxValue": 10},
    constraint="count <= 5",
)
class Counter: ...
```

## Types

### StateVar

```python
StateVar(name: str, type: str, description: str = "", initial_value: str | None = None)
```

### Action

```python
Action(
    name: str,
    parameters: Sequence[ActionParameter] = (),
    preconditions: Sequence[str] = (),
    updates: dict[str, str] = {},
)
```

### Invariant

```python
Invariant(id: str, name: str, predicate: str, description: str = "")
```

## Testing Utilities

```python
from weakincentives.formal.testing import extract_and_verify

spec, tla_file, cfg_file, result = extract_and_verify(
    Counter,
    output_dir=tmp_path,
    model_check_enabled=True,
    tlc_config={"workers": "auto"},
)

assert result.passed
```

## State Space Optimization

- Small constants: `MaxMessages: 2`, not `100`
- State constraints: `constraint="now <= 2"`
- Narrow parameter domains: `"1..3"` not `"1..10"`

## CI Integration

```bash
make verify-formal  # Runs pytest formal-tests/
```

60-second timeout per spec. Graceful skip if TLC not configured.

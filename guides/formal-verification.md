# Formal Verification with TLA+

WINK supports embedding TLA+ formal specifications directly in Python code using
the `@formal_spec` decorator. This approach prevents specification drift by
keeping specs co-located with implementation.

## Why Formal Verification?

For correctness-critical code like distributed message queues, testing alone
isn't sufficient. The `RedisMailbox` implementation, for example, must maintain
invariants like "each message exists in exactly one place" across all possible
interleavings of concurrent operations.

TLA+ model checking exhaustively explores these interleavings, catching subtle
bugs that randomized testing might miss.

## Quick Example

```python nocheck
from weakincentives.formal import formal_spec, StateVar, Action, Invariant

@formal_spec(
    module="Counter",
    state_vars=[
        StateVar("count", "Nat", "Current count value"),
    ],
    actions=[
        Action(
            name="Increment",
            preconditions=("count < MaxValue",),
            updates={"count": "count + 1"},
        ),
        Action(
            name="Decrement",
            preconditions=("count > 0",),
            updates={"count": "count - 1"},
        ),
    ],
    invariants=[
        Invariant("INV-1", "NonNegative", "count >= 0", "Count never goes negative"),
        Invariant("INV-2", "BelowMax", "count <= MaxValue", "Count never exceeds max"),
    ],
    constants={"MaxValue": 10},
    constraint="count <= 5",  # Limit state space exploration
)
class Counter:
    """Simple counter with formal spec."""

    def __init__(self):
        self.count = 0

    def increment(self):
        if self.count < 10:
            self.count += 1

    def decrement(self):
        if self.count > 0:
            self.count -= 1
```

## Running Verification

```python nocheck
# formal-tests/test_counter.py
from pathlib import Path
from weakincentives.formal.testing import extract_and_verify


def test_counter_spec(tmp_path: Path):
    """Extract and verify Counter TLA+ specification."""
    spec, tla_file, cfg_file, result = extract_and_verify(
        Counter,
        output_dir=tmp_path,
        model_check_enabled=True,
        tlc_config={"workers": "auto", "cleanup": True},
    )

    assert spec.module == "Counter"
    if result is not None:
        assert result.passed
        assert result.states_generated > 0
```

Run with:

```bash
make verify-formal  # Runs TLC model checker
```

## Key Concepts

**State variables** declare the TLA+ state space:

```python nocheck
StateVar("queue", "Seq(Message)", "Pending messages")
StateVar("inFlight", "[1..NumConsumers -> Seq(Message)]", "In-flight per consumer")
```

**Actions** define state transitions with preconditions and updates:

```python nocheck
Action(
    name="Receive",
    parameters=(ActionParameter("consumer", "1..NumConsumers"),),
    preconditions=("queue /= <<>>",),
    updates={
        "inFlight": "Append(inFlight[consumer], Head(queue))",
        "queue": "Tail(queue)",
    },
)
```

**Invariants** define safety properties that must always hold:

```python nocheck
Invariant("INV-1", "MessageExclusivity", "MessageInExactlyOnePlace(msg)")
Invariant("INV-2", "NoLostMessages", "CountMessages() = InitialMessageCount")
```

## State Space Management

The challenge with model checking is state space explosion. Strategies:

1. **Small constants**: Use `MaxMessages: 2` not `100`
1. **Tight constraints**: Add `constraint="now <= 2"` to bound exploration
1. **Narrow domains**: Use `"0..2"` not `"0..100"` for parameters

The RedisMailbox spec, for example, explores ~500K states in 60 seconds with
carefully chosen bounds.

## When to Use Formal Verification

**Use `@formal_spec` for:**

- Distributed algorithms (message queues, consensus)
- State machines with complex invariants
- Concurrent data structures
- Any code where "it works in testing" isn't enough

**Don't use it for:**

- Simple CRUD operations
- Stateless transformations
- Code where types + tests provide sufficient confidence

## Testing Utilities

```python nocheck
from weakincentives.formal.testing import (
    extract_spec,       # Extract FormalSpec from decorated class
    write_spec,         # Write .tla and .cfg files
    model_check,        # Run TLC model checker
    extract_and_verify  # Combined extraction + verification
)
```

## Installation

TLC must be installed for model checking:

```bash
# macOS
brew install tlaplus

# Linux
wget https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar
```

## Relationship to Design-by-Contract

Formal verification complements design-by-contract:

- **DbC decorators** (`@require`, `@ensure`) check conditions at runtime
- **TLA+ specs** explore all possible states at verification time

For safety-critical state machines, use both: DbC catches violations in
production, TLA+ proves the algorithm is correct.

## Next Steps

- [Code Quality](code-quality.md): Other quality mechanisms
- [specs/FORMAL_VERIFICATION.md](../specs/FORMAL_VERIFICATION.md): Complete API
  documentation
- [specs/VERIFICATION.md](../specs/VERIFICATION.md): RedisMailbox formal spec

# Appendix C: Formal Verification with TLA+

WINK supports embedding TLA+ formal specifications directly in Python code using the `@formal_spec` decorator. This approach prevents specification drift by keeping specs co-located with implementation.

## Why formal verification?

For correctness-critical code like distributed message queues, testing alone isn't sufficient. The `RedisMailbox` implementation, for example, must maintain invariants like "each message exists in exactly one place" across all possible interleavings of concurrent operations.

TLA+ model checking exhaustively explores these interleavings, catching subtle bugs that randomized testing might miss.

### The testing gap

Traditional testing approaches have limitations for concurrent systems:

| Approach | What it catches | What it misses |
| --- | --- | --- |
| **Unit tests** | Basic logic errors | Race conditions |
| **Integration tests** | Happy path scenarios | Edge cases in interleaving |
| **Property tests** | Random inputs | Rare concurrent states |
| **Stress tests** | Load issues | Timing-dependent bugs |
| **TLA+ model checking** | **All reachable states** | **Nothing in state space** |

TLA+ exhaustively explores the entire state space up to configured bounds, guaranteeing that invariants hold for all possible executions.

## Quick example

Here's a simple counter with formal verification:

```python
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

This decorator:

1. Embeds TLA+ specification alongside implementation
1. Documents state variables, actions, and invariants
1. Enables extraction and verification via TLC model checker
1. Keeps spec and code co-located to prevent drift

## Running verification

Create a test that extracts and verifies the spec:

```python
# formal-tests/test_counter.py
from pathlib import Path
from typing import Any
from weakincentives.formal.testing import extract_and_verify

# Counter class defined above with @formal_spec decorator
Counter: Any = ...  # type: ignore[assignment]


def test_counter_spec(tmp_path: Path) -> None:
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

Output shows state space exploration:

```
TLC2 Version 2.18 of Day Month 20XX
Running breadth-first search Model-Checking with fp 121 and seed -3789448849101184558 with 8 workers on 8 cores with 4096MB heap and 1024MB offheap memory [pid: 12345]
Starting... (2024-01-15 10:30:00)
Computed 2 states...
Computed 4 states...
Computed 8 states...
Finished. (2024-01-15 10:30:01)
Model checking completed. No error has been found.
  States generated: 16
  Distinct states: 11
  States left on queue: 0
```

## Key concepts

### State variables

State variables declare the TLA+ state space. They define what data the system tracks:

```python
StateVar("queue", "Seq(Message)", "Pending messages")
StateVar("inFlight", "[1..NumConsumers -> Seq(Message)]", "In-flight per consumer")
StateVar("delivered", "SUBSET Message", "Delivered message IDs")
```

**Common TLA+ types:**

- `Nat` - natural numbers (0, 1, 2, ...)
- `Int` - integers
- `BOOLEAN` - true/false
- `Seq(T)` - sequences of type T
- `SUBSET T` - sets of elements from T
- `[Domain -> Range]` - functions/maps

### Actions

Actions define state transitions with preconditions and updates:

```python
Action(
    name="Receive",
    parameters=(ActionParameter("consumer", "1..NumConsumers"),),
    preconditions=("queue /= <<>>",),  # Queue not empty
    updates={
        "inFlight": "Append(inFlight[consumer], Head(queue))",
        "queue": "Tail(queue)",
    },
)
```

**Action components:**

- **name**: Action identifier (must be valid TLA+ identifier)
- **parameters**: Optional tuple of `ActionParameter` for quantified actions
- **preconditions**: Tuple of TLA+ expressions that must hold
- **updates**: Dict mapping state var names to new values

### Invariants

Invariants define safety properties that must always hold:

```python
Invariant(
    "INV-1",
    "MessageExclusivity",
    "MessageInExactlyOnePlace(msg)",
    "Each message exists in exactly one place"
)
Invariant(
    "INV-2",
    "NoLostMessages",
    "CountMessages() = InitialMessageCount",
    "Total messages never changes"
)
```

**Invariant components:**

- **id**: Short identifier (e.g., "INV-1")
- **name**: Descriptive name
- **formula**: TLA+ expression that must be true in all states
- **description**: Human-readable explanation

If any invariant is violated in any reachable state, TLC reports the error with a trace showing how to reproduce it.

### Helper operators

Complex invariants often need helper operators:

```python
@formal_spec(
    module="Mailbox",
    helpers=[
        HelperOperator(
            "MessageInExactlyOnePlace",
            "(msg \\in Message)",
            "MessageInQueue(msg) + MessageInFlight(msg) + MessageDelivered(msg) = 1",
            "Check if message exists in exactly one location",
        ),
        HelperOperator(
            "MessageInQueue",
            "(msg \\in Message)",
            "IF \\E i \\in 1..Len(queue): queue[i] = msg THEN 1 ELSE 0",
            "Count message in queue (0 or 1)",
        ),
    ],
    invariants=[
        Invariant(
            "INV-1",
            "Exclusivity",
            "\\A msg \\in Message: MessageInExactlyOnePlace(msg)",
        ),
    ],
)
```

Helper operators:

- Decompose complex invariants into reusable pieces
- Improve readability of TLA+ specs
- Can be recursively defined

## State space management

The challenge with model checking is **state space explosion**. A system with 3 boolean flags has 2³ = 8 states. Add more variables and the state space grows exponentially.

### Strategies to manage state space

#### 1. Small constants

Use small values for constants to bound exploration:

```python
constants={
    "MaxMessages": 2,        # Not 100
    "NumConsumers": 2,       # Not 10
    "MaxRetries": 1,         # Not 5
}
```

**Trade-off**: Smaller constants mean fewer states but might miss bugs that only appear with larger values. Start small, then increase if verification passes quickly.

#### 2. Tight constraints

Add state constraints to limit what states TLC explores:

```python
constraint="Len(queue) + Len(inFlight[1]) + Len(inFlight[2]) <= 3"
```

This tells TLC: "Don't explore states where total messages exceed 3." Valid executions won't violate this, but it dramatically reduces state space.

#### 3. Narrow domains

Use bounded ranges for parameters:

```python
Action(
    name="Retry",
    parameters=(ActionParameter("msg", "1..MaxMessages"),),  # Not all Nat
    # ...
)
```

### Real-world example

The RedisMailbox spec explores ~500K states in 60 seconds with:

```python
constants={
    "MaxMessages": 2,
    "NumConsumers": 2,
    "MaxVisibilityTimeout": 2,
}
constraint="now <= 4"  # Bound time progression
```

This configuration is sufficient to catch subtle bugs like:

- Race conditions between receive and delete
- Visibility timeout edge cases
- Message loss scenarios

## When to use formal verification

### Use `@formal_spec` for:

**Distributed algorithms**

- Message queues
- Consensus protocols
- Distributed locks
- Leader election

**State machines with complex invariants**

- Transaction managers
- Workflow engines
- Resource allocators

**Concurrent data structures**

- Lock-free queues
- Reference counting
- Copy-on-write structures

**Any code where "it works in testing" isn't enough**

- Financial transactions
- Safety-critical systems
- Security-sensitive operations

### Don't use it for:

**Simple CRUD operations**

- Basic database queries
- REST API handlers
- File I/O

**Stateless transformations**

- Data parsing
- Formatting
- Pure functions without complex logic

**Code where types + tests provide sufficient confidence**

- Business logic with simple invariants
- UI components
- One-off scripts

### Cost-benefit analysis

| System Complexity | State Space Size | Verification Cost | Benefit |
| --- | --- | --- | --- |
| Simple counter | Small (~10 states) | Seconds | Low (tests sufficient) |
| Message queue | Medium (~500K states) | Minutes | **High** (catches race conditions) |
| Consensus protocol | Large (millions) | Hours | **Very high** (critical correctness) |
| Monolithic system | Enormous | Infeasible | N/A (must decompose) |

Focus formal verification on the **correctness-critical core** of your system, not the entire application.

## Testing utilities

The `weakincentives.formal.testing` module provides utilities for extracting and verifying specs:

```python
from weakincentives.formal.testing import (
    extract_spec,      # Extract FormalSpec from decorated class
    write_spec,        # Write .tla and .cfg files
    model_check,       # Run TLC model checker
    extract_and_verify # Combined extraction + verification
)
```

### extract_spec

Extract the `FormalSpec` from a decorated class:

```python
spec = extract_spec(Counter)
assert spec.module == "Counter"
assert len(spec.state_vars) == 1
assert len(spec.actions) == 2
assert len(spec.invariants) == 2
```

### write_spec

Write extracted spec to `.tla` and `.cfg` files:

```python
tla_file, cfg_file = write_spec(spec, output_dir=tmp_path)
assert tla_file.exists()
assert cfg_file.exists()
```

### model_check

Run TLC model checker on generated files:

```python
result = model_check(
    tla_file,
    cfg_file,
    workers="auto",      # Use all CPU cores
    cleanup=True,        # Delete temp files
)

assert result.passed
assert result.states_generated > 0
assert result.duration_seconds > 0
```

### extract_and_verify

Combined extraction and verification (recommended):

```python
spec, tla_file, cfg_file, result = extract_and_verify(
    Counter,
    output_dir=tmp_path,
    model_check_enabled=True,
    tlc_config={"workers": "auto", "cleanup": True},
)

# Spec extracted
assert spec.module == "Counter"

# Files written
assert tla_file.exists()
assert cfg_file.exists()

# Verification succeeded
if result is not None:
    assert result.passed
```

## Installation

TLC (the TLA+ model checker) must be installed separately:

### macOS

```bash
brew install tlaplus
```

This installs:

- `tlc2` command for model checking
- `tla2tools.jar` Java library

### Linux

Download from GitHub releases:

```bash
wget https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar
# Add to PATH or use full path in tests
```

Or use package manager if available:

```bash
# Debian/Ubuntu (if packaged)
apt-get install tlaplus-tools
```

### Verify installation

```bash
tlc2 -h
# Should print TLC usage information
```

### CI integration

GitHub Actions example:

```yaml
- name: Install TLA+ tools
  run: |
    wget https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar
    echo "TLC_JAR=$PWD/tla2tools.jar" >> $GITHUB_ENV

- name: Run formal verification
  run: make verify-formal
```

## Advanced topics

### Modeling time

For systems with timeouts, model time explicitly:

```python
StateVar("now", "Nat", "Current time"),
StateVar("visibilityDeadlines", "[1..NumConsumers -> Nat]", "Timeout per consumer"),

Action(
    name="Tick",
    preconditions=("now < MaxTime",),
    updates={"now": "now + 1"},
)
```

Then use `constraint="now <= 5"` to bound time progression.

### Symmetry reduction

If consumers are interchangeable, declare symmetry to reduce state space:

```python
@formal_spec(
    module="Mailbox",
    symmetry_sets=[
        "Consumers",  # Permutations of consumers are equivalent
    ],
    # ...
)
```

TLC explores only one representative from each symmetry class.

### Liveness properties

Beyond invariants (safety), check liveness (eventual progress):

```python nocheck
@formal_spec(
    module="Mailbox",
    temporal_properties=[
        TemporalProperty(
            "EventualDelivery",
            "[]<>(Len(queue) = 0)",  # Eventually always: queue becomes empty
        ),
    ],
)
```

**Caveat**: Liveness checking is more expensive and may not terminate.

## Diagram: Model checking workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    Python Code with @formal_spec            │
│                                                             │
│  @formal_spec(                                              │
│      module="Counter",                                      │
│      state_vars=[...],                                      │
│      actions=[...],                                         │
│      invariants=[...],                                      │
│  )                                                          │
│  class Counter: ...                                         │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   │ extract_spec()
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                     FormalSpec (Python object)              │
│                                                             │
│  FormalSpec(                                                │
│      module="Counter",                                      │
│      state_vars=[StateVar(...)],                            │
│      actions=[Action(...)],                                 │
│      invariants=[Invariant(...)],                           │
│  )                                                          │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   │ write_spec()
                   ▼
┌──────────────────────────────┐  ┌──────────────────────────┐
│      Counter.tla             │  │    Counter.cfg           │
│  (TLA+ specification)        │  │  (TLC configuration)     │
│                              │  │                          │
│  ---- MODULE Counter ----    │  │  CONSTANT MaxValue = 10  │
│  VARIABLES count             │  │  CONSTRAINT count <= 5   │
│  Init == count = 0           │  │  INVARIANT NonNegative   │
│  Increment == ...            │  │  INVARIANT BelowMax      │
│  ==========================  │  │                          │
└──────────────┬───────────────┘  └────────┬─────────────────┘
               │                           │
               │         model_check()     │
               └─────────────┬─────────────┘
                             ▼
                   ┌─────────────────────┐
                   │    TLC Model Checker │
                   │   (Java application) │
                   └──────────┬───────────┘
                              │
                              │ Explores state space
                              ▼
                   ┌─────────────────────┐
                   │  Verification Result │
                   │                     │
                   │  ✓ Passed           │
                   │  States: 11         │
                   │  Duration: 0.5s     │
                   └─────────────────────┘
```

## Real-world example: RedisMailbox

The `RedisMailbox` implementation uses formal verification to ensure correctness. Key invariants:

1. **Message exclusivity**: Each message exists in exactly one place (queue, in-flight, or delivered)
1. **No message loss**: Total message count never changes
1. **Visibility timeout semantics**: In-flight messages become visible again after timeout
1. **Delete idempotence**: Deleting an already-deleted message is safe

The spec found bugs during development:

- Race condition between receive and visibility timeout
- Edge case in delete-after-timeout scenario
- Missing null check in retry logic

Without TLC, these bugs would likely have reached production, causing silent message loss.

## Summary

Formal verification with TLA+ provides **exhaustive state space exploration**, catching subtle bugs that testing misses. WINK's `@formal_spec` decorator:

- Keeps specifications co-located with implementation
- Prevents spec drift
- Integrates with pytest for automated verification
- Uses standard TLA+ syntax for industry-standard verification

Use it for correctness-critical components where the cost of bugs is high and the state space is manageable. For everything else, comprehensive testing is sufficient.

## Further reading

- [specs/FORMAL_VERIFICATION.md](../specs/FORMAL_VERIFICATION.md) - Complete API documentation
- [TLA+ homepage](https://lamport.azurewebsites.net/tla/tla.html) - Learn TLA+ from Leslie Lamport
- [Learn TLA+](https://learntla.com/) - Interactive TLA+ tutorial
- [Practical TLA+](https://www.hillelwayne.com/post/practical-tla/) - Hillel Wayne's book

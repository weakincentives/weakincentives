# Formal Verification with Embedded TLA+

## Overview

WINK supports embedding formal specifications directly in Python code using TLA+
(Temporal Logic of Actions). This approach:

- **Prevents spec-code drift** - Specs live next to implementation
- **Enables bounded model checking** - TLC exhaustively verifies correctness
- **Simplifies workflow** - No separate `.tla` files to maintain
- **Provides high confidence** - Catch algorithm bugs before production

The framework consists of three components:

1. **`@formal_spec` decorator** - Embed TLA+ metadata in Python classes
2. **Test utilities** - Extract specs and run TLC model checker
3. **CI integration** - Automated verification via `make verify-formal`

## Quick Start

### 1. Add `@formal_spec` to Your Class

```python
from weakincentives.formal import formal_spec, StateVar, Action, Invariant, ActionParameter

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
        Invariant(
            id="INV-1",
            name="NonNegative",
            predicate="count >= 0",
            description="Count never goes negative",
        ),
        Invariant(
            id="INV-2",
            name="BelowMax",
            predicate="count <= MaxValue",
            description="Count never exceeds maximum",
        ),
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

### 2. Create Formal Verification Test

```python
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

    # Assertions
    assert spec.module == "Counter"
    assert result.passed
    assert result.states_generated > 0
```

### 3. Run Verification

```bash
# Run formal verification tests
make verify-formal

# Or run directly with pytest
pytest formal-tests/ -v
```

## API Reference

### `@formal_spec` Decorator

Attaches TLA+ specification metadata to a Python class.

**Parameters:**

- `module: str` - TLA+ module name (required)
- `state_vars: Sequence[StateVar]` - State variables (default: `()`)
- `actions: Sequence[Action]` - Actions/transitions (default: `()`)
- `invariants: Sequence[Invariant]` - Safety properties (default: `()`)
- `constants: dict[str, int | str]` - Model constants (default: `{}`)
- `constraint: str | None` - State constraint to limit exploration (default: `None`)
- `extends: Sequence[str]` - TLA+ modules to extend (default: `("Integers", "Sequences", "FiniteSets")`)

**Returns:** The decorated class with `__formal_spec__` attribute

### `StateVar`

Defines a TLA+ state variable.

```python
@dataclass(frozen=True)
class StateVar:
    name: str                    # Variable name
    type: str                    # TLA+ type (e.g., "Nat", "Seq(Message)", "Set(Int)")
    description: str = ""        # Human-readable description
    initial_value: str | None = None  # Override type-based default
```

**Type-based initial values:**
- `Nat`, `Int` → `0`
- `Seq(...)` → `<<>>`
- `Set(...)` → `{}`
- `Function` → `[x \in {} |-> 0]`
- Custom types → `NULL`

### `Action`

Defines a TLA+ action (state transition).

```python
@dataclass(frozen=True)
class Action:
    name: str                                # Action name
    parameters: Sequence[ActionParameter] = ()  # Bounded parameters
    preconditions: Sequence[str] = ()        # Enabling conditions
    updates: dict[str, str] = field(default_factory=dict)  # State updates
    description: str = ""                    # Human-readable description
```

**Example with parameters:**

```python
Action(
    name="Receive",
    parameters=(
        ActionParameter("consumer", "1..NumConsumers"),
        ActionParameter("timeout", "0..MaxTimeout"),
    ),
    preconditions=(
        "messages /= <<>>",
        "Len(inFlight[consumer]) < MaxInFlight",
    ),
    updates={
        "inFlight": "Append(inFlight[consumer], Head(messages))",
        "messages": "Tail(messages)",
        "now": "now + timeout",
    },
)
```

### `ActionParameter`

Defines a bounded parameter for action quantification.

```python
@dataclass(frozen=True)
class ActionParameter:
    name: str     # Parameter name
    domain: str   # TLA+ domain expression (e.g., "1..N", "Messages", "{1, 2, 3}")
```

### `Invariant`

Defines a TLA+ invariant (safety property).

```python
@dataclass(frozen=True)
class Invariant:
    id: str           # Unique identifier (e.g., "INV-1")
    name: str         # Invariant name for TLA+
    predicate: str    # TLA+ boolean expression
    description: str = ""  # Human-readable description
```

## Testing Utilities

### `extract_spec()`

Extract formal specification from decorated class.

```python
from weakincentives.formal.testing import extract_spec

spec = extract_spec(RedisMailbox)
assert spec.module == "RedisMailbox"
```

### `write_spec()`

Write TLA+ specification to files.

```python
from weakincentives.formal.testing import write_spec
from pathlib import Path

spec = extract_spec(Counter)
tla_file, cfg_file = write_spec(spec, Path("output"))

# Creates:
#   output/Counter.tla  - TLA+ module
#   output/Counter.cfg  - TLC configuration
```

### `model_check()`

Run TLC model checker on specification.

```python
from weakincentives.formal.testing import model_check

spec = extract_spec(Counter)
result = model_check(
    spec,
    tlc_config={
        "workers": "auto",  # Parallel workers ("auto" or "1"-"16")
        "cleanup": True,    # Clean up state files after checking
    }
)

assert result.passed
assert result.states_generated > 0
```

**Timeout behavior:**
- TLC runs with a 60-second timeout
- Timeout **without violations** → Pass (bounded verification)
- Timeout **with violations** → Fail
- Configuration errors → Raises `ModelCheckError`

### `extract_and_verify()`

Extract spec and optionally run model checking (main test entry point).

```python
from weakincentives.formal.testing import extract_and_verify
from pathlib import Path

spec, tla_file, cfg_file, result = extract_and_verify(
    RedisMailbox,
    output_dir=Path("output"),
    model_check_enabled=True,
    tlc_config={"workers": "auto", "cleanup": True},
)

assert result.passed
```

**Parameters:**
- `target_class` - Class decorated with `@formal_spec`
- `output_dir` - Directory for `.tla` and `.cfg` files
- `model_check_enabled` - Whether to run TLC (default: `False`)
- `tlc_config` - TLC configuration dict (default: `None`)

**Returns:** `(spec, tla_file, cfg_file, result)`

## Example: RedisMailbox Verification

The `RedisMailbox` class demonstrates a complete formal verification workflow
for a distributed message queue with visibility timeouts.

### Specification Highlights

**State variables:**
```python
state_vars=[
    StateVar("queue", "Seq(Message)", "Pending messages"),
    StateVar("inFlight", "[1..NumConsumers -> Seq(Message)]", "In-flight per consumer"),
    StateVar("deadLetters", "Seq(Message)", "Failed messages"),
    StateVar("deliveries", "[Message -> Nat]", "Delivery attempts"),
    StateVar("now", "Nat", "Current logical time"),
    # ...
]
```

**Actions with bounded parameters:**
```python
Action(
    name="Receive",
    parameters=(ActionParameter("consumer", "1..NumConsumers"),),
    preconditions=("queue /= <<>>",),
    updates={
        "inFlight": "Append(inFlight[consumer], Head(queue))",
        "queue": "Tail(queue)",
        "deliveries": "[deliveries EXCEPT ![Head(queue)] = @ + 1]",
    },
),
```

**Invariants:**
```python
Invariant(
    id="INV-1",
    name="MessageExclusivity",
    predicate="MessageInExactlyOnePlace(msg) FOR ALL msg",
    description="Each message in exactly one place",
),
Invariant(
    id="INV-2",
    name="NoLostMessages",
    predicate="CountMessages() = InitialMessageCount",
    description="Messages never lost",
),
```

### State Space Optimization

**Problem:** Full parameterization → 2M+ states (too slow)

**Solution:** Careful domain sizing

```python
# ❌ Too large (5 timeout values × 4 timeout values = 20 combinations)
ActionParameter("newTimeout", "0..VisibilityTimeout*2")

# ✅ Optimized (3 timeout values)
ActionParameter("newTimeout", "0..VisibilityTimeout")
```

**Constraint:** Limit exploration depth
```python
constraint="now <= 2"  # Bound logical time
```

**Constants:** Small values for bounded checking
```python
constants={
    "MaxMessages": 2,
    "NumConsumers": 2,
    "VisibilityTimeout": 2,
}
```

**Result:** ~500K states, completes in 60s

### Test Structure

```python
# formal-tests/test_redis_mailbox.py
def test_redis_mailbox_spec(
    extracted_specs_dir: Path,
    enable_model_checking: bool,
    tlc_config: dict[str, str | bool],
):
    """Extract and verify RedisMailbox TLA+ specification."""
    try:
        spec, tla_file, cfg_file, result = extract_and_verify(
            RedisMailbox,
            output_dir=extracted_specs_dir,
            model_check_enabled=enable_model_checking,
            tlc_config=tlc_config if enable_model_checking else None,
        )
    except ModelCheckError as e:
        # Skip if TLC not properly configured
        if "configuration error" in str(e).lower():
            pytest.skip(f"TLC not properly configured: {e}")
        raise

    # Verify spec structure
    assert spec.module == "RedisMailbox"
    assert len(spec.state_vars) == 11
    assert len(spec.actions) == 10
    assert len(spec.invariants) == 6

    # Verify model checking passed
    if enable_model_checking:
        assert result is not None
        assert result.passed
```

## Generated TLA+ Structure

The `@formal_spec` decorator generates complete TLA+ modules:

```tla
---- MODULE Counter ----
EXTENDS Integers, Sequences, FiniteSets

CONSTANTS MaxValue

VARIABLES count

vars == <<count>>

TypeInvariant ==
    count \in Nat

Init ==
    count = 0

Increment ==
    /\ count < MaxValue
    /\ count' = count + 1

Decrement ==
    /\ count > 0
    /\ count' = count - 1

Next ==
    \/ Increment
    \/ Decrement

Spec ==
    Init /\ [][Next]_vars

(* Invariants *)
INV-1 == count >= 0
NonNegative == INV-1

INV-2 == count <= MaxValue
BelowMax == INV-2

(* State Constraint *)
StateConstraint == count <= 5

=============================================================================
```

And TLC configuration:

```
SPECIFICATION Spec

CONSTANTS
    MaxValue = 10

INVARIANTS
    NonNegative
    BelowMax

CONSTRAINT StateConstraint

CHECK_DEADLOCK FALSE
```

## Advanced Topics

### Modeling Time

Use a logical clock for timeout-based operations:

```python
StateVar("now", "Nat", "Logical time"),

Action(
    name="AdvanceTime",
    updates={"now": "now + 1"},
),

# Timeout check
Action(
    name="ReapExpired",
    preconditions=(
        "\\E msg \\in inFlight: expiry[msg] <= now",
    ),
    updates={
        "inFlight": "RemoveExpired(inFlight, now)",
    },
),
```

### Bounding State Space

**Constraint on exploration depth:**
```python
constraint="now <= 5"  # Limit time-based exploration
```

**Small constants:**
```python
constants={
    "MaxMessages": 2,      # Not 100
    "NumWorkers": 2,       # Not 16
    "BufferSize": 3,       # Not 1024
}
```

**Careful parameter domains:**
```python
# Each additional value multiplies state space
ActionParameter("priority", "1..3")  # Not "1..10"
```

### Custom Operators

Define helper operators in the spec:

```python
@formal_spec(
    module="Queue",
    operators={
        "IsEmpty": "queue = <<>>",
        "IsFull": "Len(queue) >= MaxSize",
        "Contains(msg)": "msg \\in Range(queue)",
    },
    # ...
)
```

### Multiple Instances

Test different configurations:

```python
@pytest.mark.parametrize("max_msg,workers", [(2, 2), (3, 1)])
def test_redis_mailbox_configs(max_msg, workers, tmp_path):
    # Dynamically update constants
    spec = extract_spec(RedisMailbox)
    spec.constants["MaxMessages"] = max_msg
    spec.constants["NumConsumers"] = workers

    result = model_check(spec)
    assert result.passed
```

## CI Integration

Formal verification runs automatically via `make verify-formal`:

```makefile
verify-formal:
    pytest formal-tests/ -v --tb=short
```

**Environment requirements:**
- TLC model checker installed (`brew install tlaplus` on macOS)
- 60-second timeout per spec
- Graceful skip if TLC not configured

**GitHub Actions:**
```yaml
- name: Run formal verification
  run: make verify-formal
  continue-on-error: true  # Optional: don't block on TLC issues
```

## Troubleshooting

### TLC Not Found

**Error:** `ModelCheckError: TLC not found`

**Solution:**
```bash
# macOS
brew install tlaplus

# Linux/WSL
wget https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar
sudo mv tla2tools.jar /usr/local/lib/
echo '#!/bin/bash' | sudo tee /usr/local/bin/tlc
echo 'java -jar /usr/local/lib/tla2tools.jar "$@"' | sudo tee -a /usr/local/bin/tlc
sudo chmod +x /usr/local/bin/tlc
```

### TLC Configuration Error

**Error:** `Unable to access jarfile /usr/local/lib/tla2tools.jar`

This means `tlc` command exists but points to missing/wrong JAR file. Tests
will skip gracefully with appropriate message.

### State Space Explosion

**Symptom:** TLC runs for 60s and times out

**Solutions:**
1. Reduce constant values (`MaxMessages: 10` → `MaxMessages: 2`)
2. Add state constraint (`constraint="depth <= 5"`)
3. Narrow parameter domains (`"0..10"` → `"0..2"`)
4. Reduce number of actions modeled

### Invariant Violations

**Symptom:** `Model checking failed: Invariant violated`

**Debugging:**
1. Check TLC output for counterexample trace
2. Add intermediate invariants to narrow down issue
3. Add `ASSUME` statements for expected constraints
4. Review action preconditions

## Best Practices

1. **Start small** - Model 2-3 core actions first, expand gradually
2. **Tight bounds** - Use smallest constants that still test interesting behavior
3. **Clear invariants** - Each invariant tests one specific property
4. **Meaningful names** - Action/invariant names match domain concepts
5. **Document constraints** - Explain why constants/domains are bounded
6. **Verify incrementally** - Add one action at a time, verify each step
7. **Test the tests** - Intentionally violate invariants to verify detection

## See Also

- [TLA+ Video Course](https://lamport.azurewebsites.net/video/videos.html) - Learn TLA+
- [TLA+ Examples](https://github.com/tlaplus/Examples) - Real-world specs
- [specs/VERIFICATION.md](VERIFICATION.md) - RedisMailbox detailed spec
- [src/weakincentives/formal/](../src/weakincentives/formal/) - Implementation
- [formal-tests/](../formal-tests/) - Test examples

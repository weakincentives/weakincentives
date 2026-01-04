# TLA+ Embedding in Python: Implementation Summary

## What Was Implemented

This implementation provides a complete solution for embedding TLA+ formal
specifications directly in Python code, allowing specs and implementation to
co-exist and reducing drift.

### Key Components

1. **`weakincentives.formal` module** - Core metadata classes and decorator
2. **`pytest_tla` plugin** - Extraction and validation tooling
3. **Example code** - Demonstrations of usage patterns
4. **Documentation** - User guide and design rationale

## Files Created

```
src/weakincentives/formal/__init__.py      # Core formal spec support
tests/plugins/pytest_tla.py                # Pytest plugin for extraction
examples/formal_spec_example.py            # Complete working example
examples/simple_tla_demo.py                # Minimal demo
tests/formal/test_tla_extraction.py        # Unit tests
specs/TLA_EMBEDDING.md                     # Design exploration (5 approaches)
docs/FORMAL_SPEC_GUIDE.md                  # User guide
```

## How It Works

### 1. Decorate Your Class

Add `@formal_spec` to embed TLA+ metadata:

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
            updates={"count": "count + 1"},
        ),
    ],
    invariants=[
        Invariant("INV-1", "NonNegative", "count >= 0"),
    ],
)
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1
```

### 2. Extract TLA+ Specs

Run pytest to extract:

```bash
# Extract specs to specs/tla/extracted/
pytest --extract-tla

# Extract and validate with TLC
pytest --check-tla
```

### 3. Generated TLA+ Output

The decorator generates complete TLA+ modules:

```tla
------------------------ MODULE Counter ------------------------
(* Generated from Python formal specification metadata *)

EXTENDS Integers, Sequences, FiniteSets

VARIABLES
    count  \* Current count value

vars == <<count>>

-----------------------------------------------------------------------------
(* Actions *)

Increment ==
    /\ count' = count + 1

-----------------------------------------------------------------------------
(* Invariants *)

(* INV-1 *)
NonNegative ==
    count >= 0

=============================================================================
```

And TLC configuration files:

```
SPECIFICATION Spec

INVARIANTS
    NonNegative
```

## Design Rationale

After exploring 5 different approaches (see `specs/TLA_EMBEDDING.md`), the
**structured metadata approach** was chosen because:

1. **Type-safe**: Python dataclasses with IDE support
2. **Validatable**: Can check metadata schema at import time
3. **Composable**: Works alongside existing DbC decorators
4. **Extractable**: Clear mapping from metadata to TLA+
5. **Gradual**: Easy to adopt incrementally

### Alternative Approaches Considered

| Approach                   | Pros                | Cons                            | Verdict               |
| -------------------------- | ------------------- | ------------------------------- | --------------------- |
| **1. Docstring blocks**    | Easy, no new syntax | Mixes concerns, hard to parse   | Supplement only       |
| **2. String decorators**   | Familiar            | No validation, poor IDE support | Too brittle           |
| **3. Python DSL**          | Type-safe           | Huge implementation effort      | Overengineered        |
| **4. DbC + TLA fragments** | Minimal change      | Duplication, incomplete specs   | Good for simple cases |
| **5. Structured metadata** | ✅ Best balance     | Requires new API                | **✅ CHOSEN**         |

## Architecture

### `FormalSpec` Dataclass

Captures complete TLA+ module:

- **Module name**: TLA+ identifier
- **State variables**: `StateVar(name, type, description)`
- **Actions**: `Action(name, parameters, preconditions, updates)`
- **Invariants**: `Invariant(id, name, predicate, description)`
- **Constants**: Default values for model checking
- **Helpers**: Raw TLA+ operator definitions

Methods:

- `to_tla()` → Generate TLA+ module text
- `to_tla_config()` → Generate TLC configuration

### Pytest Plugin Flow

```
1. pytest --check-tla
2. Plugin scans src/weakincentives/**/*.py
3. Finds classes with __formal_spec__ attribute
4. Extracts FormalSpec instances
5. Calls spec.to_tla() for each
6. Writes to specs/tla/extracted/
7. Runs TLC model checker on each spec
8. Reports errors or success
```

## Integration with Existing DbC System

The formal spec system **complements** the existing DbC decorators:

```python
from weakincentives.dbc import require, ensure, invariant
from weakincentives.formal import formal_spec, StateVar, Invariant

@formal_spec(
    module="Account",
    state_vars=[StateVar("balance", "Nat")],
    invariants=[Invariant("INV-1", "NonNegative", "balance >= 0")],
)
@invariant(lambda self: self.balance >= 0)  # Runtime DbC check
class Account:
    @require(lambda self, amount: amount > 0)  # Runtime precondition
    def withdraw(self, amount: int) -> int:
        self.balance -= amount
        return self.balance
```

**Division of labor:**

- `@formal_spec`: High-level state machine semantics for TLC
- `@require/@ensure/@invariant`: Runtime validation for tests
- Both can coexist and provide complementary coverage

## Usage Examples

### Example 1: Simple Counter

See `examples/formal_spec_example.py::Counter`

```bash
uv run python -m examples.formal_spec_example
```

### Example 2: Mailbox Receive Operation

See `examples/formal_spec_example.py::SimpleMailbox`

Demonstrates:

- Multiple state variables
- Actions with preconditions
- Multiple invariants
- Helper operators

### Example 3: Running Extraction

```bash
# Extract all specs
pytest --extract-tla

# Check specific module
pytest --extract-tla examples/formal_spec_example.py

# Model check with TLC
pytest --check-tla

# Use custom output directory
pytest --extract-tla --tla-output-dir=/tmp/specs
```

## Testing

Unit tests in `tests/formal/test_tla_extraction.py`:

```bash
uv run pytest tests/formal/test_tla_extraction.py -v
```

Tests cover:

- `FormalSpec.to_tla()` generation
- `FormalSpec.to_tla_config()` generation
- Decorator attachment (`__formal_spec__`)
- Actions with parameters
- Multi-line predicates
- Helper operators
- UNCHANGED variable tracking

## Next Steps

### Immediate (Proof of Concept)

1. ✅ Implement `weakincentives.formal` module
2. ✅ Create pytest plugin
3. ✅ Write examples
4. ✅ Document usage
5. ⏳ Migrate one existing spec (RedisMailbox) as proof

### Short Term (Productionization)

1. ⏳ Register pytest plugin in `pyproject.toml`
2. ⏳ Add `make extract-tla` and `make check-tla` targets
3. ⏳ Integrate into CI pipeline
4. ⏳ Migrate `specs/tla/RedisMailbox.tla` → `@formal_spec` decorator
5. ⏳ Update `AGENTS.md` and `CLAUDE.md`

### Long Term (Enhancement)

1. ⏳ Add support for temporal properties (liveness)
2. ⏳ Generate Hypothesis property tests from invariants
3. ⏳ Support PlusCal algorithm syntax
4. ⏳ Integration with TLA+ Toolbox / VS Code extension
5. ⏳ Auto-generate DbC decorators from TLA+ preconditions

## Migration Path for Existing Specs

### Current State

```
specs/tla/RedisMailbox.tla        # 370 lines of TLA+
src/.../mailbox/_redis.py         # Python implementation
```

### Target State

```python
# src/.../mailbox/_redis.py

@formal_spec(
    module="RedisMailbox",
    state_vars=[
        StateVar("pending", "Seq(MessageId)", "Pending message queue"),
        StateVar("invisible", "Function", "Invisible messages"),
        StateVar("handles", "Function", "Valid receipt handles"),
        StateVar("deliveryCounts", "Function", "Delivery counts"),
    ],
    actions=[
        Action(
            name="Send",
            parameters=("msgId", "body"),
            preconditions=("msgId \\notin DOMAIN data",),
            updates={
                "pending": "Append(pending, msgId)",
                "data": "data @@ (msgId :> body)",
            },
        ),
        Action(
            name="Receive",
            # ... (see example)
        ),
        # ... more actions
    ],
    invariants=[
        Invariant("INV-1", "MessageStateExclusive", r"\A msgId: ..."),
        Invariant("INV-2", "HandleFreshness", r"\A msgId: ..."),
        # ... more invariants
    ],
)
class RedisMailbox:
    ...
```

### Migration Steps

1. Create `@formal_spec` decorator skeleton
2. Copy constants → `constants={...}`
3. Copy VARIABLES → `state_vars=[...]`
4. Copy each action → `Action(...)`
5. Copy invariants → `Invariant(...)`
6. Extract specs: `pytest --extract-tla`
7. Compare with original: `diff specs/tla/RedisMailbox.tla specs/tla/extracted/RedisMailbox.tla`
8. Iterate until identical
9. Run model checker: `pytest --check-tla`
10. Delete original `specs/tla/RedisMailbox.tla`

## Configuration Options

### `pyproject.toml`

```toml
[tool.pytest.ini_options]
# TLA+ extraction settings
tla_extraction_enabled = true
tla_output_dir = "specs/tla/extracted"
tla_run_model_checker = true
tla_model_checker = "tlc"  # or "apalache"

# Make TLA+ checking part of default test run
addopts = "--extract-tla"
```

### Makefile Integration

```makefile
.PHONY: extract-tla
extract-tla:  ## Extract TLA+ specs from @formal_spec decorators
	uv run pytest --extract-tla

.PHONY: check-tla
check-tla:  ## Extract and model check TLA+ specs
	uv run pytest --check-tla

.PHONY: verify
verify: check-tla  ## Full formal verification
	@echo "All formal verification checks passed"
```

## Limitations and Trade-offs

### Current Limitations

1. **No temporal operators**: Can't express `◇` (eventually) or `□` (always) yet
2. **No fairness**: Can't specify weak/strong fairness constraints
3. **No refinement**: Can't express refinement mappings
4. **Static extraction**: Specs must be defined at class definition time

### Design Trade-offs

| Trade-off               | Choice    | Rationale                            |
| ----------------------- | --------- | ------------------------------------ |
| Metadata vs Raw TLA+    | Metadata  | Type-safe, validatable, IDE-friendly |
| Decorator vs Docstring  | Decorator | Structured, extractable, composable  |
| Full DSL vs Strings     | Strings   | Simpler, uses real TLA+ syntax       |
| Runtime vs Extract-time | Extract   | Zero runtime overhead                |

## Comparison with Other Approaches

### vs. Separate TLA+ Files

**Before:**

- ✅ Full TLA+ expressiveness
- ❌ Drift from implementation
- ❌ Hard to discover
- ❌ Extra maintenance burden

**After (Embedded):**

- ✅ Co-located with code
- ✅ Harder to ignore/forget
- ✅ Discoverable via decorators
- ⚠️ Slightly less expressive (but good enough)

### vs. Runtime Verification (DbC)

**DbC decorators:**

- ✅ Catch bugs during testing
- ✅ Fast feedback
- ❌ Can't exhaustively check all states

**TLA+ model checking:**

- ✅ Exhaustively checks state space
- ✅ Finds subtle race conditions
- ❌ Slower (seconds to minutes)
- ❌ Limited by state space size

**Best of both:**
Use `@formal_spec` + `@require/@ensure/@invariant` together!

## Questions for Discussion

1. **Spec Coverage**: Should we require TLA+ specs for all critical paths?
2. **CI Integration**: Run TLC on every commit or only when specs change?
3. **Model Bounds**: How to specify bounds (MaxMessages, etc.) per-environment?
4. **Multi-module Specs**: How to handle TLA+ specs spanning multiple Python files?
5. **Property Tests**: Auto-generate Hypothesis tests from invariants?

## References

- `specs/TLA_EMBEDDING.md` - Full design exploration (5 approaches)
- `docs/FORMAL_SPEC_GUIDE.md` - User guide with examples
- `examples/formal_spec_example.py` - Working demonstrations
- `specs/VERIFICATION.md` - Current verification approach
- `specs/DBC.md` - Design-by-contract specification

## Conclusion

This implementation provides a **pragmatic, Python-native approach** to
embedding TLA+ formal specifications alongside implementation code. It:

- ✅ Reduces drift between specs and code
- ✅ Makes formal verification more accessible
- ✅ Integrates with existing DbC system
- ✅ Provides automated extraction and validation
- ✅ Maintains zero runtime overhead
- ✅ Supports gradual adoption

The structured metadata approach strikes the right balance between
expressiveness, usability, and maintainability.

**Ready to adopt?** Start with `docs/FORMAL_SPEC_GUIDE.md`!

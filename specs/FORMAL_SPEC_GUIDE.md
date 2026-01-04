# Embedding TLA+ Specifications in Python

## Quick Start

### 1. Add `@formal_spec` to your class

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

### 2. Extract TLA+ specs

```bash
# Extract specs to specs/tla/extracted/
pytest --extract-tla

# Extract and validate with TLC model checker
pytest --check-tla
```

### 3. View generated TLA+

```bash
cat specs/tla/extracted/Counter.tla
```

Output:

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

## Why Embed TLA+ Specs?

### Before (Separate Files)

```
specs/tla/Counter.tla          ← TLA+ spec (might drift)
src/counter.py                 ← Python implementation
```

**Problems:**

- Specs and code can drift apart
- Developers might not know specs exist
- Hard to keep synchronized during refactoring

### After (Embedded)

```python
@formal_spec(...)              ← TLA+ spec lives with code
class Counter:
    ...                        ← Implementation
```

**Benefits:**

- Single source of truth
- Specs visible during code review
- Automatic extraction and validation
- Less drift, more confidence

## Complete Example

See `examples/formal_spec_example.py` for working examples:

```bash
# Run the example
python examples/formal_spec_example.py

# Extract specs from examples
pytest --extract-tla examples/

# Model check the example specs
pytest --check-tla examples/
```

## `@formal_spec` Parameters

### `module` (required)

TLA+ module name. Must be a valid TLA+ identifier.

```python
@formal_spec(module="MyMailbox")
```

### `state_vars`

List of state variables. Each is a `StateVar(name, type, description)`.

```python
@formal_spec(
    module="Queue",
    state_vars=[
        StateVar("pending", "Seq(MessageId)", "Pending messages"),
        StateVar("invisible", "Function", "Invisible messages"),
    ],
)
```

**Common TLA+ types:**

- `Nat` - Natural numbers
- `Int` - Integers
- `BOOLEAN` - Boolean values
- `Seq(T)` - Sequence of type T
- `Set(T)` - Set of type T
- `Function` - Function/mapping

### `actions`

List of actions. Each is an `Action(name, parameters, preconditions, updates)`.

```python
Action(
    name="Send",
    parameters=("msgId",),
    preconditions=(
        "msgId \\notin DOMAIN messages",
        "Len(pending) < MaxMessages",
    ),
    updates={
        "pending": "Append(pending, msgId)",
        "messages": "messages @@ (msgId :> body)",
    },
    description="Send a new message",
)
```

**Notes:**

- Use raw strings (`r"..."`) for TLA+ with backslashes
- Preconditions are ANDed together
- Updates specify `var' = expr` for each variable
- Unchanged variables are automatically added

### `invariants`

List of invariants. Each is an `Invariant(id, name, predicate)`.

```python
Invariant(
    id="INV-1",
    name="MessageStateExclusive",
    predicate=r"\A msgId: (msgId \in pending) \/ (msgId \in DOMAIN invisible)",
    description="Message is either pending or invisible",
)
```

**Tips:**

- Use unique IDs (e.g., "INV-1", "INV-MessageExclusive")
- Name should be a valid TLA+ identifier
- Predicate is raw TLA+ expression
- Multi-line predicates are supported

### `constants`

Dictionary of constant definitions with default values.

```python
@formal_spec(
    module="Mailbox",
    constants={
        "MaxMessages": 100,
        "VisibilityTimeout": 30,
    },
)
```

These appear in both the `.tla` file and `.cfg` file.

### `helpers`

Dictionary of helper operators (raw TLA+).

```python
@formal_spec(
    module="Mailbox",
    helpers={
        "NULL": "0",
        "IsEmpty(seq)": "Len(seq) = 0",
        "Head(pending)": "pending[1]",
    },
)
```

## Combining with DbC Decorators

The `@formal_spec` decorator composes well with `@require`, `@ensure`, and
`@invariant` from the DbC system:

```python
from weakincentives.dbc import require, ensure, invariant
from weakincentives.formal import formal_spec, StateVar, Invariant

@formal_spec(
    module="Account",
    state_vars=[StateVar("balance", "Nat")],
    invariants=[Invariant("INV-1", "NonNegative", "balance >= 0")],
)
@invariant(lambda self: self.balance >= 0)  # Runtime check
class Account:
    @require(lambda self, amount: amount > 0)  # Runtime precondition
    @ensure(lambda self, result: self.balance >= result)  # Runtime postcondition
    def withdraw(self, amount: int) -> int:
        self.balance -= amount
        return self.balance
```

**Best practices:**

- Use `@formal_spec` for high-level state machine
- Use DbC decorators for detailed runtime checks
- Keep formal specs focused on critical invariants
- Use DbC for validation and defensive programming

## Pytest Plugin Options

### `--extract-tla`

Extract TLA+ specs from `@formal_spec` decorators.

```bash
pytest --extract-tla
```

Output: `specs/tla/extracted/*.tla` and `*.cfg` files

### `--check-tla`

Extract and validate with TLC model checker.

```bash
pytest --check-tla
```

**Requirements:**

- TLC must be installed (`brew install tlaplus` on macOS)
- Specs must be well-formed TLA+
- Model checking completes without errors

### `--tla-output-dir`

Custom output directory for extracted specs.

```bash
pytest --extract-tla --tla-output-dir=/tmp/tla-specs
```

### `--tla-checker`

Choose model checker (`tlc` or `apalache`).

```bash
pytest --check-tla --tla-checker=apalache
```

## CI Integration

Add to `.github/workflows/verify.yml`:

```yaml
name: Formal Verification

on: [push, pull_request]

jobs:
  tla-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          pip install uv
          uv sync

      - name: Install TLA+ Tools
        run: |
          wget -q https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar
          sudo mkdir -p /usr/local/lib
          sudo mv tla2tools.jar /usr/local/lib/
          echo '#!/bin/bash' | sudo tee /usr/local/bin/tlc
          echo 'java -jar /usr/local/lib/tla2tools.jar "$@"' | sudo tee -a /usr/local/bin/tlc
          sudo chmod +x /usr/local/bin/tlc

      - name: Extract and check TLA+ specs
        run: pytest --check-tla
```

## Advanced: Custom TLC Configuration

The plugin generates default `.cfg` files. For custom configuration:

1. Extract specs once:

   ```bash
   pytest --extract-tla
   ```

2. Edit generated `.cfg` file:

   ```bash
   vim specs/tla/extracted/MyModule.cfg
   ```

3. Add custom constraints, properties, or constants

4. Future runs will use your custom `.cfg`

Example custom config:

```
SPECIFICATION Spec

CONSTANTS
    MaxMessages = 5
    VisibilityTimeout = 10

CONSTRAINT
    Len(pending) <= 3

INVARIANTS
    MessageStateExclusive
    DeliveryCountMonotonic

PROPERTIES
    EventuallyProcessed
```

## Troubleshooting

### "No @formal_spec decorators found"

Make sure your module is importable:

```bash
python -c "import mymodule"
```

If import fails, fix the error first.

### "TLC not found"

Install TLA+ tools:

```bash
# macOS
brew install tlaplus

# Linux
wget https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar
```

### "Model checking failed"

Check the extracted `.tla` file for syntax errors:

```bash
cat specs/tla/extracted/MyModule.tla
```

Common issues:

- Typos in TLA+ syntax
- Unescaped backslashes (use `r"..."` raw strings)
- Missing helper operators
- Invalid TLA+ identifiers

### "Invariant violated"

This is a **good thing**! Your formal spec found a bug in your logic.

1. Check the TLC error output for counterexample
2. Review the violated invariant
3. Fix either the spec or the implementation
4. Re-run `pytest --check-tla`

## Migration Guide

### Migrating Existing TLA+ Specs

If you have `specs/tla/MyModule.tla`:

1. Create `@formal_spec` decorator from existing spec
2. Attach to corresponding Python class
3. Extract and compare:
   ```bash
   pytest --extract-tla
   diff specs/tla/MyModule.tla specs/tla/extracted/MyModule.tla
   ```
4. Adjust metadata until output matches
5. Delete original `.tla` file
6. Update references in docs

### Gradual Adoption

You don't have to migrate everything at once:

1. Start with critical classes (e.g., mailbox, session)
2. Add `@formal_spec` to new code
3. Migrate old specs during refactoring
4. Keep both approaches until migration complete

## Best Practices

### Do

✅ Co-locate specs with implementation
✅ Keep specs focused on critical invariants
✅ Run `--check-tla` in CI for critical paths
✅ Use DbC decorators for runtime validation
✅ Document invariant IDs in commit messages

### Don't

❌ Don't embed huge specs (>200 lines of TLA+)
❌ Don't duplicate all Python logic in TLA+
❌ Don't skip model checking locally before push
❌ Don't ignore invariant violations
❌ Don't forget to update specs during refactoring

## Next Steps

1. Read `specs/TLA_EMBEDDING.md` for design rationale
2. Study `examples/formal_spec_example.py`
3. Try adding `@formal_spec` to one of your classes
4. Run `pytest --check-tla` and fix any issues
5. Add to CI pipeline

## References

- [TLA+ Home](https://lamport.azurewebsites.net/tla/tla.html)
- [Learn TLA+](https://learntla.com/)
- [TLA+ Hyperbook](https://learntla.com/tla/)
- `specs/VERIFICATION.md` - Current verification approach
- `specs/DBC.md` - Design-by-contract specification

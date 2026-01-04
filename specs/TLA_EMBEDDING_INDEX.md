# TLA+ Specification Embedding: Complete Documentation Index

## Quick Navigation

| Document                                                 | Purpose                                  | Audience                        |
| -------------------------------------------------------- | ---------------------------------------- | ------------------------------- |
| **[TLA_EMBEDDING.md](TLA_EMBEDDING.md)**                 | Design exploration of 5 approaches       | Architects, contributors        |
| **[FORMAL_SPEC_GUIDE.md](FORMAL_SPEC_GUIDE.md)**         | User guide with examples                 | Developers using `@formal_spec` |
| **[TLA_MIGRATION_GUIDE.md](TLA_MIGRATION_GUIDE.md)**     | Step-by-step migration from legacy specs | Maintainers migrating code      |
| **[MAKEFILE_UPDATES.md](MAKEFILE_UPDATES.md)**           | Makefile targets and CI integration      | DevOps, contributors            |
| **[EMBEDDING_TLA_SUMMARY.md](EMBEDDING_TLA_SUMMARY.md)** | Implementation summary                   | Project overview                |

## At a Glance

### What Is This?

A system for **embedding TLA+ formal specifications directly in Python code** using
decorators. Instead of maintaining separate `.tla` files that can drift from
implementation, specs and code live together.

### Why?

**Problem:**

```
specs/tla/RedisMailbox.tla    ← TLA+ spec (can drift)
src/mailbox/_redis.py         ← Implementation (separate)
```

**Solution:**

```python
@formal_spec(...)             ← TLA+ spec embedded
class RedisMailbox:
    ...                       ← Implementation
```

**Benefits:**

- ✅ Single source of truth
- ✅ Specs visible in code review
- ✅ Automatic extraction and validation
- ✅ Zero runtime overhead
- ✅ Harder to forget during refactoring

### How It Works

#### 1. Embed specs with decorators

```python
from weakincentives.formal import formal_spec, StateVar, Action, Invariant

@formal_spec(
    module="Counter",
    state_vars=[StateVar("count", "Nat", "Current count")],
    actions=[Action("Increment", updates={"count": "count + 1"})],
    invariants=[Invariant("INV-1", "NonNegative", "count >= 0")],
)
class Counter:
    def __init__(self):
        self.count = 0
```

#### 2. Extract and validate

```bash
# Extract TLA+ specs
make extract-tla

# Extract and model check with TLC
make check-tla

# Run all verification
make verify-all
```

#### 3. Generated output

The decorator automatically generates:

- `specs/tla/extracted/Counter.tla` - Complete TLA+ module
- `specs/tla/extracted/Counter.cfg` - TLC configuration

## Reading Guide

### For First-Time Users

1. **Start here:** [FORMAL_SPEC_GUIDE.md](FORMAL_SPEC_GUIDE.md)
   - Quick start tutorial
   - Complete examples
   - Best practices
   - Troubleshooting

2. **Try the examples:**

   ```bash
   uv run python examples/formal_spec_example.py
   ```

3. **Add to your code:**
   - Copy a simple example
   - Add `@formal_spec` decorator
   - Run `make extract-tla`

### For Contributors Migrating Code

1. **Read:** [TLA_MIGRATION_GUIDE.md](TLA_MIGRATION_GUIDE.md)
   - Step-by-step migration process
   - `RedisMailbox.tla` → `@formal_spec` mapping
   - Validation checklist
   - Troubleshooting

2. **Update Makefile:** [MAKEFILE_UPDATES.md](MAKEFILE_UPDATES.md)
   - New targets (`extract-tla`, `check-tla`, etc.)
   - CI integration
   - Performance optimization

3. **Follow the process:**
   ```bash
   # 1. Add decorator to Python class
   # 2. Extract and compare
   make compare-specs
   # 3. Iterate until match
   # 4. Validate
   make check-tla
   # 5. Archive legacy spec
   ```

### For Architects and Designers

1. **Design rationale:** [TLA_EMBEDDING.md](TLA_EMBEDDING.md)
   - 5 approaches considered
   - Pros/cons analysis
   - Why structured metadata won
   - Architecture decisions

2. **Implementation summary:** [EMBEDDING_TLA_SUMMARY.md](EMBEDDING_TLA_SUMMARY.md)
   - What was built
   - How it works
   - Files created
   - Design trade-offs

## Document Summaries

### TLA_EMBEDDING.md (Design Exploration)

**Purpose:** Explore and compare different approaches for embedding TLA+ in Python.

**Approaches evaluated:**

1. ❌ Docstring blocks - mixes concerns, hard to parse
2. ❌ String decorator literals - brittle, no IDE support
3. ❌ Python DSL - huge effort, limited expressiveness
4. ⚠️ DbC + TLA fragments - good for simple cases, incomplete
5. ✅ **Structured metadata** - chosen for type safety and composability

**Key sections:**

- Approach 1-5: Detailed analysis with code examples
- Pytest plugin design
- Recommended approach (hybrid)
- Open questions

**Read this if:** You want to understand why we chose structured metadata
over other approaches.

---

### FORMAL_SPEC_GUIDE.md (User Guide)

**Purpose:** Teach developers how to use `@formal_spec` decorator.

**Contents:**

- Quick start (3 steps)
- Complete example walkthrough
- `@formal_spec` parameter reference
- Integration with DbC decorators
- Pytest plugin options
- CI integration
- Troubleshooting guide

**Read this if:** You're adding formal specs to your code for the first time.

---

### TLA_MIGRATION_GUIDE.md (Migration Instructions)

**Purpose:** Step-by-step guide for migrating existing TLA+ specs to embedded form.

**Process:**

1. Understand current spec
2. Create skeleton decorator
3. Migrate constants
4. Migrate variables
5. Migrate helper operators
6. Migrate actions
7. Migrate invariants
8. Extract and validate
9. Model check
10. Archive original spec
11. Update documentation

**Includes:**

- Complete `RedisMailbox` migration example
- Verification checklist
- Troubleshooting common issues
- Performance considerations

**Read this if:** You're migrating `specs/tla/RedisMailbox.tla` to embedded form.

---

### MAKEFILE_UPDATES.md (Build Integration)

**Purpose:** Document Makefile changes for TLA+ extraction and validation.

**New targets:**

- `make extract-tla` - Extract specs (fast, 1-2s)
- `make check-tla` - Extract + model check (slow, 5-60s)
- `make check-tla-fast` - Extract only (alias)
- `make verify-embedded` - Alias for check-tla
- `make verify-all` - All verification methods
- `make compare-specs` - Diff legacy vs embedded
- `make clean-extracted` - Remove extracted files

**Includes:**

- Complete Makefile diff
- Usage examples
- CI integration
- Incremental adoption path
- Performance optimization

**Read this if:** You're integrating TLA+ checking into CI or local workflow.

---

### EMBEDDING_TLA_SUMMARY.md (Implementation Overview)

**Purpose:** High-level summary of what was implemented and why.

**Contents:**

- What was implemented
- How it works (3 steps)
- Design rationale
- Architecture overview
- Usage examples
- Testing approach
- Next steps
- Migration path

**Read this if:** You want a quick overview before diving into details.

## Common Workflows

### Workflow 1: Add Spec to New Code

```bash
# 1. Add @formal_spec decorator to your class
vim src/mymodule.py

# 2. Extract to check syntax
make extract-tla

# 3. Validate with model checker
make check-tla

# 4. Commit
git add src/mymodule.py
git commit -m "Add formal spec for MyClass"
```

### Workflow 2: Migrate Existing Spec

```bash
# 1. Read migration guide
cat specs/TLA_MIGRATION_GUIDE.md

# 2. Add decorator to implementation
vim src/weakincentives/contrib/mailbox/_redis.py

# 3. Extract and compare
make compare-specs

# 4. Iterate until match
# (edit, extract, compare, repeat)

# 5. Validate
make check-tla

# 6. Archive legacy spec
mkdir -p specs/tla/archive
mv specs/tla/RedisMailbox.tla specs/tla/archive/

# 7. Commit
git add src/weakincentives/contrib/mailbox/_redis.py
git add specs/tla/archive/
git commit -m "Migrate RedisMailbox spec to embedded form"
```

### Workflow 3: Validate During Development

```bash
# Fast: extract only (1-2s)
make extract-tla

# Check syntax without model checking
cat specs/tla/extracted/MyModule.tla

# Full validation before commit (5-60s)
make check-tla

# All verification including property tests (1-5m)
make verify-all
```

### Workflow 4: CI Integration

```yaml
# .github/workflows/verify.yml
- name: Run formal verification
  run: make verify-all
```

This runs:

- Legacy TLA+ specs (`specs/tla/*.tla`)
- Embedded specs (`@formal_spec` decorators)
- Property-based tests (Hypothesis)

## File Structure

```
weakincentives/
├── specs/
│   ├── TLA_EMBEDDING_INDEX.md       ← You are here
│   ├── TLA_EMBEDDING.md             ← Design exploration
│   ├── FORMAL_SPEC_GUIDE.md         ← User guide
│   ├── TLA_MIGRATION_GUIDE.md       ← Migration instructions
│   ├── MAKEFILE_UPDATES.md          ← Build integration
│   ├── EMBEDDING_TLA_SUMMARY.md     ← Implementation summary
│   ├── tla/
│   │   ├── RedisMailbox.tla         ← Legacy spec (to be archived)
│   │   ├── RedisMailboxMC.tla       ← Legacy model checking config
│   │   ├── RedisMailboxMC.cfg       ← Legacy TLC config
│   │   └── extracted/               ← Generated from @formal_spec
│   │       ├── RedisMailbox.tla     ← Auto-generated
│   │       └── RedisMailbox.cfg     ← Auto-generated
│   └── VERIFICATION.md              ← Existing verification docs
├── src/weakincentives/
│   ├── formal/
│   │   └── __init__.py              ← FormalSpec, StateVar, Action, Invariant
│   └── contrib/mailbox/
│       └── _redis.py                ← Will have @formal_spec decorator
├── tests/
│   ├── plugins/
│   │   └── pytest_tla.py            ← Extraction and validation plugin
│   └── formal/
│       └── test_tla_extraction.py   ← Unit tests
├── examples/
│   ├── formal_spec_example.py       ← Complete examples
│   └── simple_tla_demo.py           ← Minimal demo
└── Makefile                          ← Updated with new targets
```

## Key Concepts

### Structured Metadata

The `@formal_spec` decorator uses structured metadata classes:

- **`StateVar(name, type, description)`** - State variable
- **`Action(name, parameters, preconditions, updates)`** - State transition
- **`Invariant(id, name, predicate)`** - Safety property
- **`FormalSpec(...)`** - Complete TLA+ module

These are **Python dataclasses**, providing:

- Type checking
- IDE autocomplete
- Validation
- Mechanical translation to TLA+

### Extraction Process

```
Python code with @formal_spec
         ↓
    pytest --extract-tla
         ↓
  FormalSpec.to_tla()
         ↓
  specs/tla/extracted/*.tla
         ↓
    TLC model checker
         ↓
  Pass/Fail + Counterexamples
```

### Integration with DbC

The formal spec system **complements** design-by-contract:

```python
@formal_spec(...)           # TLA+ state machine (exhaustive)
@invariant(...)             # Runtime check (tests only)
class MyClass:
    @require(...)           # Runtime precondition
    @ensure(...)            # Runtime postcondition
    def my_method(self):
        ...
```

**Division of labor:**

- `@formal_spec`: High-level semantics for TLC model checking
- `@require/@ensure/@invariant`: Runtime validation during tests

## FAQ

### Q: Do I need to know TLA+ to use this?

**A:** Basic TLA+ knowledge helps, but you can start by copying examples.
The structured metadata shields you from much of TLA+'s complexity.

### Q: Does this slow down my code?

**A:** No. Specs are extracted at **test time**, not runtime. Zero overhead in production.

### Q: Can I use this without migrating existing specs?

**A:** Yes! Use `@formal_spec` for new code while keeping legacy `.tla` files.
Run both with `make verify-all`.

### Q: What if TLC finds a bug?

**A:** This is **good**! The counterexample shows a sequence of actions violating
an invariant. Review the trace, fix the bug (in spec or code), and re-validate.

### Q: How big can specs be?

**A:** Recommended: <200 lines of TLA+. For larger specs, consider splitting
into multiple modules or keeping them as separate `.tla` files.

### Q: What about temporal properties (liveness)?

**A:** Not yet supported in structured metadata. Use legacy `.tla` files for
temporal operators (`◇`, `□`) and fairness constraints.

## Next Steps

1. **Read the user guide:** [FORMAL_SPEC_GUIDE.md](FORMAL_SPEC_GUIDE.md)
2. **Try the examples:**
   ```bash
   uv run python examples/formal_spec_example.py
   ```
3. **Add a simple spec to your code**
4. **Run extraction:**
   ```bash
   make extract-tla
   ```
5. **Validate:**
   ```bash
   make check-tla
   ```

## Getting Help

- **User guide:** `specs/FORMAL_SPEC_GUIDE.md`
- **Migration guide:** `specs/TLA_MIGRATION_GUIDE.md`
- **Examples:** `examples/formal_spec_example.py`
- **Design rationale:** `specs/TLA_EMBEDDING.md`
- **Makefile reference:** `specs/MAKEFILE_UPDATES.md`

## Contributing

When adding or modifying embedded specs:

1. ✅ Add `@formal_spec` decorator
2. ✅ Run `make extract-tla` locally
3. ✅ Run `make check-tla` before committing
4. ✅ Include spec changes in PR description
5. ✅ Update invariants if behavior changes

See `AGENTS.md` for full contributor guidelines.

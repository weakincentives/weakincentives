# Architecture Review Checklist

Post-`make check` reminder for architectural properties that require human judgment.

## Part 1: Ideas for New Automated Checks

### High-Value Additions

| Check | Description | Implementation |
|-------|-------------|----------------|
| **Frozen dataclass audit** | Verify all public dataclasses use `frozen=True` | AST scan for `@dataclass` without `frozen=True` in public modules |
| **Reducer purity enforcement** | All `@reducer` methods decorated with `@pure` | Grep for `@reducer` without accompanying `@pure` |
| **SliceOp return type check** | Reducers return `Append`/`Replace`/`Clear`/`Extend` | Pyright already catches, but explicit test validates patterns |
| **ToolResult consistency** | Tool handlers return `ToolResult[T]`, not raw values | Type check + grep for handlers not using `ToolResult` |
| **DbC coverage report** | Track % of public APIs with contracts | Script to count `@require`/`@ensure` on public functions |
| **Dispatch-only mutations** | Detect direct slice assignment outside `dispatch()` | AST scan for `session._slices` or slice mutation patterns |
| **Protocol-over-inheritance** | Flag class inheritance from non-Protocol bases | AST check favoring `Protocol` for abstractions |
| **No time.sleep in reducers** | Blocking calls in reducers cause deadlocks | Grep/AST for `time.sleep` in reducer-decorated functions |
| **Deadline propagation** | Tools access deadline from context, not globals | Grep for `datetime.now()` in tool handlers |
| **Resource scope validation** | SINGLETON resources don't depend on TOOL_CALL | Static analysis of `Binding` dependency graphs |

### Moderate-Value Additions

| Check | Description |
|-------|-------------|
| **Snapshot round-trip property test** | Auto-generate property tests for all dataclasses |
| **Unused event types** | Detect event dataclasses with no registered reducers |
| **Orphan reducers** | Reducers registered but never dispatched in tests |
| **Tool policy coverage** | Every tool has at least one policy test case |
| **Invariant coverage** | Classes with `@invariant` have tests that exercise the invariant |

### Low-Friction Quick Wins

```bash
# Add to Makefile
check-frozen-dataclasses:
    @uv run python build/check_frozen_dataclasses.py

check-reducer-purity:
    @grep -rn "@reducer" src/weakincentives --include="*.py" | \
        while read line; do \
            file=$(echo "$line" | cut -d: -f1); \
            if ! grep -B5 "@reducer" "$file" | grep -q "@pure"; then \
                echo "WARNING: Reducer without @pure: $file"; \
            fi; \
        done
```

---

## Part 2: Human Review Checklist

Display this after `make check` passes. These are subjective architectural properties
that require human judgment to evaluate.

### Core Invariants

- [ ] **Immutability discipline**: Are new data structures frozen dataclasses?
  New public state must be `@dataclass(slots=True, frozen=True)` or `@FrozenDataclass()`.
  Mutable state is only acceptable in clearly scoped, internal contexts.

- [ ] **Pure reducers**: Do new reducers have side effects?
  Reducers must be pure functions—no I/O, no logging, no mutations of arguments.
  If you need side effects, they belong in tool handlers, not reducers.

- [ ] **Dispatch-only state changes**: Are all state mutations going through `dispatch()`?
  Direct slice manipulation breaks auditability. Every mutation should produce
  a typed event that flows through the session's event ledger.

- [ ] **Typed contracts everywhere**: Are params, results, and state all dataclasses?
  Untyped dicts or `Any` erode type safety. If you're tempted to use `dict[str, Any]`,
  define a dataclass instead.

### Design-by-Contract

- [ ] **Appropriate DbC decorators**: Do new public APIs have contracts where meaningful?
  - `@require` for preconditions that aren't obvious from types
  - `@ensure` for postconditions that define the function's guarantee
  - `@invariant` for class-level properties that must always hold
  - `@pure` for functions that must be side-effect-free

- [ ] **Contracts are testable**: Are contract predicates exercised by tests?
  A contract that never fires is documentation at best. Tests should include
  cases that would violate contracts (when DbC is enabled).

### Module Architecture

- [ ] **Layer discipline**: Does the change respect the 4-layer architecture?
  ```
  Layer 4: contrib, evals, cli (high-level)
  Layer 3: adapters
  Layer 2: runtime, prompt, resources, filesystem, serde, skills, formal (core)
  Layer 1: types, errors, dataclasses, dbc, deadlines, budget (foundation)
  ```
  Lower layers must not import from higher layers (except via `TYPE_CHECKING`).

- [ ] **Private modules stay private**: Are `_foo.py` modules imported only within their package?
  Private modules are implementation details—expose through `__init__.py` if needed.

- [ ] **Minimal `__init__.py` exports**: Are exports curated?
  Don't export everything. Each export is a public API commitment.

### Tool Design

- [ ] **Consistent handler signature**: Do new tools follow `(params, *, context: ToolContext) -> ToolResult[T]`?
  The uniform signature enables composition, testing, and policy enforcement.

- [ ] **Fail gracefully**: Do tool handlers return `ToolResult.error()` instead of raising?
  Tool failures should be communicated back to the LLM, not crash the session.
  Only raise `ToolValidationError` for parameter problems.

- [ ] **Context-based resource access**: Do tools use `context.resources.get()` for dependencies?
  Don't instantiate resources inline—use the scoped resource registry.

- [ ] **Deadline awareness**: Do long-running tools check `context.deadline`?
  Tools should fail early if deadline has passed, not start expensive operations.

### Session & State

- [ ] **Event granularity**: Are events appropriately fine-grained?
  One event should represent one logical change. Don't pack multiple operations
  into a single event—it breaks replay granularity.

- [ ] **Snapshot round-trip**: Do new state types serialize correctly?
  Test that `snapshot() → restore() → snapshot()` produces identical results.
  Watch for: datetime handling, enum serialization, nested dataclass fidelity.

- [ ] **No implicit state**: Is there hidden state outside the session?
  Module-level variables, class attributes, closures—all can harbor implicit state
  that breaks determinism. State belongs in the session or scoped resources.

### Resource Management

- [ ] **Explicit scopes**: Are resource lifetimes appropriate?
  - `SINGLETON`: Lives for the entire prompt context (expensive to create)
  - `TOOL_CALL`: Fresh per tool invocation (needs isolation)
  - `PROTOTYPE`: Fresh every access (stateless helpers)

- [ ] **Cleanup guarantees**: Do resources implement `Closeable` if they hold external resources?
  File handles, connections, temp directories—all must clean up on scope exit.

- [ ] **No scope violations**: Do SINGLETON resources avoid depending on TOOL_CALL resources?
  A singleton can't sensibly depend on something that changes per tool call.

### Philosophy Alignment

- [ ] **Policies over workflows**: Does the change preserve agent reasoning?
  Prefer declarative constraints ("file must be read before write") over
  procedural scripts ("step 1, step 2, step 3"). Let the LLM reason.

- [ ] **Weak incentives**: Does the change make the correct path obvious?
  Good design makes it easier to do the right thing than the wrong thing.
  Co-locate documentation with capability. Type contracts guide valid outputs.

- [ ] **Fail-closed**: When uncertain, does the system deny rather than allow?
  Policies should reject on doubt. The agent can then reason about why and adjust.

- [ ] **Auditability**: Can the change be inspected and replayed?
  Every mutation recorded. Every state transition traceable. If you can't
  explain what happened by reading the event log, something is wrong.

### Testing & Verification

- [ ] **Regression test for bugs**: Does the bug fix include `test_regression_<issue>_<description>`?
  Every fix needs a test that fails before the fix and passes after.

- [ ] **Property-based tests for invariants**: Are critical invariants tested with Hypothesis?
  For state machines and complex data structures, property tests catch edge cases
  that example-based tests miss.

- [ ] **Formal spec alignment**: If touching verified code, does the implementation match the spec?
  TLA+ specs define what correctness means. Changes to verified components
  should be reflected in the spec (or justify why the spec doesn't need updating).

### Documentation

- [ ] **Spec updates**: If changing architecture, is the relevant spec updated?
  Specs are the source of truth for design guarantees. Code and spec must agree.

- [ ] **CHANGELOG entry**: Is there a user-visible change?
  Document new features, behavior changes, and breaking changes under "Unreleased".

- [ ] **No speculative features**: Does the change solve a real, present problem?
  Don't add configurability, feature flags, or abstractions "for the future."
  Build what's needed now.

---

## Quick Reference: Red Flags

These patterns suggest architectural problems:

| Red Flag | Why It's Bad | Alternative |
|----------|--------------|-------------|
| `@dataclass` without `frozen=True` | Mutable state breaks determinism | Add `frozen=True, slots=True` |
| `time.sleep()` in reducer | Blocks all dispatches | Move to tool handler |
| `session._slices[...] = ...` | Bypasses audit trail | Use `session.dispatch(event)` |
| `dict[str, Any]` in API | Erodes type safety | Define a dataclass |
| `except Exception: pass` | Swallows important failures | Handle specific exceptions |
| Import from higher layer | Violates architecture | Use `TYPE_CHECKING` or protocol |
| Tool handler raising raw Exception | Crashes instead of failing gracefully | Return `ToolResult.error()` |
| `datetime.now()` in tool | Clock dependency breaks determinism | Use `context.deadline` or explicit param |
| Resource without `Closeable` | Leaks external resources | Implement cleanup |

---

## Suggested Makefile Addition

```makefile
# Display architecture checklist after checks pass
check: format-check lint typecheck ... test
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "  ✓ All automated checks passed"
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "  Before committing, review ARCHITECTURE_REVIEW_CHECKLIST.md"
	@echo "  Key questions:"
	@echo "    • Are new data structures frozen dataclasses?"
	@echo "    • Do reducers have side effects? (They shouldn't)"
	@echo "    • Are all state changes going through dispatch()?"
	@echo "    • Do tool handlers return ToolResult, not raise?"
	@echo "    • Does the change respect layer boundaries?"
	@echo ""
```

---

## Version History

| Date | Change |
|------|--------|
| 2025-01-18 | Initial checklist based on architectural analysis |

# Exhaustiveness Checking Specification

## Purpose

This document specifies exhaustiveness checking patterns for union types in
`weakincentives`. When you add a new event type, message variant, or error case,
every match statement handling that union must be updated—but without
exhaustiveness checking, the compiler won't tell you which handlers you missed
until runtime when that branch executes. Exhaustiveness checking inverts this:
adding a variant immediately fails type checking at every incomplete handler.

For event-driven architectures like this codebase's session system, where events
flow through multiple reducers and handlers, exhaustiveness ensures that adding
`NewEventType` forces updates to every place that needs to handle it, rather
than silently falling through to a default case or raising an unexpected
exception.

## Guiding Principles

- **Compile-time safety over runtime discovery**: Catch missing handlers during
  development, not in production when the new variant finally appears.
- **Explicit totality**: Every match must handle all cases or explicitly
  acknowledge incompleteness with a typed assertion.
- **No silent defaults**: Avoid catch-all `case _:` patterns that swallow new
  variants without forcing review.
- **Incremental adoption**: Patterns should be adoptable module-by-module
  without requiring a full codebase rewrite.
- **Pyright-native**: Leverage pyright's strict mode and type narrowing rather
  than runtime machinery.

## Goals

- Establish patterns for exhaustive matching on union types using Python's
  `match` statement and pyright's type narrowing.
- Document the `assert_never` sentinel pattern for enforcing totality.
- Identify critical union types in the codebase that require exhaustive
  handling.
- Provide migration guidance for existing match statements.
- Integrate exhaustiveness violations into the existing `make typecheck` flow.

## Non-Goals

- This specification does not cover runtime exhaustiveness validation. The focus
  is purely on static type checking via pyright.
- We do not attempt to build custom exhaustiveness infrastructure. Python 3.11+
  and pyright provide sufficient primitives.
- Backwards compatibility shims for incomplete handlers are explicitly
  discouraged per project policy.

## The Problem

### Silent Failures

Consider the `SliceOp` union type:

```python
type SliceOp[T] = Append[T] | Extend[T] | Replace[T] | Clear[T]
```

A handler matching this union:

```python
def apply_op(op: SliceOp[S], slice: Slice[S]) -> None:
    match op:
        case Append(item=item):
            slice.append(item)
        case Extend(items=items):
            slice.extend(items)
        case Replace(items=items):
            slice.replace(items)
        case Clear(predicate=pred):
            slice.clear(pred)
```

This currently handles all four variants. But when someone adds `Truncate[T]` to
the union:

```python
type SliceOp[T] = Append[T] | Extend[T] | Replace[T] | Clear[T] | Truncate[T]
```

The match statement above compiles without error. At runtime, when a `Truncate`
operation arrives, the match silently falls through (no case matches, no
exception raised in Python 3.10+ unless in a function expecting a return value).

### The Cascade Problem

Event-driven systems amplify this risk. The `DataEvent` union:

```python
type DataEvent = PromptExecuted | PromptRendered | ToolInvoked
```

Events flow through:

1. Session dispatch routing
1. Reducer invocations
1. Telemetry handlers
1. Debug/logging subscribers

Adding `SessionCheckpointed` to `DataEvent` requires updates to potentially
dozens of handlers. Without exhaustiveness checking, you discover missing
handlers one at a time as each code path executes in production.

## The Solution: `assert_never`

Python's `typing` module provides `Never` (or `NoReturn` in older versions), a
type that represents values that can never exist. The `assert_never` function
leverages this:

```python
from typing import Never, assert_never

def apply_op(op: SliceOp[S], slice: Slice[S]) -> None:
    match op:
        case Append(item=item):
            slice.append(item)
        case Extend(items=items):
            slice.extend(items)
        case Replace(items=items):
            slice.replace(items)
        case Clear(predicate=pred):
            slice.clear(pred)
        case _ as unreachable:
            assert_never(unreachable)
```

### How It Works

After exhaustively matching all union members, `op` has type `Never`—there are
no remaining possibilities. The `assert_never(unreachable)` call expects a
`Never` argument. If a new variant is added to `SliceOp`, pyright narrows `op`
to that new type in the default case, causing a type error:

```
error: Argument of type "Truncate[S]" cannot be assigned to parameter
       of type "Never"
```

This error appears at **every** match statement missing the new variant,
immediately upon adding it to the union.

### Runtime Behavior

`assert_never` raises `AssertionError` if reached at runtime:

```python
def assert_never(arg: Never, /) -> Never:
    raise AssertionError(f"Expected code to be unreachable, but got: {arg!r}")
```

This provides defense-in-depth: even if types drift from reality, the runtime
assertion catches the impossible case.

## Implementation Patterns

### Pattern 1: Match Statement with Sentinel

The primary pattern for exhaustive matching:

```python
from typing import Never, assert_never

def handle_event(event: DataEvent) -> str:
    match event:
        case PromptExecuted() as e:
            return f"Executed: {e.prompt_name}"
        case PromptRendered() as e:
            return f"Rendered: {e.prompt_key}"
        case ToolInvoked() as e:
            return f"Invoked: {e.name}"
        case _ as unreachable:
            assert_never(unreachable)
```

### Pattern 2: If-Elif Chain with Sentinel

For legacy code or when match isn't suitable:

```python
from typing import Never, assert_never

def handle_event(event: DataEvent) -> str:
    if isinstance(event, PromptExecuted):
        return f"Executed: {event.prompt_name}"
    elif isinstance(event, PromptRendered):
        return f"Rendered: {event.prompt_key}"
    elif isinstance(event, ToolInvoked):
        return f"Invoked: {event.name}"
    else:
        assert_never(event)
```

Pyright's type narrowing reduces `event`'s type through each branch. After all
union members are checked, the else branch has type `Never`.

### Pattern 3: Dictionary Dispatch with Type Coverage

For handler registries, use `TypedDict` or explicit type mapping:

```python
from typing import Callable, Never, assert_never

type EventHandler[E] = Callable[[E], str]

# Explicit handler map with all variants
_HANDLERS: dict[type[DataEvent], EventHandler[DataEvent]] = {
    PromptExecuted: lambda e: f"Executed: {e.prompt_name}",
    PromptRendered: lambda e: f"Rendered: {e.prompt_key}",
    ToolInvoked: lambda e: f"Invoked: {e.name}",
}

def handle_event(event: DataEvent) -> str:
    handler = _HANDLERS.get(type(event))
    if handler is None:
        # Type checker sees event as DataEvent here, not Never
        # Use explicit exhaustive check instead
        _check_all_handlers_registered(event)
    return handler(event)

def _check_all_handlers_registered(event: DataEvent) -> Never:
    """Called when handler lookup fails - verifies at type level."""
    match event:
        case PromptExecuted() | PromptRendered() | ToolInvoked():
            raise RuntimeError(f"Handler missing for {type(event).__name__}")
        case _ as unreachable:
            assert_never(unreachable)
```

### Pattern 4: Protocol-Based Exhaustiveness

For open-ended extension points, use protocols instead of unions:

```python
from typing import Protocol

class Reducible(Protocol):
    def reduce(self, view: SliceView[Self]) -> SliceOp[Self]: ...
```

Protocols shift exhaustiveness from the handler to the implementer—each new type
must implement the protocol method.

## Critical Union Types

The following union types in `weakincentives` require exhaustive handling:

### `SliceOp[T]`

**Location**: `src/weakincentives/runtime/session/slices/_ops.py:65`

```python
type SliceOp[T: SupportsDataclass] = Append[T] | Extend[T] | Replace[T] | Clear[T]
```

**Handlers requiring exhaustiveness**:

- `Session._apply_slice_op()` - Applies reducer results to slices

### `DataEvent`

**Location**: `src/weakincentives/runtime/session/session.py:68`

```python
type DataEvent = PromptExecuted | PromptRendered | ToolInvoked
```

**Handlers requiring exhaustiveness**:

- Session subscription routing in `_attach_to_dispatcher()`
- Any custom telemetry subscribers

### `SystemEvent`

System mutation events handled by the session:

```python
type SystemEvent[T] = InitializeSlice[T] | ClearSlice[T]
```

**Handlers requiring exhaustiveness**:

- `Session._dispatch_system_event()` - Routes system mutations

### `JSONValue`

**Location**: `src/weakincentives/types/json.py`

```python
type JSONValue = _JSONPrimitive | JSONObject | JSONArray
```

**Handlers requiring exhaustiveness**:

- Serialization/deserialization utilities
- Schema validation

### `ContractResult`

**Location**: `src/weakincentives/dbc/__init__.py`

```python
type ContractResult = bool | tuple[bool, *tuple[object, ...]] | None
```

**Handlers requiring exhaustiveness**:

- Contract predicate evaluation

## Migration Guide

### Step 1: Identify Match Statements

Find all match statements on union types:

```bash
rg "match\s+\w+:" --type py -A 20 | grep -B 5 "case _:"
```

### Step 2: Replace Catch-All with Sentinel

Before:

```python
match op:
    case Append(item=item):
        slice.append(item)
    # ... other cases ...
    case _:
        pass  # Silent fallthrough
```

After:

```python
from typing import assert_never

match op:
    case Append(item=item):
        slice.append(item)
    # ... other cases ...
    case _ as unreachable:
        assert_never(unreachable)
```

### Step 3: Verify Type Narrowing

Temporarily add a variant to the union and run `make typecheck`. Errors should
appear at every match statement with `assert_never`.

### Step 4: Handle Intentionally Partial Matches

Some handlers legitimately handle a subset of variants. Document the intent:

```python
def handle_execution_events(event: DataEvent) -> None:
    """Only processes execution-related events; ignores renders."""
    match event:
        case PromptExecuted() as e:
            record_execution(e)
        case ToolInvoked() as e:
            record_tool_call(e)
        case PromptRendered():
            pass  # Intentionally ignored - render events handled elsewhere
        case _ as unreachable:
            assert_never(unreachable)
```

The explicit `case PromptRendered(): pass` documents the decision and maintains
exhaustiveness.

## Pyright Configuration

The project's `pyproject.toml` already enforces strict mode:

```toml
[tool.pyright]
typeCheckingMode = "strict"
```

Key settings that enable exhaustiveness checking:

- `reportMatchNotExhaustive = true` - Reports non-exhaustive match statements
  (enabled in strict mode)
- `reportUnnecessaryComparison = true` - Catches redundant isinstance checks
  after exhaustive narrowing

To verify exhaustiveness is enforced:

```bash
make typecheck  # Runs pyright in strict mode
```

## Reducer Registration Exhaustiveness

The declarative `@reducer` pattern provides compile-time exhaustiveness for
event handling within a slice:

```python
@dataclass(frozen=True)
class AgentPlan:
    steps: tuple[str, ...]

    @reducer(on=AddStep)
    def add_step(self, event: AddStep) -> Replace["AgentPlan"]:
        return Replace((replace(self, steps=(*self.steps, event.step)),))

    @reducer(on=RemoveStep)
    def remove_step(self, event: RemoveStep) -> Replace["AgentPlan"]:
        return Replace((replace(self, steps=tuple(
            s for s in self.steps if s != event.step
        )),))
```

When a new event type should be handled by `AgentPlan`, the developer must add a
`@reducer` method. This is enforced by convention rather than the type system,
but the pattern makes the expectation clear.

For stricter guarantees, define an event union and a validation function:

```python
type PlanEvent = AddStep | RemoveStep | CompleteStep

def validate_plan_handles_all_events(event: PlanEvent) -> None:
    """Type-level assertion that AgentPlan handles all PlanEvents."""
    plan = AgentPlan(steps=())
    match event:
        case AddStep():
            plan.add_step(event)
        case RemoveStep():
            plan.remove_step(event)
        case CompleteStep():
            plan.complete(event)
        case _ as unreachable:
            assert_never(unreachable)
```

This function is never called at runtime but serves as a type-level contract
that adding to `PlanEvent` requires updating the validation.

## Testing Strategy

### Unit Tests

Test that `assert_never` raises at runtime for defensive purposes:

```python
def test_assert_never_raises():
    """Verify assert_never provides runtime safety."""
    from typing import assert_never

    with pytest.raises(AssertionError, match="unreachable"):
        assert_never("unexpected value")  # type: ignore[arg-type]
```

### Type-Level Tests

Create a `tests/typecheck/` directory with intentionally failing type scenarios:

```python
# tests/typecheck/exhaustiveness_test.py
# pyright: strict

from typing import assert_never
from weakincentives.runtime.session.slices import SliceOp, Append

def incomplete_handler(op: SliceOp[int]) -> None:
    """This should fail type checking - missing cases."""
    match op:
        case Append(item=item):
            print(item)
        case _ as unreachable:
            assert_never(unreachable)  # Error: Extend | Replace | Clear != Never
```

Run with `pyright tests/typecheck/` and expect specific errors.

### Integration Tests

Verify end-to-end that new variants are caught:

```python
def test_slice_op_exhaustiveness():
    """Verify SliceOp match is exhaustive by attempting all variants."""
    from weakincentives.runtime.session import Session
    from weakincentives.runtime.session.slices import Append, Extend, Replace, Clear

    session = Session()
    session.install(TestSlice)

    # Exercise all SliceOp variants through the session
    ops = [
        Append(TestSlice(value=1)),
        Extend((TestSlice(value=2),)),
        Replace((TestSlice(value=3),)),
        Clear(),
    ]
    for op in ops:
        # If a variant is missing from _apply_slice_op, this would fail
        session._apply_slice_op(op, session._get_slice(TestSlice))
```

## Checklist for New Union Types

When defining a new union type:

- [ ] Document all variants in a type alias or `Union[]` annotation
- [ ] Add the union to the "Critical Union Types" section if it has multiple
  handlers
- [ ] Ensure all handlers use `assert_never` sentinel pattern
- [ ] Add type-level test verifying exhaustiveness
- [ ] Consider whether a protocol is more appropriate for open extension

When adding a variant to an existing union:

- [ ] Run `make typecheck` immediately after adding the variant
- [ ] Address all type errors at `assert_never` sites
- [ ] Update tests to exercise the new variant
- [ ] Consider whether existing `pass` cases should now handle the new variant

## Limitations

- **Runtime type erasure**: Generics like `SliceOp[T]` erase at runtime.
  `assert_never` catches the base type but cannot validate generic parameters.
- **Dynamic dispatch**: Handler registries using `dict[type, Callable]` require
  manual validation functions to achieve type-level exhaustiveness.
- **Mypy differences**: This spec targets pyright. Mypy's exhaustiveness
  checking may behave differently, particularly around type aliases.
- **Protocol limitations**: Protocols cannot enforce exhaustiveness for
  consumers—only for implementers.

## Summary

Exhaustiveness checking transforms "forgot to handle the new case" from a
runtime bug discovered months later into an immediate compile-time error.
The pattern is simple:

1. Define union types explicitly
1. End every match with `case _ as unreachable: assert_never(unreachable)`
1. Run `make typecheck` after modifying unions

This small discipline provides disproportionate safety for event-driven systems
where events flow through multiple handlers across the codebase.

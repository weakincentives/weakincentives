# Assertion Density Specification

## Purpose

This document specifies a coding discipline for using Python's built-in
`assert` statement strategically throughout the `weakincentives` codebase.
High assertion density catches bugs closer to their source rather than letting
invalid state propagate until it causes a confusing failure elsewhere. Unlike
the more sophisticated design-by-contract decorators (`@require`, `@ensure`,
`@invariant`), inline assertions provide a low-friction, zero-adoption-cost
approach that works immediately with existing code.

## Relationship to Design-by-Contract

The DbC decorators in `weakincentives.dbc` remain the preferred approach for
public API boundaries where contracts should be formally documented and
enforced during testing. Inline assertions complement DbC by:

- Documenting assumptions in private helpers and internal logic
- Catching violations at loop boundaries and state transitions
- Providing immediate feedback during development without decorator overhead
- Serving as lightweight sanity checks where full DbC would be overkill

Use DbC decorators for public contract enforcement; use inline assertions for
internal invariant documentation and early failure detection.

## Guiding Principles

- **Assert at boundaries**: Place assertions at function entry points, loop
  invariants, and state transition boundaries where assumptions are most likely
  to be violated by upstream changes.
- **Fail fast, fail loud**: Invalid state should cause immediate failure with
  clear context rather than propagating through multiple call frames.
- **Messages are mandatory**: Every assertion must include a message that
  provides debugging context. Bare `assert condition` is insufficient.
- **Context over description**: Messages should include relevant variable values
  and identifiers, not just describe what was expected.
- **Production-aware**: Assertions may be compiled out with `-O` or converted
  to monitoring in production. Do not use assertions for input validation or
  error handling that must run in production.

## Assertion Placement

### Function Entry Points

Assert preconditions on internal helper functions that are not decorated with
`@require`. Include the actual values that violated expectations:

```python
def _process_batch(items: list[Item], batch_id: str) -> Result:
    assert len(items) > 0, f"expected non-empty items for batch {batch_id}"
    assert all(item.valid for item in items), (
        f"invalid items in batch {batch_id}: "
        f"{[i.id for i in items if not i.valid]}"
    )
    ...
```

### Loop Invariants

Assert conditions that must hold at the start or end of each loop iteration:

```python
def _consume_events(events: Iterable[Event], budget: int) -> list[Result]:
    results: list[Result] = []
    remaining = budget

    for event in events:
        assert remaining >= 0, f"budget underflow: {remaining} after {len(results)} events"
        cost = event.compute_cost()
        remaining -= cost
        results.append(process(event))

    assert remaining >= 0, f"final budget check failed: {remaining}"
    return results
```

### State Transitions

Assert that state changes produce valid new states:

```python
def _transition_session(session: Session, event: Event) -> Session:
    old_version = session.version
    new_session = session.apply(event)

    assert new_session.version > old_version, (
        f"version must increase: {old_version} -> {new_session.version}"
    )
    assert new_session.event_count == session.event_count + 1, (
        f"event count mismatch: expected {session.event_count + 1}, "
        f"got {new_session.event_count}"
    )
    return new_session
```

### Data Structure Invariants

Assert structural properties after construction or mutation:

```python
def _build_index(items: Sequence[Item]) -> dict[str, Item]:
    index = {item.id: item for item in items}

    assert len(index) == len(items), (
        f"duplicate IDs detected: {len(items)} items, {len(index)} unique IDs"
    )
    return index
```

### Type Narrowing

Assert to narrow types after conditional checks where the type system cannot
infer the narrowing:

```python
def _extract_payload(message: Message) -> Payload:
    assert message.payload is not None, (
        f"message {message.id} missing payload (type={message.type})"
    )
    return message.payload
```

## Message Structure

### Required Elements

Every assertion message must include:

1. **What was expected** (implied by the assertion condition)
2. **What was found** (actual values that violated the expectation)
3. **Relevant identifiers** (IDs, keys, indices that locate the problem)

### Message Patterns

Use f-strings for interpolation. For complex conditions, break the message
across multiple lines:

```python
# Simple: single value check
assert count > 0, f"expected positive count, got {count}"

# With identifier context
assert user.active, f"user {user.id} must be active for operation {op_name}"

# Multi-value context
assert start < end, f"invalid range [{start}, {end}) for slice {slice_id}"

# Complex: multiple values, formatted for readability
assert all(item.valid for item in batch), (
    f"invalid items in batch {batch_id}: "
    f"failed={[i.id for i in batch if not i.valid]}, "
    f"total={len(batch)}"
)
```

### Avoid These Patterns

```python
# BAD: No message
assert len(items) > 0

# BAD: Message without context
assert len(items) > 0, "items must not be empty"

# BAD: Description without values
assert x < y, "x must be less than y"

# GOOD: Values included
assert x < y, f"expected x < y, got x={x}, y={y}"
```

## Production Considerations

### Compilation Behavior

Python's `-O` flag removes assertions entirely. This behavior is intentional
and should be leveraged:

- **Development/Testing**: Assertions run, catching bugs early
- **Production**: Assertions compiled out, zero runtime cost

Do not rely on assertions for:

- Input validation from external sources
- Error handling that must occur in production
- Security checks

### Conversion to Monitoring

For assertions that should become production monitoring, use a helper that
logs and optionally raises:

```python
from weakincentives.dbc import require

# For public API boundaries, use DbC (always runs in tests)
@require(lambda items: len(items) > 0, "items must not be empty")
def process_batch(items: list[Item]) -> Result:
    ...

# For internal checks that should monitor in production, use logging
def _internal_process(items: list[Item], batch_id: str) -> Result:
    if len(items) == 0:
        logger.error("empty batch received", extra={"batch_id": batch_id})
        raise ValueError(f"empty batch {batch_id}")
    ...
```

### Assertions vs. Exceptions

| Scenario | Use |
| --------------------------------- | --------------------- |
| Programming error (bug) | `assert` |
| Invalid external input | `raise ValueError` |
| Recoverable runtime condition | `raise` specific error |
| Contract on public API | `@require`/`@ensure` |
| Internal invariant documentation | `assert` |

## Integration with Testing

### Coverage Considerations

Assertions contribute to branch coverage. Ensure tests exercise both the
passing and failing paths:

```python
def test_batch_rejects_empty():
    """Assert fires on empty batch."""
    with pytest.raises(AssertionError, match="expected non-empty items"):
        _process_batch([], batch_id="test-123")
```

### DbC Interaction

When DbC is enabled during testing (`WEAKINCENTIVES_DBC=1`), both DbC
decorators and inline assertions run. They serve complementary purposes:

- DbC decorators: Formal contracts on public APIs, tested automatically
- Inline assertions: Internal invariants, require explicit test coverage

## Enforcement

### Code Review Guidelines

Reviewers should verify:

1. Assertions include descriptive messages with relevant values
2. Assertions are placed at boundaries (entry, loops, transitions)
3. Assertions are not used for production error handling
4. Complex assertions use multi-line formatting for readability

### Linting

Consider adding a custom ruff rule or pre-commit hook to detect bare
assertions without messages:

```python
# This pattern should be flagged:
assert condition

# This pattern is acceptable:
assert condition, "message with context"
```

## Examples from the Codebase

### Session State Validation

```python
def _apply_reducer(
    state: SliceView[S],
    event: E,
    reducer: Reducer[S, E],
) -> SliceOp[S]:
    assert state.latest() is not None or state.all() == (), (
        f"invalid state: latest={state.latest()}, all={len(state.all())} items"
    )
    result = reducer(state, event)
    assert isinstance(result, (Append, Replace)), (
        f"reducer returned {type(result).__name__}, expected Append or Replace"
    )
    return result
```

### Tool Handler Validation

```python
def _execute_tool(
    name: str,
    params: dict[str, Any],
    context: ToolContext,
) -> ToolResult[Any]:
    assert name in TOOL_REGISTRY, f"unknown tool: {name!r}"
    assert "action" in params, f"missing 'action' param for tool {name}"

    handler = TOOL_REGISTRY[name]
    result = handler(params, context=context)

    assert isinstance(result, ToolResult), (
        f"handler {name} returned {type(result).__name__}, expected ToolResult"
    )
    return result
```

### Event Processing Pipeline

```python
def _dispatch_events(
    events: Sequence[Event],
    handlers: Mapping[str, Handler],
) -> list[Outcome]:
    assert len(events) > 0, "dispatch called with empty event list"
    assert all(e.type in handlers for e in events), (
        f"unhandled event types: {[e.type for e in events if e.type not in handlers]}"
    )

    outcomes: list[Outcome] = []
    for i, event in enumerate(events):
        assert event.id not in {o.event_id for o in outcomes}, (
            f"duplicate event ID at index {i}: {event.id}"
        )
        outcomes.append(handlers[event.type](event))

    assert len(outcomes) == len(events), (
        f"outcome count mismatch: {len(outcomes)} != {len(events)}"
    )
    return outcomes
```

## Summary

High assertion density is a low-cost, high-value defensive programming
technique. By asserting at boundaries with structured messages that include
relevant context, bugs are caught closer to their source. This approach
complements the formal DbC decorators used on public APIs and integrates
naturally with existing code without requiring architectural changes.

Key takeaways:

1. Always include a message with relevant values
2. Assert at boundaries: entry, loops, transitions
3. Use DbC for public APIs, assertions for internal invariants
4. Do not use assertions for production error handling
5. Test both passing and failing assertion paths

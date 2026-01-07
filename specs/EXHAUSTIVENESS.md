# Exhaustiveness Checking Specification

## Purpose

When you add a new variant to a union type, every match statement handling that
union must be updated. Without exhaustiveness checking, missing handlers are
discovered at runtime. The `assert_never` pattern catches them at type-check
time.

## The Pattern

End match statements on union types with an `assert_never` sentinel:

```python
from typing import assert_never

match op:
    case Append(item=item):
        slice_instance.append(item)
    case Extend(items=items):
        slice_instance.extend(items)
    case Replace(items=items):
        slice_instance.replace(items)
    case Clear(predicate=pred):
        slice_instance.clear(pred)
    case _ as unreachable:  # pragma: no cover - exhaustiveness sentinel
        assert_never(unreachable)  # pyright: ignore[reportUnreachable]
```

If a new variant is added to `SliceOp`, pyright immediately reports a type
error at the `assert_never` call.

## Pyright Integration

The sentinel is intentionally unreachable when all cases are covered. Suppress
the strict mode warnings with inline comments:

- `# pyright: ignore[reportUnreachable]` - suppresses unreachable code error
- `# pragma: no cover` - excludes from coverage requirements

## When to Use

**Use `assert_never` for:** Match statements on union types.

**Skip for isinstance chains:** Pyright's type narrowing already handles
exhaustiveness. After checking `SectionOverride` and `ToolOverride`, pyright
knows the else branch is `TaskExampleOverride`. Adding a 4th type would change
the narrowed type, making the issue visible.

## Critical Union Types

- `SliceOp[T]` - `Session._apply_slice_op()` uses the pattern
- `DataEvent` - Subscription routing (handled by explicit subscriptions)

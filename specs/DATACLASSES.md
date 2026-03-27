# Dataclass Utilities Specification

## Purpose

Dependency-free dataclass utilities for serialization, validation, and immutable
patterns. Covers serde helpers, the `FrozenDataclass` decorator, and the
`Constructable` base for validated frozen dataclasses.

**Implementation:** `src/weakincentives/serde/`, `src/weakincentives/dataclasses/`

______________________________________________________________________

## Immutability

### Two Tiers

The codebase uses frozen dataclasses pervasively. Not all of them need
validation or normalization, so the design separates into two tiers:

**Tier 1 — Simple value objects.** Use `@FrozenDataclass()` alone. Direct
construction via `MyClass(...)` works normally. No base class needed.

```python
@FrozenDataclass()
class TokenUsage:
    input_tokens: int
    output_tokens: int
```

**Tier 2 — Validated / normalized classes.** Inherit from `Constructable`,
define a `create()` classmethod. Direct construction is blocked at runtime.

```python
@FrozenDataclass()
class Deadline(Constructable):
    expires_at: datetime
    started_at: datetime
    clock: WallClock = field(default=SYSTEM_CLOCK, repr=False, compare=False)

    @classmethod
    def create(
        cls,
        expires_at: datetime,
        started_at: datetime | None = None,
        clock: WallClock = SYSTEM_CLOCK,
    ) -> Deadline:
        if expires_at.tzinfo is None:
            raise ValueError("expires_at must be timezone-aware")
        now = clock.utcnow()
        if started_at is None:
            started_at = now
        if expires_at <= now:
            raise ValueError("expires_at must be in the future")
        with allow_construction():
            return cls(expires_at=expires_at, started_at=started_at, clock=clock)
```

### When to Use Each Tier

| Tier | Criteria |
|------|----------|
| **Tier 1** (`@FrozenDataclass()` only) | Fields are stored exactly as declared. No validation beyond type hints. No derived fields. |
| **Tier 2** (`Constructable`) | Any of: input types differ from stored types, fields need runtime validation, derived fields are computed from other fields, construction invariants must be enforced. |

______________________________________________________________________

## FrozenDataclass Decorator

```python
from weakincentives.dataclasses import FrozenDataclass
```

`@FrozenDataclass()` applies `@dataclass` with `frozen=True, slots=True`.
Freezing is mandatory and cannot be overridden — all `FrozenDataclass`
instances are immutable. Other standard dataclass options can be customized:

```python
@FrozenDataclass(order=True)     # Enable ordering comparisons
class Version:
    major: int
    minor: int
```

Uses `@dataclass_transform` (PEP 681) so pyright, mypy, and other type
checkers understand the generated `__init__`, `__eq__`, etc.

When the decorated class inherits from `Constructable`, the decorator
additionally wraps `__init__` with a guard that rejects direct construction
(see below).

______________________________________________________________________

## Constructable Base

```python
from weakincentives.dataclasses import Constructable, allow_construction
```

`Constructable` is a mixin for Tier 2 classes. It provides:

1. **`__init__` guard** — direct `MyClass(...)` raises `TypeError`
2. **`replace(**changes)`** — functional update via `create()`
3. **`allow_construction()`** — context manager for use inside `create()`

### The __init__ Guard

When `@FrozenDataclass()` detects that a class inherits from `Constructable`,
it wraps the generated `__init__` with a runtime check. Construction is only
permitted inside an `allow_construction()` context:

```python
deadline = Deadline(...)          # TypeError: use Deadline.create()
deadline = Deadline.create(...)   # OK
```

The guard uses a `ContextVar[bool]`, making it thread-safe and async-safe.
Nested `allow_construction()` contexts work correctly via token-based reset.

### The create() Classmethod

Each `Constructable` subclass defines a `create()` classmethod. This is the
**only** public construction path. It:

1. Accepts raw input types (may differ from stored field types)
2. Validates all preconditions
3. Normalizes values to their stored representation
4. Computes derived fields
5. Calls `cls(...)` inside `allow_construction()` with fully prepared values

```python
@classmethod
def create(cls, ...) -> Self:
    # 1. Validate
    # 2. Normalize
    # 3. Derive
    with allow_construction():
        return cls(validated_field=..., derived_field=..., ...)
```

**Type safety.** Because `create()` calls `cls(...)` directly (not an untyped
helper), the type checker verifies field names and types against `__init__`.
A typo like `cls(naem=...)` is caught statically.

**No `__post_init__`.** Tier 2 classes must not define `__post_init__`. All
logic belongs in `create()`. The decorator enforces this — defining
`__post_init__` on a `Constructable` subclass raises `TypeError` at class
definition time.

### Input Types vs Stored Types

A key benefit: `create()` can accept different types than what the class
stores. The gap is explicit and type-safe:

```python
@FrozenDataclass()
class Snapshot(Constructable):
    created_at: datetime                    # Always tz-aware
    slices: SnapshotState                   # Always MappingProxyType
    tags: Mapping[str, str]                 # Always MappingProxyType

    @classmethod
    def create(
        cls,
        created_at: datetime,               # May be naive
        slices: Mapping[...] | None = None, # May be mutable dict
        tags: Mapping[str, str] | None = None,
    ) -> Snapshot:
        with allow_construction():
            return cls(
                created_at=_ensure_timezone(created_at),
                slices=MappingProxyType(dict(slices or {})),
                tags=MappingProxyType(dict(tags or {})),
            )
```

### Derived Fields

Fields computed from other fields are regular init parameters (not
`field(init=False)`). Since direct construction is blocked, nobody can pass
an incorrect value:

```python
@FrozenDataclass()
class DispatchResult(Constructable):
    event: object
    handlers_invoked: tuple[EventHandler, ...]
    errors: tuple[HandlerFailure, ...]
    handled_count: int                      # Derived, but a regular field

    @classmethod
    def create(
        cls,
        event: object,
        handlers_invoked: tuple[EventHandler, ...],
        errors: tuple[HandlerFailure, ...],
    ) -> DispatchResult:
        with allow_construction():
            return cls(
                event=event,
                handlers_invoked=handlers_invoked,
                errors=errors,
                handled_count=len(handlers_invoked),
            )
```

`handled_count` is not a `create()` parameter. It cannot be set by callers.
It cannot be changed via `replace()`. It is always recomputed.

______________________________________________________________________

## replace()

`replace()` is the functional update method on `Constructable` instances. It
creates a modified copy by delegating to `create()`:

```python
d2 = deadline.replace(expires_at=new_time)
```

### How It Works

1. Introspect `create()`'s signature to discover the parameter names
2. Read each parameter's current value from the instance via `getattr`
3. Overlay the caller's `**changes`
4. Call `cls.create(**merged)` — all validation and derivation re-runs

```python
def replace(self, **changes: object) -> Self:
    cls = type(self)
    sig = inspect.signature(cls.create)
    create_params = {
        name for name, p in sig.parameters.items()
        if name != "cls"
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }

    # Reject unknown fields
    unknown = set(changes) - create_params
    if unknown:
        raise TypeError(f"unexpected field(s): {sorted(unknown)}")

    # Merge current values with changes, delegate to create()
    current = {name: getattr(self, name) for name in create_params}
    current.update(changes)
    return cls.create(**current)
```

### The Idempotency Requirement

Because `replace()` feeds stored values back through `create()`, normalization
must be idempotent. Concretely: if `create()` normalizes a value, applying
that normalization to an already-normalized value must produce an equivalent
result.

| Pattern | Idempotent? | Why |
|---------|-------------|-----|
| `_ensure_timezone(dt)` on tz-aware dt | Yes | Returns input unchanged |
| `MappingProxyType(dict(proxy))` | Yes | Creates equivalent proxy |
| `value.strip()` on stripped string | Yes | Returns input unchanged |
| `tuple(sequence)` on tuple | Yes | Returns equivalent tuple |

If a class cannot satisfy this (e.g., stored type is fundamentally
incompatible with `create()` input type), override `replace()` to raise
`NotImplementedError` with a clear message.

### When replace() Doesn't Apply

Some classes (e.g., `Tool`, `PromptTemplate`) have complex construction that
resolves type arguments from class specialization or transforms input types
structurally (e.g., `Section` → `SectionNode`). For these:

```python
def replace(self, **changes: object) -> Self:
    raise NotImplementedError(
        f"{type(self).__name__} does not support replace(). "
        f"Construct a new instance via create()."
    )
```

This is explicit and honest. Not every class needs functional update.

______________________________________________________________________

## Serde Integration

`serde.parse()` constructs dataclass instances via `cls(**kwargs)`. For
`Constructable` subclasses this hits the `__init__` guard.

The fix: `serde.parse()` detects `Constructable` and wraps construction in
`allow_construction()`:

```python
if issubclass(target_cls, Constructable):
    with allow_construction():
        instance = target_cls(**kwargs)
else:
    instance = target_cls(**kwargs)
```

This is correct because serde deserializes already-validated data from a
trusted store. It bypasses `create()` validation intentionally — the data was
validated when it was first created, and re-validation could have side effects
(e.g., reading the system clock).

The same applies to `serde.clone()`.

______________________________________________________________________

## Serde API

```python
from weakincentives.serde import parse, dump, clone, schema
```

### parse

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cls` | - | Target dataclass type (supports generic aliases) |
| `data` | - | Dict to parse |
| `extra` | `"ignore"` | `"ignore"` or `"forbid"` |
| `coerce` | `True` | Type coercion |

For generic dataclasses, use generic alias syntax: `parse(Wrapper[Data], data)`.

### dump

| Parameter | Default | Description |
|-----------|---------|-------------|
| `obj` | - | Dataclass instance |
| `by_alias` | `True` | Use alias names |
| `exclude_none` | `False` | Exclude None values |
| `computed` | `False` | Include `__computed__` props |

### clone / schema

- `clone` — wraps `dataclasses.replace` with validation re-run
- `schema` — emits JSON Schema (inlined, no `$ref`)

## Supported Types

Primitives, Enum, UUID, Path, Decimal, datetime/date/time, collections,
Literal, Union, Optional, nested dataclasses.

## Constraints and Transforms

Via `Annotated[..., {...}]` or `field(metadata=...)`:

| Key | Description |
|-----|-------------|
| `ge`, `gt`, `le`, `lt` | Numeric bounds |
| `min_length`, `max_length` | String/collection length |
| `pattern`, `regex` | Regex validation |
| `strip`, `lower`, `upper` | String normalisers |
| `in`, `not_in` | Membership validation |
| `validate`, `validators` | Custom validation |
| `convert`, `transform` | Value transformation |
| `alias` | Field-level alias for parsing/dumping |

______________________________________________________________________

## Generic Dataclass Serialization

Generic dataclasses are parsed using **generic alias** syntax so TypeVars
resolve at parse time. `parse(Wrapper[Data], data)` correctly creates a
`Wrapper` whose `payload` is typed as `Data`. Parsing without type arguments
raises a helpful error.

## Scoped Field Visibility

Fields can be hidden from schema generation and parsing in specific contexts
using scope-aware annotations. This enables post-processing patterns where
certain fields are populated after LLM evaluation (e.g., in `finalize()`).

### Why This Matters

LLMs should not know about implementation fields like `processing_time_ms` or
`model_version`. Without scoped visibility, these fields would appear in the
JSON schema sent to the LLM, confusing it and wasting context. With scoped
visibility, the LLM sees a clean schema while the full object (with populated
hidden fields) is stored and logged.

### SerdeScope

Two scopes at `src/weakincentives/serde/_scope.py`:

- `SerdeScope.DEFAULT` — standard serde, all fields visible
- `SerdeScope.STRUCTURED_OUTPUT` — LLM schema/parsing context; hidden fields excluded

### HiddenInStructuredOutput

`Annotated[type, HiddenInStructuredOutput()]` excludes a field from schema
generation and parsing when `scope=STRUCTURED_OUTPUT`. Hidden fields **must**
have a default value since the LLM cannot provide them.

```python
@dataclass
class AnalysisResult:
    summary: str                                              # LLM generates
    confidence: float                                         # LLM generates
    processing_time_ms: Annotated[int, HiddenInStructuredOutput()] = 0
    model_version: Annotated[str, HiddenInStructuredOutput()] = ""
```

### Integration with Structured Output Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Prompt Evaluation Flow                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Schema Generation (scope=STRUCTURED_OUTPUT)                  │
│     └─ HiddenInStructuredOutput fields excluded from LLM schema │
│                                                                  │
│  2. LLM Evaluation                                               │
│     └─ LLM generates values for visible fields only             │
│                                                                  │
│  3. Response Parsing (scope=STRUCTURED_OUTPUT)                   │
│     └─ Hidden fields use defaults (not expected in response)    │
│                                                                  │
│  4. finalize() Hook                                              │
│     └─ Populate hidden fields with computed values              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### finalize() Pattern

Override `AgentLoop.finalize()` to populate hidden fields after LLM evaluation.
See `src/weakincentives/runtime/agent_loop.py` for the base class and
`AGENT_LOOP.md` for usage patterns.

### Serialization Behavior

| Operation | Hidden Fields |
|-----------|---------------|
| `schema(scope=STRUCTURED_OUTPUT)` | Excluded |
| `schema(scope=DEFAULT)` | Included |
| `parse(scope=STRUCTURED_OUTPUT)` | Skipped (use defaults) |
| `parse(scope=DEFAULT)` | Included |
| `dump()` | **Always included** |
| `clone()` | Included (uses DEFAULT) |

`dump()` always serializes hidden fields to ensure logging and storage capture
complete state.

`HiddenInStructuredOutput` composes with other constraint annotations. Hidden
fields with nested dataclass types exclude the entire subtree from
`STRUCTURED_OUTPUT` scope. Validation hooks run after parsing; hidden fields
have defaults at parse time and receive real values in `finalize()`.

______________________________________________________________________

## Design Decisions

### Why not `__post_init__` + `object.__setattr__`?

This is the standard Python pattern for mutating frozen dataclasses after
construction. We reject it for Tier 2 classes because:

1. **Untyped.** `object.__setattr__(self, "name", value)` accepts any string.
   Typos are silent. The type checker cannot verify field names or value types.

2. **Input/stored type conflation.** The dataclass field declaration must serve
   as both the input type and the stored type. When these differ (e.g.,
   `datetime | None` input, `datetime` stored), the declaration lies about
   one of them.

3. **No construction guard.** Anyone can call `MyClass(...)` and bypass
   validation. The invariant is only enforced by convention.

### Why not a `_freeze(**attrs)` helper?

A method like `self._freeze(name=value)` is syntactically cleaner than raw
`object.__setattr__` but still untyped (`**kwargs: object`). It does not
solve problems 1–3 above. It is a cosmetic improvement, not a structural one.

### Why `create()` calls `cls(...)` instead of an untyped `_new()`?

Type safety. Inside `create()`:

```python
with allow_construction():
    return cls(expires_at=expires_at, started_at=started_at, clock=clock)
```

The type checker verifies `cls(...)` against the generated `__init__`
signature — catching typos, wrong types, and missing required fields at
static analysis time. An untyped `cls._new(**kwargs)` would bypass this.

### Why ContextVar for the guard?

- **Thread-safe:** each thread has its own value
- **Async-safe:** each asyncio task has its own value
- **Reentrant:** `token = var.set(True)` / `var.reset(token)` nests correctly
- **Zero cost when not guarding:** simple boolean check

### Why does replace() go through create()?

So validation and derivation re-run automatically. If `create()` computes
`handled_count = len(handlers_invoked)`, then `replace(handlers_invoked=...)`
recomputes it without any special wiring. One code path, one source of truth.

______________________________________________________________________

## Limitations

- No discriminated unions or `$ref` schema
- No assignment-time validation
- No external parsers
- `replace()` requires idempotent normalization in `create()`; classes that
  cannot satisfy this must override `replace()` to raise or provide a
  custom implementation
- **No backward compatibility guarantee for persisted data.** This is alpha
  software; serialization format may change between versions.

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
| **Tier 1** (`@FrozenDataclass()` only) | Fields are stored exactly as declared. No validation beyond type hints. No derived state. |
| **Tier 2** (`Constructable`) | Any of: input types differ from stored types, fields need runtime validation, construction invariants must be enforced. |

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
1. **`replace(**changes)`** — functional update via `create()`
1. **`allow_construction()`** — context manager for use inside `create()`

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
1. Validates all preconditions
1. Normalizes values to their stored representation
1. Calls `cls(...)` inside `allow_construction()` with fully prepared values

```python
@classmethod
def create(cls, ...) -> Self:
    # 1. Validate
    # 2. Normalize
    with allow_construction():
        return cls(validated_field=..., ...)
```

**Type safety.** Because `create()` calls `cls(...)` directly (not an untyped
helper), the type checker verifies field names and types against `__init__`.
A typo like `cls(naem=...)` is caught statically.

**No `__post_init__`.** Tier 2 classes must not define `__post_init__`. All
logic belongs in `create()`. The decorator enforces this — defining
`__post_init__` on a `Constructable` subclass raises `TypeError` at class
definition time.

### The create() == __init__ Contract

**Every `create()` parameter must map 1:1 to a dataclass field.** The
parameter names must match the field names and their types must be compatible.

This is the central invariant that makes the system work:

- **`replace()`** reads current field values via `getattr`, overlays changes,
  and calls `create(**merged)`. This only works if every `create()` parameter
  is a stored field.

- **`serde.parse()`** calls `create(**kwargs)` with deserialized data. This
  only works if `create()` accepts exactly the fields that serde extracted.

- **Round-tripping** (`dump` → `parse`) is lossless because every stored field
  has a corresponding `create()` parameter.

**Derived state must not be a dataclass field.** If a value is computed from
other fields, express it as a `@property` — not as a field that `create()`
sets but doesn't accept:

```python
@FrozenDataclass()
class DispatchResult(Constructable):
    event: object
    handlers_invoked: tuple[EventHandler, ...]
    errors: tuple[HandlerFailure, ...]

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
            )

    @property
    def handled_count(self) -> int:
        """Derived from handlers_invoked — not a stored field."""
        return len(self.handlers_invoked)

    @property
    def ok(self) -> bool:
        return not self.errors
```

`handled_count` is never stored, never serialized, and never passed to
`create()`. It is recomputed on every access. This is the correct pattern
for derived state.

### Class-Level Derived State (ClassVar)

Some classes resolve type metadata at specialization time (e.g., via
`__class_getitem__`). This state is per-class, not per-instance, and must
use `ClassVar` — not instance fields:

```python
@FrozenDataclass()
class Tool[ParamsT, ResultT](Constructable):
    name: str
    description: str
    handler: ToolHandler[ParamsT, ResultT] | None
    accepts_overrides: bool = True

    # Per-class, set by __class_getitem__:
    _specialized_params_type: ClassVar[type | None] = None
    _specialized_result_type: ClassVar[type | None] = None
    _specialized_result_container: ClassVar[Literal["object", "array"]] = "object"

    @property
    def params_type(self) -> type[ParamsT]:
        return cast(type[ParamsT], type(self)._specialized_params_type)
```

The `ClassVar` fields are invisible to `dataclasses`, so they don't appear in
`__init__`, `create()`, `replace()`, or serde. Properties provide typed access.

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

______________________________________________________________________

## replace()

`replace()` is the functional update method on `Constructable` instances. It
creates a modified copy by delegating to `create()`:

```python
d2 = deadline.replace(expires_at=new_time)
```

### How It Works

1. Look up `create()`'s parameter names (cached per-class via `@functools.cache`)
1. Verify every parameter corresponds to an instance attribute
1. Read each parameter's current value from the instance via `getattr`
1. Overlay the caller's `**changes`
1. Call `cls.create(**merged)` — all validation re-runs

The signature introspection result is cached in a module-level
`_create_param_names()` function decorated with `@functools.cache`, so the
cost is paid once per class, not per invocation.

### The Round-Trip Requirement

Every `create()` parameter must correspond to a stored instance field so
that `replace()` can read the current value back via `getattr`. If
`create()` accepts configuration-only parameters that are not stored as
fields (e.g. a `tax_rate` used only to compute a derived value), `replace()`
raises `TypeError` at runtime with guidance to override it:

```
TypeError: Order.replace() cannot round-trip create() parameter(s)
that are not instance fields: tax_rate. Override replace() in Order
to handle this.
```

This is intentional — silently dropping parameters would be a data loss
footgun. If the class needs functional update with non-field params,
override `replace()` with custom logic.

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
incompatible with `create()` input type), override `replace()` with custom
logic or raise `NotImplementedError` with a clear message.

______________________________________________________________________

## Serde Integration

`serde.parse()` calls `create()` for `Constructable` subclasses, so that
**all validation and normalization runs on deserialized data**:

```python
if issubclass(target_cls, Constructable):
    instance = cast(T, target_cls.create(**kwargs))
else:
    instance = target_cls(**kwargs)
```

This works because the `create() == __init__` contract guarantees that
`create()` accepts exactly the same keyword arguments that serde extracts
from the data. No derived fields, no extra parameters, no mismatch.

**Why call `create()` instead of bypassing it?** Deserialized data is not
inherently trusted. Calling `create()` ensures that timezone checks, name
validation, type resolution, and all other invariants are enforced regardless
of whether data arrives via the public API or from a serialized store.

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

1. **Input/stored type conflation.** The dataclass field declaration must serve
   as both the input type and the stored type. When these differ (e.g.,
   `datetime | None` input, `datetime` stored), the declaration lies about
   one of them.

1. **No construction guard.** Anyone can call `MyClass(...)` and bypass
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

So validation re-runs automatically. One code path, one source of truth. If
`create()` validates that `expires_at` is timezone-aware, then
`replace(expires_at=naive_dt)` catches the error without any special wiring.

### Why does serde.parse() call create()?

So that deserialized data is validated. The `create() == __init__` contract
guarantees parameter compatibility. This means Constructable invariants are
enforced uniformly — whether data arrives from user code, from `replace()`,
or from a serialized store.

### Why no derived fields as dataclass fields?

Derived fields break the `create() == __init__` contract. If `create()`
accepts 3 parameters but `__init__` has 4 fields (with one derived), then:

- `replace()` cannot round-trip: it reads 4 fields but `create()` only
  accepts 3.
- `serde.parse()` cannot call `create()`: it extracts 4 fields from data
  but `create()` only accepts 3.

Using `@property` for derived state avoids these issues entirely. The
property is never serialized, never passed to `create()`, and is always
recomputed on access.

### Why cache create() signature introspection?

`replace()` needs to know which parameters `create()` accepts. Calling
`inspect.signature()` on every `replace()` invocation is wasteful.
The `_create_param_names()` function uses `@functools.cache` to compute
the parameter names once per class and reuse them on subsequent calls.

______________________________________________________________________

## Checklist for New Constructable Subclasses

When writing a new Tier 2 class:

- [ ] `create()` parameters match `__init__` fields 1:1 (same names, compatible types)
- [ ] Derived state uses `@property`, not stored fields
- [ ] Per-class metadata uses `ClassVar`, not instance fields
- [ ] Normalization in `create()` is idempotent (for `replace()` round-trips)
- [ ] No `__post_init__` defined
- [ ] `allow_construction()` wraps the `cls(...)` call inside `create()`
- [ ] `replace()` works without override (or override is provided with tests)

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

# Dataclass Utilities Specification

## Purpose

Provide dependency-free dataclass utilities for serialization, validation, and
immutable patterns. This specification covers the serde helpers (`parse`,
`dump`, `clone`, `schema`) and the `FrozenDataclass` decorator.

## Guiding Principles

- **Dependency-free contracts**: Bridge stdlib `dataclasses` with validation
  without runtime decorators or third-party packages.
- **Predictable state**: Parsing and dumping remain deterministic for stable
  telemetry and provider payloads.
- **Metadata-first validation**: `field.metadata` and `Annotated` are the single
  source of truth for constraints.
- **Contract-friendly errors**: Path-aware failures with dotted/indexed notation.

## Serde API

```python
from weakincentives.serde import parse, dump, clone, schema
```

### parse

```python
parse(
    cls: type[T] | None,
    data,
    *,
    extra="ignore",           # "ignore", "forbid", "allow"
    coerce=True,
    case_insensitive=False,
    alias_generator=None,
    aliases=None,
    allow_dataclass_type=False,
    type_key="__type__",
) -> T
```

**Behavior:**

- Missing required fields raise `ValueError("Missing required field: 'field'")`
- Extras: `ignore` drops, `forbid` raises, `allow` attaches to instance
- Coercion converts strings to numerics, UUID, paths, enums, datetime, etc.
- Aliases: `aliases` arg > `field(metadata={"alias": ...})` > `alias_generator`
- Validation order: normalisers → numeric constraints → length → pattern →
  membership → validators → `convert`/`transform`
- Error paths use dotted notation: `"address.street"`, `"items[0].price"`

### dump

```python
dump(
    obj,
    *,
    by_alias=True,
    exclude_none=False,
    computed=False,
    include_dataclass_type=False,
    type_key="__type__",
    alias_generator=None,
) -> dict[str, Any]
```

**Behavior:**

- Serializes to JSON-safe primitives, recursing through nested dataclasses
- Enums emit values; datetime uses ISO format; UUID/Decimal/Path stringify
- `computed=True` materializes `__computed__` properties

### clone

```python
clone(obj, **updates) -> T
```

Wraps `dataclasses.replace` while preserving extras and re-running validation
hooks.

### schema

```python
schema(cls, *, alias_generator=None, extra="ignore") -> dict[str, Any]
```

Emits JSON Schema with inlined nested dataclasses (no `$ref`). Mirrors alias
resolution and propagates constraint metadata.

## Supported Types

- Dataclasses (including nested)
- Primitives: `str`, `int`, `float`, `bool`, `None`
- `Enum`, `UUID`, `Path`, `Decimal`
- `datetime`, `date`, `time`
- Collections: `list`, `set`, `tuple`, `dict`
- `Literal`, `Union`, `Optional`

## Constraints and Transforms

Merge keys from `Annotated[..., {...}]` and `field(metadata=...)`:

| Key | Description |
|-----|-------------|
| `ge`, `gt`, `le`, `lt` | Numeric bounds |
| `minimum`, `maximum` | Alias for `ge`, `le` bounds |
| `exclusiveMinimum`, `exclusiveMaximum` | Alias for `gt`, `lt` bounds |
| `min_length`, `max_length` | String/collection length |
| `minLength`, `maxLength` | Alias for length constraints |
| `pattern`, `regex` | Regex validation |
| `strip`, `lower`, `upper` | String normalisers |
| `lowercase`, `uppercase` | Alias for `lower`, `upper` |
| `in`, `enum` | Membership inclusion validation |
| `not_in` | Membership exclusion validation |
| `validate` | Single custom validation callable |
| `validators` | Iterable of validation callables |
| `convert`, `transform` | Final value transformation (aliases)

### Validation Hooks

Dataclasses may define:

- `__validate__` - Post-construction validation
- `__post_validate__` - After all fields validated
- `__computed__` - Property names for serialization

## FrozenDataclass Decorator

Lightweight decorator for immutable dataclasses:

```python
from weakincentives.dataclasses import FrozenDataclass

@FrozenDataclass()
class User:
    name: str
    slug: str
```

**Defaults:**

- `frozen=True`, `slots=True`
- `kw_only=False`, `order=False`, `eq=True`, `repr=True`
- `match_args=True`, `unsafe_hash=False`

### Pre-Construction Hook

```python
@FrozenDataclass()
class Invoice:
    total_cents: int
    tax_cents: int

    @classmethod
    def __pre_init__(cls, *, total_cents: int, tax_rate: float = 0.1, **_):
        return {
            "total_cents": total_cents,
            "tax_cents": int(total_cents * tax_rate),
        }
```

- Runs before `__init__`
- Returns mapping with all required fields
- Enables derivation without `__post_init__` mutation

### Copy Helpers

```python
updated = invoice.update(tax_rate=0.24)
from_mapping = invoice.merge({"tax_rate": 0.24})
remapped = invoice.map(lambda f: {"tax_cents": f["total_cents"] * 0.24})
```

- `update(**changes)` - Apply field changes
- `merge(mapping_or_obj)` - Merge from dict or object
- `map(transform)` - Transform via callable

All helpers:

- Return new instances (immutable)
- Re-run `__post_init__` for invariants
- Raise `TypeError` for unknown fields
- Do NOT invoke `__pre_init__`

## Examples

### Parsing

```python
from dataclasses import dataclass, field
from typing import Annotated
from uuid import UUID
from weakincentives.serde import parse

@dataclass
class User:
    user_id: UUID = field(metadata={"alias": "id"})
    name: Annotated[str, {"min_length": 1, "strip": True}]

user = parse(
    User,
    {"ID": "a9f95576...", "name": "  Ada  "},
    case_insensitive=True,
)
```

### Frozen with Derived Fields

```python
from weakincentives.dataclasses import FrozenDataclass

@FrozenDataclass()
class Order:
    subtotal: int
    tax: int
    total: int

    @classmethod
    def __pre_init__(cls, *, subtotal: int, tax_rate: float = 0.1, **_):
        tax = int(subtotal * tax_rate)
        return {
            "subtotal": subtotal,
            "tax": tax,
            "total": subtotal + tax,
        }

    def __post_init__(self):
        if self.total != self.subtotal + self.tax:
            raise ValueError("Total mismatch")

order = Order(subtotal=1000)
assert order.total == 1100
```

## Edge Cases

- **Slots vs extras**: With `slots=True`, extras fall back to `__extras__` dict
- **Empty strings**: With coercion, treated as missing for optional fields
- **Union errors**: Only final branch error is surfaced
- **Datetime parsing**: Uses `fromisoformat` without timezone coercion
- **Schema reuse**: Nested dataclasses are inlined, no `$ref`

## Limitations

- No discriminated unions or `$ref` schema components
- No assignment-time validation
- No external parsers (dateutil, etc.)
- Validators may mutate values; ensure they return transformed values

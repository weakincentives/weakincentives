# Dataclass Utilities Specification

## Purpose

Dependency-free dataclass utilities for serialization, validation, and immutable
patterns. Covers serde helpers and `FrozenDataclass` decorator.

**Implementation:** `src/weakincentives/serde/`, `src/weakincentives/dataclasses/`

## Serde API

```python
from weakincentives.serde import parse, dump, clone, schema
```

### parse

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cls` | - | Target dataclass type (supports generic aliases) |
| `data` | - | Dict to parse |
| `extra` | `"ignore"` | `"ignore"`, `"forbid"`, `"allow"` |
| `coerce` | `True` | Type coercion |
| `case_insensitive` | `False` | Case-insensitive keys |
| `aliases` | `None` | Field aliases |
| `alias_generator` | `None` | Function to transform field names |

For generic dataclasses, use generic alias syntax: `parse(Wrapper[Data], data)`.

### dump

| Parameter | Default | Description |
|-----------|---------|-------------|
| `obj` | - | Dataclass instance |
| `by_alias` | `True` | Use alias names |
| `exclude_none` | `False` | Exclude None values |
| `computed` | `False` | Include `__computed__` props |
| `alias_generator` | `None` | Function to transform field names |

### clone

Wraps `dataclasses.replace` with validation re-run.

### schema

Emits JSON Schema (inlined, no `$ref`).

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

## FrozenDataclass Decorator

```python
from weakincentives.dataclasses import FrozenDataclass

@FrozenDataclass()
class User:
    name: str
```

Defaults: `frozen=True`, `slots=True`

### Pre-Construction Hook

```python
@FrozenDataclass()
class Order:
    subtotal: int
    tax: int
    total: int

    @classmethod
    def __pre_init__(cls, *, subtotal: int, tax_rate: float = 0.1, **_):
        tax = int(subtotal * tax_rate)
        return {"subtotal": subtotal, "tax": tax, "total": subtotal + tax}
```

### Copy Helpers

| Method | Description |
|--------|-------------|
| `update(**changes)` | Apply field changes |
| `merge(mapping)` | Merge from dict/object |
| `map(transform)` | Transform via callable |

All return new instances (immutable).

## Generic Dataclass Serialization

Generic dataclasses like `Wrapper[T]` are parsed using **generic alias** syntax.
The type arguments resolve TypeVar fields during parsing.

```python
@dataclass
class Wrapper[T]:
    payload: T

@dataclass
class Data:
    value: int

# Serialize - standard dump
data = dump(Wrapper(payload=Data(1)))
# {"payload": {"value": 1}}

# Parse with generic alias to specify type parameter
restored = parse(Wrapper[Data], data)
assert isinstance(restored.payload, Data)
```

When parsing, TypeVar fields are resolved from the generic alias type arguments.
Parsing without type arguments raises a helpful error guiding usage.

## Limitations

- No discriminated unions or `$ref` schema
- No assignment-time validation
- No external parsers

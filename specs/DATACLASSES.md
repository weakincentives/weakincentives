# Dataclass Utilities Specification

Dependency-free serialization, validation, and immutable patterns.

**Source:** `src/weakincentives/serde.py`, `src/weakincentives/dataclasses.py`

## Serde API

```python
from weakincentives.serde import parse, dump, clone, schema
```

### parse

```python
parse(cls, data, *, extra="ignore", coerce=True, case_insensitive=False, ...)
```

- Missing required fields raise `ValueError`
- Extras: `ignore` (drop), `forbid` (raise), `allow` (attach)
- Coerces strings to numerics, UUID, Path, Enum, datetime
- Error paths: `"address.street"`, `"items[0].price"`

### dump

```python
dump(obj, *, by_alias=True, exclude_none=False, computed=False, ...)
```

Serializes to JSON-safe primitives. Enums emit values; datetime uses ISO.

### clone / schema

```python
clone(obj, **updates)  # dataclasses.replace + validation
schema(cls)            # JSON Schema (inlined, no $ref)
```

## Constraints

Via `Annotated[..., {...}]` or `field(metadata=...)`:

| Key | Purpose |
|-----|---------|
| `ge`, `gt`, `le`, `lt` | Numeric bounds |
| `min_length`, `max_length` | String/collection length |
| `pattern` | Regex validation |
| `strip`, `lower`, `upper` | String normalization |
| `in`, `not_in` | Membership validation |
| `validate`, `validators` | Custom callables |
| `convert`, `transform` | Value transformation |

**Validation order:** normalisers → bounds → length → pattern → membership → validators → transform

## FrozenDataclass

**Definition:** `dataclasses.py:FrozenDataclass`

```python
@FrozenDataclass()
class User:
    name: str
    slug: str
```

Defaults: `frozen=True`, `slots=True`, `kw_only=False`

### Pre-Construction Hook

```python
@FrozenDataclass()
class Order:
    subtotal: int
    tax: int

    @classmethod
    def __pre_init__(cls, *, subtotal: int, tax_rate: float = 0.1, **_):
        return {"subtotal": subtotal, "tax": int(subtotal * tax_rate)}
```

### Copy Helpers

```python
obj.update(**changes)   # Field changes
obj.merge(mapping)      # From dict/object
obj.map(transform)      # Via callable
```

All return new instances and re-run `__post_init__`.

## Validation Hooks

| Hook | Purpose |
|------|---------|
| `__validate__` | Post-construction validation |
| `__post_validate__` | After all fields validated |
| `__computed__` | Property names for serialization |

## Limitations

- No discriminated unions or `$ref`
- No assignment-time validation
- No external parsers

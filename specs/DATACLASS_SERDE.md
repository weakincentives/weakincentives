# Dataclass Utilities

This specification covers both the `FrozenDataclass` decorator and the serde
(serialization/deserialization) helpers.

______________________________________________________________________

## Frozen Dataclasses

### Motivation

A lightweight decorator makes immutable dataclasses easy to author without
leaning on the serde helpers or invoking `__setattr__` in `__post_init__`.

### Goals

- **Immutability by default**: Apply `frozen=True` and `slots=True` so instances
  behave predictably and stay lean.
- **Pre-construction shaping**: Provide a hook to derive or normalise inputs
  before the dataclass initialises.
- **Ergonomic copy-on-write**: Mirror `dataclasses.replace` via small helpers.
- **Serde independence**: Keep the decorator free of `weakincentives.serde`.

### Decorator Surface

`@FrozenDataclass` wraps the stdlib `@dataclass` with frozen defaults:

```python
from weakincentives.dataclasses import FrozenDataclass

@FrozenDataclass()
class Invoice:
    total_cents: int
    tax_rate: float
    tax_cents: int
    grand_total_cents: int
```

- **Defaults**: `frozen=True`, `slots=True`, `kw_only=False`, `order=False`,
  `eq=True`, `repr=True`. Keyword arguments pass through to `dataclasses.dataclass`.
- **Module location**: `src/weakincentives/dataclasses/`.

### Input Normalisation Hook

Models can define an optional `__pre_init__(cls, **kwargs) -> dict[str, Any]`
classmethod. When present, the decorator calls it before the generated
`__init__`, using the returned mapping as the constructor inputs:

- Deriving fields from others (e.g., computing `tax_cents` from `total_cents`).
- Normalising or defaulting values without mutation in `__post_init__`.

Rules:

- Must accept keyword-only arguments matching the dataclass fields.
- The returned mapping must provide every required field.
- Runs before validation in `__post_init__`.

### Copy Helpers

The decorator injects ergonomic copy-on-write helpers:

- `update(**changes)`: Constructs a new instance with the specified changes,
  re-running `__post_init__`.
- `merge(mapping_or_obj)`: Merges fields from a mapping or object.
- `map(transform)`: Provides current fields to a callable, applies replacements.

```python
updated = invoice.update(tax_rate=0.24)
from_mapping = invoice.merge({"tax_rate": 0.24})
remapped = invoice.map(lambda fields: {"tax_cents": fields["total_cents"] * 0.24})
```

These helpers do **not** invoke `__pre_init__`; callers should recompute
dependent fields explicitly.

### Frozen Dataclass Example

```python
from dataclasses import field
from weakincentives.dataclasses import FrozenDataclass

@FrozenDataclass()
class User:
    name: str
    slug: str
    tags: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def __pre_init__(cls, *, name: str, slug: str | None = None, tags=()):
        base = slug or name
        return {
            "name": name.strip(),
            "slug": base.lower().replace(" ", "-"),
            "tags": tuple(tags),
        }

    def __post_init__(self):
        if not self.name:
            raise ValueError("name is required")

user = User(name=" Ada Lovelace ")
updated = user.update(tags=("pioneer",))
```

______________________________________________________________________

## Serde Utilities

### Guiding Principles

- **Dependency-free contracts**: Bridge standard library `dataclasses` with validation and serialisation guarantees without
  runtime decorators or third-party packages.
- **Predictable, inspectable state**: Keep parsing and dumping deterministic so agent telemetry, provider payloads, and
  cached tool responses remain stable.
- **Metadata-first validation**: Treat `field.metadata` and `typing.Annotated` as the single source of truth for constraints
  and transforms; favour declarative configuration over imperative guards.
- **Ergonomic defaults with explicit escape hatches**: Coercion, alias resolution, and extras handling come pre-wired but can
  be tightened (`extra="forbid"`, `coerce=False`) or relaxed (`extra="allow"`) per call.
- **Contract-friendly errors**: Surface path-aware failures with dotted/indexed notation so DbC tests and callers can assert
  on precise failure modes.

## Scope and Ownership

- **Modules covered**: `weakincentives.serde.parse`, `.dump`, `.clone`, and `.schema` implement the behaviours described here.
- **Supported types**: dataclasses (including nested), stdlib primitives, `Enum`, `UUID`, `Path`, `Decimal`,
  `datetime`/`date`/`time`, collections (`list`, `set`, `tuple`, `dict`), `Literal`, and `Union`/`Optional` constructs.
- **Constraints and transforms**: Merge keys from `Annotated[..., {...}]` and `field(metadata=...)`; `Annotated` wins on
  conflicts. Recognised keys include numeric bounds (`ge`, `gt`, `le`, `lt`, `minimum`, `maximum`), length limits,
  membership checks, regex `pattern`, string normalisers (`strip`, `lower`, `upper`), callable `validate`/`validators`, and a
  final `convert`/`transform` hook.
- **Hooks**: Dataclasses may define `__validate__` and `__post_validate__` for post-construction checks. `__computed__`
  lists property names eligible for serialisation when requested.
- **Out of scope**: Discriminated unions, `$ref` schema components, and assignment-time validation are intentionally
  unsupported. External parsers (for example, `python-dateutil`) are not used; conversions rely on stdlib behaviour.

## API Reference

Import helpers directly from the package:

```python
from weakincentives.serde import parse, dump, clone, schema
```

### `parse`

```python
parse(
    cls: type[T] | None,
    data,
    *,
    extra="ignore",
    coerce=True,
    case_insensitive=False,
    alias_generator=None,
    aliases=None,
    allow_dataclass_type=False,
    type_key="__type__",
) -> T
```

- Accepts a dataclass type (or resolves from the payload when `allow_dataclass_type=True`) and a mapping payload. Missing required inputs raise `ValueError("Missing required field: 'field'")`.
- **Extras**: `extra` may be `"ignore"`, `"forbid"`, or `"allow"`. Allowed extras attach to the instance or fall back to
  `__extras__` when slots block dynamic attributes.
- **Coercion**: When enabled, converts strings to numerics, UUID, paths, Enums, date/time objects, nested dataclasses,
  collections, unions, and literals. Optional unions treat `None` and empty strings as missing inputs when coercion is on.
- **Aliases**: Resolution order is `aliases` argument → `field(metadata={"alias": ...})` → `alias_generator` callback.
  `case_insensitive=True` enables case folding across all alias sources.
- **Validation order**: Apply string normalisers first, then constraint/validator checks, then `convert` / `transform` for the
  final value. Model-level hooks run after construction and extras assignment.
- **Error reporting**: Path-aware messages use dotted/indexed keys such as `"address.street"` or `"line_items[0].price"`.
- **Union semantics**: Branches are attempted in declaration order; the last failure surfaces when all branches fail.
- **Type references**: When `allow_dataclass_type=True`, parse reads a type identifier from `type_key` (defaulting to
  `"__type__"`) and resolves it in the form `"{module}:{qualname}"`. The resolved type must be a dataclass and match the
  provided `cls` when present; mismatches raise `TypeError`.

### `dump`

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

- Serialises dataclasses into JSON-safe primitives, recursing through nested dataclasses and collections.
- **Aliases**: Uses field aliases or `alias_generator` when `by_alias=True`; otherwise falls back to field names.
- **Type handling**: Enums emit their values; `datetime`/`date`/`time` use ISO formatting; `UUID`/`Decimal`/`Path` stringify.
- **Nullability**: `exclude_none=True` prunes `None` values throughout the payload.
- **Computed fields**: When `computed=True`, properties listed in `__computed__` materialise under the same alias policy.
- **Type references**: When `include_dataclass_type=True`, dump embeds the dataclass type identifier at `type_key` using the
  `"{module}:{qualname}"` format so callers can round-trip instances without supplying an explicit target class to
  `parse()`.
- **Runtime contract**: Tool payloads depend on `dump(..., exclude_none=True)` for `Result.render()`, so regressions here
  surface as telemetry or provider mismatches.

### `clone`

```python
clone(obj, **updates) -> T
```

- Wraps `dataclasses.replace` to apply updates while preserving extras. Re-runs `__validate__` / `__post_validate__` hooks
  to keep invariants intact.

### `schema`

```python
schema(cls, *, alias_generator=None, extra="ignore") -> dict[str, Any]
```

- Emits JSON Schema describing the dataclass structure, inlining nested dataclasses instead of `$ref` components.
- Mirrors alias resolution rules and marks additional properties forbidden only when `extra="forbid"`.
- Propagates constraint metadata into schema keywords (numeric bounds, length, membership, `pattern`).

## Behaviour Map

| Principle | Implementation Touchpoints |
| --- | --- |
| Dependency-free contracts | `src/weakincentives/serde/__init__.py` re-exports helpers; parsing and dumping rely solely on stdlib types and helper functions within `src/weakincentives/serde/` modules. |
| Metadata-first validation | Constraint extraction and merge logic live in `src/weakincentives/serde/parse.py`; enforcement flows through `parse` in the same module. |
| Deterministic coercion | Coercers and collection handlers reside in `src/weakincentives/serde/parse.py` alongside the main parsing logic. |
| Alias-aware IO | Alias plumbing is handled in `src/weakincentives/serde/parse.py` and `src/weakincentives/serde/dump.py`. |
| Path-aware errors | Error tracking utilities are integrated in `src/weakincentives/serde/parse.py`, surfacing dotted/indexed paths. |
| Schema parity with runtime | `src/weakincentives/serde/schema.py` mirrors parse-time constraints into JSON Schema output. |
| Extras preservation | Extras attachment and cloning behaviours are implemented in `src/weakincentives/serde/parse.py` and `src/weakincentives/serde/dump.py` (which contains `clone`). |
| Contract verification | Tests in `tests/serde/test_dataclass_serde.py` and `tests/plugins/dataclass_serde.py` enforce the documented behaviours. |

## Edge Cases and Caveats

- **Slots vs extras**: When dataclasses use `slots=True`, extras fall back to a `__extras__` dict to avoid attribute errors.
- **Empty strings**: With coercion enabled, empty strings are treated as missing for optional fields; with coercion disabled
  they remain literal values and may trigger constraint errors.
- **Union error reporting**: Only the final attempted branch error is surfaced; errors are not aggregated across branches.
- **Datetime parsing**: Relies on `datetime.fromisoformat` without timezone coercion; malformed offsets will raise.
- **Schema reuse**: Nested dataclasses are inlined; there is no component registry or `$ref` reuse.
- **Validator outputs**: `validate`/`validators` may mutate values; ensure they return the transformed value to avoid silent
  loss during subsequent processing.

## Examples

### Parsing

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated
from uuid import UUID

from weakincentives.serde import parse

@dataclass
class User:
    user_id: UUID = field(metadata={"alias": "id"})
    name: Annotated[str, {"min_length": 1, "strip": True}]
    created_at: datetime

user = parse(
    User,
    {
        "ID": "a9f95576-7a80-4c79-9b90-6afee4c3f9d9",
        "name": "  Ada Lovelace  ",
        "created_at": "2025-10-28T12:34:56.789123",
    },
    case_insensitive=True,
    aliases={"user_id": "ID"},
)
```

### Dumping

```python
from weakincentives.serde import dump

payload = dump(user, computed=True, exclude_none=True)
```

### Cloning

```python
from weakincentives.serde import clone

older = clone(user, created_at=user.created_at.replace(year=2024))
```

### Schema

```python
from weakincentives.serde import schema

camel = lambda name: name.split("_", 1)[0] + "".join(part.title() for part in name.split("_")[1:])
user_schema = schema(User, alias_generator=camel, extra="forbid")
```

The tests under `tests/serde/test_dataclass_serde.py` exercise the full matrix of coercions, constraints, extras policies, and
schema expectations.

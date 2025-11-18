# Dataclass Serde Utilities

## Motivation

This module introduces a dependency-free bridge between standard library `dataclasses` and the ergonomics typically provided by
validation frameworks such as Pydantic. Agents can parse untrusted mappings, enforce field-level constraints, and emit JSON-
ready payloads without decorating dataclasses or depending on third-party packages. The helper functions live in
`weakincentives.serde` and are designed for background agents that need predictable, validated state transfer.

## Public API

Import helpers directly from the package:

```python
from weakincentives.serde import parse, dump, clone, schema
```

### `parse`

```python
parse(
    cls,
    data,
    *,
    extra="ignore",
    coerce=True,
    case_insensitive=False,
    alias_generator=None,
    aliases=None,
) -> T
```

- Accepts a dataclass type and a mapping payload.
- Unknown keys obey the `extra` policy (`"ignore"`, `"forbid"`, or `"allow"`). When allowed, extras are attached to the
  instance or stored under `__extras__` when slots prevent dynamic attributes.
- Type coercion converts strings to numerics, UUID, paths, Enums, date/time objects, nested dataclasses, collections, unions,
  and literals when feasible. Disable by passing `coerce=False`.
- Field aliases resolve in this order: the `aliases` mapping argument, `field(metadata={"alias": ...})`, and finally the
  `alias_generator` callback. Case-insensitive matching is optional.
- Constraints come from `dataclasses.field(metadata=...)` and `typing.Annotated[..., {...}]`. Supported keys include numeric
  bounds (`ge`, `gt`, `le`, `lt`, `minimum`, `maximum`), length limits, regular expressions, membership checks, string
  normalisers (`strip`, `lower`, `upper`), callable `validate`/`validators`, and a trailing `convert`/`transform` hook.
- Model-level hooks named `__validate__` or `__post_validate__` run after construction and extras assignment.

### `dump`

```python
dump(
    obj,
    *,
    by_alias=True,
    exclude_none=False,
    computed=False,
    alias_generator=None,
) -> dict[str, Any]
```

- Serialises a dataclass instance into JSON-safe primitives.
- Respects aliases or generator-driven key transformations when `by_alias=True`.
- Converts dataclasses recursively, Enums to their values, `datetime`/`date`/`time` to ISO strings, and `UUID`/`Decimal`/`Path`
  objects to strings. Collections recurse and omit `None` values when `exclude_none=True`.
- When `computed=True`, properties listed in `__computed__` are materialised and serialised with the same alias policy.
- Tool runtime payloads rely on `dump(..., exclude_none=True)` as the default backing for
  `Result.render()`, so keeping its output stable ensures telemetry, provider messages,
  and structured payloads stay in sync.

### `clone`

```python
clone(obj, **updates) -> T
```

- Thin wrapper around `dataclasses.replace` that preserves extras, applies updates, and re-runs `__validate__` /
  `__post_validate__` if present.

### `schema`

```python
schema(cls, *, alias_generator=None, extra="ignore") -> dict[str, Any]
```

- Emits a JSON Schema snippet describing the dataclass structure.
- Maps Python primitives, Enums, literals, unions/optionals, lists/sets/tuples/dicts, and nested dataclasses to schema types.
- Includes constraints derived from metadata (length, numeric bounds, patterns, membership) and mirrors alias selection.
- Sets `additionalProperties` to `False` only when `extra="forbid"`.

## Constraints & Hooks

1. Merge metadata from `Annotated` annotations and `field.metadata`. When both provide a key, the `Annotated` payload wins.
1. Apply string normalisers before constraint checks. Validators run next and may return a transformed value. `convert` /
   `transform` executes last and should return the final value.
1. Path-aware errors use dotted/indexed notation such as `"address.street"`, `"line_items[0].price"`, and
   `"attributes[sku]"`. Missing required inputs raise `ValueError("Missing required field: 'field'")`.
1. Union parsing tries branches in declaration order and surfaces the final failure message. Optional unions treat `None` and
   empty strings as missing when coercion is enabled.

## Limitations

- No assignment-time validation hooks; parsing is the validation entry point.
- Discriminated unions and `$ref` schemas are out of scope.
- External parsers (for example `python-dateutil`) are not used; conversions rely on stdlib APIs such as
  `datetime.fromisoformat`.
- Schema generation inlines nested dataclasses and does not maintain a reusable component registry.
- Error aggregation for unions re-raises the final branch error instead of collecting all failures.

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

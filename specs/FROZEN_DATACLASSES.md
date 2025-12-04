# Frozen Dataclasses

## Motivation

A lightweight decorator can make immutable dataclasses easy to author without leaning on the serde helpers or invoking
`__setattr__` in `__post_init__`. The goal is to keep model ergonomics high while keeping the serialization layer focused
solely on parse/dump concerns.

## Goals

- **Immutability by default**: Apply `frozen=True` and `slots=True` so instances behave predictably and stay lean.
- **Pre-construction shaping**: Provide a hook to derive or normalise inputs before the dataclass initialises, avoiding
  mutation inside `__post_init__`.
- **Ergonomic copy-on-write**: Mirror `dataclasses.replace` via a small helper for producing modified copies while
  preserving invariants.
- **Serde independence**: Keep the decorator and helpers free of `weakincentives.serde`; serde remains responsible only
  for serialisation and deserialisation.

## Non-goals

- Validating or coercing payloads: Callers remain responsible for shaping inputs (manually or via serde) before invoking
  the decorator-provided factories.
- Runtime mutation escape hatches: The decorator should not expose ways to bypass `frozen=True`.

## Decorator surface

`@FrozenDataclass` wraps the stdlib `@dataclass` with frozen defaults plus a few ergonomic helpers:

```python
from weakincentives.dataclasses import FrozenDataclass

@FrozenDataclass()
class Invoice:
    total_cents: int
    tax_rate: float
    tax_cents: int
    grand_total_cents: int
```

- **Defaults**: `frozen=True`, `slots=True`, `kw_only=False`, `order=False`, `eq=True`, `repr=True`. Keyword arguments
  pass through to `dataclasses.dataclass` to override defaults when needed.
- **Module location**: Implement under `src/weakincentives/dataclasses/` with a public import at
  `src/weakincentives/dataclasses/__init__.py`.

## Input normalisation hook

Allow models to define an optional `__pre_init__(cls, **kwargs) -> dict[str, Any]` classmethod. When present, the
decorator calls it before the generated `__init__`, using the returned mapping as the constructor inputs. Typical uses:

- Deriving fields from others (for example, computing `tax_cents` and `grand_total_cents` from `total_cents` and
  `tax_rate`).
- Normalising or defaulting values that would otherwise require mutation in `__post_init__`.

Rules:

- If `__pre_init__` is defined, it must accept keyword-only arguments matching the dataclass fields.
- The returned mapping must provide every required field; missing or extra keys raise `TypeError` to match dataclass
  expectations.
- `__pre_init__` runs before validation in `__post_init__`, so invariants remain in `__post_init__` or other explicit
  validators without mutating the instance.

## Post-construction validation

`__post_init__` remains available for validation or side-effect-free checks. Because instances are frozen, mutation
inside `__post_init__` is prohibited; failures should raise exceptions instead of attempting `__setattr__`.

## Copy helpers

The decorator injects a `with_updates(**changes)` instance method as a thin wrapper around `dataclasses.replace`:

```python
updated = invoice.with_updates(tax_rate=0.24)
```

- Re-applies `__post_init__` after replacement to ensure invariants still hold.
- Does **not** invoke `__pre_init__`; callers should recompute dependent fields explicitly or factor that logic into a
  shared helper callable.

## Recommended usage patterns

- Keep business rules in `__post_init__` and pure derivations in `__pre_init__`.
- Prefer `@FrozenDataclass` even for mutable-looking models to signal intent; use `@dataclass` directly when mutation is
  a requirement.
- Combine with serde by first parsing into a plain dict (via `parse(..., cls=None)`) and then calling the classmethod or
  constructor; serde does not need to know about the decorator.

## Example

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
updated = user.with_updates(tags=("pioneer",))
```

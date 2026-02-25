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

## FrozenDataclass Decorator

`@FrozenDataclass()` at `src/weakincentives/dataclasses/__init__.py` applies
`frozen=True, slots=True` by default.

**`__pre_init__` hook** — a `@classmethod` that transforms raw kwargs before
the frozen dataclass constructor runs. Useful for computing derived fields:

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

**Copy helpers** (all return new instances):

| Method | Description |
|--------|-------------|
| `update(**changes)` | Apply field changes |
| `merge(mapping)` | Merge from dict/object |
| `map(transform)` | Transform via callable |

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

## Limitations

- No discriminated unions or `$ref` schema
- No assignment-time validation
- No external parsers
- **No backward compatibility guarantee for persisted data.** This is alpha
  software; serialization format may change between versions.

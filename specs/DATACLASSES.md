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

## Scoped Field Visibility

Fields can be hidden from schema generation and parsing in specific contexts
using scope-aware annotations. This enables post-processing patterns where
certain fields are populated after LLM evaluation (e.g., in `finalize()`).

### SerdeScope Enum

```python
from weakincentives.serde import SerdeScope

class SerdeScope(Enum):
    DEFAULT = "default"                    # Standard serde, all fields visible
    STRUCTURED_OUTPUT = "structured_output" # LLM schema/parsing context
```

### HiddenInStructuredOutput Marker

Excludes a field from schema generation and parsing when `scope=STRUCTURED_OUTPUT`:

```python
from dataclasses import dataclass
from typing import Annotated
from weakincentives.serde import HiddenInStructuredOutput

@dataclass
class AnalysisResult:
    summary: str                                              # LLM generates
    confidence: float                                         # LLM generates

    # Hidden from LLM schema - populated in finalize()
    processing_time_ms: Annotated[int, HiddenInStructuredOutput()] = 0
    model_version: Annotated[str, HiddenInStructuredOutput()] = ""
```

**Marker definition:**

```python
@dataclass(frozen=True, slots=True)
class HiddenInStructuredOutput:
    """Exclude field from schema/parsing when scope is STRUCTURED_OUTPUT."""
    pass
```

### Scope Parameter on schema() and parse()

Both functions accept an optional `scope` parameter:

```python
# Schema generation with scope
schema(AnalysisResult, scope=SerdeScope.STRUCTURED_OUTPUT)
# Returns schema WITHOUT processing_time_ms and model_version

schema(AnalysisResult, scope=SerdeScope.DEFAULT)
# Returns schema WITH all fields (default behavior)

# Parsing with scope
parse(AnalysisResult, data, scope=SerdeScope.STRUCTURED_OUTPUT)
# Ignores processing_time_ms and model_version in data, uses defaults

parse(AnalysisResult, data, scope=SerdeScope.DEFAULT)
# Parses all fields normally (default behavior)
```

| Function | Scope Parameter | Default |
|----------|-----------------|---------|
| `schema()` | `scope: SerdeScope` | `SerdeScope.DEFAULT` |
| `parse()` | `scope: SerdeScope` | `SerdeScope.DEFAULT` |
| `dump()` | None | Always includes all fields |
| `clone()` | None | Uses `SerdeScope.DEFAULT` internally |

### Integration with Structured Output Pipeline

The response parser uses `STRUCTURED_OUTPUT` scope automatically:

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

Override `AgentLoop.finalize()` to populate hidden fields:

```python
from dataclasses import replace

class MyAgentLoop(AgentLoop):
    def finalize(
        self,
        prompt: Prompt[AnalysisResult],
        session: Session,
        output: AnalysisResult | None,
    ) -> AnalysisResult | None:
        if output is None:
            return None

        return replace(
            output,
            processing_time_ms=self._calculate_processing_time(),
            model_version=self._get_model_version(),
        )
```

### Default Value Requirements

Hidden fields MUST have a default value or `default_factory` since the LLM
cannot provide them:

```python
# ✓ Correct: has default
processing_time_ms: Annotated[int, HiddenInStructuredOutput()] = 0

# ✓ Correct: has default_factory
metadata: Annotated[dict[str, str], HiddenInStructuredOutput()] = field(
    default_factory=dict
)

# ✓ Correct: Optional with None default
parent_id: Annotated[str | None, HiddenInStructuredOutput()] = None

# ✗ Error: no default - would fail at parse time
timestamp: Annotated[datetime, HiddenInStructuredOutput()]  # Missing required!
```

### Composability with Other Annotations

`HiddenInStructuredOutput` composes with constraint metadata:

```python
@dataclass
class Result:
    # Hidden AND has constraints (constraints apply in finalize/manual parsing)
    score: Annotated[int, HiddenInStructuredOutput(), {"ge": 0, "le": 100}] = 0

    # Hidden AND has alias (alias used in DEFAULT scope parsing/dump)
    internal_id: Annotated[str, HiddenInStructuredOutput(), {"alias": "id"}] = ""
```

### Serialization Behavior

| Operation | Hidden Fields |
|-----------|---------------|
| `schema(scope=STRUCTURED_OUTPUT)` | Excluded |
| `schema(scope=DEFAULT)` | Included |
| `parse(scope=STRUCTURED_OUTPUT)` | Skipped (use defaults) |
| `parse(scope=DEFAULT)` | Included |
| `dump()` | **Always included** |
| `clone()` | Included (uses DEFAULT) |

**Key:** `dump()` always serializes hidden fields. This ensures:
- Logging captures complete state
- Storage/persistence includes all data
- API responses can include computed fields

### Nested Dataclass Handling

Hidden fields with nested dataclass types exclude the entire subtree:

```python
@dataclass
class Metadata:
    timestamp: datetime
    source: str

@dataclass
class Result:
    content: str
    # Entire Metadata schema excluded from STRUCTURED_OUTPUT
    meta: Annotated[Metadata, HiddenInStructuredOutput()] = field(
        default_factory=lambda: Metadata(datetime.now(), "system")
    )
```

### Validation Considerations

Validation hooks run after parsing. Hidden fields have defaults at parse time:

```python
@dataclass
class Result:
    value: int
    computed: Annotated[int, HiddenInStructuredOutput()] = 0

    def __post_validate__(self) -> None:
        # At parse time (STRUCTURED_OUTPUT): computed == 0 (default)
        # After finalize(): computed has real value
        # Consider: validate in finalize() for hidden field constraints
        pass
```

### Future Extensibility

The scope mechanism supports future expansion:

```python
class SerdeScope(Enum):
    DEFAULT = "default"
    STRUCTURED_OUTPUT = "structured_output"
    # Future scopes:
    # API = "api"              # Public API serialization
    # STORAGE = "storage"      # Database persistence
    # DEBUG = "debug"          # Verbose debug output
```

A more general marker could support multiple scopes:

```python
# Future: hide in multiple specific scopes
field: Annotated[T, HiddenInScope(SerdeScope.STRUCTURED_OUTPUT, SerdeScope.API)]
```

## Limitations

- No discriminated unions or `$ref` schema
- No assignment-time validation
- No external parsers
- **No backward compatibility guarantee for persisted data.** This is alpha
  software; serialization format may change between versions. Rehydrating
  previously persisted snapshots after schema or serde changes is not supported.

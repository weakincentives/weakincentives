# Refined Types Specification

Refined types encode value constraints directly in the type system, making
invalid states unrepresentable. Instead of scattering defensive checks
throughout code, constraints are validated once at construction boundaries,
and the type checker ensures violations cannot propagate.

## Guiding Principles

- **Construction-time validation**: Refined types validate values when they
  enter the system, not when they're used. A `NonEmpty[list[T]]` is validated
  once when created; every downstream function receives a guarantee.

- **Type-level documentation**: Constraints expressed in types are
  self-documenting and verified by static analysis. `Positive[int]` in a
  signature tells readers more than any docstring.

- **Multiplicative impact**: Every function receiving a refined type eliminates
  one defensive check. Ten functions using `NonEmpty[list[T]]` means ten fewer
  empty-list handlers.

- **Zero-cost production option**: Like DbC decorators, runtime validation can
  be disabled in production when static analysis provides sufficient
  confidence.

- **Composable constraints**: Refined types compose naturally. `NonEmpty[list[Positive[int]]]`
  guarantees both non-emptiness and positive elements.

## Goals and Non-Goals

### Goals

- Eliminate null-reference equivalents, empty-collection bugs, numeric range
  violations, and malformed string errors at their source
- Provide clear, actionable error messages at validation boundaries
- Integrate seamlessly with existing dataclass and DbC patterns
- Support both runtime validation and static type hints
- Enable gradual adoption without breaking existing code

### Non-Goals

- Full dependent type system with arbitrary predicates (too complex)
- Automatic type inference of refined constraints (requires explicit annotation)
- Replacing DbC decorators (refined types complement, not replace)
- Zero runtime cost when enabled (validation has inherent cost)

## Core Refined Types

### Numeric Refinements

```python
from weakincentives.types.refined import (
    Positive,
    NonNegative,
    Negative,
    NonPositive,
    ClosedRange,
    OpenRange,
    HalfOpenRange,
)

# Single-bound constraints
count: Positive[int]           # x > 0
age: NonNegative[int]          # x >= 0
debt: Negative[float]          # x < 0
balance: NonPositive[float]    # x <= 0

# Range constraints (closed bounds by default)
percentage: ClosedRange[int, 0, 100]      # 0 <= x <= 100
probability: ClosedRange[float, 0.0, 1.0] # 0.0 <= x <= 1.0

# Exclusive bounds
temp_kelvin: OpenRange[float, 0.0, None]  # x > 0.0, no upper bound
port: HalfOpenRange[int, 0, 65536]        # 0 <= x < 65536
```

### Collection Refinements

```python
from weakincentives.types.refined import (
    NonEmpty,
    FixedLength,
    MinLength,
    MaxLength,
    LengthRange,
)

# Non-empty guarantee
items: NonEmpty[list[str]]                # len(x) >= 1
mapping: NonEmpty[dict[str, int]]         # len(x) >= 1
text: NonEmpty[str]                       # len(x) >= 1

# Length constraints
pair: FixedLength[tuple[int, ...], 2]     # len(x) == 2
short_list: MaxLength[list[str], 10]      # len(x) <= 10
batch: LengthRange[list[Item], 1, 100]    # 1 <= len(x) <= 100
```

### String Refinements

```python
from weakincentives.types.refined import (
    NonBlank,
    Pattern,
    LowercaseStr,
    UppercaseStr,
    TrimmedStr,
)

# Content constraints
name: NonBlank[str]                        # x.strip() != ""
slug: Pattern[str, r"^[a-z0-9-]+$"]       # re.match(pattern, x)
email: Pattern[str, r"^[^@]+@[^@]+\.[^@]+$"]

# Normalization (value is transformed, not just validated)
username: LowercaseStr                     # x.lower()
code: UppercaseStr                         # x.upper()
input: TrimmedStr                          # x.strip()
```

### Membership Refinements

```python
from weakincentives.types.refined import OneOf, NoneOf
from typing import Literal

# Enumerated values (stricter than Literal for runtime)
status: OneOf[str, "pending", "active", "done"]
priority: OneOf[int, 1, 2, 3]

# Exclusion
color: NoneOf[str, "transparent", "inherit"]
```

## Type Aliases with `type` Statement

Define reusable refined types using PEP 695 syntax:

```python
# Domain-specific aliases
type UserId = Positive[int]
type Email = Pattern[str, r"^[^@]+@[^@]+\.[^@]+$"]
type Probability = ClosedRange[float, 0.0, 1.0]
type NonEmptyStr = NonEmpty[str]

# Compose constraints via intersection
type ValidUsername = TrimmedStr & Pattern[str, r"^[a-z][a-z0-9_]{2,19}$"]
```

## Dataclass Integration

Refined types integrate with dataclass field definitions:

```python
from dataclasses import dataclass
from weakincentives.types.refined import Positive, NonEmpty, Pattern

@dataclass(frozen=True, slots=True)
class Order:
    order_id: Positive[int]
    items: NonEmpty[list[str]]
    customer_email: Pattern[str, r"^[^@]+@[^@]+\.[^@]+$"]
    quantity: Positive[int] = 1
```

### Construction Validation

```python
# Valid construction
order = Order(
    order_id=123,
    items=["widget"],
    customer_email="user@example.com",
)

# Invalid construction raises RefinementError
Order(order_id=-1, ...)        # RefinementError: order_id must be positive, got -1
Order(items=[], ...)           # RefinementError: items must be non-empty
Order(customer_email="bad", ...)  # RefinementError: customer_email must match pattern
```

### Metadata Equivalence

Refined type annotations are syntactic sugar for field metadata:

```python
# These are equivalent:
@dataclass
class Config:
    count: Positive[int]

@dataclass
class Config:
    count: Annotated[int, {"gt": 0}]

@dataclass
class Config:
    count: int = field(metadata={"gt": 0})
```

The refined type syntax is preferred for readability.

## Validation Pipeline

Refined type validation follows a defined order:

```
1. Type coercion (if enabled)
2. String normalization (TrimmedStr, LowercaseStr, etc.)
3. Numeric bounds (Positive, ClosedRange, etc.)
4. Length constraints (NonEmpty, MaxLength, etc.)
5. Pattern matching (Pattern[str, ...])
6. Membership checks (OneOf, NoneOf)
7. Custom validators (via Annotated metadata)
```

Each stage produces a clear error if validation fails, identifying the field,
constraint, and actual value.

## Error Handling

### RefinementError

All validation failures raise `RefinementError` with structured information:

```python
from weakincentives.types.refined import RefinementError

try:
    order = Order(order_id=-5, items=["x"], customer_email="a@b.com")
except RefinementError as e:
    print(e.field)       # "order_id"
    print(e.constraint)  # "Positive[int]"
    print(e.value)       # -5
    print(e.message)     # "order_id must be positive, got -5"
```

### Multiple Violations

By default, validation fails fast on the first error. Enable exhaustive mode
to collect all violations:

```python
from weakincentives.types.refined import validate_exhaustive

errors = validate_exhaustive(Order, order_id=-1, items=[], customer_email="bad")
# Returns list of RefinementError for all invalid fields
```

## Integration with DbC

Refined types and DbC decorators serve complementary purposes:

| Aspect | Refined Types | DbC Decorators |
|--------|---------------|----------------|
| **Scope** | Single value | Function contract |
| **When** | Construction | Call boundaries |
| **What** | Value constraints | Relational invariants |
| **Example** | `x > 0` | `result > input` |

### Using Together

```python
from weakincentives.dbc import require, ensure
from weakincentives.types.refined import Positive, NonEmpty

@require(lambda items, n: n <= len(items), "n must not exceed items length")
@ensure(lambda result: len(result) <= len(items), "result cannot grow")
def take_first(items: NonEmpty[list[T]], n: Positive[int]) -> list[T]:
    """Take first n items from a non-empty list."""
    return items[:n]
```

The refined types guarantee `items` is non-empty and `n` is positive. The DbC
decorators express the relational constraint between them.

### Guideline

- Use **refined types** for constraints on individual values
- Use **DbC decorators** for constraints relating multiple values or
  expressing pre/postcondition relationships

## Runtime Control

### Environment Variable

```bash
# Disable all refined type validation (not recommended outside benchmarks)
WEAKINCENTIVES_REFINED=0 python app.py

# Enable with fail-fast (default)
WEAKINCENTIVES_REFINED=1 python app.py

# Enable with exhaustive validation
WEAKINCENTIVES_REFINED=exhaustive python app.py
```

### Programmatic Control

```python
from weakincentives.types.refined import (
    enable_refinement,
    disable_refinement,
    refinement_enabled,
)

# Context manager for scoped control
with refinement_enabled():
    # Validation active
    ...

# Query current state
if refinement_enabled():
    ...
```

### Production Considerations

Unlike DbC which defaults to disabled, **refined type validation defaults to
enabled**. The rationale:

1. Refined types catch bugs that would otherwise cause crashes or data
   corruption
2. Validation cost is typically negligible compared to I/O
3. Static analysis cannot catch all violations (dynamic data)

Disable only after profiling confirms validation is a bottleneck.

## Static Analysis Support

### Type Checker Integration

Refined types expose their base type for static analysis:

```python
def process(items: NonEmpty[list[str]]) -> str:
    # Type checker sees: items: list[str]
    # But also knows items is NonEmpty via plugin/stub
    return items[0]  # No "possibly empty" warning
```

### Pyright Plugin

The optional pyright plugin understands refined type semantics:

```python
items: NonEmpty[list[str]] = ["a"]
first = items[0]  # OK: NonEmpty guarantees element exists

items2: list[str] = []
first2 = items2[0]  # Warning: possibly empty

items3: NonEmpty[list[str]] = []  # Error: empty list assigned to NonEmpty
```

## Custom Refined Types

Define domain-specific refinements:

```python
from weakincentives.types.refined import Refinement, RefinementError

class Even(Refinement[int]):
    """Integer that must be even."""

    @staticmethod
    def validate(value: int) -> int:
        if value % 2 != 0:
            raise RefinementError(
                constraint="Even[int]",
                value=value,
                message=f"value must be even, got {value}",
            )
        return value

# Usage
@dataclass
class Grid:
    width: Even[int]
    height: Even[int]
```

### Parameterized Refinements

```python
from weakincentives.types.refined import ParameterizedRefinement

class Divisible(ParameterizedRefinement[int]):
    """Integer divisible by a given divisor."""

    def __class_getitem__(cls, params: tuple[type[int], int]) -> type:
        _, divisor = params
        return cls._create(divisor=divisor)

    @staticmethod
    def validate(value: int, *, divisor: int) -> int:
        if value % divisor != 0:
            raise RefinementError(
                constraint=f"Divisible[int, {divisor}]",
                value=value,
                message=f"value must be divisible by {divisor}, got {value}",
            )
        return value

# Usage
batch_size: Divisible[int, 32]  # Must be divisible by 32
```

## Implementation Sketch

### Refined Type Base

```python
from typing import Generic, TypeVar, get_args, get_origin

T = TypeVar("T")

class Refinement(Generic[T]):
    """Base class for refined types."""

    __slots__ = ()

    def __class_getitem__(cls, item: type[T]) -> type[T]:
        # Return annotated type that carries refinement metadata
        return Annotated[item, cls]

    @staticmethod
    def validate(value: T) -> T:
        """Validate and optionally transform the value."""
        raise NotImplementedError
```

### Validation Hook

```python
def validate_refined(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    """Validate all refined fields in a dataclass."""
    hints = get_type_hints(cls, include_extras=True)
    validated = {}

    for name, hint in hints.items():
        if name not in values:
            continue

        value = values[name]
        refinements = _extract_refinements(hint)

        for refinement in refinements:
            value = refinement.validate(value)

        validated[name] = value

    return validated
```

### Dataclass Integration

Integration via `__init__` wrapper or metaclass:

```python
def _wrap_init(original_init):
    @functools.wraps(original_init)
    def wrapped(self, **kwargs):
        validated = validate_refined(type(self), kwargs)
        original_init(self, **validated)
    return wrapped
```

## Testing Strategy

### Unit Tests

Each refined type requires:

1. Valid values pass through unchanged
2. Invalid values raise `RefinementError` with correct fields
3. Edge cases (boundary values, empty strings, zero, etc.)
4. Composition with other refinements

```python
def test_positive_valid():
    assert Positive.validate(1) == 1
    assert Positive.validate(0.001) == 0.001

def test_positive_invalid():
    with pytest.raises(RefinementError) as exc:
        Positive.validate(0)
    assert exc.value.constraint == "Positive"
    assert exc.value.value == 0

def test_positive_boundary():
    with pytest.raises(RefinementError):
        Positive.validate(0)
    assert Positive.validate(1) == 1  # Smallest valid
```

### Property Tests

```python
from hypothesis import given, strategies as st

@given(st.integers(min_value=1))
def test_positive_accepts_positive(n):
    assert Positive.validate(n) == n

@given(st.integers(max_value=0))
def test_positive_rejects_non_positive(n):
    with pytest.raises(RefinementError):
        Positive.validate(n)
```

### Integration Tests

```python
def test_dataclass_validation():
    @dataclass
    class Config:
        port: ClosedRange[int, 1, 65535]
        workers: Positive[int]

    # Valid
    cfg = Config(port=8080, workers=4)
    assert cfg.port == 8080

    # Invalid
    with pytest.raises(RefinementError):
        Config(port=0, workers=4)
```

## Migration Guide

### From Field Metadata

```python
# Before: metadata-based
@dataclass
class Order:
    count: int = field(metadata={"gt": 0})

# After: refined type
@dataclass
class Order:
    count: Positive[int]
```

### From Runtime Checks

```python
# Before: scattered checks
def process(items: list[str]) -> str:
    if not items:
        raise ValueError("items cannot be empty")
    return items[0].upper()

# After: type-level guarantee
def process(items: NonEmpty[list[str]]) -> str:
    return items[0].upper()  # Safe: NonEmpty guarantees element exists
```

### From DbC Preconditions

```python
# Before: DbC for value constraints
@require(lambda x: x > 0, "x must be positive")
def sqrt(x: float) -> float:
    return math.sqrt(x)

# After: refined type (simpler, same effect)
def sqrt(x: Positive[float]) -> float:
    return math.sqrt(x)
```

Keep DbC for relational constraints that refined types cannot express.

## Codebase Examples

This section shows concrete examples from the weakincentives codebase where
refined types would eliminate defensive checks.

### Budget Token Counts

**Current code** (`budget.py:64-74`):

```python
@dataclass(frozen=True, slots=True)
class Budget:
    max_total_tokens: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_total_tokens is not None and self.max_total_tokens <= 0:
            msg = "max_total_tokens must be positive."
            raise ValueError(msg)
        if self.max_input_tokens is not None and self.max_input_tokens <= 0:
            msg = "max_input_tokens must be positive."
            raise ValueError(msg)
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            msg = "max_output_tokens must be positive."
            raise ValueError(msg)
```

**With refined types**:

```python
@dataclass(frozen=True, slots=True)
class Budget:
    max_total_tokens: Positive[int] | None = None
    max_input_tokens: Positive[int] | None = None
    max_output_tokens: Positive[int] | None = None
    # No __post_init__ needed - constraints enforced by type
```

### Visibility Timeout Range

**Current code** (`runtime/mailbox/_types.py:96-113`):

```python
MAX_VISIBILITY_TIMEOUT = 43200

def validate_visibility_timeout(
    value: int, param_name: str = "visibility_timeout"
) -> None:
    """Validate visibility_timeout is within valid range [0, 43200]."""
    if value < 0:
        raise InvalidParameterError(f"{param_name} must be non-negative, got {value}")
    if value > MAX_VISIBILITY_TIMEOUT:
        raise InvalidParameterError(
            f"{param_name} must be at most {MAX_VISIBILITY_TIMEOUT} seconds, got {value}"
        )
```

**With refined types**:

```python
type VisibilityTimeout = ClosedRange[int, 0, 43200]

# Function signature encodes the constraint
def extend_visibility(timeout: VisibilityTimeout) -> None:
    # No validation needed - type guarantees valid range
    ...
```

### Skill Name Validation

**Current code** (`skills/_validation.py:114-128`):

```python
_MAX_SKILL_NAME_LENGTH = 64
_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

def validate_skill_name(name: str) -> None:
    if not name:
        msg = "Skill name cannot be empty"
        raise SkillMountError(msg)

    if len(name) > _MAX_SKILL_NAME_LENGTH:
        msg = f"Skill name exceeds {_MAX_SKILL_NAME_LENGTH} characters: {name}"
        raise SkillMountError(msg)

    if not _SKILL_NAME_PATTERN.match(name):
        msg = f"Invalid skill name format: {name}"
        raise SkillMountError(msg)
```

**With refined types**:

```python
type SkillName = LengthRange[str, 1, 64] & Pattern[str, r"^[a-z0-9]+(-[a-z0-9]+)*$"]

@dataclass(frozen=True, slots=True)
class SkillMount:
    name: SkillName  # All three constraints encoded in type
    path: Path
```

### Tool Description Bounds

**Current code** (`prompt/tool.py:284-298`):

```python
_DESCRIPTION_MAX_LENGTH = 200

def validate(self) -> None:
    description_clean = self.description.strip()
    if not description_clean or len(description_clean) > _DESCRIPTION_MAX_LENGTH:
        raise PromptValidationError(
            "Tool description must be 1-200 ASCII characters."
        )
```

**With refined types**:

```python
type ToolDescription = TrimmedStr & LengthRange[str, 1, 200]

@dataclass(frozen=True, slots=True)
class Tool:
    name: ToolName
    description: ToolDescription  # Non-blank, trimmed, 1-200 chars
    handler: Callable[..., ToolResult[Any]]
```

### Watchdog Thresholds

**Current code** (`runtime/watchdog.py:106-130`):

```python
def __init__(
    self,
    heartbeats: Sequence[Heartbeat],
    *,
    stall_threshold: float = 720.0,  # Must be > 0 (implicit)
    check_interval: float = 60.0,    # Must be > 0 (implicit)
) -> None:
    self._stall_threshold = stall_threshold
    self._check_interval = check_interval
```

**With refined types**:

```python
def __init__(
    self,
    heartbeats: NonEmpty[Sequence[Heartbeat]],  # At least one heartbeat
    *,
    stall_threshold: Positive[float] = 720.0,   # Constraint explicit
    check_interval: Positive[float] = 60.0,     # Constraint explicit
) -> None:
    ...
```

### Message Queue Bounds

**Current code** (`runtime/mailbox/_in_memory.py:309`):

```python
max_messages = min(max(1, max_messages), 10)  # Clamping to [1, 10]
```

**With refined types**:

```python
type MaxMessages = ClosedRange[int, 1, 10]

def receive(self, max_messages: MaxMessages = 1) -> list[Message]:
    # No clamping needed - type guarantees valid range
    ...
```

### Token Count Coercion

**Current code** (`adapters/token_usage.py:28`):

```python
def _coerce_token_count(value: object) -> int | None:
    if isinstance(value, (int, float)):
        coerced = int(value)
        return coerced if coerced >= 0 else None  # Reject negative
    return None
```

**With refined types**:

```python
def _coerce_token_count(value: object) -> NonNegative[int] | None:
    if isinstance(value, (int, float)):
        coerced = int(value)
        try:
            return NonNegative.validate(coerced)
        except RefinementError:
            return None
    return None
```

### Filesystem Path Depth

**Current code** (`filesystem/_types.py:158-164`):

```python
MAX_PATH_DEPTH = 10
MAX_SEGMENT_LENGTH = 200

def validate_path(path: str) -> None:
    segments = path.split("/")
    if len(segments) > MAX_PATH_DEPTH:
        msg = f"Path depth exceeds limit of {MAX_PATH_DEPTH} segments."
        raise ValueError(msg)

    for segment in segments:
        if len(segment) > MAX_SEGMENT_LENGTH:
            msg = f"Path segment exceeds limit of {MAX_SEGMENT_LENGTH} characters."
            raise ValueError(msg)
```

**With refined types**:

```python
type PathSegment = MaxLength[str, 200]
type FilesystemPath = MaxLength[tuple[PathSegment, ...], 10]

# Or as a custom refinement
class SandboxedPath(Refinement[str]):
    """Path within sandbox constraints."""

    MAX_DEPTH = 10
    MAX_SEGMENT = 200

    @staticmethod
    def validate(value: str) -> str:
        segments = value.split("/")
        if len(segments) > SandboxedPath.MAX_DEPTH:
            raise RefinementError(
                constraint="SandboxedPath",
                value=value,
                message=f"path depth exceeds {SandboxedPath.MAX_DEPTH}",
            )
        for seg in segments:
            if len(seg) > SandboxedPath.MAX_SEGMENT:
                raise RefinementError(
                    constraint="SandboxedPath",
                    value=value,
                    message=f"segment exceeds {SandboxedPath.MAX_SEGMENT} chars",
                )
        return value
```

### Session Evaluator Counts

**Current code** (`evals/_session_evaluators.py:95-110`):

```python
def tool_call_count(
    name: str,
    *,
    min_count: int = 0,       # Implicit: must be >= 0
    max_count: int | None = None,  # Implicit: must be > 0 if set
) -> SessionEvaluator:
    ...
```

**With refined types**:

```python
def tool_call_count(
    name: NonBlank[str],
    *,
    min_count: NonNegative[int] = 0,
    max_count: Positive[int] | None = None,
) -> SessionEvaluator:
    ...
```

### Domain Type Aliases

Define reusable refined types for common domain concepts:

```python
# weakincentives/types/domain.py

# Token and budget constraints
type TokenCount = NonNegative[int]
type PositiveTokens = Positive[int]

# Time constraints
type TimeoutSeconds = NonNegative[float]
type PositiveTimeout = Positive[float]
type VisibilityTimeout = ClosedRange[int, 0, 43200]

# String constraints
type SkillName = LengthRange[str, 1, 64] & Pattern[str, r"^[a-z0-9]+(-[a-z0-9]+)*$"]
type ToolName = LengthRange[str, 1, 64] & Pattern[str, r"^[a-zA-Z_][a-zA-Z0-9_]*$"]
type ToolDescription = TrimmedStr & LengthRange[str, 1, 200]

# Collection constraints
type NonEmptyMessages = NonEmpty[list[Message]]
type MaxMessages = ClosedRange[int, 1, 10]

# Filesystem constraints
type PathSegment = MaxLength[str, 200]
type PathDepth = MaxLength[tuple[str, ...], 10]
```

### Impact Summary

| Module | Current Checks | With Refined Types |
|--------|---------------|-------------------|
| `budget.py` | 3 runtime checks | 0 (type-enforced) |
| `mailbox/_types.py` | 2 validation functions | 0 (type-enforced) |
| `skills/_validation.py` | 6 validation checks | 0 (type-enforced) |
| `prompt/tool.py` | 4 validation checks | 0 (type-enforced) |
| `watchdog.py` | Implicit assumptions | Explicit in types |
| `filesystem/_types.py` | 3 validation checks | 0 (type-enforced) |

Total: **18+ runtime checks eliminated**, constraints moved to type definitions.

## Summary

Refined types shift validation from scattered defensive checks to
construction boundaries, making invalid states unrepresentable. Key benefits:

- **Type-level documentation**: Constraints are visible in signatures
- **Single validation point**: Check once at construction, trust everywhere
- **Multiplicative bug prevention**: Each refined type eliminates checks in
  every consuming function
- **Composable**: Combine constraints naturally via nesting
- **Gradual adoption**: Introduce incrementally alongside existing code

Combined with DbC decorators for relational invariants, refined types provide
comprehensive value validation with minimal ceremony.

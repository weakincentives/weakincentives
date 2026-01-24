# Serialization

*Module: `weakincentives.serde`*

The `serde` module provides dataclass serialization and deserialization without
external dependencies like Pydantic. It uses stdlib dataclasses and
`typing.Annotated` for constraints.

## Core Functions

The module exports four main functions:

```python nocheck
from weakincentives.serde import parse, dump, clone, schema
```

| Function | Purpose |
| --- | --- |
| `parse(cls, data)` | Deserialize a mapping into a dataclass instance |
| `dump(obj)` | Serialize a dataclass instance to a JSON-compatible dict |
| `clone(obj, **updates)` | Clone a dataclass with optional field updates |
| `schema(cls)` | Generate a JSON Schema for a dataclass |

## Parsing Dataclasses

Use `parse()` to convert dictionaries to typed dataclass instances:

```python nocheck
from dataclasses import dataclass
from weakincentives.serde import parse

@dataclass
class User:
    name: str
    age: int

user = parse(User, {"name": "Ada", "age": 39})
assert user.name == "Ada"
assert user.age == 39
```

### Type Coercion

By default, `parse()` coerces values to match field types:

```python nocheck
# Strings coerce to int
user = parse(User, {"name": "Ada", "age": "39"})
assert user.age == 39

# Disable coercion for strict type checking
user = parse(User, {"name": "Ada", "age": 39}, coerce=False)
```

Supported coercions:

| Target Type | Coerced From |
| --- | --- |
| `int`, `float` | Strings, numbers |
| `bool` | `"true"`, `"false"`, `"yes"`, `"no"`, `"on"`, `"off"`, `"1"`, `"0"` |
| `datetime`, `date`, `time` | ISO format strings |
| `UUID` | String representation |
| `Decimal` | Strings, numbers |
| `Path` | Strings |
| `Enum` | Enum names or values |
| `list[T]` | Single values become `[value]` |

### Optional Fields and Empty Strings

Empty or whitespace-only strings coerce to `None` for optional fields:

```python nocheck
from dataclasses import dataclass
from weakincentives.serde import parse

@dataclass
class Profile:
    bio: str | None = None

profile = parse(Profile, {"bio": "   "})
assert profile.bio is None
```

### Nested Dataclasses

Nested dataclasses are recursively parsed:

```python nocheck
from dataclasses import dataclass
from weakincentives.serde import parse

@dataclass
class Address:
    city: str
    zip: str

@dataclass
class User:
    name: str
    home: Address

user = parse(User, {
    "name": "Ada",
    "home": {"city": "London", "zip": "12345"}
})
assert user.home.city == "London"
```

Error paths include the full field path:

```python nocheck
# Raises: ValueError("home.zip: does not match pattern...")
parse(User, {"name": "Ada", "home": {"city": "London", "zip": "bad"}})
```

## Constraints via Annotated

Use `typing.Annotated` with a metadata dict to add validation constraints:

```python nocheck
from dataclasses import dataclass
from typing import Annotated
from weakincentives.serde import parse

@dataclass
class Product:
    name: Annotated[str, {"min_length": 1, "max_length": 100}]
    price: Annotated[float, {"ge": 0}]
    sku: Annotated[str, {"pattern": r"^[A-Z]{3}-\d{4}$"}]
```

### Numeric Constraints

| Key | JSON Schema | Description |
| --- | --- | --- |
| `ge` or `minimum` | `minimum` | Greater than or equal |
| `gt` or `exclusiveMinimum` | `exclusiveMinimum` | Greater than |
| `le` or `maximum` | `maximum` | Less than or equal |
| `lt` or `exclusiveMaximum` | `exclusiveMaximum` | Less than |

### String Constraints

| Key | JSON Schema | Description |
| --- | --- | --- |
| `min_length` or `minLength` | `minLength` | Minimum string length |
| `max_length` or `maxLength` | `maxLength` | Maximum string length |
| `pattern` or `regex` | `pattern` | Regex pattern (string or compiled) |

### Membership Constraints

| Key | Description |
| --- | --- |
| `in` or `enum` | Value must be in the set |
| `not_in` | Value must not be in the set |

```python nocheck
from dataclasses import dataclass
from typing import Annotated
from weakincentives.serde import parse

@dataclass
class Config:
    mode: Annotated[str, {"in": {"auto", "manual"}}]
    env: Annotated[str, {"not_in": {"test"}}]
```

### String Normalization

| Key | Description |
| --- | --- |
| `strip` | Strip leading/trailing whitespace |
| `lower` or `lowercase` | Convert to lowercase |
| `upper` or `uppercase` | Convert to uppercase |

```python nocheck
from dataclasses import dataclass
from typing import Annotated
from weakincentives.serde import parse

@dataclass
class User:
    email: Annotated[str, {"strip": True, "lower": True}]

user = parse(User, {"email": "  ADA@EXAMPLE.COM  "})
assert user.email == "ada@example.com"
```

## Custom Validators and Converters

### Validators

Add validators that run after type coercion:

```python nocheck
from dataclasses import dataclass
from typing import Annotated
from weakincentives.serde import parse

def ensure_positive(value: int) -> int:
    if value <= 0:
        raise ValueError("must be positive")
    return value

@dataclass
class Score:
    points: Annotated[int, {"validators": [ensure_positive]}]
```

Use `validate` for a single validator or `validators` for multiple.

### Converters

Add a converter that transforms the value after validation:

```python nocheck
from dataclasses import dataclass
from typing import Annotated
from weakincentives.serde import parse

def double(value: int) -> int:
    return value * 2

@dataclass
class Score:
    points: Annotated[int, {"convert": double}]

score = parse(Score, {"points": "5"})
assert score.points == 10
```

Use `convert` or `transform` as the key.

## Model-Level Validation

Dataclasses can define `__validate__` and `__post_validate__` methods:

```python nocheck
from dataclasses import dataclass
from weakincentives.serde import parse

@dataclass
class DateRange:
    start: str
    end: str

    def __validate__(self) -> None:
        if self.start > self.end:
            raise ValueError("start must be before end")
```

Both hooks run after field parsing. Use `__validate__` for basic validation and
`__post_validate__` for validation that depends on computed state.

## Field Aliases

### Per-Field Aliases

Use field metadata to define an alias:

```python nocheck
from dataclasses import dataclass, field
from weakincentives.serde import parse

@dataclass
class User:
    user_id: str = field(metadata={"alias": "id"})

user = parse(User, {"id": "abc123"})
assert user.user_id == "abc123"
```

### Alias Generator

Apply a function to all field names:

```python nocheck
from dataclasses import dataclass
from weakincentives.serde import parse

def camel_case(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(p.title() for p in parts[1:])

@dataclass
class User:
    first_name: str
    last_name: str

user = parse(User, {"firstName": "Ada", "lastName": "Lovelace"}, alias_generator=camel_case)
```

### Explicit Aliases Mapping

Pass a dict to override specific fields:

```python nocheck
user = parse(User, {"uid": "abc"}, aliases={"user_id": "uid"})
```

### Case-Insensitive Parsing

Enable case-insensitive key matching:

```python nocheck
user = parse(User, {"USER_ID": "abc"}, case_insensitive=True)
```

## Extra Fields

Control how extra fields (not in the dataclass) are handled:

| Mode | Behavior |
| --- | --- |
| `"ignore"` (default) | Extra fields are silently ignored |
| `"forbid"` | Extra fields raise `ValueError` |
| `"allow"` | Extra fields are attached to the instance |

```python nocheck
from dataclasses import dataclass
from weakincentives.serde import parse

@dataclass
class User:
    name: str

# Extra fields attached as attributes
user = parse(User, {"name": "Ada", "nickname": "Ace"}, extra="allow")
assert user.nickname == "Ace"

# Extra fields forbidden
parse(User, {"name": "Ada", "extra": "value"}, extra="forbid")
# Raises: ValueError("Extra keys not permitted: ['extra']")
```

For slotted dataclasses, extras are stored in a `__extras__` attribute.

## Serialization with dump()

Serialize dataclass instances to JSON-compatible dicts:

```python nocheck
from dataclasses import dataclass
from datetime import datetime
from weakincentives.serde import dump

@dataclass
class Event:
    name: str
    timestamp: datetime

event = Event(name="login", timestamp=datetime(2024, 1, 1, 10, 0, 0))
payload = dump(event)
assert payload == {"name": "login", "timestamp": "2024-01-01T10:00:00"}
```

### Serialization Options

| Option | Default | Description |
| --- | --- | --- |
| `by_alias` | `True` | Use field aliases in output |
| `exclude_none` | `False` | Omit fields with `None` values |
| `computed` | `False` | Include computed properties |
| `include_dataclass_type` | `False` | Add `__type__` for polymorphism |
| `alias_generator` | `None` | Function to transform field names |

### Computed Properties

Mark properties for serialization with `__computed__`:

```python nocheck
from dataclasses import dataclass
from weakincentives.serde import dump

@dataclass
class User:
    __computed__ = ("email_domain",)

    email: str

    @property
    def email_domain(self) -> str:
        return self.email.split("@")[1]

user = User(email="ada@example.com")
payload = dump(user, computed=True)
assert payload["email_domain"] == "example.com"
```

### Type Serialization

Special types are serialized as follows:

| Type | Serialization |
| --- | --- |
| `datetime`, `date`, `time` | ISO format string |
| `UUID`, `Decimal`, `Path` | String representation |
| `Enum` | Enum value (not name) |
| `set`, `frozenset` | Sorted list |

## Cloning with clone()

Clone a dataclass instance with optional field updates:

```python nocheck
from dataclasses import dataclass
from weakincentives.serde import clone

@dataclass
class User:
    name: str
    age: int

user = User(name="Ada", age=39)
updated = clone(user, age=40)
assert updated.name == "Ada"
assert updated.age == 40
```

Clone runs validation hooks, so invalid updates raise errors:

```python nocheck
@dataclass
class User:
    name: str
    age: int

    def __validate__(self) -> None:
        if self.age < 0:
            raise ValueError("age must be non-negative")

user = User(name="Ada", age=39)
clone(user, age=-1)  # Raises ValueError
```

## Polymorphic Serialization

For union types, include `__type__` to enable round-trip serialization:

```python nocheck
from dataclasses import dataclass
from weakincentives.serde import dump, parse

@dataclass
class Dog:
    breed: str

@dataclass
class Cat:
    indoor: bool

# Serialize with type reference
dog = Dog(breed="labrador")
payload = dump(dog, include_dataclass_type=True)
assert payload == {
    "__type__": "mymodule:Dog",
    "breed": "labrador"
}

# Parse without knowing the type
parsed = parse(None, payload, allow_dataclass_type=True)
assert isinstance(parsed, Dog)
```

The `__type__` format is `module:qualname` (e.g., `myapp.models:User`).

### Generic Dataclass Serialization

Generic dataclasses like `Wrapper[T]` serialize and deserialize seamlessly when
using recursive type embedding. The `__type__` field is included in all nested
dataclass values, enabling round-trip serialization without knowing the type
parameter at parse time:

```python nocheck
from dataclasses import dataclass
from weakincentives.serde import dump, parse

@dataclass
class Wrapper[T]:
    payload: T

@dataclass
class Data:
    value: int

# Serialize with type info recursively embedded
wrapper = Wrapper(payload=Data(value=42))
serialized = dump(wrapper, include_dataclass_type=True)
# {
#     "__type__": "mymodule:Wrapper",
#     "payload": {"__type__": "mymodule:Data", "value": 42}
# }

# Deserialize without knowing T - type is recovered from nested __type__
restored = parse(None, serialized, allow_dataclass_type=True)
assert isinstance(restored, Wrapper)
assert isinstance(restored.payload, Data)
assert restored.payload.value == 42
```

This works because:

1. `dump(include_dataclass_type=True)` recursively embeds `__type__` in all
   nested dataclass values, not just the root
1. `parse(allow_dataclass_type=True)` looks for `__type__` in TypeVar-typed
   fields to determine the concrete type

Without `include_dataclass_type=True`, only the root object gets `__type__`,
and generic type parameters cannot be recovered during parsing.

## JSON Schema Generation

Generate JSON Schema for dataclasses:

```python nocheck
from dataclasses import dataclass
from typing import Annotated
from weakincentives.serde import schema

@dataclass
class User:
    name: Annotated[str, {"min_length": 1}]
    age: Annotated[int, {"ge": 0, "le": 150}]

user_schema = schema(User)
# {
#     "title": "User",
#     "type": "object",
#     "properties": {
#         "name": {"type": "string", "minLength": 1},
#         "age": {"type": "integer", "minimum": 0, "maximum": 150}
#     },
#     "required": ["name", "age"],
#     "additionalProperties": True
# }
```

### Schema Options

| Option | Default | Description |
| --- | --- | --- |
| `alias_generator` | `None` | Function to transform property names |
| `extra` | `"ignore"` | Controls `additionalProperties` in schema |

## Error Handling

Parse errors include the field path and constraint that failed:

```python nocheck
# Field path in error message
# ValueError: "home.address.zip: does not match pattern ^\\d{5}$"

# Missing required field
# ValueError: "Missing required field: 'name'"

# Type coercion failure
# TypeError: "age: unable to coerce 'abc' to int"

# Constraint violation
# ValueError: "age: must be >= 0"
```

## Next Steps

- [Sessions](sessions.md): Store parsed dataclasses in session state
- [Tools](tools.md): Use dataclasses as tool parameter types
- [Testing](testing.md): Test serialization round-trips

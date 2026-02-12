# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serialization and deserialization utilities for stdlib dataclasses.

This package provides type-safe serialization (dump) and deserialization (parse)
for Python dataclasses without external dependencies like Pydantic. It supports
automatic type coercion, constraint validation via ``Annotated`` metadata, and
polymorphic union handling via ``__type__`` discriminators.

Core Functions
--------------
parse(cls, data, ...)
    Deserialize a mapping (dict) into a dataclass instance with type coercion
    and constraint validation.

dump(obj, ...)
    Serialize a dataclass instance to a JSON-compatible dictionary.

clone(obj, **overrides)
    Create a deep copy of a dataclass instance with optional field overrides.

schema(cls, ...)
    Generate a JSON Schema description for a dataclass type.

Basic Usage
-----------
::

    from dataclasses import dataclass
    from weakincentives.serde import parse, dump

    @dataclass
    class User:
        name: str
        age: int

    # Deserialize from dict
    user = parse(User, {"name": "Alice", "age": "30"})  # age coerced to int
    assert user.age == 30

    # Serialize to dict
    data = dump(user)
    assert data == {"name": "Alice", "age": 30}

Type Coercion
-------------
By default, ``parse()`` coerces values to match the declared field types:

- Strings to int, float, bool, Decimal, UUID, Path
- ISO format strings to datetime, date, time
- Strings to Enum (by name or value)
- Single values to single-element lists when needed

Set ``coerce=False`` to require exact type matches.

Constraint Validation via Annotated
-----------------------------------
Add constraints to fields using ``Annotated`` with a dict of constraint keys:

::

    from typing import Annotated
    from dataclasses import dataclass
    from weakincentives.serde import parse

    @dataclass
    class Product:
        # Numeric constraints
        price: Annotated[float, {"ge": 0, "lt": 10000}]
        quantity: Annotated[int, {"ge": 1, "le": 100}]

        # String constraints
        sku: Annotated[str, {"pattern": r"^[A-Z]{3}-\\d{4}$"}]
        name: Annotated[str, {"min_length": 1, "max_length": 200}]

        # Membership constraints
        status: Annotated[str, {"in": ["active", "inactive", "pending"]}]
        category: Annotated[str, {"not_in": ["deprecated", "removed"]}]

    product = parse(Product, {
        "price": 99.99,
        "quantity": 5,
        "sku": "ABC-1234",
        "name": "Widget",
        "status": "active",
        "category": "tools",
    })

Supported constraint keys:

- ``ge``, ``minimum``: Value must be >= bound (numeric)
- ``gt``, ``exclusiveMinimum``: Value must be > bound (numeric)
- ``le``, ``maximum``: Value must be <= bound (numeric)
- ``lt``, ``exclusiveMaximum``: Value must be < bound (numeric)
- ``min_length``, ``minLength``: Minimum length (strings, collections)
- ``max_length``, ``maxLength``: Maximum length (strings, collections)
- ``pattern``, ``regex``: Regex pattern that string must match
- ``in``, ``enum``: Value must be one of the specified options
- ``not_in``: Value must not be one of the specified options

String normalization options:

- ``strip``: Strip whitespace before validation
- ``lower``, ``lowercase``: Convert to lowercase
- ``upper``, ``uppercase``: Convert to uppercase

Custom validation and transformation:

- ``validators``, ``validate``: Callable(s) to validate the value
- ``convert``, ``transform``: Callable to transform the value after validation

Field Aliases
-------------
Map JSON keys to different field names using field metadata or alias generators:

::

    from dataclasses import dataclass, field
    from weakincentives.serde import parse, dump

    @dataclass
    class Config:
        api_key: str = field(metadata={"alias": "apiKey"})
        max_retries: int = field(metadata={"alias": "maxRetries"})

    # Parse with aliases
    config = parse(Config, {"apiKey": "secret", "maxRetries": 3})
    assert config.api_key == "secret"

    # Dump uses aliases by default
    data = dump(config)
    assert data == {"apiKey": "secret", "maxRetries": 3}

Extra Fields Handling
---------------------
Control how unrecognized fields in input data are handled:

::

    from weakincentives.serde import parse

    data = {"name": "Alice", "age": 30, "unknown_field": "value"}

    # "ignore" (default): Silently discard extra fields
    user = parse(User, data, extra="ignore")

    # "forbid": Raise ValueError if extra fields present
    user = parse(User, data, extra="forbid")  # Raises ValueError

Generic Dataclasses
-------------------
Parse generic dataclasses by providing concrete type arguments:

::

    from dataclasses import dataclass
    from weakincentives.serde import parse

    @dataclass
    class Container[T]:
        value: T
        items: list[T]

    # Provide concrete type via generic alias
    result = parse(Container[int], {"value": "42", "items": ["1", "2", "3"]})
    assert result.value == 42
    assert result.items == [1, 2, 3]

Polymorphic Unions with __type__
--------------------------------
For union types where the concrete type must be determined at runtime, use the
``__type__`` field as a discriminator. The type identifier is the fully
qualified path ``module:ClassName``.

::

    from dataclasses import dataclass
    from weakincentives.serde import parse, dump, type_identifier, resolve_type_identifier

    @dataclass
    class TextMessage:
        content: str

    @dataclass
    class ImageMessage:
        url: str
        width: int
        height: int

    # Get the type identifier for a class
    type_id = type_identifier(TextMessage)
    # Returns: "mymodule:TextMessage"

    # Include __type__ in serialized data for polymorphic deserialization
    data = {
        "__type__": "mymodule:TextMessage",
        "content": "Hello, world!",
    }

    # Resolve the type and parse
    cls = resolve_type_identifier(data["__type__"])
    message = parse(cls, data)

The ``TYPE_REF_KEY`` constant holds the discriminator field name (``"__type__"``).

JSON Schema Generation
----------------------
Generate JSON Schema for dataclasses using ``schema()``:

::

    from dataclasses import dataclass
    from typing import Annotated
    from weakincentives.serde import schema

    @dataclass
    class Person:
        name: str
        age: Annotated[int, {"ge": 0, "le": 150}]
        email: str | None = None

    json_schema = schema(Person)
    # Returns:
    # {
    #     "title": "Person",
    #     "type": "object",
    #     "properties": {
    #         "name": {"type": "string"},
    #         "age": {"type": "integer", "minimum": 0, "maximum": 150},
    #         "email": {"anyOf": [{"type": "string"}, {"type": "null"}]}
    #     },
    #     "required": ["name", "age"],
    #     "additionalProperties": True
    # }

Cloning Dataclasses
-------------------
Create modified copies of frozen dataclasses using ``clone()``:

::

    from dataclasses import dataclass
    from weakincentives.serde import clone

    @dataclass(frozen=True)
    class Point:
        x: int
        y: int

    p1 = Point(x=1, y=2)
    p2 = clone(p1, x=10)  # New instance with x=10, y=2
    assert p2.x == 10 and p2.y == 2

Security Note
-------------
The type resolution logic should only be used with trusted code and trusted type
references that are static at implementation time. Do not use ``parse()`` or
``resolve_type_identifier()`` with dynamically generated or user-provided type
annotation strings, as they could potentially resolve to unintended types.

Public API
----------
- ``parse``: Deserialize mapping to dataclass
- ``dump``: Serialize dataclass to dict
- ``clone``: Deep copy dataclass with overrides
- ``schema``: Generate JSON Schema for dataclass
- ``type_identifier``: Get fully qualified type identifier string
- ``resolve_type_identifier``: Resolve type identifier to class
- ``TYPE_REF_KEY``: The polymorphic discriminator field name (``"__type__"``)
"""

from ._scope import HiddenInStructuredOutput, SerdeScope
from ._utils import TYPE_REF_KEY, resolve_type_identifier, type_identifier
from .dump import clone, dump
from .parse import parse
from .schema import schema

__all__ = [
    "TYPE_REF_KEY",
    "HiddenInStructuredOutput",
    "SerdeScope",
    "clone",
    "dump",
    "parse",
    "resolve_type_identifier",
    "schema",
    "type_identifier",
]

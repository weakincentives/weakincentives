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

"""Common type definitions shared across weakincentives.

This package provides foundational type aliases, protocols, and utilities used
throughout the weakincentives library. These types enable consistent typing for
JSON data, dataclass handling, adapter identification, and design-by-contract
patterns.

Modules
-------
- **json**: JSON-compatible type aliases for serialization boundaries
- **dataclass**: Protocols and guards for dataclass detection
- **adapter**: Canonical adapter name constants for provider integrations

JSON Types
----------
Use these types when working with JSON-serializable data, particularly at
serialization boundaries (API responses, configuration files, tool parameters).

:data:`JSONValue`
    The recursive union of all JSON-compatible types::

        type JSONValue = str | int | float | bool | None | JSONObject | JSONArray

:data:`JSONObject`
    A mapping with string keys and JSON values::

        type JSONObject = Mapping[str, JSONValue]

:data:`JSONArray`
    A sequence of JSON values::

        type JSONArray = Sequence[JSONValue]

:data:`JSONObjectT`
    TypeVar bound to :data:`JSONObject` for generic functions that preserve
    the concrete mapping type.

:data:`JSONArrayT`
    TypeVar bound to :data:`JSONArray` for generic functions that preserve
    the concrete sequence type.

:data:`ParseableDataclassT`
    TypeVar bound to :class:`SupportsDataclass` for generic parsing functions
    that return the specific dataclass type.

:data:`ContractResult`
    Return type for design-by-contract validators. Can be:

    - ``bool``: Simple pass/fail
    - ``tuple[bool, *tuple[object, ...]]``: Pass/fail with diagnostic message(s)
    - ``None``: Treated as passing (for optional checks)

Dataclass Types
---------------
Use these types when writing generic code that operates on dataclasses.

:class:`SupportsDataclass`
    A ``@runtime_checkable`` protocol satisfied by any dataclass type or
    instance. Enables duck-typing checks via ``isinstance()``::

        @runtime_checkable
        class SupportsDataclass(Protocol):
            __dataclass_fields__: ClassVar[DataclassFieldMapping]

:data:`SupportsDataclassOrNone`
    Union of :class:`SupportsDataclass` and ``None`` for optional dataclass
    parameters.

:data:`SupportsToolResult`
    Valid return types for tool handlers::

        SupportsToolResult = SupportsDataclass | Sequence[SupportsDataclass] | None

:data:`DataclassFieldMapping`
    Type alias for the ``__dataclass_fields__`` attribute::

        type DataclassFieldMapping = dict[str, Field[Any]]

:func:`is_dataclass_instance`
    TypeGuard that returns ``True`` only for dataclass *instances* (not types).
    Use this instead of ``dataclasses.is_dataclass()`` when you need to
    distinguish instances from classes.

Adapter Constants
-----------------
Use these constants to identify provider adapters in a type-safe manner.

:data:`AdapterName`
    String type alias for adapter identifiers.

:data:`CLAUDE_AGENT_SDK_ADAPTER_NAME`
    Canonical identifier for the Claude Agent SDK adapter: ``"claude_agent_sdk"``

Examples
--------
**Type-safe JSON handling**::

    from weakincentives.types import JSONObject, JSONValue

    def process_config(config: JSONObject) -> str:
        name = config.get("name")
        if isinstance(name, str):
            return name
        return "default"

**Generic dataclass parsing**::

    from weakincentives.types import ParseableDataclassT, is_dataclass_instance

    def validate(obj: object) -> bool:
        if is_dataclass_instance(obj):
            # obj is narrowed to SupportsDataclass
            return len(obj.__dataclass_fields__) > 0
        return False

**Adapter identification**::

    from weakincentives.types import AdapterName, CLAUDE_AGENT_SDK_ADAPTER_NAME

    def get_adapter(name: AdapterName) -> object:
        if name == CLAUDE_AGENT_SDK_ADAPTER_NAME:
            return create_claude_agent_sdk_adapter()
        raise ValueError(f"Unknown adapter: {name}")

**Design-by-contract validators**::

    from weakincentives.types import ContractResult

    def positive_balance(result: float) -> ContractResult:
        if result < 0:
            return (False, f"Balance must be positive, got {result}")
        return True

When to Use These Types
-----------------------
- **JSONValue/JSONObject/JSONArray**: At serialization boundaries, API handlers,
  configuration parsing, and anywhere data crosses system boundaries as JSON.

- **SupportsDataclass**: When writing generic utilities that operate on any
  dataclass (serializers, validators, introspection tools).

- **is_dataclass_instance**: When you need to distinguish dataclass instances
  from dataclass types (e.g., in serialization code).

- **AdapterName constants**: When configuring or switching between LLM provider
  adapters. Always use the constants rather than string literals.

- **ContractResult**: When implementing custom validators for ``@require``,
  ``@ensure``, or ``@invariant`` decorators from :mod:`weakincentives.dbc`.
"""

from __future__ import annotations

from .adapter import (
    CLAUDE_AGENT_SDK_ADAPTER_NAME,
    OPENCODE_ACP_ADAPTER_NAME,
    AdapterName,
)
from .dataclass import (
    DataclassFieldMapping,
    SupportsDataclass,
    SupportsDataclassOrNone,
    SupportsToolResult,
    is_dataclass_instance,
)
from .json import (
    ContractResult,
    JSONArray,
    JSONArrayT,
    JSONObject,
    JSONObjectT,
    JSONValue,
    ParseableDataclassT,
)

__all__ = [
    "CLAUDE_AGENT_SDK_ADAPTER_NAME",
    "OPENCODE_ACP_ADAPTER_NAME",
    "AdapterName",
    "ContractResult",
    "DataclassFieldMapping",
    "JSONArray",
    "JSONArrayT",
    "JSONObject",
    "JSONObjectT",
    "JSONValue",
    "ParseableDataclassT",
    "SupportsDataclass",
    "SupportsDataclassOrNone",
    "SupportsToolResult",
    "is_dataclass_instance",
]

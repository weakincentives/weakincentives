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

"""Dataclass parsing helpers.

Security Note
-------------
The type resolution logic in this module (particularly _resolve_generic_string_type
and _get_field_types) should only be used with trusted code and trusted type
references that are static at implementation time. The AST-based type resolution
looks up types from module namespaces and could potentially resolve to unintended
types if untrusted type annotation strings are processed. Do not use parse() with
dynamically generated or user-provided type annotations.
"""

# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnnecessaryIsInstance=false, reportCallIssue=false, reportArgumentType=false, reportPossiblyUnboundVariable=false, reportPrivateUsage=false

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import MISSING
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import (
    Literal,
    Union as _TypingUnion,  # pyright: ignore[reportDeprecated]
    cast,
    get_args as typing_get_args,
    get_origin,
    get_type_hints,
)
from uuid import UUID

from ..types import JSONValue
from ._utils import (
    _UNION_TYPE,
    _AnyType,
    _apply_constraints,
    _merge_annotated_meta,
    _ParseConfig,
    _set_extras,
)

# typing.Union origin (for Optional[X] and Union[X, Y] constructs from type aliases)
_TYPING_UNION = _TypingUnion

get_args = typing_get_args


def _bool_from_str(value: str) -> bool:
    lowered = value.strip().lower()
    truthy = {"true", "1", "yes", "on"}
    falsy = {"false", "0", "no", "off"}
    if lowered in truthy:
        return True
    if lowered in falsy:
        return False
    raise TypeError(f"Cannot interpret '{value}' as boolean")


_NOT_HANDLED = object()


def _decimal_from_any(value: object) -> object:
    return Decimal(str(value))


def _uuid_from_any(value: object) -> object:
    return UUID(str(value))


def _path_from_any(value: object) -> object:
    return Path(str(value))


def _datetime_from_any(value: object) -> object:
    return datetime.fromisoformat(str(value))


def _date_from_any(value: object) -> object:
    return date.fromisoformat(str(value))


def _time_from_any(value: object) -> object:
    return time.fromisoformat(str(value))


_PRIMITIVE_COERCERS: dict[type[object], Callable[[object], object]] = cast(
    dict[type[object], Callable[[object], object]],
    {
        int: int,
        float: float,
        str: str,
        Decimal: _decimal_from_any,
        UUID: _uuid_from_any,
        Path: _path_from_any,
        datetime: _datetime_from_any,
        date: _date_from_any,
        time: _time_from_any,
    },
)


def _is_union_type(origin: object) -> bool:
    """Check if origin is a union type (either types.UnionType or typing.Union)."""
    return origin is _UNION_TYPE or origin is _TYPING_UNION


def _is_empty_string_coercible_to_none(
    value: object, base_type: object, config: _ParseConfig
) -> bool:
    """Check if empty string can be coerced to None in an optional union."""
    return (
        config.coerce
        and isinstance(value, str)
        and value.strip() == ""
        and any(arg is type(None) for arg in get_args(base_type))
    )


def _raise_union_error(last_error: Exception, path: str) -> None:
    """Raise an appropriately prefixed error from union coercion."""
    message = str(last_error)
    if message.startswith(f"{path}:") or message.startswith(f"{path}."):
        raise last_error
    if isinstance(last_error, TypeError):
        raise TypeError(f"{path}: {message}") from last_error
    raise ValueError(f"{path}: {message}") from last_error


def _coerce_union(
    value: object,
    base_type: object,
    merged_meta: Mapping[str, object],
    path: str,
    config: _ParseConfig,
) -> object:
    origin = get_origin(base_type)
    if not _is_union_type(origin):
        return _NOT_HANDLED
    if _is_empty_string_coercible_to_none(value, base_type, config):
        return _apply_constraints(None, merged_meta, path)
    last_error: Exception | None = None
    for arg in get_args(base_type):
        if arg is type(None):
            if value is None:
                return _apply_constraints(None, merged_meta, path)
            continue
        try:
            coerced = _coerce_to_type(value, arg, None, path, config)
        except (TypeError, ValueError) as error:
            last_error = error
            continue
        return _apply_constraints(coerced, merged_meta, path)
    if last_error is not None:
        _raise_union_error(last_error, path)
    raise TypeError(f"{path}: no matching type in Union")


def _try_coerce_literal(
    value: object, literal: object
) -> tuple[object | None, Exception | None]:
    """Attempt to coerce value to match a literal. Returns (coerced, error)."""
    literal_type = type(literal)
    try:
        if isinstance(literal, bool) and isinstance(value, str):
            coerced = _bool_from_str(value)
        else:
            coerced = literal_type(value)
    except (TypeError, ValueError) as error:
        return None, error
    else:
        return coerced, None


def _coerce_literal(
    value: object,
    base_type: object,
    merged_meta: Mapping[str, object],
    path: str,
    config: _ParseConfig,
) -> object:
    origin = get_origin(base_type)
    if origin is not Literal:
        return _NOT_HANDLED
    literals = get_args(base_type)
    last_literal_error: Exception | None = None
    for literal in literals:
        if value == literal:
            return _apply_constraints(literal, merged_meta, path)
        if not config.coerce:
            continue  # pragma: no cover - coerce=False rarely used with Literal
        coerced_literal, error = _try_coerce_literal(value, literal)
        if error is not None:
            last_literal_error = error
            continue
        if coerced_literal == literal:
            return _apply_constraints(literal, merged_meta, path)
    if last_literal_error is not None:
        raise type(last_literal_error)(
            f"{path}: {last_literal_error}"
        ) from last_literal_error
    raise ValueError(f"{path}: expected one of {list(literals)}")


def _coerce_none(value: object, base_type: object, path: str) -> object:
    if base_type is type(None):
        if value is not None:
            raise TypeError(f"{path}: expected None")
        return None
    if value is None:
        raise TypeError(f"{path}: value cannot be None")
    return _NOT_HANDLED


def _coerce_primitive(
    value: object,
    base_type: object,
    merged_meta: Mapping[str, object],
    path: str,
    config: _ParseConfig,
) -> object:
    coercer = _PRIMITIVE_COERCERS.get(cast(type[object], base_type))
    if coercer is None:
        return _NOT_HANDLED

    literal_type = cast(type[object], base_type)
    if isinstance(value, literal_type):
        return _apply_constraints(value, merged_meta, path)
    if not config.coerce:
        type_name = getattr(base_type, "__name__", type(base_type).__name__)
        raise TypeError(f"{path}: expected {type_name}")
    try:
        coerced_value = coercer(value)
    except Exception as error:
        type_name = getattr(base_type, "__name__", type(base_type).__name__)
        raise TypeError(f"{path}: unable to coerce {value!r} to {type_name}") from error
    return _apply_constraints(coerced_value, merged_meta, path)


def _build_nested_config(base_type: object, config: _ParseConfig) -> _ParseConfig:
    """Build config with typevar_map for nested generic type parsing.

    Resolves TypeVar arguments through the parent's typevar_map to support
    nested generics like Outer[int] containing Inner[T] where T is Outer's param.
    """
    origin = get_origin(base_type)
    if origin is None:
        return config
    nested_typevar_map = _build_typevar_map(
        cast(type[object], base_type), parent_typevar_map=config.typevar_map
    )
    if not nested_typevar_map:  # pragma: no cover - always true for PEP 695 generics
        return config
    return _ParseConfig(
        extra=config.extra,
        coerce=config.coerce,
        case_insensitive=config.case_insensitive,
        alias_generator=config.alias_generator,
        aliases=config.aliases,
        typevar_map=nested_typevar_map,
    )


def _format_nested_error(error: Exception, path: str) -> str:
    """Format error message with proper path prefix."""
    message = str(error)
    if ":" in message:
        prefix, suffix = message.split(":", 1)
        if " " not in prefix:
            return f"{path}.{prefix}:{suffix}"
    return f"{path}: {message}"


def _coerce_dataclass(
    value: object,
    base_type: object,
    merged_meta: Mapping[str, object],
    path: str,
    config: _ParseConfig,
) -> object:
    # Handle generic aliases like MyClass[int] - extract the origin class
    origin = get_origin(base_type)
    target_type = origin if origin is not None else base_type

    if not dataclasses.is_dataclass(target_type):
        return _NOT_HANDLED

    dataclass_type = target_type if isinstance(target_type, type) else type(target_type)
    if isinstance(value, dataclass_type):
        return _apply_constraints(value, merged_meta, path)
    if not isinstance(value, Mapping):
        type_name = getattr(dataclass_type, "__name__", type(dataclass_type).__name__)
        raise TypeError(f"{path}: expected mapping for dataclass {type_name}")

    nested_data = cast(Mapping[str, object], value)
    nested_config = _build_nested_config(base_type, config)

    try:
        parsed = _parse_dataclass(dataclass_type, nested_data, config=nested_config)
    except (TypeError, ValueError) as error:
        raise type(error)(_format_nested_error(error, path)) from error
    return _apply_constraints(parsed, merged_meta, path)


def _normalize_list_like(
    value: object, path: str, config: _ParseConfig
) -> list[JSONValue]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(cast(Iterable[JSONValue], value))
    if config.coerce and isinstance(value, str):
        return [value]
    raise TypeError(f"{path}: expected sequence")


def _normalize_set_value(
    value: object, path: str, config: _ParseConfig
) -> list[JSONValue]:
    if isinstance(value, (set, list, tuple)):
        return list(cast(Iterable[JSONValue], value))
    if config.coerce:
        if isinstance(value, str):
            return [value]
        if isinstance(value, Iterable):
            return list(cast(Iterable[JSONValue], value))
    raise TypeError(f"{path}: expected set")


def _normalize_tuple_value(
    value: object, path: str, config: _ParseConfig
) -> list[JSONValue]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(cast(Iterable[JSONValue], value))
    if config.coerce and isinstance(value, str):
        return [value]
    raise TypeError(f"{path}: expected tuple")


def _normalize_sequence_value(
    value: object, origin: type[object] | None, path: str, config: _ParseConfig
) -> list[JSONValue]:
    if origin in {list, Sequence}:
        return _normalize_list_like(value, path, config)
    if origin is set:
        return _normalize_set_value(value, path, config)
    return _normalize_tuple_value(value, path, config)


def _coerce_sequence_items(
    items: list[JSONValue],
    args: tuple[object, ...],
    origin: type[object] | None,
    path: str,
    config: _ParseConfig,
) -> list[object]:
    if (
        origin is tuple
        and args
        and args[-1] is not Ellipsis
        and len(args) != len(items)
    ):
        raise ValueError(f"{path}: expected {len(args)} items")
    coerced_items: list[object] = []
    for index, item in enumerate(items):
        item_path = f"{path}[{index}]"
        if origin is tuple and args:
            item_type = args[0] if args[-1] is Ellipsis else args[index]
        else:
            item_type = args[0] if args else object
        coerced_items.append(_coerce_to_type(item, item_type, None, item_path, config))
    return coerced_items


def _coerce_sequence(
    value: object,
    base_type: object,
    merged_meta: Mapping[str, object],
    path: str,
    config: _ParseConfig,
) -> object:
    origin = cast(type[object] | None, get_origin(base_type))
    if origin not in {list, Sequence, tuple, set}:
        return _NOT_HANDLED
    items = _normalize_sequence_value(value, origin, path, config)
    args = get_args(base_type)
    coerced_items = _coerce_sequence_items(items, args, origin, path, config)
    if origin is set:
        value_out: object = set(coerced_items)
    elif origin is tuple:
        value_out = tuple(coerced_items)
    else:
        value_out = list(coerced_items)
    return _apply_constraints(value_out, merged_meta, path)


def _coerce_mapping(
    value: object,
    base_type: object,
    merged_meta: Mapping[str, object],
    path: str,
    config: _ParseConfig,
) -> object:
    origin = get_origin(base_type)
    if origin is not dict and origin is not Mapping:
        return _NOT_HANDLED
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected mapping")
    key_type, value_type = (
        get_args(base_type) if get_args(base_type) else (object, object)
    )
    mapping_value = cast(Mapping[JSONValue, JSONValue], value)
    result_dict: dict[object, object] = {}
    for key, item in mapping_value.items():
        coerced_key = _coerce_to_type(key, key_type, None, f"{path} keys", config)
        coerced_value = _coerce_to_type(
            item, value_type, None, f"{path}[{coerced_key}]", config
        )
        result_dict[coerced_key] = coerced_value
    return _apply_constraints(result_dict, merged_meta, path)


def _coerce_enum(
    value: object,
    base_type: object,
    merged_meta: Mapping[str, object],
    path: str,
    config: _ParseConfig,
) -> object:
    if not (isinstance(base_type, type) and issubclass(base_type, Enum)):
        return _NOT_HANDLED
    if isinstance(value, base_type):
        enum_value = value
    elif config.coerce:
        if isinstance(value, str):
            try:
                enum_value = base_type[value]
            except KeyError:
                try:
                    enum_value = base_type(value)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{path}: invalid enum value {value!r}") from error
        else:
            try:
                enum_value = base_type(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}: invalid enum value {value!r}") from error
    else:
        type_name = getattr(base_type, "__name__", type(base_type).__name__)
        raise TypeError(f"{path}: expected {type_name}")
    return _apply_constraints(enum_value, merged_meta, path)


def _coerce_bool(
    value: object,
    base_type: object,
    merged_meta: Mapping[str, object],
    path: str,
    config: _ParseConfig,
) -> object:
    if base_type is not bool:
        return _NOT_HANDLED
    if isinstance(value, bool):
        return _apply_constraints(value, merged_meta, path)
    if config.coerce and isinstance(value, str):
        try:
            coerced_bool = _bool_from_str(value)
        except TypeError as error:
            raise TypeError(f"{path}: {error}") from error
        return _apply_constraints(coerced_bool, merged_meta, path)
    if config.coerce and isinstance(value, (int, float)):
        return _apply_constraints(bool(value), merged_meta, path)
    raise TypeError(f"{path}: expected bool")


def _is_typevar(typ: object) -> bool:
    """Check if a type is a TypeVar (unresolved generic parameter)."""
    from typing import TypeVar

    return isinstance(typ, TypeVar)


def _resolve_simple_type(type_str: str) -> object | None:
    """Resolve a simple type name to an actual type.

    Looks up type names in builtins and typing module only.
    Returns None if not found (does not fall back to object).
    """
    import builtins
    import typing
    from typing import TypeVar

    # Try builtins first (str, int, float, bool, list, dict, etc.)
    builtin_type = getattr(builtins, type_str, None)
    if builtin_type is not None and isinstance(builtin_type, type):
        return builtin_type

    # Try typing module (Any, Optional, etc.) but exclude TypeVars
    # (typing module has common TypeVar names like T, K, V which we shouldn't use)
    typing_type = getattr(typing, type_str, None)
    if typing_type is not None and not isinstance(typing_type, TypeVar):
        return typing_type

    return None


def _resolve_name(
    name: str,
    localns: dict[str, object],
    module_ns: dict[str, object],
) -> object:
    """Resolve a simple type name from namespaces."""
    # Check localns first (TypeVars)
    if name in localns:
        return localns[name]
    # Check module namespace (for class definitions like Sample)
    if name in module_ns:
        return module_ns[name]
    # Try builtins/typing
    simple = _resolve_simple_type(name)
    return simple if simple is not None else object


def _resolve_subscript(
    base_type: object,
    type_args: tuple[object, ...],
) -> object:
    """Resolve a subscripted generic type like Container[T].

    For single type arguments, passes the type directly (not as a tuple)
    since typing special forms like Optional expect a single type.
    For multiple arguments, passes them as a tuple.
    """
    if base_type is object:
        return object
    # Single arg: pass directly (e.g., Optional[str] not Optional[(str,)])
    # Multiple args: pass as tuple (e.g., Dict[str, int])
    subscript_arg = type_args[0] if len(type_args) == 1 else type_args
    try:
        return base_type[subscript_arg]  # pyright: ignore[reportIndexIssue]
    except TypeError:
        return base_type


def _make_union(left: object, right: object) -> object:
    """Create a Union type at runtime from two types using the | operator."""
    # For type objects, | creates a union at runtime (Python 3.10+)
    # This works because types support the __or__ operator for creating unions
    return left | right  # pyright: ignore[reportOperatorIssue]  # ty: ignore[unsupported-operator]


class _ASTResolver:
    """Helper class to resolve AST nodes to types with namespace context."""

    __slots__ = ("localns", "module_ns")

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self, localns: dict[str, object], module_ns: dict[str, object]
    ) -> None:
        self.localns = localns
        self.module_ns = module_ns

    @staticmethod
    def _is_literal_type(base: object) -> bool:
        """Check if base type is typing.Literal."""
        from typing import Literal, get_origin

        return base is Literal or get_origin(base) is Literal

    def _resolve_subscript_node(self, node: object, *, in_literal: bool) -> object:
        """Resolve ast.Subscript nodes like Container[T] or dict[str, int]."""
        import ast

        if not isinstance(node, ast.Subscript):  # pragma: no cover - type guard
            return object
        base = self._resolve(node.value, in_literal=False)
        # Arguments to Literal should be preserved as values, not resolved as types
        args_in_literal = self._is_literal_type(base)
        if isinstance(node.slice, ast.Tuple):
            args = tuple(
                self._resolve(elt, in_literal=args_in_literal)
                for elt in node.slice.elts
            )
        else:
            args = (self._resolve(node.slice, in_literal=args_in_literal),)
        return _resolve_subscript(base, args)

    @staticmethod
    def _resolve_unary_node(node: object) -> object | None:
        """Resolve ast.UnaryOp for signed literals like -1 or +1."""
        import ast

        if not isinstance(node, ast.UnaryOp) or not isinstance(
            node.operand, ast.Constant
        ):
            return None
        value = node.operand.value
        if isinstance(node.op, ast.USub):
            return -value  # pyright: ignore[reportOperatorIssue,reportOptionalOperand]  # ty: ignore[unsupported-operator]
        if isinstance(node.op, ast.UAdd):
            return +value  # pyright: ignore[reportOperatorIssue,reportOptionalOperand]  # ty: ignore[unsupported-operator]
        return None  # pragma: no cover - only USub/UAdd are used in type annotations

    def _resolve(self, node: object, *, in_literal: bool) -> object:
        """Recursively resolve an AST node to a type."""
        import ast

        if isinstance(node, ast.Name):
            # Inside Literal, names like True/False are resolved normally
            return _resolve_name(node.id, self.localns, self.module_ns)
        if isinstance(node, ast.Subscript):
            return self._resolve_subscript_node(node, in_literal=in_literal)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return _make_union(
                self._resolve(node.left, in_literal=False),
                self._resolve(node.right, in_literal=False),
            )
        if isinstance(node, ast.Constant):
            # Inside Literal[...], preserve constant values as-is
            if in_literal:
                return node.value
            # None constant in type context (e.g., T | None) should be NoneType
            if node.value is None:
                return type(None)
            # Outside Literal, other constants are unresolvable - fall back to object
            return object
        # Handle signed literals (always preserve value, used in Literal[-1])
        signed = self._resolve_unary_node(node)
        return signed if signed is not None else object

    def resolve(self, node: object) -> object:
        """Public entry point - resolve starting outside Literal context."""
        return self._resolve(node, in_literal=False)


def _resolve_generic_string_type(
    type_str: str,
    localns: dict[str, object],
    module_ns: dict[str, object],
) -> object:
    """Resolve a string type annotation using AST parsing.

    Handles complex generic type expressions like 'Sample[InputT, ExpectedT]'
    by parsing the string into an AST and resolving each component.

    Security Warning:
        This function looks up types from module namespaces based on string
        names. Only use with trusted type annotation strings that are static
        at implementation time. Do not pass user-provided or dynamically
        generated type strings.

    Args:
        type_str: The string type annotation to resolve.
        localns: Local namespace containing TypeVars from __type_params__.
        module_ns: Module namespace for looking up class definitions.

    Returns:
        The resolved type, or object if unresolvable.
    """
    import ast

    try:
        tree = ast.parse(type_str, mode="eval")
        return _ASTResolver(localns, module_ns).resolve(tree.body)
    except SyntaxError:
        simple = _resolve_simple_type(type_str)
        return simple if simple is not None else object


def _resolve_string_type(  # pyright: ignore[reportUnusedFunction]
    type_str: str,
) -> object:
    """Resolve a string type annotation to an actual type without eval().

    Only resolves safe, well-known types from builtins and typing module.
    Returns object for unresolvable types (e.g., TypeVar names).

    Note: This is a simplified version that doesn't handle complex generics.
    Use _resolve_generic_string_type for full generic type support.

    Tested via tests/serde/test_dataclass_serde.py.
    """
    simple = _resolve_simple_type(type_str)
    return simple if simple is not None else object


def _get_field_types(cls: type[object]) -> dict[str, object]:
    """Get field types for a dataclass, handling generic classes safely.

    Tries get_type_hints() first with TypeVars from __type_params__ in localns.
    If that fails (NameError on generic classes with postponed annotations),
    falls back to resolving field types from dataclasses.fields() using AST
    parsing to handle complex generic type annotations.

    For TypeVar string annotations (like 'T'), preserves the TypeVar object
    from __type_params__ so that TypeVar error handling still works.
    """
    import sys

    # Build mapping of TypeVar names for PEP 695 generic classes
    # This allows get_type_hints() to resolve annotations like Inner[T]
    type_params = getattr(cls, "__type_params__", ())
    localns: dict[str, object] = {tp.__name__: tp for tp in type_params}

    try:
        return get_type_hints(cls, localns=localns, include_extras=True)
    except NameError:  # pragma: no cover - defensive fallback for edge cases
        # Generic class with unresolved type parameters (and future annotations)
        # Fall back to AST-based resolution from dataclass fields.
        # Note: When get_type_hints fails with NameError, it's because
        # from __future__ import annotations is used, so all field.type
        # values are strings.

        # Get the module namespace for resolving types defined in the same module
        module_ns: dict[str, object] = {}
        module_name = getattr(cls, "__module__", None)
        if module_name and module_name in sys.modules:
            module = sys.modules[module_name]
            module_ns = vars(module)

        result: dict[str, object] = {}
        for field in dataclasses.fields(cls):
            type_str = field.type if isinstance(field.type, str) else str(field.type)
            # Check if this is a TypeVar name - preserve the TypeVar
            if type_str in localns:
                result[field.name] = localns[type_str]
            else:
                # Use AST-based resolution for complex generic types
                result[field.name] = _resolve_generic_string_type(
                    type_str, localns, module_ns
                )
        return result


def _coerce_to_type(
    value: object,
    typ: object,
    meta: Mapping[str, object] | None,
    path: str,
    config: _ParseConfig,
) -> object:
    base_type, merged_meta = _merge_annotated_meta(typ, meta)

    # Resolve TypeVar from typevar_map (populated from generic alias type args)
    if _is_typevar(base_type) and base_type in config.typevar_map:
        base_type = config.typevar_map[base_type]

    if base_type is object or base_type is _AnyType:
        return _apply_constraints(value, merged_meta, path)

    # Check for unresolved TypeVar - this means the caller didn't provide
    # a fully specialized generic alias
    if _is_typevar(base_type):
        msg = (
            f"{path}: cannot parse TypeVar field - use a fully specialized "
            f"generic type like MyClass[ConcreteType] instead of MyClass"
        )
        raise TypeError(msg)

    coercers = (
        lambda: _coerce_union(value, base_type, merged_meta, path, config),
        lambda: _coerce_none(value, base_type, path),
        lambda: _coerce_literal(value, base_type, merged_meta, path, config),
        lambda: _coerce_dataclass(value, base_type, merged_meta, path, config),
        lambda: _coerce_sequence(value, base_type, merged_meta, path, config),
        lambda: _coerce_mapping(value, base_type, merged_meta, path, config),
        lambda: _coerce_enum(value, base_type, merged_meta, path, config),
        lambda: _coerce_bool(value, base_type, merged_meta, path, config),
        lambda: _coerce_primitive(value, base_type, merged_meta, path, config),
    )
    for coercer in coercers:
        result = coercer()
        if result is not _NOT_HANDLED:
            return result

    try:
        coerced = base_type(value)
    except Exception as error:
        raise type(error)(str(error)) from error
    return _apply_constraints(coerced, merged_meta, path)


def _find_key_exact(
    data: Mapping[str, object], candidates: list[str | None]
) -> str | None:
    """Find exact match for any candidate key."""
    for candidate in candidates:
        if candidate is not None and candidate in data:
            return candidate
    return None


def _build_lowered_key_map(data: Mapping[str, object]) -> dict[str, str]:
    """Build a case-insensitive key lookup map."""
    lowered_map: dict[str, str] = {}
    for key in data:
        if isinstance(key, str):
            _ = lowered_map.setdefault(key.lower(), key)
    return lowered_map


def _find_key(
    data: Mapping[str, object], name: str, alias: str | None, case_insensitive: bool
) -> str | None:
    candidates = [alias, name]
    exact = _find_key_exact(data, candidates)
    if exact is not None or not case_insensitive:
        return exact
    lowered_map = _build_lowered_key_map(data)
    for candidate in candidates:
        if candidate is None:
            continue
        lowered = candidate.lower()
        if lowered in lowered_map:
            return lowered_map[lowered]
    return None


def _resolve_field_alias(
    field: dataclasses.Field[object],
    aliases: Mapping[str, str] | None,
    alias_generator: Callable[[str], str] | None,
    field_meta: dict[str, object],
) -> str | None:
    if aliases and field.name in aliases:
        return aliases[field.name]
    alias_value = field_meta.get("alias")
    if alias_value is not None:
        return cast(str, alias_value)
    if alias_generator is not None:
        return alias_generator(field.name)
    return None


def _coerce_field_value(
    field: dataclasses.Field[object],
    raw_value: object,
    field_meta: Mapping[str, object],
    field_type: object,
    config: _ParseConfig,
) -> object:
    try:
        return _coerce_to_type(raw_value, field_type, field_meta, field.name, config)
    except (TypeError, ValueError) as error:
        raise type(error)(str(error)) from error


def _collect_field_kwargs(
    cls: type[object],
    mapping_data: Mapping[str, object],
    type_hints: Mapping[str, object],
    config: _ParseConfig,
    *,
    aliases: Mapping[str, str] | None,
    alias_generator: Callable[[str], str] | None,
) -> tuple[dict[str, object], set[str]]:
    kwargs: dict[str, object] = {}
    used_keys: set[str] = set()

    for field in dataclasses.fields(cls):
        if not field.init:
            continue
        field_meta = dict(field.metadata)
        field_alias = _resolve_field_alias(field, aliases, alias_generator, field_meta)

        key = _find_key(mapping_data, field.name, field_alias, config.case_insensitive)
        if key is None:
            if field.default is MISSING and field.default_factory is MISSING:
                raise ValueError(f"Missing required field: '{field.name}'")
            continue
        used_keys.add(key)
        raw_value = mapping_data[key]
        field_type = type_hints.get(field.name, field.type)
        kwargs[field.name] = _coerce_field_value(
            field, raw_value, field_meta, field_type, config
        )

    return kwargs, used_keys


def _apply_extra_fields(
    instance: object,
    mapping_data: Mapping[str, object],
    used_keys: set[str],
    extra: Literal["ignore", "forbid", "allow"],
) -> None:
    extras = {key: mapping_data[key] for key in mapping_data if key not in used_keys}
    if not extras:
        return
    if extra == "forbid":
        raise ValueError(f"Extra keys not permitted: {list(extras.keys())}")
    if extra == "allow":
        if hasattr(instance, "__dict__"):
            for key, value in extras.items():
                object.__setattr__(instance, key, value)
        else:
            _set_extras(instance, extras)


def _run_validation_hooks(instance: object) -> None:
    validator = getattr(instance, "__validate__", None)
    if callable(validator):
        _ = validator()
    post_validator = getattr(instance, "__post_validate__", None)
    if callable(post_validator):
        _ = post_validator()


def _build_typevar_map(
    cls: type[object],
    *,
    parent_typevar_map: Mapping[object, type] | None = None,
) -> dict[object, type]:
    """Build a mapping from TypeVar to concrete type for a generic alias.

    For a generic alias like `MyClass[str, int]`, this maps each TypeVar
    from MyClass.__type_params__ to the corresponding type argument.

    If parent_typevar_map is provided, TypeVar arguments are resolved through
    it. This supports nested generics like Outer[int] containing Inner[T]
    where T is Outer's TypeVar - the parent map resolves T to int.
    """
    origin = get_origin(cls)
    if origin is None:
        # Not a generic alias, no typevar mapping needed
        return {}

    type_args = get_args(cls)
    if not type_args:
        return {}  # pragma: no cover - defensive

    # Get TypeVar parameters from the origin class
    type_params = getattr(origin, "__type_params__", ())
    if not type_params:
        return {}  # pragma: no cover - defensive for typing module generics

    # Map each TypeVar to its concrete type argument
    typevar_map: dict[object, type] = {}
    for param, arg in zip(type_params, type_args, strict=False):
        # Resolve TypeVar args through parent's map (for nested generics)
        if _is_typevar(arg) and parent_typevar_map and arg in parent_typevar_map:
            typevar_map[param] = parent_typevar_map[arg]
        elif isinstance(arg, type):
            typevar_map[param] = arg
        elif get_origin(arg) is not None:
            # arg is itself a generic alias (e.g., list[str])
            # Store it for recursive resolution
            typevar_map[param] = arg

    return typevar_map


def _parse_dataclass[T](
    cls: type[T],
    mapping_data: Mapping[str, object],
    *,
    config: _ParseConfig,
) -> T:
    """Internal parse implementation that takes pre-built config."""
    # Resolve generic alias to concrete class
    origin = get_origin(cls)
    target_cls = cast(type[T], origin if origin is not None else cls)

    type_hints = _get_field_types(target_cls)
    kwargs, used_keys = _collect_field_kwargs(
        target_cls,
        mapping_data,
        type_hints,
        config,
        aliases=config.aliases,
        alias_generator=config.alias_generator,
    )

    instance = target_cls(**kwargs)

    _apply_extra_fields(instance, mapping_data, used_keys, config.extra)
    _run_validation_hooks(instance)

    return instance


def parse[T](
    cls: type[T],
    data: Mapping[str, object] | object,
    *,
    extra: Literal["ignore", "forbid", "allow"] = "ignore",
    coerce: bool = True,
    case_insensitive: bool = False,
    alias_generator: Callable[[str], str] | None = None,
    aliases: Mapping[str, str] | None = None,
) -> T:
    """Parse a mapping into a dataclass instance.

    Supports generic dataclasses via generic aliases. For example:
        parse(MyGenericClass[str], data)

    The type arguments are used to resolve TypeVar fields during parsing.
    """
    if not isinstance(data, Mapping):
        raise TypeError("parse() requires a mapping input")
    if extra not in {"ignore", "forbid", "allow"}:
        raise ValueError("extra must be one of 'ignore', 'forbid', or 'allow'")

    # Resolve generic alias to concrete class
    origin = get_origin(cls)
    target_cls = cast(type[T], origin if origin is not None else cls)

    if not dataclasses.is_dataclass(target_cls) or not isinstance(target_cls, type):
        raise TypeError("parse() requires a dataclass type")

    # Build TypeVar mapping from generic alias type arguments
    typevar_map = _build_typevar_map(cls)

    mapping_data = cast(Mapping[str, object], data)
    config = _ParseConfig(
        extra=extra,
        coerce=coerce,
        case_insensitive=case_insensitive,
        alias_generator=alias_generator,
        aliases=aliases,
        typevar_map=typevar_map,
    )

    return _parse_dataclass(target_cls, mapping_data, config=config)


__all__ = ["parse"]

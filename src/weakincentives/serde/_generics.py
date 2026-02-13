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

"""TypeVar resolution and forward-reference AST scanning for serde parsing."""

# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnnecessaryIsInstance=false, reportCallIssue=false, reportArgumentType=false, reportPrivateUsage=false

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import (
    get_args,
    get_origin,
    get_type_hints,
)


def _is_typevar(typ: object) -> bool:
    """Check if a type is a TypeVar (unresolved generic parameter)."""
    from typing import TypeVar

    return isinstance(typ, TypeVar)


def _get_field_types(cls: type[object]) -> dict[str, object]:
    """Get field types for a dataclass, handling generic classes safely.

    Passes TypeVars from __type_params__ in localns so that
    get_type_hints() can resolve PEP 695 generic annotations.

    When forward references fail (e.g. TYPE_CHECKING imports), retries
    by resolving missing names from the class module's TYPE_CHECKING
    imports via ``_resolve_type_checking_imports``.
    """
    type_params = getattr(cls, "__type_params__", ())
    localns: dict[str, object] = {tp.__name__: tp for tp in type_params}

    max_retries = 10
    for _ in range(max_retries):
        try:
            return get_type_hints(cls, localns=localns, include_extras=True)
        except NameError as exc:
            missing = _extract_missing_name(exc)
            if missing is None or missing in localns:
                raise  # pragma: no cover - defensive
            resolved = _resolve_type_checking_imports(cls, missing)
            if resolved is None:
                raise  # pragma: no cover - defensive
            localns[missing] = resolved

    return get_type_hints(cls, localns=localns, include_extras=True)  # pragma: no cover


def _extract_missing_name(exc: NameError) -> str | None:
    """Extract the missing name from a NameError."""
    msg = str(exc)
    if msg.startswith("name '") and msg.endswith("' is not defined"):
        return msg[6:-16]
    return None  # pragma: no cover - defensive


def _resolve_type_checking_imports(cls: type[object], name: str) -> object | None:
    """Resolve a name from the TYPE_CHECKING imports of a module.

    Scans the module source for ``if TYPE_CHECKING:`` blocks and imports
    the missing name at runtime. Handles both absolute and relative imports.
    """
    import ast
    import inspect
    import sys

    module_name = getattr(cls, "__module__", None)
    if not module_name or module_name not in sys.modules:
        return None  # pragma: no cover - defensive

    module = sys.modules[module_name]
    try:
        source = inspect.getsource(module)
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):  # pragma: no cover - defensive
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not _is_type_checking_guard(node.test):
            continue
        result = _find_import_in_block(node.body, name, module_name)
        if result is not None:
            return result

    return None  # pragma: no cover - defensive


def _is_type_checking_guard(test: object) -> bool:
    """Check whether an AST test node is ``TYPE_CHECKING``."""
    import ast

    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute) and isinstance(test.value, ast.Name):
        return test.attr == "TYPE_CHECKING"
    return False


def _find_import_in_block(
    body: Sequence[object], name: str, module_name: str
) -> object | None:
    """Search a TYPE_CHECKING block for an import of *name*.

    Handles both ``from foo import Bar`` (module is set) and bare relative
    imports like ``from . import foo`` (module is None, level > 0).
    """
    import ast

    for stmt in body:
        if not isinstance(stmt, ast.ImportFrom):
            continue
        for alias in stmt.names:
            if (alias.asname or alias.name) != name:
                continue
            return _import_from_stmt(stmt, alias, module_name)
    return None


def _import_from_stmt(stmt: object, alias: object, module_name: str) -> object | None:
    """Resolve a single import alias from an ``ast.ImportFrom`` node."""
    import ast
    import importlib

    if not isinstance(stmt, ast.ImportFrom) or not isinstance(alias, ast.alias):
        return None  # pragma: no cover - defensive

    if stmt.module is not None:
        target = _resolve_import_module(stmt.module, stmt.level, module_name)
        try:
            mod = importlib.import_module(target)
            return getattr(mod, alias.name, None)
        except (ImportError, AttributeError):  # pragma: no cover - defensive
            return None

    # Bare relative import: ``from . import foo`` — resolve the package
    # and import the sub-module directly.
    if stmt.level > 0:
        target = _resolve_import_module(alias.name, stmt.level, module_name)
        try:
            return importlib.import_module(target)
        except ImportError:  # pragma: no cover - defensive
            return None
    return None  # pragma: no cover - defensive


def _resolve_import_module(module: str, level: int, current_module: str) -> str:
    """Resolve a possibly-relative import module path."""
    if level == 0:
        return module
    parts = current_module.split(".")
    base = ".".join(parts[:-level])
    return f"{base}.{module}" if module else base


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


__all__ = [
    "_build_typevar_map",
    "_extract_missing_name",
    "_find_import_in_block",
    "_get_field_types",
    "_import_from_stmt",
    "_is_type_checking_guard",
    "_is_typevar",
    "_resolve_import_module",
    "_resolve_type_checking_imports",
]

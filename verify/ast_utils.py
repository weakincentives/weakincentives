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

"""AST analysis utilities for the verification toolbox."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

# Minimum parts needed for a valid subpackage (e.g., "package.subpackage")
_MIN_SUBPACKAGE_PARTS = 2


@dataclass(frozen=True, slots=True)
class ImportInfo:
    """Information about an import statement.

    Attributes:
        module: The module containing the import statement.
        imported_from: The module being imported from.
        items: Items imported (empty for plain 'import' statements).
        lineno: Line number of the import statement.
        is_relative: Whether this is a relative import.
    """

    module: str
    imported_from: str
    items: tuple[str, ...]
    lineno: int
    is_relative: bool


def extract_imports(source: str, module_name: str) -> tuple[ImportInfo, ...]:
    """Extract all imports from Python source code.

    Args:
        source: The Python source code to analyze.
        module_name: The name of the module being analyzed.

    Returns:
        A tuple of ImportInfo objects describing all imports.

    Raises:
        SyntaxError: If the source code has syntax errors.
    """
    tree = ast.parse(source)
    imports: list[ImportInfo] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # For 'import foo.bar', imported_from is 'foo'
                imported_from = alias.name.split(".")[0]
                imports.append(
                    ImportInfo(
                        module=module_name,
                        imported_from=imported_from,
                        items=(),
                        lineno=node.lineno,
                        is_relative=False,
                    )
                )

        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue  # Skip 'from . import x' without module

            # Resolve relative imports
            if node.level > 0:
                base_parts = module_name.split(".")
                # Go up node.level levels
                base_parts = (
                    base_parts[: -node.level] if node.level <= len(base_parts) else []
                )
                if node.module:
                    base_parts.append(node.module)
                imported_from = ".".join(base_parts)
                is_relative = True
            else:
                imported_from = node.module
                is_relative = False

            items = tuple(alias.name for alias in node.names)

            imports.append(
                ImportInfo(
                    module=module_name,
                    imported_from=imported_from,
                    items=items,
                    lineno=node.lineno,
                    is_relative=is_relative,
                )
            )

    return tuple(imports)


def path_to_module(path: Path, src_dir: Path) -> str:
    """Convert a file path to a module name.

    Args:
        path: The file path to convert.
        src_dir: The source directory (e.g., project/src).

    Returns:
        The module name (e.g., "weakincentives.verify._ast").
    """
    try:
        rel_path = path.relative_to(src_dir)
    except ValueError:
        # Path is not relative to src_dir, use full path
        rel_path = path

    parts = list(rel_path.parts)

    # Remove .py extension
    if parts and parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]

    # Remove __init__ suffix
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]

    return ".".join(parts)


def module_to_path(
    module: str, src_dir: Path, package_name: str = "weakincentives"
) -> Path | None:
    """Convert a module name to a file path.

    Tries both __init__.py and module.py forms.

    Args:
        module: The module name to convert.
        src_dir: The source directory.
        package_name: The root package name.

    Returns:
        The path to the module, or None if not found.
    """
    parts = module.split(".")

    # Remove package prefix if present (it's implied by src_dir)
    if parts and parts[0] == package_name:
        parts = parts[1:]

    if not parts:
        return src_dir / package_name / "__init__.py"

    # Try as __init__.py first
    init_path = src_dir / package_name / "/".join(parts) / "__init__.py"
    if init_path.exists():
        return init_path

    # Try as module.py
    if len(parts) >= 1:
        module_path = src_dir / package_name / "/".join(parts[:-1]) / f"{parts[-1]}.py"
        if module_path.exists():
            return module_path

    # Try direct path
    direct_path = src_dir / package_name / "/".join(parts) / "__init__.py"
    if direct_path.exists():
        return direct_path

    return None


def get_top_level_package(
    module: str, root_package: str = "weakincentives"
) -> str | None:
    """Get the top-level package name within a root package.

    For "weakincentives.contrib.tools", returns "contrib".
    For "weakincentives.runtime", returns "runtime".

    Args:
        module: The full module name.
        root_package: The root package name.

    Returns:
        The top-level subpackage name, or None if not in root package.
    """
    parts = module.split(".")

    if not parts or parts[0] != root_package:
        return None

    if len(parts) < _MIN_SUBPACKAGE_PARTS:
        return None

    return parts[1]


def patch_ast_for_legacy_tools() -> None:
    """Restore AST nodes removed in Python 3.14.

    Some tools (like bandit) still expect deprecated AST node types
    that were removed in Python 3.14. This function patches the ast
    module to provide compatibility shims.
    """
    constant = getattr(ast, "Constant", None)
    if constant is None:
        return

    deprecated_nodes = ("Num", "Str", "Bytes", "NameConstant", "Ellipsis")
    ast_members = ast.__dict__

    for deprecated in deprecated_nodes:
        if deprecated not in ast_members:
            setattr(ast, deprecated, constant)

    # Add property shims for deprecated attributes
    def _make_property() -> property:
        return property(lambda self: self.value)

    constant_members = constant.__dict__
    for attr_name in ("n", "s", "b"):
        if attr_name not in constant_members:
            setattr(constant, attr_name, _make_property())

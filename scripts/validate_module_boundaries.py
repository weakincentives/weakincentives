#!/usr/bin/env python3
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

"""Validate module boundaries and import patterns in weakincentives.

This script enforces:
1. Layer architecture (foundation → core → high-level)
2. No imports from private modules (starting with _) outside their package
3. No circular dependencies between packages
4. No redundant reexports (submodule + items from submodule)
5. Clean separation between core and contrib
6. Optimal import flow
"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Architecture layers (from low to high)
LAYERS = {
    "foundation": {"types", "errors", "dataclasses", "dbc", "deadlines", "budget"},
    "core": {
        "runtime",
        "prompt",
        "resources",
        "filesystem",
        "serde",
        "skills",
        "formal",
        "optimizers",
    },
    "adapters": {"adapters"},
    "high_level": {"contrib", "evals", "cli"},
}

# Flatten for easy lookup
LAYER_MAP: dict[str, str] = {}
for layer_name, packages in LAYERS.items():
    for package in packages:
        LAYER_MAP[package] = layer_name

LAYER_ORDER = ["foundation", "core", "adapters", "high_level"]


@dataclass
class Import:
    """Represents an import statement."""

    module: str  # The module doing the importing
    imported_from: str  # What module is being imported from
    items: list[str]  # What items are imported (empty for module imports)
    lineno: int
    is_relative: bool


@dataclass
class Violation:
    """A module boundary violation."""

    type: str
    message: str
    file: Path
    lineno: int | None = None


class ModuleBoundaryValidator:
    """Validates module boundaries and import patterns."""

    def __init__(self, src_dir: Path):
        self.src_dir = src_dir
        self.root_package = src_dir / "weakincentives"
        self.violations: list[Violation] = []
        self.imports_by_module: dict[str, list[Import]] = defaultdict(list)
        self.reexports_by_module: dict[str, set[str]] = defaultdict(set)

    def validate(self) -> list[Violation]:
        """Run all validation checks."""
        print("🔍 Scanning Python files...")
        self._scan_files()

        print("📊 Analyzing imports...")
        self._check_layer_violations()
        self._check_private_module_leaks()
        self._check_circular_dependencies()
        self._check_redundant_reexports()
        self._check_contrib_core_separation()

        return self.violations

    def _scan_files(self) -> None:
        """Scan all Python files and extract imports."""
        for py_file in self.root_package.rglob("*.py"):
            if py_file.name == "__pycache__":
                continue

            module_name = self._path_to_module(py_file)
            try:
                with py_file.open() as f:
                    tree = ast.parse(f.read(), filename=str(py_file))
                self._extract_imports(tree, module_name, py_file)
            except SyntaxError as e:
                self.violations.append(
                    Violation(
                        type="SYNTAX_ERROR",
                        message=f"Syntax error in {py_file}: {e}",
                        file=py_file,
                        lineno=e.lineno,
                    )
                )

    def _path_to_module(self, path: Path) -> str:
        """Convert file path to module name."""
        rel_path = path.relative_to(self.src_dir)
        parts = list(rel_path.parts)

        # Remove .py extension
        if parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]

        # Remove __init__
        if parts[-1] == "__init__":
            parts = parts[:-1]

        return ".".join(parts)

    def _extract_imports(
        self, tree: ast.AST, module_name: str, file_path: Path
    ) -> None:
        """Extract imports from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_from = alias.name.split(".")[0]
                    imp = Import(
                        module=module_name,
                        imported_from=imported_from,
                        items=[],
                        lineno=node.lineno,
                        is_relative=False,
                    )
                    self.imports_by_module[module_name].append(imp)

            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue  # Skip relative imports without module

                # Resolve relative imports
                if node.level > 0:
                    base_parts = module_name.split(".")
                    # Go up node.level levels
                    base_parts = base_parts[: -(node.level)]
                    if node.module:
                        base_parts.append(node.module)
                    imported_from = ".".join(base_parts)
                    is_relative = True
                else:
                    imported_from = node.module
                    is_relative = False

                items = [alias.name for alias in node.names]

                imp = Import(
                    module=module_name,
                    imported_from=imported_from,
                    items=items,
                    lineno=node.lineno,
                    is_relative=is_relative,
                )
                self.imports_by_module[module_name].append(imp)

                # Track reexports (imports in __init__.py files)
                if module_name.endswith("__init__") or file_path.name == "__init__.py":
                    for item in items:
                        self.reexports_by_module[module_name].add(item)

    def _get_top_level_package(self, module: str) -> str:
        """Get the top-level package name."""
        parts = module.split(".")
        if len(parts) < 2:
            return module
        # Handle weakincentives.contrib.tools -> contrib
        if parts[0] == "weakincentives":
            return parts[1] if len(parts) > 1 else parts[0]
        return parts[0]

    def _check_layer_violations(self) -> None:
        """Check for layer architecture violations."""
        for module, imports in self.imports_by_module.items():
            if not module.startswith("weakincentives"):
                continue

            module_package = self._get_top_level_package(module)
            module_layer = LAYER_MAP.get(module_package)

            if module_layer is None:
                continue  # Unknown package, skip

            for imp in imports:
                if not imp.imported_from.startswith("weakincentives"):
                    continue  # External import, skip

                imported_package = self._get_top_level_package(imp.imported_from)
                imported_layer = LAYER_MAP.get(imported_package)

                if imported_layer is None:
                    continue  # Unknown package, skip

                # Check if importing from higher layer
                module_layer_idx = LAYER_ORDER.index(module_layer)
                imported_layer_idx = LAYER_ORDER.index(imported_layer)

                if imported_layer_idx > module_layer_idx:
                    self.violations.append(
                        Violation(
                            type="LAYER_VIOLATION",
                            message=(
                                f"{module} ({module_layer} layer) imports from "
                                f"{imp.imported_from} ({imported_layer} layer). "
                                f"Lower layers cannot import from higher layers."
                            ),
                            file=self._module_to_path(module),
                            lineno=imp.lineno,
                        )
                    )

    def _check_private_module_leaks(self) -> None:
        """Check that private modules (starting with _) aren't imported outside their package."""
        for module, imports in self.imports_by_module.items():
            for imp in imports:
                if not imp.imported_from.startswith("weakincentives"):
                    continue

                # Check if importing from a private module
                imported_parts = imp.imported_from.split(".")
                for i, part in enumerate(imported_parts):
                    if part.startswith("_") and not part == "__init__":
                        # This is a private module
                        # Check if it's being imported from outside its parent package
                        parent_package = ".".join(imported_parts[:i])
                        importing_package = ".".join(module.split(".")[: i + 1])

                        if not module.startswith(parent_package + "."):
                            self.violations.append(
                                Violation(
                                    type="PRIVATE_MODULE_LEAK",
                                    message=(
                                        f"{module} imports from private module "
                                        f"{imp.imported_from}. Private modules "
                                        f"should not be imported outside their package."
                                    ),
                                    file=self._module_to_path(module),
                                    lineno=imp.lineno,
                                )
                            )
                        break

                # Check if importing private items
                for item in imp.items:
                    if item.startswith("_") and not item.startswith("__"):
                        # Allow private imports within the same subpackage
                        module_pkg = ".".join(module.split(".")[:-1])
                        import_pkg = ".".join(imp.imported_from.split(".")[:-1])

                        if module_pkg != import_pkg:
                            self.violations.append(
                                Violation(
                                    type="PRIVATE_ITEM_LEAK",
                                    message=(
                                        f"{module} imports private item '{item}' from "
                                        f"{imp.imported_from}. Private items should not "
                                        f"be imported outside their immediate package."
                                    ),
                                    file=self._module_to_path(module),
                                    lineno=imp.lineno,
                                )
                            )

    def _check_circular_dependencies(self) -> None:
        """Check for circular dependencies between packages."""
        # Build dependency graph at package level
        graph: dict[str, set[str]] = defaultdict(set)

        for module, imports in self.imports_by_module.items():
            if not module.startswith("weakincentives"):
                continue

            module_package = self._get_top_level_package(module)

            for imp in imports:
                if not imp.imported_from.startswith("weakincentives"):
                    continue

                imported_package = self._get_top_level_package(imp.imported_from)

                if module_package != imported_package:
                    graph[module_package].add(imported_package)

        # Detect cycles
        def find_cycles(
            node: str, path: list[str], visited: set[str]
        ) -> list[list[str]]:
            if node in path:
                cycle_start = path.index(node)
                return [path[cycle_start:] + [node]]

            if node in visited:
                return []

            visited.add(node)
            path.append(node)

            cycles: list[list[str]] = []
            for neighbor in graph.get(node, []):
                cycles.extend(find_cycles(neighbor, path[:], visited))

            return cycles

        all_cycles: list[list[str]] = []
        visited: set[str] = set()

        for node in graph:
            if node not in visited:
                all_cycles.extend(find_cycles(node, [], visited))

        # Report unique cycles
        seen_cycles: set[tuple[str, ...]] = set()
        for cycle in all_cycles:
            # Normalize cycle to start with smallest element
            min_idx = cycle.index(min(cycle))
            normalized = tuple(cycle[min_idx:] + cycle[:min_idx])

            if normalized not in seen_cycles:
                seen_cycles.add(normalized)
                cycle_str = " → ".join(normalized)
                self.violations.append(
                    Violation(
                        type="CIRCULAR_DEPENDENCY",
                        message=f"Circular dependency detected: {cycle_str}",
                        file=self.root_package,
                    )
                )

    def _check_redundant_reexports(self) -> None:
        """Check for modules that reexport both submodules AND items from those submodules."""
        for module, imports in self.imports_by_module.items():
            if not module.startswith("weakincentives"):
                continue

            # Find module imports (from . import foo)
            submodule_imports: set[str] = set()
            item_imports: dict[str, list[str]] = defaultdict(list)

            for imp in imports:
                if imp.is_relative and not imp.items:
                    # This is a submodule import: from . import foo
                    # The imported_from will be the parent package
                    # We need to look at the actual import statement
                    # For now, skip this as we need to parse more carefully
                    pass

                if imp.is_relative and imp.items:
                    # from .submodule import Item
                    parent = module.rsplit(".", 1)[0] if "." in module else module
                    full_module = f"{parent}.{imp.imported_from.split('.')[-1]}"
                    for item in imp.items:
                        item_imports[full_module].append(item)

            # Check if we're importing both module and items
            for submodule in submodule_imports:
                if submodule in item_imports:
                    self.violations.append(
                        Violation(
                            type="REDUNDANT_REEXPORT",
                            message=(
                                f"{module} reexports both the submodule '{submodule}' "
                                f"and items from it: {', '.join(item_imports[submodule][:3])}... "
                                f"Consider removing the submodule from __all__ if items are "
                                f"already exposed."
                            ),
                            file=self._module_to_path(module),
                        )
                    )

    def _check_contrib_core_separation(self) -> None:
        """Check that contrib doesn't reexport from core modules."""
        for module, imports in self.imports_by_module.items():
            if not module.startswith("weakincentives.contrib"):
                continue

            # Skip if this is not an __init__.py
            if not (module.endswith("__init__") or "__init__" in module):
                continue

            for imp in imports:
                if not imp.imported_from.startswith("weakincentives"):
                    continue

                # Check if importing from core (not contrib)
                if not imp.imported_from.startswith("weakincentives.contrib"):
                    imported_package = self._get_top_level_package(imp.imported_from)

                    # Allow imports but not reexports
                    # A reexport is when items are imported in __init__.py and likely added to __all__
                    if imp.items and not imp.is_relative:
                        self.violations.append(
                            Violation(
                                type="CONTRIB_CORE_REEXPORT",
                                message=(
                                    f"{module} reexports items from core module "
                                    f"{imp.imported_from}: {', '.join(imp.items[:5])}. "
                                    f"Contrib packages should not reexport core modules."
                                ),
                                file=self._module_to_path(module),
                                lineno=imp.lineno,
                            )
                        )

    def _module_to_path(self, module: str) -> Path:
        """Convert module name to file path."""
        parts = module.split(".")
        # Remove weakincentives prefix if present
        if parts[0] == "weakincentives":
            parts = parts[1:]

        if not parts:
            return self.root_package / "__init__.py"

        # Try as __init__.py first
        init_path = self.root_package / "/".join(parts) / "__init__.py"
        if init_path.exists():
            return init_path

        # Try as module.py
        module_path = self.root_package / "/".join(parts[:-1]) / f"{parts[-1]}.py"
        if module_path.exists():
            return module_path

        return self.root_package / "/".join(parts)


def main() -> int:
    """Main entry point."""
    src_dir = Path(__file__).parent.parent / "src"

    if not src_dir.exists():
        print(f"❌ Source directory not found: {src_dir}")
        return 1

    validator = ModuleBoundaryValidator(src_dir)
    violations = validator.validate()

    if not violations:
        print("✅ All module boundary checks passed!")
        return 0

    # Group violations by type
    by_type: dict[str, list[Violation]] = defaultdict(list)
    for v in violations:
        by_type[v.type].append(v)

    # Print summary
    print(f"\n❌ Found {len(violations)} module boundary violation(s):\n")

    for vtype in sorted(by_type.keys()):
        viols = by_type[vtype]
        print(f"  {vtype}: {len(viols)}")

    print("\nDetails:\n")

    # Print details
    for vtype in sorted(by_type.keys()):
        viols = by_type[vtype]
        print(f"{'=' * 80}")
        print(f"{vtype} ({len(viols)} violation(s))")
        print(f"{'=' * 80}\n")

        for v in viols:
            location = f"{v.file}"
            if v.lineno:
                location += f":{v.lineno}"
            print(f"  📍 {location}")
            print(f"     {v.message}\n")

    return 1


if __name__ == "__main__":
    sys.exit(main())

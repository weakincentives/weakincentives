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

"""
Validate that production code uses Clock abstraction instead of direct time calls.

This script enforces the restriction defined in specs/CLOCK.md:
- No direct use of time.time(), time.monotonic(), time.sleep()
- No direct use of datetime.now(), datetime.utcnow(), datetime.today()
- No direct use of date.today()
- No direct use of asyncio.sleep()

Allowed exceptions:
- datetime.fromtimestamp() (converting external timestamps)
- datetime.fromisoformat() (parsing timestamps)

Note: Comments are not scanned (AST-based parsing naturally ignores them).

Exit codes:
    0: No violations found
    1: Violations found
    2: Script error
"""

import ast
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Violation:
    """A single violation of clock usage policy."""

    file_path: Path
    line: int
    column: int
    violation_type: str
    code: str

    def __str__(self) -> str:
        return (
            f"{self.file_path}:{self.line}:{self.column}: "
            f"{self.violation_type}: {self.code}"
        )


class ClockUsageValidator(ast.NodeVisitor):
    """
    AST visitor that detects forbidden time operations.

    Detects:
    - time.time(), time.time_ns(), time.monotonic(), time.monotonic_ns()
    - time.sleep(), time.perf_counter(), time.perf_counter_ns()
    - datetime.now(), datetime.utcnow()
    - asyncio.sleep()
    """

    def __init__(self, file_path: Path, source: str) -> None:
        self.file_path = file_path
        self.source_lines = source.splitlines()
        self.violations: list[Violation] = []

        # Track module import aliases: alias_name -> real_module_name
        # e.g., "import time as t" -> {"t": "time"}
        self.module_aliases: dict[str, str] = {}

        # Track names imported from forbidden modules: name -> source_module
        # e.g., "from datetime import datetime" -> {"datetime": "datetime"}
        self.imported_names: dict[str, str] = {}

        # Track direct function imports (from X import Y)
        self.time_functions: set[str] = set()
        self.datetime_functions: set[str] = set()
        self.asyncio_functions: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        """Track 'import time', 'import time as t', etc."""
        for alias in node.names:
            real_name = alias.name
            # Use alias if provided, otherwise use real name
            used_name = alias.asname if alias.asname else alias.name

            if real_name in {"time", "datetime", "asyncio"}:
                # Track the alias mapping
                self.module_aliases[used_name] = real_name

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track 'from time import monotonic', 'from datetime import datetime', etc."""
        if node.module in {"time", "datetime", "asyncio"}:
            for alias in node.names:
                real_name = alias.name
                # Use alias if provided, otherwise use real name
                used_name = alias.asname if alias.asname else alias.name

                # Track that this name came from this module
                self.imported_names[used_name] = node.module

                # Only track ACTUAL time-sourcing functions for direct calls
                # NOT data constructors like timedelta, datetime()
                if node.module == "time":
                    # Only forbidden time functions (not data constructors)
                    if real_name in {
                        "time",
                        "time_ns",
                        "monotonic",
                        "monotonic_ns",
                        "sleep",
                        "perf_counter",
                        "perf_counter_ns",
                    }:
                        self.time_functions.add(used_name)
                elif node.module == "datetime":
                    # Only forbidden datetime time-sourcing functions
                    # NOT constructors like datetime(), timedelta(), date(), time()
                    # Note: "today" is forbidden for both datetime and date classes
                    if real_name in {"now", "utcnow", "today"}:
                        self.datetime_functions.add(used_name)
                elif node.module == "asyncio" and real_name == "sleep":
                    # Only forbidden asyncio functions
                    self.asyncio_functions.add(used_name)

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Detect forbidden function calls."""
        code = self._get_source(node)

        # Check for module.function() calls (e.g., time.monotonic())
        if isinstance(node.func, ast.Attribute):
            self._check_attribute_call(node, code)
        # Check for direct function calls (e.g., monotonic() after from time import monotonic)
        elif isinstance(node.func, ast.Name):
            self._check_direct_call(node, code)

        self.generic_visit(node)

    def _check_attribute_call(self, node: ast.Call, code: str) -> None:  # noqa: C901, PLR0912 - complexity acceptable
        """Check for forbidden module.function() or name.method() calls."""
        if not isinstance(node.func, ast.Attribute):
            return

        attr_name = node.func.attr

        # Case 1: Simple attribute on a name (time.sleep, datetime.now)
        if isinstance(node.func.value, ast.Name):
            base_name = node.func.value.id

            # Check if it's an aliased module import (import time as t)
            real_module = self.module_aliases.get(base_name)
            if real_module:
                if real_module == "time":
                    self._check_time_call(node, attr_name, code)
                elif real_module == "datetime":
                    self._check_datetime_call(node, attr_name, code)
                elif real_module == "asyncio":
                    self._check_asyncio_call(node, attr_name, code)
                return

            # Check if it's a name imported from a forbidden module
            # e.g., from datetime import datetime; datetime.now()
            source_module = self.imported_names.get(base_name)
            if source_module:
                # Check if the attribute is forbidden for this module
                if source_module == "time":
                    self._check_time_call(node, attr_name, code)
                elif source_module == "datetime":
                    # Allowed: fromtimestamp, fromisoformat
                    if attr_name not in {"fromtimestamp", "fromisoformat"}:
                        self._check_datetime_call(node, attr_name, code)
                elif source_module == "asyncio":
                    self._check_asyncio_call(node, attr_name, code)
                return

        # Case 2: Nested attribute (datetime.datetime.now)
        elif isinstance(node.func.value, ast.Attribute):
            # Get the leftmost base name
            base = self._extract_leftmost_name(node.func.value)
            if (
                base
                and self.module_aliases.get(base) == "datetime"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "datetime"
            ):
                self._check_datetime_call(node, attr_name, code)

    @staticmethod
    def _extract_leftmost_name(node: ast.Attribute) -> str | None:
        """Extract the leftmost name from an attribute chain (e.g., 'a' in a.b.c)."""
        current = node
        while isinstance(current, ast.Attribute):
            if isinstance(current.value, ast.Name):
                return current.value.id
            if isinstance(current.value, ast.Attribute):
                current = current.value
            else:
                return None
        return None

    def _check_direct_call(self, node: ast.Call, code: str) -> None:
        """Check for forbidden direct function calls."""
        if not isinstance(node.func, ast.Name):
            return

        func_name = node.func.id

        # For direct calls, if the name is in our tracking sets, it was imported
        # from a forbidden module. We don't need to check the actual function name
        # because the import itself tells us it's forbidden.
        if func_name in self.time_functions:
            # Report violation using the actual name (could be aliased)
            self.violations.append(
                Violation(
                    file_path=self.file_path,
                    line=node.lineno,
                    column=node.col_offset,
                    violation_type=f"forbidden time function '{func_name}'",
                    code=code,
                )
            )
        elif func_name in self.datetime_functions:
            self.violations.append(
                Violation(
                    file_path=self.file_path,
                    line=node.lineno,
                    column=node.col_offset,
                    violation_type=f"forbidden datetime function '{func_name}'",
                    code=code,
                )
            )
        elif func_name in self.asyncio_functions:
            self.violations.append(
                Violation(
                    file_path=self.file_path,
                    line=node.lineno,
                    column=node.col_offset,
                    violation_type=f"forbidden asyncio function '{func_name}'",
                    code=code,
                )
            )

    def _check_time_call(self, node: ast.Call, func_name: str, code: str) -> None:
        """Check if time.X() call is forbidden."""
        forbidden = {
            "time",
            "time_ns",
            "monotonic",
            "monotonic_ns",
            "sleep",
            "perf_counter",
            "perf_counter_ns",
        }
        if func_name in forbidden:
            self.violations.append(
                Violation(
                    file_path=self.file_path,
                    line=node.lineno,
                    column=node.col_offset,
                    violation_type=f"forbidden time.{func_name}()",
                    code=code,
                )
            )

    def _check_datetime_call(self, node: ast.Call, func_name: str, code: str) -> None:
        """Check if datetime.X() or date.X() call is forbidden."""
        # Forbidden: now, utcnow, today (all return current time/date)
        forbidden = {"now", "utcnow", "today"}
        # Allowed: fromtimestamp, fromisoformat (for external timestamps)

        if func_name in forbidden:
            self.violations.append(
                Violation(
                    file_path=self.file_path,
                    line=node.lineno,
                    column=node.col_offset,
                    violation_type=f"forbidden datetime.{func_name}()",
                    code=code,
                )
            )
        # fromtimestamp and fromisoformat are explicitly allowed

    def _check_asyncio_call(self, node: ast.Call, func_name: str, code: str) -> None:
        """Check if asyncio.X() call is forbidden."""
        if func_name == "sleep":
            self.violations.append(
                Violation(
                    file_path=self.file_path,
                    line=node.lineno,
                    column=node.col_offset,
                    violation_type="forbidden asyncio.sleep()",
                    code=code,
                )
            )

    def _get_source(self, node: ast.AST) -> str:
        """Extract source code for AST node."""
        try:
            if hasattr(node, "lineno") and hasattr(node, "col_offset"):
                line_idx = node.lineno - 1
                if 0 <= line_idx < len(self.source_lines):
                    line = self.source_lines[line_idx]
                    # Return the portion from column offset onwards
                    return line[node.col_offset :].strip()
        except Exception:
            pass
        return "<source unavailable>"


def validate_file(file_path: Path) -> list[Violation]:
    """Validate a single Python file for clock usage violations."""
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
        validator = ClockUsageValidator(file_path, source)
        validator.visit(tree)
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return []
    else:
        return validator.violations


def find_python_files(root: Path) -> Iterator[Path]:
    """Find all Python files in src/weakincentives/."""
    src_dir = root / "src" / "weakincentives"
    if not src_dir.exists():
        print(f"Error: {src_dir} does not exist", file=sys.stderr)
        sys.exit(2)

    # Skip clock.py - it implements the Clock abstraction
    clock_impl = src_dir / "runtime" / "clock.py"

    # Skip docs/ - example code may use datetime.now() for simplicity
    docs_dir = src_dir / "docs"

    for py_file in src_dir.rglob("*.py"):
        # Skip __pycache__ and other generated files
        if "__pycache__" in py_file.parts:
            continue

        # Skip clock.py itself - it's the implementation
        if py_file == clock_impl:
            continue

        # Skip docs/ directory - example code for documentation
        if docs_dir in py_file.parents or py_file.parent == docs_dir:
            continue

        yield py_file


def main() -> int:
    """Main entry point."""
    # Find repository root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent

    print("Validating clock usage in production code...")
    print(f"Scanning: {repo_root / 'src' / 'weakincentives'}")
    print()

    all_violations: list[Violation] = []

    # Validate all Python files
    for py_file in sorted(find_python_files(repo_root)):
        violations = validate_file(py_file)
        all_violations.extend(violations)

    # Report results
    if not all_violations:
        print("✓ No violations found")
        return 0

    print(f"✗ Found {len(all_violations)} violation(s):\n")
    for violation in sorted(all_violations, key=lambda v: (str(v.file_path), v.line)):
        print(f"  {violation}")

    print()
    print("Production code must use Clock abstraction instead of direct time calls.")
    print("See specs/CLOCK.md for details and migration guide.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

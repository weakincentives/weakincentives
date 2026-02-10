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

"""Code length checker.

Enforces maximum file and function/method lengths to encourage
well-encapsulated abstractions:

- File length: max 620 lines (warning)
- Function/method length: max 120 lines (error)
"""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..result import CheckResult, Diagnostic, Location


_DEFAULT_MAX_FILE_LINES = 620
_DEFAULT_MAX_FUNCTION_LINES = 120


@dataclass
class CodeLengthChecker:
    """Checker for file and function/method length limits.

    Both file and function violations are reported as warnings by default,
    allowing incremental enforcement without blocking the build.
    """

    src_dir: Path | None = None
    test_dir: Path | None = None
    max_file_lines: int = _DEFAULT_MAX_FILE_LINES
    max_function_lines: int = _DEFAULT_MAX_FUNCTION_LINES
    file_length_severity: str = "warning"
    function_length_severity: str = "warning"
    exclude_patterns: tuple[str, ...] = field(
        default_factory=lambda: ("__pycache__",)
    )

    @property
    def name(self) -> str:
        return "code-length"

    @property
    def description(self) -> str:
        return "Check file and function length limits"

    def run(self) -> CheckResult:
        start = time.monotonic()
        root = Path(".")
        src = self.src_dir or root / "src"
        tests = self.test_dir or root / "tests"

        diagnostics: list[Diagnostic] = []
        has_errors = False

        for directory in (src, tests):
            if not directory.is_dir():
                continue
            for py_file in sorted(directory.rglob("*.py")):
                if any(pat in str(py_file) for pat in self.exclude_patterns):
                    continue
                file_diags = self._check_file(py_file)
                for d in file_diags:
                    if d.severity == "error":
                        has_errors = True
                diagnostics.extend(file_diags)

        return CheckResult(
            name=self.name,
            status="failed" if has_errors else "passed",
            duration_ms=int((time.monotonic() - start) * 1000),
            diagnostics=tuple(diagnostics),
        )

    def _check_file(self, py_file: Path) -> list[Diagnostic]:
        """Check a single Python file for length violations."""
        diagnostics: list[Diagnostic] = []

        try:
            source = py_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []

        lines = source.splitlines()
        total_lines = len(lines)

        # Check file length
        if total_lines > self.max_file_lines:
            msg = (
                f"File has {total_lines} lines (max {self.max_file_lines})\n"
                f"Fix: Split into smaller, focused modules"
            )
            diagnostics.append(
                Diagnostic(
                    message=msg,
                    location=Location(file=str(py_file)),
                    severity=self.file_length_severity,  # type: ignore[arg-type]
                )
            )

        # Check function/method lengths via AST
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            return diagnostics

        diagnostics.extend(self._check_functions(tree, py_file))
        return diagnostics

    def _check_functions(
        self, tree: ast.Module, py_file: Path
    ) -> list[Diagnostic]:
        """Walk AST to find functions/methods exceeding the length limit."""
        diagnostics: list[Diagnostic] = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            start_line = node.lineno
            end_line = node.end_lineno
            if end_line is None:
                continue

            length = end_line - start_line + 1
            if length <= self.max_function_lines:
                continue

            # Determine qualified name (Class.method or function)
            qualified_name = _qualified_name(tree, node)

            msg = (
                f"{qualified_name} has {length} lines "
                f"(max {self.max_function_lines})\n"
                f"Fix: Extract helper functions for better encapsulation"
            )
            diagnostics.append(
                Diagnostic(
                    message=msg,
                    location=Location(
                        file=str(py_file),
                        line=start_line,
                        end_line=end_line,
                    ),
                    severity=self.function_length_severity,  # type: ignore[arg-type]
                )
            )

        return diagnostics


def _qualified_name(
    tree: ast.Module, target: ast.FunctionDef | ast.AsyncFunctionDef
) -> str:
    """Get 'ClassName.method' or 'function' name for a function node."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in ast.iter_child_nodes(node):
                if child is target:
                    return f"{node.name}.{target.name}"
    return target.name

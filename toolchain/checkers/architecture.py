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

"""Architecture verification checker.

Enforces core/contrib separation: core modules cannot import from contrib.
This ensures that contrib builds on core, not vice versa.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from ..result import CheckResult, Diagnostic, Location
from ..utils import extract_imports, get_subpackage, path_to_module


@dataclass
class ArchitectureChecker:
    """Checker for core/contrib separation.

    Ensures that core modules (weakincentives.*) don't import from
    contrib modules (weakincentives.contrib.*).
    """

    src_dir: Path | None = None

    @property
    def name(self) -> str:
        return "architecture"

    @property
    def description(self) -> str:
        return "Check core/contrib separation"

    def run(self) -> CheckResult:
        start = time.monotonic()
        src = self.src_dir or Path("src")
        pkg_dir = src / "weakincentives"

        if not pkg_dir.is_dir():
            msg = (
                f"Package not found: {pkg_dir}\n"
                f"Fix: Ensure you're in the project root directory\n"
                f"Expected structure: src/weakincentives/"
            )
            return CheckResult(
                name=self.name,
                status="failed",
                duration_ms=int((time.monotonic() - start) * 1000),
                diagnostics=(Diagnostic(msg),),
            )

        diagnostics: list[Diagnostic] = []

        for py_file in pkg_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            file_diags = self._check_file(py_file, src)
            diagnostics.extend(file_diags)

        return CheckResult(
            name=self.name,
            status="passed" if not diagnostics else "failed",
            duration_ms=int((time.monotonic() - start) * 1000),
            diagnostics=tuple(diagnostics),
        )

    def _check_file(self, py_file: Path, src: Path) -> list[Diagnostic]:
        """Check a single Python file for core/contrib violations."""
        module_name = path_to_module(py_file, src)
        module_pkg = get_subpackage(module_name)

        # Skip contrib and docs - they can import from contrib
        if module_pkg in ("contrib", "docs", None):
            return []

        try:
            source = py_file.read_text(encoding="utf-8")
            imports = extract_imports(source, module_name)
        except SyntaxError as e:
            msg = (
                f"Syntax error: {e}\n"
                f"Fix: Correct the syntax error in the file\n"
                f"Run: make format lint to identify all issues"
            )
            return [
                Diagnostic(
                    message=msg,
                    location=Location(file=str(py_file), line=e.lineno),
                )
            ]

        diagnostics: list[Diagnostic] = []
        for imp in imports:
            if "contrib" in imp.imported_from:
                msg = (
                    f"Core module imports from contrib: {imp.imported_from}\n"
                    f"Import: {imp.statement}\n"
                    f"Fix: Move code to contrib or refactor to use protocols\n"
                    f"Architecture: Core modules must not depend on contrib\n"
                    f"See: CLAUDE.md for architecture guidelines"
                )
                diagnostics.append(
                    Diagnostic(
                        message=msg,
                        location=Location(file=str(py_file), line=imp.lineno),
                    )
                )

        return diagnostics

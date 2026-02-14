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

"""Private module import checker.

Enforces that private modules (``_foo.py`` files and ``_bar/`` packages)
are never imported from outside their owning package.  The owning package
is the nearest ancestor that does **not** start with an underscore.

For example, ``weakincentives.serde._scope`` is owned by
``weakincentives.serde``, so only modules under ``weakincentives.serde``
may import from it.

The ``adapters/_shared/`` pattern is handled naturally: its owning
package is ``weakincentives.adapters``, so any adapter sub-package can
import from it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from ..result import CheckResult, Diagnostic, Location
from ..utils import extract_imports, path_to_module


def _is_private(segment: str) -> bool:
    """Return whether a module segment is private (single-underscore prefix).

    Dunder names (``__future__``, ``__init__``, etc.) are **not** private.
    """
    return segment.startswith("_") and not segment.startswith("__")


def _find_private_segment(parts: list[str]) -> int | None:
    """Return the index of the first private segment, or *None*."""
    for index, segment in enumerate(parts):
        if _is_private(segment):
            return index
    return None


@dataclass
class PrivateImportChecker:
    """Checker that flags cross-package imports of private modules.

    A module ``weakincentives.X._priv`` is *private* to package
    ``weakincentives.X``.  Any import that references ``_priv`` from a
    module **outside** ``weakincentives.X`` is flagged as an error.
    """

    src_dir: Path | None = None

    @property
    def name(self) -> str:
        return "private-imports"

    @property
    def description(self) -> str:
        return "Check private module imports across package boundaries"

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

        for py_file in sorted(pkg_dir.rglob("*.py")):
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
        """Check a single file for cross-package private imports."""
        module_name = path_to_module(py_file, src)

        try:
            source = py_file.read_text(encoding="utf-8")
            imports = extract_imports(source, module_name)
        except SyntaxError:
            return []  # Architecture checker handles syntax errors

        diagnostics: list[Diagnostic] = []
        for imp in imports:
            parts = imp.imported_from.split(".")
            priv_idx = _find_private_segment(parts)
            if priv_idx is None:
                continue

            # Owning package: everything before the first private segment
            owning_package = ".".join(parts[:priv_idx])

            # Allow if importer is within the owning package
            if module_name == owning_package or module_name.startswith(
                owning_package + "."
            ):
                continue

            private_module = ".".join(parts[: priv_idx + 1])
            msg = (
                f"Private module imported from outside its package: "
                f"{private_module}\n"
                f"Import: {imp.statement}\n"
                f"Fix: Import from the public API of '{owning_package}' instead\n"
                f"Rule: Private _modules must not be imported outside their "
                f"owning package"
            )
            diagnostics.append(
                Diagnostic(
                    message=msg,
                    location=Location(file=str(py_file), line=imp.lineno),
                )
            )

        return diagnostics

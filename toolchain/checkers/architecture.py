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

Enforces the four-layer module boundary model:

    Layer 4  HIGH-LEVEL   contrib, evals, cli, docs
    Layer 3  ADAPTERS     adapters
    Layer 2  CORE         runtime, prompt, resources, filesystem, serde,
                          skills, formal, debug, optimizers
    Layer 1  FOUNDATION   types, errors, dataclasses, dbc, deadlines,
                          budget, clock, experiment

Rules:
  - A module in layer *N* may import from layers 1 .. *N* (same or lower).
  - Imports inside ``if TYPE_CHECKING:`` are always allowed.
  - Core/contrib separation is a strict subset of the layer model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from ..result import CheckResult, Diagnostic, Location
from ..utils import ImportInfo, extract_imports, get_subpackage, path_to_module

# ---------------------------------------------------------------------------
# Layer definitions
# ---------------------------------------------------------------------------

FOUNDATION = 1
CORE = 2
ADAPTERS = 3
HIGH_LEVEL = 4

_LAYER_NAME: dict[int, str] = {
    FOUNDATION: "Foundation",
    CORE: "Core",
    ADAPTERS: "Adapters",
    HIGH_LEVEL: "High-level",
}

# Maps the first sub-package component to its layer.
_PACKAGE_LAYER: dict[str, int] = {
    # Foundation (layer 1)
    "types": FOUNDATION,
    "errors": FOUNDATION,
    "dataclasses": FOUNDATION,
    "dbc": FOUNDATION,
    "deadlines": FOUNDATION,
    "budget": FOUNDATION,
    "clock": FOUNDATION,
    "experiment": FOUNDATION,
    # Core (layer 2)
    "runtime": CORE,
    "prompt": CORE,
    "resources": CORE,
    "filesystem": CORE,
    "serde": CORE,
    "skills": CORE,
    "formal": CORE,
    "debug": CORE,
    "optimizers": CORE,
    # Adapters (layer 3)
    "adapters": ADAPTERS,
    # High-level (layer 4)
    "contrib": HIGH_LEVEL,
    "evals": HIGH_LEVEL,
    "cli": HIGH_LEVEL,
    "docs": HIGH_LEVEL,
}


def _layer_of(subpackage: str | None) -> int | None:
    """Return the layer number for a subpackage, or *None* if unknown."""
    if subpackage is None:
        return None
    return _PACKAGE_LAYER.get(subpackage)


def _target_subpackage(imported_from: str) -> str | None:
    """Extract the weakincentives sub-package from a resolved import path."""
    return get_subpackage(imported_from)


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------


@dataclass
class ArchitectureChecker:
    """Checker enforcing the four-layer module boundary model.

    Also enforces the legacy core/contrib separation rule as a strict
    subset of the layer model.
    """

    src_dir: Path | None = None

    @property
    def name(self) -> str:
        return "architecture"

    @property
    def description(self) -> str:
        return "Check core/contrib separation and layer boundaries"

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
        """Check a single Python file for layer violations."""
        module_name = path_to_module(py_file, src)
        source_pkg = get_subpackage(module_name)
        source_layer = _layer_of(source_pkg)

        # Packages not in the layer map are unconstrained
        if source_layer is None:
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
            diag = self._check_import(imp, py_file, source_pkg, source_layer)
            if diag is not None:
                diagnostics.append(diag)

        return diagnostics

    @staticmethod
    def _check_import(
        imp: ImportInfo,
        py_file: Path,
        source_pkg: str | None,
        source_layer: int,
    ) -> Diagnostic | None:
        """Return a diagnostic if *imp* violates the layer model."""
        # TYPE_CHECKING imports never violate layer boundaries
        if imp.in_type_checking:
            return None

        target_pkg = _target_subpackage(imp.imported_from)
        target_layer = _layer_of(target_pkg)

        # Only enforce boundaries for internal weakincentives imports
        if target_layer is None:
            return None

        # Same or lower layer — allowed
        if target_layer <= source_layer:
            return None

        # --- violation ---
        src_label = _LAYER_NAME[source_layer]
        tgt_label = _LAYER_NAME[target_layer]
        msg = (
            f"Layer violation: {src_label} (layer {source_layer}) "
            f"imports {tgt_label} (layer {target_layer})\n"
            f"Module: {imp.module} ({source_pkg})\n"
            f"Import: {imp.statement}\n"
            f"Fix: Use TYPE_CHECKING for type-only imports, "
            f"move code to a higher layer, or refactor to use protocols\n"
            f"See: specs/MODULE_BOUNDARIES.md"
        )
        return Diagnostic(
            message=msg,
            location=Location(file=str(py_file), line=imp.lineno),
        )

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

"""Architecture verification checkers.

These checkers enforce architectural constraints:
- Layer violations (foundation -> core -> adapters -> high-level)
- Core/contrib separation (core cannot import from contrib)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from weakincentives.verify._ast import (
    extract_imports,
    get_top_level_package,
    path_to_module,
)
from weakincentives.verify._types import CheckContext, CheckResult, Finding, Severity

if TYPE_CHECKING:
    pass

# Architecture layers (from low to high)
LAYERS: dict[str, frozenset[str]] = {
    "foundation": frozenset(
        {"types", "errors", "dataclasses", "dbc", "deadlines", "budget"}
    ),
    "core": frozenset(
        {
            "runtime",
            "prompt",
            "resources",
            "filesystem",
            "serde",
            "skills",
            "formal",
            "optimizers",
            "debug",
            "verify",
        }
    ),
    "adapters": frozenset({"adapters"}),
    "high_level": frozenset({"contrib", "evals", "cli"}),
}

# Map package to layer for quick lookup
LAYER_MAP: dict[str, str] = {
    package: layer_name
    for layer_name, packages in LAYERS.items()
    for package in packages
}

LAYER_ORDER = ("foundation", "core", "adapters", "high_level")


class LayerViolationsChecker:
    """Checker for layer architecture violations.

    Ensures that lower layers don't import from higher layers:
    - foundation cannot import from core, adapters, or high_level
    - core cannot import from adapters or high_level
    - adapters cannot import from high_level
    """

    @property
    def name(self) -> str:
        return "layer_violations"

    @property
    def category(self) -> str:
        return "architecture"

    @property
    def description(self) -> str:
        return "Check that lower layers don't import from higher layers"

    def check(self, ctx: CheckContext) -> CheckResult:  # noqa: C901
        start_time = time.monotonic()
        findings: list[Finding] = []

        package_dir = ctx.src_dir / "weakincentives"
        if not package_dir.is_dir():
            return CheckResult(
                checker=f"{self.category}.{self.name}",
                findings=(
                    Finding(
                        checker=f"{self.category}.{self.name}",
                        severity=Severity.ERROR,
                        message=f"Package directory not found: {package_dir}",
                    ),
                ),
                duration_ms=int((time.monotonic() - start_time) * 1000),
            )

        # Scan all Python files
        for py_file in package_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            module_name = path_to_module(py_file, ctx.src_dir)
            module_package = get_top_level_package(module_name)

            if module_package is None:
                continue

            module_layer = LAYER_MAP.get(module_package)
            if module_layer is None:
                continue

            try:
                source = py_file.read_text(encoding="utf-8")
                imports = extract_imports(source, module_name)
            except SyntaxError as e:
                findings.append(
                    Finding(
                        checker=f"{self.category}.{self.name}",
                        severity=Severity.ERROR,
                        message=f"Syntax error: {e}",
                        file=py_file,
                        line=e.lineno,
                    )
                )
                continue

            for imp in imports:
                if not imp.imported_from.startswith("weakincentives"):
                    continue

                imported_package = get_top_level_package(imp.imported_from)
                if imported_package is None:
                    continue

                imported_layer = LAYER_MAP.get(imported_package)
                if imported_layer is None:
                    continue

                module_layer_idx = LAYER_ORDER.index(module_layer)
                imported_layer_idx = LAYER_ORDER.index(imported_layer)

                if imported_layer_idx > module_layer_idx:
                    findings.append(
                        Finding(
                            checker=f"{self.category}.{self.name}",
                            severity=Severity.ERROR,
                            message=(
                                f"{module_name} ({module_layer} layer) imports from "
                                f"{imp.imported_from} ({imported_layer} layer). "
                                f"Lower layers cannot import from higher layers."
                            ),
                            file=py_file,
                            line=imp.lineno,
                        )
                    )

        duration_ms = int((time.monotonic() - start_time) * 1000)
        return CheckResult(
            checker=f"{self.category}.{self.name}",
            findings=tuple(findings),
            duration_ms=duration_ms,
        )


class CoreContribSeparationChecker:
    """Checker for core/contrib separation.

    Ensures that core modules (weakincentives.*) don't import from
    contrib modules (weakincentives.contrib.*).
    """

    @property
    def name(self) -> str:
        return "core_contrib_separation"

    @property
    def category(self) -> str:
        return "architecture"

    @property
    def description(self) -> str:
        return "Check that core modules don't import from contrib"

    def check(self, ctx: CheckContext) -> CheckResult:
        start_time = time.monotonic()
        findings: list[Finding] = []

        package_dir = ctx.src_dir / "weakincentives"
        contrib_dir = package_dir / "contrib"
        docs_dir = package_dir / "docs"

        if not package_dir.is_dir():
            return CheckResult(
                checker=f"{self.category}.{self.name}",
                findings=(
                    Finding(
                        checker=f"{self.category}.{self.name}",
                        severity=Severity.ERROR,
                        message=f"Package directory not found: {package_dir}",
                    ),
                ),
                duration_ms=int((time.monotonic() - start_time) * 1000),
            )

        for py_file in package_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            # Skip contrib directory - it's allowed to import from contrib
            if contrib_dir.exists() and (
                contrib_dir in py_file.parents or py_file.parent == contrib_dir
            ):
                continue

            # Skip docs directory - contains bundled examples
            if docs_dir.exists() and (
                docs_dir in py_file.parents or py_file.parent == docs_dir
            ):
                continue

            try:
                source = py_file.read_text(encoding="utf-8")
                imports = extract_imports(source, path_to_module(py_file, ctx.src_dir))
            except SyntaxError:
                continue  # Syntax errors handled by other checker

            findings.extend(
                Finding(
                    checker=f"{self.category}.{self.name}",
                    severity=Severity.ERROR,
                    message=f"Core module imports from contrib: {imp.imported_from}",
                    file=py_file,
                    line=imp.lineno,
                    suggestion=(
                        "Core modules (weakincentives.*) must not import from "
                        "contrib (weakincentives.contrib.*). Contrib builds on "
                        "core, not vice versa."
                    ),
                )
                for imp in imports
                if "contrib" in imp.imported_from
            )

        duration_ms = int((time.monotonic() - start_time) * 1000)
        return CheckResult(
            checker=f"{self.category}.{self.name}",
            findings=tuple(findings),
            duration_ms=duration_ms,
        )

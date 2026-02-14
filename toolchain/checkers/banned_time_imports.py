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

"""Checker that bans direct ``time`` module usage in src/weakincentives/.

All production code must use the clock protocols (``MonotonicClock``,
``WallClock``, ``Sleeper``) from ``weakincentives.clock`` instead of
calling ``time.monotonic()``, ``time.time()``, or ``time.sleep()``
directly.  The only exception is ``clock.py`` itself, which wraps the
stdlib ``time`` module behind the protocol abstraction.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

from ..result import CheckResult, Diagnostic, Location

# Matches ``import time`` (bare or aliased) and ``from time import ...``
_IMPORT_TIME_RE = re.compile(
    r"^\s*(?:import\s+time\b|from\s+time\s+import\b)",
)

# Files allowed to import time directly.
_ALLOWED_FILES = frozenset({"clock.py"})


@dataclass
class BannedTimeImportsChecker:
    """Checker that flags direct ``import time`` in production code.

    Scans all ``.py`` files under ``src/weakincentives/`` (excluding
    ``clock.py``) for bare ``import time`` or ``from time import``
    statements.
    """

    src_dir: Path | None = None

    @property
    def name(self) -> str:
        return "banned-time-imports"

    @property
    def description(self) -> str:
        return "Ban direct time module usage (use clock protocols)"

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
            if py_file.name in _ALLOWED_FILES:
                continue

            file_diags = self._check_file(py_file)
            diagnostics.extend(file_diags)

        return CheckResult(
            name=self.name,
            status="passed" if not diagnostics else "failed",
            duration_ms=int((time.monotonic() - start) * 1000),
            diagnostics=tuple(diagnostics),
        )

    def _check_file(self, py_file: Path) -> list[Diagnostic]:
        """Check a single file for banned time imports."""
        try:
            lines = py_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []

        diagnostics: list[Diagnostic] = []
        for lineno, line in enumerate(lines, start=1):
            if _IMPORT_TIME_RE.match(line):
                msg = (
                    f"Direct time import is banned: {line.strip()}\n"
                    f"Fix: Use weakincentives.clock protocols instead\n"
                    f"  - time.monotonic()  -> MonotonicClock.monotonic()\n"
                    f"  - time.time()       -> MonotonicClock.monotonic() "
                    f"(elapsed/deadline)\n"
                    f"                      -> WallClock.utcnow() "
                    f"(wall-clock timestamps)\n"
                    f"  - time.sleep()      -> Sleeper.sleep()\n"
                    f"Inject clock parameter (default SYSTEM_CLOCK)"
                )
                diagnostics.append(
                    Diagnostic(
                        message=msg,
                        location=Location(file=str(py_file), line=lineno),
                    )
                )

        return diagnostics

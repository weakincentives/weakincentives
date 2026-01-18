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

"""Dependency verification checkers.

These checkers verify dependency hygiene:
- Deptry: Unused/missing dependency detection
"""

from __future__ import annotations

import time

from core_types import CheckContext, CheckResult, Finding, Severity
from subprocess_utils import run_python_module


class DeptryChecker:
    """Checker for dependency hygiene using deptry.

    Detects unused dependencies and missing dependencies.
    """

    @property
    def name(self) -> str:
        return "deptry"

    @property
    def category(self) -> str:
        return "dependencies"

    @property
    def description(self) -> str:
        return "Check for unused or missing dependencies"

    def check(self, ctx: CheckContext) -> CheckResult:
        start_time = time.monotonic()
        findings: list[Finding] = []

        result = run_python_module(
            "deptry",
            ["--no-ansi", str(ctx.src_dir)],
            cwd=ctx.project_root,
            timeout_seconds=60.0,
        )

        output = result.output.strip()

        if output and (not result.success or "warning" in output.lower()):
            for line in output.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Deptry outputs warnings/errors with specific codes
                if any(
                    code in line for code in ("DEP001", "DEP002", "DEP003", "DEP004")
                ):
                    findings.append(
                        Finding(
                            checker=f"{self.category}.{self.name}",
                            severity=Severity.ERROR,
                            message=line,
                        )
                    )
                elif "warning" in line.lower() or "error" in line.lower():
                    findings.append(
                        Finding(
                            checker=f"{self.category}.{self.name}",
                            severity=Severity.WARNING,
                            message=line,
                        )
                    )

        duration_ms = int((time.monotonic() - start_time) * 1000)
        return CheckResult(
            checker=f"{self.category}.{self.name}",
            findings=tuple(findings),
            duration_ms=duration_ms,
        )

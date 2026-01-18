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

"""Security verification checkers.

These checkers verify security properties:
- Bandit: Static security analysis
- Pip-audit: Known vulnerability scanning
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from weakincentives.verify._ast import patch_ast_for_legacy_tools
from weakincentives.verify._subprocess import run_python_module
from weakincentives.verify._types import CheckContext, CheckResult, Finding, Severity


class BanditChecker:
    """Checker for security anti-patterns using Bandit.

    Runs bandit static analysis to find common security issues.
    """

    @property
    def name(self) -> str:
        return "bandit"

    @property
    def category(self) -> str:
        return "security"

    @property
    def description(self) -> str:
        return "Check for security anti-patterns"

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

        # Patch AST for Python 3.14 compatibility
        patch_ast_for_legacy_tools()

        result = run_python_module(
            "bandit",
            ["-q", "-r", str(package_dir)],
            cwd=ctx.project_root,
            timeout_seconds=60.0,
        )

        if not result.success:
            output = result.output.strip()
            if output:
                # Parse bandit output for individual issues
                for line in output.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    # Bandit outputs lines like:
                    # >> Issue: [B101:assert_used] Use of assert detected.
                    if ">> Issue:" in line or ">>" in line:
                        findings.append(
                            Finding(
                                checker=f"{self.category}.{self.name}",
                                severity=Severity.ERROR,
                                message=line,
                            )
                        )
                    elif line.startswith("   "):
                        # Context line, skip
                        continue
                    elif ":" in line and not line.startswith("Run"):
                        # Could be a file:line reference
                        findings.append(
                            Finding(
                                checker=f"{self.category}.{self.name}",
                                severity=Severity.ERROR,
                                message=line,
                            )
                        )

            if not findings:
                findings.append(
                    Finding(
                        checker=f"{self.category}.{self.name}",
                        severity=Severity.ERROR,
                        message="Bandit security check failed",
                    )
                )

        duration_ms = int((time.monotonic() - start_time) * 1000)
        return CheckResult(
            checker=f"{self.category}.{self.name}",
            findings=tuple(findings),
            duration_ms=duration_ms,
        )


class PipAuditChecker:
    """Checker for known vulnerabilities using pip-audit.

    Scans dependencies for known CVEs.
    """

    @property
    def name(self) -> str:
        return "pip_audit"

    @property
    def category(self) -> str:
        return "security"

    @property
    def description(self) -> str:
        return "Check for known vulnerabilities in dependencies"

    def check(self, ctx: CheckContext) -> CheckResult:  # noqa: C901
        start_time = time.monotonic()
        findings: list[Finding] = []

        # Build environment with macOS library path fix
        env: dict[str, str] = {}
        if sys.platform == "darwin":
            base_lib_dir = Path(sys.base_prefix) / "lib"
            if base_lib_dir.exists():
                fallback_var = "DYLD_FALLBACK_LIBRARY_PATH"
                existing = os.environ.get(fallback_var, "")
                fallback_parts = [str(base_lib_dir)]
                if existing:
                    fallback_parts.append(existing)
                env[fallback_var] = ":".join(fallback_parts)

        result = run_python_module(
            "pip_audit",
            [
                "--progress-spinner",
                "off",
                "--strict",
                "--skip-editable",
                str(ctx.project_root),
            ],
            cwd=ctx.project_root,
            env=env,
            timeout_seconds=120.0,
        )

        if not result.success:
            output = result.output.strip()
            if output and "warning" not in output.lower():
                # Parse pip-audit output for vulnerabilities
                for line in output.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if "vulnerability" in line.lower() or "cve" in line.lower():
                        findings.append(
                            Finding(
                                checker=f"{self.category}.{self.name}",
                                severity=Severity.ERROR,
                                message=line,
                            )
                        )
                    elif line.startswith("Name"):
                        # Header line, skip
                        continue
                    elif "---" in line:
                        # Separator, skip
                        continue

            if not findings and not result.success:
                findings.append(
                    Finding(
                        checker=f"{self.category}.{self.name}",
                        severity=Severity.ERROR,
                        message=f"pip-audit check failed: {output or 'unknown error'}",
                    )
                )

        duration_ms = int((time.monotonic() - start_time) * 1000)
        return CheckResult(
            checker=f"{self.category}.{self.name}",
            findings=tuple(findings),
            duration_ms=duration_ms,
        )

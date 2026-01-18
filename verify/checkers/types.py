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

"""Type checking verification checkers.

These checkers verify type correctness:
- Type coverage: 100% type completeness
- Integration types: Integration tests type-check
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from core_types import CheckContext, CheckResult, Finding, Severity
from subprocess_utils import run_tool

PACKAGE_NAME = "weakincentives"
DEFAULT_TYPE_COVERAGE_THRESHOLD = 100.0


class TypeCoverageChecker:
    """Checker for type coverage using pyright --verifytypes.

    Ensures that the package has complete type annotations.
    """

    def __init__(self, threshold: float = DEFAULT_TYPE_COVERAGE_THRESHOLD) -> None:
        """Initialize the checker.

        Args:
            threshold: Minimum required type completeness percentage.
        """
        super().__init__()
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "type_coverage"

    @property
    def category(self) -> str:
        return "types"

    @property
    def description(self) -> str:
        return "Check type annotation completeness"

    def check(self, ctx: CheckContext) -> CheckResult:
        start_time = time.monotonic()
        findings: list[Finding] = []

        result = run_tool(
            [
                "pyright",
                "--verifytypes",
                PACKAGE_NAME,
                "--ignoreexternal",
                "--outputjson",
            ],
            cwd=ctx.project_root,
            timeout_seconds=120.0,
        )

        if result.returncode != 0 and not result.stdout:
            findings.append(
                Finding(
                    checker=f"{self.category}.{self.name}",
                    severity=Severity.ERROR,
                    message=f"pyright --verifytypes failed: {result.stderr}",
                )
            )
        elif result.stdout:
            try:
                data = json.loads(result.stdout)
                type_completeness = data.get("typeCompleteness", {})
                score = type_completeness.get("completenessScore", 0.0) * 100

                exported = type_completeness.get("exportedSymbolCounts", {})
                known = exported.get("withKnownType", 0)
                ambiguous = exported.get("withAmbiguousType", 0)
                unknown = exported.get("withUnknownType", 0)

                if score < self._threshold:
                    findings.append(
                        Finding(
                            checker=f"{self.category}.{self.name}",
                            severity=Severity.ERROR,
                            message=(
                                f"Type coverage {score:.1f}% is below the required "
                                f"threshold of {self._threshold:.1f}% "
                                f"({known} known, {ambiguous} ambiguous, {unknown} unknown)"
                            ),
                        )
                    )

                    # Add details about symbols with unknown types
                    symbols = type_completeness.get("symbols", [])
                    findings.extend(
                        Finding(
                            checker=f"{self.category}.{self.name}",
                            severity=Severity.WARNING,
                            message=f"Unknown type: {symbol.get('name', 'unknown')}",
                        )
                        for symbol in symbols[:10]  # Limit to first 10
                        if symbol.get("category") == "unknown"
                    )

            except json.JSONDecodeError:
                findings.append(
                    Finding(
                        checker=f"{self.category}.{self.name}",
                        severity=Severity.ERROR,
                        message="Failed to parse pyright output",
                    )
                )

        duration_ms = int((time.monotonic() - start_time) * 1000)
        return CheckResult(
            checker=f"{self.category}.{self.name}",
            findings=tuple(findings),
            duration_ms=duration_ms,
        )


class IntegrationTypesChecker:
    """Checker for integration test types.

    Validates that integration tests type-check without running them.
    """

    INTEGRATION_TESTS_DIR = "integration-tests"

    @property
    def name(self) -> str:
        return "integration_types"

    @property
    def category(self) -> str:
        return "types"

    @property
    def description(self) -> str:
        return "Check that integration tests type-check"

    def check(self, ctx: CheckContext) -> CheckResult:
        start_time = time.monotonic()
        findings: list[Finding] = []

        integration_dir = ctx.project_root / self.INTEGRATION_TESTS_DIR
        if not integration_dir.exists():
            # No integration tests, that's fine
            return CheckResult(
                checker=f"{self.category}.{self.name}",
                findings=(),
                duration_ms=int((time.monotonic() - start_time) * 1000),
            )

        # Create temporary pyright config
        config = {
            "typeCheckingMode": "basic",
            "pythonVersion": "3.12",
            "include": [self.INTEGRATION_TESTS_DIR],
        }

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix=".pyrightconfig-integration-",
            dir=ctx.project_root,
            delete=False,
        ) as f:
            json.dump(config, f)
            config_path = Path(f.name)

        try:
            result = run_tool(
                ["pyright", "--project", str(config_path)],
                cwd=ctx.project_root,
                timeout_seconds=120.0,
            )

            if not result.success:
                output = result.output.strip()
                if output:
                    # Parse pyright errors
                    for line in output.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        # Pyright outputs errors like:
                        # file.py:10:5 - error: Something wrong
                        if " - error:" in line or " - warning:" in line:
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
                            message="Integration test type checking failed",
                        )
                    )
        finally:
            config_path.unlink(missing_ok=True)

        duration_ms = int((time.monotonic() - start_time) * 1000)
        return CheckResult(
            checker=f"{self.category}.{self.name}",
            findings=tuple(findings),
            duration_ms=duration_ms,
        )

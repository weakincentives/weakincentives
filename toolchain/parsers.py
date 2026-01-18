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

"""Diagnostic parsers for common tool output formats.

Parsers extract structured Diagnostic objects from raw tool output.
Each parser takes (output: str, exit_code: int) and returns a tuple of Diagnostics.

The goal is to extract actionable information that lets you immediately navigate
to and understand the issue without reading raw output.
"""

from __future__ import annotations

import re

from .result import Diagnostic, Location


def parse_ruff(output: str, _code: int) -> tuple[Diagnostic, ...]:
    """Parse ruff output format: file:line:col: CODE message."""
    diagnostics = []
    pattern = re.compile(r"^(.+?):(\d+):(\d+): (\w+) (.+)$", re.MULTILINE)

    for match in pattern.finditer(output):
        file, line, col, code, message = match.groups()
        diagnostics.append(
            Diagnostic(
                message=f"[{code}] {message}",
                location=Location(file=file, line=int(line), column=int(col)),
            )
        )

    return tuple(diagnostics)


def parse_pyright(output: str, _code: int) -> tuple[Diagnostic, ...]:
    """Parse pyright output format: file:line:col - severity: message."""
    diagnostics = []
    pattern = re.compile(r"^  (.+?):(\d+):(\d+) - (error|warning|info): (.+)$", re.MULTILINE)

    for match in pattern.finditer(output):
        file, line, col, severity, message = match.groups()
        sev = "error" if severity == "error" else ("warning" if severity == "warning" else "info")
        diagnostics.append(
            Diagnostic(
                message=message,
                location=Location(file=file, line=int(line), column=int(col)),
                severity=sev,  # type: ignore[arg-type]
            )
        )

    return tuple(diagnostics)


def parse_ty(output: str, _code: int) -> tuple[Diagnostic, ...]:
    """Parse ty output format: error[code]: message --> file:line:col."""
    diagnostics = []
    pattern = re.compile(
        r"(error|warning)\[([^\]]+)\]: (.+?)\n\s*--> (.+?):(\d+):(\d+)",
        re.MULTILINE,
    )

    for match in pattern.finditer(output):
        severity, _code, message, file, line, col = match.groups()
        sev = "error" if severity == "error" else "warning"
        diagnostics.append(
            Diagnostic(
                message=message.strip(),
                location=Location(file=file, line=int(line), column=int(col)),
                severity=sev,  # type: ignore[arg-type]
            )
        )

    return tuple(diagnostics)


def parse_pytest(output: str, code: int) -> tuple[Diagnostic, ...]:
    """Parse pytest output for failures with assertion details.

    Extracts:
    - Test file and function name
    - Line number where assertion failed
    - The actual assertion error message
    """
    if code == 0:
        return ()

    diagnostics: list[Diagnostic] = []

    # Parse FAILED lines with short test failure summary
    # Format: FAILED tests/path/test.py::test_name - AssertionError: message
    failed_pattern = re.compile(
        r"^FAILED (.+?)::(\S+?)(?:\s+-\s+(.+))?$",
        re.MULTILINE,
    )

    for match in failed_pattern.finditer(output):
        file, test, reason = match.groups()
        if reason:
            # Clean up the reason - take first line if multiline
            reason = reason.split("\n")[0].strip()
            msg = f"{test}: {reason}"
        else:
            msg = f"{test}"

        # Try to find line number from traceback
        line_num = _find_test_failure_line(output, file, test)

        diagnostics.append(
            Diagnostic(
                message=msg,
                location=Location(file=file, line=line_num),
            )
        )

    # Check for collection errors (syntax errors, import errors)
    collection_error = re.search(
        r"ERROR collecting (.+?)\n.*?(?:SyntaxError|ImportError|ModuleNotFoundError): (.+?)(?:\n|$)",
        output,
        re.DOTALL,
    )
    if collection_error:
        file, error = collection_error.groups()
        diagnostics.append(
            Diagnostic(
                message=f"Collection error: {error.strip()}",
                location=Location(file=file.strip()),
            )
        )

    # Check for coverage failures
    cov_match = re.search(
        r"FAIL Required test coverage of (\d+(?:\.\d+)?)% not reached\. Total coverage: (\d+(?:\.\d+)?)%",
        output,
    )
    if cov_match:
        required, actual = cov_match.groups()
        diagnostics.append(
            Diagnostic(message=f"Coverage {actual}% < {required}% required")
        )

    # If nothing parsed but failed, add generic message
    if not diagnostics and code != 0:
        # Look for any error hint
        if "no tests ran" in output.lower():
            diagnostics.append(Diagnostic("No tests ran"))
        else:
            diagnostics.append(Diagnostic("Tests failed (run with -v for details)"))

    return tuple(diagnostics)


def _find_test_failure_line(output: str, file: str, test: str) -> int | None:
    """Extract line number from pytest traceback for a specific test."""
    # Look for the traceback section for this test
    # Pattern: file.py:123: in test_name
    pattern = re.compile(rf"{re.escape(file)}:(\d+).*?{re.escape(test)}")
    match = pattern.search(output)
    if match:
        return int(match.group(1))

    # Alternative: look for assertion line in traceback
    # >       assert x == y
    # file.py:42: AssertionError
    assertion_pattern = re.compile(rf"{re.escape(file)}:(\d+): (?:AssertionError|assert)")
    match = assertion_pattern.search(output)
    if match:
        return int(match.group(1))

    return None


def parse_bandit(output: str, code: int) -> tuple[Diagnostic, ...]:
    """Parse bandit output for security issues."""
    if code == 0:
        return ()

    diagnostics = []
    # Pattern: >> Issue: [CODE:SEVERITY:CONFIDENCE] message
    # Location: file:line:col
    issue_pattern = re.compile(
        r">> Issue: \[([^\]]+)\] (.+?)\n.*?Location: (.+?):(\d+)",
        re.MULTILINE | re.DOTALL,
    )

    for match in issue_pattern.finditer(output):
        code_info, message, file, line = match.groups()
        diagnostics.append(
            Diagnostic(
                message=f"[{code_info}] {message.strip()}",
                location=Location(file=file, line=int(line)),
            )
        )

    return tuple(diagnostics)


def parse_deptry(output: str, code: int) -> tuple[Diagnostic, ...]:
    """Parse deptry output for dependency issues."""
    if code == 0:
        return ()

    diagnostics = []
    # Deptry format: file:line:col: DEP00X 'package' message
    pattern = re.compile(r"^(.+?):(\d+):(\d+): (DEP\d+) (.+)$", re.MULTILINE)

    for match in pattern.finditer(output):
        file, line, col, code, message = match.groups()
        diagnostics.append(
            Diagnostic(
                message=f"[{code}] {message}",
                location=Location(file=file, line=int(line), column=int(col)),
            )
        )

    # Fallback for older deptry format
    if not diagnostics:
        for line in output.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith(("-", "=", "Found", "Scanning")):
                diagnostics.append(Diagnostic(message=line))

    return tuple(diagnostics)


def parse_pip_audit(output: str, code: int) -> tuple[Diagnostic, ...]:
    """Parse pip-audit output for vulnerabilities."""
    if code == 0:
        return ()

    diagnostics = []
    # Look for vulnerability entries
    # Format varies, but typically: package version has vulnerability ID
    vuln_pattern = re.compile(r"(\S+)\s+(\S+)\s+.*?(CVE-\d+-\d+|GHSA-\S+)", re.IGNORECASE)

    for match in vuln_pattern.finditer(output):
        package, version, vuln_id = match.groups()
        diagnostics.append(
            Diagnostic(message=f"{package}=={version} has {vuln_id}")
        )

    if not diagnostics and code != 0:
        diagnostics.append(Diagnostic("Vulnerability check failed"))

    return tuple(diagnostics)


def parse_mdformat(output: str, code: int) -> tuple[Diagnostic, ...]:
    """Parse mdformat --check output."""
    if code == 0:
        return ()

    diagnostics = []
    # mdformat outputs files that would be changed
    for line in output.strip().split("\n"):
        line = line.strip()
        if line.endswith(".md"):
            diagnostics.append(
                Diagnostic(
                    message="Needs formatting",
                    location=Location(file=line),
                )
            )

    return tuple(diagnostics)

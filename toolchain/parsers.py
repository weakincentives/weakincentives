"""Diagnostic parsers for common tool output formats.

Parsers extract structured Diagnostic objects from raw tool output.
Each parser takes (output: str, exit_code: int) and returns a tuple of Diagnostics.
"""

from __future__ import annotations

import re

from .result import Diagnostic, Location


def parse_ruff(output: str, _code: int) -> tuple[Diagnostic, ...]:
    """Parse ruff output format: file:line:col: CODE message."""
    diagnostics = []
    # Pattern: path:line:col: CODE message
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
    # Pattern: path:line:col - error: message
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
    """Parse ty output format: error[code]: message\n  --> file:line:col."""
    diagnostics = []
    # Pattern: error[code]: message followed by --> file:line:col
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
    """Parse pytest output for failures."""
    if code == 0:
        return ()

    diagnostics = []
    # Look for FAILED lines: FAILED tests/path/test.py::test_name
    pattern = re.compile(r"^FAILED (.+?)::(.+?)(?:\s|$)", re.MULTILINE)

    for match in pattern.finditer(output):
        file, test = match.groups()
        diagnostics.append(
            Diagnostic(
                message=f"Test failed: {test}",
                location=Location(file=file),
            )
        )

    # If no specific failures found but exit code non-zero, add generic diagnostic
    if not diagnostics and code != 0:
        # Check for common error patterns
        if "SyntaxError" in output:
            diagnostics.append(Diagnostic("Syntax error in test file"))
        elif "ImportError" in output or "ModuleNotFoundError" in output:
            diagnostics.append(Diagnostic("Import error in tests"))
        elif "coverage" in output.lower() and "fail" in output.lower():
            # Coverage failure
            cov_match = re.search(r"Coverage failure.*?(\d+(?:\.\d+)?%)", output, re.IGNORECASE)
            if cov_match:
                diagnostics.append(Diagnostic(f"Coverage below threshold: {cov_match.group(1)}"))
            else:
                diagnostics.append(Diagnostic("Coverage below required threshold"))
        else:
            diagnostics.append(Diagnostic("Tests failed"))

    return tuple(diagnostics)


def parse_bandit(output: str, code: int) -> tuple[Diagnostic, ...]:
    """Parse bandit output for security issues."""
    if code == 0:
        return ()

    diagnostics = []
    # Pattern: >> Issue: [CODE:SEVERITY:CONFIDENCE] message
    # followed by    Location: file:line:col
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
    # Deptry outputs lines like: package: error message
    for line in output.strip().split("\n"):
        line = line.strip()
        if line and not line.startswith(("-", "=", "Found")):
            diagnostics.append(Diagnostic(message=line))

    return tuple(diagnostics)


def parse_pip_audit(output: str, code: int) -> tuple[Diagnostic, ...]:
    """Parse pip-audit output for vulnerabilities."""
    if code == 0:
        return ()

    diagnostics = []
    # pip-audit outputs vulnerability info
    lines = output.strip().split("\n")
    for line in lines:
        if "vulnerability" in line.lower() or "CVE" in line:
            diagnostics.append(Diagnostic(message=line.strip()))

    if not diagnostics and code != 0:
        diagnostics.append(Diagnostic("Vulnerability check failed"))

    return tuple(diagnostics)


def parse_mdformat(output: str, code: int) -> tuple[Diagnostic, ...]:
    """Parse mdformat --check output."""
    if code == 0:
        return ()

    diagnostics = []
    # mdformat --check outputs files that would be reformatted
    for line in output.strip().split("\n"):
        line = line.strip()
        if line.endswith(".md"):
            diagnostics.append(
                Diagnostic(
                    message="File needs formatting",
                    location=Location(file=line),
                )
            )

    return tuple(diagnostics)

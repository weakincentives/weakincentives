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

# Limits for diagnostic output to avoid overwhelming the user
MAX_UNCOVERED_FILES = 10  # Max files shown in coverage failure
MAX_TEST_FILES_PREVIEW = 5  # Max test files shown in failure message
MAX_TRACEBACK_LINES = 5  # Max traceback lines per test failure
MAX_OUTPUT_PREVIEW_LINES = 5  # Max raw output lines in fallback messages


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
    """Parse pyright output format: file:line:col - severity: message.

    Preserves rule names like (reportGeneralTypeIssues) in the message.
    """
    diagnostics = []
    pattern = re.compile(r"^  (.+?):(\d+):(\d+) - (error|warning|info): (.+)$", re.MULTILINE)

    for match in pattern.finditer(output):
        file, line, col, severity, message = match.groups()
        sev = "error" if severity == "error" else ("warning" if severity == "warning" else "info")

        # Extract rule name if present (e.g., "(reportGeneralTypeIssues)")
        rule_match = re.search(r"\((\w+)\)$", message)
        if rule_match:
            rule_name = rule_match.group(1)
            # Remove rule from end of message and prepend as code
            base_message = message[: rule_match.start()].strip()
            formatted_message = f"[{rule_name}] {base_message}"
        else:
            formatted_message = message

        diagnostics.append(
            Diagnostic(
                message=formatted_message,
                location=Location(file=file, line=int(line), column=int(col)),
                severity=sev,  # type: ignore[arg-type]
            )
        )

    return tuple(diagnostics)


def parse_ty(output: str, _code: int) -> tuple[Diagnostic, ...]:
    """Parse ty output format: error[code]: message --> file:line:col.

    Preserves the error code in the message as [code] prefix.
    """
    diagnostics = []
    pattern = re.compile(
        r"(error|warning)\[([^\]]+)\]: (.+?)\n\s*--> (.+?):(\d+):(\d+)",
        re.MULTILINE,
    )

    for match in pattern.finditer(output):
        severity, error_code, message, file, line, col = match.groups()
        sev = "error" if severity == "error" else "warning"
        # Include error code in message for actionable diagnostics
        formatted_message = f"[{error_code}] {message.strip()}"
        diagnostics.append(
            Diagnostic(
                message=formatted_message,
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
    - Traceback context for better debugging
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

        # Try to find line number and full context from traceback
        line_num, traceback_msg = _find_test_failure_line(output, file, test)

        # Build comprehensive error message
        if traceback_msg:
            # We have traceback context
            msg = f"{test}\n{traceback_msg}"
        elif reason:
            # Clean up the reason - take first line if multiline
            reason = reason.split("\n")[0].strip()
            msg = f"{test}: {reason}"
        else:
            msg = f"{test}"

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
        # Extract uncovered files from coverage report
        uncovered_info = _extract_uncovered_files(output)
        if uncovered_info:
            msg = f"Coverage {actual}% < {required}% required\nUncovered:\n{uncovered_info}"
        else:
            msg = f"Coverage {actual}% < {required}% required"
        diagnostics.append(Diagnostic(message=msg))

    # If nothing parsed but failed, add generic message
    if not diagnostics and code != 0:
        # Look for any error hint
        if "no tests ran" in output.lower():
            msg = (
                "No tests ran\n"
                "Fix: Check test collection and file naming\n"
                "Run: uv run pytest tests -v"
            )
            diagnostics.append(Diagnostic(msg))
        else:
            # Try to extract test files mentioned in output
            test_files = _extract_test_files(output)
            if test_files:
                files_str = ", ".join(test_files[:MAX_TEST_FILES_PREVIEW])  # Limit to 5 files
                if len(test_files) > MAX_TEST_FILES_PREVIEW:
                    files_str += f" (and {len(test_files) - MAX_TEST_FILES_PREVIEW} more)"
                msg = (
                    f"Tests failed\n"
                    f"Files involved: {files_str}\n"
                    "Reproduce: uv run pytest tests -v\n"
                    "Focus on specific test: uv run pytest tests/path/to/test.py::test_name -vv"
                )
            else:
                msg = (
                    "Tests failed\n"
                    "Reproduce: uv run pytest tests -v\n"
                    "Focus on specific test: uv run pytest tests/path/to/test.py::test_name -vv"
                )
            diagnostics.append(Diagnostic(msg))

    return tuple(diagnostics)


def _find_test_failure_line(output: str, file: str, test: str) -> tuple[int | None, str | None]:
    """Extract line number and traceback context from pytest output for a specific test.

    Returns:
        Tuple of (line_number, traceback_message) where traceback_message includes
        the assertion details and context.
    """
    # Look for the test failure section in output
    # pytest --tb=short format includes:
    # file.py:123: in test_name
    #     assert x == y
    # E   AssertionError: message
    # or
    # >   assert x == y
    # E   assert False

    # Try to find the section for this specific test
    test_section_pattern = rf"_+ {re.escape(test)} _+.*?(?=_+ \w+ _+|$)"
    test_section_match = re.search(test_section_pattern, output, re.DOTALL)

    if test_section_match:
        section = test_section_match.group(0)

        # Extract line number and assertion context
        line_pattern = re.compile(rf"{re.escape(file)}:(\d+)")
        line_match = line_pattern.search(section)
        line_num = int(line_match.group(1)) if line_match else None

        # Extract assertion details (lines starting with 'E   ')
        error_lines = []
        for line in section.split("\n"):
            if line.startswith("E   "):
                error_lines.append(line[4:])  # Remove 'E   ' prefix
            elif line.strip().startswith(">") and "assert" in line:
                error_lines.append(line.strip()[1:].strip())  # Add assertion line

        if error_lines:
            traceback_msg = "\n".join(error_lines[:MAX_TRACEBACK_LINES])  # Limit to first 5 lines
            return line_num, traceback_msg

        return line_num, None

    # Fallback: look for assertion line in general traceback
    pattern = re.compile(rf"{re.escape(file)}:(\d+).*?{re.escape(test)}")
    match = pattern.search(output)
    if match:
        return int(match.group(1)), None

    # Alternative fallback
    assertion_pattern = re.compile(rf"{re.escape(file)}:(\d+): (?:AssertionError|assert)")
    match = assertion_pattern.search(output)
    if match:
        return int(match.group(1)), None

    return None, None


def _extract_uncovered_files(output: str) -> str | None:
    """Extract files with incomplete coverage from pytest coverage output.

    Parses pytest-cov output format (with or without branch coverage):
    Name                                      Stmts   Miss  Cover
    Name                                      Stmts   Miss Branch BrPart  Cover
    -------------------------------------------------------------
    src/module.py                               100     10    90%
    src/module.py                               100     10     20      5    90%   1-5, 10
    TOTAL                                       500     50    90%

    Returns a formatted string listing uncovered files and their missing line info.
    """
    uncovered_files = []

    # Look for the coverage table in output
    # Pattern matches lines with variable columns before percentage:
    # - Basic: file   stmts   miss   cover%
    # - Branch: file   stmts   miss   branch   brpart   cover%
    # Capture file, miss count, coverage %, and optional missing lines
    # Use {0,2} instead of *? to avoid potential regex backtracking on malformed input
    cov_line_pattern = re.compile(
        r"^([\w/.-]+\.py)\s+"  # file path
        r"(\d+)\s+"  # stmts
        r"(\d+)\s+"  # miss
        r"(?:\d+\s+){0,2}"  # 0-2 optional branch columns (Branch, BrPart)
        r"(\d+)%"  # coverage percentage
        r"(?:\s+([\d,\s-]+))?$",  # optional missing lines
        re.MULTILINE,
    )

    for match in cov_line_pattern.finditer(output):
        file, _stmts, miss, cover, missing_lines = match.groups()
        if file == "TOTAL" or int(miss) == 0:
            continue
        if int(cover) < 100:
            if missing_lines and missing_lines.strip():
                uncovered_files.append(f"  {file} ({cover}%): lines {missing_lines.strip()}")
            else:
                uncovered_files.append(f"  {file} ({cover}%): {miss} lines missing")

    if uncovered_files:
        # Limit to first 10 files
        result = uncovered_files[:MAX_UNCOVERED_FILES]
        if len(uncovered_files) > MAX_UNCOVERED_FILES:
            result.append(f"  ... and {len(uncovered_files) - MAX_UNCOVERED_FILES} more files")
        return "\n".join(result)

    return None


def _extract_test_files(output: str) -> list[str]:
    """Extract test file names mentioned in pytest output.

    Looks for patterns like:
    - tests/test_foo.py
    - test_*.py
    - path/to/test_something.py
    """
    test_files: set[str] = set()

    # Pattern for test file paths
    test_file_pattern = re.compile(r"((?:[\w./]*)?tests?/[\w/]*test_\w+\.py|test_\w+\.py)")

    for match in test_file_pattern.finditer(output):
        test_files.add(match.group(1))

    return sorted(test_files)


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
                msg = (
                    f"{line}\n"
                    "Reproduce: uv run deptry src/weakincentives\n"
                    "Fix: Remove unused imports or add missing dependencies to pyproject.toml"
                )
                diagnostics.append(Diagnostic(message=msg))

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
        # Include first few meaningful lines of output for context
        output_lines = [
            line.strip()
            for line in output.strip().split("\n")
            if line.strip() and not line.startswith(("-", "=", "#"))
        ]
        output_preview = "\n".join(output_lines[:MAX_OUTPUT_PREVIEW_LINES])
        if len(output_lines) > MAX_OUTPUT_PREVIEW_LINES:
            output_preview += "\n..."

        if output_preview:
            msg = (
                f"Vulnerability check failed\n"
                f"Output:\n{output_preview}\n"
                "Reproduce: uv run pip-audit\n"
                "Fix: Update dependencies in pyproject.toml"
            )
        else:
            msg = (
                "Vulnerability check failed\n"
                "Reproduce: uv run pip-audit\n"
                "Fix: Update dependencies in pyproject.toml"
            )
        diagnostics.append(Diagnostic(msg))

    return tuple(diagnostics)


def parse_mdformat(output: str, code: int) -> tuple[Diagnostic, ...]:
    """Parse mdformat --check output."""
    if code == 0:
        return ()

    diagnostics = []

    # mdformat outputs in format: Error: File "path/to/file.md" is not formatted.
    # or just the file path on older versions
    error_pattern = re.compile(r'Error: File "([^"]+\.md)" is not formatted\.')
    for match in error_pattern.finditer(output):
        file_path = match.group(1)
        diagnostics.append(
            Diagnostic(
                message=(
                    f"Markdown formatting required\n"
                    f"Fix: Run `mdformat {file_path}` to auto-format"
                ),
                location=Location(file=file_path),
            )
        )

    # Fallback: look for lines that are just file paths (older mdformat versions)
    if not diagnostics:
        for line in output.strip().split("\n"):
            line = line.strip()
            if line.endswith(".md") and not line.startswith("Error"):
                diagnostics.append(
                    Diagnostic(
                        message=(
                            f"Markdown formatting required\n"
                            f"Fix: Run `mdformat {line}` to auto-format"
                        ),
                        location=Location(file=line),
                    )
                )

    return tuple(diagnostics)

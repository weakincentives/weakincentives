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

"""Built-in verification checkers.

This module provides factory functions to create all standard checkers
for the weakincentives project.
"""

from __future__ import annotations

from ..checker import AutoFormatChecker, Checker, SubprocessChecker
from ..parsers import (
    parse_bandit,
    parse_bun_test,
    parse_deptry,
    parse_mdformat,
    parse_pip_audit,
    parse_pyright,
    parse_pytest,
    parse_ruff,
    parse_ty,
)
from .architecture import ArchitectureChecker
from .code_length import CodeLengthChecker
from .docs import DocsChecker
from .private_imports import PrivateImportChecker


def create_format_checker() -> AutoFormatChecker:
    """Create the code formatting checker (ruff format).

    In local environments: auto-fixes formatting and reports changes.
    In CI environments: checks formatting without modifications.

    Uses JSON output internally for precise file path reporting.
    """
    return AutoFormatChecker(
        name="format",
        description="Check code formatting with ruff",
        check_command=["uv", "run", "ruff", "format", "--check", "."],
        fix_command=["uv", "run", "ruff", "format", "."],
        json_check_command=[
            "uv", "run", "ruff", "format",
            "--check", "--output-format=json", ".",
        ],
        parser=parse_ruff,
    )


def create_lint_checker() -> SubprocessChecker:
    """Create the linting checker (ruff check)."""
    return SubprocessChecker(
        name="lint",
        description="Check code style with ruff",
        command=["uv", "run", "ruff", "check", "--preview", "."],
        parser=parse_ruff,
    )


def create_typecheck_checker() -> SubprocessChecker:
    """Create the type checking checker (ty + pyright)."""
    # ty checks src/ with warnings as errors, pyright uses pyproject.toml config.
    return SubprocessChecker(
        name="typecheck",
        description="Check types with ty and pyright",
        command=[
            "bash",
            "-c",
            "uv run ty check --error-on-warning src && uv run pyright",
        ],
        parser=_parse_typecheck,
    )


def _parse_typecheck(output: str, code: int) -> tuple:
    """Parse combined ty + pyright output.

    Each diagnostic's message is prefixed with the tool name ([ty] or [pyright])
    to indicate which type checker produced the error.
    """
    from ..result import Diagnostic

    diagnostics: list[Diagnostic] = []

    # Try ty parser - prefix messages with [ty]
    for diag in parse_ty(output, code):
        prefixed = Diagnostic(
            message=f"[ty] {diag.message}",
            location=diag.location,
            severity=diag.severity,
        )
        diagnostics.append(prefixed)

    # Try pyright parser - prefix messages with [pyright]
    for diag in parse_pyright(output, code):
        prefixed = Diagnostic(
            message=f"[pyright] {diag.message}",
            location=diag.location,
            severity=diag.severity,
        )
        diagnostics.append(prefixed)

    return tuple(diagnostics)


def create_test_checker() -> SubprocessChecker:
    """Create the test checker (pytest)."""
    return SubprocessChecker(
        name="test",
        description="Run tests with pytest and coverage",
        command=[
            "uv",
            "run",
            "--all-extras",
            "pytest",
            "--strict-config",
            "--strict-markers",
            "--cov-fail-under=100",
            "--timeout=10",
            "--timeout-method=thread",
            "--tb=short",  # Short traceback format - shows enough context without being verbose
            "--no-header",
            "--cov-report=term-missing",
            "tests",
        ],
        parser=parse_pytest,
        timeout=600,  # 10 minutes for tests
    )


def create_bun_test_checker() -> SubprocessChecker:
    """Create the JavaScript test checker (bun test).

    Uses --coverage for coverage reporting and --only-failures to reduce output noise.
    The bash wrapper handles the case where bun is not installed by exiting 0.
    """
    return SubprocessChecker(
        name="bun-test",
        description="Run JavaScript tests with bun",
        command=[
            "bash",
            "-c",
            'command -v bun >/dev/null 2>&1 || { echo "bun not installed, skipping"; exit 0; }; '
            "bun test --coverage --only-failures tests/js/",
        ],
        parser=parse_bun_test,
        timeout=120,  # 2 minutes for JS tests
    )


def create_bandit_checker() -> SubprocessChecker:
    """Create the security checker (bandit)."""
    return SubprocessChecker(
        name="bandit",
        description="Security scanning with bandit",
        command=[
            "uv",
            "run",
            "bandit",
            "-r",
            "src/weakincentives",
            "-c",
            "pyproject.toml",
            "-q",
        ],
        parser=parse_bandit,
    )


def create_deptry_checker() -> SubprocessChecker:
    """Create the dependency checker (deptry)."""
    return SubprocessChecker(
        name="deptry",
        description="Check dependencies with deptry",
        command=["uv", "run", "deptry", "src/weakincentives"],
        parser=parse_deptry,
    )


def create_pip_audit_checker() -> SubprocessChecker:
    """Create the vulnerability checker (pip-audit)."""
    return SubprocessChecker(
        name="pip-audit",
        description="Vulnerability scanning with pip-audit",
        command=["uv", "run", "pip-audit"],
        parser=parse_pip_audit,
    )


def _parse_mdformat_file_list(output: str) -> list[str]:
    """Parse file paths from mdformat check output.

    mdformat outputs: Error: File "path/to/file.md" is not formatted.
    """
    import re

    files = []
    error_pattern = re.compile(r'Error: File "([^"]+)" is not formatted\.')
    for match in error_pattern.finditer(output):
        files.append(match.group(1))
    return sorted(files)


# Markdown targets for formatting
_MARKDOWN_TARGETS = [
    "README.md",
    "AGENTS.md",
    "CLAUDE.md",
    "llms.md",
    "GLOSSARY.md",
    "guides",
    "specs",
]


def create_markdown_checker() -> AutoFormatChecker:
    """Create the markdown formatting checker.

    In local environments: auto-fixes formatting and reports changes.
    In CI environments: checks formatting without modifications.
    """
    return AutoFormatChecker(
        name="markdown",
        description="Check markdown formatting with mdformat",
        check_command=["uv", "run", "mdformat", "--check", *_MARKDOWN_TARGETS],
        fix_command=["uv", "run", "mdformat", *_MARKDOWN_TARGETS],
        file_list_parser=_parse_mdformat_file_list,
        parser=parse_mdformat,
    )


def create_architecture_checker() -> ArchitectureChecker:
    """Create the architecture checker."""
    return ArchitectureChecker()


def create_code_length_checker() -> CodeLengthChecker:
    """Create the code length checker.

    Enforces max function/method length (120 lines) and max file
    length (720 lines).  Known violations in the baseline file are
    warnings; new violations are errors.
    """
    return CodeLengthChecker()


def create_docs_checker() -> DocsChecker:
    """Create the documentation checker."""
    return DocsChecker()


def create_private_imports_checker() -> PrivateImportChecker:
    """Create the private module import checker."""
    return PrivateImportChecker()


def create_all_checkers() -> list[Checker]:
    """Create all standard checkers in recommended execution order."""
    return [
        create_format_checker(),
        create_lint_checker(),
        create_typecheck_checker(),
        create_bandit_checker(),
        create_deptry_checker(),
        create_pip_audit_checker(),
        create_architecture_checker(),
        create_private_imports_checker(),
        create_code_length_checker(),
        create_docs_checker(),
        create_markdown_checker(),
        create_bun_test_checker(),
        create_test_checker(),
    ]


__all__ = [
    "create_all_checkers",
    "create_format_checker",
    "create_lint_checker",
    "create_typecheck_checker",
    "create_test_checker",
    "create_bun_test_checker",
    "create_bandit_checker",
    "create_deptry_checker",
    "create_pip_audit_checker",
    "create_markdown_checker",
    "create_architecture_checker",
    "create_private_imports_checker",
    "create_code_length_checker",
    "create_docs_checker",
]

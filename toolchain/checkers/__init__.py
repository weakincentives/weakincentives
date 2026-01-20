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

from ..checker import Checker, SubprocessChecker
from ..parsers import (
    parse_bandit,
    parse_deptry,
    parse_mdformat,
    parse_pip_audit,
    parse_pyright,
    parse_pytest,
    parse_ruff,
    parse_ty,
)
from .architecture import ArchitectureChecker
from .docs import DocsChecker


def create_format_checker() -> SubprocessChecker:
    """Create the code formatting checker (ruff format)."""
    return SubprocessChecker(
        name="format",
        description="Check code formatting with ruff",
        command=["uv", "run", "ruff", "format", "--check", "."],
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
    """Parse combined ty + pyright output."""
    from ..result import Diagnostic

    diagnostics: list[Diagnostic] = []

    # Try ty parser
    ty_diags = parse_ty(output, code)
    diagnostics.extend(ty_diags)

    # Try pyright parser
    pyright_diags = parse_pyright(output, code)
    diagnostics.extend(pyright_diags)

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
            "--cov-report=",
            "tests",
        ],
        parser=parse_pytest,
        timeout=600,  # 10 minutes for tests
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


def create_markdown_checker() -> SubprocessChecker:
    """Create the markdown formatting checker."""
    return SubprocessChecker(
        name="markdown",
        description="Check markdown formatting with mdformat",
        command=[
            "uv",
            "run",
            "mdformat",
            "--check",
            "README.md",
            "AGENTS.md",
            "CLAUDE.md",
            "llms.md",
            "GLOSSARY.md",
            "guides",
            "specs",
        ],
        parser=parse_mdformat,
    )


def create_architecture_checker() -> ArchitectureChecker:
    """Create the architecture checker."""
    return ArchitectureChecker()


def create_docs_checker() -> DocsChecker:
    """Create the documentation checker."""
    return DocsChecker()


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
        create_docs_checker(),
        create_markdown_checker(),
        create_test_checker(),
    ]


__all__ = [
    "create_all_checkers",
    "create_format_checker",
    "create_lint_checker",
    "create_typecheck_checker",
    "create_test_checker",
    "create_bandit_checker",
    "create_deptry_checker",
    "create_pip_audit_checker",
    "create_markdown_checker",
    "create_architecture_checker",
    "create_docs_checker",
]

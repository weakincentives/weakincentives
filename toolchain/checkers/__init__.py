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

from ..checker import (
    AutoFormatChecker,
    Checker,
    SubprocessChecker,
    is_ci_environment,
)
from ..parsers import (
    parse_bandit,
    parse_biome,
    parse_biome_files,
    parse_bun_test,
    parse_deptry,
    parse_mdformat,
    parse_pip_audit,
    parse_pytest,
    parse_ruff_format,
    parse_ruff_format_files,
    parse_ruff_json,
    parse_vulture,
)
from .architecture import ArchitectureChecker
from .banned_time_imports import BannedTimeImportsChecker
from .code_length import CodeLengthChecker
from .docs import DocsChecker
from .private_imports import PrivateImportChecker
from .typecheck import TypecheckChecker


def create_format_checker() -> AutoFormatChecker:
    """Create the code formatting checker (ruff format).

    In local environments: auto-fixes formatting and reports changes.
    In CI environments: checks formatting without modifications.

    File paths are extracted from the `Would reformat: <path>` check output.
    """
    return AutoFormatChecker(
        name="format",
        description="Check code formatting with ruff",
        check_command=["uv", "run", "ruff", "format", "--check", "."],
        fix_command=["uv", "run", "ruff", "format", "."],
        file_list_parser=parse_ruff_format_files,
        parser=parse_ruff_format,
    )


def create_lint_checker() -> SubprocessChecker:
    """Create the linting checker (ruff check).

    Uses JSON output - ruff's stable machine-readable format - so diagnostics
    survive changes to the human-readable output across ruff releases.
    """
    return SubprocessChecker(
        name="lint",
        description="Check code style with ruff",
        command=[
            "uv", "run", "ruff", "check",
            "--preview", "--output-format=json", ".",
        ],
        parser=parse_ruff_json,
    )


def create_typecheck_checker() -> TypecheckChecker:
    """Create the type checking checker (ty + pyright).

    Both tools always run - one failing tool does not hide the other's
    errors - and same-line findings are merged into [ty+pyright] entries.
    """
    return TypecheckChecker()


def create_test_checker() -> SubprocessChecker:
    """Create the test checker (pytest).

    In CI: full test suite with 100% coverage enforcement.
    Locally: only tests affected by changes, via the testmon database
    (the first run builds it; later runs skip unaffected tests).
    """
    if is_ci_environment():
        command = [
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
        ]
        description = "Run tests with pytest and coverage"
    else:
        command = [
            "uv",
            "run",
            "--all-extras",
            "pytest",
            "-p",
            "no:cov",
            "-o",
            "addopts=",
            "--testmon",
            "--strict-config",
            "--strict-markers",
            "--timeout=10",
            "--timeout-method=thread",
            "--tb=short",
            "--no-header",
            "--reruns=2",
            "--reruns-delay=0.5",
            "tests",
        ]
        description = "Run tests affected by changes (testmon)"
    return SubprocessChecker(
        name="test",
        description=description,
        command=command,
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


# Guard shared by the biome commands: skip when npx is unavailable, install
# node_modules on first run.
_BIOME_GUARD = (
    'command -v npx >/dev/null 2>&1 || { echo "npx not installed, skipping"; exit 0; }; '
    "[ -d node_modules ] || npm install --silent; "
)
_BIOME_TARGET = "src/weakincentives/cli/static/"


def create_biome_checker() -> AutoFormatChecker:
    """Create the frontend lint checker (biome).

    In local environments: applies biome's safe fixes (formatting and lint)
    and reports the affected files; issues without an auto-fix still fail the
    check with structured diagnostics. In CI environments: check-only.

    Exits 0 with a skip message when npx is unavailable. The GitHub reporter
    gives stable one-line-per-issue output that parses into structured
    diagnostics.
    """
    check_script = f"{_BIOME_GUARD}npx biome check --reporter=github {_BIOME_TARGET}"
    fix_script = (
        f"{_BIOME_GUARD}npx biome check --write --reporter=github {_BIOME_TARGET}"
    )
    return AutoFormatChecker(
        name="biome",
        description="Lint frontend static files with Biome",
        check_command=["bash", "-c", check_script],
        fix_command=["bash", "-c", fix_script],
        file_list_parser=parse_biome_files,
        parser=parse_biome,
        timeout=120,
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
        command=["uv", "run", "pip-audit", "--ignore-vuln", "CVE-2026-4539"],
        parser=parse_pip_audit,
    )


def _parse_mdformat_file_list(output: str) -> list[str]:
    """Parse file paths from mdformat check output.

    mdformat outputs: Error: File "path/to/file.md" is not formatted.
    Long messages word-wrap at arbitrary points depending on path length, so
    every inter-token gap must tolerate newlines.
    """
    import re

    files = []
    error_pattern = re.compile(r'Error:\s+File\s+"([^"]+)"\s+is\s+not\s+formatted\.')
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


def create_banned_time_imports_checker() -> BannedTimeImportsChecker:
    """Create the banned time imports checker.

    Flags direct ``import time`` in src/weakincentives/ (excluding clock.py).
    Production code must use clock protocols instead.
    """
    return BannedTimeImportsChecker()


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


def create_dead_code_checker() -> SubprocessChecker:
    """Create the dead code checker (vulture).

    Detects unused code (functions, classes, variables, imports) at the
    configured minimum confidence level. Configuration is read from the
    ``[tool.vulture]`` section of ``pyproject.toml``.
    """
    return SubprocessChecker(
        name="dead-code",
        description="Detect unused code with vulture",
        command=["uv", "run", "vulture"],
        parser=parse_vulture,
    )


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
        create_banned_time_imports_checker(),
        create_code_length_checker(),
        create_dead_code_checker(),
        create_docs_checker(),
        create_markdown_checker(),
        create_biome_checker(),
        create_bun_test_checker(),
        create_test_checker(),
    ]


__all__ = [
    "create_all_checkers",
    "create_format_checker",
    "create_lint_checker",
    "create_typecheck_checker",
    "create_test_checker",
    "create_biome_checker",
    "create_bun_test_checker",
    "create_bandit_checker",
    "create_deptry_checker",
    "create_pip_audit_checker",
    "create_markdown_checker",
    "create_architecture_checker",
    "create_private_imports_checker",
    "create_banned_time_imports_checker",
    "create_code_length_checker",
    "create_dead_code_checker",
    "create_docs_checker",
]

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

"""Core types for the verification toolbox."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    pass


class Severity(Enum):
    """Severity level for verification findings."""

    ERROR = auto()
    WARNING = auto()
    INFO = auto()


@dataclass(frozen=True, slots=True)
class Finding:
    """A single verification finding.

    Attributes:
        checker: The checker that produced this finding (e.g., "architecture.layer_violations").
        severity: The severity level of the finding.
        message: Human-readable description of the issue.
        file: The file where the issue was found, if applicable.
        line: The line number where the issue was found, if applicable.
        suggestion: A suggested fix, if available.
    """

    checker: str
    severity: Severity
    message: str
    file: Path | None = None
    line: int | None = None
    suggestion: str | None = None

    def format(self, *, show_checker: bool = True) -> str:
        """Format the finding for display."""
        parts: list[str] = []

        if show_checker:
            parts.append(f"[{self.checker}]")

        severity_str = self.severity.name.lower()
        parts.append(f"{severity_str}:")

        if self.file is not None:
            location = str(self.file)
            if self.line is not None:
                location = f"{location}:{self.line}"
            parts.append(location)

        parts.append(self.message)

        result = " ".join(parts)

        if self.suggestion:
            result = f"{result}\n  suggestion: {self.suggestion}"

        return result


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Result of running a checker.

    Attributes:
        checker: The name of the checker that produced this result.
        findings: All findings from the check.
        duration_ms: How long the check took in milliseconds.
    """

    checker: str
    findings: tuple[Finding, ...]
    duration_ms: int

    @property
    def passed(self) -> bool:
        """Check if the result indicates success (no errors)."""
        return not any(f.severity == Severity.ERROR for f in self.findings)

    @property
    def error_count(self) -> int:
        """Count the number of errors."""
        return sum(1 for f in self.findings if f.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        """Count the number of warnings."""
        return sum(1 for f in self.findings if f.severity == Severity.WARNING)


@dataclass(frozen=True, slots=True)
class CheckContext:
    """Context passed to all checkers.

    Attributes:
        project_root: The root directory of the project.
        src_dir: The source directory (project_root / "src").
        quiet: Whether to suppress non-error output.
        fix: Whether to apply auto-fixes where supported.
    """

    project_root: Path
    src_dir: Path
    quiet: bool = False
    fix: bool = False

    @classmethod
    def from_project_root(
        cls,
        project_root: Path,
        *,
        quiet: bool = False,
        fix: bool = False,
    ) -> CheckContext:
        """Create a context from a project root directory."""
        return cls(
            project_root=project_root.resolve(),
            src_dir=(project_root / "src").resolve(),
            quiet=quiet,
            fix=fix,
        )


@runtime_checkable
class Checker(Protocol):
    """Protocol for verification checkers.

    All checkers must implement this protocol to be usable with the
    verification runner.
    """

    @property
    def name(self) -> str:
        """Short identifier for the checker (e.g., 'layer_violations')."""
        ...

    @property
    def category(self) -> str:
        """Category the checker belongs to (e.g., 'architecture')."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of what this checker verifies."""
        ...

    def check(self, ctx: CheckContext) -> CheckResult:
        """Run the check and return findings.

        Args:
            ctx: The check context with project paths and options.

        Returns:
            A CheckResult containing all findings from this check.
        """
        ...


@dataclass(slots=True)
class RunConfig:
    """Configuration for running checkers.

    Attributes:
        max_failures: Stop after this many failures (None = no limit).
        max_parallel: Maximum number of checkers to run in parallel.
        categories: If set, only run checkers in these categories.
        checkers: If set, only run checkers with these names.
    """

    max_failures: int | None = None
    max_parallel: int | None = None
    categories: frozenset[str] | None = None
    checkers: frozenset[str] | None = None

    def should_run(self, checker: Checker) -> bool:
        """Check if a checker should be run given this config."""
        if self.categories is not None and checker.category not in self.categories:
            return False
        return not (self.checkers is not None and checker.name not in self.checkers)

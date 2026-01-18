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

"""Result types for verification checks.

This module defines the core data structures for representing check results:
- Location: File position (file:line:column)
- Diagnostic: A single issue with location and severity
- CheckResult: Complete result of running a single checker
- Report: Aggregated results from multiple checkers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class Location:
    """A position in a file."""

    file: str
    line: int | None = None
    column: int | None = None

    def __str__(self) -> str:
        if self.line is None:
            return self.file
        if self.column is None:
            return f"{self.file}:{self.line}"
        return f"{self.file}:{self.line}:{self.column}"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A single issue found by a checker."""

    message: str
    location: Location | None = None
    severity: Literal["error", "warning", "info"] = "error"

    def __str__(self) -> str:
        if self.location:
            return f"{self.location}: {self.message}"
        return self.message


Status = Literal["passed", "failed", "skipped"]


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Result of running a single checker."""

    name: str
    status: Status
    duration_ms: int
    diagnostics: tuple[Diagnostic, ...] = ()
    output: str = ""  # Raw output for debugging

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    @property
    def failed(self) -> bool:
        return self.status == "failed"

    @property
    def error_count(self) -> int:
        return sum(1 for d in self.diagnostics if d.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for d in self.diagnostics if d.severity == "warning")


@dataclass(frozen=True, slots=True)
class Report:
    """Aggregated results from running multiple checkers."""

    results: tuple[CheckResult, ...]
    total_duration_ms: int

    @property
    def passed(self) -> bool:
        return all(r.status != "failed" for r in self.results)

    @property
    def failed_results(self) -> tuple[CheckResult, ...]:
        return tuple(r for r in self.results if r.failed)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if r.failed)

    @property
    def skipped_count(self) -> int:
        return sum(1 for r in self.results if r.status == "skipped")

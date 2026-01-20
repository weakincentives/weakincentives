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

"""Runner for executing verification checkers.

The Runner orchestrates executing multiple checkers and collecting their
results into a Report.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .result import Report

if TYPE_CHECKING:
    from .checker import Checker


@dataclass
class Runner:
    """Orchestrates running multiple checkers."""

    checkers: dict[str, Checker] = field(default_factory=dict)

    def register(self, checker: Checker) -> None:
        """Register a checker."""
        self.checkers[checker.name] = checker

    def list_checkers(self) -> list[tuple[str, str]]:
        """Return list of (name, description) for all registered checkers."""
        return [(c.name, c.description) for c in self.checkers.values()]

    def run(self, names: list[str] | None = None) -> Report:
        """Run specified checkers (or all if none specified).

        Args:
            names: List of checker names to run. If None, runs all checkers.

        Returns:
            Report containing results from all executed checkers.
        """
        start = time.monotonic()

        if names:
            # Validate requested names
            unknown = set(names) - set(self.checkers.keys())
            if unknown:
                available = sorted(self.checkers.keys())
                msg = (
                    f"Unknown checker(s): {', '.join(sorted(unknown))}\n"
                    f"Available checkers: {', '.join(available)}\n"
                    f"Run all: make check\n"
                    f"Run specific: uv run python check.py <checker-name>"
                )
                raise ValueError(msg)
            to_run = [self.checkers[n] for n in names]
        else:
            to_run = list(self.checkers.values())

        results = []
        for checker in to_run:
            result = checker.run()
            results.append(result)

        total_ms = int((time.monotonic() - start) * 1000)
        return Report(results=tuple(results), total_duration_ms=total_ms)

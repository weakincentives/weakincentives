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

"""Tests for checker runner."""

from __future__ import annotations

import asyncio
from pathlib import Path

from core_types import (
    CheckContext,
    CheckResult,
    Finding,
    RunConfig,
    Severity,
)
from runner import run_checkers, run_checkers_async


class FakeChecker:
    """A fake checker for testing."""

    def __init__(
        self,
        name: str,
        category: str = "test",
        *,
        passed: bool = True,
        duration_ms: int = 10,
    ) -> None:
        self._name = name
        self._category = category
        self._passed = passed
        self._duration_ms = duration_ms

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> str:
        return self._category

    @property
    def description(self) -> str:
        return f"Fake checker: {self._name}"

    def check(self, ctx: CheckContext) -> CheckResult:
        findings = ()
        if not self._passed:
            findings = (
                Finding(
                    checker=f"{self._category}.{self._name}",
                    severity=Severity.ERROR,
                    message="Fake error",
                ),
            )
        return CheckResult(
            checker=f"{self._category}.{self._name}",
            findings=findings,
            duration_ms=self._duration_ms,
        )


class TestRunCheckers:
    """Tests for run_checkers function."""

    def test_run_empty_list(self, tmp_path: Path) -> None:
        """Running no checkers returns empty results."""
        ctx = CheckContext.from_project_root(tmp_path)
        results = run_checkers([], ctx)
        assert results == ()

    def test_run_single_passing_checker(self, tmp_path: Path) -> None:
        """Run a single passing checker."""
        ctx = CheckContext.from_project_root(tmp_path)
        checker = FakeChecker("test1", passed=True)

        results = run_checkers([checker], ctx)

        assert len(results) == 1
        assert results[0].passed is True
        assert results[0].checker == "test.test1"

    def test_run_single_failing_checker(self, tmp_path: Path) -> None:
        """Run a single failing checker."""
        ctx = CheckContext.from_project_root(tmp_path)
        checker = FakeChecker("test1", passed=False)

        results = run_checkers([checker], ctx)

        assert len(results) == 1
        assert results[0].passed is False

    def test_run_multiple_checkers(self, tmp_path: Path) -> None:
        """Run multiple checkers."""
        ctx = CheckContext.from_project_root(tmp_path)
        checkers = [
            FakeChecker("test1", passed=True),
            FakeChecker("test2", passed=True),
            FakeChecker("test3", passed=False),
        ]

        results = run_checkers(checkers, ctx)

        assert len(results) == 3
        assert sum(1 for r in results if r.passed) == 2

    def test_run_with_max_failures(self, tmp_path: Path) -> None:
        """Stop after max failures."""
        ctx = CheckContext.from_project_root(tmp_path)
        checkers = [
            FakeChecker("test1", passed=False),
            FakeChecker("test2", passed=False),
            FakeChecker("test3", passed=True),
        ]
        config = RunConfig(max_failures=1)

        results = run_checkers(checkers, ctx, config=config)

        assert len(results) == 1

    def test_run_with_category_filter(self, tmp_path: Path) -> None:
        """Filter by category."""
        ctx = CheckContext.from_project_root(tmp_path)
        checkers = [
            FakeChecker("test1", category="arch"),
            FakeChecker("test2", category="docs"),
            FakeChecker("test3", category="arch"),
        ]
        config = RunConfig(categories=frozenset({"arch"}))

        results = run_checkers(checkers, ctx, config=config)

        assert len(results) == 2
        assert all("arch" in r.checker for r in results)

    def test_run_with_checker_filter(self, tmp_path: Path) -> None:
        """Filter by checker name."""
        ctx = CheckContext.from_project_root(tmp_path)
        checkers = [
            FakeChecker("checker1"),
            FakeChecker("checker2"),
            FakeChecker("checker3"),
        ]
        config = RunConfig(checkers=frozenset({"checker2"}))

        results = run_checkers(checkers, ctx, config=config)

        assert len(results) == 1
        assert "checker2" in results[0].checker


class TestRunCheckersAsync:
    """Tests for run_checkers_async function."""

    def test_run_async_empty_list(self, tmp_path: Path) -> None:
        """Running no checkers returns empty results."""
        ctx = CheckContext.from_project_root(tmp_path)

        results = asyncio.run(run_checkers_async([], ctx))

        assert results == ()

    def test_run_async_single_checker(self, tmp_path: Path) -> None:
        """Run a single checker asynchronously."""
        ctx = CheckContext.from_project_root(tmp_path)
        checker = FakeChecker("test1", passed=True)

        results = asyncio.run(run_checkers_async([checker], ctx))

        assert len(results) == 1
        assert results[0].passed is True

    def test_run_async_multiple_checkers(self, tmp_path: Path) -> None:
        """Run multiple checkers asynchronously."""
        ctx = CheckContext.from_project_root(tmp_path)
        checkers = [FakeChecker(f"test{i}", passed=True) for i in range(5)]

        results = asyncio.run(run_checkers_async(checkers, ctx))

        assert len(results) == 5
        assert all(r.passed for r in results)

    def test_run_async_with_max_parallel(self, tmp_path: Path) -> None:
        """Respect max_parallel setting."""
        ctx = CheckContext.from_project_root(tmp_path)
        checkers = [FakeChecker(f"test{i}") for i in range(5)]
        config = RunConfig(max_parallel=2)

        results = asyncio.run(run_checkers_async(checkers, ctx, config=config))

        assert len(results) == 5

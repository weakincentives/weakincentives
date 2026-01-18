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

"""Tests for toolchain runner module."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from toolchain.result import CheckResult
from toolchain.runner import Runner


@dataclass
class MockChecker:
    """A simple mock checker for testing."""

    _name: str
    _description: str
    _status: str = "passed"

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def run(self) -> CheckResult:
        return CheckResult(
            name=self._name,
            status=self._status,  # type: ignore[arg-type]
            duration_ms=10,
        )


class TestRunner:
    """Tests for Runner."""

    def test_register_and_list_checkers(self) -> None:
        runner = Runner()
        runner.register(MockChecker(_name="lint", _description="Check lint"))
        runner.register(MockChecker(_name="test", _description="Run tests"))

        checkers = runner.list_checkers()
        assert len(checkers) == 2
        assert ("lint", "Check lint") in checkers
        assert ("test", "Run tests") in checkers

    def test_run_all_checkers(self) -> None:
        runner = Runner()
        runner.register(MockChecker(_name="lint", _description="Lint"))
        runner.register(MockChecker(_name="test", _description="Test"))

        report = runner.run()
        assert len(report.results) == 2
        assert report.total_duration_ms >= 0

    def test_run_specific_checkers(self) -> None:
        runner = Runner()
        runner.register(MockChecker(_name="lint", _description="Lint"))
        runner.register(MockChecker(_name="test", _description="Test"))
        runner.register(MockChecker(_name="typecheck", _description="Types"))

        report = runner.run(["lint", "typecheck"])
        assert len(report.results) == 2
        names = [r.name for r in report.results]
        assert "lint" in names
        assert "typecheck" in names
        assert "test" not in names

    def test_run_unknown_checker_raises(self) -> None:
        runner = Runner()
        runner.register(MockChecker(_name="lint", _description="Lint"))

        with pytest.raises(ValueError, match="Unknown checker.*nonexistent"):
            runner.run(["lint", "nonexistent"])

    def test_run_preserves_order(self) -> None:
        runner = Runner()
        runner.register(MockChecker(_name="a", _description="A"))
        runner.register(MockChecker(_name="b", _description="B"))
        runner.register(MockChecker(_name="c", _description="C"))

        report = runner.run(["c", "a", "b"])
        names = [r.name for r in report.results]
        assert names == ["c", "a", "b"]

    def test_run_empty_returns_empty_report(self) -> None:
        runner = Runner()
        report = runner.run()
        assert len(report.results) == 0
        assert report.passed is True

    def test_report_tracks_failures(self) -> None:
        runner = Runner()
        runner.register(MockChecker(_name="pass1", _description="Pass", _status="passed"))
        runner.register(MockChecker(_name="fail1", _description="Fail", _status="failed"))
        runner.register(MockChecker(_name="pass2", _description="Pass", _status="passed"))

        report = runner.run()
        assert report.passed is False
        assert report.passed_count == 2
        assert report.failed_count == 1

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

"""Unit tests for task completion module: FileOutputChecker, CompositeChecker, and types."""

from __future__ import annotations

import pytest

from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
from weakincentives.prompt import (
    CompositeChecker,
    FileOutputChecker,
    Prompt,
    PromptTemplate,
    TaskCompletionChecker,
    TaskCompletionContext,
    TaskCompletionResult,
)
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session


@pytest.fixture
def session() -> Session:
    return Session(dispatcher=InProcessDispatcher())


@pytest.fixture
def fs() -> InMemoryFilesystem:
    return InMemoryFilesystem()


# ---------------------------------------------------------------------------
# TaskCompletionResult
# ---------------------------------------------------------------------------
class TestTaskCompletionResult:
    """Tests for the TaskCompletionResult frozen dataclass."""

    def test_ok_no_feedback(self) -> None:
        result = TaskCompletionResult.ok()
        assert result.complete is True
        assert result.feedback is None

    def test_ok_with_feedback(self) -> None:
        result = TaskCompletionResult.ok("All done.")
        assert result.complete is True
        assert result.feedback == "All done."

    def test_incomplete(self) -> None:
        result = TaskCompletionResult.incomplete("Missing file X.")
        assert result.complete is False
        assert result.feedback == "Missing file X."


# ---------------------------------------------------------------------------
# FileOutputChecker
# ---------------------------------------------------------------------------
class TestFileOutputChecker:
    """Tests for the FileOutputChecker built-in implementation."""

    def test_no_filesystem_returns_incomplete(self, session: Session) -> None:
        """Fail-closed: when no filesystem is available and files required, checker returns incomplete."""
        checker = FileOutputChecker(files=("report.md",))
        context = TaskCompletionContext(session=session, filesystem=None)

        result = checker.check(context)

        assert result.complete is False
        assert "No filesystem" in result.feedback
        assert "1 required" in result.feedback

    def test_no_filesystem_empty_files_returns_ok(self, session: Session) -> None:
        """No filesystem but no files required => ok (vacuously true)."""
        checker = FileOutputChecker(files=())
        context = TaskCompletionContext(session=session, filesystem=None)

        result = checker.check(context)

        assert result.complete is True
        assert "No files required" in result.feedback

    def test_all_files_exist(self, session: Session, fs: InMemoryFilesystem) -> None:
        """All required files present => ok."""
        fs.write("a.txt", "data")
        fs.write("b.txt", "data")

        checker = FileOutputChecker(files=("a.txt", "b.txt"))
        context = TaskCompletionContext(session=session, filesystem=fs)

        result = checker.check(context)

        assert result.complete is True
        assert "2" in result.feedback

    def test_some_files_missing(self, session: Session, fs: InMemoryFilesystem) -> None:
        """Some files missing => incomplete with listing."""
        fs.write("a.txt", "data")

        checker = FileOutputChecker(files=("a.txt", "b.txt"))
        context = TaskCompletionContext(session=session, filesystem=fs)

        result = checker.check(context)

        assert result.complete is False
        assert "b.txt" in result.feedback
        assert "1 required" in result.feedback

    def test_all_files_missing(self, session: Session, fs: InMemoryFilesystem) -> None:
        """All files missing => incomplete."""
        checker = FileOutputChecker(files=("x.txt", "y.txt"))
        context = TaskCompletionContext(session=session, filesystem=fs)

        result = checker.check(context)

        assert result.complete is False
        assert "x.txt" in result.feedback
        assert "y.txt" in result.feedback

    def test_empty_files_list(self, session: Session, fs: InMemoryFilesystem) -> None:
        """Empty file list => ok (vacuously true)."""
        checker = FileOutputChecker(files=())
        context = TaskCompletionContext(session=session, filesystem=fs)

        result = checker.check(context)

        assert result.complete is True

    def test_truncates_at_three_files(
        self, session: Session, fs: InMemoryFilesystem
    ) -> None:
        """Missing file list truncated to 3 with ellipsis."""
        files = tuple(f"f{i}.txt" for i in range(6))
        checker = FileOutputChecker(files=files)
        context = TaskCompletionContext(session=session, filesystem=fs)

        result = checker.check(context)

        assert result.complete is False
        assert "f0.txt" in result.feedback
        assert "f1.txt" in result.feedback
        assert "f2.txt" in result.feedback
        assert "..." in result.feedback
        assert "f5.txt" not in result.feedback

    def test_is_runtime_checkable(self) -> None:
        """FileOutputChecker satisfies the TaskCompletionChecker protocol."""
        checker = FileOutputChecker(files=("output.txt",))
        assert isinstance(checker, TaskCompletionChecker)


# ---------------------------------------------------------------------------
# CompositeChecker
# ---------------------------------------------------------------------------
class TestCompositeChecker:
    """Tests for the CompositeChecker composition logic."""

    def test_empty_checkers(self, session: Session) -> None:
        """Empty composite => ok."""
        checker = CompositeChecker(checkers=())
        context = TaskCompletionContext(session=session)

        result = checker.check(context)

        assert result.complete is True
        assert "No checkers configured" in result.feedback

    def test_all_must_pass_all_pass(
        self, session: Session, fs: InMemoryFilesystem
    ) -> None:
        """AND mode: all pass => ok."""
        fs.write("a.txt", "data")
        fs.write("b.txt", "data")

        checker = CompositeChecker(
            checkers=(
                FileOutputChecker(files=("a.txt",)),
                FileOutputChecker(files=("b.txt",)),
            ),
            all_must_pass=True,
        )
        context = TaskCompletionContext(session=session, filesystem=fs)

        result = checker.check(context)

        assert result.complete is True

    def test_all_must_pass_one_fails(
        self, session: Session, fs: InMemoryFilesystem
    ) -> None:
        """AND mode: first failure short-circuits."""
        fs.write("a.txt", "data")
        # b.txt missing

        checker = CompositeChecker(
            checkers=(
                FileOutputChecker(files=("b.txt",)),  # Fails first
                FileOutputChecker(files=("a.txt",)),  # Not evaluated
            ),
            all_must_pass=True,
        )
        context = TaskCompletionContext(session=session, filesystem=fs)

        result = checker.check(context)

        assert result.complete is False
        assert "b.txt" in result.feedback

    def test_any_pass_first_passes(
        self, session: Session, fs: InMemoryFilesystem
    ) -> None:
        """OR mode: first success short-circuits."""
        fs.write("a.txt", "data")

        checker = CompositeChecker(
            checkers=(
                FileOutputChecker(files=("a.txt",)),  # Passes
                FileOutputChecker(files=("missing.txt",)),  # Not evaluated
            ),
            all_must_pass=False,
        )
        context = TaskCompletionContext(session=session, filesystem=fs)

        result = checker.check(context)

        assert result.complete is True

    def test_any_pass_none_pass(self, session: Session, fs: InMemoryFilesystem) -> None:
        """OR mode: all fail => incomplete."""
        checker = CompositeChecker(
            checkers=(
                FileOutputChecker(files=("x.txt",)),
                FileOutputChecker(files=("y.txt",)),
            ),
            all_must_pass=False,
        )
        context = TaskCompletionContext(session=session, filesystem=fs)

        result = checker.check(context)

        assert result.complete is False

    def test_is_runtime_checkable(self) -> None:
        """CompositeChecker satisfies the protocol."""
        checker = CompositeChecker(checkers=())
        assert isinstance(checker, TaskCompletionChecker)


# ---------------------------------------------------------------------------
# PromptTemplate integration
# ---------------------------------------------------------------------------
class TestPromptTemplateIntegration:
    """Tests for task_completion_checker on PromptTemplate."""

    def test_default_is_none(self) -> None:
        """PromptTemplate defaults to no task completion checker."""
        template: PromptTemplate[object] = PromptTemplate(ns="test", key="test")
        assert template.task_completion_checker is None

    def test_set_checker_on_template(self) -> None:
        """PromptTemplate accepts a task completion checker."""
        checker = FileOutputChecker(files=("output.txt",))
        template: PromptTemplate[object] = PromptTemplate(
            ns="test", key="test", task_completion_checker=checker
        )
        assert template.task_completion_checker is checker

    def test_prompt_forwards_checker(self) -> None:
        """Prompt.task_completion_checker forwards from template."""
        checker = FileOutputChecker(files=("output.txt",))
        template: PromptTemplate[object] = PromptTemplate(
            ns="test", key="test", task_completion_checker=checker
        )
        prompt: Prompt[object] = Prompt(template)
        assert prompt.task_completion_checker is checker

    def test_prompt_forwards_none(self) -> None:
        """Prompt.task_completion_checker returns None when not configured."""
        template: PromptTemplate[object] = PromptTemplate(ns="test", key="test")
        prompt: Prompt[object] = Prompt(template)
        assert prompt.task_completion_checker is None

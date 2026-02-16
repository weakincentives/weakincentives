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

"""Tests for verify_task_completion and check_task_completion functions."""

from __future__ import annotations

import logging
import warnings
from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest

from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
)
from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
from weakincentives.filesystem import Filesystem
from weakincentives.prompt import Prompt, PromptTemplate
from weakincentives.prompt.task_completion import (
    FileOutputChecker,
    TaskCompletionChecker,
    TaskCompletionContext,
    TaskCompletionResult,
)
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session

from ._hook_helpers import _make_prompt_with_fs


class TestVerifyTaskCompletion:
    """Tests for verify_task_completion function."""

    @pytest.fixture
    def session(self) -> Session:
        return Session(dispatcher=InProcessDispatcher())

    @pytest.fixture
    def adapter(self) -> ClaudeAgentSDKAdapter:
        return ClaudeAgentSDKAdapter()

    @staticmethod
    def _call_verify(adapter: ClaudeAgentSDKAdapter, **kwargs: Any) -> None:
        from weakincentives.adapters.claude_agent_sdk._result_extraction import (
            verify_task_completion,
        )

        verify_task_completion(
            **kwargs, client_config=adapter._client_config, adapter=adapter
        )

    def test_no_checker_configured_does_nothing(
        self, adapter: ClaudeAgentSDKAdapter, session: Session
    ) -> None:
        """When no checker is configured, verification passes."""
        self._call_verify(
            adapter,
            output={"key": "value"},
            session=session,
            stop_reason="structured_output",
            prompt_name="test",
        )

    def test_no_output_does_nothing(self, session: Session) -> None:
        """When output is None, verification passes."""
        checker = FileOutputChecker(files=("output.txt",))
        prompt: Prompt[object] = Prompt(
            PromptTemplate(ns="test", key="test", task_completion_checker=checker)
        )
        prompt.resources.__enter__()
        adapter = ClaudeAgentSDKAdapter()
        self._call_verify(
            adapter,
            output=None,
            session=session,
            stop_reason="structured_output",
            prompt_name="test",
            prompt=prompt,
        )

    def test_logs_warning_when_files_missing(
        self, session: Session, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When required files are missing, logs warning but doesn't raise error."""
        fs = InMemoryFilesystem()
        # Don't create any files

        checker = FileOutputChecker(files=("output.txt",))
        adapter = ClaudeAgentSDKAdapter()

        caplog.set_level(logging.WARNING)

        prompt = _make_prompt_with_fs(fs, task_completion_checker=checker)
        self._call_verify(
            adapter,
            output={"summary": "done"},
            session=session,
            stop_reason="structured_output",
            prompt_name="test_prompt",
            prompt=prompt,
        )

        assert any("incomplete_tasks" in record.message for record in caplog.records), (
            "Should log warning about incomplete tasks"
        )

    def test_passes_when_files_exist(self, session: Session) -> None:
        """When all required files exist, verification passes."""
        fs = InMemoryFilesystem()
        fs.write("output.txt", "done")
        fs.write("summary.md", "# Summary")

        checker = FileOutputChecker(files=("output.txt", "summary.md"))
        adapter = ClaudeAgentSDKAdapter()

        prompt = _make_prompt_with_fs(fs, task_completion_checker=checker)
        self._call_verify(
            adapter,
            output={"summary": "done"},
            session=session,
            stop_reason="structured_output",
            prompt_name="test_prompt",
            prompt=prompt,
        )

    def test_skips_when_deadline_exceeded(self, session: Session) -> None:
        """When deadline is exceeded, verification is skipped."""
        checker = FileOutputChecker(files=("output.txt",))
        prompt: Prompt[object] = Prompt(
            PromptTemplate(ns="test", key="test", task_completion_checker=checker)
        )
        prompt.resources.__enter__()
        adapter = ClaudeAgentSDKAdapter()

        exceeded_deadline = MagicMock()
        exceeded_deadline.remaining.return_value = timedelta(seconds=-1)

        self._call_verify(
            adapter,
            output={"summary": "partial"},
            session=session,
            stop_reason="structured_output",
            prompt_name="test_prompt",
            deadline=exceeded_deadline,
            prompt=prompt,
        )

    def test_skips_when_budget_exhausted(self, session: Session) -> None:
        """When budget is exhausted, verification is skipped."""
        from weakincentives.budget import Budget, BudgetTracker
        from weakincentives.runtime.events.types import TokenUsage

        checker = FileOutputChecker(files=("output.txt",))
        prompt: Prompt[object] = Prompt(
            PromptTemplate(ns="test", key="test", task_completion_checker=checker)
        )
        prompt.resources.__enter__()
        adapter = ClaudeAgentSDKAdapter()

        budget = Budget(max_total_tokens=100)
        tracker = BudgetTracker(budget)
        tracker.record_cumulative("test", TokenUsage(input_tokens=50, output_tokens=50))

        self._call_verify(
            adapter,
            output={"summary": "partial"},
            session=session,
            stop_reason="structured_output",
            prompt_name="test_prompt",
            budget_tracker=tracker,
            prompt=prompt,
        )

    def test_passes_filesystem_and_adapter_to_context(self, session: Session) -> None:
        """Filesystem and adapter are passed to TaskCompletionContext."""
        captured_context: list[TaskCompletionContext] = []

        class CapturingChecker(TaskCompletionChecker):
            def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
                captured_context.append(context)
                return TaskCompletionResult.ok()

        capturing_checker = CapturingChecker()
        adapter = ClaudeAgentSDKAdapter()

        mock_filesystem = MagicMock(spec=Filesystem)
        mock_resources = MagicMock()
        mock_resources.get.return_value = mock_filesystem
        mock_prompt = MagicMock()
        mock_prompt.resources = mock_resources
        mock_prompt.task_completion_checker = capturing_checker

        self._call_verify(
            adapter,
            output={"summary": "done"},
            session=session,
            stop_reason="structured_output",
            prompt_name="test_prompt",
            prompt=mock_prompt,
        )

        assert len(captured_context) == 1
        ctx = captured_context[0]
        assert ctx.filesystem is mock_filesystem
        assert ctx.adapter is adapter

    def test_handles_filesystem_lookup_failure(self, session: Session) -> None:
        """When filesystem lookup fails, context still gets adapter but no filesystem."""
        from weakincentives.resources.errors import UnboundResourceError

        captured_context: list[TaskCompletionContext] = []

        class CapturingChecker(TaskCompletionChecker):
            def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
                captured_context.append(context)
                return TaskCompletionResult.ok()

        capturing_checker = CapturingChecker()
        adapter = ClaudeAgentSDKAdapter()

        mock_resources = MagicMock()
        mock_resources.get.side_effect = UnboundResourceError(object)
        mock_prompt = MagicMock()
        mock_prompt.resources = mock_resources
        mock_prompt.task_completion_checker = capturing_checker

        self._call_verify(
            adapter,
            output={"summary": "done"},
            session=session,
            stop_reason="structured_output",
            prompt_name="test_prompt",
            prompt=mock_prompt,
        )

        assert len(captured_context) == 1
        ctx = captured_context[0]
        assert ctx.filesystem is None
        assert ctx.adapter is adapter

    def test_logs_warning_when_budget_not_exhausted(
        self, session: Session, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When budget_tracker is provided but not exhausted, logs warning."""
        from weakincentives.budget import Budget, BudgetTracker
        from weakincentives.runtime.events.types import TokenUsage

        fs = InMemoryFilesystem()
        # Don't create the required file

        checker = FileOutputChecker(files=("output.txt",))
        adapter = ClaudeAgentSDKAdapter()

        budget = Budget(max_total_tokens=1000)
        tracker = BudgetTracker(budget)
        tracker.record_cumulative("test", TokenUsage(input_tokens=50, output_tokens=50))

        caplog.set_level(logging.WARNING)

        prompt = _make_prompt_with_fs(fs, task_completion_checker=checker)
        self._call_verify(
            adapter,
            output={"summary": "partial"},
            session=session,
            stop_reason="structured_output",
            prompt_name="test_prompt",
            budget_tracker=tracker,
            prompt=prompt,
        )

        assert any("incomplete_tasks" in record.message for record in caplog.records), (
            "Should log warning about incomplete tasks when budget remains"
        )

    def test_config_checker_fallback_with_deprecation(self, session: Session) -> None:
        """Checker on config without prompt checker produces deprecation warning."""
        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(
                task_completion_checker=FileOutputChecker(files=("output.txt",)),
            ),
        )

        prompt: Prompt[object] = Prompt(PromptTemplate(ns="test", key="test"))
        prompt.resources.__enter__()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._call_verify(
                adapter,
                output={"summary": "done"},
                session=session,
                stop_reason="structured_output",
                prompt_name="test",
                prompt=prompt,
            )

        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_no_prompt_with_config_checker(self, session: Session) -> None:
        """Checker resolves from config when prompt is None (no filesystem)."""
        captured_context: list[TaskCompletionContext] = []

        class CapturingChecker(TaskCompletionChecker):
            def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
                captured_context.append(context)
                return TaskCompletionResult.ok()

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(
                task_completion_checker=CapturingChecker(),
            ),
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            self._call_verify(
                adapter,
                output={"summary": "done"},
                session=session,
                stop_reason="structured_output",
                prompt_name="test",
                prompt=None,
            )

        assert len(captured_context) == 1
        ctx = captured_context[0]
        assert ctx.filesystem is None
        assert ctx.adapter is adapter


class TestCheckTaskCompletion:
    """Tests for check_task_completion function."""

    def test_returns_false_none_for_empty_messages(self, session: Session) -> None:
        """check_task_completion returns (False, None) for empty message list."""
        from weakincentives.adapters.claude_agent_sdk._hooks import (
            HookConstraints,
            HookContext,
        )
        from weakincentives.adapters.claude_agent_sdk._sdk_execution import (
            check_task_completion,
        )

        checker = MagicMock()
        template: PromptTemplate[None] = PromptTemplate(
            ns="test", key="test", name="test"
        )
        prompt: Prompt[None] = Prompt(template)
        constraints = HookConstraints()
        hook_context = HookContext(
            prompt=prompt,
            session=session,
            adapter_name="test",
            prompt_name="test",
            constraints=constraints,
        )

        result = check_task_completion(checker, [], hook_context)

        assert result == (False, None)
        checker.check.assert_not_called()

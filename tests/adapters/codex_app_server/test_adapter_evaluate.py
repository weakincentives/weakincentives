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

"""Tests for adapter evaluate: end-to-end, tools, and CWD resolution."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from weakincentives.adapters.codex_app_server.adapter import CodexAppServerAdapter
from weakincentives.adapters.codex_app_server.client import CodexClientError
from weakincentives.adapters.codex_app_server.config import (
    CodexAppServerClientConfig,
)
from weakincentives.adapters.core import PromptEvaluationError
from weakincentives.budget import Budget
from weakincentives.clock import FakeClock
from weakincentives.deadlines import Deadline
from weakincentives.filesystem import Filesystem
from weakincentives.runtime.events import PromptExecuted, PromptRendered

from .conftest import (
    make_mock_client,
    make_prompt_with_tool,
    make_session,
    make_simple_prompt,
    messages_iterator,
)


class TestEvaluateExpiredDeadline:
    def test_raises_on_expired_deadline(self) -> None:
        adapter = CodexAppServerAdapter()
        session, _ = make_session()
        prompt = make_simple_prompt()
        clock = FakeClock()
        anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        clock.set_wall(anchor)
        deadline = Deadline(expires_at=anchor + timedelta(seconds=5), clock=clock)
        clock.advance(10)

        with pytest.raises(PromptEvaluationError, match="Deadline expired"):
            adapter.evaluate(prompt, session=session, deadline=deadline)


class TestEvaluateEndToEnd:
    """End-to-end tests mocking the CodexAppServerClient."""

    def test_simple_evaluation(self) -> None:
        """Test basic prompt evaluation with mocked client."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, dispatcher = make_session()

        # Track events
        rendered_events: list[PromptRendered] = []
        executed_events: list[PromptExecuted] = []
        dispatcher.subscribe(PromptRendered, lambda e: rendered_events.append(e))
        dispatcher.subscribe(PromptExecuted, lambda e: executed_events.append(e))

        prompt = make_simple_prompt()

        messages = [
            {"method": "item/agentMessage/delta", "params": {"delta": "Hello "}},
            {"method": "item/agentMessage/delta", "params": {"delta": "world"}},
            # Unknown notification — should be silently ignored
            {"method": "unknown/notification", "params": {}},
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "Hello world"}},
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = make_mock_client()
            # Initialize returns capabilities
            mock_client.send_request.side_effect = [
                {"capabilities": {}},  # initialize
                {"thread": {"id": "t-1"}},  # thread/start
                {"turn": {"id": 1}},  # turn/start
            ]
            mock_client.read_messages.return_value = messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session)

        assert result.text == "Hello world"
        assert len(rendered_events) == 1
        assert len(executed_events) == 1
        assert executed_events[0].adapter == "codex_app_server"

    def test_evaluation_with_token_usage(self) -> None:
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = make_session()
        prompt = make_simple_prompt()

        messages = [
            {
                "method": "thread/tokenUsage/updated",
                "params": {
                    "tokenUsage": {
                        "last": {
                            "inputTokens": 100,
                            "outputTokens": 50,
                        }
                    }
                },
            },
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "answer"}},
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session)

        assert result.text == "answer"

    def test_client_error_wrapped(self) -> None:
        """CodexClientError is wrapped in PromptEvaluationError."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = make_session()
        prompt = make_simple_prompt()

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = make_mock_client()
            mock_client.send_request.side_effect = CodexClientError("conn failed")
            MockClient.return_value = mock_client

            with pytest.raises(PromptEvaluationError, match="conn failed"):
                adapter.evaluate(prompt, session=session)

    def test_generic_error_wrapped(self) -> None:
        """Generic exceptions are wrapped in PromptEvaluationError."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = make_session()
        prompt = make_simple_prompt()

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = make_mock_client()
            mock_client.start.side_effect = RuntimeError("unexpected")
            MockClient.return_value = mock_client

            with pytest.raises(PromptEvaluationError, match="unexpected"):
                adapter.evaluate(prompt, session=session)

    def test_prompt_eval_error_passthrough(self) -> None:
        """PromptEvaluationError passes through unwrapped."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = make_session()
        prompt = make_simple_prompt()

        messages = [
            {
                "method": "turn/completed",
                "params": {
                    "turn": {
                        "status": "failed",
                        "codexErrorInfo": "unauthorized",
                        "additionalDetails": "bad key",
                    }
                },
            },
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = messages_iterator(messages)
            MockClient.return_value = mock_client

            with pytest.raises(PromptEvaluationError) as exc_info:
                adapter.evaluate(prompt, session=session)
            assert exc_info.value.phase == "request"

    def test_stream_eof_before_turn_completed(self) -> None:
        """Stream ends with messages but no turn/completed -> raises."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = make_session()
        prompt = make_simple_prompt()

        messages = [
            {"method": "item/agentMessage/delta", "params": {"delta": "partial"}},
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "partial"}},
            },
            # No turn/completed — stream ends
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = messages_iterator(messages)
            MockClient.return_value = mock_client

            with pytest.raises(PromptEvaluationError, match="stream ended before"):
                adapter.evaluate(prompt, session=session)

    def test_stream_eof_empty_stream(self) -> None:
        """Zero messages in stream -> raises."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = make_session()
        prompt = make_simple_prompt()

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = messages_iterator([])
            MockClient.return_value = mock_client

            with pytest.raises(PromptEvaluationError, match="stream ended before"):
                adapter.evaluate(prompt, session=session)

    def test_budget_creates_tracker(self) -> None:
        """Budget without tracker creates one automatically."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = make_session()
        prompt = make_simple_prompt()
        budget = Budget(
            max_input_tokens=1000,
            max_output_tokens=500,
        )

        messages = [
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "ok"}},
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session, budget=budget)
            assert result.text == "ok"

    def test_rendered_tools_dispatch_failure_logs_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that RenderedTools dispatch failures are logged."""
        import logging

        from weakincentives.runtime.session.rendered_tools import RenderedTools

        def failing_handler(event: RenderedTools) -> None:
            raise RuntimeError("Subscriber error")

        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, dispatcher = make_session()
        dispatcher.subscribe(RenderedTools, failing_handler)

        prompt = make_simple_prompt()

        messages = [
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "Done"}},
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        caplog.set_level(logging.ERROR)

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session)

        assert result.text == "Done"
        assert any(
            "rendered_tools_dispatch_failed" in record.message
            for record in caplog.records
        )


class TestEvaluateWithTools:
    """Test that prompt tools are extracted into RenderedTools schemas."""

    def test_tool_schemas_dispatched(self) -> None:
        from weakincentives.runtime.session.rendered_tools import RenderedTools

        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, dispatcher = make_session()
        prompt = make_prompt_with_tool()

        dispatched: list[RenderedTools] = []
        dispatcher.subscribe(RenderedTools, lambda e: dispatched.append(e))

        messages = [
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "ok"}},
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session)

        assert result.text == "ok"
        assert len(dispatched) == 1
        assert len(dispatched[0].tools) == 1
        assert dispatched[0].tools[0].name == "add"


class TestResolveCwd:
    def test_no_filesystem_no_cwd_creates_temp(self) -> None:
        import shutil as _shutil

        adapter = CodexAppServerAdapter()
        prompt = make_simple_prompt()

        cwd, temp_dir, _new_prompt = adapter._resolve_cwd(prompt)
        try:
            assert cwd is not None
            assert temp_dir is not None
            assert cwd == temp_dir
        finally:
            if temp_dir:
                _shutil.rmtree(temp_dir, ignore_errors=True)

    def test_no_filesystem_with_cwd_uses_configured(self) -> None:
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/configured")
        )
        prompt = make_simple_prompt()

        cwd, temp_dir, _ = adapter._resolve_cwd(prompt)
        assert cwd == "/tmp/configured"
        assert temp_dir is None

    def test_workspace_section_extracts_root(self) -> None:
        """When prompt has a workspace section with HostFilesystem and no cwd."""
        from weakincentives.prompt import Prompt, PromptTemplate, WorkspaceSection

        adapter = CodexAppServerAdapter()
        session, _ = make_session()

        workspace = WorkspaceSection(session=session)
        workspace_root = str(workspace.temp_dir)
        try:
            template: PromptTemplate[object] = PromptTemplate(
                ns="test",
                key="with-ws",
                sections=(workspace,),
                name="ws-prompt",
            )
            prompt = Prompt(template)

            cwd, temp_dir, _ = adapter._resolve_cwd(prompt)
            assert cwd == workspace_root
            assert temp_dir is None
        finally:
            workspace.cleanup()

    def test_non_host_filesystem_falls_back_to_cwd(self) -> None:
        """When filesystem is not HostFilesystem, falls back to Path.cwd()."""
        adapter = CodexAppServerAdapter()
        prompt = make_simple_prompt()

        # Mock prompt.filesystem() to return a non-HostFilesystem
        mock_fs = MagicMock(spec=Filesystem)
        with patch.object(type(prompt), "filesystem", return_value=mock_fs):
            cwd, temp_dir, _ = adapter._resolve_cwd(prompt)
            assert cwd is not None
            assert temp_dir is None

    def test_workspace_with_configured_cwd(self) -> None:
        """When prompt has workspace section AND cwd is configured, cwd wins."""
        from weakincentives.prompt import Prompt, PromptTemplate, WorkspaceSection

        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/configured")
        )
        session, _ = make_session()

        workspace = WorkspaceSection(session=session)
        try:
            template: PromptTemplate[object] = PromptTemplate(
                ns="test",
                key="with-ws2",
                sections=(workspace,),
                name="ws-prompt2",
            )
            prompt = Prompt(template)

            cwd, temp_dir, _ = adapter._resolve_cwd(prompt)
            # Configured cwd should win over workspace root
            assert cwd == "/tmp/configured"
            assert temp_dir is None
        finally:
            workspace.cleanup()

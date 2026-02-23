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

"""Tests for Codex App Server protocol helpers, deadline utilities, and skills."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weakincentives.adapters.codex_app_server._ephemeral_home import CodexEphemeralHome
from weakincentives.adapters.codex_app_server._protocol import (
    authenticate,
    create_thread,
    deadline_remaining_s,
    handle_tool_call,
    start_turn,
)
from weakincentives.adapters.codex_app_server.adapter import CodexAppServerAdapter
from weakincentives.adapters.codex_app_server.client import (
    CodexAppServerClient,
    CodexClientError,
)
from weakincentives.adapters.codex_app_server.config import (
    ApiKeyAuth,
    CodexAppServerClientConfig,
    CodexAppServerModelConfig,
)
from weakincentives.adapters.core import PromptEvaluationError
from weakincentives.clock import FakeClock
from weakincentives.deadlines import Deadline
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session
from weakincentives.skills import SkillMount

# ---- Helpers ----


def _make_session() -> tuple[Session, InProcessDispatcher]:
    dispatcher = InProcessDispatcher()
    session = Session(dispatcher=dispatcher, tags={"suite": "tests"})
    return session, dispatcher


def _make_simple_prompt(name: str = "test-prompt") -> Any:
    from weakincentives.prompt import Prompt, PromptTemplate

    template: PromptTemplate[object] = PromptTemplate(
        ns="test",
        key="basic",
        sections=(),
        name=name,
    )
    return Prompt(template)


def _make_mock_client() -> AsyncMock:
    """Create a mock CodexAppServerClient."""
    client = AsyncMock(spec=CodexAppServerClient)
    client.stderr_output = ""
    client.start = AsyncMock()
    client.stop = AsyncMock()
    client.send_request = AsyncMock(return_value={})
    client.send_notification = AsyncMock()
    client.send_response = AsyncMock()
    return client


def _messages_iterator(
    messages: list[dict[str, Any]],
) -> Any:
    """Create an async iterator from a list of messages."""

    async def _iter() -> Any:
        for msg in messages:
            yield msg

    return _iter()


# ---------------------------------------------------------------------------
# Helpers for skill frontmatter
# ---------------------------------------------------------------------------

_SKILL_MD = "---\nname: {name}\ndescription: A test skill\n---\n\n# {name}\n"


def _make_dir_skill(base: Path, name: str) -> Path:
    d = base / name
    d.mkdir()
    (d / "SKILL.md").write_text(_SKILL_MD.format(name=name))
    return d


class TestToolCallRunsInThread:
    def test_tool_call_runs_in_thread(self) -> None:
        """Tool dispatch is wrapped in asyncio.to_thread."""

        async def _run() -> None:
            client = _make_mock_client()
            mock_tool = MagicMock()
            mock_tool.return_value = {
                "content": [{"type": "text", "text": "threaded"}],
                "isError": False,
            }
            tool_lookup = {"calc": mock_tool}

            with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
                mock_to_thread.return_value = {
                    "content": [{"type": "text", "text": "threaded"}],
                    "isError": False,
                }
                await handle_tool_call(
                    client, 20, {"tool": "calc", "arguments": {"x": 1}}, tool_lookup
                )
                mock_to_thread.assert_called_once_with(mock_tool, {"x": 1})

        asyncio.run(_run())


class TestDeadlineRemainingS:
    def test_no_deadline_returns_none(self) -> None:
        assert deadline_remaining_s(None, "p") is None

    def test_expired_deadline_raises(self) -> None:
        clock = FakeClock()
        anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        clock.set_wall(anchor)
        deadline = Deadline(expires_at=anchor + timedelta(seconds=5), clock=clock)
        clock.advance(10)

        with pytest.raises(PromptEvaluationError, match="Deadline expired during"):
            deadline_remaining_s(deadline, "test-prompt")

    def test_active_deadline_returns_seconds(self) -> None:
        clock = FakeClock()
        anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        clock.set_wall(anchor)
        deadline = Deadline(expires_at=anchor + timedelta(seconds=30), clock=clock)

        remaining = deadline_remaining_s(deadline, "test-prompt")
        assert remaining is not None
        assert remaining > 0


class TestSetupRPCDeadlineBounding:
    def test_setup_timeout_wraps_client_error(self) -> None:
        """When thread/start times out, PromptEvaluationError has phase='request'."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()
        clock = FakeClock()
        anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        clock.set_wall(anchor)
        deadline = Deadline(expires_at=anchor + timedelta(seconds=30), clock=clock)

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            # initialize succeeds, thread/start raises client timeout
            mock_client.send_request.side_effect = [
                {"capabilities": {}},  # initialize
                CodexClientError("Timeout waiting for response to thread/start"),
            ]
            MockClient.return_value = mock_client

            with pytest.raises(PromptEvaluationError) as exc_info:
                adapter.evaluate(prompt, session=session, deadline=deadline)
            assert exc_info.value.phase == "request"

    def test_setup_passes_timeout_to_send_request(self) -> None:
        """Verify timeout is forwarded to send_request for setup RPCs."""

        async def _run() -> None:
            client = _make_mock_client()
            client.send_request.return_value = {"thread": {"id": "t-1"}}

            await create_thread(
                client,
                "/tmp",
                [],
                client_config=CodexAppServerClientConfig(),
                model_config=CodexAppServerModelConfig(),
                timeout=5.0,
            )
            call_args = client.send_request.call_args
            assert call_args[1].get("timeout") == 5.0 or call_args[0][2] == 5.0

        asyncio.run(_run())

    def test_authenticate_passes_timeout(self) -> None:
        async def _run() -> None:
            client = _make_mock_client()

            await authenticate(client, ApiKeyAuth(api_key="sk-test"), timeout=3.0)
            call_args = client.send_request.call_args
            assert call_args[1].get("timeout") == 3.0

        asyncio.run(_run())

    def test_start_turn_passes_timeout(self) -> None:
        async def _run() -> None:
            client = _make_mock_client()
            client.send_request.return_value = {"turn": {"id": 1}}

            await start_turn(
                client,
                "thread-1",
                "Hello",
                None,
                model_config=CodexAppServerModelConfig(),
                timeout=7.0,
            )
            call_args = client.send_request.call_args
            assert call_args[1].get("timeout") == 7.0

        asyncio.run(_run())


class TestTranscriptBridgeIntegration:
    """Tests for transcript bridge integration in the adapter."""

    def test_evaluate_with_transcript_disabled(self) -> None:
        """When transcript=False, bridge is not created."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test", transcript=False),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

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
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session)

        assert result.text == "ok"

    def test_handle_tool_call_with_bridge(self) -> None:
        """Tool call emits transcript entries via bridge."""

        async def _run() -> None:
            client = _make_mock_client()
            mock_tool = MagicMock()
            mock_tool.return_value = {
                "content": [{"type": "text", "text": "result: 42"}],
                "isError": False,
            }
            tool_lookup = {"calc": mock_tool}

            bridge = MagicMock()
            await handle_tool_call(
                client,
                10,
                {"tool": "calc", "arguments": {"x": 1}},
                tool_lookup,
                bridge=bridge,
            )
            bridge.on_tool_call.assert_called_once_with(
                {"tool": "calc", "arguments": {"x": 1}}
            )
            bridge.on_tool_result.assert_called_once()
            resp = bridge.on_tool_result.call_args[0][1]
            assert resp["success"] is True

        asyncio.run(_run())

    def test_handle_tool_call_unknown_tool_with_bridge(self) -> None:
        """Unknown tool call still emits tool_result via bridge."""

        async def _run() -> None:
            client = _make_mock_client()
            tool_lookup: dict[str, Any] = {}

            bridge = MagicMock()
            await handle_tool_call(
                client,
                10,
                {"tool": "missing", "arguments": {}},
                tool_lookup,
                bridge=bridge,
            )
            bridge.on_tool_call.assert_called_once()
            bridge.on_tool_result.assert_called_once()
            resp = bridge.on_tool_result.call_args[0][1]
            assert resp["success"] is False

        asyncio.run(_run())


class TestApprovalPolicyUntrusted:
    def test_approval_untrusted_declines(self) -> None:
        from weakincentives.adapters.codex_app_server._protocol import (
            handle_server_request,
        )

        async def _run() -> None:
            client = _make_mock_client()
            msg = {
                "id": 6,
                "method": "item/commandExecution/requestApproval",
                "params": {},
            }
            await handle_server_request(client, msg, {}, approval_policy="untrusted")
            resp = client.send_response.call_args[0][1]
            assert resp["decision"] == "decline"

        asyncio.run(_run())

    def test_approval_on_failure_accepts_requested_approval(self) -> None:
        from weakincentives.adapters.codex_app_server._protocol import (
            handle_server_request,
        )

        async def _run() -> None:
            client = _make_mock_client()
            msg = {
                "id": 7,
                "method": "item/commandExecution/requestApproval",
                "params": {},
            }
            await handle_server_request(client, msg, {}, approval_policy="on-failure")
            resp = client.send_response.call_args[0][1]
            assert resp["decision"] == "accept"

        asyncio.run(_run())


class TestEvaluateWithSkills:
    """Test that skills trigger ephemeral home creation and env merging."""

    def _success_messages(self) -> list[dict[str, Any]]:
        return [
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "ok"}},
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

    def test_skills_create_ephemeral_home(self, tmp_path: Path) -> None:
        """When rendered prompt has skills, client env includes HOME + CODEX_HOME."""
        skill_dir = _make_dir_skill(tmp_path, "my-skill")
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        original_render = prompt.render

        def patched_render(**kwargs: Any) -> Any:
            from weakincentives.prompt.rendering import RenderedPrompt

            rendered = original_render(**kwargs)
            return RenderedPrompt(
                text=rendered.text,
                _tools=rendered.tools,
                _skills=(SkillMount(source=skill_dir),),
            )

        with (
            patch(
                "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
            ) as MockClient,
            patch.object(prompt, "render", side_effect=patched_render),
        ):
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(
                self._success_messages()
            )
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session)

        assert result.text == "ok"
        # Verify client was constructed with env containing HOME
        constructor_kwargs = MockClient.call_args[1]
        env = constructor_kwargs.get("env") or {}
        assert "HOME" in env

    def test_no_skills_no_ephemeral_home(self) -> None:
        """When rendered prompt has no skills, env is unchanged."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(
                self._success_messages()
            )
            MockClient.return_value = mock_client

            adapter.evaluate(prompt, session=session)

        # No skills -> env should be None (from empty dict)
        constructor_kwargs = MockClient.call_args[1]
        assert constructor_kwargs.get("env") is None

    def test_ephemeral_home_cleaned_up_on_error(self, tmp_path: Path) -> None:
        """Ephemeral home is cleaned up even when protocol raises."""
        skill_dir = _make_dir_skill(tmp_path, "my-skill")
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        original_render = prompt.render

        def patched_render(**kwargs: Any) -> Any:
            from weakincentives.prompt.rendering import RenderedPrompt

            rendered = original_render(**kwargs)
            return RenderedPrompt(
                text=rendered.text,
                _tools=rendered.tools,
                _skills=(SkillMount(source=skill_dir),),
            )

        with (
            patch(
                "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
            ) as MockClient,
            patch.object(prompt, "render", side_effect=patched_render),
            patch.object(CodexEphemeralHome, "cleanup") as mock_cleanup,
        ):
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = CodexClientError("fail")
            MockClient.return_value = mock_client

            with pytest.raises(PromptEvaluationError):
                adapter.evaluate(prompt, session=session)

        mock_cleanup.assert_called()

    def test_client_receives_merged_env(self, tmp_path: Path) -> None:
        """Config env and ephemeral home env are merged."""
        skill_dir = _make_dir_skill(tmp_path, "my-skill")
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(
                cwd="/tmp/test",
                env={"CUSTOM_VAR": "value"},
            ),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        original_render = prompt.render

        def patched_render(**kwargs: Any) -> Any:
            from weakincentives.prompt.rendering import RenderedPrompt

            rendered = original_render(**kwargs)
            return RenderedPrompt(
                text=rendered.text,
                _tools=rendered.tools,
                _skills=(SkillMount(source=skill_dir),),
            )

        with (
            patch(
                "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
            ) as MockClient,
            patch.object(prompt, "render", side_effect=patched_render),
        ):
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(
                self._success_messages()
            )
            MockClient.return_value = mock_client

            adapter.evaluate(prompt, session=session)

        constructor_kwargs = MockClient.call_args[1]
        env = constructor_kwargs.get("env") or {}
        assert env.get("CUSTOM_VAR") == "value"
        assert "HOME" in env

    def test_ephemeral_home_cleaned_up_on_mount_failure(self, tmp_path: Path) -> None:
        """Ephemeral home is cleaned up when mount_skills raises."""
        nonexistent = tmp_path / "does-not-exist"
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        original_render = prompt.render

        def patched_render(**kwargs: Any) -> Any:
            from weakincentives.prompt.rendering import RenderedPrompt

            rendered = original_render(**kwargs)
            return RenderedPrompt(
                text=rendered.text,
                _tools=rendered.tools,
                _skills=(SkillMount(source=nonexistent),),
            )

        with (
            patch.object(prompt, "render", side_effect=patched_render),
            patch.object(CodexEphemeralHome, "cleanup") as mock_cleanup,
        ):
            from weakincentives.skills import SkillNotFoundError

            with pytest.raises(SkillNotFoundError):
                adapter.evaluate(prompt, session=session)

        mock_cleanup.assert_called()

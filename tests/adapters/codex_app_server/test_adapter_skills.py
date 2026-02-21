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

"""Tests for transcript bridge integration and skill mounting in the adapter."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from weakincentives.adapters.codex_app_server._ephemeral_home import CodexEphemeralHome
from weakincentives.adapters.codex_app_server._protocol import handle_tool_call
from weakincentives.adapters.codex_app_server.adapter import CodexAppServerAdapter
from weakincentives.adapters.codex_app_server.client import CodexClientError
from weakincentives.adapters.codex_app_server.config import (
    CodexAppServerClientConfig,
)
from weakincentives.adapters.core import PromptEvaluationError
from weakincentives.skills import SkillMount

from .conftest import (
    make_mock_client,
    make_session,
    make_simple_prompt,
    messages_iterator,
)

# ---------------------------------------------------------------------------
# Helpers for skill frontmatter
# ---------------------------------------------------------------------------

_SKILL_MD = "---\nname: {name}\ndescription: A test skill\n---\n\n# {name}\n"


def _make_dir_skill(base: Path, name: str) -> Path:
    d = base / name
    d.mkdir()
    (d / "SKILL.md").write_text(_SKILL_MD.format(name=name))
    return d


# ---------------------------------------------------------------------------
# Transcript bridge integration
# ---------------------------------------------------------------------------


class TestTranscriptBridgeIntegration:
    """Tests for transcript bridge integration in the adapter."""

    def test_evaluate_with_transcript_disabled(self) -> None:
        """When transcript=False, bridge is not created."""
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test", transcript=False),
        )
        session, _ = make_session()
        prompt = make_simple_prompt()

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

    def test_handle_tool_call_with_bridge(self) -> None:
        """Tool call emits transcript entries via bridge."""

        async def _run() -> None:
            client = make_mock_client()
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
            client = make_mock_client()
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


# ---------------------------------------------------------------------------
# Skill mounting integration in _run_codex
# ---------------------------------------------------------------------------


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
        session, _ = make_session()
        prompt = make_simple_prompt()

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
            mock_client = make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = messages_iterator(
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
            mock_client.read_messages.return_value = messages_iterator(
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
        session, _ = make_session()
        prompt = make_simple_prompt()

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
            mock_client = make_mock_client()
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
        session, _ = make_session()
        prompt = make_simple_prompt()

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
            mock_client = make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = messages_iterator(
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
        session, _ = make_session()
        prompt = make_simple_prompt()

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

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

"""Unit tests for the generic ACP adapter."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from weakincentives.adapters.acp._env import build_env
from weakincentives.adapters.acp._prompt_loop import drain_quiet_period, extract_text
from weakincentives.adapters.acp.adapter import ACPAdapter
from weakincentives.adapters.acp.config import ACPAdapterConfig, ACPClientConfig
from weakincentives.adapters.core import PromptEvaluationError
from weakincentives.clock import SYSTEM_CLOCK
from weakincentives.runtime.events import InProcessDispatcher

from .conftest import (
    AgentMessageChunk,
    AgentThoughtChunk,
)


def _make_mock_session() -> MagicMock:
    """Create a mock session with dispatcher."""
    dispatcher = InProcessDispatcher()
    session = MagicMock()
    session.dispatcher = dispatcher
    session.session_id = "test-session"
    return session


def _make_mock_prompt(
    *,
    text: str = "Hello",
    output_type: type[Any] | None = None,
    container: str | None = None,
) -> MagicMock:
    """Create a mock prompt."""
    prompt = MagicMock()
    prompt.ns = "test"
    prompt.key = "prompt"
    prompt.name = "test-prompt"

    rendered = MagicMock()
    rendered.text = text
    rendered.tools = ()
    rendered.output_type = output_type
    rendered.container = container
    rendered.allow_extra_keys = False

    prompt.render.return_value = rendered
    prompt.filesystem.return_value = None
    prompt.bind.return_value = prompt
    prompt.resources = MagicMock()
    prompt.resources.__enter__ = MagicMock(return_value=None)
    prompt.resources.__exit__ = MagicMock(return_value=False)
    return prompt


def _make_mock_deadline(*, remaining_s: float = 60.0) -> MagicMock:
    """Create a mock deadline with remaining time."""
    deadline = MagicMock()
    deadline.remaining.return_value = timedelta(seconds=remaining_s)
    return deadline


@pytest.fixture()
def acp_adapter() -> ACPAdapter:
    """Create a default ACPAdapter instance."""
    return ACPAdapter()


@pytest.fixture()
def mock_session() -> MagicMock:
    """Create a mock session with dispatcher."""
    return _make_mock_session()


@pytest.fixture()
def mock_prompt() -> MagicMock:
    """Create a mock prompt with defaults."""
    return _make_mock_prompt()


@pytest.fixture()
def rendered_with_structured_output() -> MagicMock:
    """Create a rendered mock expecting structured output."""
    rendered = MagicMock()
    rendered.output_type = str
    rendered.container = "object"
    return rendered


@pytest.fixture()
def rendered_without_structured_output() -> MagicMock:
    """Create a rendered mock with no structured output."""
    rendered = MagicMock()
    rendered.tools = ()
    rendered.output_type = None
    rendered.container = None
    return rendered


class TestACPAdapterInit:
    def test_init_defaults_quiet_period_and_agent_bin(self) -> None:
        adapter = ACPAdapter()
        assert adapter._adapter_config.quiet_period_ms == 500
        assert adapter._client_config.agent_bin == "opencode"

    def test_init_custom_config_applies_model_and_quiet_period(self) -> None:
        cfg = ACPAdapterConfig(model_id="test-model", quiet_period_ms=100)
        adapter = ACPAdapter(adapter_config=cfg)
        assert adapter._adapter_config.model_id == "test-model"
        assert adapter._adapter_config.quiet_period_ms == 100


class TestACPAdapterName:
    def test_adapter_name_method_returns_acp(self) -> None:
        adapter = ACPAdapter()
        assert adapter._adapter_name() == "acp"

    def test_adapter_name_property_returns_acp(self) -> None:
        adapter = ACPAdapter()
        assert adapter.adapter_name == "acp"


class TestExpiredDeadline:
    def test_evaluate_expired_deadline_raises_prompt_evaluation_error(
        self, acp_adapter: ACPAdapter, mock_prompt: MagicMock
    ) -> None:
        session = _make_mock_session()
        deadline = _make_mock_deadline(remaining_s=-1.0)

        with pytest.raises(PromptEvaluationError, match="Deadline expired"):
            acp_adapter.evaluate(mock_prompt, session=session, deadline=deadline)


class TestResolveCwd:
    def test_resolve_cwd_config_cwd_returns_configured_path(self) -> None:
        adapter = ACPAdapter(client_config=ACPClientConfig(cwd="/tmp/work"))
        prompt = _make_mock_prompt()
        effective_cwd, temp_dir, _ = adapter._resolve_cwd(prompt)
        assert effective_cwd == "/tmp/work"
        assert temp_dir is None

    def test_resolve_cwd_no_cwd_creates_temp_dir(self) -> None:
        adapter = ACPAdapter(client_config=ACPClientConfig(cwd=None))
        prompt = _make_mock_prompt()
        effective_cwd, temp_dir, _ = adapter._resolve_cwd(prompt)
        assert effective_cwd is not None
        assert temp_dir is not None
        assert effective_cwd == temp_dir

        # Clean up
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_resolve_cwd_filesystem_root_used_when_no_config(self) -> None:
        from weakincentives.filesystem import HostFilesystem

        adapter = ACPAdapter(client_config=ACPClientConfig(cwd=None))
        prompt = _make_mock_prompt()
        fs = HostFilesystem(_root="/tmp/fs-root")
        prompt.filesystem.return_value = fs
        effective_cwd, temp_dir, _ = adapter._resolve_cwd(prompt)
        assert effective_cwd == "/tmp/fs-root"
        assert temp_dir is None

    def test_resolve_cwd_config_takes_precedence_over_filesystem(self) -> None:
        from weakincentives.filesystem import HostFilesystem

        adapter = ACPAdapter(client_config=ACPClientConfig(cwd="/tmp/configured"))
        prompt = _make_mock_prompt()
        fs = HostFilesystem(_root="/tmp/fs-root")
        prompt.filesystem.return_value = fs
        effective_cwd, temp_dir, _ = adapter._resolve_cwd(prompt)
        assert effective_cwd == "/tmp/configured"
        assert temp_dir is None

    def test_resolve_cwd_non_host_filesystem_falls_back_to_cwd(self) -> None:
        from pathlib import Path

        adapter = ACPAdapter(client_config=ACPClientConfig(cwd=None))
        prompt = _make_mock_prompt()
        mock_fs = MagicMock()
        mock_fs.__class__.__name__ = "SomeOtherFilesystem"
        prompt.filesystem.return_value = mock_fs
        effective_cwd, temp_dir, _ = adapter._resolve_cwd(prompt)
        assert effective_cwd is not None
        assert temp_dir is None
        assert effective_cwd == str(Path.cwd().resolve())


class TestExtractText:
    def test_extract_text_no_chunks_returns_none(self) -> None:
        client = MagicMock()
        client.message_chunks = []
        client.thought_chunks = []
        assert extract_text(client, emit_thought_chunks=False) is None

    def test_extract_text_message_chunks_concatenated(self) -> None:
        client = MagicMock()
        client.message_chunks = [
            AgentMessageChunk("Hello "),
            AgentMessageChunk("world"),
        ]
        client.thought_chunks = []
        assert extract_text(client, emit_thought_chunks=False) == "Hello world"

    def test_extract_text_thoughts_enabled_prepends_thoughts(self) -> None:
        client = MagicMock()
        client.message_chunks = [AgentMessageChunk("answer")]
        client.thought_chunks = [AgentThoughtChunk("thinking...")]
        result = extract_text(client, emit_thought_chunks=True)
        assert result == "thinking...answer"

    def test_extract_text_thoughts_disabled_excludes_thoughts(self) -> None:
        client = MagicMock()
        client.message_chunks = [AgentMessageChunk("answer")]
        client.thought_chunks = [AgentThoughtChunk("thinking...")]
        result = extract_text(client, emit_thought_chunks=False)
        assert result == "answer"

    def test_extract_text_empty_thought_content_skipped(self) -> None:
        client = MagicMock()
        client.message_chunks = [AgentMessageChunk("answer")]
        client.thought_chunks = [AgentThoughtChunk(""), AgentThoughtChunk("ok")]
        result = extract_text(client, emit_thought_chunks=True)
        assert result == "okanswer"

    def test_extract_text_empty_message_content_skipped(self) -> None:
        client = MagicMock()
        client.message_chunks = [
            AgentMessageChunk(""),
            AgentMessageChunk("text"),
        ]
        client.thought_chunks = []
        result = extract_text(client, emit_thought_chunks=False)
        assert result == "text"


class TestExtractChunkTextListContent:
    def test_extract_chunk_text_list_of_text_blocks_concatenated(self) -> None:
        from weakincentives.adapters.acp._prompt_loop import _extract_chunk_text

        block1 = MagicMock()
        block1.text = "hello "
        block2 = MagicMock()
        block2.text = "world"
        chunk = MagicMock()
        chunk.content = [block1, block2]
        assert _extract_chunk_text(chunk) == "hello world"

    def test_extract_chunk_text_non_text_blocks_use_str_fallback(self) -> None:
        from weakincentives.adapters.acp._prompt_loop import _extract_chunk_text

        class PlainBlock:
            def __str__(self) -> str:
                return "fallback"

        chunk = MagicMock()
        chunk.content = [PlainBlock()]
        assert _extract_chunk_text(chunk) == "fallback"

    def test_extract_chunk_text_list_skips_falsy_blocks(self) -> None:
        from weakincentives.adapters.acp._prompt_loop import _extract_chunk_text

        block = MagicMock()
        block.text = "kept"
        chunk = MagicMock()
        chunk.content = [None, block, 0, False]
        assert _extract_chunk_text(chunk) == "kept"

    def test_extract_chunk_text_text_content_block(self) -> None:
        from weakincentives.adapters.acp._prompt_loop import _extract_chunk_text

        inner = type("TextContentBlock", (), {"text": "hello"})()
        chunk = MagicMock()
        chunk.content = inner
        assert _extract_chunk_text(chunk) == "hello"

    def test_extract_chunk_text_non_string_non_list_uses_str(self) -> None:
        from weakincentives.adapters.acp._prompt_loop import _extract_chunk_text

        chunk = MagicMock()
        chunk.content = 42
        assert _extract_chunk_text(chunk) == "42"

    def test_extract_chunk_text_none_content_returns_empty(self) -> None:
        from weakincentives.adapters.acp._prompt_loop import _extract_chunk_text

        chunk = MagicMock()
        chunk.content = None
        assert _extract_chunk_text(chunk) == ""


class TestSubclassHooks:
    def test_validate_model_noop(self, acp_adapter: ACPAdapter) -> None:
        acp_adapter._validate_model("any-model", [])

    def test_handle_mode_error_no_raise(self, acp_adapter: ACPAdapter) -> None:
        acp_adapter._handle_mode_error(RuntimeError("test error"))

    def test_detect_empty_response_noop(self, acp_adapter: ACPAdapter) -> None:
        acp_adapter._detect_empty_response(MagicMock(), MagicMock())


class TestStructuredOutputResolution:
    """Test structured output edge cases."""

    def test_resolve_structured_output_missing_text_raises(
        self, acp_adapter: ACPAdapter
    ) -> None:
        rendered = MagicMock()
        rendered.output_type = str

        with pytest.raises(PromptEvaluationError, match="Structured output required"):
            acp_adapter._resolve_structured_output(None, rendered, "p", None)

    def test_resolve_structured_output_capture_parse_error_raises(
        self, acp_adapter: ACPAdapter
    ) -> None:
        rendered = MagicMock()
        rendered.output_type = str
        capture = MagicMock()
        capture.called = True
        capture.data = "invalid"

        with (
            patch(
                "weakincentives.adapters.acp.adapter.parse_structured_output",
                side_effect=ValueError("bad"),
            ),
            pytest.raises(PromptEvaluationError, match="Failed to parse"),
        ):
            acp_adapter._resolve_structured_output("text", rendered, "p", capture)

    def test_resolve_structured_output_text_parse_error_raises(
        self, acp_adapter: ACPAdapter
    ) -> None:
        rendered = MagicMock()
        rendered.output_type = str
        capture = MagicMock()
        capture.called = False

        with (
            patch(
                "weakincentives.adapters.acp.adapter.parse_structured_output",
                side_effect=ValueError("bad text"),
            ),
            pytest.raises(PromptEvaluationError, match="Failed to parse"),
        ):
            acp_adapter._resolve_structured_output("raw text", rendered, "p", capture)

    def test_resolve_structured_output_capture_succeeds(
        self, acp_adapter: ACPAdapter
    ) -> None:
        rendered = MagicMock()
        rendered.output_type = str
        capture = MagicMock()
        capture.called = True
        capture.data = {"key": "value"}

        with patch(
            "weakincentives.adapters.acp.adapter.parse_structured_output",
            return_value="parsed",
        ) as mock_parse:
            result = acp_adapter._resolve_structured_output(
                "text", rendered, "p", capture
            )

        assert result == "parsed"
        mock_parse.assert_called_once_with('{"key": "value"}', rendered)


class TestBuildEnv:
    def test_build_env_no_config_returns_none(self) -> None:
        adapter = ACPAdapter(client_config=ACPClientConfig(env=None))
        assert build_env(adapter._client_config.env) is None

    def test_build_env_merges_with_os_environ(self) -> None:
        adapter = ACPAdapter(client_config=ACPClientConfig(env={"MY_VAR": "my_val"}))
        result = build_env(adapter._client_config.env)
        assert result is not None
        assert result["MY_VAR"] == "my_val"
        assert "PATH" in result


class TestPrepareExecutionEnv:
    def test_prepare_env_no_config_returns_none_env(
        self, acp_adapter: ACPAdapter
    ) -> None:
        env, cleanup = acp_adapter._prepare_execution_env(
            rendered=MagicMock(),
            effective_cwd="/tmp",
        )
        assert env is None
        cleanup()  # Should be a no-op, must not raise

    def test_prepare_env_with_config_returns_merged_env(self) -> None:
        adapter = ACPAdapter(client_config=ACPClientConfig(env={"KEY": "val"}))
        env, cleanup = adapter._prepare_execution_env(
            rendered=MagicMock(),
            effective_cwd="/tmp",
        )
        assert env is not None
        assert env["KEY"] == "val"
        cleanup()


@pytest.fixture()
def _prepare_tools_common(
    mock_session: MagicMock, mock_prompt: MagicMock
) -> dict[str, Any]:
    """Common keyword arguments for _prepare_tools calls."""
    return {
        "session": mock_session,
        "prompt": mock_prompt,
        "deadline": None,
        "budget_tracker": None,
        "adapter_name": "acp",
        "prompt_name": "test",
        "heartbeat": None,
        "run_context": None,
        "visibility_signal": MagicMock(),
    }


class TestPrepareTools:
    def test_prepare_tools_no_structured_output_returns_empty(
        self,
        acp_adapter: ACPAdapter,
        rendered_without_structured_output: MagicMock,
        _prepare_tools_common: dict[str, Any],
    ) -> None:
        with patch(
            "weakincentives.adapters.acp.adapter.create_bridged_tools"
        ) as mock_bt:
            mock_bt.return_value = ()
            tools, capture = acp_adapter._prepare_tools(
                rendered=rendered_without_structured_output,
                **_prepare_tools_common,
            )

        assert capture is None
        assert tools == []

    def test_prepare_tools_with_structured_output_includes_capture_tool(
        self,
        acp_adapter: ACPAdapter,
        rendered_with_structured_output: MagicMock,
        _prepare_tools_common: dict[str, Any],
    ) -> None:
        with (
            patch(
                "weakincentives.adapters.acp.adapter.create_bridged_tools"
            ) as mock_bt,
            patch(
                "weakincentives.adapters.acp.adapter.create_structured_output_tool"
            ) as mock_sot,
        ):
            mock_bt.return_value = ()
            mock_tool = MagicMock()
            mock_capture = MagicMock()
            mock_sot.return_value = (mock_tool, mock_capture)

            tools, capture = acp_adapter._prepare_tools(
                rendered=rendered_with_structured_output,
                **_prepare_tools_common,
            )

        assert capture is mock_capture
        assert mock_tool in tools


class TestFinalizeResponse:
    def test_finalize_response_basic_text(
        self, acp_adapter: ACPAdapter, mock_session: MagicMock
    ) -> None:
        rendered = MagicMock()
        rendered.output_type = None

        response = acp_adapter._finalize_response(
            result=("response text", None),
            rendered=rendered,
            prompt_name="test-prompt",
            adapter_name="acp",
            session=mock_session,
            budget_tracker=None,
            run_context=None,
            start_time=datetime.now(UTC),
            structured_capture=None,
        )

        assert response.text == "response text"
        assert response.prompt_name == "test-prompt"

    def test_finalize_response_with_structured_output(
        self, acp_adapter: ACPAdapter, mock_session: MagicMock
    ) -> None:
        rendered = MagicMock()
        rendered.output_type = str
        rendered.container = "object"

        capture = MagicMock()
        capture.called = True
        capture.data = "parsed_data"

        with patch(
            "weakincentives.adapters.acp.adapter.parse_structured_output",
            return_value="parsed_result",
        ):
            response = acp_adapter._finalize_response(
                result=("text", None),
                rendered=rendered,
                prompt_name="test-prompt",
                adapter_name="acp",
                session=mock_session,
                budget_tracker=None,
                run_context=None,
                start_time=datetime.now(UTC),
                structured_capture=capture,
            )

        assert response.output == "parsed_result"


class TestDrainQuietPeriod:
    def test_drain_zero_quiet_period_returns_immediately(self) -> None:
        client = MagicMock()
        client.last_update_time = 0.0

        asyncio.run(
            drain_quiet_period(
                client,
                None,
                quiet_period_ms=0,
                clock=SYSTEM_CLOCK,
                async_sleeper=SYSTEM_CLOCK,
            )
        )

    def test_drain_with_deadline_respects_timeout(self) -> None:
        client = MagicMock()
        client.last_update_time = time.monotonic()

        deadline = _make_mock_deadline(remaining_s=0.001)

        asyncio.run(
            drain_quiet_period(
                client,
                deadline,
                quiet_period_ms=50,
                clock=SYSTEM_CLOCK,
                async_sleeper=SYSTEM_CLOCK,
            )
        )

    def test_drain_waits_for_quiet_period(self) -> None:
        client = MagicMock()
        client.last_update_time = time.monotonic()

        asyncio.run(
            drain_quiet_period(
                client,
                None,
                quiet_period_ms=10,
                clock=SYSTEM_CLOCK,
                async_sleeper=SYSTEM_CLOCK,
            )
        )

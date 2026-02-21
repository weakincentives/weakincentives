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

from weakincentives.adapters.acp._execution import (
    build_env,
    drain_quiet_period,
    extract_chunk_text,
    extract_client_text,
    resolve_structured_output,
)
from weakincentives.adapters.acp.config import ACPAdapterConfig, ACPClientConfig
from weakincentives.adapters.core import PromptEvaluationError
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


class TestACPAdapterInit:
    def test_defaults(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter

        adapter = ACPAdapter()
        assert adapter._adapter_config.quiet_period_ms == 500
        assert adapter._client_config.agent_bin == "opencode"

    def test_custom_config(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter

        cfg = ACPAdapterConfig(model_id="test-model", quiet_period_ms=100)
        adapter = ACPAdapter(adapter_config=cfg)
        assert adapter._adapter_config.model_id == "test-model"
        assert adapter._adapter_config.quiet_period_ms == 100


class TestACPAdapterName:
    def test_returns_acp(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter

        adapter = ACPAdapter()
        assert adapter._adapter_name() == "acp"

    def test_adapter_name_property(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter

        adapter = ACPAdapter()
        assert adapter.adapter_name == "acp"


class TestExpiredDeadline:
    def test_raises_on_expired_deadline(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter

        adapter = ACPAdapter()
        prompt = _make_mock_prompt()
        session = _make_mock_session()
        deadline = _make_mock_deadline(remaining_s=-1.0)

        with pytest.raises(PromptEvaluationError, match="Deadline expired"):
            adapter.evaluate(prompt, session=session, deadline=deadline)


class TestResolveCwd:
    def test_uses_config_cwd(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter

        adapter = ACPAdapter(client_config=ACPClientConfig(cwd="/tmp/work"))
        prompt = _make_mock_prompt()
        effective_cwd, temp_dir, _ = adapter._resolve_cwd(prompt)
        assert effective_cwd == "/tmp/work"
        assert temp_dir is None

    def test_creates_temp_dir_when_no_cwd(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter

        adapter = ACPAdapter(client_config=ACPClientConfig(cwd=None))
        prompt = _make_mock_prompt()
        effective_cwd, temp_dir, _ = adapter._resolve_cwd(prompt)
        assert effective_cwd is not None
        assert temp_dir is not None
        assert effective_cwd == temp_dir

        # Clean up
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_uses_filesystem_root(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter
        from weakincentives.filesystem import HostFilesystem

        adapter = ACPAdapter(client_config=ACPClientConfig(cwd=None))
        prompt = _make_mock_prompt()
        fs = HostFilesystem(_root="/tmp/fs-root")
        prompt.filesystem.return_value = fs
        effective_cwd, temp_dir, _ = adapter._resolve_cwd(prompt)
        assert effective_cwd == "/tmp/fs-root"
        assert temp_dir is None

    def test_config_cwd_with_existing_filesystem(self) -> None:
        """Covers branch where filesystem exists AND cwd is configured."""
        from weakincentives.adapters.acp.adapter import ACPAdapter
        from weakincentives.filesystem import HostFilesystem

        adapter = ACPAdapter(client_config=ACPClientConfig(cwd="/tmp/configured"))
        prompt = _make_mock_prompt()
        fs = HostFilesystem(_root="/tmp/fs-root")
        prompt.filesystem.return_value = fs
        effective_cwd, temp_dir, _ = adapter._resolve_cwd(prompt)
        # Config cwd takes precedence, filesystem is not None so elif is skipped
        assert effective_cwd == "/tmp/configured"
        assert temp_dir is None

    def test_non_host_filesystem_falls_back_to_cwd(self) -> None:
        """Covers the fallback to Path.cwd() when filesystem is not HostFilesystem."""
        from weakincentives.adapters.acp.adapter import ACPAdapter

        adapter = ACPAdapter(client_config=ACPClientConfig(cwd=None))
        prompt = _make_mock_prompt()
        # Non-HostFilesystem object
        mock_fs = MagicMock()
        mock_fs.__class__.__name__ = "SomeOtherFilesystem"
        prompt.filesystem.return_value = mock_fs
        effective_cwd, temp_dir, _ = adapter._resolve_cwd(prompt)
        assert effective_cwd is not None
        assert temp_dir is None
        # Falls back to current directory
        from pathlib import Path

        assert effective_cwd == str(Path.cwd().resolve())


class TestExtractText:
    def test_no_message_chunks(self) -> None:
        client = MagicMock()
        client.message_chunks = []
        client.thought_chunks = []
        assert extract_client_text(client, emit_thought_chunks=False) is None

    def test_message_chunks(self) -> None:
        client = MagicMock()
        client.message_chunks = [
            AgentMessageChunk("Hello "),
            AgentMessageChunk("world"),
        ]
        client.thought_chunks = []
        assert extract_client_text(client, emit_thought_chunks=False) == "Hello world"

    def test_with_thoughts_enabled(self) -> None:
        client = MagicMock()
        client.message_chunks = [AgentMessageChunk("answer")]
        client.thought_chunks = [AgentThoughtChunk("thinking...")]
        result = extract_client_text(client, emit_thought_chunks=True)
        assert result == "thinking...answer"

    def test_with_thoughts_disabled(self) -> None:
        client = MagicMock()
        client.message_chunks = [AgentMessageChunk("answer")]
        client.thought_chunks = [AgentThoughtChunk("thinking...")]
        result = extract_client_text(client, emit_thought_chunks=False)
        assert result == "answer"

    def test_empty_thought_content_skipped(self) -> None:
        """Covers branch where thought chunk content is empty."""
        client = MagicMock()
        client.message_chunks = [AgentMessageChunk("answer")]
        client.thought_chunks = [AgentThoughtChunk(""), AgentThoughtChunk("ok")]
        result = extract_client_text(client, emit_thought_chunks=True)
        assert result == "okanswer"

    def test_empty_message_content_skipped(self) -> None:
        """Covers branch where message chunk content is empty."""
        client = MagicMock()
        client.message_chunks = [
            AgentMessageChunk(""),
            AgentMessageChunk("text"),
        ]
        client.thought_chunks = []
        result = extract_client_text(client, emit_thought_chunks=False)
        assert result == "text"


class TestExtractChunkTextListContent:
    def test_list_of_content_blocks(self) -> None:
        block1 = MagicMock()
        block1.text = "hello "
        block2 = MagicMock()
        block2.text = "world"
        chunk = MagicMock()
        chunk.content = [block1, block2]
        assert extract_chunk_text(chunk) == "hello world"

    def test_list_with_non_text_blocks(self) -> None:
        class PlainBlock:
            def __str__(self) -> str:
                return "fallback"

        chunk = MagicMock()
        chunk.content = [PlainBlock()]
        assert extract_chunk_text(chunk) == "fallback"

    def test_list_skips_falsy_blocks(self) -> None:
        block = MagicMock()
        block.text = "kept"
        chunk = MagicMock()
        chunk.content = [None, block, 0, False]
        assert extract_chunk_text(chunk) == "kept"

    def test_text_content_block(self) -> None:
        inner = type("TextContentBlock", (), {"text": "hello"})()
        chunk = MagicMock()
        chunk.content = inner
        assert extract_chunk_text(chunk) == "hello"

    def test_non_string_non_list_fallback(self) -> None:
        chunk = MagicMock()
        chunk.content = 42
        assert extract_chunk_text(chunk) == "42"

    def test_falsy_non_string_content(self) -> None:
        chunk = MagicMock()
        chunk.content = None
        assert extract_chunk_text(chunk) == ""


class TestSubclassHooks:
    def test_validate_model_is_noop(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter

        adapter = ACPAdapter()
        # Should not raise
        adapter._validate_model("any-model", [])

    def test_handle_mode_error_logs_warning(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter

        adapter = ACPAdapter()
        # Should not raise
        adapter._handle_mode_error(RuntimeError("test error"))

    def test_detect_empty_response_is_noop(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter

        adapter = ACPAdapter()
        client = MagicMock()
        # Should not raise
        adapter._detect_empty_response(client, MagicMock())


class TestStructuredOutputResolution:
    """Test structured output edge cases."""

    def test_missing_output_raises(self) -> None:
        rendered = MagicMock()
        rendered.output_type = str

        with pytest.raises(PromptEvaluationError, match="Structured output required"):
            resolve_structured_output(None, rendered, "p", None)

    def test_capture_parse_error_raises(self) -> None:
        rendered = MagicMock()
        rendered.output_type = str

        capture = MagicMock()
        capture.called = True
        capture.data = "invalid"

        with (
            patch(
                "weakincentives.adapters.acp._execution.parse_structured_output",
                side_effect=ValueError("bad"),
            ),
            pytest.raises(PromptEvaluationError, match="Failed to parse"),
        ):
            resolve_structured_output("text", rendered, "p", capture)

    def test_text_parse_error_raises(self) -> None:
        rendered = MagicMock()
        rendered.output_type = str

        capture = MagicMock()
        capture.called = False

        with (
            patch(
                "weakincentives.adapters.acp._execution.parse_structured_output",
                side_effect=ValueError("bad text"),
            ),
            pytest.raises(PromptEvaluationError, match="Failed to parse"),
        ):
            resolve_structured_output("raw text", rendered, "p", capture)

    def test_capture_succeeds(self) -> None:
        rendered = MagicMock()
        rendered.output_type = str

        capture = MagicMock()
        capture.called = True
        capture.data = {"key": "value"}

        with patch(
            "weakincentives.adapters.acp._execution.parse_structured_output",
            return_value="parsed",
        ) as mock_parse:
            result = resolve_structured_output("text", rendered, "p", capture)

        assert result == "parsed"
        # Verify capture data is JSON-serialized before parsing
        mock_parse.assert_called_once_with('{"key": "value"}', rendered)


class TestBuildEnv:
    def test_returns_none_when_no_env(self) -> None:
        client_config = ACPClientConfig(env=None)
        assert build_env(client_config) is None

    def test_merges_with_os_environ(self) -> None:
        client_config = ACPClientConfig(env={"MY_VAR": "my_val"})
        result = build_env(client_config)
        assert result is not None
        assert result["MY_VAR"] == "my_val"
        # os.environ keys should also be present
        assert "PATH" in result


class TestPrepareExecutionEnv:
    def test_default_returns_build_env_and_noop(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter

        adapter = ACPAdapter(client_config=ACPClientConfig(env=None))
        rendered = MagicMock()

        env, cleanup = adapter._prepare_execution_env(
            rendered=rendered,
            effective_cwd="/tmp",
        )

        assert env is None  # _build_env returns None when no config env
        cleanup()  # Should be a no-op, must not raise

    def test_default_with_config_env(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter

        adapter = ACPAdapter(
            client_config=ACPClientConfig(env={"KEY": "val"}),
        )
        rendered = MagicMock()

        env, cleanup = adapter._prepare_execution_env(
            rendered=rendered,
            effective_cwd="/tmp",
        )

        assert env is not None
        assert env["KEY"] == "val"
        cleanup()


class TestPrepareTools:
    def test_no_structured_output(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter

        adapter = ACPAdapter()
        rendered = MagicMock()
        rendered.tools = ()
        rendered.output_type = None
        rendered.container = None

        prompt = _make_mock_prompt()
        session = _make_mock_session()

        with patch(
            "weakincentives.adapters.acp.adapter.create_bridged_tools"
        ) as mock_bt:
            mock_bt.return_value = ()
            tools, capture = adapter._prepare_tools(
                rendered=rendered,
                session=session,
                prompt=prompt,
                deadline=None,
                budget_tracker=None,
                adapter_name="acp",
                prompt_name="test",
                heartbeat=None,
                run_context=None,
                visibility_signal=MagicMock(),
            )

        assert capture is None
        assert tools == []

    def test_with_structured_output(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter

        adapter = ACPAdapter()
        rendered = MagicMock()
        rendered.tools = ()
        rendered.output_type = str
        rendered.container = "object"

        prompt = _make_mock_prompt()
        session = _make_mock_session()

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

            tools, capture = adapter._prepare_tools(
                rendered=rendered,
                session=session,
                prompt=prompt,
                deadline=None,
                budget_tracker=None,
                adapter_name="acp",
                prompt_name="test",
                heartbeat=None,
                run_context=None,
                visibility_signal=MagicMock(),
            )

        assert capture is mock_capture
        assert mock_tool in tools


class TestFinalizeResponse:
    def test_basic_response(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter

        adapter = ACPAdapter()
        session = _make_mock_session()
        rendered = MagicMock()
        rendered.output_type = None

        result = (
            "response text",
            None,
        )

        response = adapter._finalize_response(
            result=result,
            rendered=rendered,
            prompt_name="test-prompt",
            adapter_name="acp",
            session=session,
            budget_tracker=None,
            run_context=None,
            start_time=datetime.now(UTC),
            structured_capture=None,
        )

        assert response.text == "response text"
        assert response.prompt_name == "test-prompt"

    def test_with_structured_output(self) -> None:
        from weakincentives.adapters.acp.adapter import ACPAdapter

        adapter = ACPAdapter()
        session = _make_mock_session()
        rendered = MagicMock()
        rendered.output_type = str
        rendered.container = "object"

        capture = MagicMock()
        capture.called = True
        capture.data = "parsed_data"

        with patch(
            "weakincentives.adapters.acp._execution.parse_structured_output",
            return_value="parsed_result",
        ):
            response = adapter._finalize_response(
                result=("text", None),
                rendered=rendered,
                prompt_name="test-prompt",
                adapter_name="acp",
                session=session,
                budget_tracker=None,
                run_context=None,
                start_time=datetime.now(UTC),
                structured_capture=capture,
            )

        assert response.output == "parsed_result"


class TestDrainQuietPeriod:
    def test_immediate_drain_when_zero_quiet_period(self) -> None:
        from weakincentives.clock import SYSTEM_CLOCK

        client = MagicMock()
        client.last_update_time = 0.0

        # Should return immediately
        asyncio.run(
            drain_quiet_period(
                client,
                None,
                quiet_period_ms=0,
                clock=SYSTEM_CLOCK,
                async_sleeper=SYSTEM_CLOCK,
            )
        )

    def test_drain_with_deadline(self) -> None:
        from weakincentives.clock import SYSTEM_CLOCK

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

    def test_drain_waits_for_quiet(self) -> None:
        from weakincentives.clock import SYSTEM_CLOCK

        client = MagicMock()
        # Recent update — should wait
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

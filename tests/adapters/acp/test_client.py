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

"""Tests for ACP Client implementation."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from weakincentives.adapters.acp.config import ACPClientConfig

from .conftest import (
    MockAgentMessageChunk,
    MockAgentThoughtChunk,
    MockPermissionRequestResponse,
    MockReadTextFileResponse,
    MockRequestError,
    MockSessionAccumulator,
    MockSessionNotification,
    MockToolCallProgress,
    MockToolCallStart,
    MockWriteTextFileResponse,
)


def _mock_acp_modules() -> dict[str, Any]:
    mock_acp = MagicMock()
    mock_acp.RequestError = MockRequestError
    mock_schema = MagicMock()
    mock_schema.SessionNotification = MockSessionNotification
    mock_schema.PermissionRequestResponse = MockPermissionRequestResponse
    mock_schema.ReadTextFileResponse = MockReadTextFileResponse
    mock_schema.WriteTextFileResponse = MockWriteTextFileResponse
    return {"acp": mock_acp, "acp.schema": mock_schema}


def _make_client(
    *,
    permission_mode: str = "auto",
    allow_file_reads: bool = False,
    allow_file_writes: bool = False,
    workspace_root: str | None = None,
) -> Any:
    from weakincentives.adapters.acp.client import ACPClient

    config = ACPClientConfig(
        permission_mode=permission_mode,  # type: ignore[arg-type]
        allow_file_reads=allow_file_reads,
        allow_file_writes=allow_file_writes,
    )
    return ACPClient(config, workspace_root=workspace_root)


class TestACPClientSessionUpdate:
    def test_wraps_in_session_notification(self) -> None:
        async def _run() -> None:
            client = _make_client()
            acc = MockSessionAccumulator()
            client.set_accumulator(acc)
            update = MockAgentMessageChunk(content="hello")
            await client.session_update("sess-1", update)
            assert len(acc._notifications) == 1
            notif = acc._notifications[0]
            assert isinstance(notif, MockSessionNotification)
            assert notif.sessionId == "sess-1"
            assert notif.update is update

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_tracks_message_chunks(self) -> None:
        async def _run() -> None:
            client = _make_client()
            msg = MockAgentMessageChunk(content="hi")
            await client.session_update("sess-1", msg)
            assert len(client.message_chunks) == 1
            assert client.message_chunks[0] is msg

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_tracks_thought_chunks(self) -> None:
        async def _run() -> None:
            client = _make_client()
            thought = MockAgentThoughtChunk(content="thinking")
            await client.session_update("sess-1", thought)
            assert len(client.thought_chunks) == 1
            assert client.thought_chunks[0] is thought

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_tracks_tool_call_start(self) -> None:
        async def _run() -> None:
            client = _make_client()
            tool = MockToolCallStart(id="tc-1", title="bash")
            await client.session_update("sess-1", tool)
            assert "tc-1" in client.tool_call_tracker
            assert client.tool_call_tracker["tc-1"] == {
                "title": "bash",
                "status": "started",
                "output": "",
            }

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_tracks_tool_call_progress_existing(self) -> None:
        async def _run() -> None:
            client = _make_client()
            start = MockToolCallStart(id="tc-1", title="bash")
            await client.session_update("sess-1", start)
            progress = MockToolCallProgress(
                id="tc-1", title="bash", status="completed", output="ok"
            )
            await client.session_update("sess-1", progress)
            assert client.tool_call_tracker["tc-1"] == {
                "title": "bash",
                "status": "completed",
                "output": "ok",
            }

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_tracks_tool_call_progress_existing_empty_output(self) -> None:
        async def _run() -> None:
            client = _make_client()
            start = MockToolCallStart(id="tc-1", title="bash")
            await client.session_update("sess-1", start)
            progress = MockToolCallProgress(
                id="tc-1", title="bash", status="completed", output=""
            )
            await client.session_update("sess-1", progress)
            assert client.tool_call_tracker["tc-1"]["status"] == "completed"
            assert client.tool_call_tracker["tc-1"]["output"] == ""

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_tracks_tool_call_progress_new(self) -> None:
        async def _run() -> None:
            client = _make_client()
            progress = MockToolCallProgress(
                id="tc-2", title="search", status="completed", output="found"
            )
            await client.session_update("sess-1", progress)
            assert client.tool_call_tracker["tc-2"] == {
                "title": "search",
                "status": "completed",
                "output": "found",
            }

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_tracks_tool_call_progress_truncates_output(self) -> None:
        async def _run() -> None:
            client = _make_client()
            progress = MockToolCallProgress(
                id="tc-3", title="bash", status="completed", output="x" * 2000
            )
            await client.session_update("sess-1", progress)
            assert len(client.tool_call_tracker["tc-3"]["output"]) == 1000

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_synthetic_id_for_empty_tool_call_start(self) -> None:
        async def _run() -> None:
            client = _make_client()
            t1 = MockToolCallStart(id="", title="glob")
            t2 = MockToolCallStart(id="", title="read")
            await client.session_update("sess-1", t1)
            await client.session_update("sess-1", t2)
            # Each gets a unique synthetic ID
            assert len(client.tool_call_tracker) == 2
            assert "_tc_1" in client.tool_call_tracker
            assert "_tc_2" in client.tool_call_tracker
            assert client.tool_call_tracker["_tc_1"]["title"] == "glob"
            assert client.tool_call_tracker["_tc_2"]["title"] == "read"

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_synthetic_id_progress_follows_start(self) -> None:
        async def _run() -> None:
            client = _make_client()
            await client.session_update(
                "sess-1", MockToolCallStart(id="", title="bash")
            )
            await client.session_update(
                "sess-1",
                MockToolCallProgress(
                    id="", title="bash", status="completed", output="ok"
                ),
            )
            assert client.tool_call_tracker["_tc_1"]["status"] == "completed"
            assert client.tool_call_tracker["_tc_1"]["output"] == "ok"

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_real_ids_not_replaced(self) -> None:
        async def _run() -> None:
            client = _make_client()
            await client.session_update(
                "sess-1", MockToolCallStart(id="real-1", title="bash")
            )
            assert "real-1" in client.tool_call_tracker
            assert "_tc_" not in str(client.tool_call_tracker.keys())

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_forwards_to_transcript_bridge(self) -> None:
        async def _run() -> None:
            client = _make_client()
            bridge = MagicMock()
            client.set_transcript_bridge(bridge)
            msg = MockAgentMessageChunk(content="hi")
            await client.session_update("sess-1", msg)
            bridge.on_update.assert_called_once_with(msg)

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_updates_last_update_time(self) -> None:
        async def _run() -> None:
            client = _make_client()
            assert client.last_update_time is None
            await client.session_update("sess-1", MagicMock())
            assert client.last_update_time is not None
            assert client.last_update_time > 0.0

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_no_accumulator(self) -> None:
        async def _run() -> None:
            client = _make_client()
            msg = MockAgentMessageChunk(content="hello")
            await client.session_update("sess-1", msg)
            assert len(client.message_chunks) == 1

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())


class TestACPClientRequestPermission:
    def test_auto_mode_approves(self) -> None:
        async def _run() -> None:
            client = _make_client(permission_mode="auto")
            result = await client.request_permission(
                options=None, session_id="s", tool_call=None
            )
            assert isinstance(result, MockPermissionRequestResponse)
            assert result.approved

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_deny_mode_denies(self) -> None:
        async def _run() -> None:
            client = _make_client(permission_mode="deny")
            result = await client.request_permission(
                options=None, session_id="s", tool_call=None
            )
            assert not result.approved
            assert "policy" in (result.reason or "")

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_prompt_mode_denies(self) -> None:
        async def _run() -> None:
            client = _make_client(permission_mode="prompt")
            result = await client.request_permission(
                options=None, session_id="s", tool_call=None
            )
            assert not result.approved
            assert "non-interactive" in (result.reason or "")

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())


class TestACPClientTerminalMethods:
    def test_create_terminal_raises(self) -> None:
        async def _run() -> None:
            client = _make_client()
            with pytest.raises(MockRequestError) as exc_info:
                await client.create_terminal(command="ls", session_id="s")
            assert exc_info.value.code == -32601

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_terminal_output_raises(self) -> None:
        async def _run() -> None:
            client = _make_client()
            with pytest.raises(MockRequestError):
                await client.terminal_output(session_id="s", terminal_id="t")

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_release_terminal_raises(self) -> None:
        async def _run() -> None:
            client = _make_client()
            with pytest.raises(MockRequestError):
                await client.release_terminal(session_id="s", terminal_id="t")

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_wait_for_terminal_exit_raises(self) -> None:
        async def _run() -> None:
            client = _make_client()
            with pytest.raises(MockRequestError):
                await client.wait_for_terminal_exit(session_id="s", terminal_id="t")

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_kill_terminal_raises(self) -> None:
        async def _run() -> None:
            client = _make_client()
            with pytest.raises(MockRequestError):
                await client.kill_terminal(session_id="s", terminal_id="t")

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())


class TestACPClientExtMethods:
    def test_ext_method_raises(self) -> None:
        async def _run() -> None:
            client = _make_client()
            with pytest.raises(MockRequestError) as exc_info:
                await client.ext_method(method="custom", params={})
            assert exc_info.value.code == -32601

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_ext_notification_returns_none(self) -> None:
        async def _run() -> None:
            client = _make_client()
            result = await client.ext_notification(method="custom", params={})
            assert result is None

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())


class TestACPClientFileOps:
    def test_read_not_allowed(self) -> None:
        async def _run() -> None:
            client = _make_client(allow_file_reads=False)
            with pytest.raises(MockRequestError) as exc_info:
                await client.read_text_file(path="/f", session_id="s")
            assert exc_info.value.code == -32601

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_read_no_workspace(self) -> None:
        async def _run() -> None:
            client = _make_client(allow_file_reads=True, workspace_root=None)
            with pytest.raises(MockRequestError) as exc_info:
                await client.read_text_file(path="/f", session_id="s")
            assert exc_info.value.code == -32601

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_read_outside_workspace(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")

        async def _run() -> None:
            client = _make_client(allow_file_reads=True, workspace_root=str(workspace))
            with pytest.raises(MockRequestError) as exc_info:
                await client.read_text_file(path=str(outside), session_id="s")
            assert exc_info.value.code == -32600

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_read_prefix_sibling_directory_blocked(self, tmp_path: Path) -> None:
        """A sibling directory sharing a prefix must not bypass containment."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        evil = tmp_path / "ws-evil"
        evil.mkdir()
        secret = evil / "secret.txt"
        secret.write_text("stolen")

        async def _run() -> None:
            client = _make_client(allow_file_reads=True, workspace_root=str(workspace))
            with pytest.raises(MockRequestError) as exc_info:
                await client.read_text_file(path=str(secret), session_id="s")
            assert exc_info.value.code == -32600

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_read_parent_traversal_blocked(self, tmp_path: Path) -> None:
        """Path traversal via ``../`` must be rejected."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        outside = tmp_path / "secret.txt"
        outside.write_text("stolen")

        async def _run() -> None:
            client = _make_client(allow_file_reads=True, workspace_root=str(workspace))
            traversal_path = str(workspace / ".." / "secret.txt")
            with pytest.raises(MockRequestError) as exc_info:
                await client.read_text_file(path=traversal_path, session_id="s")
            assert exc_info.value.code == -32600

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_read_success(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        target = workspace / "hello.txt"
        target.write_text("hello world")

        async def _run() -> None:
            client = _make_client(allow_file_reads=True, workspace_root=str(workspace))
            result = await client.read_text_file(path=str(target), session_id="s")
            assert isinstance(result, MockReadTextFileResponse)
            assert result.content == "hello world"

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_write_not_allowed(self) -> None:
        async def _run() -> None:
            client = _make_client(allow_file_writes=False)
            with pytest.raises(MockRequestError) as exc_info:
                await client.write_text_file(content="x", path="/f", session_id="s")
            assert exc_info.value.code == -32601

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_write_no_workspace(self) -> None:
        async def _run() -> None:
            client = _make_client(allow_file_writes=True, workspace_root=None)
            with pytest.raises(MockRequestError) as exc_info:
                await client.write_text_file(content="x", path="/f", session_id="s")
            assert exc_info.value.code == -32601

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_write_outside_workspace(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        outside = tmp_path / "outside.txt"

        async def _run() -> None:
            client = _make_client(allow_file_writes=True, workspace_root=str(workspace))
            with pytest.raises(MockRequestError) as exc_info:
                await client.write_text_file(
                    content="x", path=str(outside), session_id="s"
                )
            assert exc_info.value.code == -32600

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_write_prefix_sibling_directory_blocked(self, tmp_path: Path) -> None:
        """A sibling directory sharing a prefix must not bypass containment."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        evil = tmp_path / "ws-evil"
        evil.mkdir()

        async def _run() -> None:
            client = _make_client(allow_file_writes=True, workspace_root=str(workspace))
            target = str(evil / "payload.txt")
            with pytest.raises(MockRequestError) as exc_info:
                await client.write_text_file(content="x", path=target, session_id="s")
            assert exc_info.value.code == -32600

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())

    def test_write_success(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        target = workspace / "output.txt"

        async def _run() -> None:
            client = _make_client(allow_file_writes=True, workspace_root=str(workspace))
            result = await client.write_text_file(
                content="written!", path=str(target), session_id="s"
            )
            assert isinstance(result, MockWriteTextFileResponse)

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())
        assert target.read_text() == "written!"

    def test_write_creates_parents(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        target = workspace / "sub" / "dir" / "file.txt"

        async def _run() -> None:
            client = _make_client(allow_file_writes=True, workspace_root=str(workspace))
            await client.write_text_file(
                content="deep", path=str(target), session_id="s"
            )

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())
        assert target.read_text() == "deep"

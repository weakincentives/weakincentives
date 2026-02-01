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

"""Tests for OpenCode ACP client."""

from __future__ import annotations

import asyncio

from weakincentives.adapters.opencode_acp.client import (
    OpenCodeACPClient,
    SessionAccumulator,
    ToolCallUpdate,
)
from weakincentives.adapters.opencode_acp.config import OpenCodeACPClientConfig


class TestToolCallUpdate:
    def test_defaults(self) -> None:
        update = ToolCallUpdate(
            tool_use_id="call-123",
            tool_name="my_tool",
        )
        assert update.tool_use_id == "call-123"
        assert update.tool_name == "my_tool"
        assert update.params is None
        assert update.status == "pending"
        assert update.result is None
        assert update.mcp_server_name is None

    def test_with_all_fields(self) -> None:
        update = ToolCallUpdate(
            tool_use_id="call-456",
            tool_name="another_tool",
            params={"key": "value"},
            status="completed",
            result="Success",
            mcp_server_name="wink",
        )
        assert update.tool_use_id == "call-456"
        assert update.tool_name == "another_tool"
        assert update.params == {"key": "value"}
        assert update.status == "completed"
        assert update.result == "Success"
        assert update.mcp_server_name == "wink"


class TestSessionAccumulator:
    def test_initial_state(self) -> None:
        acc = SessionAccumulator()
        assert acc.agent_messages == []
        assert acc.tool_calls == {}
        assert acc.thoughts == []
        assert acc.final_text == ""

    def test_handle_agent_message_chunk(self) -> None:
        acc = SessionAccumulator()
        acc.handle_agent_message_chunk("Hello, ")
        acc.handle_agent_message_chunk("world!")
        assert acc.agent_messages == ["Hello, ", "world!"]
        assert acc.final_text == "Hello, world!"

    def test_handle_thought_chunk_disabled_by_default(self) -> None:
        acc = SessionAccumulator()
        acc.handle_thought_chunk("Thinking...")
        # Thoughts not captured when _emit_thoughts is False
        assert acc.thoughts == []

    def test_handle_thought_chunk_enabled(self) -> None:
        acc = SessionAccumulator()
        acc._emit_thoughts = True
        acc.handle_thought_chunk("Thinking about X...")
        acc.handle_thought_chunk("Now considering Y...")
        assert acc.thoughts == ["Thinking about X...", "Now considering Y..."]

    def test_final_text_with_thoughts(self) -> None:
        acc = SessionAccumulator()
        acc._emit_thoughts = True
        acc.handle_thought_chunk("I'm thinking...")
        acc.handle_agent_message_chunk("The answer is 42.")

        result = acc.final_text_with_thoughts
        assert "I'm thinking..." in result
        assert "The answer is 42." in result

    def test_handle_tool_call(self) -> None:
        acc = SessionAccumulator()
        acc.handle_tool_call(
            tool_use_id="call-123",
            tool_name="my_tool",
            params={"arg": "value"},
            mcp_server_name="wink",
        )

        assert "call-123" in acc.tool_calls
        call = acc.tool_calls["call-123"]
        assert call.tool_name == "my_tool"
        assert call.params == {"arg": "value"}
        assert call.status == "running"
        assert call.mcp_server_name == "wink"

    def test_handle_tool_call_update_existing(self) -> None:
        acc = SessionAccumulator()
        acc.handle_tool_call("call-123", "my_tool")
        acc.handle_tool_call_update("call-123", "completed", "Success!")

        call = acc.tool_calls["call-123"]
        assert call.status == "completed"
        assert call.result == "Success!"

    def test_handle_tool_call_update_new(self) -> None:
        acc = SessionAccumulator()
        # Update without prior tool_call notification
        acc.handle_tool_call_update("call-456", "failed", "Error occurred")

        assert "call-456" in acc.tool_calls
        call = acc.tool_calls["call-456"]
        assert call.tool_name == "unknown"
        assert call.status == "failed"
        assert call.result == "Error occurred"

    def test_completed_tool_calls(self) -> None:
        acc = SessionAccumulator()
        acc.handle_tool_call("call-1", "tool_a")
        acc.handle_tool_call("call-2", "tool_b")
        acc.handle_tool_call("call-3", "tool_c")

        acc.handle_tool_call_update("call-1", "completed", "Done")
        acc.handle_tool_call_update("call-2", "failed", "Error")
        # call-3 remains running

        completed = acc.completed_tool_calls()
        assert len(completed) == 2
        tool_ids = {c.tool_use_id for c in completed}
        assert tool_ids == {"call-1", "call-2"}


class TestOpenCodeACPClientInit:
    def test_creates_with_default_config(self) -> None:
        config = OpenCodeACPClientConfig()
        client = OpenCodeACPClient(config)
        assert client.session_id is None
        assert isinstance(client.accumulator, SessionAccumulator)

    def test_creates_with_workspace(self) -> None:
        from weakincentives.adapters.opencode_acp import OpenCodeWorkspaceSection
        from weakincentives.runtime import InProcessDispatcher, Session

        config = OpenCodeACPClientConfig()
        dispatcher = InProcessDispatcher()
        session = Session(dispatcher=dispatcher)
        workspace = OpenCodeWorkspaceSection(session=session)

        try:
            client = OpenCodeACPClient(config, workspace=workspace)
            assert client.session_id is None
        finally:
            workspace.cleanup()


class TestOpenCodeACPClientPermissions:
    def test_request_permission_auto_approves(self) -> None:
        config = OpenCodeACPClientConfig(permission_mode="auto")
        client = OpenCodeACPClient(config)

        result = asyncio.run(
            client.request_permission(
                "session-123",
                {"type": "file_write", "path": "/some/path"},
            )
        )

        assert result["approved"] is True
        assert result["reason"] is None

    def test_request_permission_deny_rejects(self) -> None:
        config = OpenCodeACPClientConfig(permission_mode="deny")
        client = OpenCodeACPClient(config)

        result = asyncio.run(
            client.request_permission(
                "session-123",
                {"type": "file_write", "path": "/some/path"},
            )
        )

        assert result["approved"] is False
        assert "denied" in result["reason"].lower()

    def test_request_permission_prompt_rejects_non_interactive(self) -> None:
        config = OpenCodeACPClientConfig(permission_mode="prompt")
        client = OpenCodeACPClient(config)

        result = asyncio.run(
            client.request_permission(
                "session-123",
                {"type": "file_write", "path": "/some/path"},
            )
        )

        assert result["approved"] is False
        assert "not supported" in result["reason"].lower()


class TestOpenCodeACPClientFileOperations:
    def test_read_text_file_disabled_without_config(self) -> None:
        config = OpenCodeACPClientConfig(allow_file_reads=False)
        client = OpenCodeACPClient(config)

        result = asyncio.run(client.read_text_file("session-123", "/some/file.txt"))

        assert "error" in result
        assert "not enabled" in result["error"].lower()

    def test_read_text_file_disabled_without_workspace(self) -> None:
        config = OpenCodeACPClientConfig(allow_file_reads=True)
        # No workspace provided
        client = OpenCodeACPClient(config)

        result = asyncio.run(client.read_text_file("session-123", "/some/file.txt"))

        assert "error" in result
        assert "not enabled" in result["error"].lower()

    def test_read_text_file_with_workspace(self) -> None:
        from weakincentives.adapters.opencode_acp import OpenCodeWorkspaceSection
        from weakincentives.runtime import InProcessDispatcher, Session

        config = OpenCodeACPClientConfig(allow_file_reads=True)
        dispatcher = InProcessDispatcher()
        session = Session(dispatcher=dispatcher)
        workspace = OpenCodeWorkspaceSection(session=session)

        try:
            # Write a test file to the workspace
            test_content = "Hello, world!"
            (workspace.temp_dir / "test.txt").write_text(test_content)

            client = OpenCodeACPClient(config, workspace=workspace)
            result = asyncio.run(client.read_text_file("session-123", "test.txt"))

            assert "content" in result
            assert result["content"] == test_content
        finally:
            workspace.cleanup()

    def test_read_text_file_rejects_path_outside_workspace(self) -> None:
        from weakincentives.adapters.opencode_acp import OpenCodeWorkspaceSection
        from weakincentives.runtime import InProcessDispatcher, Session

        config = OpenCodeACPClientConfig(allow_file_reads=True)
        dispatcher = InProcessDispatcher()
        session = Session(dispatcher=dispatcher)
        workspace = OpenCodeWorkspaceSection(session=session)

        try:
            client = OpenCodeACPClient(config, workspace=workspace)
            result = asyncio.run(client.read_text_file("session-123", "/etc/passwd"))

            assert "error" in result
            assert "outside workspace" in result["error"].lower()
        finally:
            workspace.cleanup()

    def test_write_text_file_disabled_without_config(self) -> None:
        config = OpenCodeACPClientConfig(allow_file_writes=False)
        client = OpenCodeACPClient(config)

        result = asyncio.run(
            client.write_text_file("session-123", "/some/file.txt", "content")
        )

        assert "error" in result
        assert "not enabled" in result["error"].lower()

    def test_write_text_file_with_workspace(self) -> None:
        from weakincentives.adapters.opencode_acp import OpenCodeWorkspaceSection
        from weakincentives.runtime import InProcessDispatcher, Session

        config = OpenCodeACPClientConfig(allow_file_writes=True)
        dispatcher = InProcessDispatcher()
        session = Session(dispatcher=dispatcher)
        workspace = OpenCodeWorkspaceSection(session=session)

        try:
            client = OpenCodeACPClient(config, workspace=workspace)
            result = asyncio.run(
                client.write_text_file("session-123", "output.txt", "New content")
            )

            assert result.get("success") is True
            # Verify file was written
            assert (workspace.temp_dir / "output.txt").read_text() == "New content"
        finally:
            workspace.cleanup()

    def test_write_text_file_creates_parent_dirs(self) -> None:
        from weakincentives.adapters.opencode_acp import OpenCodeWorkspaceSection
        from weakincentives.runtime import InProcessDispatcher, Session

        config = OpenCodeACPClientConfig(allow_file_writes=True)
        dispatcher = InProcessDispatcher()
        session = Session(dispatcher=dispatcher)
        workspace = OpenCodeWorkspaceSection(session=session)

        try:
            client = OpenCodeACPClient(config, workspace=workspace)
            result = asyncio.run(
                client.write_text_file(
                    "session-123", "subdir/nested/file.txt", "Nested content"
                )
            )

            assert result.get("success") is True
            assert (
                workspace.temp_dir / "subdir" / "nested" / "file.txt"
            ).read_text() == "Nested content"
        finally:
            workspace.cleanup()


class TestOpenCodeACPClientTerminal:
    def test_create_terminal_disabled(self) -> None:
        config = OpenCodeACPClientConfig(allow_terminal=False)
        client = OpenCodeACPClient(config)

        result = asyncio.run(client.create_terminal("session-123", {}))

        assert "error" in result
        assert "not supported" in result["error"].lower()

    def test_create_terminal_not_implemented(self) -> None:
        config = OpenCodeACPClientConfig(allow_terminal=True)
        client = OpenCodeACPClient(config)

        result = asyncio.run(client.create_terminal("session-123", {}))

        assert "error" in result
        assert "not implemented" in result["error"].lower()


class TestOpenCodeACPClientSessionUpdate:
    def test_handles_agent_message_chunk(self) -> None:
        config = OpenCodeACPClientConfig()
        client = OpenCodeACPClient(config)

        asyncio.run(
            client.session_update(
                "session-123",
                {"type": "agent_message_chunk", "text": "Hello!"},
            )
        )

        assert client.accumulator.final_text == "Hello!"

    def test_handles_thought_update(self) -> None:
        config = OpenCodeACPClientConfig()
        client = OpenCodeACPClient(config)
        client.accumulator._emit_thoughts = True

        asyncio.run(
            client.session_update(
                "session-123",
                {"type": "thought", "text": "Analyzing..."},
            )
        )

        assert "Analyzing..." in client.accumulator.thoughts

    def test_handles_tool_call_update(self) -> None:
        config = OpenCodeACPClientConfig()
        client = OpenCodeACPClient(config)

        asyncio.run(
            client.session_update(
                "session-123",
                {
                    "type": "tool_call",
                    "tool_use_id": "call-123",
                    "tool_name": "my_tool",
                    "params": {"arg": "value"},
                },
            )
        )

        assert "call-123" in client.accumulator.tool_calls
        call = client.accumulator.tool_calls["call-123"]
        assert call.tool_name == "my_tool"
        assert call.params == {"arg": "value"}

    def test_handles_tool_call_status_update(self) -> None:
        config = OpenCodeACPClientConfig()
        client = OpenCodeACPClient(config)

        # First, register the tool call
        asyncio.run(
            client.session_update(
                "session-123",
                {
                    "type": "tool_call",
                    "tool_use_id": "call-123",
                    "tool_name": "my_tool",
                },
            )
        )

        # Then update its status
        asyncio.run(
            client.session_update(
                "session-123",
                {
                    "type": "tool_call_update",
                    "tool_use_id": "call-123",
                    "status": "completed",
                    "result": "Success",
                },
            )
        )

        call = client.accumulator.tool_calls["call-123"]
        assert call.status == "completed"
        assert call.result == "Success"

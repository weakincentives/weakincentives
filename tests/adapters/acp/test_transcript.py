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

"""Tests for ACP transcript bridge."""

from __future__ import annotations

from unittest.mock import MagicMock

from weakincentives.adapters.acp._transcript import (
    ACPTranscriptBridge,
    _extract_chunk_text,
)

from .conftest import (
    MockAgentMessageChunk,
    MockAgentThoughtChunk,
    MockToolCallProgress,
    MockToolCallStart,
)


def _make_bridge() -> tuple[ACPTranscriptBridge, MagicMock]:
    """Create a bridge with a mock emitter."""
    emitter = MagicMock()
    bridge = ACPTranscriptBridge(emitter)
    return bridge, emitter


class TestACPTranscriptBridge:
    def test_emitter_property(self) -> None:
        bridge, emitter = _make_bridge()
        assert bridge.emitter is emitter

    def test_on_user_message(self) -> None:
        bridge, emitter = _make_bridge()
        bridge.on_user_message("Hello world")
        emitter.emit.assert_called_once_with(
            "user_message",
            detail={"text": "Hello world"},
        )

    def test_on_user_message_truncates(self) -> None:
        bridge, emitter = _make_bridge()
        bridge.on_user_message("x" * 1000)
        call_args = emitter.emit.call_args
        assert len(call_args.kwargs["detail"]["text"]) == 500


class TestChunkConsolidation:
    """Test that streaming deltas are consolidated into single entries."""

    def test_message_chunks_consolidated_on_flush(self) -> None:
        bridge, emitter = _make_bridge()
        bridge.on_update(MockAgentMessageChunk(content="Hello"))
        bridge.on_update(MockAgentMessageChunk(content=" world"))
        # Not emitted yet (still buffering)
        emitter.emit.assert_not_called()
        bridge.flush()
        emitter.emit.assert_called_once_with(
            "assistant_message",
            detail={"text": "Hello world"},
        )

    def test_thought_chunks_consolidated_on_flush(self) -> None:
        bridge, emitter = _make_bridge()
        bridge.on_update(MockAgentThoughtChunk(content="Let me"))
        bridge.on_update(MockAgentThoughtChunk(content=" think"))
        emitter.emit.assert_not_called()
        bridge.flush()
        emitter.emit.assert_called_once_with(
            "thinking",
            detail={"text": "Let me think"},
        )

    def test_message_flushed_on_tool_start(self) -> None:
        bridge, emitter = _make_bridge()
        bridge.on_update(MockAgentMessageChunk(content="I will read the file."))
        bridge.on_update(MockToolCallStart(tool_call_id="tc-1", title="read"))
        # flush + tool_use = 2 calls
        assert emitter.emit.call_count == 2
        first_call = emitter.emit.call_args_list[0]
        assert first_call.args[0] == "assistant_message"
        assert first_call.kwargs["detail"]["text"] == "I will read the file."
        second_call = emitter.emit.call_args_list[1]
        assert second_call.args[0] == "tool_use"

    def test_message_flushed_on_tool_progress(self) -> None:
        bridge, emitter = _make_bridge()
        bridge.on_update(MockAgentMessageChunk(content="text"))
        # Terminal status emits both the flushed text and the tool_result.
        bridge.on_update(
            MockToolCallProgress(tool_call_id="tc-1", title="bash", status="completed")
        )
        assert emitter.emit.call_count == 2
        assert emitter.emit.call_args_list[0].args[0] == "assistant_message"
        assert emitter.emit.call_args_list[1].args[0] == "tool_result"

    def test_message_flushed_on_non_terminal_progress(self) -> None:
        bridge, emitter = _make_bridge()
        bridge.on_update(MockAgentMessageChunk(content="text"))
        # Non-terminal progress flushes the text but does not emit tool_result yet.
        bridge.on_update(
            MockToolCallProgress(
                tool_call_id="tc-1", title="bash", status="in_progress"
            )
        )
        assert emitter.emit.call_count == 1
        assert emitter.emit.call_args_list[0].args[0] == "assistant_message"

    def test_thought_flushed_when_message_starts(self) -> None:
        bridge, emitter = _make_bridge()
        bridge.on_update(MockAgentThoughtChunk(content="thinking"))
        bridge.on_update(MockAgentMessageChunk(content="response"))
        # thought was flushed when message type started
        assert emitter.emit.call_count == 1
        assert emitter.emit.call_args_list[0].args[0] == "thinking"
        # message still buffered
        bridge.flush()
        assert emitter.emit.call_count == 2
        assert emitter.emit.call_args_list[1].args[0] == "assistant_message"

    def test_message_flushed_when_thought_starts(self) -> None:
        bridge, emitter = _make_bridge()
        bridge.on_update(MockAgentMessageChunk(content="partial"))
        bridge.on_update(MockAgentThoughtChunk(content="thinking"))
        assert emitter.emit.call_count == 1
        assert emitter.emit.call_args_list[0].args[0] == "assistant_message"
        bridge.flush()
        assert emitter.emit.call_count == 2

    def test_consolidated_text_truncated(self) -> None:
        bridge, emitter = _make_bridge()
        for _ in range(100):
            bridge.on_update(MockAgentMessageChunk(content="x" * 10))
        bridge.flush()
        call_args = emitter.emit.call_args
        assert len(call_args.kwargs["detail"]["text"]) == 500

    def test_flush_noop_when_empty(self) -> None:
        bridge, emitter = _make_bridge()
        bridge.flush()
        emitter.emit.assert_not_called()

    def test_flush_skips_empty_text(self) -> None:
        bridge, emitter = _make_bridge()
        bridge.on_update(MockAgentMessageChunk(content=""))
        bridge.on_update(MockAgentMessageChunk(content=""))
        bridge.flush()
        # No emit since accumulated text is empty
        emitter.emit.assert_not_called()

    def test_double_flush_is_noop(self) -> None:
        bridge, emitter = _make_bridge()
        bridge.on_update(MockAgentMessageChunk(content="text"))
        bridge.flush()
        assert emitter.emit.call_count == 1
        bridge.flush()
        assert emitter.emit.call_count == 1

    def test_unknown_update_type_ignored(self) -> None:
        bridge, emitter = _make_bridge()
        bridge.on_update(MagicMock())
        emitter.emit.assert_not_called()

    def test_full_conversation_flow(self) -> None:
        """Simulate: think → tool_use → tool_result → message."""
        bridge, emitter = _make_bridge()
        # Thinking phase (buffered)
        bridge.on_update(MockAgentThoughtChunk(content="I need to "))
        bridge.on_update(MockAgentThoughtChunk(content="read files."))
        # Tool call flushes thinking, then emits tool_use
        bridge.on_update(MockToolCallStart(tool_call_id="tc-1", title="read"))
        # Tool progress emits tool_result (no buffered text to flush)
        bridge.on_update(
            MockToolCallProgress(
                tool_call_id="tc-1",
                title="read",
                status="completed",
                raw_output="data",
            )
        )
        # Response phase (buffered)
        bridge.on_update(MockAgentMessageChunk(content="Here is "))
        bridge.on_update(MockAgentMessageChunk(content="the result."))
        bridge.flush()

        calls = emitter.emit.call_args_list
        assert len(calls) == 4
        assert calls[0].args[0] == "thinking"
        assert calls[0].kwargs["detail"]["text"] == "I need to read files."
        assert calls[1].args[0] == "tool_use"
        assert calls[2].args[0] == "tool_result"
        assert calls[3].args[0] == "assistant_message"
        assert calls[3].kwargs["detail"]["text"] == "Here is the result."


class TestToolEvents:
    def test_tool_start(self) -> None:
        bridge, emitter = _make_bridge()
        update = MockToolCallStart(tool_call_id="tc-1", title="bash")
        bridge.on_update(update)
        emitter.emit.assert_called_once_with(
            "tool_use",
            detail={"tool_name": "bash", "tool_call_id": "tc-1"},
        )

    def test_tool_start_with_input(self) -> None:
        bridge, emitter = _make_bridge()
        update = MockToolCallStart(
            tool_call_id="tc-1",
            title="read",
            raw_input={"path": "/tmp/foo.py"},
        )
        bridge.on_update(update)
        detail = emitter.emit.call_args.kwargs["detail"]
        assert detail["tool_name"] == "read"
        assert detail["tool_call_id"] == "tc-1"
        assert "path" in detail["input"]

    def test_tool_progress_emits_on_completed(self) -> None:
        bridge, emitter = _make_bridge()
        bridge.on_update(
            MockToolCallProgress(
                tool_call_id="tc-1",
                title="bash",
                status="completed",
                raw_output="done",
            )
        )
        emitter.emit.assert_called_once()
        detail = emitter.emit.call_args.kwargs["detail"]
        assert detail["tool_name"] == "bash"
        assert detail["tool_call_id"] == "tc-1"
        assert detail["status"] == "completed"
        assert detail["output"] == "done"

    def test_tool_progress_emits_on_failed(self) -> None:
        bridge, emitter = _make_bridge()
        bridge.on_update(
            MockToolCallProgress(
                tool_call_id="tc-1",
                title="bash",
                status="failed",
                raw_output="error",
            )
        )
        emitter.emit.assert_called_once()
        assert emitter.emit.call_args.kwargs["detail"]["status"] == "failed"

    def test_tool_progress_buffers_non_terminal(self) -> None:
        """Non-terminal (in_progress) updates do not emit immediately."""
        bridge, emitter = _make_bridge()
        bridge.on_update(
            MockToolCallProgress(
                tool_call_id="tc-1",
                title="bash",
                status="in_progress",
                raw_input={"cmd": "ls"},
            )
        )
        emitter.emit.assert_not_called()

    def test_tool_progress_consolidates_deltas(self) -> None:
        """Multiple progress updates merge into one emitted tool_result."""
        bridge, emitter = _make_bridge()
        # First: in_progress with input
        bridge.on_update(
            MockToolCallProgress(
                tool_call_id="tc-1",
                title="read",
                status="in_progress",
                raw_input={"filePath": "/tmp/foo.py"},
            )
        )
        emitter.emit.assert_not_called()
        # Second: completed with output
        bridge.on_update(
            MockToolCallProgress(
                tool_call_id="tc-1",
                title="/tmp/foo.py",  # OpenCode overwrites title
                status="completed",
                raw_output="file contents",
            )
        )
        emitter.emit.assert_called_once()
        detail = emitter.emit.call_args.kwargs["detail"]
        # Original tool name preserved (not overwritten to file path).
        assert detail["tool_name"] == "read"
        assert detail["tool_call_id"] == "tc-1"
        assert detail["status"] == "completed"
        assert "filePath" in detail["input"]
        assert detail["output"] == "file contents"

    def test_tool_progress_preserves_first_tool_name(self) -> None:
        """First non-empty title sticks even if later updates change it."""
        bridge, emitter = _make_bridge()
        bridge.on_update(
            MockToolCallProgress(
                tool_call_id="tc-1", title="glob", status="in_progress"
            )
        )
        bridge.on_update(
            MockToolCallProgress(
                tool_call_id="tc-1", title="/some/path", status="completed"
            )
        )
        detail = emitter.emit.call_args.kwargs["detail"]
        assert detail["tool_name"] == "glob"

    def test_tool_progress_fills_empty_tool_name(self) -> None:
        """If first update has empty title, a later update can fill it."""
        bridge, emitter = _make_bridge()
        bridge.on_update(
            MockToolCallProgress(tool_call_id="tc-1", title="", status="in_progress")
        )
        bridge.on_update(
            MockToolCallProgress(tool_call_id="tc-1", title="read", status="completed")
        )
        detail = emitter.emit.call_args.kwargs["detail"]
        assert detail["tool_name"] == "read"

    def test_tool_progress_empty_status_preserves_existing(self) -> None:
        """An update with empty status does not overwrite existing status."""
        bridge, emitter = _make_bridge()
        bridge.on_update(
            MockToolCallProgress(
                tool_call_id="tc-1", title="bash", status="in_progress"
            )
        )
        # Update with empty status and some input.
        bridge.on_update(
            MockToolCallProgress(
                tool_call_id="tc-1",
                title="",
                status="",
                raw_input={"cmd": "ls"},
            )
        )
        # Complete to emit.
        bridge.on_update(
            MockToolCallProgress(tool_call_id="tc-1", title="", status="completed")
        )
        detail = emitter.emit.call_args.kwargs["detail"]
        assert detail["status"] == "completed"
        assert "cmd" in detail["input"]

    def test_tool_progress_with_input_on_completed(self) -> None:
        bridge, emitter = _make_bridge()
        bridge.on_update(
            MockToolCallProgress(
                tool_call_id="tc-1",
                title="bash",
                status="completed",
                raw_input={"cmd": "ls"},
                raw_output="file.py",
            )
        )
        detail = emitter.emit.call_args.kwargs["detail"]
        assert "cmd" in detail["input"]
        assert detail["output"] == "file.py"

    def test_tool_progress_truncates_output(self) -> None:
        bridge, emitter = _make_bridge()
        bridge.on_update(
            MockToolCallProgress(
                tool_call_id="tc-1",
                title="bash",
                status="completed",
                raw_output="x" * 1000,
            )
        )
        detail = emitter.emit.call_args.kwargs["detail"]
        assert len(detail["output"]) == 500

    def test_tool_progress_non_string_output(self) -> None:
        bridge, emitter = _make_bridge()
        update = MockToolCallProgress(
            tool_call_id="tc-1", title="bash", status="completed"
        )
        object.__setattr__(update, "raw_output", 42)
        bridge.on_update(update)
        detail = emitter.emit.call_args.kwargs["detail"]
        assert detail["output"] == "42"

    def test_tool_progress_none_output(self) -> None:
        bridge, emitter = _make_bridge()
        bridge.on_update(
            MockToolCallProgress(tool_call_id="tc-1", title="bash", status="completed")
        )
        detail = emitter.emit.call_args.kwargs["detail"]
        assert detail["output"] == ""

    def test_flush_emits_pending_tool_buffers(self) -> None:
        """flush() emits non-terminal tool buffers (e.g. session ended early)."""
        bridge, emitter = _make_bridge()
        bridge.on_update(
            MockToolCallProgress(
                tool_call_id="tc-1",
                title="bash",
                status="in_progress",
                raw_input={"cmd": "ls"},
            )
        )
        emitter.emit.assert_not_called()
        bridge.flush()
        emitter.emit.assert_called_once()
        detail = emitter.emit.call_args.kwargs["detail"]
        assert detail["tool_name"] == "bash"
        assert detail["status"] == "in_progress"

    def test_flush_clears_tool_buffers(self) -> None:
        """After flush, pending tool buffers are cleared."""
        bridge, emitter = _make_bridge()
        bridge.on_update(
            MockToolCallProgress(
                tool_call_id="tc-1", title="bash", status="in_progress"
            )
        )
        bridge.flush()
        assert emitter.emit.call_count == 1
        bridge.flush()
        # No additional emit.
        assert emitter.emit.call_count == 1

    def test_multiple_concurrent_tools(self) -> None:
        """Interleaved progress from different tool_call_ids consolidates each."""
        bridge, emitter = _make_bridge()
        bridge.on_update(
            MockToolCallProgress(
                tool_call_id="tc-1",
                title="read",
                status="in_progress",
                raw_input={"path": "a.py"},
            )
        )
        bridge.on_update(
            MockToolCallProgress(
                tool_call_id="tc-2",
                title="glob",
                status="in_progress",
                raw_input={"pattern": "*.py"},
            )
        )
        emitter.emit.assert_not_called()
        # tc-1 completes first
        bridge.on_update(
            MockToolCallProgress(
                tool_call_id="tc-1",
                title="/tmp/a.py",
                status="completed",
                raw_output="contents-a",
            )
        )
        assert emitter.emit.call_count == 1
        d1 = emitter.emit.call_args.kwargs["detail"]
        assert d1["tool_name"] == "read"
        assert d1["output"] == "contents-a"
        # tc-2 completes next
        bridge.on_update(
            MockToolCallProgress(
                tool_call_id="tc-2",
                title="/tmp",
                status="completed",
                raw_output="b.py\nc.py",
            )
        )
        assert emitter.emit.call_count == 2
        d2 = emitter.emit.call_args.kwargs["detail"]
        assert d2["tool_name"] == "glob"
        assert d2["output"] == "b.py\nc.py"


class TestExtractChunkText:
    def test_string_content(self) -> None:
        chunk = MagicMock(content="hello")
        assert _extract_chunk_text(chunk) == "hello"

    def test_object_with_text_attr(self) -> None:
        inner = MagicMock(text="inner text")
        inner.__class__ = type("ContentBlock", (), {"text": "inner text"})
        chunk = MagicMock(content=inner)
        assert _extract_chunk_text(chunk) == "inner text"

    def test_non_string_no_text_attr(self) -> None:
        inner = 42
        chunk = MagicMock(content=inner)
        assert _extract_chunk_text(chunk) == "42"

    def test_falsy_content(self) -> None:
        chunk = MagicMock(content=None)
        assert _extract_chunk_text(chunk) == ""

    def test_no_content_attr(self) -> None:
        chunk = object()
        assert _extract_chunk_text(chunk) == ""

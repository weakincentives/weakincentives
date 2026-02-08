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

"""Tests for CodexTranscriptBridge."""

from __future__ import annotations

from unittest.mock import patch

from weakincentives.adapters.codex_app_server._transcript import (
    CodexTranscriptBridge,
)
from weakincentives.runtime.transcript import TranscriptEmitter


class TestCodexTranscriptBridge:
    """Tests for CodexTranscriptBridge."""

    def _make_bridge(self, *, emit_raw: bool = True) -> CodexTranscriptBridge:
        emitter = TranscriptEmitter(
            prompt_name="test-prompt",
            adapter="codex_app_server",
            emit_raw=emit_raw,
        )
        return CodexTranscriptBridge(emitter)

    def test_on_user_message(self) -> None:
        """on_user_message emits user_message entry."""
        bridge = self._make_bridge()
        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            bridge.on_user_message("Hello, world")
            entry_calls = [
                c
                for c in mock_logger.debug.call_args_list
                if c[1].get("event") == "transcript.entry"
            ]
            assert len(entry_calls) == 1
            ctx = entry_calls[0][1]["context"]
            assert ctx["entry_type"] == "user_message"
            assert ctx["detail"]["text"] == "Hello, world"

    def test_on_notification_streaming_deltas_suppressed(self) -> None:
        """Streaming delta notifications are not emitted."""
        bridge = self._make_bridge()
        delta_methods = [
            "item/agentMessage/delta",
            "item/reasoning/summaryTextDelta",
            "item/reasoning/summaryPartAdded",
            "item/commandExecution/outputDelta",
            "item/commandExecution/terminalInteraction",
        ]
        for method in delta_methods:
            with patch("weakincentives.runtime.transcript._logger") as mock_logger:
                bridge.on_notification(method, {"delta": "chunk"})
                entry_calls = [
                    c
                    for c in mock_logger.debug.call_args_list
                    if c[1].get("event") == "transcript.entry"
                ]
                assert len(entry_calls) == 0, f"Delta {method} should be suppressed"

    def test_on_notification_item_completed_agent_message(self) -> None:
        """item/completed with agentMessage emits assistant_message."""
        bridge = self._make_bridge()
        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            bridge.on_notification(
                "item/completed",
                {"item": {"type": "agentMessage", "text": "Hi there"}},
            )
            entry_calls = [
                c
                for c in mock_logger.debug.call_args_list
                if c[1].get("event") == "transcript.entry"
            ]
            assert len(entry_calls) == 1
            ctx = entry_calls[0][1]["context"]
            assert ctx["entry_type"] == "assistant_message"

    def test_on_notification_item_completed_tool(self) -> None:
        """item/completed with tool type emits tool_result."""
        for tool_type in ("commandExecution", "fileChange", "mcpToolCall", "webSearch"):
            bridge = self._make_bridge()
            with patch("weakincentives.runtime.transcript._logger") as mock_logger:
                bridge.on_notification(
                    "item/completed",
                    {"item": {"type": tool_type, "result": "ok"}},
                )
                entry_calls = [
                    c
                    for c in mock_logger.debug.call_args_list
                    if c[1].get("event") == "transcript.entry"
                ]
                assert len(entry_calls) == 1, f"Failed for {tool_type}"
                ctx = entry_calls[0][1]["context"]
                assert ctx["entry_type"] == "tool_result"

    def test_on_notification_item_completed_compaction(self) -> None:
        """item/completed with contextCompaction emits system_event."""
        bridge = self._make_bridge()
        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            bridge.on_notification(
                "item/completed",
                {"item": {"type": "contextCompaction"}},
            )
            entry_calls = [
                c
                for c in mock_logger.debug.call_args_list
                if c[1].get("event") == "transcript.entry"
            ]
            assert len(entry_calls) == 1
            ctx = entry_calls[0][1]["context"]
            assert ctx["entry_type"] == "system_event"
            assert ctx["detail"]["subtype"] == "compaction"

    def test_on_notification_item_completed_unknown_type(self) -> None:
        """item/completed with unrecognized type emits nothing."""
        bridge = self._make_bridge()
        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            bridge.on_notification(
                "item/completed",
                {"item": {"type": "unknownItemType"}},
            )
            entry_calls = [
                c
                for c in mock_logger.debug.call_args_list
                if c[1].get("event") == "transcript.entry"
            ]
            assert len(entry_calls) == 0

    def test_on_notification_item_started_tool(self) -> None:
        """item/started with tool type emits tool_use."""
        bridge = self._make_bridge()
        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            bridge.on_notification(
                "item/started",
                {"item": {"type": "commandExecution", "command": "ls"}},
            )
            entry_calls = [
                c
                for c in mock_logger.debug.call_args_list
                if c[1].get("event") == "transcript.entry"
            ]
            assert len(entry_calls) == 1
            ctx = entry_calls[0][1]["context"]
            assert ctx["entry_type"] == "tool_use"

    def test_on_notification_item_started_non_tool_ignored(self) -> None:
        """item/started with non-tool type is not emitted."""
        bridge = self._make_bridge()
        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            bridge.on_notification(
                "item/started",
                {"item": {"type": "agentMessage"}},
            )
            entry_calls = [
                c
                for c in mock_logger.debug.call_args_list
                if c[1].get("event") == "transcript.entry"
            ]
            assert len(entry_calls) == 0

    def test_on_notification_reasoning_completed(self) -> None:
        """item/reasoning/completed emits thinking entry."""
        bridge = self._make_bridge()
        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            bridge.on_notification(
                "item/reasoning/completed",
                {"text": "thinking..."},
            )
            entry_calls = [
                c
                for c in mock_logger.debug.call_args_list
                if c[1].get("event") == "transcript.entry"
            ]
            assert len(entry_calls) == 1
            ctx = entry_calls[0][1]["context"]
            assert ctx["entry_type"] == "thinking"

    def test_on_notification_token_usage(self) -> None:
        """thread/tokenUsage/updated emits token_usage entry."""
        bridge = self._make_bridge()
        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            bridge.on_notification(
                "thread/tokenUsage/updated",
                {"input_tokens": 100, "output_tokens": 50},
            )
            entry_calls = [
                c
                for c in mock_logger.debug.call_args_list
                if c[1].get("event") == "transcript.entry"
            ]
            assert len(entry_calls) == 1
            ctx = entry_calls[0][1]["context"]
            assert ctx["entry_type"] == "token_usage"

    def test_on_notification_turn_started(self) -> None:
        """turn/started emits system_event with turn_started subtype."""
        bridge = self._make_bridge()
        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            bridge.on_notification(
                "turn/started",
                {"turn": {"id": "0", "status": "inProgress"}},
            )
            entry_calls = [
                c
                for c in mock_logger.debug.call_args_list
                if c[1].get("event") == "transcript.entry"
            ]
            assert len(entry_calls) == 1
            ctx = entry_calls[0][1]["context"]
            assert ctx["entry_type"] == "system_event"
            assert ctx["detail"]["subtype"] == "turn_started"

    def test_on_notification_turn_completed_failed(self) -> None:
        """turn/completed with failed status emits error entry."""
        bridge = self._make_bridge()
        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            bridge.on_notification(
                "turn/completed",
                {"turn": {"status": "failed", "error": "timeout"}},
            )
            entry_calls = [
                c
                for c in mock_logger.debug.call_args_list
                if c[1].get("event") == "transcript.entry"
            ]
            assert len(entry_calls) == 1
            ctx = entry_calls[0][1]["context"]
            assert ctx["entry_type"] == "error"

    def test_on_notification_turn_completed_success_no_emit(self) -> None:
        """turn/completed with success status does not emit."""
        bridge = self._make_bridge()
        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            bridge.on_notification(
                "turn/completed",
                {"turn": {"status": "completed"}},
            )
            entry_calls = [
                c
                for c in mock_logger.debug.call_args_list
                if c[1].get("event") == "transcript.entry"
            ]
            assert len(entry_calls) == 0

    def test_on_notification_unknown_item_method(self) -> None:
        """Unrecognized item/* methods emit unknown entry type."""
        bridge = self._make_bridge()
        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            bridge.on_notification("item/future/feature", {"data": "value"})
            entry_calls = [
                c
                for c in mock_logger.debug.call_args_list
                if c[1].get("event") == "transcript.entry"
            ]
            assert len(entry_calls) == 1
            ctx = entry_calls[0][1]["context"]
            assert ctx["entry_type"] == "unknown"

    def test_on_notification_unrelated_method_ignored(self) -> None:
        """Methods not starting with item/ or turn/ are ignored."""
        bridge = self._make_bridge()
        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            bridge.on_notification("system/ping", {})
            entry_calls = [
                c
                for c in mock_logger.debug.call_args_list
                if c[1].get("event") == "transcript.entry"
            ]
            assert len(entry_calls) == 0

    def test_on_tool_call(self) -> None:
        """on_tool_call emits tool_use entry."""
        bridge = self._make_bridge()
        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            bridge.on_tool_call({"tool": "my_tool", "arguments": {"x": 1}})
            entry_calls = [
                c
                for c in mock_logger.debug.call_args_list
                if c[1].get("event") == "transcript.entry"
            ]
            assert len(entry_calls) == 1
            ctx = entry_calls[0][1]["context"]
            assert ctx["entry_type"] == "tool_use"
            assert "raw" in ctx

    def test_on_tool_result(self) -> None:
        """on_tool_result emits tool_result entry."""
        bridge = self._make_bridge()
        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            bridge.on_tool_result(
                {"tool": "my_tool"},
                {"success": True, "contentItems": []},
            )
            entry_calls = [
                c
                for c in mock_logger.debug.call_args_list
                if c[1].get("event") == "transcript.entry"
            ]
            assert len(entry_calls) == 1
            ctx = entry_calls[0][1]["context"]
            assert ctx["entry_type"] == "tool_result"
            assert ctx["detail"]["result"]["success"] is True

    def test_raw_included_when_enabled(self) -> None:
        """raw field is present when emit_raw=True."""
        bridge = self._make_bridge(emit_raw=True)
        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            bridge.on_notification("thread/tokenUsage/updated", {"input_tokens": 10})
            ctx = mock_logger.debug.call_args[1]["context"]
            assert "raw" in ctx

    def test_raw_omitted_when_disabled(self) -> None:
        """raw field is absent when emit_raw=False."""
        bridge = self._make_bridge(emit_raw=False)
        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            bridge.on_notification("thread/tokenUsage/updated", {"input_tokens": 10})
            ctx = mock_logger.debug.call_args[1]["context"]
            assert "raw" not in ctx

    def test_sequence_numbers_across_calls(self) -> None:
        """Sequence numbers increment across multiple calls."""
        bridge = self._make_bridge()
        with patch("weakincentives.runtime.transcript._logger") as mock_logger:
            bridge.on_user_message("hi")
            bridge.on_notification(
                "item/completed",
                {"item": {"type": "agentMessage", "text": "reply"}},
            )
            bridge.on_tool_call({"tool": "t"})

            entry_calls = [
                c
                for c in mock_logger.debug.call_args_list
                if c[1].get("event") == "transcript.entry"
            ]
            assert len(entry_calls) == 3
            seqs = [c[1]["context"]["sequence_number"] for c in entry_calls]
            assert seqs == [1, 2, 3]

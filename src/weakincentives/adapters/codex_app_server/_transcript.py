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

"""Transcript bridge for the Codex App Server adapter.

Maps Codex stdio JSON-RPC notifications to canonical transcript entry types
and emits them via :class:`~weakincentives.runtime.transcript.TranscriptEmitter`.

See ``specs/TRANSCRIPT.md`` for the full specification.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, ClassVar

from ...runtime.transcript import TranscriptEmitter

# Type alias for notification handler methods.
_HandlerFn = Callable[["CodexTranscriptBridge", dict[str, Any], str], None]

__all__ = ["CodexTranscriptBridge"]

# Item types that correspond to tool operations.
_TOOL_ITEM_TYPES: frozenset[str] = frozenset(
    {"commandExecution", "fileChange", "mcpToolCall", "webSearch"}
)

# Streaming delta methods to suppress — their content is consolidated
# in the corresponding ``item/completed`` or ``item/reasoning/completed``
# notifications.
_DELTA_METHODS: frozenset[str] = frozenset(
    {
        "item/agentMessage/delta",
        "item/reasoning/summaryTextDelta",
        "item/reasoning/summaryPartAdded",
        "item/commandExecution/outputDelta",
        "item/commandExecution/terminalInteraction",
    }
)


class CodexTranscriptBridge:
    """Bridge between Codex notifications and the transcript emitter.

    Called from ``_process_notification()`` and ``_handle_server_request()``
    in the Codex adapter to emit transcript entries for each notification.
    """

    def __init__(self, emitter: TranscriptEmitter) -> None:
        super().__init__()
        self._emitter = emitter

    @property
    def emitter(self) -> TranscriptEmitter:
        """Return the underlying emitter."""
        return self._emitter

    def on_user_message(self, text: str) -> None:
        """Emit ``user_message`` entry before turn start.

        Args:
            text: The rendered prompt text sent to Codex.
        """
        self._emitter.emit(
            "user_message",
            detail={"text": text},
        )

    def on_notification(self, method: str, params: dict[str, Any]) -> None:
        """Map a Codex notification to a transcript entry and emit.

        Args:
            method: The JSON-RPC notification method name.
            params: Notification parameters.
        """
        # Suppress streaming delta notifications — the completed events
        # carry the full consolidated content.
        if method in _DELTA_METHODS:
            return

        raw = json.dumps({"method": method, "params": params})
        handler = self._NOTIFICATION_HANDLERS.get(method)
        if handler is not None:
            handler(self, params, raw)
        elif method.startswith(("item/", "turn/")):
            self._emitter.emit(
                "unknown",
                detail={"notification": params},
                raw=raw,
            )

    def _handle_item_started(self, params: dict[str, Any], raw: str) -> None:
        item: dict[str, Any] = params.get("item", {})
        if item.get("type", "") in _TOOL_ITEM_TYPES:
            self._emitter.emit("tool_use", detail={"notification": params}, raw=raw)

    def _handle_reasoning_completed(self, params: dict[str, Any], raw: str) -> None:
        self._emitter.emit("thinking", detail={"notification": params}, raw=raw)

    def _handle_token_usage(self, params: dict[str, Any], raw: str) -> None:
        self._emitter.emit("token_usage", detail={"notification": params}, raw=raw)

    def _handle_turn_completed(self, params: dict[str, Any], raw: str) -> None:
        turn: dict[str, Any] = params.get("turn", {})
        if turn.get("status") == "failed":
            self._emitter.emit("error", detail={"notification": params}, raw=raw)

    def _handle_turn_started(self, params: dict[str, Any], raw: str) -> None:
        self._emitter.emit(
            "system_event",
            detail={"notification": params, "subtype": "turn_started"},
            raw=raw,
        )

    _NOTIFICATION_HANDLERS: ClassVar[dict[str, _HandlerFn]] = {
        "turn/started": _handle_turn_started,
        "item/started": _handle_item_started,
        "item/completed": lambda self, p, r: self._handle_item_completed(p, r),
        "item/reasoning/completed": _handle_reasoning_completed,
        "thread/tokenUsage/updated": _handle_token_usage,
        "turn/completed": _handle_turn_completed,
    }

    def _handle_item_completed(self, params: dict[str, Any], raw: str) -> None:
        """Handle ``item/completed`` notifications."""
        item: dict[str, Any] = params.get("item", {})
        item_type = item.get("type", "")

        if item_type == "agentMessage":
            self._emitter.emit(
                "assistant_message",
                detail={"notification": params},
                raw=raw,
            )
        elif item_type in _TOOL_ITEM_TYPES:
            self._emitter.emit(
                "tool_result",
                detail={"notification": params},
                raw=raw,
            )
        elif item_type == "contextCompaction":
            self._emitter.emit(
                "system_event",
                detail={"notification": params, "subtype": "compaction"},
                raw=raw,
            )

    def on_tool_call(self, params: dict[str, Any]) -> None:
        """Emit ``tool_use`` for a WINK bridged tool call.

        Args:
            params: The ``item/tool/call`` server request parameters.
        """
        raw = json.dumps({"method": "item/tool/call", "params": params})
        self._emitter.emit(
            "tool_use",
            detail={"notification": params},
            raw=raw,
        )

    def on_tool_result(self, params: dict[str, Any], result: dict[str, Any]) -> None:
        """Emit ``tool_result`` after a WINK bridged tool completes.

        Args:
            params: Original ``item/tool/call`` request parameters.
            result: The tool execution result sent back to Codex.
        """
        raw = json.dumps(
            {"method": "item/tool/result", "params": params, "result": result}
        )
        self._emitter.emit(
            "tool_result",
            detail={"notification": params, "result": result},
            raw=raw,
        )

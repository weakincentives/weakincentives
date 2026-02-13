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

"""ACP Client implementation for the generic ACP adapter."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...runtime.logging import StructuredLogger, get_logger

if TYPE_CHECKING:
    from ._transcript import ACPTranscriptBridge
    from .config import ACPClientConfig

__all__ = ["ACPClient"]

logger: StructuredLogger = get_logger(__name__, context={"component": "acp_client"})


class ACPClient:
    """ACP Client that handles session updates and permission requests.

    Implements the ``acp.interfaces.Client`` protocol.  All methods are async.
    Uses lazy imports for all ACP types so the ``acp`` package is only required
    at call time.
    """

    def __init__(
        self,
        config: ACPClientConfig,
        *,
        workspace_root: str | None = None,
    ) -> None:
        self._config = config
        self._workspace_root = workspace_root
        self._accumulator: Any = None  # Lazy: SessionAccumulator
        self._transcript_bridge: ACPTranscriptBridge | None = None
        self._last_update_time: float = 0.0
        self._message_chunks: list[Any] = []
        self._thought_chunks: list[Any] = []
        # tool_call_id -> {title, status, output}
        self._tool_call_tracker: dict[str, dict[str, str]] = {}
        self._tool_counter: int = 0
        self._current_tool_id: str = ""

    def set_accumulator(self, accumulator: Any) -> None:
        """Set the SessionAccumulator instance."""
        self._accumulator = accumulator

    def set_transcript_bridge(self, bridge: ACPTranscriptBridge) -> None:
        """Set the transcript bridge for emitting transcript entries."""
        self._transcript_bridge = bridge

    @property
    def last_update_time(self) -> float:
        """Monotonic time of the last session update."""
        return self._last_update_time

    @property
    def message_chunks(self) -> list[Any]:
        """Accumulated AgentMessageChunk updates."""
        return self._message_chunks

    @property
    def thought_chunks(self) -> list[Any]:
        """Accumulated AgentThoughtChunk updates."""
        return self._thought_chunks

    @property
    def tool_call_tracker(self) -> dict[str, dict[str, str]]:
        """Mapping of tool_call_id to tool data dict."""
        return self._tool_call_tracker

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        """Handle ``session/update`` notification."""
        from acp.schema import SessionNotification

        self._last_update_time = time.monotonic()

        # Wrap in SessionNotification for accumulator
        notification = SessionNotification(sessionId=session_id, update=update)
        if self._accumulator is not None:
            self._accumulator.apply(notification)

        # Emit transcript entry
        if self._transcript_bridge is not None:
            self._transcript_bridge.on_update(update)

        self._track_update(update)

    def _track_update(self, update: Any) -> None:
        """Track message, thought, and tool call updates."""
        update_type = type(update).__name__
        if update_type == "AgentMessageChunk":
            self._message_chunks.append(update)
        elif update_type == "AgentThoughtChunk":
            self._thought_chunks.append(update)
        elif update_type == "ToolCallStart":
            tc_id = self._resolve_tool_id(getattr(update, "id", ""), is_start=True)
            self._tool_call_tracker[tc_id] = {
                "title": getattr(update, "title", ""),
                "status": "started",
                "output": "",
            }
        elif update_type == "ToolCallProgress":
            self._track_tool_progress(update)

    def _resolve_tool_id(self, raw_id: str, *, is_start: bool) -> str:
        """Resolve a tool call ID, generating a synthetic one if empty.

        Some ACP agents (e.g. OpenCode) send empty ``id`` on every
        ToolCallStart / ToolCallProgress.  We assign monotonic synthetic
        IDs so each tool call gets its own tracker entry.
        """
        if raw_id:
            if is_start:
                self._current_tool_id = raw_id
            return raw_id
        if is_start:
            self._tool_counter += 1
            synthetic = f"_tc_{self._tool_counter}"
            self._current_tool_id = synthetic
            return synthetic
        return self._current_tool_id or f"_tc_{self._tool_counter}"

    def _track_tool_progress(self, update: Any) -> None:
        """Track a ToolCallProgress update."""
        tc_id = self._resolve_tool_id(getattr(update, "id", ""), is_start=False)
        status = getattr(update, "status", "")
        output = getattr(update, "output", "")
        if tc_id in self._tool_call_tracker:
            self._tool_call_tracker[tc_id]["status"] = status
            if output:
                self._tool_call_tracker[tc_id]["output"] = str(output)[:1000]
        else:
            self._tool_call_tracker[tc_id] = {
                "title": getattr(update, "title", ""),
                "status": status,
                "output": str(output)[:1000] if output else "",
            }

    async def request_permission(
        self, options: Any, session_id: str, tool_call: Any, **kwargs: Any
    ) -> Any:
        """Handle ``session/request_permission``."""
        from acp.schema import PermissionRequestResponse

        mode = self._config.permission_mode
        if mode == "auto":
            return PermissionRequestResponse(approved=True)
        # "deny" and "prompt" both deny (prompt must not block)
        reason = (
            "Permission denied: non-interactive mode"
            if mode == "prompt"
            else "Permission denied by policy"
        )
        return PermissionRequestResponse(approved=False, reason=reason)

    async def read_text_file(self, path: str, session_id: str, **kwargs: Any) -> Any:
        """Handle ``fs/read_text_file`` request."""
        from acp import RequestError
        from acp.schema import ReadTextFileResponse

        if not self._config.allow_file_reads:
            raise RequestError.method_not_found("read_text_file")

        if self._workspace_root is None:
            raise RequestError.method_not_found("read_text_file")

        resolved = Path(path).resolve()
        workspace = Path(self._workspace_root).resolve()
        if not str(resolved).startswith(str(workspace)):
            raise RequestError("Path outside workspace", code=-32600)

        content = resolved.read_text()
        return ReadTextFileResponse(content=content)

    async def write_text_file(
        self, content: str, path: str, session_id: str, **kwargs: Any
    ) -> Any:
        """Handle ``fs/write_text_file`` request."""
        from acp import RequestError
        from acp.schema import WriteTextFileResponse

        if not self._config.allow_file_writes:
            raise RequestError.method_not_found("write_text_file")

        if self._workspace_root is None:
            raise RequestError.method_not_found("write_text_file")

        resolved = Path(path).resolve()
        workspace = Path(self._workspace_root).resolve()
        if not str(resolved).startswith(str(workspace)):
            raise RequestError("Path outside workspace", code=-32600)

        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)
        return WriteTextFileResponse()

    async def create_terminal(
        self, command: str, session_id: str, **kwargs: Any
    ) -> Any:
        """Raise method_not_found --- terminal not supported."""
        from acp import RequestError

        raise RequestError.method_not_found("create_terminal")

    async def terminal_output(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> Any:
        """Raise method_not_found --- terminal not supported."""
        from acp import RequestError

        raise RequestError.method_not_found("terminal_output")

    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> Any:
        """Raise method_not_found --- terminal not supported."""
        from acp import RequestError

        raise RequestError.method_not_found("release_terminal")

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> Any:
        """Raise method_not_found --- terminal not supported."""
        from acp import RequestError

        raise RequestError.method_not_found("wait_for_terminal_exit")

    async def kill_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> Any:
        """Raise method_not_found --- terminal not supported."""
        from acp import RequestError

        raise RequestError.method_not_found("kill_terminal")

    async def ext_method(self, method: str, params: Any, **kwargs: Any) -> Any:
        """Extension method --- not supported."""
        from acp import RequestError

        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: Any, **kwargs: Any) -> None:
        """Extension notification --- silently ignored."""

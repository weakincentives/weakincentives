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

"""Transcript bridge for the ACP adapter.

Maps ACP session updates to canonical transcript entry types and emits
them via :class:`~weakincentives.runtime.transcript.TranscriptEmitter`.

Streaming deltas (``AgentMessageChunk``, ``AgentThoughtChunk``) are
**consolidated** into single transcript entries.  Unlike Codex (which
sends ``item/completed`` with accumulated text), ACP only streams
deltas.  The bridge buffers text and flushes when the update type
changes or :meth:`flush` is called at session end.

See ``specs/TRANSCRIPT.md`` for the full specification.
"""

from __future__ import annotations

from ...runtime.transcript import TranscriptEmitter

__all__ = ["ACPTranscriptBridge"]

# Maximum text length stored in a single transcript detail entry.
_MAX_TEXT = 500


def _truncate(value: object, limit: int) -> str:
    """Convert *value* to string and truncate to *limit* characters."""
    text = str(value)
    return text[:limit] if len(text) > limit else text


def _extract_chunk_text(chunk: object) -> str:
    """Extract text from an ACP content chunk."""
    raw = getattr(chunk, "content", "")
    if isinstance(raw, str):
        return raw
    text = getattr(raw, "text", None)
    if isinstance(text, str):
        return text
    return str(raw) if raw else ""


class ACPTranscriptBridge:
    """Bridge between ACP session updates and the transcript emitter.

    Called from ``ACPClient.session_update()`` for each update notification.

    Streaming text (``AgentMessageChunk`` / ``AgentThoughtChunk``) is
    buffered and emitted as a single consolidated entry when the stream
    type changes or :meth:`flush` is called.
    """

    def __init__(self, emitter: TranscriptEmitter) -> None:
        super().__init__()
        self._emitter = emitter
        # Buffering state: _buf_type is "message" | "thought" | None
        self._buf_type: str | None = None
        self._buf_parts: list[str] = []

    @property
    def emitter(self) -> TranscriptEmitter:
        """Return the underlying emitter."""
        return self._emitter

    def on_user_message(self, text: str) -> None:
        """Emit ``user_message`` entry for the prompt text."""
        self._emitter.emit(
            "user_message",
            detail={"text": text[:_MAX_TEXT]},
        )

    def flush(self) -> None:
        """Emit any buffered message/thought text."""
        if self._buf_type is None:
            return
        text = "".join(self._buf_parts)
        if text:
            entry_type = (
                "assistant_message" if self._buf_type == "message" else "thinking"
            )
            self._emitter.emit(entry_type, detail={"text": text[:_MAX_TEXT]})
        self._buf_type = None
        self._buf_parts.clear()

    def on_update(self, update: object) -> None:
        """Map an ACP session update to a transcript entry and emit."""
        update_type = type(update).__name__

        if update_type == "AgentMessageChunk":
            self._buffer_chunk("message", update)
        elif update_type == "AgentThoughtChunk":
            self._buffer_chunk("thought", update)
        elif update_type == "ToolCallStart":
            self.flush()
            self._handle_tool_start(update)
        elif update_type == "ToolCallProgress":
            self.flush()
            self._handle_tool_progress(update)

    def _buffer_chunk(self, buf_type: str, update: object) -> None:
        """Buffer a streaming chunk, flushing if the type changed."""
        if self._buf_type is not None and self._buf_type != buf_type:
            self.flush()
        self._buf_type = buf_type
        self._buf_parts.append(_extract_chunk_text(update))

    def _handle_tool_start(self, update: object) -> None:
        title = getattr(update, "title", "")
        tool_id = getattr(update, "tool_call_id", "")
        raw_input = getattr(update, "raw_input", None)

        detail: dict[str, object] = {
            "tool_name": title,
            "tool_call_id": tool_id,
        }
        if raw_input is not None:
            detail["input"] = _truncate(raw_input, _MAX_TEXT)

        self._emitter.emit("tool_use", detail=detail)

    def _handle_tool_progress(self, update: object) -> None:
        status = getattr(update, "status", "")
        tool_id = getattr(update, "tool_call_id", "")
        title = getattr(update, "title", "")
        raw_output = getattr(update, "raw_output", None)

        detail: dict[str, object] = {
            "tool_name": title,
            "tool_call_id": tool_id,
            "status": str(status) if status else "",
            "output": _truncate(raw_output, _MAX_TEXT)
            if raw_output is not None
            else "",
        }

        # Include input if delivered on the update (some agents send it here).
        raw_input = getattr(update, "raw_input", None)
        if raw_input is not None:
            detail["input"] = _truncate(raw_input, _MAX_TEXT)

        self._emitter.emit("tool_result", detail=detail)

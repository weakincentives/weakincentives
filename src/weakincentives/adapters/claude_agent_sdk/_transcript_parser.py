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

"""Entry transformation for Claude Agent SDK transcripts.

Pure functions that parse JSONL transcript entries and emit them through
a :class:`~weakincentives.runtime.transcript.TranscriptEmitter`.  Separated
from file-monitoring concerns in :mod:`._transcript_collector`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ...runtime.transcript import TranscriptEmitter

if TYPE_CHECKING:
    from ._transcript_collector import _TailerState

__all__ = [
    "emit_assistant_split",
    "emit_entry",
    "emit_user_tool_result_split",
]

# Mapping from Claude SDK transcript entry ``type`` to canonical entry type.
_SDK_TYPE_MAP: dict[str, str] = {
    "user": "user_message",
    "assistant": "assistant_message",
    "tool_result": "tool_result",
    "thinking": "thinking",
    "summary": "system_event",
    "system": "system_event",
}


def _has_block_type(content: list[Any], block_type: str) -> bool:
    """Check if a content block list contains any blocks of the given type."""
    return any(isinstance(b, dict) and b.get("type") == block_type for b in content)


def emit_entry(
    emitter: TranscriptEmitter,
    tailer: _TailerState,
    line: str,
) -> None:
    """Parse and emit a single transcript JSONL line.

    For assistant messages containing tool_use content blocks, splits into
    separate entries: one assistant_message (text-only blocks) plus one
    tool_use entry per tool_use block.  Similarly, user messages with
    tool_result content blocks are split into separate tool_result entries.
    This makes the Claude SDK adapter structurally isomorphic with the
    Codex adapter for tool analysis.

    Args:
        emitter: Transcript emitter.
        tailer: Tailer state for source and entry counting.
        line: Raw JSONL line.
    """
    # Always parse to determine canonical entry type
    try:
        entry = json.loads(line)
        sdk_type = entry.get("type", "unknown")
        entry_type = _SDK_TYPE_MAP.get(sdk_type, "unknown")
        detail: dict[str, Any] = {"sdk_entry": entry}

        # Add compaction subtype for summary entries
        if sdk_type == "summary":
            detail["subtype"] = "compaction"
    except json.JSONDecodeError:
        entry_type = "unknown"
        detail = {"parse_error": "Invalid JSON"}
        tailer.entry_count += 1
        emitter.emit(
            entry_type,
            source=tailer.source,
            detail=detail,
            raw=line,
        )
        return

    # Split messages that mix content with tool blocks
    if _try_emit_split(emitter, tailer, entry, entry_type, line):
        return

    tailer.entry_count += 1
    emitter.emit(
        entry_type,
        source=tailer.source,
        detail=detail,
        raw=line,
    )


def _try_emit_split(
    emitter: TranscriptEmitter,
    tailer: _TailerState,
    entry: dict[str, Any],
    entry_type: str,
    line: str,
) -> bool:
    """Try to split an entry with mixed content blocks. Returns True if split."""
    message = entry.get("message")
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False

    if entry_type == "assistant_message" and _has_block_type(content, "tool_use"):
        emit_assistant_split(emitter, tailer, entry, line)
        return True

    if entry_type == "user_message" and _has_block_type(content, "tool_result"):
        emit_user_tool_result_split(emitter, tailer, entry, line)
        return True

    return False


def emit_assistant_split(
    emitter: TranscriptEmitter,
    tailer: _TailerState,
    entry: dict[str, Any],
    line: str,
) -> None:
    """Split an assistant message into text and tool_use entries.

    Args:
        emitter: Transcript emitter.
        tailer: Tailer state for entry counting.
        entry: Parsed SDK entry dict.
        line: Raw JSONL line (attached only to the assistant_message).
    """
    content = entry["message"]["content"]

    # Partition content blocks
    text_blocks = [
        b for b in content if not (isinstance(b, dict) and b.get("type") == "tool_use")
    ]
    tool_blocks = [
        b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"
    ]

    # Emit assistant_message with text-only blocks (if any)
    if text_blocks:
        text_entry = {
            **entry,
            "message": {**entry["message"], "content": text_blocks},
        }
        tailer.entry_count += 1
        emitter.emit(
            "assistant_message",
            source=tailer.source,
            detail={"sdk_entry": text_entry},
            raw=line,
        )

    # Record tool_use_id → tool_name mappings for later tool_result correlation
    for block in tool_blocks:
        block_id = block.get("id")
        block_name = block.get("name")
        if block_id and block_name:
            tailer.tool_names[block_id] = block_name

    # Emit one tool_use entry per tool_use block
    for block in tool_blocks:
        tailer.entry_count += 1
        emitter.emit(
            "tool_use",
            source=tailer.source,
            detail={"sdk_entry": block},
            raw=None,
        )


def emit_user_tool_result_split(
    emitter: TranscriptEmitter,
    tailer: _TailerState,
    entry: dict[str, Any],
    line: str,
) -> None:
    """Split a user message into non-tool_result user_message + tool_result entries.

    The Claude SDK wraps tool results as user messages with ``tool_result``
    content blocks.  Splitting these out as ``tool_result`` entries enables
    the ``transcript_tools`` view to correlate calls with results via
    ``tool_use_id``.

    Args:
        emitter: Transcript emitter.
        tailer: Tailer state for entry counting.
        entry: Parsed SDK entry dict.
        line: Raw JSONL line (attached only to the user_message if any
              non-tool_result blocks remain).
    """
    content = entry["message"]["content"]

    # Partition content blocks
    other_blocks = [
        b
        for b in content
        if not (isinstance(b, dict) and b.get("type") == "tool_result")
    ]
    result_blocks = [
        b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"
    ]

    # Emit user_message with non-tool_result blocks (if any)
    if other_blocks:
        other_entry = {
            **entry,
            "message": {**entry["message"], "content": other_blocks},
        }
        tailer.entry_count += 1
        emitter.emit(
            "user_message",
            source=tailer.source,
            detail={"sdk_entry": other_entry},
            raw=line,
        )

    # Emit one tool_result entry per tool_result block
    for block in result_blocks:
        tool_use_id = block.get("tool_use_id", "")
        tool_name = tailer.tool_names.get(tool_use_id, "")
        detail: dict[str, Any] = {"sdk_entry": block}
        if tool_name:
            detail["tool_name"] = tool_name
        tailer.entry_count += 1
        emitter.emit(
            "tool_result",
            source=tailer.source,
            detail=detail,
            raw=None if other_blocks else line,
        )

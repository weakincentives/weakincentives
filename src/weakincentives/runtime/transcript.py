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

"""Transcript capture for conversation logging.

This module provides the TranscriptEntry dataclass for capturing full
conversation transcripts in a human-readable format. Transcripts are
stored in a LOG slice for append-only semantics.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, cast
from uuid import UUID, uuid4

__all__ = [
    "TranscriptEntry",
    "TranscriptRole",
    "convert_claude_transcript_entry",
    "format_transcript",
]


TranscriptRole = Literal["system", "user", "assistant", "tool"]
"""Role of a transcript entry: system, user, assistant, or tool."""


@dataclass(frozen=True, slots=True)
class TranscriptEntry:
    """A single message in the conversation transcript.

    Represents one turn in the conversation, whether from the system,
    user, assistant, or a tool response. Designed for human readability.

    Attributes:
        role: Who produced this message: system, user, assistant, or tool.
        content: The text content of the message.
        created_at: When this entry was recorded.
        tool_name: For tool role: the name of the tool that was called.
        tool_call_id: For tool role: the unique identifier for this invocation.
        tool_input: For assistant role with tool calls: the structured arguments.
        metadata: Provider-specific metadata (model name, token counts, etc.).
        entry_id: Unique identifier for this transcript entry.
        source: Source identifier. 'primary' for main agent, 'subagent:<id>'
            for nested agents.
        sequence: Monotonically increasing sequence number within the source.
    """

    role: TranscriptRole
    content: str
    created_at: datetime
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_input: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    entry_id: UUID = field(default_factory=uuid4)
    source: str = "primary"
    sequence: int = 0


def convert_claude_transcript_entry(
    raw: dict[str, Any],
    *,
    source: str = "primary",
    sequence: int = 0,
) -> TranscriptEntry:
    """Convert a Claude Code transcript entry to TranscriptEntry.

    Claude Code JSONL format includes fields like:
    - type: "user" | "assistant" | "tool_use" | "tool_result"
    - content: message content (text or structured)
    - timestamp: ISO 8601 timestamp
    - tool_name: for tool entries
    - tool_use_id: correlation ID

    Args:
        raw: Raw dict from Claude Code transcript JSONL.
        source: Source identifier for multi-agent scenarios.
        sequence: Sequence number within the source.

    Returns:
        Converted TranscriptEntry instance.
    """
    entry_type = raw.get("type", "")
    timestamp_str = raw.get("timestamp", "")

    # Map Claude Code types to TranscriptRole
    role_map: dict[str, TranscriptRole] = {
        "user": "user",
        "assistant": "assistant",
        "tool_use": "assistant",  # Tool calls come from assistant
        "tool_result": "tool",
        "system": "system",
    }
    role: TranscriptRole = role_map.get(entry_type, "assistant")

    # Extract content
    content_raw: object = raw.get("content", "")
    content: str
    if isinstance(content_raw, dict):
        content = json.dumps(content_raw, indent=2)
    elif isinstance(content_raw, list):
        # Claude sometimes uses content blocks
        parts: list[str] = []
        content_blocks = cast(list[object], content_raw)
        for block in content_blocks:
            if isinstance(block, dict) and "text" in block:
                text_val = cast(object, block["text"])
                parts.append(str(text_val))
            elif isinstance(block, str):
                parts.append(block)
        content = "\n".join(parts)
    else:
        content = str(content_raw) if content_raw else ""

    # Parse timestamp
    try:
        created_at = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        created_at = datetime.now(UTC)

    return TranscriptEntry(
        role=role,
        content=content,
        created_at=created_at,
        tool_name=raw.get("tool_name"),
        tool_call_id=raw.get("tool_use_id"),
        tool_input=raw.get("input") if entry_type == "tool_use" else None,
        metadata={"raw_type": entry_type, "claude_session_id": raw.get("session_id")},
        source=source,
        sequence=sequence,
    )


def format_transcript(entries: Iterable[TranscriptEntry]) -> str:
    """Format transcript entries for human reading.

    Produces a readable output with timestamps, role indicators, and content.

    Args:
        entries: Iterable of TranscriptEntry objects.

    Returns:
        Human-readable formatted transcript string.

    Example output::

        [09:15:32] ⚙️ SYSTEM
        You are a helpful coding assistant...

        [09:15:35] 🤖 ASSISTANT
        I'll review the code...

        [09:15:36] 🔧 Read
        {"path": "src/main.py"}
    """
    lines: list[str] = []
    role_emoji: dict[str, str] = {
        "system": "⚙️",
        "user": "👤",
        "assistant": "🤖",
        "tool": "🔧",
    }

    for entry in entries:
        timestamp = entry.created_at.strftime("%H:%M:%S")
        source_prefix = f"[{entry.source}] " if entry.source != "primary" else ""

        if entry.role == "tool" and entry.tool_name:
            header = f"{source_prefix}[{timestamp}] 🔧 {entry.tool_name}"
        else:
            emoji = role_emoji.get(entry.role, "•")
            header = f"{source_prefix}[{timestamp}] {emoji} {entry.role.upper()}"

        lines.append(header)
        lines.append(entry.content)
        lines.append("")  # Blank line between entries

    return "\n".join(lines)

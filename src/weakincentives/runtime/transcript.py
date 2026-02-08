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

"""Transcript types and emitter for adapter-agnostic evaluation logging.

This module provides the shared ``TranscriptEmitter`` used by both the Claude
Agent SDK and Codex App Server adapters to emit transcript entries as
DEBUG-level structured log records.  It also provides ``TranscriptEntry`` and
``TranscriptSummary`` dataclasses for reconstructing typed transcripts from
captured log records.

See ``specs/TRANSCRIPT.md`` for the full specification.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .logging import StructuredLogger, get_logger

__all__ = [
    "CANONICAL_ENTRY_TYPES",
    "TranscriptEmitter",
    "TranscriptEntry",
    "TranscriptSummary",
    "reconstruct_transcript",
]

CANONICAL_ENTRY_TYPES: frozenset[str] = frozenset(
    {
        "user_message",
        "assistant_message",
        "tool_use",
        "tool_result",
        "thinking",
        "system_event",
        "token_usage",
        "error",
        "unknown",
    }
)
"""The set of canonical entry types shared across all adapters."""

_logger: StructuredLogger = get_logger(
    "weakincentives.runtime.transcript",
    context={"component": "transcript"},
)


@dataclass(slots=True, frozen=True)
class TranscriptEntry:
    """A single entry in a reconstructed transcript."""

    prompt_name: str
    adapter: str
    entry_type: str
    sequence_number: int
    source: str
    timestamp: datetime
    session_id: str | None = None
    detail: dict[str, Any] = field(default_factory=lambda: {})
    raw: str | None = None


@dataclass(slots=True, frozen=True)
class TranscriptSummary:
    """Summary statistics for a transcript."""

    total_entries: int
    entries_by_type: Mapping[str, int]
    entries_by_source: Mapping[str, int]
    sources: tuple[str, ...]
    adapter: str
    prompt_name: str
    first_timestamp: datetime | None
    last_timestamp: datetime | None


class TranscriptEmitter:
    """Emit transcript entries as structured DEBUG logs.

    Shared helper that both adapters use to construct and emit entries.
    Lives in the runtime layer so adapters don't duplicate envelope logic.
    Not a public API -- internal to the framework.
    """

    def __init__(
        self,
        *,
        prompt_name: str,
        adapter: str,
        session_id: str | None = None,
        emit_raw: bool = True,
    ) -> None:
        super().__init__()
        self._prompt_name = prompt_name
        self._adapter = adapter
        self._session_id = session_id
        self._emit_raw = emit_raw
        self._counters: dict[str, int] = {}
        self._type_counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def emit(
        self,
        entry_type: str,
        *,
        source: str = "main",
        detail: dict[str, Any] | None = None,
        raw: str | None = None,
    ) -> None:
        """Emit a transcript entry. Thread-safe, never raises."""
        try:
            with self._lock:
                seq = self._counters.get(source, 0) + 1
                self._counters[source] = seq
                self._type_counts[entry_type] = self._type_counts.get(entry_type, 0) + 1

            ts = datetime.now(UTC).isoformat()
            context: dict[str, Any] = {
                "prompt_name": self._prompt_name,
                "adapter": self._adapter,
                "entry_type": entry_type,
                "sequence_number": seq,
                "source": source,
                "timestamp": ts,
            }
            if self._session_id is not None:
                context["session_id"] = self._session_id
            if detail is not None:
                context["detail"] = detail
            if raw is not None and self._emit_raw:
                context["raw"] = raw

            _logger.debug(
                f"transcript entry: {entry_type}",
                event="transcript.entry",
                context=context,
            )
        except Exception:
            with contextlib.suppress(Exception):
                _logger.warning(
                    "failed to emit transcript entry",
                    event="transcript.error",
                    context={
                        "prompt_name": self._prompt_name,
                        "adapter": self._adapter,
                        "entry_type": entry_type,
                        "source": source,
                    },
                )

    def start(self) -> None:
        """Emit transcript.start event."""
        with contextlib.suppress(Exception):  # pragma: no cover - defensive
            _logger.debug(
                "transcript emission started",
                event="transcript.start",
                context={
                    "prompt_name": self._prompt_name,
                    "adapter": self._adapter,
                },
            )

    def stop(self) -> None:
        """Emit transcript.stop event with summary statistics."""
        with contextlib.suppress(Exception):  # pragma: no cover - defensive
            with self._lock:
                entries_by_source = dict(self._counters)
                entries_by_type = dict(self._type_counts)
                total = sum(entries_by_source.values())

            _logger.debug(
                "transcript emission stopped",
                event="transcript.stop",
                context={
                    "prompt_name": self._prompt_name,
                    "adapter": self._adapter,
                    "total_entries": total,
                    "entries_by_source": entries_by_source,
                    "entries_by_type": entries_by_type,
                },
            )

    @property
    def total_entries(self) -> int:
        """Total entries emitted across all sources."""
        with self._lock:
            return sum(self._counters.values())

    def source_count(self, source: str) -> int:
        """Number of entries emitted for a given source."""
        with self._lock:
            return self._counters.get(source, 0)


def reconstruct_transcript(
    records: Sequence[dict[str, object]],
) -> list[TranscriptEntry]:
    """Reconstruct a typed transcript from log records.

    Accepts a sequence of log record dicts (as captured by debug bundle
    ``logs/app.jsonl`` or ``transcript.jsonl``).  Each record must have
    ``event == "transcript.entry"`` and a ``context`` dict containing the
    common envelope keys.

    Records that lack required fields are silently skipped.

    Args:
        records: Sequence of log record dicts.

    Returns:
        Ordered list of ``TranscriptEntry`` instances.
    """
    entries: list[TranscriptEntry] = []
    for record in records:
        if record.get("event") != "transcript.entry":
            continue
        ctx_raw = record.get("context")
        if not isinstance(ctx_raw, dict):
            continue
        ctx: dict[str, object] = ctx_raw  # type: ignore[assignment]
        try:
            entry = _parse_transcript_record(ctx)
            entries.append(entry)
        except (KeyError, TypeError, ValueError):
            continue
    return entries


def _parse_transcript_record(ctx: dict[str, object]) -> TranscriptEntry:
    """Parse a single transcript record context into a TranscriptEntry.

    Raises KeyError/TypeError/ValueError on invalid data.
    """
    ts_raw = ctx.get("timestamp")
    ts_str = str(ts_raw) if ts_raw is not None else ""
    ts = datetime.fromisoformat(ts_str) if ts_str else datetime.now(UTC)

    prompt_name = str(ctx["prompt_name"])
    adapter = str(ctx["adapter"])
    entry_type = str(ctx["entry_type"])
    sequence_number = int(str(ctx["sequence_number"]))
    source = str(ctx["source"])

    sid_raw = ctx.get("session_id")
    session_id: str | None = str(sid_raw) if sid_raw is not None else None

    detail_raw = ctx.get("detail")
    if isinstance(detail_raw, dict):
        detail_typed: dict[str, object] = detail_raw  # type: ignore[assignment]
        detail: dict[str, Any] = {str(k): v for k, v in detail_typed.items()}
    else:
        detail = {}

    raw_raw = ctx.get("raw")
    raw: str | None = str(raw_raw) if raw_raw is not None else None

    return TranscriptEntry(
        prompt_name=prompt_name,
        adapter=adapter,
        entry_type=entry_type,
        sequence_number=sequence_number,
        source=source,
        timestamp=ts,
        session_id=session_id,
        detail=detail,
        raw=raw,
    )

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

"""Unified event-log for sessions.

A :class:`TranscriptListener` subscribes to a :class:`Session` and records
every published event into a flat, immutable, JSON-friendly stream of
:class:`TranscriptEntry` values. ``debug`` and ``evals`` build on this.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from typing import cast, final

from weakincentives.clock import SYSTEM_CLOCK, WallClock

__all__ = [
    "Transcript",
    "TranscriptEntry",
    "TranscriptListener",
]


def _serialize(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, tuple | list):
        sequence = cast("tuple[object, ...] | list[object]", value)
        return [_serialize(item) for item in sequence]
    if isinstance(value, dict):
        mapping = cast("dict[object, object]", value)
        items: dict[str, object] = {}
        for key, val in mapping.items():
            items[str(key)] = _serialize(val)
        return items
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        encoded: dict[str, object] = {}
        for f in fields(value):
            encoded[f.name] = _serialize(getattr(value, f.name))
        return encoded
    return repr(value)


def _payload_of(event: object) -> Mapping[str, object]:
    if is_dataclass(event) and not isinstance(event, type):
        raw = asdict(event)
        return {key: _serialize(val) for key, val in raw.items()}
    return {"value": _serialize(event)}


@final
@dataclass(frozen=True, slots=True)
class TranscriptEntry:
    """Single immutable line in a transcript."""

    timestamp: datetime
    kind: str
    payload: Mapping[str, object] = field(default_factory=lambda: {})


@final
class Transcript:
    """Append-only collection of :class:`TranscriptEntry` values."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.RLock()
        self._entries: list[TranscriptEntry] = []

    def append(self, entry: TranscriptEntry) -> None:
        with self._lock:
            self._entries.append(entry)

    def entries(self) -> tuple[TranscriptEntry, ...]:
        with self._lock:
            return tuple(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


@final
class TranscriptListener:
    """Subscribe to a :class:`Session` and record every published event.

    Implements :class:`weakincentives.core.EventListener`. Pass the same
    instance to multiple sessions to merge their transcripts.
    """

    def __init__(
        self,
        *,
        transcript: Transcript | None = None,
        clock: WallClock = SYSTEM_CLOCK,
    ) -> None:
        super().__init__()
        self._transcript = transcript if transcript is not None else Transcript()
        self._clock = clock

    @property
    def transcript(self) -> Transcript:
        return self._transcript

    def on_event(self, event: object) -> None:
        kind = type(event).__qualname__
        entry = TranscriptEntry(
            timestamp=self._clock.utcnow().astimezone(UTC),
            kind=kind,
            payload=_payload_of(event),
        )
        self._transcript.append(entry)

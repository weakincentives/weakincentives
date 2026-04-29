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

"""Tests for ``weakincentives.transcript``."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from weakincentives.clock import FakeClock
from weakincentives.core import EventListener, ToolInvoked
from weakincentives.transcript import (
    Transcript,
    TranscriptEntry,
    TranscriptListener,
)


@dataclass(frozen=True)
class CustomEvent:
    label: str
    when: datetime
    payload: dict[str, int]


def test_listener_satisfies_event_listener_protocol() -> None:
    assert isinstance(TranscriptListener(), EventListener)


def test_listener_records_dataclass_event() -> None:
    listener = TranscriptListener(clock=FakeClock())
    listener.on_event(ToolInvoked(name="t", success=True, message="ok"))
    entries = listener.transcript.entries()
    assert len(entries) == 1
    assert entries[0].kind == "ToolInvoked"
    assert entries[0].payload["name"] == "t"
    assert entries[0].payload["success"] is True


def test_listener_records_non_dataclass_event() -> None:
    listener = TranscriptListener(clock=FakeClock())
    listener.on_event("plain string")
    entry = listener.transcript.entries()[0]
    assert entry.kind == "str"
    assert entry.payload == {"value": "plain string"}


def test_serialize_handles_nested_collections() -> None:
    listener = TranscriptListener(clock=FakeClock())
    event = CustomEvent(
        label="x",
        when=datetime.fromisoformat("2025-01-01T00:00:00+00:00"),
        payload={"a": 1},
    )
    listener.on_event(event)
    entry = listener.transcript.entries()[0]
    assert entry.kind == "CustomEvent"
    assert entry.payload["label"] == "x"
    assert entry.payload["when"] == "2025-01-01T00:00:00+00:00"
    assert entry.payload["payload"] == {"a": 1}


def test_serialize_falls_back_to_repr_for_unknown_types() -> None:
    listener = TranscriptListener(clock=FakeClock())
    listener.on_event(complex(1, 2))
    entry = listener.transcript.entries()[0]
    assert entry.payload == {"value": "(1+2j)"}


def test_serialize_handles_lists_and_tuples() -> None:
    @dataclass(frozen=True)
    class Wrap:
        items: tuple[int, ...]

    listener = TranscriptListener(clock=FakeClock())
    listener.on_event(Wrap(items=(1, 2, 3)))
    entry = listener.transcript.entries()[0]
    assert entry.payload["items"] == [1, 2, 3]


def test_transcript_append_query_clear_len() -> None:
    transcript = Transcript()
    assert len(transcript) == 0
    entry = TranscriptEntry(
        timestamp=datetime.fromisoformat("2025-01-01T00:00:00+00:00"),
        kind="x",
        payload={"k": "v"},
    )
    transcript.append(entry)
    assert len(transcript) == 1
    assert transcript.entries() == (entry,)
    transcript.clear()
    assert transcript.entries() == ()


def test_serialize_handles_nested_dataclass() -> None:
    @dataclass(frozen=True)
    class Leaf:
        value: int

    @dataclass(frozen=True)
    class Root:
        leaf: Leaf

    listener = TranscriptListener(clock=FakeClock())
    listener.on_event(Root(leaf=Leaf(value=7)))
    payload = listener.transcript.entries()[0].payload
    assert payload["leaf"] == {"value": 7}


def test_serialize_handles_dataclass_inside_non_dataclass_event() -> None:
    @dataclass(frozen=True)
    class Item:
        v: int

    listener = TranscriptListener(clock=FakeClock())
    # The event itself is a tuple (not a dataclass), but contains a
    # dataclass instance — exercises the `_serialize` recursion branch.
    listener.on_event((Item(v=1), Item(v=2)))
    payload = listener.transcript.entries()[0].payload
    assert payload["value"] == [{"v": 1}, {"v": 2}]


def test_listener_shares_transcript_across_sessions() -> None:
    shared = Transcript()
    listener_a = TranscriptListener(transcript=shared, clock=FakeClock())
    listener_b = TranscriptListener(transcript=shared, clock=FakeClock())
    listener_a.on_event(ToolInvoked(name="a", success=True, message=""))
    listener_b.on_event(ToolInvoked(name="b", success=True, message=""))
    names = [entry.payload["name"] for entry in shared.entries()]
    assert names == ["a", "b"]

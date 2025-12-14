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

"""Tests for snapshot helpers in :mod:`weakincentives.debug`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from weakincentives.debug import dump_session
from weakincentives.runtime.annotations import is_header_line
from weakincentives.runtime.session import Session


@dataclass(slots=True, frozen=True)
class _Slice:
    value: str


def _extract_snapshot_lines(lines: list[str]) -> list[str]:
    """Extract snapshot lines from JSONL, skipping the header."""
    return [line for line in lines if not is_header_line(line)]


def test_dump_session_preserves_root_ordering(tmp_path: Path) -> None:
    root = Session()
    child = Session(parent=root)
    root.mutate(_Slice).seed((_Slice("root"),))
    child.mutate(_Slice).seed((_Slice("child"),))
    target = tmp_path / f"{root.session_id}.jsonl"

    output_path = dump_session(root, target)

    assert output_path == target
    assert output_path is not None
    lines = output_path.read_text().splitlines()

    # First line should be the header
    assert is_header_line(lines[0])
    header = json.loads(lines[0])
    assert header["header"] is True
    assert header["annotation_version"] == "1"

    # Remaining lines should be snapshots
    snapshot_lines = _extract_snapshot_lines(lines)
    session_ids = [json.loads(line)["tags"]["session_id"] for line in snapshot_lines]
    assert session_ids == [str(root.session_id), str(child.session_id)]


def test_dump_session_normalizes_target(tmp_path: Path) -> None:
    session = Session()
    session.mutate(_Slice).seed((_Slice("value"),))

    explicit = tmp_path / "custom.json"
    rewritten = dump_session(session, explicit)
    assert rewritten is not None
    assert rewritten.name == f"{session.session_id}.jsonl"
    assert rewritten.parent == tmp_path

    renamed = dump_session(session, tmp_path / "custom.jsonl")
    assert renamed is not None
    assert renamed.name == f"{session.session_id}.jsonl"


def test_dump_session_includes_header(tmp_path: Path) -> None:
    session = Session()
    session.mutate(_Slice).seed((_Slice("test"),))
    target = tmp_path / f"{session.session_id}.jsonl"

    output_path = dump_session(session, target)
    assert output_path is not None

    lines = output_path.read_text().splitlines()

    # Should have header + at least one snapshot
    assert len(lines) >= 2

    # First line is header
    header = json.loads(lines[0])
    assert header["header"] is True
    assert header["annotation_version"] == "1"
    assert "slices" in header

    # Second line is snapshot
    snapshot = json.loads(lines[1])
    assert "version" in snapshot
    assert "tags" in snapshot

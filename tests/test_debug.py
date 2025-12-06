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
from typing import TYPE_CHECKING

from weakincentives.debug import dump_session
from weakincentives.runtime.session import Session

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(slots=True, frozen=True)
class _Slice:
    value: str


def test_dump_session_preserves_root_ordering(tmp_path: Path) -> None:
    root = Session()
    child = Session(parent=root)
    root.seed_slice(_Slice, (_Slice("root"),))
    child.seed_slice(_Slice, (_Slice("child"),))
    target = tmp_path / f"{root.session_id}.jsonl"

    output_path = dump_session(root, target)

    assert output_path == target
    assert output_path is not None
    lines = output_path.read_text().splitlines()
    session_ids = [json.loads(line)["tags"]["session_id"] for line in lines]
    assert session_ids == [str(root.session_id), str(child.session_id)]


def test_dump_session_normalizes_target(tmp_path: Path) -> None:
    session = Session()
    session.seed_slice(_Slice, (_Slice("value"),))

    explicit = tmp_path / "custom.json"
    rewritten = dump_session(session, explicit)
    assert rewritten is not None
    assert rewritten.name == f"{session.session_id}.jsonl"
    assert rewritten.parent == tmp_path

    renamed = dump_session(session, tmp_path / "custom.jsonl")
    assert renamed is not None
    assert renamed.name == f"{session.session_id}.jsonl"

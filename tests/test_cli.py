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

"""Tests for ``weakincentives.cli``."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pytest

from weakincentives.cli import main
from weakincentives.core import Session, TypeRegistry
from weakincentives.debug import from_session
from weakincentives.transcript import Transcript, TranscriptEntry


@dataclass(frozen=True)
class Plan:
    steps: tuple[str, ...] = ()


def _bundle_path(tmp_path: Path) -> Path:
    session = Session()
    session[Plan].seed(Plan(steps=("a", "b")))
    transcript = Transcript()
    transcript.append(
        TranscriptEntry(
            timestamp=__import__("datetime").datetime.fromisoformat(
                "2025-01-01T00:00:00+00:00"
            ),
            kind="kind",
            payload={"k": "v"},
        )
    )
    bundle = from_session(session, transcript, metadata={"label": "demo"})
    registry = TypeRegistry()
    registry.register(Plan)
    path = tmp_path / "bundle.json"
    path.write_text(bundle.to_json(registry=registry), encoding="utf-8")
    return path


def test_debug_prints_bundle_summary(tmp_path: Path) -> None:
    path = _bundle_path(tmp_path)
    buffer = io.StringIO()
    code = main(["debug", str(path)], stdout=buffer)
    output = buffer.getvalue()
    assert code == 0
    assert "schema_version: 1" in output
    assert "metadata:" in output
    assert "label = 'demo'" in output
    assert "entries: 1" in output
    assert "[2025-01-01T00:00:00+00:00] kind" in output
    assert "slices: 1" in output
    assert "Plan" in output


def test_debug_reports_missing_path(tmp_path: Path) -> None:
    buffer = io.StringIO()
    code = main(["debug", str(tmp_path / "nope.json")], stdout=buffer)
    assert code == 1
    assert "not found" in buffer.getvalue()


def test_debug_reports_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{", encoding="utf-8")
    buffer = io.StringIO()
    code = main(["debug", str(path)], stdout=buffer)
    assert code == 1
    assert "invalid JSON" in buffer.getvalue()


def test_debug_reports_non_object_payload(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")
    buffer = io.StringIO()
    code = main(["debug", str(path)], stdout=buffer)
    assert code == 1
    assert "must be an object" in buffer.getvalue()


def test_debug_skips_metadata_section_when_empty(tmp_path: Path) -> None:
    bundle = from_session(Session(), Transcript())
    registry = TypeRegistry()
    path = tmp_path / "bundle.json"
    path.write_text(bundle.to_json(registry=registry), encoding="utf-8")
    buffer = io.StringIO()
    main(["debug", str(path)], stdout=buffer)
    output = buffer.getvalue()
    assert "metadata:" not in output
    assert "entries: 0" in output


def test_main_requires_subcommand() -> None:
    buffer = io.StringIO()
    with pytest.raises(SystemExit):
        main([], stdout=buffer)

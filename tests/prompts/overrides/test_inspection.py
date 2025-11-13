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

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from weakincentives.prompt.overrides import (
    HexDigest,
    PromptOverridesError,
    iter_override_files,
    resolve_overrides_root,
)


def test_resolve_overrides_root_uses_store_rules(tmp_path: Path) -> None:
    overrides_root = resolve_overrides_root(root_path=tmp_path)
    expected = tmp_path / ".weakincentives" / "prompts" / "overrides"
    assert overrides_root == expected


def test_resolve_overrides_root_accepts_custom_relative_path(tmp_path: Path) -> None:
    overrides_root = resolve_overrides_root(
        root_path=tmp_path,
        overrides_relative_path=Path("custom") / "path",
    )
    assert overrides_root == tmp_path / "custom" / "path"


def test_iter_override_files_yields_metadata(tmp_path: Path) -> None:
    overrides_root = resolve_overrides_root(root_path=tmp_path)
    target_dir = overrides_root / "example" / "prompt"
    target_dir.mkdir(parents=True)
    override_path = target_dir / "latest.json"
    section_hash = "a" * 64
    tool_hash = "b" * 64
    payload = {
        "version": 1,
        "ns": "example/ns",
        "prompt_key": "prompt",
        "tag": "latest",
        "sections": {
            "section": {
                "expected_hash": section_hash,
                "body": "replacement",
            }
        },
        "tools": {
            "echo": {
                "name": "echo",
                "expected_contract_hash": tool_hash,
            }
        },
    }
    override_path.write_text(json.dumps(payload), encoding="utf-8")

    [metadata] = list(iter_override_files(overrides_root=overrides_root))

    assert metadata.path == override_path.resolve()
    assert (
        metadata.relative_segments
        == override_path.resolve().relative_to(overrides_root).parts
    )
    assert metadata.section_count == 1
    assert metadata.tool_count == 1
    assert metadata.modified_time == pytest.approx(override_path.stat().st_mtime)
    assert metadata.content_hash == HexDigest(
        sha256(override_path.read_bytes()).hexdigest()
    )


def test_iter_override_files_handles_missing_directory(tmp_path: Path) -> None:
    overrides_root = tmp_path / "missing"
    assert list(iter_override_files(overrides_root=overrides_root)) == []


def test_iter_override_files_raises_on_invalid_json(tmp_path: Path) -> None:
    overrides_root = resolve_overrides_root(root_path=tmp_path)
    overrides_root.mkdir(parents=True)
    override_path = overrides_root / "broken.json"
    override_path.write_text("not json", encoding="utf-8")

    with pytest.raises(PromptOverridesError):
        list(iter_override_files(overrides_root=overrides_root))


def test_iter_override_files_raises_on_invalid_sections(tmp_path: Path) -> None:
    overrides_root = resolve_overrides_root(root_path=tmp_path)
    overrides_root.mkdir(parents=True)
    override_path = overrides_root / "invalid.json"
    payload = {
        "version": 1,
        "ns": "example/ns",
        "prompt_key": "prompt",
        "tag": "latest",
        "sections": ["unexpected"],
        "tools": {},
    }
    override_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PromptOverridesError):
        list(iter_override_files(overrides_root=overrides_root))


def test_iter_override_files_raises_on_invalid_tools(tmp_path: Path) -> None:
    overrides_root = resolve_overrides_root(root_path=tmp_path)
    overrides_root.mkdir(parents=True)
    override_path = overrides_root / "invalid_tools.json"
    payload = {
        "version": 1,
        "ns": "example/ns",
        "prompt_key": "prompt",
        "tag": "latest",
        "sections": {},
        "tools": ["unexpected"],
    }
    override_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PromptOverridesError):
        list(iter_override_files(overrides_root=overrides_root))


def test_iter_override_files_skips_non_file_matches(tmp_path: Path) -> None:
    overrides_root = resolve_overrides_root(root_path=tmp_path)
    nested_dir = overrides_root / "dir.json"
    nested_dir.mkdir(parents=True)
    target_dir = overrides_root / "example" / "prompt"
    target_dir.mkdir(parents=True)
    override_path = target_dir / "latest.json"
    payload = {
        "version": 1,
        "ns": "example/ns",
        "prompt_key": "prompt",
        "tag": "latest",
        "sections": {},
        "tools": {},
    }
    override_path.write_text(json.dumps(payload), encoding="utf-8")

    result = list(iter_override_files(overrides_root=overrides_root))

    assert [metadata.path for metadata in result] == [override_path.resolve()]


def test_iter_override_files_raises_when_payload_not_object(tmp_path: Path) -> None:
    overrides_root = resolve_overrides_root(root_path=tmp_path)
    overrides_root.mkdir(parents=True)
    override_path = overrides_root / "invalid_payload.json"
    override_path.write_text(json.dumps(["unexpected"]), encoding="utf-8")

    with pytest.raises(PromptOverridesError):
        list(iter_override_files(overrides_root=overrides_root))

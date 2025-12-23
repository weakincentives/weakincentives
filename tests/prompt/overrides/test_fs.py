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
from pathlib import Path

import pytest

from weakincentives.contrib.overrides._fs import OverrideFilesystem
from weakincentives.prompt.overrides import PromptOverridesError


def _build_filesystem(tmp_path: Path) -> OverrideFilesystem:
    return OverrideFilesystem(
        explicit_root=tmp_path,
        overrides_relative_path=Path("overrides"),
    )


def test_override_filesystem_builds_paths(tmp_path: Path) -> None:
    # When explicit_root is provided, it is used directly as the overrides
    # directory (overrides_relative_path is ignored)
    filesystem = _build_filesystem(tmp_path)

    file_path = filesystem.override_file_path(
        ns="demo/example",
        prompt_key="intro",
        tag="latest",
    )

    expected = tmp_path / "demo" / "example" / "intro" / "latest.json"
    assert file_path == expected


def test_override_filesystem_rejects_invalid_identifier(tmp_path: Path) -> None:
    filesystem = _build_filesystem(tmp_path)

    with pytest.raises(PromptOverridesError):
        filesystem.validate_identifier("Invalid Tag", "tag")


def test_override_filesystem_atomic_write(tmp_path: Path) -> None:
    filesystem = _build_filesystem(tmp_path)
    payload = {"key": "value"}

    file_path = filesystem.override_file_path(
        ns="demo",
        prompt_key="example",
        tag="latest",
    )

    with filesystem.locked_override_path(file_path):
        filesystem.atomic_write(file_path, payload)

    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    assert data == payload


def test_git_toplevel_returns_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filesystem = _build_filesystem(tmp_path)

    class _Result:
        stdout = str(tmp_path)

    monkeypatch.setattr(
        "weakincentives.contrib.overrides._fs.subprocess.run",
        lambda *_args, **_kwargs: _Result(),
    )

    assert filesystem._git_toplevel() == tmp_path


def test_git_toplevel_handles_missing_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filesystem = _build_filesystem(tmp_path)

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(
        "weakincentives.contrib.overrides._fs.subprocess.run",
        _raise,
    )

    assert filesystem._git_toplevel() is None


def test_git_toplevel_handles_empty_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filesystem = _build_filesystem(tmp_path)

    class _BlankResult:
        stdout = ""

    monkeypatch.setattr(
        "weakincentives.contrib.overrides._fs.subprocess.run",
        lambda *_args, **_kwargs: _BlankResult(),
    )

    assert filesystem._git_toplevel() is None

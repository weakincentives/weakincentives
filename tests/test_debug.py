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
import zipfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from weakincentives.debug import (
    dump_filesystem_snapshot,
    dump_session,
    dump_session_with_filesystem,
)
from weakincentives.runtime.session import Session


@dataclass(slots=True, frozen=True)
class _Slice:
    value: str


def test_dump_session_preserves_root_ordering(tmp_path: Path) -> None:
    root = Session()
    child = Session(parent=root)
    root[_Slice].seed((_Slice("root"),))
    child[_Slice].seed((_Slice("child"),))
    target = tmp_path / f"{root.session_id}.jsonl"

    output_path = dump_session(root, target)

    assert output_path == target
    assert output_path is not None
    lines = output_path.read_text().splitlines()
    session_ids = [json.loads(line)["tags"]["session_id"] for line in lines]
    assert session_ids == [str(root.session_id), str(child.session_id)]


def test_dump_session_normalizes_target(tmp_path: Path) -> None:
    session = Session()
    session[_Slice].seed((_Slice("value"),))

    explicit = tmp_path / "custom.json"
    rewritten = dump_session(session, explicit)
    assert rewritten is not None
    assert rewritten.name == f"{session.session_id}.jsonl"
    assert rewritten.parent == tmp_path

    renamed = dump_session(session, tmp_path / "custom.jsonl")
    assert renamed is not None
    assert renamed.name == f"{session.session_id}.jsonl"


def _create_mock_filesystem(files: dict[str, str]) -> MagicMock:
    """Create a mock filesystem with the given files."""
    mock_fs = MagicMock()
    mock_fs.root = "/"

    # Mock glob to return all files
    def mock_glob(pattern: str) -> list[MagicMock]:
        matches = []
        for path in files:
            match = MagicMock()
            match.path = path
            match.is_file = True
            matches.append(match)
        return matches

    mock_fs.glob = mock_glob

    # Mock read to return file contents
    def mock_read(path: str, limit: int = -1) -> MagicMock:
        if path not in files:
            raise FileNotFoundError(path)
        result = MagicMock()
        result.content = files[path]
        return result

    mock_fs.read = mock_read

    return mock_fs


def test_dump_filesystem_snapshot_creates_zip(tmp_path: Path) -> None:
    """Test that dump_filesystem_snapshot creates a valid ZIP archive."""
    mock_fs = _create_mock_filesystem(
        {
            "src/main.py": "print('hello')",
            "src/utils.py": "# utils",
            "README.md": "# Project",
        }
    )

    output_path = dump_filesystem_snapshot(mock_fs, tmp_path, session_id="test-session")

    assert output_path is not None
    assert output_path.name == "test-session.fs.zip"
    assert output_path.exists()

    with zipfile.ZipFile(output_path, "r") as zf:
        names = zf.namelist()
        assert "src/main.py" in names
        assert "src/utils.py" in names
        assert "README.md" in names
        assert "_wink_metadata.json" in names

        content = zf.read("src/main.py").decode("utf-8")
        assert content == "print('hello')"

        metadata = json.loads(zf.read("_wink_metadata.json"))
        assert metadata["version"] == "1"
        assert metadata["session_id"] == "test-session"
        assert metadata["file_count"] == 3


def test_dump_filesystem_snapshot_with_jsonl_target(tmp_path: Path) -> None:
    """Test that dump_filesystem_snapshot creates sibling .fs.zip for .jsonl target."""
    mock_fs = _create_mock_filesystem({"file.txt": "content"})

    jsonl_path = tmp_path / "my-session.jsonl"
    jsonl_path.write_text("")

    output_path = dump_filesystem_snapshot(mock_fs, jsonl_path, session_id="my-session")

    assert output_path is not None
    assert output_path.name == "my-session.fs.zip"
    assert output_path.parent == tmp_path


def test_dump_filesystem_snapshot_returns_none_for_empty(tmp_path: Path) -> None:
    """Test that dump_filesystem_snapshot returns None for empty filesystem."""
    mock_fs = _create_mock_filesystem({})

    output_path = dump_filesystem_snapshot(
        mock_fs, tmp_path, session_id="empty-session"
    )

    assert output_path is None


def test_dump_filesystem_snapshot_requires_session_id_for_directory() -> None:
    """Test that dump_filesystem_snapshot requires session_id for directory target."""
    mock_fs = _create_mock_filesystem({"file.txt": "content"})

    with pytest.raises(ValueError, match="session_id is required"):
        dump_filesystem_snapshot(mock_fs, Path("/tmp"))


def test_dump_session_with_filesystem_creates_both_files(tmp_path: Path) -> None:
    """Test that dump_session_with_filesystem creates both JSONL and ZIP files."""
    session = Session()
    session[_Slice].seed((_Slice("value"),))

    mock_fs = _create_mock_filesystem(
        {
            "code.py": "# code",
        }
    )

    session_path, fs_path = dump_session_with_filesystem(
        session, tmp_path, filesystem=mock_fs
    )

    assert session_path is not None
    assert session_path.suffix == ".jsonl"
    assert session_path.exists()

    assert fs_path is not None
    assert fs_path.name.endswith(".fs.zip")
    assert fs_path.exists()

    # Verify they share the same stem
    assert session_path.stem == fs_path.name.replace(".fs.zip", "")


def test_dump_session_with_filesystem_without_fs(tmp_path: Path) -> None:
    """Test that dump_session_with_filesystem works without filesystem."""
    session = Session()
    session[_Slice].seed((_Slice("value"),))

    session_path, fs_path = dump_session_with_filesystem(session, tmp_path)

    assert session_path is not None
    assert session_path.exists()
    assert fs_path is None


def test_dump_session_with_filesystem_handles_empty_session(tmp_path: Path) -> None:
    """Test dump_session_with_filesystem handles empty session with filesystem."""
    session = Session()  # No slices

    mock_fs = _create_mock_filesystem({"file.txt": "content"})

    session_path, fs_path = dump_session_with_filesystem(
        session, tmp_path, filesystem=mock_fs
    )

    # Session is empty so no JSONL
    assert session_path is None

    # But filesystem should still be dumped
    assert fs_path is not None
    assert fs_path.exists()


def test_dump_filesystem_snapshot_handles_read_errors(tmp_path: Path) -> None:
    """Test dump_filesystem_snapshot handles file read errors gracefully."""
    mock_fs = MagicMock()
    mock_fs.root = "/"

    # Mock glob to return files
    def mock_glob(pattern: str) -> list[MagicMock]:
        match1 = MagicMock()
        match1.path = "ok.txt"
        match1.is_file = True
        match2 = MagicMock()
        match2.path = "error.txt"
        match2.is_file = True
        return [match1, match2]

    mock_fs.glob = mock_glob

    # Mock read to succeed for one file and fail for another
    def mock_read(path: str, limit: int = -1) -> MagicMock:
        if path == "error.txt":
            raise FileNotFoundError(path)
        result = MagicMock()
        result.content = "ok content"
        return result

    mock_fs.read = mock_read

    output_path = dump_filesystem_snapshot(mock_fs, tmp_path, session_id="test")

    # Should still create archive with the successful file
    assert output_path is not None
    with zipfile.ZipFile(output_path, "r") as zf:
        names = zf.namelist()
        assert "ok.txt" in names
        assert "error.txt" not in names


def test_dump_filesystem_snapshot_with_fs_zip_suffix(tmp_path: Path) -> None:
    """Test dump_filesystem_snapshot handles .fs.zip target directly."""
    mock_fs = _create_mock_filesystem({"file.txt": "content"})

    # Target with .fs.zip suffix should be used directly
    target = tmp_path / "custom.fs.zip"
    output_path = dump_filesystem_snapshot(mock_fs, target, session_id="test")

    assert output_path is not None
    assert output_path.name == "custom.fs.zip"


def test_dump_filesystem_snapshot_with_other_suffix(tmp_path: Path) -> None:
    """Test dump_filesystem_snapshot handles other suffix targets."""
    mock_fs = _create_mock_filesystem({"file.txt": "content"})

    # Target with other suffix gets .fs.zip added
    target = tmp_path / "archive.other"
    output_path = dump_filesystem_snapshot(mock_fs, target, session_id="test")

    assert output_path is not None
    assert output_path.name == "archive.fs.zip"


def test_dump_filesystem_snapshot_handles_encoding_errors(tmp_path: Path) -> None:
    """Test dump_filesystem_snapshot skips files with encoding issues."""
    mock_fs = MagicMock()
    mock_fs.root = "/"

    # Mock glob to return files
    def mock_glob(pattern: str) -> list[MagicMock]:
        match1 = MagicMock()
        match1.path = "ok.txt"
        match1.is_file = True
        match2 = MagicMock()
        match2.path = "bad.txt"
        match2.is_file = True
        return [match1, match2]

    mock_fs.glob = mock_glob

    # Mock read to return content with invalid surrogate for bad.txt
    def mock_read(path: str, limit: int = -1) -> MagicMock:
        result = MagicMock()
        if path == "bad.txt":
            # String with lone surrogate that can't be encoded to UTF-8
            result.content = "text with bad \ud800 surrogate"
        else:
            result.content = "ok content"
        return result

    mock_fs.read = mock_read

    output_path = dump_filesystem_snapshot(mock_fs, tmp_path, session_id="test")

    # Should still create archive with the successful file
    assert output_path is not None
    with zipfile.ZipFile(output_path, "r") as zf:
        names = zf.namelist()
        assert "ok.txt" in names
        # bad.txt should be skipped due to encoding issues
        assert "bad.txt" not in names

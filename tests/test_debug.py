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
from uuid import uuid4

from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
from weakincentives.debug import archive_filesystem, dump_session
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


def test_archive_filesystem_creates_zip(tmp_path: Path) -> None:
    fs = InMemoryFilesystem()
    fs.write("src/main.py", "print('hello')")
    fs.write("src/utils.py", "def helper(): pass")
    fs.write("README.md", "# Project")

    archive_id = uuid4()
    output_path = archive_filesystem(fs, tmp_path, archive_id=archive_id)

    assert output_path is not None
    assert output_path == tmp_path / f"{archive_id}.zip"
    assert output_path.exists()

    with zipfile.ZipFile(output_path, "r") as zf:
        names = set(zf.namelist())
        assert names == {"src/main.py", "src/utils.py", "README.md"}
        assert zf.read("src/main.py") == b"print('hello')"
        assert zf.read("README.md") == b"# Project"


def test_archive_filesystem_returns_none_for_empty(tmp_path: Path) -> None:
    fs = InMemoryFilesystem()

    output_path = archive_filesystem(fs, tmp_path)

    assert output_path is None


def test_archive_filesystem_generates_uuid_if_not_provided(tmp_path: Path) -> None:
    fs = InMemoryFilesystem()
    fs.write("file.txt", "content")

    output_path = archive_filesystem(fs, tmp_path)

    assert output_path is not None
    assert output_path.suffix == ".zip"
    assert output_path.parent == tmp_path


def test_archive_filesystem_creates_target_directory(tmp_path: Path) -> None:
    fs = InMemoryFilesystem()
    fs.write("file.txt", "content")
    nested_target = tmp_path / "nested" / "dir"

    output_path = archive_filesystem(fs, nested_target)

    assert output_path is not None
    assert output_path.parent == nested_target
    assert output_path.exists()


def test_archive_filesystem_with_file_target(tmp_path: Path) -> None:
    """When target is a file path, use its parent directory."""
    fs = InMemoryFilesystem()
    fs.write("file.txt", "content")
    file_target = tmp_path / "some_file.txt"
    file_target.touch()  # Create the file

    output_path = archive_filesystem(fs, file_target)

    assert output_path is not None
    # Should use parent directory of the file
    assert output_path.parent == tmp_path
    assert output_path.suffix == ".zip"


def test_archive_filesystem_skips_unreadable_files(tmp_path: Path) -> None:
    """Files that raise errors during read are skipped."""
    from unittest.mock import Mock

    # Create a mock filesystem that raises on one file
    fs = Mock()
    fs.list.return_value = [
        Mock(name="good.txt", path="good.txt", is_file=True, is_directory=False),
        Mock(name="bad.txt", path="bad.txt", is_file=True, is_directory=False),
    ]

    def mock_read_bytes(path: str) -> Mock:
        if path == "bad.txt":
            raise PermissionError("Access denied")
        result = Mock()
        result.content = b"good content"
        return result

    fs.read_bytes.side_effect = mock_read_bytes

    archive_id = uuid4()
    output_path = archive_filesystem(fs, tmp_path, archive_id=archive_id)

    assert output_path is not None
    with zipfile.ZipFile(output_path, "r") as zf:
        # Only the good file should be in the archive
        assert zf.namelist() == ["good.txt"]
        assert zf.read("good.txt") == b"good content"


def test_archive_filesystem_handles_list_errors(tmp_path: Path) -> None:
    """FileNotFoundError during directory listing is handled gracefully."""
    from unittest.mock import Mock

    # Create a mock filesystem that raises on list
    fs = Mock()
    fs.list.side_effect = FileNotFoundError("Directory not found")

    output_path = archive_filesystem(fs, tmp_path)

    # Should return None since no files could be collected
    assert output_path is None


def test_archive_filesystem_skips_non_file_non_directory(tmp_path: Path) -> None:
    """Entries that are neither file nor directory are skipped."""
    from unittest.mock import Mock

    # Create a mock filesystem with a mixed entry list
    fs = Mock()
    fs.list.return_value = [
        Mock(name="file.txt", path="file.txt", is_file=True, is_directory=False),
        Mock(name="symlink", path="symlink", is_file=False, is_directory=False),
    ]

    def mock_read_bytes(path: str) -> Mock:
        result = Mock()
        result.content = b"content"
        return result

    fs.read_bytes.side_effect = mock_read_bytes

    archive_id = uuid4()
    output_path = archive_filesystem(fs, tmp_path, archive_id=archive_id)

    assert output_path is not None
    with zipfile.ZipFile(output_path, "r") as zf:
        # Only the file should be in the archive
        assert zf.namelist() == ["file.txt"]


def test_archive_filesystem_raises_on_write_failure_with_cleanup(
    tmp_path: Path,
) -> None:
    """OSError during archive creation cleans up partial archive."""
    from unittest.mock import Mock, patch

    import pytest

    fs = InMemoryFilesystem()
    fs.write("file.txt", "content")

    archive_id = uuid4()
    archive_path = tmp_path / f"{archive_id}.zip"

    # Create a partial archive file to verify cleanup
    archive_path.touch()
    assert archive_path.exists()

    # Mock ZipFile to raise OSError on write
    with patch("weakincentives.debug._dump.zipfile.ZipFile") as mock_zipfile:
        mock_zf = Mock()
        mock_zf.__enter__ = Mock(return_value=mock_zf)
        mock_zf.__exit__ = Mock(return_value=False)
        mock_zf.writestr.side_effect = OSError("Disk full")
        mock_zipfile.return_value = mock_zf

        with pytest.raises(OSError, match="Disk full"):
            archive_filesystem(fs, tmp_path, archive_id=archive_id)

    # Verify the partial archive was cleaned up
    assert not archive_path.exists()


def test_archive_filesystem_raises_on_write_failure_no_file(tmp_path: Path) -> None:
    """OSError during archive creation re-raises when no partial file exists."""
    from unittest.mock import patch

    import pytest

    fs = InMemoryFilesystem()
    fs.write("file.txt", "content")

    archive_id = uuid4()
    archive_path = tmp_path / f"{archive_id}.zip"

    # No archive file created yet
    assert not archive_path.exists()

    # Mock ZipFile to raise OSError before any file is created
    with patch("weakincentives.debug._dump.zipfile.ZipFile") as mock_zipfile:
        mock_zipfile.side_effect = OSError("Permission denied")

        with pytest.raises(OSError, match="Permission denied"):
            archive_filesystem(fs, tmp_path, archive_id=archive_id)

    # Verify no archive was left behind
    assert not archive_path.exists()

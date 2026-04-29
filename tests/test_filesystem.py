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

"""Tests for ``weakincentives.filesystem``."""

from __future__ import annotations

import pytest

from weakincentives.core import Snapshotable
from weakincentives.filesystem import (
    Filesystem,
    FilesystemError,
    FilesystemSnapshot,
    InMemoryFilesystem,
)


def test_satisfies_filesystem_protocol() -> None:
    assert isinstance(InMemoryFilesystem(), Filesystem)


def test_satisfies_snapshotable_protocol() -> None:
    assert isinstance(InMemoryFilesystem(), Snapshotable)


def test_write_then_read() -> None:
    fs = InMemoryFilesystem()
    fs.write_text("/a/b.txt", "hello")
    assert fs.read_text("/a/b.txt") == "hello"


def test_read_missing_file_raises() -> None:
    fs = InMemoryFilesystem()
    with pytest.raises(FilesystemError, match="missing file"):
        fs.read_text("/nope")


def test_exists_for_files_and_directories() -> None:
    fs = InMemoryFilesystem()
    fs.write_text("/a/b.txt", "hi")
    assert fs.exists("/a/b.txt")
    assert fs.exists("/a")
    assert not fs.exists("/c")


def test_list_dir_returns_immediate_children() -> None:
    fs = InMemoryFilesystem()
    fs.write_text("/a/b.txt", "1")
    fs.write_text("/a/c/d.txt", "2")
    fs.write_text("/a/c/e.txt", "3")
    assert fs.list_dir("/a") == ("b.txt", "c")
    assert fs.list_dir("/a/c") == ("d.txt", "e.txt")


def test_list_dir_root() -> None:
    fs = InMemoryFilesystem()
    fs.write_text("/x.txt", "1")
    fs.write_text("/y/z.txt", "2")
    assert fs.list_dir("/") == ("x.txt", "y")


def test_remove_missing_raises() -> None:
    fs = InMemoryFilesystem()
    with pytest.raises(FilesystemError, match="cannot remove"):
        fs.remove("/nope")


def test_remove_drops_file() -> None:
    fs = InMemoryFilesystem()
    fs.write_text("/a", "x")
    fs.remove("/a")
    assert not fs.exists("/a")


def test_walk_returns_sorted_pairs() -> None:
    fs = InMemoryFilesystem()
    fs.write_text("/b", "2")
    fs.write_text("/a", "1")
    assert list(fs.walk()) == [("/a", "1"), ("/b", "2")]


def test_normalize_rejects_empty_path() -> None:
    fs = InMemoryFilesystem()
    with pytest.raises(FilesystemError, match="non-empty"):
        fs.write_text("", "x")


def test_relative_path_normalizes_to_absolute() -> None:
    fs = InMemoryFilesystem()
    fs.write_text("a/b", "x")
    assert fs.read_text("/a/b") == "x"


def test_snapshot_round_trip() -> None:
    fs = InMemoryFilesystem()
    fs.write_text("/a", "1")
    fs.write_text("/b", "2")
    snapshot = fs.snapshot()
    fs.write_text("/a", "MUTATED")
    fs.write_text("/c", "new")
    fs.restore(snapshot)
    assert fs.read_text("/a") == "1"
    assert fs.read_text("/b") == "2"
    assert not fs.exists("/c")


def test_filesystem_snapshot_to_dict() -> None:
    snapshot = FilesystemSnapshot(files=(("/a", "1"), ("/b", "2")))
    assert snapshot.to_dict() == {"/a": "1", "/b": "2"}


def test_list_dir_skips_unrelated_keys() -> None:
    fs = InMemoryFilesystem()
    fs.write_text("/different/branch.txt", "x")
    fs.write_text("/target/inside.txt", "y")
    assert fs.list_dir("/target") == ("inside.txt",)

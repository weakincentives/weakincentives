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

"""Tests for the Filesystem protocol and InMemoryFilesystem implementation."""

from __future__ import annotations

import pytest

from weakincentives.contrib.tools.filesystem import (
    FileEntry,
    FileStat,
    GlobMatch,
    GrepMatch,
    InMemoryFilesystem,
    ReadResult,
    WriteResult,
)


class TestInMemoryFilesystemBasics:
    """Basic tests for InMemoryFilesystem."""

    def test_root_property(self) -> None:
        fs = InMemoryFilesystem()
        assert fs.root == "/"

    def test_read_only_property_default(self) -> None:
        fs = InMemoryFilesystem()
        assert fs.read_only is False

    def test_read_only_property_set(self) -> None:
        fs = InMemoryFilesystem(_read_only=True)
        assert fs.read_only is True

    def test_exists_returns_false_for_missing(self) -> None:
        fs = InMemoryFilesystem()
        assert fs.exists("missing.txt") is False

    def test_exists_returns_true_after_write(self) -> None:
        fs = InMemoryFilesystem()
        fs.write("file.txt", "content")
        assert fs.exists("file.txt") is True


class TestInMemoryFilesystemRead:
    """Tests for read operations."""

    def test_read_file_content(self) -> None:
        fs = InMemoryFilesystem()
        fs.write("file.txt", "hello world")
        result = fs.read("file.txt")
        assert result.content == "hello world"
        assert result.path == "file.txt"

    def test_read_with_offset(self) -> None:
        fs = InMemoryFilesystem()
        fs.write("file.txt", "line1\nline2\nline3")
        result = fs.read("file.txt", offset=1)
        assert "line2" in result.content

    def test_read_with_limit(self) -> None:
        fs = InMemoryFilesystem()
        lines = "\n".join([f"line{i}" for i in range(100)])
        fs.write("file.txt", lines)
        result = fs.read("file.txt", limit=5)
        assert result.limit == 5
        assert result.truncated is True

    def test_read_missing_file_raises(self) -> None:
        fs = InMemoryFilesystem()
        with pytest.raises(FileNotFoundError):
            fs.read("missing.txt")

    def test_read_directory_raises(self) -> None:
        fs = InMemoryFilesystem()
        fs.mkdir("mydir")
        with pytest.raises(IsADirectoryError):
            fs.read("mydir")


class TestInMemoryFilesystemWrite:
    """Tests for write operations."""

    def test_write_creates_file(self) -> None:
        fs = InMemoryFilesystem()
        result = fs.write("file.txt", "content", mode="create")
        assert result.path == "file.txt"
        assert result.bytes_written > 0

    def test_write_create_mode_fails_if_exists(self) -> None:
        fs = InMemoryFilesystem()
        fs.write("file.txt", "original")
        with pytest.raises(FileExistsError):
            fs.write("file.txt", "new", mode="create")

    def test_write_overwrite_mode(self) -> None:
        fs = InMemoryFilesystem()
        fs.write("file.txt", "original")
        fs.write("file.txt", "updated", mode="overwrite")
        result = fs.read("file.txt")
        assert result.content == "updated"

    def test_write_append_mode(self) -> None:
        fs = InMemoryFilesystem()
        fs.write("file.txt", "hello")
        fs.write("file.txt", " world", mode="append")
        result = fs.read("file.txt")
        assert result.content == "hello world"

    def test_write_append_creates_file(self) -> None:
        fs = InMemoryFilesystem()
        fs.write("missing.txt", "content", mode="append")
        result = fs.read("missing.txt")
        assert result.content == "content"

    def test_write_creates_parents(self) -> None:
        fs = InMemoryFilesystem()
        fs.write("dir1/dir2/file.txt", "content")
        assert fs.exists("dir1/dir2/file.txt")

    def test_write_without_parents_fails(self) -> None:
        fs = InMemoryFilesystem()
        with pytest.raises(FileNotFoundError):
            fs.write("dir1/dir2/file.txt", "content", create_parents=False)

    def test_write_readonly_fails(self) -> None:
        fs = InMemoryFilesystem(_read_only=True)
        with pytest.raises(PermissionError):
            fs.write("file.txt", "content")

    def test_write_validates_path_depth(self) -> None:
        fs = InMemoryFilesystem()
        with pytest.raises(ValueError, match="Path depth exceeds"):
            deep_path = "/".join(["d"] * 20)
            fs.write(deep_path, "content")

    def test_write_validates_path_segment_length(self) -> None:
        fs = InMemoryFilesystem()
        with pytest.raises(ValueError, match="Path segment exceeds"):
            long_segment = "a" * 300
            fs.write(long_segment, "content")

    def test_write_validates_path_ascii(self) -> None:
        fs = InMemoryFilesystem()
        with pytest.raises(ValueError, match="ASCII"):
            fs.write("ファイル.txt", "content")

    def test_path_with_parent_reference(self) -> None:
        fs = InMemoryFilesystem()
        fs.mkdir("a/b/c", parents=True)
        fs.write("a/b/c/file.txt", "content")
        # Access file using .. in path
        result = fs.read("a/b/../b/c/file.txt")
        assert result.content == "content"

    def test_write_to_root_fails(self) -> None:
        fs = InMemoryFilesystem()
        with pytest.raises(ValueError, match="Cannot write to root"):
            fs.write(".", "content")

    def test_write_content_too_long_fails(self) -> None:
        fs = InMemoryFilesystem()
        with pytest.raises(ValueError, match="Content exceeds maximum"):
            fs.write("file.txt", "x" * 100000)


class TestInMemoryFilesystemStat:
    """Tests for stat operations."""

    def test_stat_file(self) -> None:
        fs = InMemoryFilesystem()
        fs.write("file.txt", "hello")
        stat = fs.stat("file.txt")
        assert stat.path == "file.txt"
        assert stat.is_file is True
        assert stat.is_directory is False
        assert stat.size_bytes == len("hello")

    def test_stat_directory(self) -> None:
        fs = InMemoryFilesystem()
        fs.mkdir("mydir")
        stat = fs.stat("mydir")
        assert stat.is_directory is True
        assert stat.is_file is False

    def test_stat_missing_raises(self) -> None:
        fs = InMemoryFilesystem()
        with pytest.raises(FileNotFoundError):
            fs.stat("missing")


class TestInMemoryFilesystemList:
    """Tests for directory listing."""

    def test_list_root(self) -> None:
        fs = InMemoryFilesystem()
        fs.write("file.txt", "content")
        fs.mkdir("mydir")
        entries = fs.list(".")
        assert len(entries) == 2

    def test_list_subdirectory(self) -> None:
        fs = InMemoryFilesystem()
        fs.write("dir/file.txt", "content")
        entries = fs.list("dir")
        assert len(entries) == 1
        assert entries[0].name == "file.txt"

    def test_list_missing_raises(self) -> None:
        fs = InMemoryFilesystem()
        with pytest.raises(FileNotFoundError):
            fs.list("missing")

    def test_list_file_raises(self) -> None:
        fs = InMemoryFilesystem()
        fs.write("file.txt", "content")
        with pytest.raises(NotADirectoryError):
            fs.list("file.txt")


class TestInMemoryFilesystemGlob:
    """Tests for glob operations."""

    def test_glob_all_files(self) -> None:
        fs = InMemoryFilesystem()
        fs.write("a.txt", "a")
        fs.write("b.txt", "b")
        matches = fs.glob("*.txt")
        assert len(matches) == 2

    def test_glob_recursive(self) -> None:
        fs = InMemoryFilesystem()
        fs.write("dir/file.py", "code")
        fs.write("subdir/nested/deep.py", "more code")
        matches = fs.glob("**/*.py")
        # ** in fnmatch requires directory prefix
        assert len(matches) >= 1

    def test_glob_in_subdirectory(self) -> None:
        fs = InMemoryFilesystem()
        fs.write("src/main.py", "code")
        matches = fs.glob("*.py", path="src")
        assert len(matches) == 1

    def test_glob_returns_only_files(self) -> None:
        fs = InMemoryFilesystem()
        fs.mkdir("mydir")
        fs.write("mydir/file.txt", "content")
        fs.write("root.txt", "root content")
        matches = fs.glob("*")
        # Glob returns files (not directories)
        # Pattern "*" matches both files in the root
        assert len(matches) == 2
        assert all(m.is_file for m in matches)


class TestInMemoryFilesystemGrep:
    """Tests for grep operations."""

    def test_grep_invalid_regex_raises(self) -> None:
        fs = InMemoryFilesystem()
        fs.write("file.txt", "content")
        with pytest.raises(ValueError, match="Invalid regex"):
            fs.grep("[invalid")  # Unclosed bracket

    def test_grep_finds_matches(self) -> None:
        fs = InMemoryFilesystem()
        fs.write("file.txt", "hello world\nfoo bar\nhello again")
        matches = fs.grep("hello")
        assert len(matches) == 2

    def test_grep_with_glob_filter(self) -> None:
        fs = InMemoryFilesystem()
        fs.write("file.txt", "hello")
        fs.write("file.py", "hello")
        matches = fs.grep("hello", glob="*.txt")
        assert len(matches) == 1

    def test_grep_with_max_matches(self) -> None:
        fs = InMemoryFilesystem()
        fs.write("file.txt", "match\nmatch\nmatch")
        matches = fs.grep("match", max_matches=2)
        assert len(matches) == 2

    def test_grep_in_subdirectory(self) -> None:
        fs = InMemoryFilesystem()
        fs.write("src/code.py", "hello world")
        fs.write("docs/readme.md", "hello docs")
        matches = fs.grep("hello", path="src")
        assert len(matches) == 1


class TestInMemoryFilesystemDelete:
    """Tests for delete operations."""

    def test_delete_file(self) -> None:
        fs = InMemoryFilesystem()
        fs.write("file.txt", "content")
        fs.delete("file.txt")
        assert fs.exists("file.txt") is False

    def test_delete_empty_directory(self) -> None:
        fs = InMemoryFilesystem()
        fs.mkdir("mydir")
        fs.delete("mydir")
        assert fs.exists("mydir") is False

    def test_delete_nonempty_directory_fails(self) -> None:
        fs = InMemoryFilesystem()
        fs.write("mydir/file.txt", "content")
        with pytest.raises(OSError, match="not empty"):
            fs.delete("mydir", recursive=False)

    def test_delete_nonempty_directory_recursive(self) -> None:
        fs = InMemoryFilesystem()
        fs.write("mydir/file.txt", "content")
        fs.delete("mydir", recursive=True)
        assert fs.exists("mydir") is False

    def test_delete_nested_directories_recursive(self) -> None:
        fs = InMemoryFilesystem()
        fs.mkdir("a/b/c", parents=True)
        fs.write("a/b/c/file.txt", "content")
        fs.write("a/b/file2.txt", "content2")
        fs.delete("a", recursive=True)
        assert fs.exists("a") is False
        assert fs.exists("a/b") is False
        assert fs.exists("a/b/c") is False

    def test_delete_missing_raises(self) -> None:
        fs = InMemoryFilesystem()
        with pytest.raises(FileNotFoundError):
            fs.delete("missing")

    def test_delete_readonly_fails(self) -> None:
        fs = InMemoryFilesystem()
        fs.write("file.txt", "content")
        fs = InMemoryFilesystem(_read_only=True)
        with pytest.raises(PermissionError):
            fs.delete("file.txt")


class TestInMemoryFilesystemMkdir:
    """Tests for mkdir operations."""

    def test_mkdir_creates_directory(self) -> None:
        fs = InMemoryFilesystem()
        fs.mkdir("mydir")
        assert fs.exists("mydir")
        assert fs.stat("mydir").is_directory

    def test_mkdir_with_parents(self) -> None:
        fs = InMemoryFilesystem()
        fs.mkdir("a/b/c", parents=True)
        assert fs.exists("a/b/c")

    def test_mkdir_without_parents_fails(self) -> None:
        fs = InMemoryFilesystem()
        with pytest.raises(FileNotFoundError):
            fs.mkdir("a/b/c", parents=False)

    def test_mkdir_exist_ok(self) -> None:
        fs = InMemoryFilesystem()
        fs.mkdir("mydir")
        fs.mkdir("mydir", exist_ok=True)  # Should not raise
        assert fs.exists("mydir")

    def test_mkdir_exist_not_ok_fails(self) -> None:
        fs = InMemoryFilesystem()
        fs.mkdir("mydir")
        with pytest.raises(FileExistsError):
            fs.mkdir("mydir", exist_ok=False)

    def test_mkdir_over_file_fails(self) -> None:
        fs = InMemoryFilesystem()
        fs.write("file", "content")
        with pytest.raises(FileExistsError):
            fs.mkdir("file")

    def test_mkdir_readonly_fails(self) -> None:
        fs = InMemoryFilesystem(_read_only=True)
        with pytest.raises(PermissionError):
            fs.mkdir("mydir")

    def test_mkdir_root_is_noop(self) -> None:
        fs = InMemoryFilesystem()
        fs.mkdir(".")  # Should not raise, root always exists


class TestDataclassRendering:
    """Tests for dataclass render methods."""

    def test_file_stat_str(self) -> None:
        stat = FileStat(
            path="file.txt", is_file=True, is_directory=False, size_bytes=100
        )
        rendered = str(stat)
        assert "file.txt" in rendered

    def test_file_entry_str(self) -> None:
        entry = FileEntry(
            name="file.txt", path="file.txt", is_file=True, is_directory=False
        )
        rendered = str(entry)
        assert "file.txt" in rendered

    def test_glob_match_str(self) -> None:
        match = GlobMatch(path="file.txt", is_file=True)
        rendered = str(match)
        assert "file.txt" in rendered

    def test_grep_match_str(self) -> None:
        match = GrepMatch(
            path="file.txt",
            line_number=1,
            line_content="hello world",
            match_start=0,
            match_end=5,
        )
        rendered = str(match)
        assert "file.txt" in rendered
        assert "hello" in rendered

    def test_read_result_str(self) -> None:
        result = ReadResult(
            content="hello",
            path="file.txt",
            total_lines=1,
            offset=0,
            limit=100,
            truncated=False,
        )
        rendered = str(result)
        assert "file.txt" in rendered

    def test_write_result_str(self) -> None:
        result = WriteResult(path="file.txt", bytes_written=5, mode="create")
        rendered = str(result)
        assert "file.txt" in rendered

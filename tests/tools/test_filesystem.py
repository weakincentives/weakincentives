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

from pathlib import Path

import pytest

from weakincentives.contrib.tools.filesystem import (
    READ_ENTIRE_FILE,
    FileEntry,
    FileStat,
    GlobMatch,
    GrepMatch,
    HostFilesystem,
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

    def test_read_entire_file_with_constant(self) -> None:
        """Test that READ_ENTIRE_FILE reads the entire file without truncation."""
        fs = InMemoryFilesystem()
        # Create a file with more than the default 2000 line limit
        lines = "\n".join([f"line{i}" for i in range(2500)])
        fs.write("big.txt", lines)
        result = fs.read("big.txt", limit=READ_ENTIRE_FILE)
        assert result.total_lines == 2500
        assert result.truncated is False
        # Verify all content is present
        assert "line0" in result.content
        assert "line2499" in result.content


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


# HostFilesystem tests


class TestHostFilesystemBasics:
    """Basic tests for HostFilesystem."""

    def test_root_property(self, tmp_path: Path) -> None:
        fs = HostFilesystem(_root=str(tmp_path))
        assert fs.root == str(tmp_path)

    def test_read_only_property_default(self, tmp_path: Path) -> None:
        fs = HostFilesystem(_root=str(tmp_path))
        assert fs.read_only is False

    def test_read_only_property_set(self, tmp_path: Path) -> None:
        fs = HostFilesystem(_root=str(tmp_path), _read_only=True)
        assert fs.read_only is True

    def test_exists_returns_false_for_missing(self, tmp_path: Path) -> None:
        fs = HostFilesystem(_root=str(tmp_path))
        assert fs.exists("missing.txt") is False

    def test_exists_returns_true_for_file(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("content")
        fs = HostFilesystem(_root=str(tmp_path))
        assert fs.exists("file.txt") is True


class TestHostFilesystemRead:
    """Tests for HostFilesystem read operations."""

    def test_read_file_content(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("hello world")
        fs = HostFilesystem(_root=str(tmp_path))
        result = fs.read("file.txt")
        assert result.content == "hello world"
        assert result.path == "file.txt"

    def test_read_with_offset(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("line1\nline2\nline3")
        fs = HostFilesystem(_root=str(tmp_path))
        result = fs.read("file.txt", offset=1)
        assert "line2" in result.content

    def test_read_with_limit(self, tmp_path: Path) -> None:
        lines = "\n".join([f"line{i}" for i in range(100)])
        (tmp_path / "file.txt").write_text(lines)
        fs = HostFilesystem(_root=str(tmp_path))
        result = fs.read("file.txt", limit=5)
        assert result.limit == 5
        assert result.truncated is True

    def test_read_missing_file_raises(self, tmp_path: Path) -> None:
        fs = HostFilesystem(_root=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            fs.read("missing.txt")

    def test_read_directory_raises(self, tmp_path: Path) -> None:
        (tmp_path / "mydir").mkdir()
        fs = HostFilesystem(_root=str(tmp_path))
        with pytest.raises(IsADirectoryError):
            fs.read("mydir")

    def test_read_entire_file_with_constant(self, tmp_path: Path) -> None:
        """Test that READ_ENTIRE_FILE reads the entire file without truncation."""
        # Create a file with more than the default 2000 line limit
        lines = "\n".join([f"line{i}" for i in range(2500)])
        (tmp_path / "big.txt").write_text(lines)
        fs = HostFilesystem(_root=str(tmp_path))
        result = fs.read("big.txt", limit=READ_ENTIRE_FILE)
        assert result.total_lines == 2500
        assert result.truncated is False
        # Verify all content is present
        assert "line0" in result.content
        assert "line2499" in result.content


class TestHostFilesystemWrite:
    """Tests for HostFilesystem write operations."""

    def test_write_creates_file(self, tmp_path: Path) -> None:
        fs = HostFilesystem(_root=str(tmp_path))
        result = fs.write("file.txt", "content", mode="create")
        assert result.path == "file.txt"
        assert result.bytes_written > 0
        assert (tmp_path / "file.txt").read_text() == "content"

    def test_write_create_mode_fails_if_exists(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("original")
        fs = HostFilesystem(_root=str(tmp_path))
        with pytest.raises(FileExistsError):
            fs.write("file.txt", "new", mode="create")

    def test_write_overwrite_mode(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("original")
        fs = HostFilesystem(_root=str(tmp_path))
        fs.write("file.txt", "updated", mode="overwrite")
        assert (tmp_path / "file.txt").read_text() == "updated"

    def test_write_append_mode(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("hello")
        fs = HostFilesystem(_root=str(tmp_path))
        fs.write("file.txt", " world", mode="append")
        assert (tmp_path / "file.txt").read_text() == "hello world"

    def test_write_creates_parents(self, tmp_path: Path) -> None:
        fs = HostFilesystem(_root=str(tmp_path))
        fs.write("dir1/dir2/file.txt", "content")
        assert (tmp_path / "dir1/dir2/file.txt").exists()

    def test_write_without_parents_fails(self, tmp_path: Path) -> None:
        fs = HostFilesystem(_root=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            fs.write("dir1/dir2/file.txt", "content", create_parents=False)

    def test_write_readonly_fails(self, tmp_path: Path) -> None:
        fs = HostFilesystem(_root=str(tmp_path), _read_only=True)
        with pytest.raises(PermissionError):
            fs.write("file.txt", "content")


class TestHostFilesystemStat:
    """Tests for HostFilesystem stat operations."""

    def test_stat_file(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("hello")
        fs = HostFilesystem(_root=str(tmp_path))
        stat = fs.stat("file.txt")
        assert stat.path == "file.txt"
        assert stat.is_file is True
        assert stat.is_directory is False
        assert stat.size_bytes == len("hello")

    def test_stat_directory(self, tmp_path: Path) -> None:
        (tmp_path / "mydir").mkdir()
        fs = HostFilesystem(_root=str(tmp_path))
        stat = fs.stat("mydir")
        assert stat.is_directory is True
        assert stat.is_file is False

    def test_stat_missing_raises(self, tmp_path: Path) -> None:
        fs = HostFilesystem(_root=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            fs.stat("missing")


class TestHostFilesystemList:
    """Tests for HostFilesystem directory listing."""

    def test_list_root(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("content")
        (tmp_path / "mydir").mkdir()
        fs = HostFilesystem(_root=str(tmp_path))
        entries = fs.list(".")
        assert len(entries) == 2

    def test_list_subdirectory(self, tmp_path: Path) -> None:
        (tmp_path / "dir").mkdir()
        (tmp_path / "dir/file.txt").write_text("content")
        fs = HostFilesystem(_root=str(tmp_path))
        entries = fs.list("dir")
        assert len(entries) == 1
        assert entries[0].name == "file.txt"

    def test_list_missing_raises(self, tmp_path: Path) -> None:
        fs = HostFilesystem(_root=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            fs.list("missing")

    def test_list_file_raises(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("content")
        fs = HostFilesystem(_root=str(tmp_path))
        with pytest.raises(NotADirectoryError):
            fs.list("file.txt")


class TestHostFilesystemGlob:
    """Tests for HostFilesystem glob operations."""

    def test_glob_all_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        fs = HostFilesystem(_root=str(tmp_path))
        matches = fs.glob("*.txt")
        assert len(matches) == 2

    def test_glob_in_subdirectory(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src/main.py").write_text("code")
        fs = HostFilesystem(_root=str(tmp_path))
        matches = fs.glob("*.py", path="src")
        assert len(matches) == 1

    def test_glob_nonexistent_path(self, tmp_path: Path) -> None:
        fs = HostFilesystem(_root=str(tmp_path))
        matches = fs.glob("*.py", path="missing")
        assert len(matches) == 0


class TestHostFilesystemGrep:
    """Tests for HostFilesystem grep operations."""

    def test_grep_finds_matches(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("hello world\nfoo bar\nhello again")
        fs = HostFilesystem(_root=str(tmp_path))
        matches = fs.grep("hello")
        assert len(matches) == 2

    def test_grep_with_glob_filter(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("hello")
        (tmp_path / "file.py").write_text("hello")
        fs = HostFilesystem(_root=str(tmp_path))
        matches = fs.grep("hello", glob="*.txt")
        assert len(matches) == 1

    def test_grep_with_max_matches(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("match\nmatch\nmatch")
        fs = HostFilesystem(_root=str(tmp_path))
        matches = fs.grep("match", max_matches=2)
        assert len(matches) == 2

    def test_grep_invalid_regex_raises(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("content")
        fs = HostFilesystem(_root=str(tmp_path))
        with pytest.raises(ValueError, match="Invalid regex"):
            fs.grep("[invalid")

    def test_grep_nonexistent_path(self, tmp_path: Path) -> None:
        fs = HostFilesystem(_root=str(tmp_path))
        matches = fs.grep("pattern", path="missing")
        assert len(matches) == 0


class TestHostFilesystemDelete:
    """Tests for HostFilesystem delete operations."""

    def test_delete_file(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("content")
        fs = HostFilesystem(_root=str(tmp_path))
        fs.delete("file.txt")
        assert not (tmp_path / "file.txt").exists()

    def test_delete_empty_directory(self, tmp_path: Path) -> None:
        (tmp_path / "mydir").mkdir()
        fs = HostFilesystem(_root=str(tmp_path))
        fs.delete("mydir")
        assert not (tmp_path / "mydir").exists()

    def test_delete_nonempty_directory_fails(self, tmp_path: Path) -> None:
        (tmp_path / "mydir").mkdir()
        (tmp_path / "mydir/file.txt").write_text("content")
        fs = HostFilesystem(_root=str(tmp_path))
        with pytest.raises(IsADirectoryError, match="not empty"):
            fs.delete("mydir", recursive=False)

    def test_delete_nonempty_directory_recursive(self, tmp_path: Path) -> None:
        (tmp_path / "mydir").mkdir()
        (tmp_path / "mydir/file.txt").write_text("content")
        fs = HostFilesystem(_root=str(tmp_path))
        fs.delete("mydir", recursive=True)
        assert not (tmp_path / "mydir").exists()

    def test_delete_missing_raises(self, tmp_path: Path) -> None:
        fs = HostFilesystem(_root=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            fs.delete("missing")

    def test_delete_readonly_fails(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("content")
        fs = HostFilesystem(_root=str(tmp_path), _read_only=True)
        with pytest.raises(PermissionError):
            fs.delete("file.txt")


class TestHostFilesystemMkdir:
    """Tests for HostFilesystem mkdir operations."""

    def test_mkdir_creates_directory(self, tmp_path: Path) -> None:
        fs = HostFilesystem(_root=str(tmp_path))
        fs.mkdir("mydir")
        assert (tmp_path / "mydir").is_dir()

    def test_mkdir_with_parents(self, tmp_path: Path) -> None:
        fs = HostFilesystem(_root=str(tmp_path))
        fs.mkdir("a/b/c", parents=True)
        assert (tmp_path / "a/b/c").is_dir()

    def test_mkdir_without_parents_fails(self, tmp_path: Path) -> None:
        fs = HostFilesystem(_root=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            fs.mkdir("a/b/c", parents=False)

    def test_mkdir_exist_ok(self, tmp_path: Path) -> None:
        (tmp_path / "mydir").mkdir()
        fs = HostFilesystem(_root=str(tmp_path))
        fs.mkdir("mydir", exist_ok=True)  # Should not raise

    def test_mkdir_exist_not_ok_fails(self, tmp_path: Path) -> None:
        (tmp_path / "mydir").mkdir()
        fs = HostFilesystem(_root=str(tmp_path))
        with pytest.raises(FileExistsError):
            fs.mkdir("mydir", exist_ok=False)

    def test_mkdir_over_file_fails(self, tmp_path: Path) -> None:
        (tmp_path / "file").write_text("content")
        fs = HostFilesystem(_root=str(tmp_path))
        with pytest.raises(FileExistsError):
            fs.mkdir("file")

    def test_mkdir_readonly_fails(self, tmp_path: Path) -> None:
        fs = HostFilesystem(_root=str(tmp_path), _read_only=True)
        with pytest.raises(PermissionError):
            fs.mkdir("mydir")


class TestHostFilesystemPathSecurity:
    """Tests for HostFilesystem path security."""

    def test_path_escape_prevented(self, tmp_path: Path) -> None:
        fs = HostFilesystem(_root=str(tmp_path))
        # Attempting to escape the root should fail
        with pytest.raises(PermissionError, match="escapes root"):
            fs.read("../etc/passwd")

    def test_exists_returns_false_for_escape_attempt(self, tmp_path: Path) -> None:
        fs = HostFilesystem(_root=str(tmp_path))
        # Should return False rather than raise
        assert fs.exists("../etc/passwd") is False


class TestHostFilesystemEdgeCases:
    """Tests for HostFilesystem edge cases and validation."""

    def test_glob_skips_directories(self, tmp_path: Path) -> None:
        """Glob should only return files, not directories."""
        (tmp_path / "subdir").mkdir()
        (tmp_path / "file.txt").write_text("content")
        fs = HostFilesystem(_root=str(tmp_path))
        matches = fs.glob("*")
        # Should only match the file, not the directory
        assert len(matches) == 1
        assert matches[0].path == "file.txt"

    def test_grep_skips_directories(self, tmp_path: Path) -> None:
        """Grep should only search files, not directories."""
        (tmp_path / "subdir").mkdir()
        (tmp_path / "file.txt").write_text("hello")
        fs = HostFilesystem(_root=str(tmp_path))
        matches = fs.grep("hello")
        assert len(matches) == 1

    def test_grep_skips_binary_files(self, tmp_path: Path) -> None:
        """Grep should skip files that fail to decode as UTF-8."""
        # Write a file with invalid UTF-8 bytes
        (tmp_path / "binary.bin").write_bytes(b"\xff\xfe\x00\x01")
        (tmp_path / "text.txt").write_text("hello")
        fs = HostFilesystem(_root=str(tmp_path))
        matches = fs.grep(".*")  # Match anything
        # Should only match the text file
        assert len(matches) == 1
        assert matches[0].path == "text.txt"

    def test_write_to_root_fails(self, tmp_path: Path) -> None:
        """Writing to root path should fail."""
        fs = HostFilesystem(_root=str(tmp_path))
        with pytest.raises(ValueError, match="Cannot write to root"):
            fs.write(".", "content")

    def test_write_content_too_long_fails(self, tmp_path: Path) -> None:
        """Writing content that exceeds max length should fail."""
        fs = HostFilesystem(_root=str(tmp_path))
        with pytest.raises(ValueError, match="Content exceeds maximum"):
            fs.write("file.txt", "x" * 50000)

    def test_delete_root_fails(self, tmp_path: Path) -> None:
        """Deleting root directory should fail."""
        fs = HostFilesystem(_root=str(tmp_path))
        with pytest.raises(PermissionError, match="Cannot delete root"):
            fs.delete(".")

    def test_mkdir_root_is_noop(self, tmp_path: Path) -> None:
        """Creating root directory should be a no-op."""
        fs = HostFilesystem(_root=str(tmp_path))
        fs.mkdir(".")  # Should not raise

    def test_mkdir_single_level_without_parents(self, tmp_path: Path) -> None:
        """Creating a single directory level without parents should work."""
        fs = HostFilesystem(_root=str(tmp_path))
        fs.mkdir("newdir", parents=False)
        assert (tmp_path / "newdir").is_dir()

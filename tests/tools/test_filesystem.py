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
from uuid import UUID

import pytest

from weakincentives.contrib.tools.filesystem import (
    READ_ENTIRE_FILE,
    FileEntry,
    FileStat,
    FilesystemSnapshot,
    GlobMatch,
    GrepMatch,
    HostFilesystem,
    InMemoryFilesystem,
    ReadResult,
    SnapshotableFilesystem,
    WriteResult,
    normalize_path,
)
from weakincentives.errors import SnapshotRestoreError


class TestNormalizePath:
    """Tests for normalize_path utility function."""

    def test_empty_path(self) -> None:
        assert normalize_path("") == ""

    def test_dot_path(self) -> None:
        assert normalize_path(".") == ""

    def test_root_path(self) -> None:
        assert normalize_path("/") == ""

    def test_only_slashes_results_in_empty_segments(self) -> None:
        """Paths with only slashes should result in empty result."""
        assert normalize_path("//") == ""
        assert normalize_path("///") == ""

    def test_only_dots_and_slashes(self) -> None:
        """Paths with only dots and slashes should result in empty."""
        assert normalize_path("/./") == ""
        assert normalize_path("././") == ""

    def test_simple_path(self) -> None:
        assert normalize_path("foo/bar") == "foo/bar"

    def test_path_with_leading_slash(self) -> None:
        assert normalize_path("/foo/bar") == "foo/bar"

    def test_path_with_trailing_slash(self) -> None:
        assert normalize_path("foo/bar/") == "foo/bar"

    def test_path_with_parent_refs(self) -> None:
        assert normalize_path("foo/../bar") == "bar"
        assert normalize_path("foo/bar/../baz") == "foo/baz"

    def test_parent_refs_at_root(self) -> None:
        """Parent refs that go beyond root are ignored."""
        assert normalize_path("../foo") == "foo"
        assert normalize_path("../../foo") == "foo"


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

    def test_delete_root_clears_contents_but_keeps_root(self) -> None:
        """Deleting root recursively clears contents but root directory persists."""
        fs = InMemoryFilesystem()
        fs.write("file.txt", "content")
        fs.mkdir("subdir")
        fs.write("subdir/nested.txt", "nested content")

        # Delete root recursively
        fs.delete("/", recursive=True)

        # Root should still exist
        stat = fs.stat("/")
        assert stat.is_directory

        # Contents should be gone
        assert not fs.exists("file.txt")
        assert not fs.exists("subdir")


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


# Snapshot tests for InMemoryFilesystem


class TestInMemoryFilesystemSnapshots:
    """Tests for InMemoryFilesystem snapshot operations."""

    def test_implements_snapshotable_protocol(self) -> None:
        """InMemoryFilesystem should implement SnapshotableFilesystem."""
        fs = InMemoryFilesystem()
        assert isinstance(fs, SnapshotableFilesystem)

    def test_snapshot_returns_filesystem_snapshot(self) -> None:
        """snapshot() should return a FilesystemSnapshot."""
        fs = InMemoryFilesystem()
        fs.write("file.txt", "content")
        snapshot = fs.snapshot()
        assert isinstance(snapshot, FilesystemSnapshot)
        assert isinstance(snapshot.snapshot_id, UUID)
        assert snapshot.root_path == "/"
        assert snapshot.commit_ref.startswith("mem-")

    def test_snapshot_with_tag(self) -> None:
        """snapshot() should accept an optional tag."""
        fs = InMemoryFilesystem()
        fs.write("file.txt", "content")
        snapshot = fs.snapshot(tag="my-tag")
        assert snapshot.tag == "my-tag"

    def test_snapshot_and_restore_roundtrip(self) -> None:
        """Basic snapshot and restore should preserve file contents."""
        fs = InMemoryFilesystem()
        fs.write("config.py", "DEBUG = True")
        snapshot_v1 = fs.snapshot(tag="initial")

        # Modify file
        fs.write("config.py", "DEBUG = False")
        assert fs.read("config.py").content == "DEBUG = False"

        # Restore
        fs.restore(snapshot_v1)
        assert fs.read("config.py").content == "DEBUG = True"

    def test_restore_removes_new_files(self) -> None:
        """restore() should remove files created after snapshot."""
        fs = InMemoryFilesystem()
        fs.write("original.txt", "original")
        snapshot = fs.snapshot()

        # Add new file
        fs.write("new.txt", "new content")
        assert fs.exists("new.txt")

        # Restore should remove the new file
        fs.restore(snapshot)
        assert not fs.exists("new.txt")
        assert fs.exists("original.txt")

    def test_restore_restores_deleted_files(self) -> None:
        """restore() should bring back deleted files."""
        fs = InMemoryFilesystem()
        fs.write("file.txt", "content")
        snapshot = fs.snapshot()

        # Delete file
        fs.delete("file.txt")
        assert not fs.exists("file.txt")

        # Restore should bring it back
        fs.restore(snapshot)
        assert fs.exists("file.txt")
        assert fs.read("file.txt").content == "content"

    def test_multiple_snapshots(self) -> None:
        """Multiple snapshots can be restored independently."""
        fs = InMemoryFilesystem()

        fs.write("file.txt", "v1")
        snapshot_v1 = fs.snapshot(tag="v1")

        fs.write("file.txt", "v2")
        snapshot_v2 = fs.snapshot(tag="v2")

        fs.write("file.txt", "v3")

        # Restore to v1
        fs.restore(snapshot_v1)
        assert fs.read("file.txt").content == "v1"

        # Restore to v2
        fs.restore(snapshot_v2)
        assert fs.read("file.txt").content == "v2"

    def test_restore_unknown_snapshot_raises(self) -> None:
        """restore() should raise SnapshotRestoreError for unknown snapshot."""
        from datetime import UTC, datetime

        fs = InMemoryFilesystem()
        fake_snapshot = FilesystemSnapshot(
            snapshot_id=UUID("00000000-0000-0000-0000-000000000000"),
            created_at=datetime.now(UTC),
            commit_ref="mem-nonexistent",
            root_path="/",
        )
        with pytest.raises(SnapshotRestoreError, match="Unknown snapshot"):
            fs.restore(fake_snapshot)

    def test_restore_directories(self) -> None:
        """restore() should restore directory structure."""
        fs = InMemoryFilesystem()
        fs.mkdir("a/b/c", parents=True)
        fs.write("a/b/c/file.txt", "content")
        snapshot = fs.snapshot()

        # Delete directory
        fs.delete("a", recursive=True)
        assert not fs.exists("a/b/c")

        # Restore
        fs.restore(snapshot)
        assert fs.exists("a/b/c")
        assert fs.read("a/b/c/file.txt").content == "content"


# Snapshot tests for HostFilesystem


class TestHostFilesystemSnapshots:
    """Tests for HostFilesystem snapshot operations."""

    def test_implements_snapshotable_protocol(self, tmp_path: Path) -> None:
        """HostFilesystem should implement SnapshotableFilesystem."""
        fs = HostFilesystem(_root=str(tmp_path))
        assert isinstance(fs, SnapshotableFilesystem)

    def test_snapshot_returns_filesystem_snapshot(self, tmp_path: Path) -> None:
        """snapshot() should return a FilesystemSnapshot with git commit."""
        fs = HostFilesystem(_root=str(tmp_path))
        (tmp_path / "file.txt").write_text("content")
        snapshot = fs.snapshot()
        assert isinstance(snapshot, FilesystemSnapshot)
        assert isinstance(snapshot.snapshot_id, UUID)
        assert snapshot.root_path == str(tmp_path)
        # Git commit hash is 40 hex characters
        assert len(snapshot.commit_ref) == 40
        assert all(c in "0123456789abcdef" for c in snapshot.commit_ref)

    def test_snapshot_with_tag(self, tmp_path: Path) -> None:
        """snapshot() should accept an optional tag."""
        fs = HostFilesystem(_root=str(tmp_path))
        (tmp_path / "file.txt").write_text("content")
        snapshot = fs.snapshot(tag="my-tag")
        assert snapshot.tag == "my-tag"

    def test_snapshot_and_restore_roundtrip(self, tmp_path: Path) -> None:
        """Basic snapshot and restore should preserve file contents."""
        fs = HostFilesystem(_root=str(tmp_path))
        fs.write("config.py", "DEBUG = True")
        snapshot_v1 = fs.snapshot(tag="initial")

        # Modify file
        fs.write("config.py", "DEBUG = False")
        assert fs.read("config.py").content == "DEBUG = False"

        # Restore
        fs.restore(snapshot_v1)
        assert fs.read("config.py").content == "DEBUG = True"

    def test_restore_removes_new_files(self, tmp_path: Path) -> None:
        """restore() should remove files created after snapshot."""
        fs = HostFilesystem(_root=str(tmp_path))
        fs.write("original.txt", "original")
        snapshot = fs.snapshot()

        # Add new file
        fs.write("new.txt", "new content")
        assert fs.exists("new.txt")

        # Restore should remove the new file
        fs.restore(snapshot)
        assert not fs.exists("new.txt")
        assert fs.exists("original.txt")

    def test_restore_removes_gitignored_files(self, tmp_path: Path) -> None:
        """restore() should remove gitignored files for strict rollback."""
        fs = HostFilesystem(_root=str(tmp_path))
        fs.write("src/main.py", "print('hello')")
        # Create .gitignore to ignore cache files
        fs.write(".gitignore", "*.pyc\n__pycache__/\n*.log\n")
        snapshot = fs.snapshot()

        # Simulate tool creating ignored files (cache, logs)
        (tmp_path / "src" / "main.pyc").write_bytes(b"compiled")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "main.cpython-311.pyc").write_bytes(b"cache")
        (tmp_path / "debug.log").write_text("log output")

        # Verify ignored files exist
        assert (tmp_path / "src" / "main.pyc").exists()
        assert (tmp_path / "__pycache__").exists()
        assert (tmp_path / "debug.log").exists()

        # Restore should remove ignored files too (strict rollback)
        fs.restore(snapshot)

        # All ignored files should be gone
        assert not (tmp_path / "src" / "main.pyc").exists()
        assert not (tmp_path / "__pycache__").exists()
        assert not (tmp_path / "debug.log").exists()
        # Original files remain
        assert fs.exists("src/main.py")
        assert fs.exists(".gitignore")

    def test_restore_restores_deleted_files(self, tmp_path: Path) -> None:
        """restore() should bring back deleted files."""
        fs = HostFilesystem(_root=str(tmp_path))
        fs.write("file.txt", "content")
        snapshot = fs.snapshot()

        # Delete file
        fs.delete("file.txt")
        assert not fs.exists("file.txt")

        # Restore should bring it back
        fs.restore(snapshot)
        assert fs.exists("file.txt")
        assert fs.read("file.txt").content == "content"

    def test_multiple_snapshots(self, tmp_path: Path) -> None:
        """Multiple snapshots can be restored independently."""
        fs = HostFilesystem(_root=str(tmp_path))

        fs.write("file.txt", "v1")
        snapshot_v1 = fs.snapshot(tag="v1")

        fs.write("file.txt", "v2")
        snapshot_v2 = fs.snapshot(tag="v2")

        fs.write("file.txt", "v3")

        # Restore to v1
        fs.restore(snapshot_v1)
        assert fs.read("file.txt").content == "v1"

        # Restore to v2
        fs.restore(snapshot_v2)
        assert fs.read("file.txt").content == "v2"

    def test_restore_invalid_commit_raises(self, tmp_path: Path) -> None:
        """restore() should raise SnapshotRestoreError for invalid commit."""
        from datetime import UTC, datetime

        fs = HostFilesystem(_root=str(tmp_path))
        # Need to initialize git first
        fs.write("file.txt", "content")
        _ = fs.snapshot()

        fake_snapshot = FilesystemSnapshot(
            snapshot_id=UUID("00000000-0000-0000-0000-000000000000"),
            created_at=datetime.now(UTC),
            commit_ref="0" * 40,  # Invalid commit
            root_path=str(tmp_path),
        )
        with pytest.raises(SnapshotRestoreError, match="Failed to restore"):
            fs.restore(fake_snapshot)

    def test_restore_directories(self, tmp_path: Path) -> None:
        """restore() should restore directory structure."""
        fs = HostFilesystem(_root=str(tmp_path))
        fs.mkdir("a/b/c", parents=True)
        fs.write("a/b/c/file.txt", "content")
        snapshot = fs.snapshot()

        # Delete directory
        fs.delete("a", recursive=True)
        assert not fs.exists("a/b/c")

        # Restore
        fs.restore(snapshot)
        assert fs.exists("a/b/c")
        assert fs.read("a/b/c/file.txt").content == "content"

    def test_git_initialized_once(self, tmp_path: Path) -> None:
        """Git should only be initialized once."""
        fs = HostFilesystem(_root=str(tmp_path))
        fs.write("file.txt", "content")

        # First snapshot initializes git
        _ = fs.snapshot()
        assert fs._git_initialized
        git_dir = tmp_path / ".git"
        assert git_dir.exists()

        # Second snapshot reuses existing git
        _ = fs.snapshot()
        # Still the same git dir
        assert git_dir.exists()

    def test_idempotent_empty_snapshot(self, tmp_path: Path) -> None:
        """Creating snapshots without changes should work (allow-empty)."""
        fs = HostFilesystem(_root=str(tmp_path))
        fs.write("file.txt", "content")

        snapshot1 = fs.snapshot()
        # No changes
        snapshot2 = fs.snapshot()

        # Both should be valid snapshots (different commit hashes due to timestamp)
        assert snapshot1.commit_ref != snapshot2.commit_ref
        assert len(snapshot1.commit_ref) == 40
        assert len(snapshot2.commit_ref) == 40

    def test_snapshot_empty_repo_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Snapshot should handle edge case where commit fails on empty repo."""
        import subprocess
        from unittest.mock import MagicMock

        fs = HostFilesystem(_root=str(tmp_path))

        # Store original subprocess.run
        original_run = subprocess.run

        call_count = 0

        def mock_subprocess_run(
            args: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[bytes] | subprocess.CompletedProcess[str]:
            nonlocal call_count
            call_count += 1

            # Let git init and config calls through
            if args[0:2] in (
                ["git", "init"],
                ["git", "config"],
                ["git", "add"],
                ["git", "rev-parse"],
            ):
                return original_run(args, **kwargs)

            # For git commit: fail the first time, succeed the second
            if args[0:2] == ["git", "commit"]:
                # First commit call should fail (simulating edge case)
                if call_count <= 5:  # After init, 2 config, add, first commit
                    mock_result = MagicMock()
                    mock_result.returncode = 1
                    mock_result.stderr = "simulated failure"
                    return mock_result
                # Second commit (fallback) should succeed
                return original_run(args, **kwargs)

            return original_run(args, **kwargs)

        monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

        # This should trigger the fallback path
        snapshot = fs.snapshot()
        assert isinstance(snapshot, FilesystemSnapshot)
        assert len(snapshot.commit_ref) == 40

    def test_snapshot_on_existing_git_repo(self, tmp_path: Path) -> None:
        """Snapshot should work on pre-existing git repo without re-initializing."""
        import subprocess

        # Initialize git repo before creating HostFilesystem
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        # Create a file and commit it
        (tmp_path / "existing.txt").write_text("pre-existing content")
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "--no-gpg-sign", "-m", "initial commit"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        # Now create HostFilesystem on this existing repo
        fs = HostFilesystem(_root=str(tmp_path))
        fs.write("new_file.txt", "new content")

        # Snapshot should work without re-initializing git
        snapshot = fs.snapshot()
        assert isinstance(snapshot, FilesystemSnapshot)
        assert len(snapshot.commit_ref) == 40

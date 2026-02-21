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

"""Core filesystem validation suite: basic operations, read/write, stat, list, glob, grep, delete, mkdir.

This module contains the non-streaming portion of the filesystem validation
tests. See ``filesystem_streaming.py`` for streaming operations.
"""

from __future__ import annotations

from abc import abstractmethod

import pytest

from weakincentives.filesystem import (
    READ_ENTIRE_FILE,
    Filesystem,
)


class FilesystemCoreValidationSuite:
    """Abstract test suite for core Filesystem protocol compliance.

    Subclasses must implement the ``fs`` fixture to provide a filesystem
    instance to test. The filesystem should be empty at the start of each test.

    This suite validates:
    - Read operations (read, read_bytes, exists, stat, list, glob, grep)
    - Write operations (write, write_bytes, delete, mkdir)
    - Properties (root, read_only)
    - Error handling (FileNotFoundError, IsADirectoryError, etc.)
    - Path validation (depth, segment length)

    Implementation-specific tests (e.g., HostFilesystem path escape detection)
    should remain in their own test modules.
    """

    @pytest.fixture
    @abstractmethod
    def fs(self) -> Filesystem:
        """Provide a fresh filesystem instance for testing.

        Subclasses must implement this fixture to return a filesystem
        implementation to test.
        """
        ...

    # -------------------------------------------------------------------------
    # Basic Properties
    # -------------------------------------------------------------------------

    def test_root_property_exists(self, fs: Filesystem) -> None:
        """Filesystem should have a root property."""
        assert fs.root is not None

    def test_read_only_property_default_false(self, fs: Filesystem) -> None:
        """Default filesystem should not be read-only."""
        assert fs.read_only is False

    # -------------------------------------------------------------------------
    # Exists Operation
    # -------------------------------------------------------------------------

    def test_exists_returns_false_for_missing(self, fs: Filesystem) -> None:
        """exists() should return False for nonexistent paths."""
        assert fs.exists("missing.txt") is False

    def test_exists_returns_true_after_write(self, fs: Filesystem) -> None:
        """exists() should return True after writing a file."""
        fs.write("file.txt", "content")
        assert fs.exists("file.txt") is True

    def test_exists_returns_true_for_directory(self, fs: Filesystem) -> None:
        """exists() should return True for directories."""
        fs.mkdir("mydir")
        assert fs.exists("mydir") is True

    def test_exists_returns_true_for_root(self, fs: Filesystem) -> None:
        """exists() should return True for root directory."""
        assert fs.exists(".") is True

    # -------------------------------------------------------------------------
    # Read Operation
    # -------------------------------------------------------------------------

    def test_read_file_content(self, fs: Filesystem) -> None:
        """read() should return file content."""
        fs.write("file.txt", "hello world")
        result = fs.read("file.txt")
        assert result.content == "hello world"
        assert result.path == "file.txt"

    def test_read_multiline_content(self, fs: Filesystem) -> None:
        """read() should handle multiline content."""
        content = "line1\nline2\nline3"
        fs.write("file.txt", content)
        result = fs.read("file.txt")
        assert "line1" in result.content
        assert "line2" in result.content
        assert "line3" in result.content

    def test_read_with_offset(self, fs: Filesystem) -> None:
        """read() should support offset parameter."""
        fs.write("file.txt", "line1\nline2\nline3")
        result = fs.read("file.txt", offset=1)
        assert "line2" in result.content
        assert result.offset == 1

    def test_read_with_limit(self, fs: Filesystem) -> None:
        """read() should support limit parameter."""
        lines = "\n".join([f"line{i}" for i in range(100)])
        fs.write("file.txt", lines)
        result = fs.read("file.txt", limit=5)
        assert result.limit == 5
        assert result.truncated is True

    def test_read_entire_file_constant(self, fs: Filesystem) -> None:
        """READ_ENTIRE_FILE should read entire file without truncation."""
        lines = "\n".join([f"line{i}" for i in range(2500)])
        fs.write("big.txt", lines)
        result = fs.read("big.txt", limit=READ_ENTIRE_FILE)
        assert result.total_lines == 2500
        assert result.truncated is False
        assert "line0" in result.content
        assert "line2499" in result.content

    def test_read_missing_file_raises(self, fs: Filesystem) -> None:
        """read() should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            fs.read("missing.txt")

    def test_read_directory_raises(self, fs: Filesystem) -> None:
        """read() should raise IsADirectoryError for directories."""
        fs.mkdir("mydir")
        with pytest.raises(IsADirectoryError):
            fs.read("mydir")

    # -------------------------------------------------------------------------
    # Read Bytes Operation
    # -------------------------------------------------------------------------

    def test_read_bytes_file_content(self, fs: Filesystem) -> None:
        """read_bytes() should return raw file content."""
        fs.write_bytes("file.bin", b"\x00\x01\x02\x03")
        result = fs.read_bytes("file.bin")
        assert result.content == b"\x00\x01\x02\x03"
        assert result.path == "file.bin"
        assert result.size_bytes == 4

    def test_read_bytes_with_offset(self, fs: Filesystem) -> None:
        """read_bytes() should support offset parameter."""
        fs.write_bytes("file.bin", b"hello world")
        result = fs.read_bytes("file.bin", offset=6)
        assert result.content == b"world"
        assert result.offset == 6

    def test_read_bytes_with_limit(self, fs: Filesystem) -> None:
        """read_bytes() should support limit parameter."""
        fs.write_bytes("file.bin", b"hello world")
        result = fs.read_bytes("file.bin", limit=5)
        assert result.content == b"hello"
        assert result.limit == 5
        assert result.truncated is True

    def test_read_bytes_missing_file_raises(self, fs: Filesystem) -> None:
        """read_bytes() should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            fs.read_bytes("missing.bin")

    def test_read_bytes_directory_raises(self, fs: Filesystem) -> None:
        """read_bytes() should raise IsADirectoryError for directories."""
        fs.mkdir("mydir")
        with pytest.raises(IsADirectoryError):
            fs.read_bytes("mydir")

    def test_read_bytes_binary_content(self, fs: Filesystem) -> None:
        """read_bytes() should handle binary content that can't be decoded as UTF-8."""
        binary_data = b"\xff\xfe\x00\x01\x80\x81"  # Invalid UTF-8
        fs.write_bytes("binary.bin", binary_data)
        result = fs.read_bytes("binary.bin")
        assert result.content == binary_data

    def test_read_text_fails_on_binary_content(self, fs: Filesystem) -> None:
        """read() should fail when file contains binary content that can't be decoded."""
        binary_data = b"\xff\xfe\x00\x01\x80\x81"  # Invalid UTF-8
        fs.write_bytes("binary.bin", binary_data)
        with pytest.raises(ValueError, match=r"binary content.*read_bytes"):
            fs.read("binary.bin")

    def test_read_bytes_negative_offset_raises(self, fs: Filesystem) -> None:
        """read_bytes() should raise ValueError for negative offset."""
        fs.write_bytes("file.bin", b"content")
        with pytest.raises(ValueError, match=r"offset must be non-negative"):
            fs.read_bytes("file.bin", offset=-1)

    def test_read_bytes_negative_limit_raises(self, fs: Filesystem) -> None:
        """read_bytes() should raise ValueError for negative limit."""
        fs.write_bytes("file.bin", b"content")
        with pytest.raises(ValueError, match=r"limit must be non-negative"):
            fs.read_bytes("file.bin", limit=-1)

    # -------------------------------------------------------------------------
    # Write Operation
    # -------------------------------------------------------------------------

    def test_write_creates_file(self, fs: Filesystem) -> None:
        """write() should create a new file."""
        result = fs.write("file.txt", "content", mode="create")
        assert result.path == "file.txt"
        assert result.bytes_written > 0
        assert fs.exists("file.txt")

    def test_write_create_mode_fails_if_exists(self, fs: Filesystem) -> None:
        """write() with mode='create' should fail if file exists."""
        fs.write("file.txt", "original")
        with pytest.raises(FileExistsError):
            fs.write("file.txt", "new", mode="create")

    def test_write_overwrite_mode(self, fs: Filesystem) -> None:
        """write() with mode='overwrite' should replace content."""
        fs.write("file.txt", "original")
        fs.write("file.txt", "updated", mode="overwrite")
        result = fs.read("file.txt")
        assert result.content == "updated"

    def test_write_append_mode(self, fs: Filesystem) -> None:
        """write() with mode='append' should append content."""
        fs.write("file.txt", "hello")
        fs.write("file.txt", " world", mode="append")
        result = fs.read("file.txt")
        assert result.content == "hello world"

    def test_write_append_creates_file(self, fs: Filesystem) -> None:
        """write() with mode='append' should create file if missing."""
        fs.write("missing.txt", "content", mode="append")
        result = fs.read("missing.txt")
        assert result.content == "content"

    def test_write_append_bytes_written(self, fs: Filesystem) -> None:
        """write() with mode='append' should report bytes written, not total size."""
        fs.write("file.txt", "hello")  # 5 bytes
        result = fs.write("file.txt", " world", mode="append")  # 6 bytes
        # bytes_written should be 6 (what we just wrote), not 11 (total size)
        assert result.bytes_written == len(b" world")

    def test_write_creates_parents(self, fs: Filesystem) -> None:
        """write() should create parent directories by default."""
        fs.write("dir1/dir2/file.txt", "content")
        assert fs.exists("dir1/dir2/file.txt")

    def test_write_without_parents_fails(self, fs: Filesystem) -> None:
        """write() with create_parents=False should fail if parent missing."""
        with pytest.raises(FileNotFoundError):
            fs.write("dir1/dir2/file.txt", "content", create_parents=False)

    def test_write_to_root_fails(self, fs: Filesystem) -> None:
        """write() to root directory should fail."""
        with pytest.raises(ValueError, match=r"[Cc]annot write to root"):
            fs.write(".", "content")

    def test_write_content_too_long_fails(
        self, fs: Filesystem, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """write() should fail if content exceeds max length."""
        # Patch both implementations since we don't know which one is being tested
        monkeypatch.setattr(
            "weakincentives.contrib.tools.filesystem_memory.MAX_WRITE_LENGTH", 100
        )
        monkeypatch.setattr("weakincentives.filesystem._host.MAX_WRITE_LENGTH", 100)
        with pytest.raises(ValueError, match=r"[Cc]ontent exceeds maximum"):
            fs.write("file.txt", "x" * 101)

    def test_write_validates_path_depth(self, fs: Filesystem) -> None:
        """write() should validate path depth."""
        deep_path = "/".join(["d"] * 20)
        with pytest.raises(ValueError, match=r"[Pp]ath depth exceeds"):
            fs.write(deep_path, "content")

    def test_write_validates_path_segment_length(self, fs: Filesystem) -> None:
        """write() should validate path segment length."""
        long_segment = "a" * 300
        with pytest.raises(ValueError, match=r"[Pp]ath segment exceeds"):
            fs.write(long_segment, "content")

    # -------------------------------------------------------------------------
    # Write Bytes Operation
    # -------------------------------------------------------------------------

    def test_write_bytes_creates_file(self, fs: Filesystem) -> None:
        """write_bytes() should create a new file."""
        result = fs.write_bytes("file.bin", b"\x00\x01\x02", mode="create")
        assert result.path == "file.bin"
        assert result.bytes_written == 3
        assert fs.exists("file.bin")

    def test_write_bytes_create_mode_fails_if_exists(self, fs: Filesystem) -> None:
        """write_bytes() with mode='create' should fail if file exists."""
        fs.write_bytes("file.bin", b"original")
        with pytest.raises(FileExistsError):
            fs.write_bytes("file.bin", b"new", mode="create")

    def test_write_bytes_overwrite_mode(self, fs: Filesystem) -> None:
        """write_bytes() with mode='overwrite' should replace content."""
        fs.write_bytes("file.bin", b"original")
        fs.write_bytes("file.bin", b"updated", mode="overwrite")
        result = fs.read_bytes("file.bin")
        assert result.content == b"updated"

    def test_write_bytes_append_mode(self, fs: Filesystem) -> None:
        """write_bytes() with mode='append' should append content."""
        fs.write_bytes("file.bin", b"hello")
        fs.write_bytes("file.bin", b" world", mode="append")
        result = fs.read_bytes("file.bin")
        assert result.content == b"hello world"

    def test_write_bytes_append_bytes_written(self, fs: Filesystem) -> None:
        """write_bytes() with mode='append' should report bytes written, not total size."""
        fs.write_bytes("file.bin", b"hello")  # 5 bytes
        result = fs.write_bytes("file.bin", b" world", mode="append")  # 6 bytes
        # bytes_written should be 6 (what we just wrote), not 11 (total size)
        assert result.bytes_written == 6

    def test_write_bytes_creates_parents(self, fs: Filesystem) -> None:
        """write_bytes() should create parent directories by default."""
        fs.write_bytes("dir1/dir2/file.bin", b"content")
        assert fs.exists("dir1/dir2/file.bin")

    def test_write_bytes_binary_content(self, fs: Filesystem) -> None:
        """write_bytes() should handle binary content that can't be decoded as UTF-8."""
        binary_data = b"\xff\xfe\x00\x01\x80\x81"  # Invalid UTF-8
        fs.write_bytes("binary.bin", binary_data)
        result = fs.read_bytes("binary.bin")
        assert result.content == binary_data

    def test_write_bytes_content_too_long_fails(
        self, fs: Filesystem, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """write_bytes() should fail if content exceeds max size."""
        # Patch both implementations since we don't know which one is being tested
        monkeypatch.setattr(
            "weakincentives.contrib.tools.filesystem_memory.MAX_WRITE_BYTES", 100
        )
        monkeypatch.setattr("weakincentives.filesystem._host.MAX_WRITE_BYTES", 100)
        with pytest.raises(ValueError, match=r"[Cc]ontent exceeds maximum"):
            fs.write_bytes("file.bin", b"x" * 101)

    def test_write_bytes_to_root_fails(self, fs: Filesystem) -> None:
        """write_bytes() to root directory should fail."""
        with pytest.raises(ValueError, match=r"[Cc]annot write to root"):
            fs.write_bytes(".", b"content")

    def test_write_bytes_without_parents_fails(self, fs: Filesystem) -> None:
        """write_bytes() with create_parents=False should fail if parent missing."""
        with pytest.raises(FileNotFoundError):
            fs.write_bytes("dir1/dir2/file.bin", b"content", create_parents=False)

    # -------------------------------------------------------------------------
    # Stat Operation
    # -------------------------------------------------------------------------

    def test_stat_file(self, fs: Filesystem) -> None:
        """stat() should return correct metadata for files."""
        fs.write("file.txt", "hello")
        stat = fs.stat("file.txt")
        assert stat.path == "file.txt"
        assert stat.is_file is True
        assert stat.is_directory is False
        assert stat.size_bytes == len("hello")

    def test_stat_directory(self, fs: Filesystem) -> None:
        """stat() should return correct metadata for directories."""
        fs.mkdir("mydir")
        stat = fs.stat("mydir")
        assert stat.is_directory is True
        assert stat.is_file is False

    def test_stat_missing_raises(self, fs: Filesystem) -> None:
        """stat() should raise FileNotFoundError for missing paths."""
        with pytest.raises(FileNotFoundError):
            fs.stat("missing")

    # -------------------------------------------------------------------------
    # List Operation
    # -------------------------------------------------------------------------

    def test_list_root(self, fs: Filesystem) -> None:
        """list() should return entries in root directory."""
        fs.write("file.txt", "content")
        fs.mkdir("mydir")
        entries = fs.list(".")
        assert len(entries) == 2

    def test_list_subdirectory(self, fs: Filesystem) -> None:
        """list() should return entries in subdirectory."""
        fs.write("dir/file.txt", "content")
        entries = fs.list("dir")
        assert len(entries) == 1
        assert entries[0].name == "file.txt"

    def test_list_entries_sorted(self, fs: Filesystem) -> None:
        """list() should return entries sorted by name."""
        fs.write("c.txt", "c")
        fs.write("a.txt", "a")
        fs.write("b.txt", "b")
        entries = fs.list(".")
        names = [e.name for e in entries]
        assert names == sorted(names)

    def test_list_missing_raises(self, fs: Filesystem) -> None:
        """list() should raise FileNotFoundError for missing directories."""
        with pytest.raises(FileNotFoundError):
            fs.list("missing")

    def test_list_file_raises(self, fs: Filesystem) -> None:
        """list() should raise NotADirectoryError for files."""
        fs.write("file.txt", "content")
        with pytest.raises(NotADirectoryError):
            fs.list("file.txt")

    # -------------------------------------------------------------------------
    # Glob Operation
    # -------------------------------------------------------------------------

    def test_glob_matches_pattern(self, fs: Filesystem) -> None:
        """glob() should match files by pattern."""
        fs.write("a.txt", "a")
        fs.write("b.txt", "b")
        fs.write("c.py", "c")
        matches = fs.glob("*.txt")
        assert len(matches) == 2

    def test_glob_in_subdirectory(self, fs: Filesystem) -> None:
        """glob() should support path parameter."""
        fs.write("src/main.py", "code")
        fs.write("tests/test.py", "test")
        matches = fs.glob("*.py", path="src")
        assert len(matches) == 1

    def test_glob_returns_only_files(self, fs: Filesystem) -> None:
        """glob() should return only files, not directories."""
        fs.mkdir("mydir")
        fs.write("file.txt", "content")
        matches = fs.glob("*")
        assert all(m.is_file for m in matches)

    def test_glob_results_sorted(self, fs: Filesystem) -> None:
        """glob() should return results sorted by path."""
        fs.write("c.txt", "c")
        fs.write("a.txt", "a")
        fs.write("b.txt", "b")
        matches = fs.glob("*.txt")
        paths = [m.path for m in matches]
        assert paths == sorted(paths)

    # -------------------------------------------------------------------------
    # Grep Operation
    # -------------------------------------------------------------------------

    def test_grep_finds_matches(self, fs: Filesystem) -> None:
        """grep() should find regex matches in files."""
        fs.write("file.txt", "hello world\nfoo bar\nhello again")
        matches = fs.grep("hello")
        assert len(matches) == 2

    def test_grep_with_glob_filter(self, fs: Filesystem) -> None:
        """grep() should support glob filter."""
        fs.write("file.txt", "hello")
        fs.write("file.py", "hello")
        matches = fs.grep("hello", glob="*.txt")
        assert len(matches) == 1

    def test_grep_with_max_matches(self, fs: Filesystem) -> None:
        """grep() should respect max_matches limit."""
        fs.write("file.txt", "match\nmatch\nmatch")
        matches = fs.grep("match", max_matches=2)
        assert len(matches) == 2

    def test_grep_in_subdirectory(self, fs: Filesystem) -> None:
        """grep() should support path parameter."""
        fs.write("src/code.py", "hello world")
        fs.write("docs/readme.md", "hello docs")
        matches = fs.grep("hello", path="src")
        assert len(matches) == 1

    def test_grep_invalid_regex_raises(self, fs: Filesystem) -> None:
        """grep() should raise ValueError for invalid regex."""
        fs.write("file.txt", "content")
        with pytest.raises(ValueError, match=r"[Ii]nvalid regex"):
            fs.grep("[invalid")  # Unclosed bracket

    def test_grep_results_sorted(self, fs: Filesystem) -> None:
        """grep() should return results sorted by (path, line_number)."""
        fs.write("b.txt", "match")
        fs.write("a.txt", "match\nmatch")
        matches = fs.grep("match")
        # Check sorting by path, then line number
        for i in range(len(matches) - 1):
            curr = (matches[i].path, matches[i].line_number)
            next_ = (matches[i + 1].path, matches[i + 1].line_number)
            assert curr <= next_

    def test_grep_skips_binary_files(self, fs: Filesystem) -> None:
        """grep() should skip files that can't be decoded as UTF-8."""
        binary_data = b"\xff\xfe\x00\x01\x80\x81"  # Invalid UTF-8
        fs.write_bytes("binary.bin", binary_data)
        fs.write("text.txt", "match")
        matches = fs.grep(".*")
        # Should only match text.txt
        assert len(matches) == 1
        assert matches[0].path == "text.txt"

    # -------------------------------------------------------------------------
    # Delete Operation
    # -------------------------------------------------------------------------

    def test_delete_file(self, fs: Filesystem) -> None:
        """delete() should remove a file."""
        fs.write("file.txt", "content")
        fs.delete("file.txt")
        assert fs.exists("file.txt") is False

    def test_delete_empty_directory(self, fs: Filesystem) -> None:
        """delete() should remove an empty directory."""
        fs.mkdir("mydir")
        fs.delete("mydir")
        assert fs.exists("mydir") is False

    def test_delete_nonempty_directory_fails(self, fs: Filesystem) -> None:
        """delete() should fail for non-empty directory without recursive."""
        fs.write("mydir/file.txt", "content")
        with pytest.raises(OSError, match="not empty"):
            fs.delete("mydir", recursive=False)

    def test_delete_nonempty_directory_recursive(self, fs: Filesystem) -> None:
        """delete() with recursive=True should remove non-empty directory."""
        fs.write("mydir/file.txt", "content")
        fs.delete("mydir", recursive=True)
        assert fs.exists("mydir") is False

    def test_delete_nested_recursive(self, fs: Filesystem) -> None:
        """delete() with recursive=True should remove nested directories."""
        fs.mkdir("a/b/c", parents=True)
        fs.write("a/b/c/file.txt", "content")
        fs.write("a/b/file2.txt", "content2")
        fs.delete("a", recursive=True)
        assert fs.exists("a") is False
        assert fs.exists("a/b") is False
        assert fs.exists("a/b/c") is False

    def test_delete_missing_raises(self, fs: Filesystem) -> None:
        """delete() should raise FileNotFoundError for missing paths."""
        with pytest.raises(FileNotFoundError):
            fs.delete("missing")

    # -------------------------------------------------------------------------
    # Mkdir Operation
    # -------------------------------------------------------------------------

    def test_mkdir_creates_directory(self, fs: Filesystem) -> None:
        """mkdir() should create a directory."""
        fs.mkdir("mydir")
        assert fs.exists("mydir")
        assert fs.stat("mydir").is_directory

    def test_mkdir_with_parents(self, fs: Filesystem) -> None:
        """mkdir() with parents=True should create parent directories."""
        fs.mkdir("a/b/c", parents=True)
        assert fs.exists("a/b/c")

    def test_mkdir_without_parents_fails(self, fs: Filesystem) -> None:
        """mkdir() with parents=False should fail if parent missing."""
        with pytest.raises(FileNotFoundError):
            fs.mkdir("a/b/c", parents=False)

    def test_mkdir_exist_ok(self, fs: Filesystem) -> None:
        """mkdir() with exist_ok=True should not raise for existing directory."""
        fs.mkdir("mydir")
        fs.mkdir("mydir", exist_ok=True)  # Should not raise
        assert fs.exists("mydir")

    def test_mkdir_exist_not_ok_fails(self, fs: Filesystem) -> None:
        """mkdir() with exist_ok=False should raise for existing directory."""
        fs.mkdir("mydir")
        with pytest.raises(FileExistsError):
            fs.mkdir("mydir", exist_ok=False)

    def test_mkdir_over_file_fails(self, fs: Filesystem) -> None:
        """mkdir() should fail if path is an existing file."""
        fs.write("file", "content")
        with pytest.raises(FileExistsError):
            fs.mkdir("file")

    def test_mkdir_root_is_noop(self, fs: Filesystem) -> None:
        """mkdir() on root should be a no-op."""
        fs.mkdir(".")  # Should not raise

    def test_mkdir_single_level_without_parents(self, fs: Filesystem) -> None:
        """mkdir() should work for single-level without parents."""
        fs.mkdir("topdir", parents=False)
        assert fs.exists("topdir")

    # -------------------------------------------------------------------------
    # Path Normalization
    # -------------------------------------------------------------------------

    def test_path_with_parent_reference(self, fs: Filesystem) -> None:
        """Paths with .. should be normalized correctly."""
        fs.mkdir("a/b/c", parents=True)
        fs.write("a/b/c/file.txt", "content")
        result = fs.read("a/b/../b/c/file.txt")
        assert result.content == "content"

    def test_utf8_path_handling(self, fs: Filesystem) -> None:
        """Filesystem should handle UTF-8 characters in paths."""
        # Test various UTF-8 characters: accented, emoji, CJK
        fs.write(
            "caf\u00e9/\u65e5\u672c\u8a9e/\u0444\u0430\u0439\u043b.txt",
            "content with \u00e9mojis \U0001f389",
        )
        assert fs.exists("caf\u00e9/\u65e5\u672c\u8a9e/\u0444\u0430\u0439\u043b.txt")
        result = fs.read("caf\u00e9/\u65e5\u672c\u8a9e/\u0444\u0430\u0439\u043b.txt")
        assert result.content == "content with \u00e9mojis \U0001f389"


__all__ = [
    "FilesystemCoreValidationSuite",
]

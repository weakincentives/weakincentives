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

"""Generic validation suite for Filesystem protocol implementations.

This module provides a reusable test suite that validates any implementation
of the Filesystem protocol. Tests are designed to be subclassed with a
concrete filesystem factory.

Example usage::

    from tests.helpers.filesystem import FilesystemValidationSuite

    class TestMyFilesystem(FilesystemValidationSuite):
        @pytest.fixture
        def fs(self) -> MyFilesystem:
            return MyFilesystem()

All tests in the suite will run against the filesystem returned by the
``fs`` fixture. Implementations must provide this fixture.
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

import pytest

from weakincentives.filesystem import (
    READ_ENTIRE_FILE,
    Filesystem,
    FilesystemSnapshot,
    SnapshotableFilesystem,
)


class FilesystemValidationSuite:
    """Abstract test suite for Filesystem protocol compliance.

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
        fs.write("café/日本語/файл.txt", "content with émojis 🎉")
        assert fs.exists("café/日本語/файл.txt")
        result = fs.read("café/日本語/файл.txt")
        assert result.content == "content with émojis 🎉"

    # -------------------------------------------------------------------------
    # Streaming Read Operations (open_read)
    # -------------------------------------------------------------------------

    def test_open_read_basic(self, fs: Filesystem) -> None:
        """open_read() should return a ByteReader for reading file content."""
        fs.write_bytes("file.bin", b"hello world")
        with fs.open_read("file.bin") as reader:
            content = reader.read()
            assert content == b"hello world"
            assert reader.path == "file.bin"
            assert reader.size == 11

    def test_open_read_chunks(self, fs: Filesystem) -> None:
        """open_read() should support chunk iteration."""
        data = b"a" * 100000  # 100KB
        fs.write_bytes("large.bin", data)
        with fs.open_read("large.bin") as reader:
            chunks = list(reader.chunks(size=10000))  # 10KB chunks
            assert len(chunks) == 10
            assert b"".join(chunks) == data

    def test_open_read_default_iteration(self, fs: Filesystem) -> None:
        """open_read() should support default chunk iteration via __iter__."""
        data = b"x" * 1000
        fs.write_bytes("file.bin", data)
        with fs.open_read("file.bin") as reader:
            chunks = list(reader)
            assert b"".join(chunks) == data

    def test_open_read_seek(self, fs: Filesystem) -> None:
        """open_read() should support seek operations."""
        fs.write_bytes("file.bin", b"0123456789")
        with fs.open_read("file.bin") as reader:
            pos = reader.seek(5)
            assert pos == 5
            assert reader.position == 5
            content = reader.read()
            assert content == b"56789"

    def test_open_read_seek_from_end(self, fs: Filesystem) -> None:
        """open_read() should support seeking from end."""
        fs.write_bytes("file.bin", b"0123456789")
        with fs.open_read("file.bin") as reader:
            pos = reader.seek(-3, 2)  # whence=2 is from end
            assert pos == 7
            content = reader.read()
            assert content == b"789"

    def test_open_read_seek_invalid_whence_raises(self, fs: Filesystem) -> None:
        """open_read() should raise ValueError for invalid whence."""
        fs.write_bytes("file.bin", b"content")
        with fs.open_read("file.bin") as reader:
            with pytest.raises(ValueError, match="whence"):
                reader.seek(0, 99)  # Invalid whence value

    def test_open_read_missing_file_raises(self, fs: Filesystem) -> None:
        """open_read() should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            fs.open_read("missing.bin")

    def test_open_read_directory_raises(self, fs: Filesystem) -> None:
        """open_read() should raise IsADirectoryError for directories."""
        fs.mkdir("mydir")
        with pytest.raises(IsADirectoryError):
            fs.open_read("mydir")

    def test_open_read_closed_raises(self, fs: Filesystem) -> None:
        """Reading from closed reader should raise ValueError."""
        fs.write_bytes("file.bin", b"content")
        reader = fs.open_read("file.bin")
        reader.close()
        with pytest.raises(ValueError, match="closed"):
            reader.read()

    def test_open_read_context_manager(self, fs: Filesystem) -> None:
        """open_read() should work as context manager."""
        fs.write_bytes("file.bin", b"content")
        with fs.open_read("file.bin") as reader:
            assert not reader.closed
            _ = reader.read()
        assert reader.closed

    def test_open_read_close_idempotent(self, fs: Filesystem) -> None:
        """close() on reader should be idempotent."""
        fs.write_bytes("file.bin", b"content")
        reader = fs.open_read("file.bin")
        _ = reader.read()
        reader.close()
        reader.close()  # Should not raise
        assert reader.closed

    # -------------------------------------------------------------------------
    # Streaming Write Operations (open_write)
    # -------------------------------------------------------------------------

    def test_open_write_basic(self, fs: Filesystem) -> None:
        """open_write() should return a ByteWriter for writing file content."""
        with fs.open_write("file.bin", mode="create") as writer:
            written = writer.write(b"hello world")
            assert written == 11
            assert writer.path == "file.bin"
        assert fs.read_bytes("file.bin").content == b"hello world"

    def test_open_write_write_all(self, fs: Filesystem) -> None:
        """open_write() should support write_all for iterables."""
        chunks = [b"hello", b" ", b"world"]
        with fs.open_write("file.bin") as writer:
            total = writer.write_all(chunks)
            assert total == 11
            assert writer.bytes_written == 11
        assert fs.read_bytes("file.bin").content == b"hello world"

    def test_open_write_overwrite_mode(self, fs: Filesystem) -> None:
        """open_write() with mode='overwrite' should replace content."""
        fs.write_bytes("file.bin", b"original")
        with fs.open_write("file.bin", mode="overwrite") as writer:
            writer.write(b"new")
        assert fs.read_bytes("file.bin").content == b"new"

    def test_open_write_append_mode(self, fs: Filesystem) -> None:
        """open_write() with mode='append' should add to existing content."""
        fs.write_bytes("file.bin", b"hello")
        with fs.open_write("file.bin", mode="append") as writer:
            writer.write(b" world")
        assert fs.read_bytes("file.bin").content == b"hello world"

    def test_open_write_create_mode_existing_raises(self, fs: Filesystem) -> None:
        """open_write() with mode='create' should raise if file exists."""
        fs.write_bytes("file.bin", b"existing")
        with pytest.raises(FileExistsError):
            fs.open_write("file.bin", mode="create")

    def test_open_write_creates_parents(self, fs: Filesystem) -> None:
        """open_write() should create parent directories by default."""
        with fs.open_write("a/b/c/file.bin") as writer:
            writer.write(b"content")
        assert fs.exists("a/b/c/file.bin")

    def test_open_write_no_create_parents_raises(self, fs: Filesystem) -> None:
        """open_write() with create_parents=False should raise if parent missing."""
        with pytest.raises(FileNotFoundError):
            fs.open_write("missing_parent/file.bin", create_parents=False)

    def test_open_write_root_raises(self, fs: Filesystem) -> None:
        """open_write() should raise ValueError when trying to write to root."""
        with pytest.raises(ValueError, match="root"):
            fs.open_write(".")

    def test_open_write_closed_raises(self, fs: Filesystem) -> None:
        """Writing to closed writer should raise ValueError."""
        writer = fs.open_write("file.bin")
        writer.close()
        with pytest.raises(ValueError, match="closed"):
            writer.write(b"content")

    def test_open_write_context_manager(self, fs: Filesystem) -> None:
        """open_write() should work as context manager."""
        with fs.open_write("file.bin") as writer:
            assert not writer.closed
            writer.write(b"content")
        assert writer.closed

    def test_open_write_close_idempotent(self, fs: Filesystem) -> None:
        """close() on writer should be idempotent."""
        writer = fs.open_write("file.bin")
        writer.write(b"content")
        writer.close()
        writer.close()  # Should not raise or double-commit
        assert writer.closed
        assert fs.read_bytes("file.bin").content == b"content"

    def test_open_write_abort_on_error_overwrite(self, fs: Filesystem) -> None:
        """On error exit, overwrite should not commit partial writes."""
        fs.write_bytes("file.bin", b"original")

        with pytest.raises(RuntimeError, match="simulated"):
            with fs.open_write("file.bin", mode="overwrite") as writer:
                writer.write(b"partial data that should be discarded")
                msg = "simulated error"
                raise RuntimeError(msg)

        assert fs.read_bytes("file.bin").content == b"original"

    def test_open_write_abort_on_error_create(self, fs: Filesystem) -> None:
        """On error exit, create should not leave partial file."""
        with pytest.raises(RuntimeError, match="simulated"):
            with fs.open_write("new_file.bin", mode="create") as writer:
                writer.write(b"partial data")
                msg = "simulated error"
                raise RuntimeError(msg)

        assert not fs.exists("new_file.bin")

    def test_open_write_abort_after_close_is_noop(self, fs: Filesystem) -> None:
        """Abort after explicit close should not discard committed data."""
        writer = fs.open_write("file.bin")
        writer.write(b"content")
        writer.close()
        # Simulate __exit__ with error after explicit close
        writer.__exit__(RuntimeError, RuntimeError("late"), None)
        assert writer.closed
        assert fs.read_bytes("file.bin").content == b"content"

    # -------------------------------------------------------------------------
    # Streaming Text Operations (open_text)
    # -------------------------------------------------------------------------

    def test_open_text_basic(self, fs: Filesystem) -> None:
        """open_text() should return a TextReader for reading text content."""
        fs.write("file.txt", "line1\nline2\nline3")
        with fs.open_text("file.txt") as reader:
            content = reader.read()
            assert "line1" in content
            assert "line2" in content
            assert reader.path == "file.txt"
            assert reader.encoding == "utf-8"

    def test_open_text_readline(self, fs: Filesystem) -> None:
        """open_text() should support readline()."""
        fs.write("file.txt", "line1\nline2\nline3\n")
        with fs.open_text("file.txt") as reader:
            line1 = reader.readline()
            assert line1 == "line1\n"
            line2 = reader.readline()
            assert line2 == "line2\n"
            assert reader.line_number == 2

    def test_open_text_iteration(self, fs: Filesystem) -> None:
        """open_text() should support line iteration."""
        fs.write("file.txt", "line1\nline2\nline3\n")
        with fs.open_text("file.txt") as reader:
            lines = list(reader)
            assert len(lines) == 3
            assert lines[0] == "line1\n"
            assert reader.line_number == 3

    def test_open_text_lines_with_strip(self, fs: Filesystem) -> None:
        """open_text() lines() should support stripping."""
        fs.write("file.txt", "  line1  \n  line2  \n")
        with fs.open_text("file.txt") as reader:
            lines = list(reader.lines(strip=True))
            assert lines == ["  line1", "  line2"]

    def test_open_text_lines_without_strip(self, fs: Filesystem) -> None:
        """open_text() lines() should preserve whitespace by default."""
        fs.write("file.txt", "  line1  \n  line2  \n")
        with fs.open_text("file.txt") as reader:
            lines = list(reader.lines(strip=False))
            assert lines == ["  line1  \n", "  line2  \n"]

    def test_open_text_missing_file_raises(self, fs: Filesystem) -> None:
        """open_text() should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            fs.open_text("missing.txt")

    def test_open_text_directory_raises(self, fs: Filesystem) -> None:
        """open_text() should raise IsADirectoryError for directories."""
        fs.mkdir("mydir")
        with pytest.raises(IsADirectoryError):
            fs.open_text("mydir")

    def test_open_text_unsupported_encoding_raises(self, fs: Filesystem) -> None:
        """open_text() should raise ValueError for unsupported encoding."""
        fs.write("file.txt", "content")
        with pytest.raises(ValueError, match="utf-8"):
            fs.open_text("file.txt", encoding="latin-1")

    def test_open_text_closed_raises(self, fs: Filesystem) -> None:
        """Reading from closed TextReader should raise ValueError."""
        fs.write("file.txt", "content")
        reader = fs.open_text("file.txt")
        reader.close()
        with pytest.raises(ValueError, match="closed"):
            reader.read()

    def test_open_text_context_manager(self, fs: Filesystem) -> None:
        """open_text() should work as context manager."""
        fs.write("file.txt", "content")
        with fs.open_text("file.txt") as reader:
            assert not reader.closed
            _ = reader.read()
        assert reader.closed

    # -------------------------------------------------------------------------
    # Streaming Copy Pattern
    # -------------------------------------------------------------------------

    def test_streaming_copy(self, fs: Filesystem) -> None:
        """Streaming copy should work with open_read and open_write."""
        data = b"x" * 50000  # 50KB
        fs.write_bytes("source.bin", data)

        with fs.open_read("source.bin") as src:
            with fs.open_write("dest.bin") as dst:
                dst.write_all(src)

        assert fs.read_bytes("dest.bin").content == data


class ReadOnlyFilesystemValidationSuite:
    """Test suite for read-only Filesystem behavior.

    Subclasses must implement the ``fs_readonly`` fixture to provide a
    read-only filesystem instance to test.
    """

    @pytest.fixture
    @abstractmethod
    def fs_readonly(self) -> Filesystem:
        """Provide a read-only filesystem instance for testing."""
        ...

    def test_read_only_property_true(self, fs_readonly: Filesystem) -> None:
        """Read-only filesystem should have read_only=True."""
        assert fs_readonly.read_only is True

    def test_write_fails_on_readonly(self, fs_readonly: Filesystem) -> None:
        """write() should raise PermissionError on read-only filesystem."""
        with pytest.raises(PermissionError):
            fs_readonly.write("file.txt", "content")

    def test_delete_fails_on_readonly(self, fs_readonly: Filesystem) -> None:
        """delete() should raise PermissionError on read-only filesystem."""
        with pytest.raises(PermissionError):
            fs_readonly.delete("file.txt")

    def test_mkdir_fails_on_readonly(self, fs_readonly: Filesystem) -> None:
        """mkdir() should raise PermissionError on read-only filesystem."""
        with pytest.raises(PermissionError):
            fs_readonly.mkdir("mydir")

    def test_write_bytes_fails_on_readonly(self, fs_readonly: Filesystem) -> None:
        """write_bytes() should raise PermissionError on read-only filesystem."""
        with pytest.raises(PermissionError):
            fs_readonly.write_bytes("file.bin", b"content")

    def test_open_write_fails_on_readonly(self, fs_readonly: Filesystem) -> None:
        """open_write() should raise PermissionError on read-only filesystem."""
        with pytest.raises(PermissionError):
            fs_readonly.open_write("file.bin")


class SnapshotableFilesystemValidationSuite:
    """Test suite for SnapshotableFilesystem protocol compliance.

    Subclasses must implement the ``fs_snapshotable`` fixture to provide
    a snapshotable filesystem instance to test.
    """

    @pytest.fixture
    @abstractmethod
    def fs_snapshotable(self) -> SnapshotableFilesystem:
        """Provide a snapshotable filesystem instance for testing."""
        ...

    def test_implements_snapshotable_protocol(
        self, fs_snapshotable: SnapshotableFilesystem
    ) -> None:
        """Filesystem should implement SnapshotableFilesystem protocol."""
        assert isinstance(fs_snapshotable, SnapshotableFilesystem)

    def test_snapshot_returns_filesystem_snapshot(
        self, fs_snapshotable: SnapshotableFilesystem
    ) -> None:
        """snapshot() should return a FilesystemSnapshot."""
        fs_snapshotable.write("file.txt", "content")
        snapshot = fs_snapshotable.snapshot()
        assert isinstance(snapshot, FilesystemSnapshot)
        assert isinstance(snapshot.snapshot_id, UUID)

    def test_snapshot_with_tag(self, fs_snapshotable: SnapshotableFilesystem) -> None:
        """snapshot() should accept an optional tag."""
        fs_snapshotable.write("file.txt", "content")
        snapshot = fs_snapshotable.snapshot(tag="my-tag")
        assert snapshot.tag == "my-tag"

    def test_snapshot_and_restore_roundtrip(
        self, fs_snapshotable: SnapshotableFilesystem
    ) -> None:
        """Basic snapshot and restore should preserve file contents."""
        fs_snapshotable.write("config.py", "DEBUG = True")
        snapshot_v1 = fs_snapshotable.snapshot(tag="initial")

        # Modify file
        fs_snapshotable.write("config.py", "DEBUG = False")
        assert fs_snapshotable.read("config.py").content == "DEBUG = False"

        # Restore
        fs_snapshotable.restore(snapshot_v1)
        assert fs_snapshotable.read("config.py").content == "DEBUG = True"

    def test_restore_removes_new_files(
        self, fs_snapshotable: SnapshotableFilesystem
    ) -> None:
        """restore() should remove files created after snapshot."""
        fs_snapshotable.write("original.txt", "original")
        snapshot = fs_snapshotable.snapshot()

        # Add new file
        fs_snapshotable.write("new.txt", "new content")
        assert fs_snapshotable.exists("new.txt")

        # Restore should remove the new file
        fs_snapshotable.restore(snapshot)
        assert not fs_snapshotable.exists("new.txt")
        assert fs_snapshotable.exists("original.txt")

    def test_restore_restores_deleted_files(
        self, fs_snapshotable: SnapshotableFilesystem
    ) -> None:
        """restore() should bring back deleted files."""
        fs_snapshotable.write("file.txt", "content")
        snapshot = fs_snapshotable.snapshot()

        # Delete file
        fs_snapshotable.delete("file.txt")
        assert not fs_snapshotable.exists("file.txt")

        # Restore should bring it back
        fs_snapshotable.restore(snapshot)
        assert fs_snapshotable.exists("file.txt")
        assert fs_snapshotable.read("file.txt").content == "content"

    def test_multiple_snapshots(self, fs_snapshotable: SnapshotableFilesystem) -> None:
        """Multiple snapshots can be restored independently."""
        fs_snapshotable.write("file.txt", "v1")
        snapshot_v1 = fs_snapshotable.snapshot(tag="v1")

        fs_snapshotable.write("file.txt", "v2")
        snapshot_v2 = fs_snapshotable.snapshot(tag="v2")

        fs_snapshotable.write("file.txt", "v3")

        # Restore to v1
        fs_snapshotable.restore(snapshot_v1)
        assert fs_snapshotable.read("file.txt").content == "v1"

        # Restore to v2
        fs_snapshotable.restore(snapshot_v2)
        assert fs_snapshotable.read("file.txt").content == "v2"

    def test_restore_directories(self, fs_snapshotable: SnapshotableFilesystem) -> None:
        """restore() should restore directory structure."""
        fs_snapshotable.mkdir("a/b/c", parents=True)
        fs_snapshotable.write("a/b/c/file.txt", "content")
        snapshot = fs_snapshotable.snapshot()

        # Delete directory
        fs_snapshotable.delete("a", recursive=True)
        assert not fs_snapshotable.exists("a/b/c")

        # Restore
        fs_snapshotable.restore(snapshot)
        assert fs_snapshotable.exists("a/b/c")
        assert fs_snapshotable.read("a/b/c/file.txt").content == "content"


__all__ = [
    "FilesystemValidationSuite",
    "ReadOnlyFilesystemValidationSuite",
    "SnapshotableFilesystemValidationSuite",
]

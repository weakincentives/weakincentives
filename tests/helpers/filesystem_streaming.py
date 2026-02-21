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

"""Streaming and advanced validation suites for Filesystem protocol implementations.

This module provides additional test suites that complement
``FilesystemValidationSuite`` from ``tests.helpers.filesystem``:

- ``FilesystemStreamingValidationSuite``: Tests for streaming I/O operations
  (open_write, open_text, streaming copy pattern).
- ``ReadOnlyFilesystemValidationSuite``: Tests for read-only Filesystem behavior.
- ``SnapshotableFilesystemValidationSuite``: Tests for SnapshotableFilesystem
  protocol compliance.

Example usage::

    from tests.helpers.filesystem import FilesystemValidationSuite
    from tests.helpers.filesystem_streaming import FilesystemStreamingValidationSuite

    class TestMyFilesystem(FilesystemValidationSuite, FilesystemStreamingValidationSuite):
        @pytest.fixture
        def fs(self) -> MyFilesystem:
            return MyFilesystem()
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

import pytest

from weakincentives.filesystem import (
    Filesystem,
    FilesystemSnapshot,
    SnapshotableFilesystem,
)


class FilesystemStreamingValidationSuite:
    """Test suite for streaming write and text I/O Filesystem operations.

    Subclasses must implement the ``fs`` fixture to provide a filesystem
    instance to test. The filesystem should be empty at the start of each test.

    This suite validates:
    - Streaming write operations (open_write)
    - Streaming text read operations (open_text)
    - Streaming copy pattern
    """

    @pytest.fixture
    @abstractmethod
    def fs(self) -> Filesystem:
        """Provide a fresh filesystem instance for testing."""
        ...

    # -------------------------------------------------------------------------
    # Streaming Read lifecycle (open_read - continued from FilesystemValidationSuite)
    # -------------------------------------------------------------------------

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
    "FilesystemStreamingValidationSuite",
    "ReadOnlyFilesystemValidationSuite",
    "SnapshotableFilesystemValidationSuite",
]

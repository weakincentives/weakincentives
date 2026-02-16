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

"""Unit tests for the filesystem streaming implementations.

These tests cover edge cases in the stream implementations that may not
be exercised through the full filesystem protocol tests.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from weakincentives.filesystem._streams import (
    DefaultTextReader,
    HostByteReader,
    HostByteWriter,
    MemoryByteReader,
    MemoryByteWriter,
)


class TestMemoryByteReader:
    """Tests for MemoryByteReader implementation."""

    def test_from_bytes_creates_reader(self) -> None:
        """from_bytes should create a reader from content."""
        reader = MemoryByteReader.from_bytes("test.bin", b"hello world")
        assert reader.path == "test.bin"
        assert reader.size == 11
        assert not reader.closed

    def test_read_returns_content(self) -> None:
        """read() should return content from buffer."""
        reader = MemoryByteReader.from_bytes("test.bin", b"hello")
        content = reader.read()
        assert content == b"hello"

    def test_position_tracks_reads(self) -> None:
        """position should track current read position."""
        reader = MemoryByteReader.from_bytes("test.bin", b"hello world")
        assert reader.position == 0
        reader.read(5)
        assert reader.position == 5

    def test_seek_returns_position(self) -> None:
        """seek should return new position."""
        reader = MemoryByteReader.from_bytes("test.bin", b"hello world")
        pos = reader.seek(5)
        assert pos == 5
        assert reader.position == 5

    def test_seek_invalid_whence_raises(self) -> None:
        """seek with invalid whence should raise ValueError."""
        reader = MemoryByteReader.from_bytes("test.bin", b"content")
        with pytest.raises(ValueError, match="whence"):
            reader.seek(0, 99)

    def test_chunks_yields_data(self) -> None:
        """chunks should yield data in specified sizes."""
        data = b"a" * 100
        reader = MemoryByteReader.from_bytes("test.bin", data)
        chunks = list(reader.chunks(size=30))
        assert len(chunks) == 4  # 30 + 30 + 30 + 10
        assert sum(len(c) for c in chunks) == 100

    def test_context_manager_closes(self) -> None:
        """Context manager should close the reader."""
        reader = MemoryByteReader.from_bytes("test.bin", b"content")
        with reader:
            assert not reader.closed
        assert reader.closed

    def test_close_idempotent(self) -> None:
        """close() should be safe to call multiple times."""
        reader = MemoryByteReader.from_bytes("test.bin", b"content")
        reader.close()
        assert reader.closed
        reader.close()  # Should not raise and not double-close
        assert reader.closed

    def test_read_after_close_raises(self) -> None:
        """Reading after close should raise ValueError."""
        reader = MemoryByteReader.from_bytes("test.bin", b"content")
        reader.close()
        with pytest.raises(ValueError, match="closed"):
            reader.read()

    def test_seek_after_close_raises(self) -> None:
        """Seeking after close should raise ValueError."""
        reader = MemoryByteReader.from_bytes("test.bin", b"content")
        reader.close()
        with pytest.raises(ValueError, match="closed"):
            reader.seek(0)

    def test_position_after_close_raises(self) -> None:
        """Accessing position after close should raise ValueError."""
        reader = MemoryByteReader.from_bytes("test.bin", b"content")
        reader.close()
        with pytest.raises(ValueError, match="closed"):
            _ = reader.position

    def test_chunks_after_close_raises(self) -> None:
        """Iterating chunks after close should raise ValueError."""
        reader = MemoryByteReader.from_bytes("test.bin", b"content")
        reader.close()
        with pytest.raises(ValueError, match="closed"):
            list(reader.chunks())


class TestMemoryByteWriter:
    """Tests for MemoryByteWriter implementation."""

    def test_create_for_new_file(self) -> None:
        """create() should create writer for new file."""
        writer = MemoryByteWriter.create("test.bin", mode="create")
        assert writer.path == "test.bin"
        assert not writer.closed
        assert writer.bytes_written == 0

    def test_write_returns_bytes_written(self) -> None:
        """write() should return number of bytes written."""
        writer = MemoryByteWriter.create("test.bin", mode="create")
        n = writer.write(b"hello")
        assert n == 5
        assert writer.bytes_written == 5

    def test_write_all_writes_all_chunks(self) -> None:
        """write_all() should write all chunks."""
        writer = MemoryByteWriter.create("test.bin", mode="create")
        total = writer.write_all([b"hello", b" ", b"world"])
        assert total == 11
        assert writer.bytes_written == 11

    def test_get_content_returns_written_data(self) -> None:
        """get_content() should return all written bytes."""
        writer = MemoryByteWriter.create("test.bin", mode="create")
        writer.write(b"hello")
        writer.write(b" world")
        assert writer.get_content() == b"hello world"

    def test_append_mode_includes_existing(self) -> None:
        """append mode should include existing content."""
        writer = MemoryByteWriter.create(
            "test.bin", mode="append", existing_content=b"hello"
        )
        writer.write(b" world")
        assert writer.get_content() == b"hello world"
        assert writer.bytes_written == 6  # Only counts new writes

    def test_context_manager_closes(self) -> None:
        """Context manager should close the writer."""
        writer = MemoryByteWriter.create("test.bin", mode="create")
        with writer:
            writer.write(b"content")
            assert not writer.closed
        assert writer.closed

    def test_close_sets_closed_flag(self) -> None:
        """close() should set closed flag."""
        writer = MemoryByteWriter.create("test.bin", mode="create")
        assert not writer.closed
        writer.close()
        assert writer.closed

    def test_write_after_close_raises(self) -> None:
        """Writing after close should raise ValueError."""
        writer = MemoryByteWriter.create("test.bin", mode="create")
        writer.close()
        with pytest.raises(ValueError, match="closed"):
            writer.write(b"content")

    def test_write_all_after_close_raises(self) -> None:
        """write_all after close should raise ValueError."""
        writer = MemoryByteWriter.create("test.bin", mode="create")
        writer.close()
        with pytest.raises(ValueError, match="closed"):
            writer.write_all([b"content"])


class TestDefaultTextReader:
    """Tests for DefaultTextReader implementation."""

    def test_wrap_creates_reader(self) -> None:
        """wrap() should create a text reader from byte reader."""
        byte_reader = MemoryByteReader.from_bytes("test.txt", b"hello world")
        reader = DefaultTextReader.wrap(byte_reader)
        assert reader.path == "test.txt"
        assert reader.encoding == "utf-8"
        assert not reader.closed

    def test_wrap_unsupported_encoding_raises(self) -> None:
        """wrap() should raise for unsupported encoding."""
        byte_reader = MemoryByteReader.from_bytes("test.txt", b"content")
        with pytest.raises(ValueError, match="utf-8"):
            DefaultTextReader.wrap(byte_reader, encoding="latin-1")

    def test_read_returns_text(self) -> None:
        """read() should return decoded text."""
        byte_reader = MemoryByteReader.from_bytes("test.txt", b"hello world")
        reader = DefaultTextReader.wrap(byte_reader)
        content = reader.read()
        assert content == "hello world"

    def test_readline_returns_line(self) -> None:
        """readline() should return next line."""
        byte_reader = MemoryByteReader.from_bytes("test.txt", b"line1\nline2\n")
        reader = DefaultTextReader.wrap(byte_reader)
        line1 = reader.readline()
        assert line1 == "line1\n"
        assert reader.line_number == 1
        line2 = reader.readline()
        assert line2 == "line2\n"
        assert reader.line_number == 2

    def test_readline_empty_at_eof(self) -> None:
        """readline() should return empty string at EOF without incrementing line number."""
        byte_reader = MemoryByteReader.from_bytes("test.txt", b"line1\n")
        reader = DefaultTextReader.wrap(byte_reader)
        line1 = reader.readline()
        assert line1 == "line1\n"
        assert reader.line_number == 1
        eof = reader.readline()
        assert eof == ""
        assert reader.line_number == 1  # Should not increment for empty line

    def test_iteration_yields_lines(self) -> None:
        """Iteration should yield lines."""
        byte_reader = MemoryByteReader.from_bytes("test.txt", b"line1\nline2\n")
        reader = DefaultTextReader.wrap(byte_reader)
        lines = list(reader)
        assert lines == ["line1\n", "line2\n"]
        assert reader.line_number == 2

    def test_lines_with_strip(self) -> None:
        """lines() with strip should strip trailing whitespace."""
        byte_reader = MemoryByteReader.from_bytes("test.txt", b"  line1  \n  line2  \n")
        reader = DefaultTextReader.wrap(byte_reader)
        lines = list(reader.lines(strip=True))
        assert lines == ["  line1", "  line2"]

    def test_lines_without_strip(self) -> None:
        """lines() without strip should preserve whitespace."""
        byte_reader = MemoryByteReader.from_bytes("test.txt", b"  line1  \n  line2  \n")
        reader = DefaultTextReader.wrap(byte_reader)
        lines = list(reader.lines(strip=False))
        assert lines == ["  line1  \n", "  line2  \n"]

    def test_context_manager_closes(self) -> None:
        """Context manager should close the reader."""
        byte_reader = MemoryByteReader.from_bytes("test.txt", b"content")
        reader = DefaultTextReader.wrap(byte_reader)
        with reader:
            assert not reader.closed
        assert reader.closed

    def test_close_closes_underlying_reader(self) -> None:
        """close() should close the underlying byte reader."""
        byte_reader = MemoryByteReader.from_bytes("test.txt", b"content")
        reader = DefaultTextReader.wrap(byte_reader)
        reader.close()
        assert reader.closed
        assert byte_reader.closed

    def test_close_idempotent(self) -> None:
        """close() should be idempotent."""
        byte_reader = MemoryByteReader.from_bytes("test.txt", b"content")
        reader = DefaultTextReader.wrap(byte_reader)
        reader.close()
        assert reader.closed
        reader.close()  # Should not raise
        assert reader.closed

    def test_read_after_close_raises(self) -> None:
        """Reading after close should raise ValueError."""
        byte_reader = MemoryByteReader.from_bytes("test.txt", b"content")
        reader = DefaultTextReader.wrap(byte_reader)
        reader.close()
        with pytest.raises(ValueError, match="closed"):
            reader.read()

    def test_readline_after_close_raises(self) -> None:
        """readline after close should raise ValueError."""
        byte_reader = MemoryByteReader.from_bytes("test.txt", b"content")
        reader = DefaultTextReader.wrap(byte_reader)
        reader.close()
        with pytest.raises(ValueError, match="closed"):
            reader.readline()

    def test_iteration_after_close_raises(self) -> None:
        """Iteration after close should raise ValueError."""
        byte_reader = MemoryByteReader.from_bytes("test.txt", b"content")
        reader = DefaultTextReader.wrap(byte_reader)
        reader.close()
        with pytest.raises(ValueError, match="closed"):
            list(reader)


class TestHostByteReader:
    """Tests for HostByteReader with host filesystem."""

    def test_open_directory_raises(self, tmp_path: Path) -> None:
        """Opening a directory should raise IsADirectoryError and close handle."""
        subdir = tmp_path / "mydir"
        subdir.mkdir()
        with pytest.raises(IsADirectoryError, match="mydir"):
            HostByteReader.open(subdir, "mydir")

    def test_open_uses_fstat_for_size(self, tmp_path: Path) -> None:
        """open() should get file size from os.fstat, not stat before open."""
        f = tmp_path / "file.bin"
        f.write_bytes(b"hello world")
        reader = HostByteReader.open(f, "file.bin")
        assert reader.size == 11
        reader.close()


class TestHostByteWriter:
    """Tests for HostByteWriter atomic write behavior."""

    def test_overwrite_is_atomic(self, tmp_path: Path) -> None:
        """Overwrite should use temp file + rename for atomicity."""
        f = tmp_path / "file.bin"
        f.write_bytes(b"original")

        with HostByteWriter.open(
            f, "file.bin", mode="overwrite", create_parents=False
        ) as w:
            w.write(b"new content")

        assert f.read_bytes() == b"new content"
        # No temp files should remain
        remaining = list(tmp_path.glob(".wink_tmp_*"))
        assert remaining == []

    def test_abort_on_error_preserves_original(self, tmp_path: Path) -> None:
        """On error exit, original file should be preserved (no partial write)."""
        f = tmp_path / "file.bin"
        f.write_bytes(b"original")

        with pytest.raises(RuntimeError, match="simulated"):
            with HostByteWriter.open(
                f, "file.bin", mode="overwrite", create_parents=False
            ) as w:
                w.write(b"partial data that should be discarded")
                msg = "simulated error"
                raise RuntimeError(msg)

        # Original file should be untouched
        assert f.read_bytes() == b"original"
        # No temp files should remain
        remaining = list(tmp_path.glob(".wink_tmp_*"))
        assert remaining == []

    def test_abort_when_already_closed_is_noop(self, tmp_path: Path) -> None:
        """_abort() on already-closed writer should be a no-op."""
        f = tmp_path / "file.bin"
        writer = HostByteWriter.open(
            f, "file.bin", mode="overwrite", create_parents=False
        )
        writer.write(b"content")
        writer.close()
        assert f.read_bytes() == b"content"
        # Calling _abort after close should not raise or change anything
        writer._abort()
        assert f.read_bytes() == b"content"

    def test_create_mode_atomic(self, tmp_path: Path) -> None:
        """Create mode should atomically fail if file already exists."""
        f = tmp_path / "file.bin"
        f.write_bytes(b"existing")
        with pytest.raises(FileExistsError):
            HostByteWriter.open(f, "file.bin", mode="create", create_parents=False)

    def test_append_mode_direct(self, tmp_path: Path) -> None:
        """Append mode should write directly (no temp file)."""
        f = tmp_path / "file.bin"
        f.write_bytes(b"hello")
        with HostByteWriter.open(
            f, "file.bin", mode="append", create_parents=False
        ) as w:
            w.write(b" world")
        assert f.read_bytes() == b"hello world"

    def test_close_flush_failure_cleans_temp(self, tmp_path: Path) -> None:
        """If flush/fsync fails during close, temp file should be cleaned up."""
        f = tmp_path / "file.bin"
        writer = HostByteWriter.open(
            f, "file.bin", mode="overwrite", create_parents=False
        )
        writer.write(b"data")

        with patch("os.fsync", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                writer.close()

        # Temp file should have been cleaned up
        remaining = list(tmp_path.glob(".wink_tmp_*"))
        assert remaining == []
        assert writer.closed

    def test_abort_on_error_append_mode(self, tmp_path: Path) -> None:
        """Abort in append mode (no temp file) should still close cleanly."""
        f = tmp_path / "file.bin"
        f.write_bytes(b"hello")

        with pytest.raises(RuntimeError, match="simulated"):
            with HostByteWriter.open(
                f, "file.bin", mode="append", create_parents=False
            ) as w:
                w.write(b" world")
                msg = "simulated error"
                raise RuntimeError(msg)

        assert w.closed

    def test_close_flush_failure_create_mode_cleans_file(self, tmp_path: Path) -> None:
        """If flush/fsync fails in create mode, the new file should be removed."""
        f = tmp_path / "new_file.bin"
        writer = HostByteWriter.open(
            f, "new_file.bin", mode="create", create_parents=False
        )
        writer.write(b"data")

        with patch("os.fsync", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                writer.close()

        assert not f.exists()
        assert writer.closed

    def test_close_flush_failure_append_mode(self, tmp_path: Path) -> None:
        """Flush failure in append mode (no temp file) should still propagate."""
        f = tmp_path / "file.bin"
        f.write_bytes(b"hello")
        writer = HostByteWriter.open(f, "file.bin", mode="append", create_parents=False)
        writer.write(b" world")

        with patch("os.fsync", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                writer.close()

        assert writer.closed

    def test_close_flush_failure_closes_handle(self, tmp_path: Path) -> None:
        """File handle must be closed even when flush/fsync raises."""
        f = tmp_path / "file.bin"
        writer = HostByteWriter.open(
            f, "file.bin", mode="overwrite", create_parents=False
        )
        writer.write(b"data")
        handle = writer._handle

        with patch("os.fsync", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                writer.close()

        assert handle.closed

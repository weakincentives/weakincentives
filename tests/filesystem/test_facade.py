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

"""Facade-specific tests: streaming lifecycle, validation, result types."""

from __future__ import annotations

from pathlib import Path

import pytest

from weakincentives.errors import SnapshotRestoreError
from weakincentives.filesystem import (
    FileEntry,
    FileStat,
    Filesystem,
    GlobMatch,
    GrepMatch,
    HostBackend,
    MemoryBackend,
    ReadResult,
    WriteResult,
    normalize_path,
)


@pytest.fixture
def fs() -> Filesystem:
    return Filesystem.in_memory()


class TestConstruction:
    """Facade construction over backends."""

    def test_backend_property_returns_wrapped_backend(self) -> None:
        backend = MemoryBackend()
        fs = Filesystem(backend)
        assert fs.backend is backend

    def test_in_memory_uses_memory_backend(self) -> None:
        assert isinstance(Filesystem.in_memory().backend, MemoryBackend)


class TestReadValidation:
    """Argument validation composed once in the facade."""

    def test_read_negative_offset_raises(self, fs: Filesystem) -> None:
        fs.write("file.txt", "content")
        with pytest.raises(ValueError, match="offset must be non-negative"):
            fs.read("file.txt", offset=-1)

    def test_read_negative_limit_raises(self, fs: Filesystem) -> None:
        fs.write("file.txt", "content")
        with pytest.raises(ValueError, match="limit must be non-negative"):
            fs.read("file.txt", limit=-2)

    def test_read_validates_path_depth(self, fs: Filesystem) -> None:
        """Validation applies to reads, not only writes."""
        deep_path = "/".join(["d"] * 20)
        with pytest.raises(ValueError, match=r"[Pp]ath depth exceeds"):
            fs.read(deep_path)

    def test_read_validates_segment_length(self, fs: Filesystem) -> None:
        with pytest.raises(ValueError, match=r"[Pp]ath segment exceeds"):
            fs.stat("a" * 300)


class TestByteReaderLifecycle:
    """ByteReader behaviors beyond the shared validation suite."""

    def test_position_tracks_reads(self, fs: Filesystem) -> None:
        fs.write_bytes("file.bin", b"hello world")
        with fs.open_read("file.bin") as reader:
            assert reader.position == 0
            _ = reader.read(5)
            assert reader.position == 5

    def test_seek_whence_current(self, fs: Filesystem) -> None:
        fs.write_bytes("file.bin", b"0123456789")
        with fs.open_read("file.bin") as reader:
            _ = reader.read(2)
            pos = reader.seek(3, 1)
            assert pos == 5
            assert reader.read() == b"56789"

    def test_seek_negative_position_raises(self, fs: Filesystem) -> None:
        fs.write_bytes("file.bin", b"content")
        with fs.open_read("file.bin") as reader:
            with pytest.raises(ValueError, match="negative seek"):
                _ = reader.seek(-1)

    def test_seek_after_close_raises(self, fs: Filesystem) -> None:
        fs.write_bytes("file.bin", b"content")
        reader = fs.open_read("file.bin")
        reader.close()
        with pytest.raises(ValueError, match="closed"):
            _ = reader.seek(0)

    def test_position_after_close_raises(self, fs: Filesystem) -> None:
        fs.write_bytes("file.bin", b"content")
        reader = fs.open_read("file.bin")
        reader.close()
        with pytest.raises(ValueError, match="closed"):
            _ = reader.position

    def test_chunks_after_close_raises(self, fs: Filesystem) -> None:
        fs.write_bytes("file.bin", b"content")
        reader = fs.open_read("file.bin")
        reader.close()
        with pytest.raises(ValueError, match="closed"):
            _ = list(reader.chunks())

    def test_reads_paginate_through_backend(self, fs: Filesystem) -> None:
        """Each chunk is a separate backend read (fixed memory footprint)."""
        data = bytes(range(256)) * 1024  # 256 KB
        fs.write_bytes("file.bin", data)
        with fs.open_read("file.bin") as reader:
            chunks = list(reader.chunks(size=65_536))
        assert len(chunks) == 4
        assert b"".join(chunks) == data


class TestByteWriterLifecycle:
    """ByteWriter behaviors beyond the shared validation suite."""

    def test_write_all_after_close_raises(self, fs: Filesystem) -> None:
        writer = fs.open_write("file.bin")
        writer.close()
        with pytest.raises(ValueError, match="closed"):
            _ = writer.write_all([b"chunk"])

    def test_create_mode_commit_fails_if_file_appeared(self, fs: Filesystem) -> None:
        """A file created between open and close fails the create commit."""
        writer = fs.open_write("file.bin", mode="create")
        _ = writer.write(b"buffered")
        fs.write_bytes("file.bin", b"raced ahead")
        with pytest.raises(FileExistsError):
            writer.close()


class TestTextReaderLifecycle:
    """TextReader behaviors beyond the shared validation suite."""

    def test_read_with_size(self, fs: Filesystem) -> None:
        fs.write("file.txt", "hello world")
        with fs.open_text("file.txt") as reader:
            assert reader.read(5) == "hello"
            assert reader.read() == " world"

    def test_readline_at_eof_returns_empty(self, fs: Filesystem) -> None:
        fs.write("file.txt", "only line")
        with fs.open_text("file.txt") as reader:
            assert reader.readline() == "only line"
            assert reader.readline() == ""
            assert reader.line_number == 1

    def test_close_idempotent(self, fs: Filesystem) -> None:
        fs.write("file.txt", "content")
        reader = fs.open_text("file.txt")
        reader.close()
        reader.close()  # Should not raise
        assert reader.closed


class TestSnapshotRefsAcrossBackends:
    """Refs are backend-private: the wrong backend rejects them."""

    def test_memory_rejects_host_style_ref(
        self, fs: Filesystem, tmp_path: Path
    ) -> None:
        host_fs = Filesystem.host(tmp_path)
        host_fs.write("f.txt", "x")
        host_ref = host_fs.snapshot()
        backend = host_fs.backend
        with pytest.raises(SnapshotRestoreError):
            fs.restore(host_ref)
        assert isinstance(backend, HostBackend)
        backend.cleanup()


class TestDataclassRendering:
    """Result dataclasses render their key fields."""

    def test_file_stat_str(self) -> None:
        stat = FileStat(
            path="file.txt", is_file=True, is_directory=False, size_bytes=100
        )
        assert "file.txt" in str(stat)

    def test_file_entry_str(self) -> None:
        entry = FileEntry(
            name="file.txt", path="file.txt", is_file=True, is_directory=False
        )
        assert "file.txt" in str(entry)

    def test_glob_match_str(self) -> None:
        assert "file.txt" in str(GlobMatch(path="file.txt", is_file=True))

    def test_grep_match_str(self) -> None:
        match = GrepMatch(
            path="file.txt",
            line_number=1,
            line_content="hello world",
            match_start=0,
            match_end=5,
        )
        assert "file.txt" in str(match)
        assert "hello" in str(match)

    def test_read_result_str(self) -> None:
        result = ReadResult(
            content="hello",
            path="file.txt",
            total_lines=1,
            offset=0,
            limit=100,
            truncated=False,
        )
        assert "file.txt" in str(result)

    def test_write_result_str(self) -> None:
        result = WriteResult(path="file.txt", bytes_written=5, mode="create")
        assert "file.txt" in str(result)


class TestNormalizePath:
    """Tests for normalize_path utility function."""

    def test_normalize_path_with_parent_refs_and_result(self) -> None:
        assert normalize_path("a/b/../c") == "a/c"
        assert normalize_path("a/b/c/../../d") == "a/d"

    def test_normalize_path_with_leading_dotdot(self) -> None:
        assert normalize_path("../file.txt") == "file.txt"
        assert normalize_path("../../file.txt") == "file.txt"
        assert normalize_path("dir/../../../file.txt") == "file.txt"

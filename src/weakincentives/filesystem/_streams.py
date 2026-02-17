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

"""Streaming protocols and implementations for filesystem I/O.

This module provides streaming abstractions for reading and writing files
with fixed memory footprint, enabling operations on files of arbitrary size.

The design follows a bytes-first approach where all core I/O operates on raw
bytes, with text operations layered on top via lazy UTF-8 decoding.

Protocols:
    ByteReader: Streaming byte reader with seek support and chunk iteration.
    ByteWriter: Streaming byte writer with immediate writes.
    TextReader: Line-oriented text reader with lazy UTF-8 decoding.

Implementations:
    HostByteReader: ByteReader backed by a native file handle.
    HostByteWriter: ByteWriter backed by a native file handle.
    MemoryByteReader: ByteReader backed by an io.BytesIO buffer.
    MemoryByteWriter: ByteWriter backed by an io.BytesIO buffer.
    DefaultTextReader: TextReader wrapping any ByteReader.
"""

from __future__ import annotations

import io
import os
import tempfile
from collections.abc import Buffer, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    IO,
    BinaryIO,
    Final,
    Literal,
    Protocol,
    Self,
    override,
    runtime_checkable,
)

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "ByteReader",
    "ByteWriter",
    "DefaultTextReader",
    "HostByteReader",
    "HostByteWriter",
    "MemoryByteReader",
    "MemoryByteWriter",
    "TextReader",
]

#: Default chunk size for iteration (64KB).
DEFAULT_CHUNK_SIZE: Final[int] = 65_536


# ---------------------------------------------------------------------------
# Stream Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class ByteReader(Protocol):
    """Streaming byte reader with fixed memory footprint.

    Provides sequential and random access to file bytes via read() and seek().
    Supports iteration over chunks for streaming processing.

    Example::

        with filesystem.open_read("large_file.bin") as reader:
            for chunk in reader.chunks(size=65536):
                process(chunk)

        # Or with default chunk size
        with filesystem.open_read("data.bin") as reader:
            for chunk in reader:
                hash_update(chunk)
    """

    @property
    def path(self) -> str:
        """Path being read (relative to filesystem root)."""
        ...

    @property
    def size(self) -> int:
        """Total file size in bytes."""
        ...

    @property
    def position(self) -> int:
        """Current read position in bytes."""
        ...

    @property
    def closed(self) -> bool:
        """True if the reader has been closed."""
        ...

    def read(self, size: int = -1) -> bytes:
        """Read up to size bytes from the current position.

        Args:
            size: Maximum bytes to read. -1 means read to EOF.

        Returns:
            Bytes read. Empty bytes at EOF.

        Raises:
            ValueError: If reader is closed.
        """
        ...

    def seek(self, offset: int, whence: int = 0) -> int:
        """Seek to a position in the file.

        Args:
            offset: Offset relative to whence.
            whence: Reference point (0=start, 1=current, 2=end).

        Returns:
            New absolute position.

        Raises:
            ValueError: If reader is closed or whence is invalid.
        """
        ...

    def __iter__(self) -> Iterator[bytes]:
        """Iterate over chunks of default size (64KB)."""
        ...

    def chunks(self, size: int = DEFAULT_CHUNK_SIZE) -> Iterator[bytes]:
        """Iterate over chunks of specified size.

        Args:
            size: Chunk size in bytes. Defaults to 64KB.

        Yields:
            Byte chunks up to the specified size.
        """
        ...

    def __enter__(self) -> Self:
        """Enter context manager."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager, closing the reader."""
        ...

    def close(self) -> None:
        """Close the reader and release resources."""
        ...


@runtime_checkable
class ByteWriter(Protocol):
    """Streaming byte writer with fixed memory footprint.

    Writes bytes immediately without buffering entire files in memory.

    Example::

        with filesystem.open_write("output.bin", mode="create") as writer:
            for chunk in data_source:
                writer.write(chunk)

        # Or write from an iterable
        with filesystem.open_write("output.bin") as writer:
            writer.write_all(generate_chunks())
    """

    @property
    def path(self) -> str:
        """Path being written (relative to filesystem root)."""
        ...

    @property
    def bytes_written(self) -> int:
        """Total bytes written so far."""
        ...

    @property
    def closed(self) -> bool:
        """True if the writer has been closed."""
        ...

    def write(self, data: bytes) -> int:
        """Write bytes to the file.

        Args:
            data: Bytes to write.

        Returns:
            Number of bytes written.

        Raises:
            ValueError: If writer is closed.
        """
        ...

    def write_all(self, chunks: Iterable[bytes]) -> int:
        """Write all chunks from an iterable.

        Args:
            chunks: Iterable of byte chunks.

        Returns:
            Total bytes written.

        Raises:
            ValueError: If writer is closed.
        """
        ...

    def __enter__(self) -> Self:
        """Enter context manager."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager, closing the writer."""
        ...

    def close(self) -> None:
        """Close the writer and flush any buffered data."""
        ...


@runtime_checkable
class TextReader(Protocol):
    """Line-oriented text reader with lazy UTF-8 decoding.

    Decodes bytes to text lazily as lines are requested, avoiding
    the memory cost of decoding entire files upfront.

    Example::

        with filesystem.open_text("log.txt") as reader:
            for line in reader.lines(strip=True):
                if "ERROR" in line:
                    print(f"Line {reader.line_number}: {line}")
    """

    @property
    def path(self) -> str:
        """Path being read (relative to filesystem root)."""
        ...

    @property
    def encoding(self) -> str:
        """Text encoding (always 'utf-8')."""
        ...

    @property
    def line_number(self) -> int:
        """Current 0-indexed line number (lines read so far)."""
        ...

    @property
    def closed(self) -> bool:
        """True if the reader has been closed."""
        ...

    def readline(self) -> str:
        """Read next line including newline character.

        Returns:
            Next line with newline, or empty string at EOF.

        Raises:
            ValueError: If reader is closed.
            UnicodeDecodeError: If bytes cannot be decoded as UTF-8.
        """
        ...

    def read(self, size: int = -1) -> str:
        """Read up to size characters.

        Args:
            size: Maximum characters to read. -1 means read to EOF.

        Returns:
            Text read. Empty string at EOF.

        Raises:
            ValueError: If reader is closed.
            UnicodeDecodeError: If bytes cannot be decoded as UTF-8.
        """
        ...

    def __iter__(self) -> Iterator[str]:
        """Iterate over lines (including newlines)."""
        ...

    def lines(self, *, strip: bool = False) -> Iterator[str]:
        """Iterate over lines with optional stripping.

        Args:
            strip: If True, strip trailing whitespace from each line.

        Yields:
            Lines from the file.
        """
        ...

    def __enter__(self) -> Self:
        """Enter context manager."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager, closing the reader."""
        ...

    def close(self) -> None:
        """Close the reader and release resources."""
        ...


# ---------------------------------------------------------------------------
# Host Filesystem Implementations
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HostByteReader:
    """ByteReader implementation backed by a native file handle.

    Wraps a host filesystem file for streaming byte reads with
    seek support and chunk iteration.
    """

    _path: str
    _handle: BinaryIO
    _size: int
    _closed: bool = field(default=False, init=False)

    @classmethod
    def open(cls, resolved_path: Path, relative_path: str) -> HostByteReader:
        """Open a file for reading.

        Opens the file first, then gets size from the file descriptor
        via ``os.fstat`` to avoid TOCTOU races.

        Args:
            resolved_path: Absolute path to the file on disk.
            relative_path: Path relative to filesystem root.

        Returns:
            New HostByteReader instance.

        Raises:
            FileNotFoundError: If file does not exist.
            IsADirectoryError: If path is a directory.
        """
        try:
            handle = resolved_path.open("rb")
        except FileNotFoundError:
            raise FileNotFoundError(relative_path) from None
        except IsADirectoryError:
            msg = f"Is a directory: {relative_path}"
            raise IsADirectoryError(msg) from None
        size = os.fstat(handle.fileno()).st_size
        return cls(_path=relative_path, _handle=handle, _size=size)

    @property
    def path(self) -> str:
        """Path being read."""
        return self._path

    @property
    def size(self) -> int:
        """Total file size in bytes."""
        return self._size

    @property
    def position(self) -> int:
        """Current read position in bytes."""
        self._check_closed()
        return self._handle.tell()

    @property
    def closed(self) -> bool:
        """True if the reader has been closed."""
        return self._closed

    def _check_closed(self) -> None:
        """Raise ValueError if closed."""
        if self._closed:
            msg = "I/O operation on closed file"
            raise ValueError(msg)

    def read(self, size: int = -1) -> bytes:
        """Read up to size bytes."""
        self._check_closed()
        return self._handle.read(size)

    def seek(self, offset: int, whence: int = 0) -> int:
        """Seek to position."""
        self._check_closed()
        if whence not in {0, 1, 2}:
            msg = f"Invalid whence value: {whence}"
            raise ValueError(msg)
        return self._handle.seek(offset, whence)

    def __iter__(self) -> Iterator[bytes]:
        """Iterate over chunks of default size."""
        return self.chunks(DEFAULT_CHUNK_SIZE)

    def chunks(self, size: int = DEFAULT_CHUNK_SIZE) -> Iterator[bytes]:
        """Iterate over chunks of specified size."""
        self._check_closed()
        while True:
            chunk = self._handle.read(size)
            if not chunk:
                break
            yield chunk

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager."""
        self.close()

    def close(self) -> None:
        """Close the reader."""
        if not self._closed:
            self._handle.close()
            self._closed = True


@dataclass(slots=True)
class HostByteWriter:
    """ByteWriter implementation backed by a native file handle.

    Wraps a host filesystem file for streaming byte writes.

    Mode behavior:

    ``overwrite``
        Writes to a temporary file in the same directory, atomically
        renamed to the target on ``close()``.  On abort the temp file
        is removed and the original is left untouched.

    ``create``
        Opens the target directly with ``"xb"`` (exclusive create).
        On abort the newly created file is deleted.  No temp-file
        indirection is needed because there is no pre-existing file
        to protect.

    ``append``
        Writes directly to the target file.  Append is inherently
        non-transactional: bytes already written are *not* rolled
        back on abort.
    """

    _path: str
    _handle: BinaryIO
    _final_path: Path | None
    _temp_path: Path | None
    _bytes_written: int = field(default=0, init=False)
    _closed: bool = field(default=False, init=False)

    @classmethod
    def open(
        cls,
        resolved_path: Path,
        relative_path: str,
        *,
        mode: Literal["create", "overwrite", "append"],
        create_parents: bool,
    ) -> HostByteWriter:
        """Open a file for writing.

        For ``create`` mode, uses ``"xb"`` to atomically fail if the file
        already exists (no TOCTOU race) and writes directly to the new
        file.  For ``overwrite`` mode, writes to a temporary file that
        is renamed on close for atomic writes.  For ``append`` mode,
        writes directly.

        Args:
            resolved_path: Absolute path to the file on disk.
            relative_path: Path relative to filesystem root.
            mode: Write mode.
            create_parents: Create parent directories if missing.

        Returns:
            New HostByteWriter instance.

        Raises:
            FileExistsError: If mode="create" and file exists.
            FileNotFoundError: If parent missing and create_parents=False.
        """
        parent_dir = resolved_path.parent

        if not parent_dir.exists():
            if not create_parents:
                raise FileNotFoundError(
                    f"Parent directory does not exist: {parent_dir}"
                )
            parent_dir.mkdir(parents=True, exist_ok=True)

        if mode == "append":
            handle = resolved_path.open("ab")
            return cls(
                _path=relative_path,
                _handle=handle,
                _final_path=None,
                _temp_path=None,
            )

        if mode == "create":
            # "xb" atomically fails if the file exists and writes
            # directly to the target — no temp-file indirection needed
            # since there is no pre-existing file to protect.
            try:
                handle = resolved_path.open("xb")
            except FileExistsError:
                raise FileExistsError(f"File already exists: {relative_path}") from None
            return cls(
                _path=relative_path,
                _handle=handle,
                _final_path=resolved_path,
                _temp_path=None,
            )

        # overwrite: write to temp file, rename on close
        fd, temp_path_str = tempfile.mkstemp(dir=str(parent_dir), prefix=".wink_tmp_")
        temp_path = Path(temp_path_str)
        handle = os.fdopen(fd, "wb")
        return cls(
            _path=relative_path,
            _handle=handle,
            _final_path=resolved_path,
            _temp_path=temp_path,
        )

    @property
    def path(self) -> str:
        """Path being written."""
        return self._path

    @property
    def bytes_written(self) -> int:
        """Total bytes written so far."""
        return self._bytes_written

    @property
    def closed(self) -> bool:
        """True if the writer has been closed."""
        return self._closed

    def _check_closed(self) -> None:
        """Raise ValueError if closed."""
        if self._closed:
            msg = "I/O operation on closed file"
            raise ValueError(msg)

    def write(self, data: bytes) -> int:
        """Write bytes to the file."""
        self._check_closed()
        written = self._handle.write(data)
        self._bytes_written += written
        return written

    def write_all(self, chunks: Iterable[bytes]) -> int:
        """Write all chunks from an iterable."""
        self._check_closed()
        total = 0
        for chunk in chunks:
            total += self.write(chunk)
        return total

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager.

        On error, calls ``_abort()`` to discard writes. On success,
        calls ``close()`` to commit.
        """
        if exc_type is not None:
            self._abort()
        else:
            self.close()

    def _abort(self) -> None:
        """Discard writes and clean up without committing.

        ``overwrite``: removes the temp file; original is preserved.
        ``create``: deletes the newly created file.
        ``append``: closes the handle; bytes already appended are not
        rolled back.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._handle.close()
        finally:
            if self._temp_path is not None:
                self._temp_path.unlink(missing_ok=True)
            elif self._final_path is not None:
                self._final_path.unlink(missing_ok=True)

    def close(self) -> None:
        """Close the writer and commit writes.

        ``overwrite``: flushes, fsyncs, then atomically renames the
        temp file to the target path.  On flush/fsync failure the
        temp file is removed and the original is left untouched.
        ``create``: flushes and fsyncs the file in place.
        ``append``: flushes and fsyncs the file in place.

        For ``create`` and ``append`` modes, if flush/fsync fails
        the file is **left in place** (data may already be persisted
        by the OS).  Use ``_abort()`` when the intent is to discard.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._handle.flush()
            os.fsync(self._handle.fileno())
        except BaseException:
            self._handle.close()
            if self._temp_path is not None:
                # Overwrite mode: discard the temp file; original is untouched.
                self._temp_path.unlink(missing_ok=True)
            # Create/append modes write directly to the target file.
            # Data may already be persisted by the OS, so we leave the
            # file in place rather than destroying potentially-good data.
            # Callers can inspect or retry; _abort() handles cleanup for
            # with-block exceptions where the intent is to discard.
            raise
        self._handle.close()
        if self._final_path is not None and self._temp_path is not None:
            _ = self._temp_path.replace(self._final_path)


# ---------------------------------------------------------------------------
# In-Memory Implementations
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MemoryByteReader:
    """ByteReader implementation backed by an io.BytesIO buffer.

    Used by InMemoryFilesystem for streaming reads from in-memory content.
    Creates a copy of the content to avoid mutation issues.
    """

    _path: str
    _buffer: io.BytesIO
    _size: int
    _closed: bool = field(default=False, init=False)

    @classmethod
    def from_bytes(cls, path: str, content: bytes) -> MemoryByteReader:
        """Create a reader from bytes content.

        Args:
            path: Path for error messages.
            content: Bytes to read from (copied).

        Returns:
            New MemoryByteReader instance.
        """
        buffer = io.BytesIO(content)
        return cls(_path=path, _buffer=buffer, _size=len(content))

    @property
    def path(self) -> str:
        """Path being read."""
        return self._path

    @property
    def size(self) -> int:
        """Total size in bytes."""
        return self._size

    @property
    def position(self) -> int:
        """Current read position."""
        self._check_closed()
        return self._buffer.tell()

    @property
    def closed(self) -> bool:
        """True if the reader has been closed."""
        return self._closed

    def _check_closed(self) -> None:
        """Raise ValueError if closed."""
        if self._closed:
            msg = "I/O operation on closed file"
            raise ValueError(msg)

    def read(self, size: int = -1) -> bytes:
        """Read up to size bytes."""
        self._check_closed()
        return self._buffer.read(size)

    def seek(self, offset: int, whence: int = 0) -> int:
        """Seek to position."""
        self._check_closed()
        if whence not in {0, 1, 2}:
            msg = f"Invalid whence value: {whence}"
            raise ValueError(msg)
        return self._buffer.seek(offset, whence)

    def __iter__(self) -> Iterator[bytes]:
        """Iterate over chunks of default size."""
        return self.chunks(DEFAULT_CHUNK_SIZE)

    def chunks(self, size: int = DEFAULT_CHUNK_SIZE) -> Iterator[bytes]:
        """Iterate over chunks of specified size."""
        self._check_closed()
        while True:
            chunk = self._buffer.read(size)
            if not chunk:
                break
            yield chunk

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager."""
        self.close()

    def close(self) -> None:
        """Close the reader."""
        if not self._closed:
            self._buffer.close()
            self._closed = True


@dataclass(slots=True)
class MemoryByteWriter:
    """ByteWriter implementation that collects bytes in memory.

    Used by InMemoryFilesystem. Collects written bytes in a buffer,
    which is retrieved via get_content() after closing.
    """

    _path: str
    _buffer: io.BytesIO = field(default_factory=io.BytesIO)
    _bytes_written: int = field(default=0, init=False)
    _closed: bool = field(default=False, init=False)
    _initial_content: bytes = field(default=b"")

    @classmethod
    def create(
        cls,
        path: str,
        *,
        mode: Literal["create", "overwrite", "append"],
        existing_content: bytes | None = None,
    ) -> MemoryByteWriter:
        """Create a writer for the given path.

        Args:
            path: Path for metadata.
            mode: Write mode.
            existing_content: Existing file content for append mode.

        Returns:
            New MemoryByteWriter instance.
        """
        buffer = io.BytesIO()
        initial = b""

        if mode == "append" and existing_content:
            initial = existing_content
            _ = buffer.write(existing_content)

        return cls(_path=path, _buffer=buffer, _initial_content=initial)

    @property
    def path(self) -> str:
        """Path being written."""
        return self._path

    @property
    def bytes_written(self) -> int:
        """Total bytes written (excluding initial append content)."""
        return self._bytes_written

    @property
    def closed(self) -> bool:
        """True if the writer has been closed."""
        return self._closed

    def _check_closed(self) -> None:
        """Raise ValueError if closed."""
        if self._closed:
            msg = "I/O operation on closed file"
            raise ValueError(msg)

    def write(self, data: bytes) -> int:
        """Write bytes to the buffer."""
        self._check_closed()
        written = self._buffer.write(data)
        self._bytes_written += written
        return written

    def write_all(self, chunks: Iterable[bytes]) -> int:
        """Write all chunks from an iterable."""
        self._check_closed()
        total = 0
        for chunk in chunks:
            total += self.write(chunk)
        return total

    def get_content(self) -> bytes:
        """Get the complete content written to the buffer.

        Must be called **before** ``close()``; after close the
        underlying buffer is released.
        """
        return self._buffer.getvalue()

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager."""
        self.close()

    def close(self) -> None:
        """Close the writer and release the buffer."""
        if not self._closed:
            self._buffer.close()
            self._closed = True


# ---------------------------------------------------------------------------
# Text Reader Implementation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DefaultTextReader:
    """TextReader implementation wrapping a ByteReader.

    Provides lazy UTF-8 decoding of bytes as lines are requested.
    Uses Python's io.TextIOWrapper for efficient incremental decoding.
    """

    _path: str
    _byte_reader: ByteReader
    _text_wrapper: IO[str]
    _encoding: str = field(default="utf-8")
    _line_number: int = field(default=0, init=False)
    _closed: bool = field(default=False, init=False)

    @classmethod
    def wrap(
        cls, byte_reader: ByteReader, *, encoding: str = "utf-8"
    ) -> DefaultTextReader:
        """Wrap a ByteReader for text reading.

        Args:
            byte_reader: The underlying byte reader.
            encoding: Text encoding (only 'utf-8' supported).

        Returns:
            New DefaultTextReader instance.

        Raises:
            ValueError: If encoding is not 'utf-8'.
        """
        if encoding != "utf-8":
            msg = f"Only 'utf-8' encoding is supported, got: {encoding}"
            raise ValueError(msg)

        # Create a wrapper that reads from the byte reader
        raw_wrapper = _ByteReaderWrapper(byte_reader)
        buffered_wrapper = io.BufferedReader(raw_wrapper)
        text_wrapper = io.TextIOWrapper(buffered_wrapper, encoding=encoding)

        return cls(
            _path=byte_reader.path,
            _byte_reader=byte_reader,
            _text_wrapper=text_wrapper,
            _encoding=encoding,
        )

    @property
    def path(self) -> str:
        """Path being read."""
        return self._path

    @property
    def encoding(self) -> str:
        """Text encoding."""
        return self._encoding

    @property
    def line_number(self) -> int:
        """Current 0-indexed line number."""
        return self._line_number

    @property
    def closed(self) -> bool:
        """True if the reader has been closed."""
        return self._closed

    def _check_closed(self) -> None:
        """Raise ValueError if closed."""
        if self._closed:
            msg = "I/O operation on closed file"
            raise ValueError(msg)

    def readline(self) -> str:
        """Read next line including newline."""
        self._check_closed()
        line = self._text_wrapper.readline()
        if line:
            self._line_number += 1
        return line

    def read(self, size: int = -1) -> str:
        """Read up to size characters."""
        self._check_closed()
        return self._text_wrapper.read(size)

    def __iter__(self) -> Iterator[str]:
        """Iterate over lines (including newlines)."""
        self._check_closed()
        for line in self._text_wrapper:
            self._line_number += 1
            yield line

    def lines(self, *, strip: bool = False) -> Iterator[str]:
        """Iterate over lines with optional stripping."""
        for line in self:
            if strip:
                yield line.rstrip()
            else:
                yield line

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager."""
        self.close()

    def close(self) -> None:
        """Close the reader."""
        if not self._closed:
            self._closed = True
            try:
                self._text_wrapper.close()
            finally:
                self._byte_reader.close()


class _ByteReaderWrapper(io.RawIOBase):
    """Adapter to make ByteReader usable with io.TextIOWrapper.

    Implements the minimal RawIOBase interface needed for TextIOWrapper.
    """

    def __init__(self, byte_reader: ByteReader) -> None:
        """Initialize wrapper.

        Args:
            byte_reader: The underlying byte reader.
        """
        super().__init__()
        self._reader = byte_reader

    @override
    def readable(self) -> bool:
        """Return True - this is a readable stream."""
        return True

    @override
    def readinto(self, buffer: Buffer, /) -> int | None:
        """Read bytes into buffer.

        Args:
            buffer: Buffer to read into.

        Returns:
            Number of bytes read (0 at EOF), or None for non-blocking.
        """
        # Cast to memoryview for slicing support
        mv = memoryview(buffer).cast("B")
        data = self._reader.read(len(mv))
        n = len(data)
        mv[:n] = data
        return n

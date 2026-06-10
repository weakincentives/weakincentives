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

"""Streaming reader/writer types for the ``Filesystem`` facade.

One implementation over any backend: ``ByteReader`` pulls chunks through
``FilesystemBackend.read_range``; ``ByteWriter`` buffers and commits through
one ``FilesystemBackend.write`` call; ``TextReader`` decodes a ``ByteReader``
lazily. Obtain instances via ``Filesystem.open_read``/``open_write``/
``open_text``.
"""

from __future__ import annotations

import io
import os
from collections.abc import Buffer, Iterable, Iterator
from typing import IO, Final, Self, final, override

from ._backend import FilesystemBackend
from ._types import WriteMode

#: Default chunk size for streaming iteration (64KB).
DEFAULT_CHUNK_SIZE: Final[int] = 65_536

_CLOSED_MSG: Final[str] = "I/O operation on closed file"


def _check_open(closed: bool) -> None:
    if closed:
        raise ValueError(_CLOSED_MSG)


@final
class ByteReader:
    """Streaming byte reader over backend ``read_range`` calls.

    Reads chunks on demand with a fixed memory footprint; supports
    ``seek()`` and iteration over chunks. Obtain via
    :meth:`Filesystem.open_read`.
    """

    def __init__(self, backend: FilesystemBackend, path: str, size: int) -> None:
        self._backend = backend
        self._path = path
        self._size = size
        self._position = 0
        self._closed = False

    @property
    def path(self) -> str:
        """Path being read (relative to filesystem root)."""
        return self._path

    @property
    def size(self) -> int:
        """Total file size in bytes at open time."""
        return self._size

    @property
    def position(self) -> int:
        """Current read position in bytes."""
        _check_open(self._closed)
        return self._position

    @property
    def closed(self) -> bool:
        """True if the reader has been closed."""
        return self._closed

    def read(self, size: int = -1) -> bytes:
        """Read up to ``size`` bytes (``-1`` reads to EOF)."""
        _check_open(self._closed)
        length = None if size < 0 else size
        data = self._backend.read_range(
            self._path, offset=self._position, length=length
        )
        self._position += len(data)
        return data

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        """Seek to a position (whence: 0=start, 1=current, 2=end)."""
        _check_open(self._closed)
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            position = self._size + offset
        else:
            msg = f"Invalid whence value: {whence}"
            raise ValueError(msg)
        if position < 0:
            msg = f"negative seek value {position}"
            raise ValueError(msg)
        self._position = position
        return position

    def __iter__(self) -> Iterator[bytes]:
        """Iterate over chunks of default size (64KB)."""
        return self.chunks(DEFAULT_CHUNK_SIZE)

    def chunks(self, size: int = DEFAULT_CHUNK_SIZE) -> Iterator[bytes]:
        """Iterate over chunks of the specified size."""
        _check_open(self._closed)
        while True:
            chunk = self.read(size)
            if not chunk:
                break
            yield chunk

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the reader. Idempotent."""
        self._closed = True


@final
class ByteWriter:
    """Streaming byte writer that commits through one backend ``write``.

    Bytes are buffered and committed atomically on ``close()``; exiting the
    context manager with an exception discards the buffer without touching
    the backend. Obtain via :meth:`Filesystem.open_write`.
    """

    def __init__(self, backend: FilesystemBackend, path: str, mode: WriteMode) -> None:
        self._backend = backend
        self._path = path
        self._mode: WriteMode = mode
        self._buffer = bytearray()
        self._bytes_written = 0
        self._closed = False

    @property
    def path(self) -> str:
        """Path being written (relative to filesystem root)."""
        return self._path

    @property
    def bytes_written(self) -> int:
        """Total bytes written so far."""
        return self._bytes_written

    @property
    def closed(self) -> bool:
        """True if the writer has been closed."""
        return self._closed

    def write(self, data: bytes) -> int:
        """Buffer bytes for the commit on ``close()``."""
        _check_open(self._closed)
        self._buffer.extend(data)
        self._bytes_written += len(data)
        return len(data)

    def write_all(self, chunks: Iterable[bytes]) -> int:
        """Write all chunks from an iterable; returns total bytes."""
        _check_open(self._closed)
        total = 0
        for chunk in chunks:
            total += self.write(chunk)
        return total

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if exc_type is not None:
            self._abort()
        else:
            self.close()

    def _abort(self) -> None:
        """Discard buffered bytes without committing. No-op when closed."""
        if self._closed:
            return
        self._closed = True
        self._buffer.clear()

    def close(self) -> None:
        """Commit buffered bytes to the backend. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._backend.write(self._path, bytes(self._buffer), mode=self._mode)
        self._buffer.clear()


class _ByteReaderRawIO(io.RawIOBase):
    """Adapter exposing a ByteReader as a RawIOBase for TextIOWrapper."""

    def __init__(self, reader: ByteReader) -> None:
        super().__init__()
        self._reader = reader

    @override
    def readable(self) -> bool:
        return True

    @override
    def readinto(self, buffer: Buffer, /) -> int | None:
        view = memoryview(buffer).cast("B")
        data = self._reader.read(len(view))
        view[: len(data)] = data
        return len(data)


@final
class TextReader:
    """Line-oriented text reader with lazy UTF-8 decoding.

    Decodes bytes incrementally as lines are requested. Obtain via
    :meth:`Filesystem.open_text`.
    """

    def __init__(self, reader: ByteReader, encoding: str) -> None:
        self._reader = reader
        self._encoding = encoding
        self._text: IO[str] = io.TextIOWrapper(
            io.BufferedReader(_ByteReaderRawIO(reader)), encoding=encoding
        )
        self._line_number = 0
        self._closed = False

    @property
    def path(self) -> str:
        """Path being read (relative to filesystem root)."""
        return self._reader.path

    @property
    def encoding(self) -> str:
        """Text encoding (always ``utf-8``)."""
        return self._encoding

    @property
    def line_number(self) -> int:
        """Number of lines read so far."""
        return self._line_number

    @property
    def closed(self) -> bool:
        """True if the reader has been closed."""
        return self._closed

    def readline(self) -> str:
        """Read the next line including its newline; ``""`` at EOF."""
        _check_open(self._closed)
        line = self._text.readline()
        if line:
            self._line_number += 1
        return line

    def read(self, size: int = -1) -> str:
        """Read up to ``size`` characters (``-1`` reads to EOF)."""
        _check_open(self._closed)
        return self._text.read(size)

    def __iter__(self) -> Iterator[str]:
        """Iterate over lines (including newlines)."""
        _check_open(self._closed)
        for line in self._text:
            self._line_number += 1
            yield line

    def lines(self, *, strip: bool = False) -> Iterator[str]:
        """Iterate over lines, optionally stripping trailing whitespace."""
        for line in self:
            yield line.rstrip() if strip else line

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the reader. Idempotent."""
        if not self._closed:
            self._closed = True
            try:
                self._text.close()
            finally:
                self._reader.close()


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "ByteReader",
    "ByteWriter",
    "TextReader",
]

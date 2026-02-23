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

"""Text streaming implementation with lazy UTF-8 decoding.

Provides DefaultTextReader wrapping any ByteReader, and the internal
_ByteReaderWrapper adapter for io.TextIOWrapper compatibility.
"""

from __future__ import annotations

import io
from collections.abc import Buffer, Iterator
from dataclasses import dataclass, field
from typing import IO, Self, override

from ._stream_protocols import ByteReader

__all__ = [
    "DefaultTextReader",
]


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

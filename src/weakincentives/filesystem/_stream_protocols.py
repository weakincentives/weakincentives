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

"""Streaming protocol definitions for filesystem I/O.

Defines the ByteReader, ByteWriter, and TextReader protocols that all
streaming implementations must satisfy.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Final, Protocol, Self, runtime_checkable

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "ByteReader",
    "ByteWriter",
    "TextReader",
]

#: Default chunk size for iteration (64KB).
DEFAULT_CHUNK_SIZE: Final[int] = 65_536


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

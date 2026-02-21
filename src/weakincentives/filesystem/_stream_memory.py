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

"""In-memory streaming implementations.

Provides MemoryByteReader and MemoryByteWriter backed by io.BytesIO,
used by InMemoryFilesystem for streaming reads and writes.
"""

from __future__ import annotations

import io
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Literal, Self

from ._stream_protocols import DEFAULT_CHUNK_SIZE

__all__ = [
    "MemoryByteReader",
    "MemoryByteWriter",
]


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

        Raises:
            ValueError: If the writer has been closed.
        """
        self._check_closed()
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

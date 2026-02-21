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

"""Host filesystem streaming implementations.

Provides ByteReader and ByteWriter implementations backed by native file
handles for streaming I/O on host filesystems.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Literal, Self

from ._streams import DEFAULT_CHUNK_SIZE

__all__ = [
    "HostByteReader",
    "HostByteWriter",
]


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

        if create_parents:
            parent_dir.mkdir(parents=True, exist_ok=True)
        elif not parent_dir.exists():
            raise FileNotFoundError(f"Parent directory does not exist: {parent_dir}")

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

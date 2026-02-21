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

# pyright: reportImportCycles=false

"""Streaming byte writer for the in-memory filesystem backend."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from weakincentives.filesystem import MemoryByteWriter

if TYPE_CHECKING:
    from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem


@dataclass(slots=True)
class InMemoryByteWriter:
    """ByteWriter that commits to InMemoryFilesystem on close.

    This writer collects bytes in a MemoryByteWriter and commits
    the content to the filesystem when closed.
    """

    filesystem: InMemoryFilesystem
    path: str
    mode: Literal["create", "overwrite", "append"]
    existing_content: bytes | None = None
    _writer: MemoryByteWriter = field(init=False)
    _closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._writer = MemoryByteWriter.create(
            self.path,
            mode=self.mode,
            existing_content=self.existing_content,
        )

    @property
    def bytes_written(self) -> int:
        """Total bytes written (excluding initial append content)."""
        return self._writer.bytes_written

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
        return self._writer.write(data)

    def write_all(self, chunks: Iterable[bytes]) -> int:
        """Write all chunks from an iterable."""
        self._check_closed()
        return self._writer.write_all(chunks)

    def __enter__(self) -> InMemoryByteWriter:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager.

        On error, discards buffered writes without committing (matching
        the HostByteWriter abort-on-error contract). On success, commits
        the content to the filesystem.
        """
        if exc_type is not None:
            self._abort()
        else:
            self.close()

    def _abort(self) -> None:
        """Discard buffered writes without committing to filesystem.

        No filesystem cleanup is needed because ``commit_streaming_write``
        is only called in ``close()``; abort skips the commit so the
        filesystem is never modified.
        """
        if self._closed:
            return
        self._closed = True
        self._writer.close()

    def close(self) -> None:
        """Close the writer and commit content to filesystem."""
        if self._closed:
            return
        self._closed = True

        try:
            # Commit the written content to the filesystem
            content = self._writer.get_content()
            self.filesystem.commit_streaming_write(self.path, content, self.mode)
        finally:
            self._writer.close()

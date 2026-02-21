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

This module re-exports all streaming types from their focused sub-modules:

- ``_stream_protocols``: ByteReader, ByteWriter, TextReader protocols
- ``_stream_host``: HostByteReader, HostByteWriter (native file handles)
- ``_stream_memory``: MemoryByteReader, MemoryByteWriter (in-memory buffers)
- ``_stream_text``: DefaultTextReader with lazy UTF-8 decoding
"""

from __future__ import annotations

from ._stream_host import HostByteReader, HostByteWriter
from ._stream_memory import MemoryByteReader, MemoryByteWriter
from ._stream_protocols import DEFAULT_CHUNK_SIZE, ByteReader, ByteWriter, TextReader
from ._stream_text import DefaultTextReader

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

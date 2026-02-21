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

"""Generic validation suite for Filesystem protocol implementations.

This module provides a reusable test suite that validates any implementation
of the Filesystem protocol. Tests are designed to be subclassed with a
concrete filesystem factory.

Example usage::

    from tests.helpers.filesystem import FilesystemValidationSuite

    class TestMyFilesystem(FilesystemValidationSuite):
        @pytest.fixture
        def fs(self) -> MyFilesystem:
            return MyFilesystem()

All tests in the suite will run against the filesystem returned by the
``fs`` fixture. Implementations must provide this fixture.

The implementation is split across two modules for maintainability:

- ``filesystem_core``: basic operations (read, write, stat, list, etc.)
- ``filesystem_streaming``: streaming operations (open_read, open_write, etc.)
  plus read-only and snapshotable validation suites.
"""

from __future__ import annotations

from tests.helpers.filesystem_core import FilesystemCoreValidationSuite
from tests.helpers.filesystem_streaming import (
    FilesystemStreamingValidationSuite,
    ReadOnlyFilesystemValidationSuite,
    SnapshotableFilesystemValidationSuite,
)


class FilesystemValidationSuite(
    FilesystemCoreValidationSuite,
    FilesystemStreamingValidationSuite,
):
    """Abstract test suite for Filesystem protocol compliance.

    Subclasses must implement the ``fs`` fixture to provide a filesystem
    instance to test. The filesystem should be empty at the start of each test.

    This suite validates:
    - Read operations (read, read_bytes, exists, stat, list, glob, grep)
    - Write operations (write, write_bytes, delete, mkdir)
    - Properties (root, read_only)
    - Error handling (FileNotFoundError, IsADirectoryError, etc.)
    - Path validation (depth, segment length)
    - Streaming operations (open_read, open_write, open_text)

    Implementation-specific tests (e.g., HostFilesystem path escape detection)
    should remain in their own test modules.
    """


__all__ = [
    "FilesystemValidationSuite",
    "ReadOnlyFilesystemValidationSuite",
    "SnapshotableFilesystemValidationSuite",
]

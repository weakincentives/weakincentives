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

"""Test-only helper utilities for weakincentives.

This package provides centralized test infrastructure to avoid scattered
ad-hoc mocks across test files. Import helpers from here or from their
specific submodules.

Core helpers available here:
    - NullDispatcher: No-op event dispatcher for isolated testing
    - FilesystemValidationSuite: Validation suite for Filesystem implementations
    - ReadOnlyFilesystemValidationSuite: Validation for read-only filesystems
    - SnapshotableFilesystemValidationSuite: Validation for snapshotable filesystems

Additional helpers in submodules:
    - tests.helpers.time: FakeClock and fake_clock fixture for time control
    - tests.helpers.adapters: Adapter name constants
    - tests.helpers.session: Event factory functions

Domain-specific helpers:
    - tests.cli.helpers: FakeLogger, FakeContextManager for CLI testing
    - tests.adapters.claude_agent_sdk.error_mocks: SDK error mocks
    - tests.tools.helpers: Tool testing utilities
    - tests.tools.podman_test_helpers: Podman mocks

Pytest fixtures (auto-registered via conftest.py):
    - fake_clock: Fresh FakeClock instance (from tests.helpers.time)
    - session_factory: Session/dispatcher pair factory

Example::

    from tests.helpers import NullDispatcher
    from tests.helpers.time import FakeClock

    def test_with_controlled_time(fake_clock: FakeClock) -> None:
        # Use the fixture for simple cases
        fake_clock.advance(10)

    def test_manual_clock() -> None:
        # Or import FakeClock for direct use
        clock = FakeClock()
        clock.set_wall(datetime(2024, 1, 1, tzinfo=UTC))
"""

from .events import NullDispatcher
from .filesystem import (
    FilesystemValidationSuite,
    ReadOnlyFilesystemValidationSuite,
    SnapshotableFilesystemValidationSuite,
)

__all__ = [
    "FilesystemValidationSuite",
    "NullDispatcher",
    "ReadOnlyFilesystemValidationSuite",
    "SnapshotableFilesystemValidationSuite",
]

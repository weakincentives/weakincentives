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

"""Test-only helper utilities for weakincentives."""

from .events import NullDispatcher
from .filesystem import (
    FilesystemValidationSuite,
    ReadOnlyFilesystemValidationSuite,
    SnapshotableFilesystemValidationSuite,
)
from .threading import (
    FakeBackgroundWorker,
    FakeCheckpoint,
    FakeExecutor,
    FakeGate,
    FakeLatch,
    FakeScheduler,
    fake_checkpoint,
    fake_executor,
    fake_gate,
    fake_latch_factory,
    fake_scheduler,
)

__all__ = [
    # Filesystem
    "FilesystemValidationSuite",
    "ReadOnlyFilesystemValidationSuite",
    "SnapshotableFilesystemValidationSuite",
    # Events
    "NullDispatcher",
    # Threading
    "FakeBackgroundWorker",
    "FakeCheckpoint",
    "FakeExecutor",
    "FakeGate",
    "FakeLatch",
    "FakeScheduler",
    "fake_checkpoint",
    "fake_executor",
    "fake_gate",
    "fake_latch_factory",
    "fake_scheduler",
]

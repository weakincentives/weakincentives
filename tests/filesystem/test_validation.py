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

"""Validation suites run against the facade over every backend.

The same ``FilesystemValidationSuite`` (plus streaming, read-only, and
snapshot suites) runs over ``HostBackend``, ``MemoryBackend``, and the toy
``FakeBackend`` — proving the narrow waist: ergonomics live in the facade,
backends only provide primitives.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from tests.helpers.filesystem import FilesystemValidationSuite
from tests.helpers.filesystem_streaming import (
    FilesystemStreamingValidationSuite,
    ReadOnlyFilesystemValidationSuite,
    SnapshotableFilesystemValidationSuite,
)
from weakincentives.filesystem import Filesystem, FilesystemBackend

from .fake_backend import FakeBackend

# =============================================================================
# MemoryBackend
# =============================================================================


class TestMemoryFilesystem(
    FilesystemValidationSuite, FilesystemStreamingValidationSuite
):
    """Full validation + streaming suites over MemoryBackend."""

    @pytest.fixture
    def fs(self) -> Filesystem:
        return Filesystem.in_memory()


class TestMemoryFilesystemReadOnly(ReadOnlyFilesystemValidationSuite):
    """Read-only behavior over MemoryBackend."""

    @pytest.fixture
    def fs_readonly(self) -> Filesystem:
        return Filesystem.in_memory(read_only=True)


class TestMemoryFilesystemSnapshots(SnapshotableFilesystemValidationSuite):
    """Snapshot behavior over MemoryBackend."""

    @pytest.fixture
    def fs_snapshotable(self) -> Filesystem:
        return Filesystem.in_memory()


# =============================================================================
# HostBackend
# =============================================================================


class TestHostFilesystem(FilesystemValidationSuite, FilesystemStreamingValidationSuite):
    """Full validation + streaming suites over HostBackend."""

    @pytest.fixture
    def fs(self, tmp_path: Path) -> Filesystem:
        return Filesystem.host(tmp_path)


class TestHostFilesystemReadOnly(ReadOnlyFilesystemValidationSuite):
    """Read-only behavior over HostBackend."""

    @pytest.fixture
    def fs_readonly(self, tmp_path: Path) -> Filesystem:
        return Filesystem.host(tmp_path, read_only=True)


class TestHostFilesystemSnapshots(SnapshotableFilesystemValidationSuite):
    """Snapshot behavior over HostBackend."""

    @pytest.fixture
    def fs_snapshotable(self, tmp_path: Path) -> Filesystem:
        return Filesystem.host(tmp_path)


# =============================================================================
# FakeBackend (toy fixture)
# =============================================================================


class TestFakeFilesystem(FilesystemValidationSuite, FilesystemStreamingValidationSuite):
    """Full validation + streaming suites over the toy FakeBackend."""

    @pytest.fixture
    def fs(self) -> Filesystem:
        return Filesystem(FakeBackend())


class TestFakeFilesystemReadOnly(ReadOnlyFilesystemValidationSuite):
    """Read-only behavior over the toy FakeBackend."""

    @pytest.fixture
    def fs_readonly(self) -> Filesystem:
        return Filesystem(FakeBackend(read_only=True))


class TestFakeFilesystemSnapshots(SnapshotableFilesystemValidationSuite):
    """Snapshot behavior over the toy FakeBackend."""

    @pytest.fixture
    def fs_snapshotable(self) -> Filesystem:
        return Filesystem(FakeBackend())


def test_fake_backend_satisfies_protocol() -> None:
    """FakeBackend structurally satisfies FilesystemBackend."""
    assert isinstance(FakeBackend(), FilesystemBackend)


def test_fake_backend_is_under_100_lines() -> None:
    """The narrow waist keeps a complete fake under 100 lines."""
    source = inspect.getsource(FakeBackend)
    assert len(source.splitlines()) < 100

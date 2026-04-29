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

"""Tests for ``weakincentives.workspace``."""

from __future__ import annotations

import pytest

from weakincentives.core import Snapshotable
from weakincentives.filesystem import InMemoryFilesystem
from weakincentives.workspace import Workspace, WorkspaceDigest


def _seeded_workspace(*, read_only: bool = False) -> Workspace:
    fs = InMemoryFilesystem()
    fs.write_text("/a.txt", "alpha")
    fs.write_text("/dir/b.txt", "beta")
    return Workspace(fs, read_only=read_only)


def test_workspace_satisfies_snapshotable_protocol() -> None:
    assert isinstance(_seeded_workspace(), Snapshotable)


def test_workspace_read_methods_proxy_to_filesystem() -> None:
    ws = _seeded_workspace()
    assert ws.read_text("/a.txt") == "alpha"
    assert ws.exists("/dir/b.txt")
    assert ws.list_dir("/dir") == ("b.txt",)


def test_workspace_write_and_remove() -> None:
    ws = _seeded_workspace()
    ws.write_text("/c.txt", "gamma")
    assert ws.read_text("/c.txt") == "gamma"
    ws.remove("/c.txt")
    assert not ws.exists("/c.txt")


def test_read_only_blocks_mutations() -> None:
    ws = _seeded_workspace(read_only=True)
    with pytest.raises(PermissionError, match="read-only"):
        ws.write_text("/a.txt", "x")
    with pytest.raises(PermissionError, match="read-only"):
        ws.remove("/a.txt")


def test_digest_stable_for_same_contents() -> None:
    a = _seeded_workspace().digest()
    b = _seeded_workspace().digest()
    assert a == b
    assert a.file_count == 2
    assert a.total_bytes == len(b"alpha") + len(b"beta")


def test_digest_changes_when_content_changes() -> None:
    base = _seeded_workspace()
    before = base.digest()
    base.write_text("/a.txt", "alpha-modified")
    after = base.digest()
    assert before != after


def test_snapshot_round_trip_via_workspace() -> None:
    ws = _seeded_workspace()
    snapshot = ws.snapshot()
    ws.write_text("/a.txt", "MUTATED")
    ws.write_text("/d.txt", "new")
    ws.restore(snapshot)
    assert ws.read_text("/a.txt") == "alpha"
    assert not ws.exists("/d.txt")


def test_snapshot_requires_filesystem_to_be_snapshotable() -> None:
    class NotSnapshotable:
        def read_text(self, path: str) -> str:
            del path
            return ""

        def write_text(self, path: str, content: str) -> None:
            del path, content

        def exists(self, path: str) -> bool:
            del path
            return False

        def list_dir(self, path: str) -> tuple[str, ...]:
            del path
            return ()

        def remove(self, path: str) -> None:
            del path

    ws = Workspace(NotSnapshotable())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="cannot snapshot"):
        ws.snapshot()


def test_restore_requires_filesystem_restore() -> None:
    class HalfSnapshotable:
        def read_text(self, path: str) -> str:
            del path
            return ""

        def write_text(self, path: str, content: str) -> None:
            del path, content

        def exists(self, path: str) -> bool:
            del path
            return False

        def list_dir(self, path: str) -> tuple[str, ...]:
            del path
            return ()

        def remove(self, path: str) -> None:
            del path

        def snapshot(self) -> object:
            from weakincentives.filesystem import FilesystemSnapshot

            return FilesystemSnapshot(files=())

    ws = Workspace(HalfSnapshotable())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Snapshotable"):
        from weakincentives.filesystem import FilesystemSnapshot

        ws.restore(FilesystemSnapshot(files=()))


def test_workspace_digest_returns_workspacedigest() -> None:
    digest = _seeded_workspace().digest()
    assert isinstance(digest, WorkspaceDigest)
    assert len(digest.sha256) == 64


def test_workspace_name_and_filesystem_attrs() -> None:
    fs = InMemoryFilesystem()
    ws = Workspace(fs, name="demo")
    assert ws.name == "demo"
    assert ws.filesystem is fs
    assert ws.read_only is False

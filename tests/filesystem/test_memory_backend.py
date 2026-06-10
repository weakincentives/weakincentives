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

"""MemoryBackend-specific tests."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest

from weakincentives.errors import SnapshotRestoreError
from weakincentives.filesystem import Filesystem, SnapshotRef


class TestMemoryBackendSpecific:
    """Behaviors specific to the in-memory backend."""

    def test_root_property_is_slash(self) -> None:
        assert Filesystem.in_memory().root == "/"

    def test_snapshot_token_prefix(self) -> None:
        fs = Filesystem.in_memory()
        fs.write("file.txt", "content")
        ref = fs.snapshot()
        assert ref.token.startswith("mem-")

    def test_restore_unknown_snapshot_raises(self) -> None:
        fs = Filesystem.in_memory()
        fake_ref = SnapshotRef(
            snapshot_id=UUID("00000000-0000-0000-0000-000000000000"),
            created_at=datetime.now(UTC),
            token="mem-nonexistent",
        )
        with pytest.raises(SnapshotRestoreError, match="Unknown snapshot"):
            fs.restore(fake_ref)

    def test_list_with_implicit_subdirectories(self) -> None:
        """Listing should show directories implicitly created by file writes."""
        fs = Filesystem.in_memory()
        fs.write("dir1/subdir/file1.txt", "content1")
        fs.write("dir1/subdir/file2.txt", "content2")
        fs.write("dir1/other/file3.txt", "content3")

        entries = fs.list("dir1")
        dir_names = [e.name for e in entries if e.is_directory]
        assert "subdir" in dir_names
        assert "other" in dir_names

    def test_list_subdirectory_excludes_outside_files(self) -> None:
        fs = Filesystem.in_memory()
        fs.write("dir1/file.txt", "inside")
        fs.write("outside.txt", "outside")
        entries = fs.list("dir1")
        assert [e.name for e in entries] == ["file.txt"]

    def test_delete_root_is_rejected(self) -> None:
        """Root deletion is uniformly rejected by the facade."""
        fs = Filesystem.in_memory()
        fs.write("dir/file.txt", "content")
        with pytest.raises(PermissionError, match="Cannot delete root"):
            fs.delete("/", recursive=True)

    def test_delete_directory_with_only_subdirectories_requires_recursive(
        self,
    ) -> None:
        fs = Filesystem.in_memory()
        fs.mkdir("a/b", parents=True)
        with pytest.raises(IsADirectoryError, match="not empty"):
            fs.delete("a")
        fs.delete("a", recursive=True)
        assert not fs.exists("a")

    def test_glob_with_recursive_pattern(self) -> None:
        fs = Filesystem.in_memory()
        fs.write("dir/file.py", "code")
        fs.write("subdir/nested/deep.py", "more code")
        matches = fs.glob("**/*.py")
        assert len(matches) >= 1

    def test_glob_in_root_includes_nested_files(self) -> None:
        fs = Filesystem.in_memory()
        fs.mkdir("mydir")
        fs.write("mydir/file.txt", "content")
        fs.write("root.txt", "root content")
        matches = fs.glob("*")
        assert len(matches) == 2
        assert all(m.is_file for m in matches)

    def test_backend_write_to_directory_raises_defensively(self) -> None:
        """The backend itself rejects directory targets, not only the facade."""
        from weakincentives.filesystem import MemoryBackend

        backend = MemoryBackend()
        backend.mkdir("mydir")
        with pytest.raises(IsADirectoryError):
            backend.write("mydir", b"x", mode="overwrite")

    def test_backend_mkdir_through_file_raises_defensively(self) -> None:
        """The backend rejects directory chains crossing a file."""
        from weakincentives.filesystem import MemoryBackend

        backend = MemoryBackend()
        backend.write("blocker", b"x", mode="overwrite")
        with pytest.raises(NotADirectoryError):
            backend.mkdir("blocker/sub")
        # The file is intact and no shadow directory entry was recorded
        assert backend.stat("blocker").is_file
        with pytest.raises(NotADirectoryError):
            backend.write("blocker/child.txt", b"y", mode="overwrite")

    def test_streaming_write_commit_rejects_path_turned_directory(self) -> None:
        """A directory created between open_write and close fails the commit."""
        fs = Filesystem.in_memory()
        writer = fs.open_write("out.bin")
        _ = writer.write(b"data")
        fs.mkdir("out.bin")
        with pytest.raises(IsADirectoryError):
            writer.close()
        assert fs.stat("out.bin").is_directory

    def test_overwrite_preserves_created_at(self) -> None:
        fs = Filesystem.in_memory()
        fs.write("file.txt", "v1")
        created = fs.stat("file.txt").created_at
        fs.write("file.txt", "v2")
        assert fs.stat("file.txt").created_at == created

    def test_snapshots_are_structurally_shared(self) -> None:
        """Restoring one snapshot does not corrupt another."""
        fs = Filesystem.in_memory()
        fs.write("file.txt", "v1")
        ref1 = fs.snapshot()
        fs.write("file.txt", "v2")
        ref2 = fs.snapshot()

        fs.restore(ref1)
        assert fs.read("file.txt").content == "v1"
        fs.restore(ref2)
        assert fs.read("file.txt").content == "v2"

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

"""Tests for filesystem archiving functionality and edge cases."""

from __future__ import annotations

from pathlib import Path

import pytest

from weakincentives.debug import BundleWriter, DebugBundle
from weakincentives.debug.bundle import BundleConfig


class TestFilesystemArchiving:
    """Tests for filesystem archiving functionality."""

    def test_writer_archives_filesystem(self, tmp_path: Path) -> None:
        """Test filesystem archiving with InMemoryFilesystem."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem

        fs = InMemoryFilesystem()
        _ = fs.write("/test.txt", "Hello, World!")
        _ = fs.write("/subdir/nested.txt", "Nested content")

        with BundleWriter(tmp_path) as writer:
            writer.write_filesystem(fs)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        files = bundle.list_files()
        assert any("filesystem" in f for f in files)

    def test_writer_handles_filesystem_errors(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test filesystem archiving handles errors gracefully."""

        class FailingFilesystem:
            """Filesystem that always raises."""

            def list(self, _path: str) -> list:
                raise OSError("Simulated error")

        config = BundleConfig(target=tmp_path)

        with BundleWriter(tmp_path, config=config) as writer:
            writer.write_filesystem(FailingFilesystem())  # type: ignore[arg-type]

        # Should not fail, just log the error
        assert writer.path is not None
        assert "Failed to write filesystem" in caplog.text

    def test_writer_respects_max_file_size(self, tmp_path: Path) -> None:
        """Test files larger than max_file_size are skipped."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem

        fs = InMemoryFilesystem()
        # Write a small file
        _ = fs.write("/small.txt", "Small content")
        # Write a large file
        _ = fs.write("/large.txt", "x" * 1000)

        config = BundleConfig(target=tmp_path, max_file_size=100)

        with BundleWriter(tmp_path, config=config) as writer:
            writer.write_filesystem(fs)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        files = bundle.list_files()
        # small.txt should be present, large.txt should not
        assert any("small.txt" in f for f in files)
        assert not any("large.txt" in f for f in files)

    def test_writer_respects_max_total_size(self, tmp_path: Path) -> None:
        """Test filesystem capture stops at max_total_size."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem

        fs = InMemoryFilesystem()
        # Write multiple files
        for i in range(10):
            _ = fs.write(f"/file{i}.txt", "x" * 100)

        # Set max_total_size to less than total file content
        config = BundleConfig(target=tmp_path, max_total_size=250)

        with BundleWriter(tmp_path, config=config) as writer:
            writer.write_filesystem(fs)

        assert writer.path is not None
        # The filesystem_truncated flag should be set
        bundle = DebugBundle.load(writer.path)
        manifest = bundle.manifest
        assert manifest.capture.limits_applied.get("filesystem_truncated") is True


class TestFilesystemEdgeCases:
    """Tests for filesystem archiving edge cases."""

    def test_filesystem_handles_permission_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test filesystem archiving handles PermissionError gracefully."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem

        class PermissionErrorFilesystem(InMemoryFilesystem):
            """Filesystem that raises PermissionError on read."""

            def read_bytes(self, path: str) -> object:
                raise PermissionError("No permission")

        fs = PermissionErrorFilesystem()
        _ = fs.write("/test.txt", "content")

        config = BundleConfig(target=tmp_path)

        with BundleWriter(tmp_path, config=config) as writer:
            writer.write_filesystem(fs)

        assert writer.path is not None
        # Should complete without error, file just skipped
        bundle = DebugBundle.load(writer.path)
        files = bundle.list_files()
        assert not any("test.txt" in f for f in files)

    def test_filesystem_handles_file_not_found(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test filesystem archiving handles FileNotFoundError gracefully."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem

        class DisappearingFilesystem(InMemoryFilesystem):
            """Filesystem where files disappear between list and read."""

            def stat(self, path: str) -> object:
                raise FileNotFoundError("File gone")

        fs = DisappearingFilesystem()
        _ = fs.write("/test.txt", "content")

        config = BundleConfig(target=tmp_path)

        with BundleWriter(tmp_path, config=config) as writer:
            writer.write_filesystem(fs)

        assert writer.path is not None

    def test_filesystem_handles_is_a_directory_error(self, tmp_path: Path) -> None:
        """Test filesystem archiving handles IsADirectoryError gracefully."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem

        class DirectoryAsFileFilesystem(InMemoryFilesystem):
            """Filesystem where read fails with IsADirectoryError."""

            def read_bytes(self, path: str) -> object:
                raise IsADirectoryError("Is a directory")

        fs = DirectoryAsFileFilesystem()
        _ = fs.write("/test.txt", "content")

        config = BundleConfig(target=tmp_path)

        with BundleWriter(tmp_path, config=config) as writer:
            writer.write_filesystem(fs)

        assert writer.path is not None

    def test_collect_files_handles_file_not_found_on_list(self, tmp_path: Path) -> None:
        """Test _collect_files handles FileNotFoundError gracefully."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem

        class ListFailsFilesystem(InMemoryFilesystem):
            """Filesystem where list fails with FileNotFoundError."""

            def list(self, path: str) -> list:
                raise FileNotFoundError("Directory gone")

        fs = ListFailsFilesystem()

        config = BundleConfig(target=tmp_path)

        with BundleWriter(tmp_path, config=config) as writer:
            writer.write_filesystem(fs)

        assert writer.path is not None

    def test_collect_files_handles_not_a_directory_error(self, tmp_path: Path) -> None:
        """Test _collect_files handles NotADirectoryError gracefully."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem

        class NotADirFilesystem(InMemoryFilesystem):
            """Filesystem where list fails with NotADirectoryError."""

            def list(self, path: str) -> list:
                raise NotADirectoryError("Not a directory")

        fs = NotADirFilesystem()

        config = BundleConfig(target=tmp_path)

        with BundleWriter(tmp_path, config=config) as writer:
            writer.write_filesystem(fs)

        assert writer.path is not None

    def test_collect_files_recurses_directories(self, tmp_path: Path) -> None:
        """Test _collect_files recursively collects from subdirectories."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem

        fs = InMemoryFilesystem()
        _ = fs.write("/level1/level2/deep.txt", "deep content")

        config = BundleConfig(target=tmp_path)

        with BundleWriter(tmp_path, config=config) as writer:
            writer.write_filesystem(fs)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        files = bundle.list_files()
        assert any("deep.txt" in f for f in files)

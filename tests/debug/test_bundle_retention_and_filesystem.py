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

"""Tests for debug bundle retention and filesystem capture."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from tests.debug._bundle_fixtures import create_bundle, create_bundle_path
from weakincentives.debug.bundle import (
    BundleConfig,
    BundleManifest,
    BundleRetentionPolicy,
    BundleWriter,
    DebugBundle,
)


class TestFilesystemArchiving:
    """Tests for filesystem archiving functionality."""

    def test_writer_archives_filesystem(self, tmp_path: Path) -> None:
        """Test filesystem archiving with InMemoryFilesystem."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem

        fs = InMemoryFilesystem()
        _ = fs.write("/test.txt", "Hello, World!")
        _ = fs.write("/subdir/nested.txt", "Nested content")

        bundle = create_bundle(
            tmp_path, write_fn=lambda writer: writer.write_filesystem(fs)
        )
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

        _ = create_bundle(
            tmp_path,
            config=config,
            write_fn=lambda writer: writer.write_filesystem(
                FailingFilesystem()  # type: ignore[arg-type]
            ),
        )

        # Should not fail, just log the error
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

        bundle = create_bundle(
            tmp_path, config=config, write_fn=lambda writer: writer.write_filesystem(fs)
        )
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

        bundle = create_bundle(
            tmp_path, config=config, write_fn=lambda writer: writer.write_filesystem(fs)
        )
        # The filesystem_truncated flag should be set
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

        bundle = create_bundle(
            tmp_path, config=config, write_fn=lambda writer: writer.write_filesystem(fs)
        )
        # Should complete without error, file just skipped
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

        _ = create_bundle(
            tmp_path, config=config, write_fn=lambda writer: writer.write_filesystem(fs)
        )

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

        _ = create_bundle(
            tmp_path, config=config, write_fn=lambda writer: writer.write_filesystem(fs)
        )

    def test_collect_files_handles_file_not_found_on_list(self, tmp_path: Path) -> None:
        """Test _collect_files handles FileNotFoundError gracefully."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem

        class ListFailsFilesystem(InMemoryFilesystem):
            """Filesystem where list fails with FileNotFoundError."""

            def list(self, path: str) -> list:
                raise FileNotFoundError("Directory gone")

        fs = ListFailsFilesystem()

        config = BundleConfig(target=tmp_path)

        _ = create_bundle(
            tmp_path, config=config, write_fn=lambda writer: writer.write_filesystem(fs)
        )

    def test_collect_files_handles_not_a_directory_error(self, tmp_path: Path) -> None:
        """Test _collect_files handles NotADirectoryError gracefully."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem

        class NotADirFilesystem(InMemoryFilesystem):
            """Filesystem where list fails with NotADirectoryError."""

            def list(self, path: str) -> list:
                raise NotADirectoryError("Not a directory")

        fs = NotADirFilesystem()

        config = BundleConfig(target=tmp_path)

        _ = create_bundle(
            tmp_path, config=config, write_fn=lambda writer: writer.write_filesystem(fs)
        )

    def test_collect_files_recurses_directories(self, tmp_path: Path) -> None:
        """Test _collect_files recursively collects from subdirectories."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem

        fs = InMemoryFilesystem()
        _ = fs.write("/level1/level2/deep.txt", "deep content")

        config = BundleConfig(target=tmp_path)

        bundle = create_bundle(
            tmp_path, config=config, write_fn=lambda writer: writer.write_filesystem(fs)
        )
        files = bundle.list_files()
        assert any("deep.txt" in f for f in files)


class TestBundleRetentionPolicy:
    """Tests for BundleRetentionPolicy dataclass."""

    def test_default_policy(self) -> None:
        """Test default retention policy has no limits."""
        policy = BundleRetentionPolicy()
        assert policy.max_bundles is None
        assert policy.max_age_seconds is None
        assert policy.max_total_bytes is None

    def test_policy_with_max_bundles(self) -> None:
        """Test policy with max_bundles limit."""
        policy = BundleRetentionPolicy(max_bundles=5)
        assert policy.max_bundles == 5

    def test_policy_with_all_limits(self) -> None:
        """Test policy with all limits configured."""
        policy = BundleRetentionPolicy(
            max_bundles=10,
            max_age_seconds=86400,
            max_total_bytes=100_000_000,
        )
        assert policy.max_bundles == 10
        assert policy.max_age_seconds == 86400
        assert policy.max_total_bytes == 100_000_000


class TestRetentionPolicyIntegration:
    """Tests for retention policy integration with BundleWriter."""

    def test_retention_max_bundles_deletes_oldest(self, tmp_path: Path) -> None:
        """Test max_bundles limit deletes oldest bundles."""
        retention = BundleRetentionPolicy(max_bundles=2)
        config = BundleConfig(target=tmp_path, retention=retention)

        # Create 3 bundles
        paths: list[Path] = []
        for i in range(3):
            path = create_bundle_path(
                tmp_path,
                config=config,
                write_fn=lambda writer, idx=i: writer.write_request_input(
                    {"bundle": idx}
                ),
            )
            paths.append(path)

        # After creating the 3rd bundle, only 2 should remain
        remaining = list(tmp_path.glob("*.zip"))
        assert len(remaining) == 2

        # The newest bundle should definitely exist
        assert paths[-1].exists()

    def test_retention_max_age_deletes_old_bundles(self, tmp_path: Path) -> None:
        """Test max_age_seconds limit deletes old bundles."""
        from unittest.mock import patch

        # Create a bundle with old timestamp
        old_path = create_bundle_path(
            tmp_path, write_fn=lambda writer: writer.write_request_input({"old": True})
        )

        # Patch datetime.now to return a time far in the future for retention check
        original_now = datetime.now

        def future_now(tz: object = None) -> datetime:
            if tz is not None:
                return original_now(tz) + __import__("datetime").timedelta(days=2)
            return original_now() + __import__("datetime").timedelta(days=2)

        # Create a new bundle with retention policy
        retention = BundleRetentionPolicy(max_age_seconds=86400)  # 24 hours
        config = BundleConfig(target=tmp_path, retention=retention)

        with patch("weakincentives.debug.bundle.datetime") as mock_datetime:
            mock_datetime.now = future_now
            mock_datetime.fromisoformat = datetime.fromisoformat
            # Need to patch at the class level for the retention check
            new_path = create_bundle_path(
                tmp_path,
                config=config,
                write_fn=lambda writer: writer.write_request_input({"new": True}),
            )

        # The old bundle should be deleted
        assert not old_path.exists()
        assert new_path.exists()

    def test_retention_max_total_bytes_deletes_largest_oldest(
        self, tmp_path: Path
    ) -> None:
        """Test max_total_bytes limit deletes bundles to stay under limit."""
        first_path = create_bundle_path(
            tmp_path,
            write_fn=lambda writer: writer.write_request_input({"data": "x" * 1000}),
        )
        first_size = first_path.stat().st_size

        # Create second bundle with retention that allows only ~one bundle worth
        retention = BundleRetentionPolicy(max_total_bytes=first_size + 100)
        config = BundleConfig(target=tmp_path, retention=retention)

        second_path = create_bundle_path(
            tmp_path,
            config=config,
            write_fn=lambda writer: writer.write_request_input({"data": "y" * 1000}),
        )

        # The first bundle should be deleted to stay under limit
        assert not first_path.exists()
        assert second_path.exists()

    def test_retention_handles_invalid_bundles_gracefully(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test retention skips invalid bundle files."""
        # Create an invalid "bundle" file
        invalid_file = tmp_path / "invalid_bundle.zip"
        _ = invalid_file.write_text("not a valid zip")

        retention = BundleRetentionPolicy(max_bundles=10)
        config = BundleConfig(target=tmp_path, retention=retention)

        # This should not raise
        path = create_bundle_path(
            tmp_path,
            config=config,
            write_fn=lambda writer: writer.write_request_input({"test": True}),
        )

        assert path.exists()
        # Invalid file should still be there (not processed)
        assert invalid_file.exists()

    def test_retention_error_is_logged_not_raised(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test retention errors are logged but don't fail bundle creation."""
        from unittest.mock import patch

        retention = BundleRetentionPolicy(max_bundles=1)
        config = BundleConfig(target=tmp_path, retention=retention)

        # Mock _enforce_retention to raise
        def failing_enforce(self: object, policy: object, exclude_path: object) -> None:
            raise RuntimeError("Simulated retention failure")

        with patch.object(BundleWriter, "_enforce_retention", failing_enforce):
            path = create_bundle_path(
                tmp_path,
                config=config,
                write_fn=lambda writer: writer.write_request_input({"test": True}),
            )

        # Bundle should still be created
        assert path.exists()
        assert "Failed to apply retention policy" in caplog.text

    def test_retention_none_does_nothing(self, tmp_path: Path) -> None:
        """Test that no retention policy means no cleanup."""
        config = BundleConfig(target=tmp_path, retention=None)

        # Create multiple bundles
        paths: list[Path] = []
        for i in range(5):
            path = create_bundle_path(
                tmp_path,
                config=config,
                write_fn=lambda writer, idx=i: writer.write_request_input(
                    {"bundle": idx}
                ),
            )
            paths.append(path)

        # All bundles should still exist
        for path in paths:
            assert path.exists()

    def test_retention_size_limit_skips_already_marked_bundles(
        self, tmp_path: Path
    ) -> None:
        """Test size limit skips bundles already marked for deletion by other limits."""
        # Create multiple bundles without retention first
        for i in range(3):
            _ = create_bundle_path(
                tmp_path,
                write_fn=lambda writer, idx=i: writer.write_request_input(
                    {"bundle": idx}
                ),
            )

        # Now create a new bundle with both max_bundles and max_total_bytes
        # max_bundles will mark some for deletion, size limit should skip those
        retention = BundleRetentionPolicy(
            max_bundles=2,  # Will mark oldest for deletion
            max_total_bytes=100_000_000,  # Large enough to not delete more
        )
        config = BundleConfig(target=tmp_path, retention=retention)

        path = create_bundle_path(
            tmp_path,
            config=config,
            write_fn=lambda writer: writer.write_request_input({"bundle": 3}),
        )

        assert path.exists()
        # Should have 2 bundles remaining (max_bundles=2)
        remaining = list(tmp_path.glob("*.zip"))
        assert len(remaining) == 2

    def test_retention_size_limit_keeps_bundles_under_limit(
        self, tmp_path: Path
    ) -> None:
        """Test size limit keeps bundles that fit under the total size limit."""
        # Create a bundle first
        first_path = create_bundle_path(
            tmp_path, write_fn=lambda writer: writer.write_request_input({"data": "s"})
        )
        first_size = first_path.stat().st_size

        # Create a second bundle with generous size limit
        # that should keep both bundles
        retention = BundleRetentionPolicy(
            max_total_bytes=first_size * 10  # Plenty of room
        )
        config = BundleConfig(target=tmp_path, retention=retention)

        second_path = create_bundle_path(
            tmp_path,
            config=config,
            write_fn=lambda writer: writer.write_request_input({"data": "small"}),
        )

        # Both bundles should still exist
        assert first_path.exists()
        assert second_path.exists()
        remaining = list(tmp_path.glob("*.zip"))
        assert len(remaining) == 2

    def test_retention_size_limit_deletes_oldest_keeps_newest(
        self, tmp_path: Path
    ) -> None:
        """Test size limit deletes oldest bundles, keeping newest ones."""
        # Create 3 bundles: A (oldest), B, C (newest before new)
        bundle_paths: list[Path] = []
        for i in range(3):
            path = create_bundle_path(
                tmp_path,
                write_fn=lambda writer, idx=i: writer.write_request_input(
                    {"bundle": idx, "data": "x" * 100}
                ),
            )
            bundle_paths.append(path)

        # Get the size of one bundle (they should all be similar)
        bundle_size = bundle_paths[0].stat().st_size

        # Create new bundle with size limit that fits only 2 bundles
        # (new bundle + one existing = 2 bundles worth)
        retention = BundleRetentionPolicy(max_total_bytes=bundle_size * 2 + 100)
        config = BundleConfig(target=tmp_path, retention=retention)

        new_path = create_bundle_path(
            tmp_path,
            config=config,
            write_fn=lambda writer: writer.write_request_input(
                {"bundle": "new", "data": "x" * 100}
            ),
        )

        # The newest bundles should be kept: new_path and bundle_paths[2] (C)
        # The oldest should be deleted: bundle_paths[0] (A) and bundle_paths[1] (B)
        assert not bundle_paths[0].exists(), "Oldest bundle A should be deleted"
        assert not bundle_paths[1].exists(), "Second oldest bundle B should be deleted"
        assert bundle_paths[2].exists(), "Newest existing bundle C should be kept"
        assert new_path.exists(), "New bundle should be kept"

        remaining = list(tmp_path.glob("*.zip"))
        assert len(remaining) == 2

    def test_retention_delete_failure_is_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test bundle deletion failure is logged but doesn't fail."""
        from unittest.mock import patch

        # Create a bundle first
        first_path = create_bundle_path(
            tmp_path, write_fn=lambda writer: writer.write_request_input({"bundle": 0})
        )

        # Create a new bundle with retention that will try to delete the first
        retention = BundleRetentionPolicy(max_bundles=1)
        config = BundleConfig(target=tmp_path, retention=retention)

        # Patch Path.unlink to fail
        original_unlink = Path.unlink

        def failing_unlink(self: Path, missing_ok: bool = False) -> None:
            if str(self).endswith(".zip") and self != first_path:
                return original_unlink(self, missing_ok)
            raise OSError("Simulated deletion failure")

        with patch.object(Path, "unlink", failing_unlink):
            new_path = create_bundle_path(
                tmp_path,
                config=config,
                write_fn=lambda writer: writer.write_request_input({"bundle": 1}),
            )

        # Bundle should still be created
        assert new_path.exists()
        assert "Failed to delete old bundle" in caplog.text

    def test_retention_age_handles_timezone_naive_timestamps(
        self, tmp_path: Path
    ) -> None:
        """Test age limit works with timezone-naive timestamps in manifests."""
        from unittest.mock import MagicMock, patch

        # Create two real bundles first
        old_path = create_bundle_path(
            tmp_path,
            write_fn=lambda writer: writer.write_request_input({"bundle": "o"}),
        )
        recent_path = create_bundle_path(
            tmp_path,
            write_fn=lambda writer: writer.write_request_input({"bundle": "recent"}),
        )

        # Create mock bundles with different timestamps
        mock_old_bundle = MagicMock()
        # Use naive datetime (no tzinfo) - old date
        naive_old_timestamp = datetime(2020, 1, 1, 0, 0, 0)
        mock_old_bundle.manifest.created_at = naive_old_timestamp.isoformat()

        mock_recent_bundle = MagicMock()
        # Use naive datetime (no tzinfo) - recent date (now)
        naive_recent_timestamp = datetime.now().replace(tzinfo=None)
        mock_recent_bundle.manifest.created_at = naive_recent_timestamp.isoformat()

        original_load = DebugBundle.load

        def mock_load(path: Path) -> DebugBundle | MagicMock:
            if path == old_path:
                return mock_old_bundle
            if path == recent_path:
                return mock_recent_bundle
            return original_load(path)

        # Create a new bundle with max_age retention
        retention = BundleRetentionPolicy(max_age_seconds=86400)  # 24 hours
        config = BundleConfig(target=tmp_path, retention=retention)

        with patch.object(DebugBundle, "load", mock_load):
            new_path = create_bundle_path(
                tmp_path,
                config=config,
                write_fn=lambda writer: writer.write_request_input({"bundle": "new"}),
            )

        # The old bundle with naive timestamp should be deleted
        assert not old_path.exists()
        # The recent bundle should be kept
        assert recent_path.exists()
        assert new_path.exists()

    def test_retention_skips_deletion_if_file_changed(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test TOCTOU protection: skip deletion if file identity changed."""
        # Create initial bundles
        paths: list[Path] = []
        for i in range(3):
            path = create_bundle_path(
                tmp_path,
                write_fn=lambda writer, idx=i: writer.write_request_input(
                    {"bundle": idx}
                ),
            )
            paths.append(path)

        # Capture the original inode of the oldest bundle
        original_inode = paths[0].stat().st_ino

        # Create stale file_identity with the original inode
        stale_identity: dict[Path, tuple[int, int]] = {
            paths[0]: (original_inode, paths[0].stat().st_dev)
        }

        # Replace the bundle file (creates new inode)
        original_content = paths[0].read_bytes()
        paths[0].unlink()

        # Create intermediate files to occupy the freed inode slot
        # (some filesystems like tmpfs reuse inodes immediately)
        intermediates: list[Path] = []
        for j in range(10):
            intermediate = tmp_path / f"_inode_occupier_{j}.tmp"
            intermediate.write_bytes(b"x")
            intermediates.append(intermediate)

        # Now recreate the original file - should get a new inode
        paths[0].write_bytes(original_content)
        new_inode = paths[0].stat().st_ino

        # Clean up intermediate files
        for intermediate in intermediates:
            intermediate.unlink()

        # Skip if filesystem still reuses inodes (cannot test TOCTOU in this env)
        if new_inode == original_inode:
            pytest.skip("Filesystem reuses inodes immediately; cannot test TOCTOU")

        # Call _delete_marked_bundles with stale identity
        BundleWriter._delete_marked_bundles({paths[0]}, stale_identity)

        # File should NOT be deleted due to inode mismatch
        assert paths[0].exists()
        assert "Bundle file changed since collection" in caplog.text

    def test_retention_deletes_bundles_without_identity_tracking(
        self, tmp_path: Path
    ) -> None:
        """Test deletion proceeds when file_identity has no entry for bundle.

        This covers the branch where expected_identity is None (no TOCTOU
        verification possible, proceed with deletion).
        """
        bundle_path = create_bundle_path(
            tmp_path, write_fn=lambda writer: writer.write_request_input({"bundle": 0})
        )

        # Call _delete_marked_bundles with empty identity dict (no tracking)
        empty_identity: dict[Path, tuple[int, int]] = {}
        BundleWriter._delete_marked_bundles({bundle_path}, empty_identity)

        # File should be deleted (no identity to verify against)
        assert not bundle_path.exists()


class TestRetentionWithNestedDirectories:
    """Tests for retention policy with nested directory structures (EvalLoop).

    EvalLoop creates bundles at ``{debug_bundle_dir}/{request_id}/{bundle}.zip``.
    The retention policy must search recursively from the config.target directory
    to find all bundles in subdirectories.
    """

    def test_retention_finds_bundles_in_subdirectories(self, tmp_path: Path) -> None:
        """Test retention policy finds and deletes bundles in subdirectories."""
        # Create bundles in nested directories (simulating EvalLoop structure)
        subdir1 = tmp_path / "request-1"
        subdir1.mkdir()
        subdir2 = tmp_path / "request-2"
        subdir2.mkdir()

        # Create bundles in subdirectories (no retention yet)
        paths: list[Path] = []
        for subdir in [subdir1, subdir2]:
            path = create_bundle_path(
                subdir,
                write_fn=lambda writer, sd=subdir: writer.write_request_input(
                    {"subdir": str(sd)}
                ),
            )
            paths.append(path)

        # Create new bundle with retention that limits to 2 total bundles
        # Key: config.target points to the root (tmp_path) for recursive search
        retention = BundleRetentionPolicy(max_bundles=2)
        config = BundleConfig(target=tmp_path, retention=retention)

        subdir3 = tmp_path / "request-3"
        subdir3.mkdir()
        # Bundle written to subdir3, but retention searches from config.target (tmp_path)
        new_path = create_bundle_path(
            subdir3,
            config=config,
            write_fn=lambda writer, sd=subdir3: writer.write_request_input(
                {"subdir": str(sd)}
            ),
        )

        # Should have 2 bundles remaining (oldest deleted)
        remaining = list(tmp_path.glob("**/*.zip"))
        assert len(remaining) == 2
        # The newest bundle should exist
        assert new_path.exists()

    def test_retention_max_bundles_across_nested_dirs(self, tmp_path: Path) -> None:
        """Test max_bundles counts bundles across all subdirectories."""
        # Create bundles in multiple subdirectories (no retention)
        for i in range(4):
            subdir = tmp_path / f"request-{i}"
            subdir.mkdir()
            _ = create_bundle_path(
                subdir,
                write_fn=lambda writer, idx=i: writer.write_request_input(
                    {"request": idx}
                ),
            )

        # Verify 4 bundles exist
        assert len(list(tmp_path.glob("**/*.zip"))) == 4

        # Create new bundle with retention limit
        # config.target is the root for recursive search
        retention = BundleRetentionPolicy(max_bundles=3)
        config = BundleConfig(target=tmp_path, retention=retention)

        subdir = tmp_path / "request-4"
        subdir.mkdir()
        _ = create_bundle_path(
            subdir,
            config=config,
            write_fn=lambda writer: writer.write_request_input({"request": 4}),
        )

        # Should have 3 bundles remaining
        remaining = list(tmp_path.glob("**/*.zip"))
        assert len(remaining) == 3

    def test_retention_size_limit_across_nested_dirs(self, tmp_path: Path) -> None:
        """Test max_total_bytes considers bundles across all subdirectories."""
        # Create bundles in subdirectories (no retention)
        total_size = 0
        for i in range(2):
            subdir = tmp_path / f"request-{i}"
            subdir.mkdir()
            path = create_bundle_path(
                subdir,
                write_fn=lambda writer: writer.write_request_input({"data": "x" * 500}),
            )
            total_size += path.stat().st_size

        # Create new bundle with tight size limit
        retention = BundleRetentionPolicy(max_total_bytes=total_size)
        config = BundleConfig(target=tmp_path, retention=retention)

        subdir = tmp_path / "request-2"
        subdir.mkdir()
        _ = create_bundle_path(
            subdir,
            config=config,
            write_fn=lambda writer: writer.write_request_input({"data": "y" * 500}),
        )

        # Oldest bundles should be deleted to fit under limit
        remaining = list(tmp_path.glob("**/*.zip"))
        # Should have fewer bundles due to size limit
        assert len(remaining) < 3

    def test_retention_skips_newly_created_bundle_in_nested_dir(
        self, tmp_path: Path
    ) -> None:
        """Test that the newly created bundle is not considered for deletion."""
        # Create a bundle in a subdirectory (no retention)
        subdir = tmp_path / "request-1"
        subdir.mkdir()
        old_path = create_bundle_path(
            subdir, write_fn=lambda writer: writer.write_request_input({"old": True})
        )

        # Create new bundle with max_bundles=1 in same target tree
        retention = BundleRetentionPolicy(max_bundles=1)
        config = BundleConfig(target=tmp_path, retention=retention)

        new_subdir = tmp_path / "request-2"
        new_subdir.mkdir()
        new_path = create_bundle_path(
            new_subdir,
            config=config,
            write_fn=lambda writer: writer.write_request_input({"new": True}),
        )

        # Old bundle should be deleted, new one kept
        assert not old_path.exists()
        assert new_path.exists()

    def test_retention_without_config_target_uses_writer_target(
        self, tmp_path: Path
    ) -> None:
        """Test retention falls back to writer target if config.target is None."""
        # Create bundles in tmp_path
        first_path = create_bundle_path(
            tmp_path,
            write_fn=lambda writer: writer.write_request_input({"first": True}),
        )

        # Create new bundle with retention but no config.target
        # This means retention will only search in the writer's target (tmp_path)
        retention = BundleRetentionPolicy(max_bundles=1)
        config = BundleConfig(retention=retention)  # No target set

        second_path = create_bundle_path(
            tmp_path,
            config=config,
            write_fn=lambda writer: writer.write_request_input({"second": True}),
        )

        # Old bundle should be deleted
        assert not first_path.exists()
        assert second_path.exists()

    def test_retention_skips_symlinks(self, tmp_path: Path) -> None:
        """Test retention policy skips symlinks to prevent loops."""
        # Create initial bundles
        paths: list[Path] = []
        for i in range(3):
            path = create_bundle_path(
                tmp_path,
                write_fn=lambda writer, idx=i: writer.write_request_input(
                    {"bundle": idx}
                ),
            )
            paths.append(path)

        # Create a symlink to one of the bundles
        symlink_path = tmp_path / "symlink_bundle.zip"
        symlink_path.symlink_to(paths[0])

        # Also create a symlink in a subdirectory
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        nested_symlink = subdir / "nested_symlink.zip"
        nested_symlink.symlink_to(paths[1])

        # Now create new bundle with retention that limits to 2
        # Symlinks should be skipped, so only real bundles count
        retention = BundleRetentionPolicy(max_bundles=2)
        config = BundleConfig(target=tmp_path, retention=retention)

        new_path = create_bundle_path(
            tmp_path,
            config=config,
            write_fn=lambda writer: writer.write_request_input({"new": True}),
        )

        # Oldest bundle should be deleted (paths[0])
        assert not paths[0].exists()
        # Symlinks should still exist (not deleted, just skipped)
        assert symlink_path.is_symlink()
        assert nested_symlink.is_symlink()
        # New bundle and one old bundle should remain
        assert new_path.exists()


class TestBundleConfigWithRetentionAndStorage:
    """Tests for BundleConfig with retention and storage handler fields."""

    def test_config_default_values(self) -> None:
        """Test BundleConfig has None defaults for retention and storage."""
        config = BundleConfig()
        assert config.retention is None
        assert config.storage_handler is None

    def test_config_with_retention(self, tmp_path: Path) -> None:
        """Test BundleConfig accepts retention policy."""
        retention = BundleRetentionPolicy(max_bundles=5)
        config = BundleConfig(target=tmp_path, retention=retention)
        assert config.retention is retention
        assert config.retention.max_bundles == 5

    def test_config_with_storage_handler(self, tmp_path: Path) -> None:
        """Test BundleConfig accepts storage handler."""

        class TestHandler:
            def store_bundle(self, bundle_path: Path, manifest: BundleManifest) -> None:
                pass

        handler = TestHandler()
        config = BundleConfig(target=tmp_path, storage_handler=handler)
        assert config.storage_handler is handler

    def test_config_with_both_retention_and_storage(self, tmp_path: Path) -> None:
        """Test BundleConfig accepts both retention and storage handler."""

        class TestHandler:
            def store_bundle(self, bundle_path: Path, manifest: BundleManifest) -> None:
                pass

        retention = BundleRetentionPolicy(max_bundles=10)
        handler = TestHandler()
        config = BundleConfig(
            target=tmp_path,
            retention=retention,
            storage_handler=handler,
        )
        assert config.retention is retention
        assert config.storage_handler is handler

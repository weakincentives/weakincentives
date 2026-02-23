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

"""Tests for filesystem history in debug bundles."""

from __future__ import annotations

from pathlib import Path

import pytest

from weakincentives.debug import BundleWriter, DebugBundle
from weakincentives.filesystem import HostFilesystem


class TestBundleWriterFilesystemHistory:
    """Tests for BundleWriter.write_filesystem_history."""

    def test_write_history_with_host_filesystem(self, tmp_path: Path) -> None:
        """Test writing filesystem history from a HostFilesystem with snapshots."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        fs = HostFilesystem(_root=str(workspace))

        # Create snapshots
        _ = fs.snapshot(tag="pre:write_file:call_001")
        (workspace / "hello.txt").write_text("Hello")
        _ = fs.snapshot(tag="pre:read_file:call_002")

        bundle_dir = tmp_path / "bundles"
        with BundleWriter(bundle_dir) as writer:
            writer.write_filesystem_history(fs)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)

        # Should have filesystem history
        assert bundle.has_filesystem_history
        history = bundle.filesystem_history
        assert history is not None
        assert history["format_version"] == "1.0.0"
        assert history["snapshot_count"] >= 2
        assert len(history["snapshots"]) >= 2

        # Check manifest files include history
        files = bundle.list_files()
        assert "filesystem_history/manifest.json" in files
        assert "filesystem_history/history.bundle" in files

    def test_write_history_noop_for_inmemory(self, tmp_path: Path) -> None:
        """Test write_filesystem_history is a no-op for InMemoryFilesystem."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem

        fs = InMemoryFilesystem()
        _ = fs.write("/test.txt", "hello")

        with BundleWriter(tmp_path) as writer:
            writer.write_filesystem_history(fs)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert not bundle.has_filesystem_history
        assert bundle.filesystem_history is None

    def test_write_history_noop_without_git_init(self, tmp_path: Path) -> None:
        """Test write_filesystem_history is a no-op if git is not initialized."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        fs = HostFilesystem(_root=str(workspace))
        # Don't call snapshot(), so git is not initialized

        with BundleWriter(tmp_path / "bundles") as writer:
            writer.write_filesystem_history(fs)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert not bundle.has_filesystem_history

    def test_write_history_handles_errors(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test write_filesystem_history handles exceptions gracefully."""

        class FailingFS:
            pass

        with BundleWriter(tmp_path) as writer:
            writer.write_filesystem_history(FailingFS())  # type: ignore[arg-type]

        # Should not fail, just log
        assert writer.path is not None

    def test_snapshot_metadata_in_manifest(self, tmp_path: Path) -> None:
        """Test that snapshot entries contain expected metadata fields."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        fs = HostFilesystem(_root=str(workspace))

        _ = fs.snapshot(tag="pre:write_file:call_001")
        (workspace / "foo.py").write_text("print('hi')")
        _ = fs.snapshot(tag="pre:read_file:call_002")

        bundle_dir = tmp_path / "bundles"
        with BundleWriter(bundle_dir) as writer:
            writer.write_filesystem_history(fs)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        history = bundle.filesystem_history
        assert history is not None

        for snap in history["snapshots"]:
            assert "commit_ref" in snap
            assert "created_at" in snap
            assert "tag" in snap
            assert "parent_ref" in snap
            assert "tool_call_id" in snap
            assert "tool_name" in snap
            assert "files_changed" in snap
            assert "insertions" in snap
            assert "deletions" in snap
            assert "rolled_back" in snap


class TestDebugBundleFilesystemHistory:
    """Tests for DebugBundle filesystem history reading."""

    def test_has_filesystem_history_false_without_history(self, tmp_path: Path) -> None:
        """Test has_filesystem_history is False when no history captured."""
        with BundleWriter(tmp_path) as writer:
            writer.write_request_input({"task": "test"})

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert not bundle.has_filesystem_history

    def test_extract_filesystem_history(self, tmp_path: Path) -> None:
        """Test extracting git bundle to a local repository."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        fs = HostFilesystem(_root=str(workspace))

        _ = fs.snapshot(tag="pre:init:call_000")
        (workspace / "test.txt").write_text("content")
        _ = fs.snapshot(tag="pre:write:call_001")

        bundle_dir = tmp_path / "bundles"
        with BundleWriter(bundle_dir) as writer:
            writer.write_filesystem_history(fs)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)

        extract_dir = tmp_path / "extracted"
        repo_path = bundle.extract_filesystem_history(extract_dir)
        assert repo_path is not None
        assert repo_path.exists()
        assert (repo_path / ".git").exists()

    def test_extract_returns_none_without_history(self, tmp_path: Path) -> None:
        """Test extract_filesystem_history returns None when no history."""
        with BundleWriter(tmp_path) as writer:
            writer.write_request_input({"task": "test"})

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        result = bundle.extract_filesystem_history(tmp_path / "extracted")
        assert result is None

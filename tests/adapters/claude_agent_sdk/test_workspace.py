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

"""Tests for Claude Agent SDK workspace management."""

from __future__ import annotations

from pathlib import Path

import pytest

from weakincentives.adapters.claude_agent_sdk.workspace import (
    ClaudeAgentWorkspace,
    HostMount,
    HostMountPreview,
    WorkspaceBudgetExceededError,
    WorkspaceSecurityError,
    cleanup_workspace,
    create_workspace,
)


class TestHostMount:
    """Tests for HostMount dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        mount = HostMount(host_path="/path/to/dir")
        assert mount.host_path == "/path/to/dir"
        assert mount.mount_path is None
        assert mount.include_glob == ()
        assert mount.exclude_glob == ()
        assert mount.max_bytes is None
        assert mount.follow_symlinks is False

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        mount = HostMount(
            host_path="/src",
            mount_path="project/src",
            include_glob=("*.py", "*.txt"),
            exclude_glob=("*_test.py",),
            max_bytes=1000000,
            follow_symlinks=True,
        )
        assert mount.host_path == "/src"
        assert mount.mount_path == "project/src"
        assert mount.include_glob == ("*.py", "*.txt")
        assert mount.exclude_glob == ("*_test.py",)
        assert mount.max_bytes == 1000000
        assert mount.follow_symlinks is True


class TestHostMountPreview:
    """Tests for HostMountPreview dataclass."""

    def test_creation(self) -> None:
        """Test creating a preview."""
        preview = HostMountPreview(
            host_path="/src",
            resolved_host=Path("/home/user/project/src"),
            mount_path="src",
            entries=("main.py", "utils.py"),
            is_directory=True,
            bytes_copied=1500,
        )
        assert preview.host_path == "/src"
        assert preview.resolved_host == Path("/home/user/project/src")
        assert preview.mount_path == "src"
        assert preview.entries == ("main.py", "utils.py")
        assert preview.is_directory is True
        assert preview.bytes_copied == 1500


class TestClaudeAgentWorkspace:
    """Tests for ClaudeAgentWorkspace dataclass."""

    def test_creation(self) -> None:
        """Test creating a workspace."""
        temp_dir = Path("/tmp/test-workspace")
        preview = HostMountPreview(
            host_path="/src",
            resolved_host=Path("/home/user/src"),
            mount_path="src",
            entries=(),
            is_directory=True,
            bytes_copied=0,
        )
        from datetime import UTC, datetime

        workspace = ClaudeAgentWorkspace(
            temp_dir=temp_dir,
            mount_previews=(preview,),
            created_at=datetime.now(UTC),
        )
        assert workspace.temp_dir == temp_dir
        assert len(workspace.mount_previews) == 1


class TestCreateWorkspace:
    """Tests for create_workspace function."""

    def test_create_workspace_with_file(self, tmp_path: Path) -> None:
        """Test creating workspace with a single file."""
        # Create a source file
        source_file = tmp_path / "test.txt"
        source_file.write_text("hello world")

        # Create workspace
        workspace = create_workspace(
            mounts=[HostMount(host_path=str(source_file))],
            allowed_host_roots=[str(tmp_path)],
        )

        try:
            # Verify workspace was created
            assert workspace.temp_dir.exists()
            assert len(workspace.mount_previews) == 1

            # Verify file was copied
            preview = workspace.mount_previews[0]
            assert preview.is_directory is False
            assert preview.bytes_copied == 11

            copied_file = workspace.temp_dir / "test.txt"
            assert copied_file.exists()
            assert copied_file.read_text() == "hello world"
        finally:
            cleanup_workspace(workspace)

    def test_create_workspace_with_directory(self, tmp_path: Path) -> None:
        """Test creating workspace with a directory."""
        # Create source directory with files
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        (source_dir / "main.py").write_text("print('hello')")
        (source_dir / "utils.py").write_text("def helper(): pass")

        # Create workspace
        workspace = create_workspace(
            mounts=[HostMount(host_path=str(source_dir))],
            allowed_host_roots=[str(tmp_path)],
        )

        try:
            # Verify files were copied
            assert workspace.temp_dir.exists()
            preview = workspace.mount_previews[0]
            assert preview.is_directory is True
            assert len(preview.entries) == 2

            copied_dir = workspace.temp_dir / "src"
            assert (copied_dir / "main.py").exists()
            assert (copied_dir / "utils.py").exists()
        finally:
            cleanup_workspace(workspace)

    def test_create_workspace_with_include_glob(self, tmp_path: Path) -> None:
        """Test creating workspace with include glob filter."""
        # Create source directory with mixed files
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        (source_dir / "main.py").write_text("code")
        (source_dir / "data.json").write_text("{}")
        (source_dir / "readme.md").write_text("# Readme")

        # Create workspace with include glob
        workspace = create_workspace(
            mounts=[
                HostMount(
                    host_path=str(source_dir),
                    include_glob=("*.py",),
                ),
            ],
            allowed_host_roots=[str(tmp_path)],
        )

        try:
            # Verify only Python files were copied
            preview = workspace.mount_previews[0]
            assert "main.py" in preview.entries
            assert "data.json" not in preview.entries
            assert "readme.md" not in preview.entries
        finally:
            cleanup_workspace(workspace)

    def test_create_workspace_with_exclude_glob(self, tmp_path: Path) -> None:
        """Test creating workspace with exclude glob filter."""
        # Create source directory with files
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        (source_dir / "main.py").write_text("code")
        (source_dir / "main_test.py").write_text("test")
        (source_dir / "utils.py").write_text("utils")

        # Create workspace with exclude glob
        workspace = create_workspace(
            mounts=[
                HostMount(
                    host_path=str(source_dir),
                    exclude_glob=("*_test.py",),
                ),
            ],
            allowed_host_roots=[str(tmp_path)],
        )

        try:
            # Verify test files were excluded
            preview = workspace.mount_previews[0]
            assert "main.py" in preview.entries
            assert "utils.py" in preview.entries
            assert "main_test.py" not in preview.entries
        finally:
            cleanup_workspace(workspace)

    def test_create_workspace_with_max_bytes(self, tmp_path: Path) -> None:
        """Test creating workspace with byte limit."""
        # Create source directory with files
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        (source_dir / "small.txt").write_text("small")
        (source_dir / "large.txt").write_text("x" * 1000)

        # Create workspace with small byte limit
        with pytest.raises(WorkspaceBudgetExceededError):
            create_workspace(
                mounts=[
                    HostMount(
                        host_path=str(source_dir),
                        max_bytes=100,
                    ),
                ],
                allowed_host_roots=[str(tmp_path)],
            )

    def test_create_workspace_security_boundary(self, tmp_path: Path) -> None:
        """Test that workspace enforces security boundaries."""
        # Create file outside allowed roots
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        (other_dir / "secret.txt").write_text("secret")

        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        # Try to mount file outside allowed roots
        with pytest.raises(WorkspaceSecurityError):
            create_workspace(
                mounts=[HostMount(host_path=str(other_dir / "secret.txt"))],
                allowed_host_roots=[str(allowed_dir)],
            )

    def test_create_workspace_nonexistent_path(self, tmp_path: Path) -> None:
        """Test creating workspace with nonexistent path."""
        with pytest.raises(WorkspaceSecurityError):
            create_workspace(
                mounts=[HostMount(host_path=str(tmp_path / "nonexistent"))],
                allowed_host_roots=[str(tmp_path)],
            )

    def test_create_workspace_custom_mount_path(self, tmp_path: Path) -> None:
        """Test creating workspace with custom mount path."""
        source_file = tmp_path / "test.txt"
        source_file.write_text("content")

        workspace = create_workspace(
            mounts=[
                HostMount(
                    host_path=str(source_file),
                    mount_path="custom/path/file.txt",
                ),
            ],
            allowed_host_roots=[str(tmp_path)],
        )

        try:
            copied_file = workspace.temp_dir / "custom/path/file.txt"
            assert copied_file.exists()
            assert copied_file.read_text() == "content"
        finally:
            cleanup_workspace(workspace)


class TestCleanupWorkspace:
    """Tests for cleanup_workspace function."""

    def test_cleanup_removes_temp_dir(self, tmp_path: Path) -> None:
        """Test that cleanup removes the temporary directory."""
        source_file = tmp_path / "test.txt"
        source_file.write_text("content")

        workspace = create_workspace(
            mounts=[HostMount(host_path=str(source_file))],
            allowed_host_roots=[str(tmp_path)],
        )

        temp_dir = workspace.temp_dir
        assert temp_dir.exists()

        cleanup_workspace(workspace)
        assert not temp_dir.exists()

    def test_cleanup_handles_already_removed(self, tmp_path: Path) -> None:
        """Test that cleanup handles already-removed directories gracefully."""
        source_file = tmp_path / "test.txt"
        source_file.write_text("content")

        workspace = create_workspace(
            mounts=[HostMount(host_path=str(source_file))],
            allowed_host_roots=[str(tmp_path)],
        )

        # Remove manually first
        import shutil

        shutil.rmtree(workspace.temp_dir)

        # Should not raise
        cleanup_workspace(workspace)

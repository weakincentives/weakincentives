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

import tempfile
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
    def test_defaults(self) -> None:
        mount = HostMount(host_path="/home/user/src")
        assert mount.host_path == "/home/user/src"
        assert mount.mount_path is None
        assert mount.include_glob == ()
        assert mount.exclude_glob == ()
        assert mount.max_bytes is None
        assert mount.follow_symlinks is False

    def test_with_all_options(self) -> None:
        mount = HostMount(
            host_path="/home/user/src",
            mount_path="project/src",
            include_glob=("*.py", "*.txt"),
            exclude_glob=("__pycache__/*",),
            max_bytes=1000000,
            follow_symlinks=True,
        )
        assert mount.host_path == "/home/user/src"
        assert mount.mount_path == "project/src"
        assert mount.include_glob == ("*.py", "*.txt")
        assert mount.exclude_glob == ("__pycache__/*",)
        assert mount.max_bytes == 1000000
        assert mount.follow_symlinks is True


class TestHostMountPreview:
    def test_construction(self) -> None:
        preview = HostMountPreview(
            host_path="src",
            resolved_host=Path("/home/user/project/src"),
            mount_path="src",
            entries=("main.py", "utils.py"),
            is_directory=True,
            bytes_copied=5000,
        )
        assert preview.host_path == "src"
        assert preview.resolved_host == Path("/home/user/project/src")
        assert preview.mount_path == "src"
        assert preview.entries == ("main.py", "utils.py")
        assert preview.is_directory is True
        assert preview.bytes_copied == 5000


class TestClaudeAgentWorkspace:
    def test_construction(self) -> None:
        from datetime import UTC, datetime

        temp_dir = Path("/tmp/wink-test")
        now = datetime.now(UTC)
        workspace = ClaudeAgentWorkspace(
            temp_dir=temp_dir,
            mount_previews=(),
            created_at=now,
        )
        assert workspace.temp_dir == temp_dir
        assert workspace.mount_previews == ()
        assert workspace.created_at == now


class TestCreateWorkspace:
    def test_creates_temp_directory(self) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            (host_path / "test.txt").write_text("hello")

            workspace = create_workspace(
                mounts=[HostMount(host_path=host_dir)],
            )

            try:
                assert workspace.temp_dir.exists()
                assert workspace.temp_dir.is_dir()
                assert len(workspace.mount_previews) == 1
            finally:
                cleanup_workspace(workspace)

    def test_copies_single_file(self) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            test_file = host_path / "test.txt"
            test_file.write_text("hello world")

            workspace = create_workspace(
                mounts=[HostMount(host_path=str(test_file))],
            )

            try:
                copied_file = workspace.temp_dir / "test.txt"
                assert copied_file.exists()
                assert copied_file.read_text() == "hello world"
                assert workspace.mount_previews[0].is_directory is False
                assert workspace.mount_previews[0].bytes_copied == 11
            finally:
                cleanup_workspace(workspace)

    def test_copies_directory(self) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            (host_path / "file1.txt").write_text("content1")
            (host_path / "subdir").mkdir()
            (host_path / "subdir" / "file2.txt").write_text("content2")

            workspace = create_workspace(
                mounts=[HostMount(host_path=host_dir)],
            )

            try:
                base_name = Path(host_dir).name
                assert (workspace.temp_dir / base_name / "file1.txt").exists()
                assert (
                    workspace.temp_dir / base_name / "subdir" / "file2.txt"
                ).exists()
                assert workspace.mount_previews[0].is_directory is True
            finally:
                cleanup_workspace(workspace)

    def test_custom_mount_path(self) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            (host_path / "test.txt").write_text("hello")

            workspace = create_workspace(
                mounts=[HostMount(host_path=host_dir, mount_path="custom/path")],
            )

            try:
                assert (workspace.temp_dir / "custom/path" / "test.txt").exists()
            finally:
                cleanup_workspace(workspace)

    def test_include_glob_filter(self) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            (host_path / "include.py").write_text("python")
            (host_path / "exclude.txt").write_text("text")

            workspace = create_workspace(
                mounts=[HostMount(host_path=host_dir, include_glob=("*.py",))],
            )

            try:
                base_name = Path(host_dir).name
                assert (workspace.temp_dir / base_name / "include.py").exists()
                assert not (workspace.temp_dir / base_name / "exclude.txt").exists()
            finally:
                cleanup_workspace(workspace)

    def test_exclude_glob_filter(self) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            (host_path / "keep.txt").write_text("keep")
            (host_path / "remove.log").write_text("remove")

            workspace = create_workspace(
                mounts=[HostMount(host_path=host_dir, exclude_glob=("*.log",))],
            )

            try:
                base_name = Path(host_dir).name
                assert (workspace.temp_dir / base_name / "keep.txt").exists()
                assert not (workspace.temp_dir / base_name / "remove.log").exists()
            finally:
                cleanup_workspace(workspace)

    def test_byte_budget_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            (host_path / "large.txt").write_text("x" * 1000)

            with pytest.raises(WorkspaceBudgetExceededError, match="byte budget"):
                create_workspace(
                    mounts=[HostMount(host_path=host_dir, max_bytes=100)],
                )

    def test_byte_budget_file_exceeds(self) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            large_file = host_path / "large.txt"
            large_file.write_text("x" * 1000)

            with pytest.raises(WorkspaceBudgetExceededError):
                create_workspace(
                    mounts=[HostMount(host_path=str(large_file), max_bytes=100)],
                )

    def test_security_boundary_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as allowed_dir:
            with tempfile.TemporaryDirectory() as forbidden_dir:
                (Path(forbidden_dir) / "secret.txt").write_text("secret")

                with pytest.raises(WorkspaceSecurityError, match="outside allowed"):
                    create_workspace(
                        mounts=[HostMount(host_path=forbidden_dir)],
                        allowed_host_roots=[allowed_dir],
                    )

    def test_security_boundary_allows_within_root(self) -> None:
        with tempfile.TemporaryDirectory() as allowed_dir:
            host_path = Path(allowed_dir)
            (host_path / "allowed.txt").write_text("allowed")

            workspace = create_workspace(
                mounts=[HostMount(host_path=allowed_dir)],
                allowed_host_roots=[allowed_dir],
            )

            try:
                assert workspace.temp_dir.exists()
            finally:
                cleanup_workspace(workspace)

    def test_nonexistent_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            create_workspace(
                mounts=[HostMount(host_path="/nonexistent/path/12345")],
            )

    def test_custom_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            (Path(host_dir) / "test.txt").write_text("test")

            workspace = create_workspace(
                mounts=[HostMount(host_path=host_dir)],
                temp_dir_prefix="custom-prefix-",
            )

            try:
                assert "custom-prefix-" in workspace.temp_dir.name
            finally:
                cleanup_workspace(workspace)

    def test_multiple_mounts(self) -> None:
        with tempfile.TemporaryDirectory() as dir1:
            with tempfile.TemporaryDirectory() as dir2:
                (Path(dir1) / "file1.txt").write_text("content1")
                (Path(dir2) / "file2.txt").write_text("content2")

                workspace = create_workspace(
                    mounts=[
                        HostMount(host_path=dir1, mount_path="mount1"),
                        HostMount(host_path=dir2, mount_path="mount2"),
                    ],
                )

                try:
                    assert (workspace.temp_dir / "mount1" / "file1.txt").exists()
                    assert (workspace.temp_dir / "mount2" / "file2.txt").exists()
                    assert len(workspace.mount_previews) == 2
                finally:
                    cleanup_workspace(workspace)


class TestCleanupWorkspace:
    def test_removes_temp_directory(self) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            (Path(host_dir) / "test.txt").write_text("test")

            workspace = create_workspace(
                mounts=[HostMount(host_path=host_dir)],
            )

            temp_dir = workspace.temp_dir
            assert temp_dir.exists()

            cleanup_workspace(workspace)

            assert not temp_dir.exists()

    def test_handles_already_deleted(self) -> None:
        import shutil

        with tempfile.TemporaryDirectory() as host_dir:
            (Path(host_dir) / "test.txt").write_text("test")

            workspace = create_workspace(
                mounts=[HostMount(host_path=host_dir)],
            )

            shutil.rmtree(workspace.temp_dir)

            cleanup_workspace(workspace)

    def test_handles_nonexistent(self) -> None:
        from datetime import UTC, datetime

        workspace = ClaudeAgentWorkspace(
            temp_dir=Path("/nonexistent/12345"),
            mount_previews=(),
            created_at=datetime.now(UTC),
        )

        cleanup_workspace(workspace)

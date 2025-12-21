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

import shutil
import tempfile
from pathlib import Path

import pytest

from weakincentives.adapters.claude_agent_sdk.workspace import (
    ClaudeAgentWorkspaceSection,
    HostMount,
    HostMountPreview,
    WorkspaceBudgetExceededError,
    WorkspaceSecurityError,
)
from weakincentives.contrib.tools.filesystem import Filesystem
from weakincentives.contrib.tools.workspace import WorkspaceSection
from weakincentives.runtime import InProcessEventBus, Session


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


@pytest.fixture
def session() -> Session:
    bus = InProcessEventBus()
    return Session(bus=bus)


class TestClaudeAgentWorkspaceSectionCore:
    """Core functionality tests for ClaudeAgentWorkspaceSection."""

    def test_implements_workspace_section_protocol(self, session: Session) -> None:
        section = ClaudeAgentWorkspaceSection(session=session)
        try:
            assert isinstance(section, WorkspaceSection)
        finally:
            section.cleanup()

    def test_creates_empty_workspace_when_no_mounts(self, session: Session) -> None:
        section = ClaudeAgentWorkspaceSection(session=session)
        try:
            assert section.temp_dir.exists()
            assert section.mount_previews == ()
        finally:
            section.cleanup()

    def test_creates_workspace_from_mounts(self, session: Session) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            (host_path / "test.py").write_text("print('hello')")

            section = ClaudeAgentWorkspaceSection(
                session=session,
                mounts=[HostMount(host_path=host_dir)],
            )

            try:
                assert section.temp_dir.exists()
                assert len(section.mount_previews) == 1
            finally:
                section.cleanup()

    def test_session_property_returns_session(self, session: Session) -> None:
        section = ClaudeAgentWorkspaceSection(session=session)
        try:
            assert section.session is session
        finally:
            section.cleanup()

    def test_temp_dir_property(self, session: Session) -> None:
        section = ClaudeAgentWorkspaceSection(session=session)
        try:
            assert isinstance(section.temp_dir, Path)
            assert section.temp_dir.exists()
        finally:
            section.cleanup()

    def test_mount_previews_property(self, session: Session) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            (host_path / "test.txt").write_text("content")

            section = ClaudeAgentWorkspaceSection(
                session=session,
                mounts=[HostMount(host_path=host_dir)],
            )

            try:
                assert isinstance(section.mount_previews, tuple)
                assert len(section.mount_previews) == 1
                assert isinstance(section.mount_previews[0], HostMountPreview)
            finally:
                section.cleanup()

    def test_created_at_property(self, session: Session) -> None:
        from datetime import datetime

        section = ClaudeAgentWorkspaceSection(session=session)
        try:
            assert isinstance(section.created_at, datetime)
        finally:
            section.cleanup()

    def test_filesystem_property(self, session: Session) -> None:
        section = ClaudeAgentWorkspaceSection(session=session)
        try:
            fs = section.filesystem
            assert isinstance(fs, Filesystem)
            # Filesystem should be empty by default
            assert fs.list(".") == []
        finally:
            section.cleanup()


class TestClaudeAgentWorkspaceSectionCleanup:
    """Cleanup and clone tests for ClaudeAgentWorkspaceSection."""

    def test_cleanup_removes_temp_directory(self, session: Session) -> None:
        section = ClaudeAgentWorkspaceSection(session=session)
        temp_dir = section.temp_dir
        assert temp_dir.exists()
        section.cleanup()
        assert not temp_dir.exists()

    def test_cleanup_handles_already_deleted(self, session: Session) -> None:
        section = ClaudeAgentWorkspaceSection(session=session)
        shutil.rmtree(section.temp_dir)
        section.cleanup()  # Should not raise

    def test_clone_creates_new_section_with_same_workspace(
        self, session: Session
    ) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            (Path(host_dir) / "file.txt").write_text("content")

            section = ClaudeAgentWorkspaceSection(
                session=session,
                mounts=[HostMount(host_path=host_dir)],
            )

            try:
                new_bus = InProcessEventBus()
                new_session = Session(bus=new_bus)
                cloned = section.clone(session=new_session)

                assert cloned is not section
                assert cloned.session is new_session
                assert cloned.temp_dir == section.temp_dir
                assert cloned.mount_previews == section.mount_previews
            finally:
                section.cleanup()

    def test_clone_preserves_filesystem_instance(self, session: Session) -> None:
        section = ClaudeAgentWorkspaceSection(session=session)

        try:
            # Write a file through the filesystem
            fs = section.filesystem
            _ = fs.write("test_file.txt", "test content")

            # Clone to a new session
            new_bus = InProcessEventBus()
            new_session = Session(bus=new_bus)
            cloned = section.clone(session=new_session)

            # Filesystem instance should be the same
            assert cloned.filesystem is section.filesystem

            # Content should be accessible through the cloned section's filesystem
            result = cloned.filesystem.read("test_file.txt")
            assert result.content == "test content"
        finally:
            section.cleanup()

    def test_clone_requires_session(self, session: Session) -> None:
        section = ClaudeAgentWorkspaceSection(session=session)

        try:
            with pytest.raises(TypeError, match="session is required"):
                section.clone()
        finally:
            section.cleanup()

    def test_clone_rejects_mismatched_bus(self, session: Session) -> None:
        section = ClaudeAgentWorkspaceSection(session=session)

        try:
            new_bus = InProcessEventBus()
            new_session = Session(bus=new_bus)
            other_bus = InProcessEventBus()

            with pytest.raises(TypeError, match="bus must match"):
                section.clone(session=new_session, bus=other_bus)
        finally:
            section.cleanup()


class TestClaudeAgentWorkspaceSectionTemplate:
    """Template rendering tests for ClaudeAgentWorkspaceSection."""

    def test_template_renders_mounted_content(self, session: Session) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            (host_path / "main.py").write_text("print('main')")
            (host_path / "utils.py").write_text("def helper(): pass")

            section = ClaudeAgentWorkspaceSection(
                session=session,
                mounts=[HostMount(host_path=host_dir)],
            )

            try:
                rendered = section.render(None, depth=1, number="1")
                assert "mounted content:" in rendered
                assert "(directory):" in rendered
            finally:
                section.cleanup()

    def test_template_shows_no_mounts_message(self, session: Session) -> None:
        section = ClaudeAgentWorkspaceSection(session=session)

        try:
            rendered = section.render(None, depth=1, number="1")
            assert "(no host mounts configured)" in rendered
        finally:
            section.cleanup()

    def test_template_shows_mount_with_no_entries(self, session: Session) -> None:
        """Test template rendering when a mount has no entries (empty directory)."""
        with tempfile.TemporaryDirectory() as host_dir:
            # Create an empty directory - no files
            section = ClaudeAgentWorkspaceSection(
                session=session,
                mounts=[HostMount(host_path=host_dir)],
            )

            try:
                rendered = section.render(None, depth=1, number="1")
                # Should still show the mount with total bytes
                assert "(directory):" in rendered
                assert "Total:" in rendered
            finally:
                section.cleanup()

    def test_template_shows_file_mount(self, session: Session) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            test_file = host_path / "single.txt"
            test_file.write_text("single file content")

            section = ClaudeAgentWorkspaceSection(
                session=session,
                mounts=[HostMount(host_path=str(test_file))],
            )

            try:
                rendered = section.render(None, depth=1, number="1")
                assert "(file):" in rendered
            finally:
                section.cleanup()

    def test_template_truncates_large_entry_lists(self, session: Session) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            # Create more than _TEMPLATE_PREVIEW_LIMIT (10) files
            for i in range(15):
                (host_path / f"file{i:02d}.txt").write_text(f"content {i}")

            section = ClaudeAgentWorkspaceSection(
                session=session,
                mounts=[HostMount(host_path=host_dir)],
            )

            try:
                rendered = section.render(None, depth=1, number="1")
                assert "... and 5 more" in rendered
            finally:
                section.cleanup()


class TestClaudeAgentWorkspaceSectionMounting:
    """File and directory mounting tests for ClaudeAgentWorkspaceSection."""

    def test_copies_single_file(self, session: Session) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            test_file = host_path / "test.txt"
            test_file.write_text("hello world")

            section = ClaudeAgentWorkspaceSection(
                session=session,
                mounts=[HostMount(host_path=str(test_file))],
            )

            try:
                copied_file = section.temp_dir / "test.txt"
                assert copied_file.exists()
                assert copied_file.read_text() == "hello world"
                assert section.mount_previews[0].is_directory is False
                assert section.mount_previews[0].bytes_copied == 11
            finally:
                section.cleanup()

    def test_copies_directory(self, session: Session) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            (host_path / "file1.txt").write_text("content1")
            (host_path / "subdir").mkdir()
            (host_path / "subdir" / "file2.txt").write_text("content2")

            section = ClaudeAgentWorkspaceSection(
                session=session,
                mounts=[HostMount(host_path=host_dir)],
            )

            try:
                base_name = Path(host_dir).name
                assert (section.temp_dir / base_name / "file1.txt").exists()
                assert (section.temp_dir / base_name / "subdir" / "file2.txt").exists()
                assert section.mount_previews[0].is_directory is True
            finally:
                section.cleanup()

    def test_custom_mount_path(self, session: Session) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            (host_path / "test.txt").write_text("hello")

            section = ClaudeAgentWorkspaceSection(
                session=session,
                mounts=[HostMount(host_path=host_dir, mount_path="custom/path")],
            )

            try:
                assert (section.temp_dir / "custom/path" / "test.txt").exists()
            finally:
                section.cleanup()

    def test_multiple_mounts(self, session: Session) -> None:
        with tempfile.TemporaryDirectory() as dir1:
            with tempfile.TemporaryDirectory() as dir2:
                (Path(dir1) / "file1.txt").write_text("content1")
                (Path(dir2) / "file2.txt").write_text("content2")

                section = ClaudeAgentWorkspaceSection(
                    session=session,
                    mounts=[
                        HostMount(host_path=dir1, mount_path="mount1"),
                        HostMount(host_path=dir2, mount_path="mount2"),
                    ],
                )

                try:
                    assert (section.temp_dir / "mount1" / "file1.txt").exists()
                    assert (section.temp_dir / "mount2" / "file2.txt").exists()
                    assert len(section.mount_previews) == 2
                finally:
                    section.cleanup()


class TestClaudeAgentWorkspaceSectionFiltering:
    """Glob filtering tests for ClaudeAgentWorkspaceSection."""

    def test_include_glob_filter(self, session: Session) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            (host_path / "include.py").write_text("python")
            (host_path / "exclude.txt").write_text("text")

            section = ClaudeAgentWorkspaceSection(
                session=session,
                mounts=[HostMount(host_path=host_dir, include_glob=("*.py",))],
            )

            try:
                base_name = Path(host_dir).name
                assert (section.temp_dir / base_name / "include.py").exists()
                assert not (section.temp_dir / base_name / "exclude.txt").exists()
            finally:
                section.cleanup()

    def test_exclude_glob_filter(self, session: Session) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            (host_path / "keep.txt").write_text("keep")
            (host_path / "remove.log").write_text("remove")

            section = ClaudeAgentWorkspaceSection(
                session=session,
                mounts=[HostMount(host_path=host_dir, exclude_glob=("*.log",))],
            )

            try:
                base_name = Path(host_dir).name
                assert (section.temp_dir / base_name / "keep.txt").exists()
                assert not (section.temp_dir / base_name / "remove.log").exists()
            finally:
                section.cleanup()


class TestClaudeAgentWorkspaceSectionSecurity:
    """Security and error handling tests for ClaudeAgentWorkspaceSection."""

    def test_byte_budget_enforced(self, session: Session) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            (host_path / "large.txt").write_text("x" * 1000)

            with pytest.raises(WorkspaceBudgetExceededError, match="byte budget"):
                ClaudeAgentWorkspaceSection(
                    session=session,
                    mounts=[HostMount(host_path=host_dir, max_bytes=100)],
                )

    def test_byte_budget_file_exceeds(self, session: Session) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            large_file = host_path / "large.txt"
            large_file.write_text("x" * 1000)

            with pytest.raises(WorkspaceBudgetExceededError):
                ClaudeAgentWorkspaceSection(
                    session=session,
                    mounts=[HostMount(host_path=str(large_file), max_bytes=100)],
                )

    def test_security_boundary_enforced(self, session: Session) -> None:
        with tempfile.TemporaryDirectory() as allowed_dir:
            with tempfile.TemporaryDirectory() as forbidden_dir:
                (Path(forbidden_dir) / "secret.txt").write_text("secret")

                with pytest.raises(WorkspaceSecurityError, match="outside allowed"):
                    ClaudeAgentWorkspaceSection(
                        session=session,
                        mounts=[HostMount(host_path=forbidden_dir)],
                        allowed_host_roots=[allowed_dir],
                    )

    def test_security_boundary_allows_within_root(self, session: Session) -> None:
        with tempfile.TemporaryDirectory() as allowed_dir:
            host_path = Path(allowed_dir)
            (host_path / "allowed.txt").write_text("allowed")

            section = ClaudeAgentWorkspaceSection(
                session=session,
                mounts=[HostMount(host_path=allowed_dir)],
                allowed_host_roots=[allowed_dir],
            )

            try:
                assert section.temp_dir.exists()
            finally:
                section.cleanup()

    def test_nonexistent_path_raises(self, session: Session) -> None:
        with pytest.raises(FileNotFoundError):
            ClaudeAgentWorkspaceSection(
                session=session,
                mounts=[HostMount(host_path="/nonexistent/path/12345")],
            )

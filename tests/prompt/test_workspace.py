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

"""Tests for the generic WorkspaceSection class."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from weakincentives.filesystem import Filesystem
from weakincentives.prompt.protocols import WorkspaceSectionProtocol
from weakincentives.prompt.workspace import (
    HostMount,
    HostMountPreview,
    WorkspaceBudgetExceededError,
    WorkspaceSection,
    WorkspaceSecurityError,
)
from weakincentives.runtime import InProcessDispatcher, Session


@pytest.fixture
def session() -> Session:
    dispatcher = InProcessDispatcher()
    return Session(dispatcher=dispatcher)


@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory for test fixtures."""
    d = Path(tempfile.mkdtemp(prefix="wink-test-"))
    yield d  # type: ignore[misc]
    shutil.rmtree(d, ignore_errors=True)


class TestWorkspaceSectionProtocolConformance:
    def test_implements_workspace_section_protocol(self, session: Session) -> None:
        section = WorkspaceSection(session=session)
        try:
            assert isinstance(section, WorkspaceSectionProtocol)
        finally:
            section.cleanup()


class TestWorkspaceSectionCore:
    """Core functionality tests for WorkspaceSection."""

    def test_creates_empty_workspace_when_no_mounts(self, session: Session) -> None:
        section = WorkspaceSection(session=session)
        try:
            assert section.temp_dir.exists()
            assert section.mount_previews == ()
        finally:
            section.cleanup()

    def test_creates_workspace_from_mounts(self, session: Session) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            (host_path / "test.py").write_text("print('hello')")

            section = WorkspaceSection(
                session=session,
                mounts=[HostMount(host_path=host_dir)],
            )

            try:
                assert section.temp_dir.exists()
                assert len(section.mount_previews) == 1
            finally:
                section.cleanup()

    def test_session_property_returns_session(self, session: Session) -> None:
        section = WorkspaceSection(session=session)
        try:
            assert section.session is session
        finally:
            section.cleanup()

    def test_temp_dir_property(self, session: Session) -> None:
        section = WorkspaceSection(session=session)
        try:
            assert isinstance(section.temp_dir, Path)
            assert section.temp_dir.exists()
        finally:
            section.cleanup()

    def test_mount_previews_property(self, session: Session) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            (host_path / "test.txt").write_text("content")

            section = WorkspaceSection(
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

        section = WorkspaceSection(session=session)
        try:
            assert isinstance(section.created_at, datetime)
        finally:
            section.cleanup()

    def test_filesystem_property(self, session: Session) -> None:
        section = WorkspaceSection(session=session)
        try:
            fs = section.filesystem
            assert isinstance(fs, Filesystem)
            assert fs.list(".") == []
        finally:
            section.cleanup()

    def test_workspace_fingerprint(self, session: Session) -> None:
        section = WorkspaceSection(session=session)
        try:
            fp = section.workspace_fingerprint
            assert isinstance(fp, str)
            assert len(fp) == 16
        finally:
            section.cleanup()

    def test_custom_key(self, session: Session) -> None:
        section = WorkspaceSection(session=session, key="my-workspace")
        try:
            assert section.key == "my-workspace"
        finally:
            section.cleanup()

    def test_default_key(self, session: Session) -> None:
        section = WorkspaceSection(session=session)
        try:
            assert section.key == "workspace"
        finally:
            section.cleanup()


class TestWorkspaceSectionCleanup:
    """Cleanup and clone tests for WorkspaceSection."""

    def test_cleanup_removes_temp_directory(self, session: Session) -> None:
        section = WorkspaceSection(session=session)
        temp_dir = section.temp_dir
        assert temp_dir.exists()
        section.cleanup()
        assert not temp_dir.exists()

    def test_cleanup_handles_already_deleted(self, session: Session) -> None:
        section = WorkspaceSection(session=session)
        shutil.rmtree(section.temp_dir)
        section.cleanup()  # Should not raise

    def test_cleanup_removes_filesystem_git_directory(self, session: Session) -> None:
        section = WorkspaceSection(session=session)
        fs = section.filesystem

        _ = fs.write("test.txt", "content")
        _ = fs.snapshot(tag="test-snapshot")

        git_dir = fs.git_dir
        assert git_dir is not None
        assert Path(git_dir).exists()

        temp_dir = section.temp_dir
        section.cleanup()

        assert not temp_dir.exists()
        assert not Path(git_dir).exists()

    def test_cleanup_handles_non_host_filesystem(self, session: Session) -> None:
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem

        section = WorkspaceSection(session=session)
        temp_dir = section.temp_dir
        assert temp_dir.exists()

        section._filesystem = InMemoryFilesystem()

        section.cleanup()
        assert not temp_dir.exists()

    def test_cleanup_already_cleaned(self, session: Session) -> None:
        """Calling cleanup twice should not raise."""
        section = WorkspaceSection(session=session)
        section.cleanup()
        section.cleanup()

    def test_clone_creates_new_section_with_same_workspace(
        self, session: Session
    ) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            (Path(host_dir) / "file.txt").write_text("content")

            section = WorkspaceSection(
                session=session,
                mounts=[HostMount(host_path=host_dir)],
            )

            try:
                new_dispatcher = InProcessDispatcher()
                new_session = Session(dispatcher=new_dispatcher)
                cloned = section.clone(session=new_session)

                assert cloned is not section
                assert cloned.session is new_session
                assert cloned.temp_dir == section.temp_dir
                assert cloned.mount_previews == section.mount_previews
            finally:
                section.cleanup()

    def test_clone_preserves_filesystem_instance(self, session: Session) -> None:
        section = WorkspaceSection(session=session)

        try:
            fs = section.filesystem
            _ = fs.write("test_file.txt", "test content")

            new_dispatcher = InProcessDispatcher()
            new_session = Session(dispatcher=new_dispatcher)
            cloned = section.clone(session=new_session)

            assert cloned.filesystem is section.filesystem

            result = cloned.filesystem.read("test_file.txt")
            assert result.content == "test content"
        finally:
            section.cleanup()

    def test_clone_requires_session(self, session: Session) -> None:
        section = WorkspaceSection(session=session)

        try:
            with pytest.raises(TypeError, match="session is required"):
                section.clone()
        finally:
            section.cleanup()

    def test_clone_rejects_mismatched_dispatcher(self, session: Session) -> None:
        section = WorkspaceSection(session=session)

        try:
            new_dispatcher = InProcessDispatcher()
            new_session = Session(dispatcher=new_dispatcher)
            other_dispatcher = InProcessDispatcher()

            with pytest.raises(TypeError, match="dispatcher must match"):
                section.clone(session=new_session, dispatcher=other_dispatcher)
        finally:
            section.cleanup()

    def test_clone_requires_session_type(self, session: Session) -> None:
        section = WorkspaceSection(session=session)
        try:
            with pytest.raises(TypeError, match="session is required"):
                section.clone(session="not a session")
        finally:
            section.cleanup()


class TestWorkspaceSectionRefCounting:
    """Reference counting tests for clone/cleanup."""

    def test_cleanup_original_preserves_temp_dir_when_clone_alive(
        self, session: Session
    ) -> None:
        section = WorkspaceSection(session=session)
        new_session = Session(dispatcher=InProcessDispatcher())
        cloned = section.clone(session=new_session)
        temp = section.temp_dir
        assert temp.exists()

        section.cleanup()
        assert temp.exists()

        cloned.cleanup()
        assert not temp.exists()

    def test_cleanup_clone_preserves_temp_dir_when_original_alive(
        self, session: Session
    ) -> None:
        section = WorkspaceSection(session=session)
        new_session = Session(dispatcher=InProcessDispatcher())
        cloned = section.clone(session=new_session)
        temp = section.temp_dir

        cloned.cleanup()
        assert temp.exists()

        section.cleanup()
        assert not temp.exists()


class TestWorkspaceSectionTemplate:
    """Template rendering tests for WorkspaceSection."""

    def test_template_renders_mounted_content(self, session: Session) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            (host_path / "main.py").write_text("print('main')")
            (host_path / "utils.py").write_text("def helper(): pass")

            section = WorkspaceSection(
                session=session,
                mounts=[HostMount(host_path=host_dir)],
            )

            try:
                rendered = section.render(None, depth=1, number="1")
                assert "mounted content" in rendered
                assert "(directory):" in rendered
            finally:
                section.cleanup()

    def test_template_shows_no_mounts_message(self, session: Session) -> None:
        section = WorkspaceSection(session=session)

        try:
            rendered = section.render(None, depth=1, number="1")
            assert "(no host mounts configured)" in rendered
        finally:
            section.cleanup()

    def test_template_shows_mount_with_no_entries(self, session: Session) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            section = WorkspaceSection(
                session=session,
                mounts=[HostMount(host_path=host_dir)],
            )

            try:
                rendered = section.render(None, depth=1, number="1")
                assert "(directory):" in rendered
                assert "Total:" in rendered
            finally:
                section.cleanup()

    def test_template_shows_file_mount(self, session: Session) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            test_file = host_path / "single.txt"
            test_file.write_text("single file content")

            section = WorkspaceSection(
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
            for i in range(15):
                (host_path / f"file{i:02d}.txt").write_text(f"content {i}")

            section = WorkspaceSection(
                session=session,
                mounts=[HostMount(host_path=host_dir)],
            )

            try:
                rendered = section.render(None, depth=1, number="1")
                assert "... and 5 more" in rendered
            finally:
                section.cleanup()


class TestWorkspaceSectionMounting:
    """File and directory mounting tests for WorkspaceSection."""

    def test_copies_single_file(self, session: Session) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            test_file = host_path / "test.txt"
            test_file.write_text("hello world")

            section = WorkspaceSection(
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

            section = WorkspaceSection(
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

            section = WorkspaceSection(
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

                section = WorkspaceSection(
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


class TestWorkspaceSectionFiltering:
    """Glob filtering tests for WorkspaceSection."""

    def test_include_glob_filter(self, session: Session) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            (host_path / "include.py").write_text("python")
            (host_path / "exclude.txt").write_text("text")

            section = WorkspaceSection(
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

            section = WorkspaceSection(
                session=session,
                mounts=[HostMount(host_path=host_dir, exclude_glob=("*.log",))],
            )

            try:
                base_name = Path(host_dir).name
                assert (section.temp_dir / base_name / "keep.txt").exists()
                assert not (section.temp_dir / base_name / "remove.log").exists()
            finally:
                section.cleanup()


class TestWorkspaceSectionSecurity:
    """Security and error handling tests for WorkspaceSection."""

    def test_byte_budget_enforced(self, session: Session) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            (host_path / "large.txt").write_text("x" * 1000)

            with pytest.raises(WorkspaceBudgetExceededError, match="byte budget"):
                WorkspaceSection(
                    session=session,
                    mounts=[HostMount(host_path=host_dir, max_bytes=100)],
                )

    def test_byte_budget_file_exceeds(self, session: Session) -> None:
        with tempfile.TemporaryDirectory() as host_dir:
            host_path = Path(host_dir)
            large_file = host_path / "large.txt"
            large_file.write_text("x" * 1000)

            with pytest.raises(WorkspaceBudgetExceededError):
                WorkspaceSection(
                    session=session,
                    mounts=[HostMount(host_path=str(large_file), max_bytes=100)],
                )

    def test_security_boundary_enforced(self, session: Session) -> None:
        with tempfile.TemporaryDirectory() as allowed_dir:
            with tempfile.TemporaryDirectory() as forbidden_dir:
                (Path(forbidden_dir) / "secret.txt").write_text("secret")

                with pytest.raises(WorkspaceSecurityError, match="outside allowed"):
                    WorkspaceSection(
                        session=session,
                        mounts=[HostMount(host_path=forbidden_dir)],
                        allowed_host_roots=[allowed_dir],
                    )

    def test_security_boundary_allows_within_root(self, session: Session) -> None:
        with tempfile.TemporaryDirectory() as allowed_dir:
            host_path = Path(allowed_dir)
            (host_path / "allowed.txt").write_text("allowed")

            section = WorkspaceSection(
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
            WorkspaceSection(
                session=session,
                mounts=[HostMount(host_path="/nonexistent/path/12345")],
            )


class TestWorkspaceSectionResources:
    """Tests for WorkspaceSection.configure() method."""

    def test_configure_registers_filesystem(self, session: Session) -> None:
        from weakincentives.resources.builder import RegistryBuilder

        with tempfile.TemporaryDirectory() as temp_dir:
            section = WorkspaceSection(
                session=session,
                mounts=[HostMount(host_path=temp_dir)],
            )

            try:
                builder = RegistryBuilder()
                section.configure(builder)
                registry = builder.build()

                with registry.open() as context:
                    fs = context.get(Filesystem)
                    assert fs is section._filesystem
            finally:
                section.cleanup()


class TestPreCreatedWorkspace:
    def test_pre_created_workspace(self, session: Session, temp_dir: Path) -> None:
        workspace = temp_dir / "prebuilt"
        workspace.mkdir()
        previews = (
            HostMountPreview(
                host_path="/src",
                resolved_host=Path("/src"),
                mount_path="src",
                entries=(),
                is_directory=True,
                bytes_copied=0,
            ),
        )
        section = WorkspaceSection(
            session=session,
            _temp_dir=workspace,
            _mount_previews=previews,
        )
        assert section.temp_dir == workspace
        assert section.mount_previews == previews

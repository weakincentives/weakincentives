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

"""Sandbox security tests for PodmanSandboxSection."""

from __future__ import annotations

from pathlib import Path

import pytest

import weakincentives.contrib.tools.podman as podman_module
import weakincentives.contrib.tools.vfs as vfs_module
from tests.tools.podman_test_helpers import (
    prepare_resolved_mount,
)
from weakincentives import ToolValidationError
from weakincentives.contrib.tools import HostMount


def test_host_mount_resolver_rejects_empty_path(tmp_path: Path) -> None:
    with pytest.raises(ToolValidationError):
        podman_module._resolve_single_host_mount(
            HostMount(host_path=""),
            (tmp_path,),
        )


def test_resolve_host_path_requires_allowed_roots() -> None:
    with pytest.raises(ToolValidationError):
        podman_module._resolve_host_path("docs", ())


def test_resolve_host_path_rejects_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ToolValidationError):
        podman_module._resolve_host_path("../outside", (root,))


def test_normalize_mount_globs_discards_empty_entries() -> None:
    result = podman_module._normalize_mount_globs(
        (" *.py ", " ", "*.md"),
        "include_glob",
    )
    assert result == ("*.py", "*.md")


def test_container_path_for_root() -> None:
    assert podman_module._container_path_for(vfs_module.VfsPath(())) == "/workspace"


def test_assert_within_overlay_raises_for_outside(tmp_path: Path) -> None:
    root = tmp_path / "overlay"
    root.mkdir()
    outside = root.parent / "other"
    outside.mkdir()
    with pytest.raises(ToolValidationError):
        podman_module._assert_within_overlay(root, outside)


def test_assert_within_overlay_handles_missing_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "overlay"
    root.mkdir()
    missing = root / "missing" / "child"
    original = Path.resolve

    def _fake_resolve(self: Path, strict: bool = False) -> Path:
        if self == missing:
            raise FileNotFoundError("missing")
        return original(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", _fake_resolve)
    podman_module._assert_within_overlay(root, missing)


def test_copy_mount_respects_include_glob(tmp_path: Path) -> None:
    resolved, file_path, overlay = prepare_resolved_mount(
        tmp_path, include_glob=("*.py",)
    )
    podman_module.PodmanSandboxSection._copy_mount_into_overlay(
        overlay=overlay,
        mount=resolved,
    )
    target = overlay / "sunfish" / file_path.name
    assert not target.exists()


def test_copy_mount_respects_exclude_glob(tmp_path: Path) -> None:
    resolved, file_path, overlay = prepare_resolved_mount(
        tmp_path, exclude_glob=("*.txt",)
    )
    podman_module.PodmanSandboxSection._copy_mount_into_overlay(
        overlay=overlay,
        mount=resolved,
    )
    target = overlay / "sunfish" / file_path.name
    assert not target.exists()


def test_copy_mount_stat_failure_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    resolved, file_path, overlay = prepare_resolved_mount(tmp_path)
    original_stat = Path.stat

    def _raise(self: Path) -> os.stat_result:
        if self == file_path:
            raise OSError("boom")
        return original_stat(self)

    monkeypatch.setattr(Path, "stat", _raise)
    with pytest.raises(ToolValidationError):
        podman_module.PodmanSandboxSection._copy_mount_into_overlay(
            overlay=overlay,
            mount=resolved,
        )


def test_copy_mount_max_bytes_guard(tmp_path: Path) -> None:
    resolved, _file_path, overlay = prepare_resolved_mount(tmp_path, max_bytes=1)
    with pytest.raises(ToolValidationError):
        podman_module.PodmanSandboxSection._copy_mount_into_overlay(
            overlay=overlay,
            mount=resolved,
        )


def test_resolve_host_path_finds_file_in_allowed_root() -> None:
    """Test branch 411->405: _resolve_host_path finds existing file."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Resolve the temp directory to handle macOS /var -> /private/var symlink
        resolved_tmpdir = Path(tmpdir).resolve()

        # Create a test file in the allowed root
        test_file = resolved_tmpdir / "test.txt"
        test_file.write_text("content")

        # Should find the file in the allowed root (branch 411: candidate.exists())
        # Use resolved path to avoid symlink resolution mismatches
        allowed_roots = (resolved_tmpdir,)
        resolved = podman_module._resolve_host_path("test.txt", allowed_roots)
        assert resolved == test_file


def test_resolve_host_path_continues_when_file_not_in_first_root() -> None:
    """Test branch 411->405: continue when candidate doesn't exist in first root."""
    import tempfile

    with tempfile.TemporaryDirectory() as first_root:
        with tempfile.TemporaryDirectory() as second_root:
            # Resolve paths to handle macOS /var -> /private/var symlink
            resolved_first = Path(first_root).resolve()
            resolved_second = Path(second_root).resolve()

            # Create file only in second root
            test_file = resolved_second / "test.txt"
            test_file.write_text("content")

            # First root doesn't have file, should continue to second root
            allowed_roots = (resolved_first, resolved_second)
            resolved = podman_module._resolve_host_path("test.txt", allowed_roots)
            assert resolved == test_file

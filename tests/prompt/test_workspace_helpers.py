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

"""Tests for workspace template rendering.

The mount machinery moved to :mod:`weakincentives.sandbox`; its tests live
in ``tests/sandbox/test_mounts.py``. Only the prompt-side template helper
is covered here.
"""

from __future__ import annotations

from pathlib import Path

from weakincentives.prompt.workspace import (
    HostMountPreview,
    _render_workspace_template,
)


class TestRenderWorkspaceTemplate:
    def test_no_previews(self) -> None:
        result = _render_workspace_template(())
        assert "no host mounts" in result

    def test_with_directory_preview(self) -> None:
        preview = HostMountPreview(
            host_path="/src",
            resolved_host=Path("/src"),
            mount_path="src",
            entries=("file1.py", "file2.py"),
            is_directory=True,
            bytes_copied=100,
        )
        result = _render_workspace_template((preview,))
        assert "src" in result
        assert "directory" in result
        assert "file1.py" in result
        assert "100" in result

    def test_with_file_preview(self) -> None:
        preview = HostMountPreview(
            host_path="/src/main.py",
            resolved_host=Path("/src/main.py"),
            mount_path="main.py",
            entries=("main.py",),
            is_directory=False,
            bytes_copied=42,
        )
        result = _render_workspace_template((preview,))
        assert "file" in result
        assert "main.py" in result

    def test_many_entries_truncated(self) -> None:
        entries = tuple(f"file{i}.py" for i in range(25))
        preview = HostMountPreview(
            host_path="/src",
            resolved_host=Path("/src"),
            mount_path="src",
            entries=entries,
            is_directory=True,
            bytes_copied=1000,
        )
        result = _render_workspace_template((preview,))
        assert "more" in result

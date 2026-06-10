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

"""Tests for the workspace preview rendered from an opened sandbox.

The mount machinery lives in :mod:`weakincentives.sandbox` and is covered
by ``tests/sandbox/``. These tests cover the prompt-side preview: the
section auto-appended by ``PromptTemplate.create(sandbox=...)`` and the
params built from the opened sandbox's filesystem.
"""

from __future__ import annotations

from weakincentives.filesystem import Filesystem
from weakincentives.prompt import (
    WORKSPACE_PREVIEW_KEY,
    Prompt,
    PromptTemplate,
    WorkspacePreviewParams,
    workspace_preview_params,
    workspace_preview_section,
)
from weakincentives.sandbox import SandboxConfig


class TestWorkspacePreviewParams:
    def test_listing_from_filesystem(self) -> None:
        fs = Filesystem.in_memory()
        _ = fs.write("notes.md", "hello")
        _ = fs.write("src/main.py", "print('hi')")

        params = workspace_preview_params(fs)

        assert "- notes.md" in params.listing
        assert "- src/" in params.listing

    def test_empty_filesystem_renders_placeholder(self) -> None:
        fs = Filesystem.in_memory()

        params = workspace_preview_params(fs)

        assert params.listing == "(empty workspace)"

    def test_listing_truncates_beyond_limit(self) -> None:
        fs = Filesystem.in_memory()
        for index in range(6):
            _ = fs.write(f"file{index:02d}.txt", "x")

        params = workspace_preview_params(fs, limit=4)

        assert "- file03.txt" in params.listing
        assert "- file04.txt" not in params.listing
        assert "- ... and 2 more" in params.listing

    def test_default_params_show_pending_placeholder(self) -> None:
        assert "not yet materialized" in WorkspacePreviewParams().listing


class TestWorkspacePreviewSection:
    def test_renders_default_placeholder(self) -> None:
        section = workspace_preview_section()

        rendered = section.render(WorkspacePreviewParams(), depth=1, number="1")

        assert "Workspace" in rendered
        assert "not yet materialized" in rendered

    def test_renders_bound_listing(self) -> None:
        section = workspace_preview_section()
        params = WorkspacePreviewParams(listing="- a.txt\n- b/")

        rendered = section.render(params, depth=1, number="1")

        assert "- a.txt" in rendered
        assert "- b/" in rendered

    def test_custom_key(self) -> None:
        section = workspace_preview_section(key="env-preview")
        assert section.key == "env-preview"

    def test_rejects_overrides(self) -> None:
        section = workspace_preview_section()
        assert section.accepts_overrides is False


class TestTemplateSandboxIntegration:
    def test_sandbox_config_appends_preview_section(self) -> None:
        template = PromptTemplate.create(
            ns="tests",
            key="with-sandbox",
            sandbox=SandboxConfig(),
        )

        assert template.sandbox == SandboxConfig()
        keys = [section.key for section in template.sections]
        assert keys[-1] == WORKSPACE_PREVIEW_KEY

    def test_no_sandbox_means_no_preview_section(self) -> None:
        template = PromptTemplate.create(ns="tests", key="plain")

        assert template.sandbox is None
        assert template.sections == ()

    def test_render_uses_bound_preview_params(self) -> None:
        fs = Filesystem.in_memory()
        _ = fs.write("README.md", "docs")
        template = PromptTemplate.create(
            ns="tests",
            key="render-preview",
            sandbox=SandboxConfig(),
        )
        prompt = Prompt(template).bind(workspace_preview_params(fs))

        rendered = prompt.render()

        assert "- README.md" in rendered.text

    def test_render_without_binding_shows_placeholder(self) -> None:
        template = PromptTemplate.create(
            ns="tests",
            key="render-placeholder",
            sandbox=SandboxConfig(),
        )

        rendered = Prompt(template).render()

        assert "not yet materialized" in rendered.text

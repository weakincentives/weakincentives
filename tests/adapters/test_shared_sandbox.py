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

"""Tests for the shared adapter sandbox lifecycle helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers.sandbox import make_memory_sandbox
from weakincentives.adapters._shared import open_prompt_sandbox
from weakincentives.prompt import (
    Prompt,
    PromptTemplate,
    ToolContext,
)
from weakincentives.sandbox import (
    HostMount,
    LocalSandboxProvider,
    Sandbox,
    SandboxConfig,
)


class TestOpenPromptSandbox:
    def test_opens_empty_sandbox_without_template_config(self) -> None:
        prompt: Prompt[object] = Prompt(PromptTemplate.create(ns="t", key="plain"))
        provider = LocalSandboxProvider()

        sandbox = open_prompt_sandbox(prompt, provider)
        try:
            assert Path(sandbox.root).is_dir()
            assert sandbox.filesystem.root == sandbox.root
        finally:
            sandbox.close()

    def test_binds_preview_from_opened_sandbox(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("hello")
        template: PromptTemplate[object] = PromptTemplate.create(
            ns="t",
            key="with-sandbox",
            sandbox=SandboxConfig(
                mounts=(HostMount(host_path=str(tmp_path), mount_path="src"),)
            ),
        )
        prompt = Prompt(template)
        provider = LocalSandboxProvider()

        sandbox = open_prompt_sandbox(prompt, provider)
        try:
            rendered = prompt.render()
            assert "- src/" in rendered.text
            # Exit criterion: context filesystem root == sandbox root == cwd
            assert sandbox.filesystem.root == sandbox.root
        finally:
            sandbox.close()

    def test_closes_sandbox_when_preview_binding_fails(self) -> None:
        template: PromptTemplate[object] = PromptTemplate.create(
            ns="t",
            key="failing-preview",
            sandbox=SandboxConfig(),
        )
        prompt = Prompt(template)
        opened: list[Sandbox] = []

        class ExplodingFilesystem:
            backend = None

            def list(self, path: str = ".") -> list[object]:
                raise RuntimeError("listing failed")

        class ExplodingProvider:
            def open(self, config: SandboxConfig) -> Sandbox:
                sandbox = make_memory_sandbox()
                sandbox._filesystem = ExplodingFilesystem()  # type: ignore[assignment]
                opened.append(sandbox)
                return sandbox

        with pytest.raises(RuntimeError, match="listing failed"):
            _ = open_prompt_sandbox(prompt, ExplodingProvider())

        assert len(opened) == 1
        with pytest.raises(Exception):  # noqa: B017 - closed sandbox rejects access
            _ = opened[0].filesystem


class TestToolContextShell:
    def test_shell_facet_exposed_from_sandbox(self) -> None:
        sandbox = make_memory_sandbox()
        context = ToolContext(
            prompt=Prompt(PromptTemplate.create(ns="t", key="shell")),  # type: ignore[arg-type]
            rendered_prompt=None,
            adapter=None,  # type: ignore[arg-type]
            session=None,  # type: ignore[arg-type]
            sandbox=sandbox,
        )

        assert context.shell is sandbox.shell
        assert context.filesystem is sandbox.filesystem

    def test_facets_none_without_sandbox(self) -> None:
        context = ToolContext(
            prompt=Prompt(PromptTemplate.create(ns="t", key="no-shell")),  # type: ignore[arg-type]
            rendered_prompt=None,
            adapter=None,  # type: ignore[arg-type]
            session=None,  # type: ignore[arg-type]
        )

        assert context.shell is None
        assert context.filesystem is None

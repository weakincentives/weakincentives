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

"""Tests for the adapter sandbox lease and the workspace preview binding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.helpers.sandbox import make_memory_sandbox
from weakincentives.adapters._shared import bind_workspace_preview
from weakincentives.adapters.core import PromptResponse, ProviderAdapter
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.deadlines import Deadline
from weakincentives.prompt import (
    Prompt,
    PromptTemplate,
    ToolContext,
)
from weakincentives.runtime.run_context import RunContext
from weakincentives.runtime.session.protocols import SessionProtocol
from weakincentives.sandbox import (
    HostMount,
    Sandbox,
    SandboxConfig,
    SandboxProvider,
)


class _RecordingAdapter(ProviderAdapter[Any]):
    """Minimal adapter recording the sandboxes its core contract receives."""

    def __init__(self, *, sandbox_provider: SandboxProvider | None = None) -> None:
        super().__init__(sandbox_provider=sandbox_provider)
        self.seen: list[Sandbox] = []

    def _evaluate[OutputT](
        self,
        prompt: Prompt[OutputT],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
        heartbeat: object = None,
        run_context: RunContext | None = None,
        sandbox: Sandbox,
    ) -> PromptResponse[OutputT]:
        del session, deadline, budget, budget_tracker, heartbeat, run_context
        self.seen.append(sandbox)
        _ = sandbox.filesystem.write(f"round-{len(self.seen)}.txt", "x")
        return PromptResponse(prompt_name=prompt.key, text="ok", output=None)


def _plain_prompt(key: str = "plain") -> Prompt[Any]:
    return Prompt(PromptTemplate.create(ns="t", key=key))


class TestOpenSandboxLease:
    def test_lease_materializes_template_config(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("hello")
        prompt: Prompt[Any] = Prompt(
            PromptTemplate.create(
                ns="t",
                key="with-mounts",
                sandbox=SandboxConfig(
                    mounts=(HostMount(host_path=str(tmp_path), mount_path="src"),)
                ),
            )
        )
        adapter = _RecordingAdapter()

        with adapter.open_sandbox(prompt) as sandbox:
            assert sandbox.filesystem.exists("src/README.md")
            assert sandbox.filesystem.root == sandbox.root
            root = Path(sandbox.root)
            assert root.is_dir()

        # Lease released on exit: locally provisioned sandboxes are removed.
        assert not root.exists()

    def test_lease_defaults_to_empty_config(self) -> None:
        opened: list[SandboxConfig] = []

        class _SpyProvider:
            def open(self, config: SandboxConfig) -> Sandbox:
                opened.append(config)
                return make_memory_sandbox()

        adapter = _RecordingAdapter(sandbox_provider=_SpyProvider())

        with adapter.open_sandbox(_plain_prompt()):
            pass

        assert opened == [SandboxConfig()]


class TestEvaluateLeaseFork:
    def test_owned_path_opens_and_releases(self) -> None:
        adapter = _RecordingAdapter()
        prompt = _plain_prompt()

        response = adapter.evaluate(prompt, session=None)  # type: ignore[arg-type]

        assert response.text == "ok"
        assert len(adapter.seen) == 1
        # The adapter-owned lease is released after evaluate returns.
        assert not Path(adapter.seen[0].root).exists()

    def test_borrowed_sandbox_spans_multiple_evaluations(self) -> None:
        adapter = _RecordingAdapter()
        prompt = _plain_prompt(key="borrowed")

        with adapter.open_sandbox(prompt) as sandbox:
            _ = adapter.evaluate(prompt, session=None, sandbox=sandbox)  # type: ignore[arg-type]
            _ = adapter.evaluate(prompt, session=None, sandbox=sandbox)  # type: ignore[arg-type]

            # Same lease both rounds; round-1 output visible in round 2.
            assert adapter.seen == [sandbox, sandbox]
            assert sandbox.filesystem.exists("round-1.txt")
            assert sandbox.filesystem.exists("round-2.txt")

        assert not Path(sandbox.root).exists()

    def test_borrowed_sandbox_not_closed_by_adapter(self) -> None:
        adapter = _RecordingAdapter()
        prompt = _plain_prompt(key="not-closed")
        sandbox = make_memory_sandbox()

        _ = adapter.evaluate(prompt, session=None, sandbox=sandbox)  # type: ignore[arg-type]

        # Borrowed lease stays open: facets remain accessible.
        assert sandbox.filesystem.exists("round-1.txt")
        sandbox.close()


class TestBindWorkspacePreview:
    def test_binds_listing_from_sandbox(self) -> None:
        prompt: Prompt[Any] = Prompt(
            PromptTemplate.create(ns="t", key="preview", sandbox=SandboxConfig())
        )
        sandbox = make_memory_sandbox()
        _ = sandbox.filesystem.write("README.md", "docs")

        bind_workspace_preview(prompt, sandbox)

        assert "- README.md" in prompt.render().text

    def test_rebind_refreshes_listing(self) -> None:
        prompt: Prompt[Any] = Prompt(
            PromptTemplate.create(ns="t", key="refresh", sandbox=SandboxConfig())
        )
        sandbox = make_memory_sandbox()
        _ = sandbox.filesystem.write("first.txt", "1")

        bind_workspace_preview(prompt, sandbox)
        assert "- second.txt" not in prompt.render().text

        _ = sandbox.filesystem.write("second.txt", "2")
        bind_workspace_preview(prompt, sandbox)

        rendered = prompt.render().text
        assert "- first.txt" in rendered
        assert "- second.txt" in rendered

    def test_noop_without_template_sandbox(self) -> None:
        prompt = _plain_prompt(key="no-preview")
        sandbox = make_memory_sandbox()
        _ = sandbox.filesystem.write("README.md", "docs")

        bind_workspace_preview(prompt, sandbox)

        assert prompt.params == ()
        assert "README.md" not in prompt.render().text


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

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

"""Adapter-specific ACK scenarios for workspace isolation behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from weakincentives.prompt import (
    HostMount,
    MarkdownSection,
    Prompt,
    PromptTemplate,
    WorkspaceSection,
)
from weakincentives.runtime.session import Session

from ..adapters import AdapterFixture

pytestmark = pytest.mark.ack_capability("workspace_isolation")


def test_workspace_section_mounts_host_files(
    adapter_fixture: AdapterFixture,
    session: Session,
    tmp_path: Path,
) -> None:
    """Workspace mounts expose host files to the adapter execution sandbox."""
    host_dir = tmp_path / "host"
    host_dir.mkdir()
    source_file = host_dir / "mounted_file.txt"
    source_file.write_text("mounted content from host")

    workspace = WorkspaceSection(
        session=session,
        mounts=[HostMount(host_path=str(host_dir), mount_path="data")],
        allowed_host_roots=[tmp_path],
    )

    try:
        adapter = adapter_fixture.create_adapter(workspace.temp_dir)
        prompt = Prompt(
            PromptTemplate(
                ns="integration/ack/workspace",
                key="host-mount",
                name="ack_workspace_mount",
                sections=[
                    workspace,
                    MarkdownSection(
                        title="Task",
                        key="task",
                        template=(
                            "Read data/mounted_file.txt and reply with its exact content."
                        ),
                    ),
                ],
            )
        )

        response = adapter.evaluate(prompt, session=session)

        assert response.text is not None
        assert "mounted content from host" in response.text
    finally:
        workspace.cleanup()


def test_workspace_uses_temp_dir_as_cwd(
    adapter_fixture: AdapterFixture,
    session: Session,
) -> None:
    """Workspace temp_dir is used as effective cwd during execution."""
    workspace = WorkspaceSection(session=session)

    try:
        adapter = adapter_fixture.create_adapter(workspace.temp_dir)
        prompt = Prompt(
            PromptTemplate(
                ns="integration/ack/workspace",
                key="cwd",
                name="ack_workspace_cwd",
                sections=[
                    workspace,
                    MarkdownSection(
                        title="Task",
                        key="task",
                        template=(
                            "Run `pwd` and reply with the current directory path only."
                        ),
                    ),
                ],
            )
        )

        response = adapter.evaluate(prompt, session=session)

        assert response.text is not None
        assert str(workspace.temp_dir) in response.text
    finally:
        workspace.cleanup()


@pytest.mark.ack_capability("custom_env_forwarding")
def test_custom_env_forwarded(
    adapter_fixture: AdapterFixture,
    session: Session,
    tmp_path: Path,
) -> None:
    """Adapter forwards configured environment variables to subprocess execution."""
    from weakincentives.adapters.codex_app_server import (
        CodexAppServerAdapter,
        CodexAppServerClientConfig,
        CodexAppServerModelConfig,
    )

    adapter = CodexAppServerAdapter(
        model_config=CodexAppServerModelConfig(model=adapter_fixture.get_model()),
        client_config=CodexAppServerClientConfig(
            cwd=str(tmp_path),
            approval_policy="never",
            env={"ACK_TEST_ENV": "ack_env_value"},
        ),
    )

    prompt = Prompt(
        PromptTemplate(
            ns="integration/ack/workspace",
            key="env",
            name="ack_workspace_env",
            sections=[
                MarkdownSection(
                    title="Task",
                    key="task",
                    template=(
                        "Print environment variable ACK_TEST_ENV and reply with the value."
                    ),
                )
            ],
        )
    )

    response = adapter.evaluate(prompt, session=session)

    assert response.text is not None
    assert "ack_env_value" in response.text

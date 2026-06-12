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

"""Adapter-specific ACK scenarios for sandbox isolation behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptTemplate,
)
from weakincentives.runtime.session import Session
from weakincentives.sandbox import HostMount, WorkspaceConfig

from ..adapters import AdapterFixture

pytestmark = pytest.mark.ack_capability("workspace_isolation")


def test_sandbox_mounts_host_files(
    adapter_fixture: AdapterFixture,
    session: Session,
    tmp_path: Path,
) -> None:
    """Sandbox mounts expose host files to the adapter execution environment."""
    host_dir = tmp_path / "host"
    host_dir.mkdir()
    source_file = host_dir / "mounted_file.txt"
    source_file.write_text("mounted content from host")

    adapter = adapter_fixture.create_adapter(tmp_path)
    prompt = Prompt(
        PromptTemplate.create(
            ns="integration.ack.workspace",
            key="host-mount",
            name="ack_workspace_mount",
            sections=[
                MarkdownSection(
                    title="Task",
                    key="task",
                    template=(
                        "Read data/mounted_file.txt and reply with its exact content."
                    ),
                ),
            ],
            workspace=WorkspaceConfig(
                mounts=(HostMount(host_path=str(host_dir), mount_path="data"),),
                allowed_host_roots=(str(tmp_path),),
            ),
        )
    )

    response = adapter.evaluate(prompt, session=session)

    assert response.text is not None
    assert "mounted content from host" in response.text


def test_sandbox_root_is_cwd(
    adapter_fixture: AdapterFixture,
    session: Session,
    tmp_path: Path,
) -> None:
    """The opened sandbox root is used as effective cwd during execution."""
    adapter = adapter_fixture.create_adapter(tmp_path)
    prompt = Prompt(
        PromptTemplate.create(
            ns="integration.ack.workspace",
            key="cwd",
            name="ack_workspace_cwd",
            sections=[
                MarkdownSection(
                    title="Task",
                    key="task",
                    template=(
                        "Run `pwd` and reply with the current directory path only."
                    ),
                ),
            ],
            workspace=WorkspaceConfig(),
        )
    )

    response = adapter.evaluate(prompt, session=session)

    assert response.text is not None
    assert "wink-sandbox-" in response.text


@pytest.mark.ack_capability("custom_env_forwarding")
def test_custom_env_forwarded(
    adapter_fixture: AdapterFixture,
    session: Session,
    tmp_path: Path,
) -> None:
    """Adapter forwards configured environment variables to subprocess execution."""
    adapter = adapter_fixture.create_adapter_with_env(
        tmp_path,
        env={"ACK_TEST_ENV": "ack_env_value"},
    )

    prompt = Prompt(
        PromptTemplate.create(
            ns="integration.ack.workspace",
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

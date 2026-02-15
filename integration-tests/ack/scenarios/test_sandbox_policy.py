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

"""Adapter-specific ACK scenarios for sandbox policy enforcement."""

from __future__ import annotations

from pathlib import Path

import pytest

from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.runtime.session import Session

from ..adapters import AdapterFixture, FileSystemMode, NetworkMode, SandboxSpec

pytestmark = pytest.mark.ack_capability("sandbox_policy")


def test_sandbox_restricts_writes_outside_workspace(
    adapter_fixture: AdapterFixture,
    session: Session,
    tmp_path: Path,
) -> None:
    """Read-only sandbox blocks writes outside the working directory."""
    # Use a subdirectory as the workspace cwd so that "outside" is truly
    # outside the working directory (not a child of cwd).
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "blocked.txt"

    adapter = adapter_fixture.create_adapter_with_sandbox(
        workspace_dir,
        sandbox=SandboxSpec(
            filesystem=FileSystemMode.READ_ONLY,
            network=NetworkMode.BLOCKED,
        ),
    )

    prompt = Prompt(
        PromptTemplate(
            ns="integration/ack/sandbox",
            key="outside-write",
            name="ack_sandbox_outside",
            sections=[
                MarkdownSection(
                    title="Task",
                    key="task",
                    template=(
                        f"Try to write a file at {outside_file}. If blocked, reply 'blocked'."
                    ),
                )
            ],
        )
    )

    _ = adapter.evaluate(prompt, session=session)
    assert not outside_file.exists()


def test_sandbox_allows_writes_inside_workspace(
    adapter_fixture: AdapterFixture,
    session: Session,
    tmp_path: Path,
) -> None:
    """Workspace-write sandbox allows writes under cwd."""
    target = tmp_path / "sandbox_allowed.txt"

    adapter = adapter_fixture.create_adapter_with_sandbox(
        tmp_path,
        sandbox=SandboxSpec(
            filesystem=FileSystemMode.WORKSPACE_WRITE,
            network=NetworkMode.ENABLED,
        ),
    )

    prompt = Prompt(
        PromptTemplate(
            ns="integration/ack/sandbox",
            key="inside-write",
            name="ack_sandbox_inside",
            sections=[
                MarkdownSection(
                    title="Task",
                    key="task",
                    template=(
                        "Create a file named sandbox_allowed.txt in the current directory and "
                        "write 'ok' to it."
                    ),
                )
            ],
        )
    )

    _ = adapter.evaluate(prompt, session=session)
    assert target.exists()


def test_sandbox_allows_writes_with_network_disabled(
    adapter_fixture: AdapterFixture,
    session: Session,
    tmp_path: Path,
) -> None:
    """Workspace writes succeed even when network is blocked."""
    target = tmp_path / "offline_write.txt"

    adapter = adapter_fixture.create_adapter_with_sandbox(
        tmp_path,
        sandbox=SandboxSpec(
            filesystem=FileSystemMode.WORKSPACE_WRITE,
            network=NetworkMode.BLOCKED,
        ),
    )

    prompt = Prompt(
        PromptTemplate(
            ns="integration/ack/sandbox",
            key="offline-write",
            name="ack_sandbox_offline_write",
            sections=[
                MarkdownSection(
                    title="Task",
                    key="task",
                    template=(
                        "Create a file named offline_write.txt in the current directory and "
                        "write 'ok' to it."
                    ),
                )
            ],
        )
    )

    _ = adapter.evaluate(prompt, session=session)
    assert target.exists()

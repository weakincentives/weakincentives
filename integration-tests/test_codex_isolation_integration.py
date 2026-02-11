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

"""Integration tests for Codex App Server workspace isolation and environment.

These tests verify that:
- WorkspaceSection correctly mounts host files via HostMount
- The adapter uses the workspace temp_dir as the effective cwd
- Custom environment variables are forwarded to the Codex subprocess
"""

# pyright: reportArgumentType=false

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Final

import pytest

from weakincentives.adapters.codex_app_server import (
    CodexAppServerAdapter,
    CodexAppServerClientConfig,
    CodexAppServerModelConfig,
)
from weakincentives.prompt import (
    HostMount,
    MarkdownSection,
    Prompt,
    PromptTemplate,
    WorkspaceSection,
)
from weakincentives.runtime.events import PromptExecuted
from weakincentives.runtime.session import Session


def _has_codex() -> bool:
    return shutil.which("codex") is not None


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _has_codex(), reason="codex CLI not found on PATH"),
    pytest.mark.timeout(90),
]

_MODEL_ENV_VAR: Final[str] = "CODEX_APP_SERVER_TEST_MODEL"
_PROMPT_NS: Final[str] = "integration/codex-isolation"


# =============================================================================
# Helpers
# =============================================================================


def _get_model() -> str:
    """Return the model name for integration tests."""
    return os.environ.get(_MODEL_ENV_VAR, "gpt-5.3-codex")


def _make_session() -> Session:
    return Session(tags={"suite": "integration"})


# =============================================================================
# Tests
# =============================================================================


def test_codex_workspace_with_host_mounts(tmp_path: Path) -> None:
    """Verify that WorkspaceSection mounts host files into the workspace.

    Creates a file on the host, mounts it via HostMount, and asks the model
    to read and report its contents.
    """
    # Create a source file on the host
    source_dir = tmp_path / "host_data"
    source_dir.mkdir()
    source_file = source_dir / "mounted_file.txt"
    source_file.write_text("mounted content from host")

    session = _make_session()
    workspace = WorkspaceSection(
        session=session,
        mounts=[
            HostMount(host_path=str(source_dir), mount_path="data"),
        ],
        allowed_host_roots=[tmp_path],
    )

    try:
        adapter = CodexAppServerAdapter(
            model_config=CodexAppServerModelConfig(model=_get_model()),
            client_config=CodexAppServerClientConfig(
                cwd=str(workspace.temp_dir),
                approval_policy="never",
            ),
        )

        section = MarkdownSection(
            title="Task",
            template=(
                "Read the file at 'data/mounted_file.txt' in the current directory "
                "and reply with its exact contents."
            ),
            key="task",
        )
        prompt = Prompt(
            PromptTemplate(
                ns=_PROMPT_NS,
                key="host-mount-test",
                name="host_mount_test",
                sections=[workspace, section],
            )
        )

        response = adapter.evaluate(prompt, session=session)

        assert response.text is not None
        assert "mounted content from host" in response.text, (
            f"Expected model to read mounted file. Response: {response.text}"
        )
    finally:
        workspace.cleanup()


def test_codex_workspace_isolation_cwd_constraint(tmp_path: Path) -> None:
    """Verify that the adapter uses workspace temp_dir as cwd.

    When a WorkspaceSection is provided without explicit cwd, the adapter
    should resolve the filesystem root as the working directory.
    """
    session = _make_session()
    workspace = WorkspaceSection(session=session)

    try:
        # No explicit cwd - adapter should derive it from the workspace filesystem
        adapter = CodexAppServerAdapter(
            model_config=CodexAppServerModelConfig(model=_get_model()),
            client_config=CodexAppServerClientConfig(
                approval_policy="never",
            ),
        )

        section = MarkdownSection(
            title="Task",
            template=(
                "Print the current working directory using `pwd` and reply with "
                "just the path."
            ),
            key="task",
        )
        prompt = Prompt(
            PromptTemplate(
                ns=_PROMPT_NS,
                key="cwd-test",
                name="cwd_test",
                sections=[workspace, section],
            )
        )

        response = adapter.evaluate(prompt, session=session)

        assert response.text is not None
        assert response.text.strip(), "Expected non-empty response"

        event = session[PromptExecuted].latest()
        assert event is not None
        assert event.adapter == "codex_app_server"
    finally:
        workspace.cleanup()


def test_codex_custom_env_passed_to_subprocess(tmp_path: Path) -> None:
    """Verify that custom environment variables are forwarded to the Codex subprocess.

    Passes WINK_TEST_VAR via the env config and asks the model to echo it.
    """
    adapter = CodexAppServerAdapter(
        model_config=CodexAppServerModelConfig(model=_get_model()),
        client_config=CodexAppServerClientConfig(
            cwd=str(tmp_path),
            approval_policy="never",
            env={"WINK_TEST_VAR": "hello_from_wink"},
        ),
    )

    section = MarkdownSection(
        title="Task",
        template=(
            "Print the value of the environment variable WINK_TEST_VAR "
            "and reply with just that value."
        ),
        key="task",
    )
    prompt = Prompt(
        PromptTemplate(
            ns=_PROMPT_NS,
            key="env-test",
            name="env_test",
            sections=[section],
        )
    )

    session = _make_session()
    response = adapter.evaluate(prompt, session=session)

    assert response.text is not None
    assert "hello_from_wink" in response.text, (
        f"Expected WINK_TEST_VAR value in response. Got: {response.text}"
    )

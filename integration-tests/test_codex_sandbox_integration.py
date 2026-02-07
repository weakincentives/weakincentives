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

"""Integration tests for Codex App Server sandbox mode configuration.

These tests verify that the ``sandbox_mode`` option (``"read-only"``,
``"workspace-write"``, ``"danger-full-access"``) correctly controls the
model's ability to write files during execution.
"""

# pyright: reportArgumentType=false

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Final

import pytest

from weakincentives.adapters.codex_app_server import (
    CodexAppServerAdapter,
    CodexAppServerClientConfig,
    CodexAppServerModelConfig,
)
from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptTemplate,
)
from weakincentives.runtime.events import PromptExecuted
from weakincentives.runtime.session import Session


def _has_codex() -> bool:
    return shutil.which("codex") is not None


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _has_codex(), reason="codex CLI not found on PATH"),
    pytest.mark.timeout(120),
]

_MODEL_ENV_VAR: Final[str] = "CODEX_APP_SERVER_TEST_MODEL"
_PROMPT_NS: Final[str] = "integration/codex-sandbox"


# =============================================================================
# Helpers
# =============================================================================


def _get_model() -> str:
    """Return the model name for integration tests."""
    return os.environ.get(_MODEL_ENV_VAR, "gpt-5.3-codex")


def _make_adapter(
    cwd: str | Path,
    *,
    sandbox_mode: str = "workspace-write",
) -> CodexAppServerAdapter:
    """Create an adapter configured for sandbox testing."""
    return CodexAppServerAdapter(
        model_config=CodexAppServerModelConfig(model=_get_model()),
        client_config=CodexAppServerClientConfig(
            cwd=str(cwd),
            approval_policy="never",
            sandbox_mode=sandbox_mode,
        ),
    )


def _make_session() -> Session:
    return Session(tags={"suite": "integration"})


# =============================================================================
# Tests
# =============================================================================


def test_codex_sandbox_workspace_write_allows_writes_in_cwd(
    tmp_path: Path,
) -> None:
    """Verify that sandbox_mode='workspace-write' allows file writes in cwd."""
    adapter = _make_adapter(tmp_path, sandbox_mode="workspace-write")

    section = MarkdownSection(
        title="Task",
        template=(
            "Create a file called 'sandbox_test.txt' in the current directory "
            "with the content 'sandbox write test'. Use only shell commands. "
            "Reply with 'done' when finished."
        ),
        key="task",
    )
    prompt = Prompt(
        PromptTemplate(
            ns=_PROMPT_NS,
            key="workspace-write-test",
            name="workspace_write_test",
            sections=[section],
        )
    )

    session = _make_session()
    response = adapter.evaluate(prompt, session=session)

    assert response.text is not None
    event = session[PromptExecuted].latest()
    assert event is not None
    assert event.adapter == "codex_app_server"

    # The model should have created the file in workspace
    target = tmp_path / "sandbox_test.txt"
    assert target.exists(), (
        f"Expected sandbox_test.txt to exist in {tmp_path}. "
        f"Files present: {list(tmp_path.iterdir())}"
    )


def test_codex_sandbox_read_only_blocks_writes(tmp_path: Path) -> None:
    """Verify that sandbox_mode='read-only' prevents the model from writing files."""
    # Seed a file so the model has something to read
    (tmp_path / "existing.txt").write_text("read only content")

    adapter = _make_adapter(tmp_path, sandbox_mode="read-only")

    section = MarkdownSection(
        title="Task",
        template=(
            "Try to create a file called 'blocked.txt' in the current directory "
            "with the content 'should not exist'. If the write fails or is blocked, "
            "just reply 'write blocked'."
        ),
        key="task",
    )
    prompt = Prompt(
        PromptTemplate(
            ns=_PROMPT_NS,
            key="read-only-test",
            name="read_only_test",
            sections=[section],
        )
    )

    session = _make_session()
    response = adapter.evaluate(prompt, session=session)

    assert response.text is not None
    # The file should NOT have been created
    blocked = tmp_path / "blocked.txt"
    assert not blocked.exists(), (
        f"Expected blocked.txt to NOT exist under read-only sandbox. "
        f"Files present: {list(tmp_path.iterdir())}"
    )


def test_codex_sandbox_danger_full_access_allows_writes_outside_cwd(
    tmp_path: Path,
) -> None:
    """Verify that sandbox_mode='danger-full-access' allows writes outside cwd."""
    # Create a separate directory outside cwd to write to
    outside_dir = Path(tempfile.mkdtemp(prefix="wink-sandbox-outside-"))
    try:
        adapter = _make_adapter(tmp_path, sandbox_mode="danger-full-access")

        section = MarkdownSection(
            title="Task",
            template=(
                f"Create a file at '{outside_dir}/full_access.txt' with the "
                "content 'full access write'. Use only shell commands. "
                "Reply with 'done' when finished."
            ),
            key="task",
        )
        prompt = Prompt(
            PromptTemplate(
                ns=_PROMPT_NS,
                key="full-access-test",
                name="full_access_test",
                sections=[section],
            )
        )

        session = _make_session()
        response = adapter.evaluate(prompt, session=session)

        assert response.text is not None
        target = outside_dir / "full_access.txt"
        assert target.exists(), (
            f"Expected full_access.txt to exist at {outside_dir}. "
            f"Files present: {list(outside_dir.iterdir())}"
        )
    finally:
        shutil.rmtree(outside_dir, ignore_errors=True)

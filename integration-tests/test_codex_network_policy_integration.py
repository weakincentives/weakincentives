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

"""Integration tests for Codex App Server network access under sandbox modes.

Codex does not have an explicit ``NetworkPolicy`` knob — network access is
controlled implicitly by ``sandbox_mode``:

- ``"read-only"`` restricts network access
- ``"workspace-write"`` allows network access

These tests verify that the sandbox mode correctly gates network operations.
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
_PROMPT_NS: Final[str] = "integration/codex-network-policy"


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
    """Create an adapter configured for network policy testing."""
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


def test_codex_read_only_sandbox_restricts_network(tmp_path: Path) -> None:
    """Verify that sandbox_mode='read-only' restricts network access.

    The model attempts to fetch a URL. Under read-only sandbox, the request
    should fail or the model should report inability to access the network.
    """
    adapter = _make_adapter(tmp_path, sandbox_mode="read-only")

    section = MarkdownSection(
        title="Task",
        template=(
            "Try to fetch the URL https://httpbin.org/get using curl and "
            "report the HTTP status code. If the network request fails or "
            "is blocked, reply with 'network blocked'."
        ),
        key="task",
    )
    prompt = Prompt(
        PromptTemplate(
            ns=_PROMPT_NS,
            key="read-only-network-test",
            name="read_only_network_test",
            sections=[section],
        )
    )

    session = _make_session()
    response = adapter.evaluate(prompt, session=session)

    assert response.text is not None
    text_lower = response.text.lower()

    event = session[PromptExecuted].latest()
    assert event is not None
    assert event.adapter == "codex_app_server"

    # Under read-only sandbox, network should be blocked.
    # The model should report failure, blocked, error, denied, or similar.
    failure_indicators = (
        "blocked",
        "denied",
        "fail",
        "error",
        "unable",
        "cannot",
        "refused",
        "not allowed",
        "permission",
        "restricted",
    )
    assert any(indicator in text_lower for indicator in failure_indicators), (
        f"Expected network to be blocked under read-only sandbox. "
        f"Response: {response.text}"
    )


def test_codex_workspace_write_sandbox_allows_network(tmp_path: Path) -> None:
    """Verify that sandbox_mode='workspace-write' allows network access.

    The model performs a network request; under workspace-write sandbox,
    the request should succeed.
    """
    adapter = _make_adapter(tmp_path, sandbox_mode="workspace-write")

    section = MarkdownSection(
        title="Task",
        template=(
            "Fetch the URL https://httpbin.org/get using curl and "
            "report the HTTP status code in your reply. Include the "
            "status code number (e.g. 200)."
        ),
        key="task",
    )
    prompt = Prompt(
        PromptTemplate(
            ns=_PROMPT_NS,
            key="workspace-write-network-test",
            name="workspace_write_network_test",
            sections=[section],
        )
    )

    session = _make_session()
    response = adapter.evaluate(prompt, session=session)

    assert response.text is not None

    event = session[PromptExecuted].latest()
    assert event is not None
    assert event.adapter == "codex_app_server"

    # Under workspace-write, network should work. The response should
    # contain a success indicator (status 200 or the fetched content).
    text_lower = response.text.lower()
    success_indicators = ("200", "ok", "success", "httpbin", "origin", "headers")
    assert any(indicator in text_lower for indicator in success_indicators), (
        f"Expected network request to succeed under workspace-write sandbox. "
        f"Response: {response.text}"
    )

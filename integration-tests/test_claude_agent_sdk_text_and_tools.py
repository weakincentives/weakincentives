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

"""Claude Agent SDK integration scenarios for text and tool usage."""

from __future__ import annotations

from pathlib import Path

import pytest
from claude_agent_sdk_fixtures import (
    GreetingParams,
    TransformRequest,
    _assert_prompt_usage,
    _build_greeting_prompt,
    _build_tool_prompt,
    _build_uppercase_tool,
    _make_adapter,
    _make_session_with_usage_tracking,
    pytestmark as claude_agent_sdk_pytestmark,
)

from weakincentives.prompt import Prompt

pytest.importorskip("claude_agent_sdk")

pytestmark = claude_agent_sdk_pytestmark


def test_claude_agent_sdk_adapter_returns_text(tmp_path: Path) -> None:
    """Test that the adapter returns text from a simple prompt."""
    adapter = _make_adapter(tmp_path)
    prompt_template = _build_greeting_prompt()
    params = GreetingParams(audience="integration tests")
    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(prompt, session=session)

    assert response.prompt_name == "greeting"
    assert response.text is not None
    assert response.text.strip()
    _assert_prompt_usage(session)


def test_claude_agent_sdk_adapter_processes_tool_invocation(tmp_path: Path) -> None:
    """Test that the adapter processes custom tool invocations via MCP bridge.

    This test validates that weakincentives tools are correctly bridged to the
    SDK via an in-process MCP server. The streaming mode approach enables proper
    MCP server initialization.
    """
    tool = _build_uppercase_tool()
    prompt_template = _build_tool_prompt(tool)
    params = TransformRequest(text="integration tests")

    adapter = _make_adapter(tmp_path)

    prompt = Prompt(prompt_template).bind(params)

    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(prompt, session=session)

    assert response.prompt_name == "uppercase_workflow"
    assert response.text is not None and response.text.strip()
    # The uppercase text should appear in the response
    assert params.text.upper() in response.text
    _assert_prompt_usage(session)

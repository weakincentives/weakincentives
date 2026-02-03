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

"""Claude Agent SDK integration scenarios for structured output."""

from __future__ import annotations

from pathlib import Path

import pytest
from claude_agent_sdk_fixtures import (
    ReviewAnalysis,
    ReviewParams,
    _assert_prompt_usage,
    _build_structured_prompt,
    _make_adapter,
    _make_session_with_usage_tracking,
    pytestmark as claude_agent_sdk_pytestmark,
)

from weakincentives.prompt import Prompt

pytest.importorskip("claude_agent_sdk")

pytestmark = claude_agent_sdk_pytestmark


def test_claude_agent_sdk_adapter_parses_structured_output(tmp_path: Path) -> None:
    """Test that the adapter parses structured output correctly."""
    adapter = _make_adapter(tmp_path)
    prompt_template = _build_structured_prompt()
    sample = ReviewParams(
        text=(
            "The new release shipped important bug fixes and improved the onboarding flow."
            " Early adopters report smoother setup, though some mention learning curves."
        ),
    )

    prompt = Prompt(prompt_template).bind(sample)

    session = _make_session_with_usage_tracking()
    response = adapter.evaluate(prompt, session=session)

    assert response.prompt_name == "structured_review"
    assert response.output is not None
    assert isinstance(response.output, ReviewAnalysis)
    assert response.output.summary
    assert response.output.sentiment
    _assert_prompt_usage(session)

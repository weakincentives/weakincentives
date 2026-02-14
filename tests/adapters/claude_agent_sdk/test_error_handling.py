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

"""Tests for SDK error handling and visibility expansion."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.adapters.claude_agent_sdk.conftest import (
    MockResultMessage,
    MockSDKQuery,
    SimpleOutput,
    sdk_patches,
    setup_mock_query,
)
from weakincentives.adapters.claude_agent_sdk import ClaudeAgentSDKAdapter
from weakincentives.adapters.core import PromptEvaluationError
from weakincentives.prompt import Prompt, SectionVisibility
from weakincentives.prompt.errors import VisibilityExpansionRequired
from weakincentives.runtime.session import Session


class TestSDKErrorHandling:
    def test_normalizes_sdk_error(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        MockSDKQuery.reset()
        MockSDKQuery.set_error(RuntimeError("SDK crashed"))

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            with pytest.raises(PromptEvaluationError, match="SDK crashed"):
                adapter.evaluate(simple_prompt, session=session)


class TestVisibilityExpansionRequired:
    """Tests for VisibilityExpansionRequired exception propagation."""

    def test_propagates_visibility_expansion_required(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """Test that VisibilityExpansionRequired propagates through the adapter."""
        MockSDKQuery.reset()
        MockSDKQuery.set_error(
            VisibilityExpansionRequired(
                "Model requested expansion",
                requested_overrides={("section", "key"): SectionVisibility.FULL},
                reason="Need more details",
                section_keys=("section.key",),
            )
        )

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            with pytest.raises(VisibilityExpansionRequired) as exc_info:
                adapter.evaluate(simple_prompt, session=session)

        exc = exc_info.value
        assert isinstance(exc, VisibilityExpansionRequired)
        assert exc.requested_overrides == {("section", "key"): SectionVisibility.FULL}
        assert exc.section_keys == ("section.key",)
        assert exc.reason == "Need more details"

    def test_propagates_visibility_expansion_from_signal(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """Test that VisibilityExpansionRequired from signal propagates correctly."""
        from weakincentives.adapters.claude_agent_sdk._visibility_signal import (
            VisibilityExpansionSignal,
        )

        setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        test_exc = VisibilityExpansionRequired(
            "Signal-based expansion",
            requested_overrides={("signal", "test"): SectionVisibility.FULL},
            reason="From signal",
            section_keys=("signal.test",),
        )

        original_init = VisibilityExpansionSignal.__init__

        def patched_init(self: VisibilityExpansionSignal) -> None:
            original_init(self)
            self.set(test_exc)

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            with patch.object(VisibilityExpansionSignal, "__init__", patched_init):
                with pytest.raises(VisibilityExpansionRequired) as exc_info:
                    adapter.evaluate(simple_prompt, session=session)

        exc = exc_info.value
        assert exc is test_exc
        assert exc.section_keys == ("signal.test",)
        assert exc.reason == "From signal"

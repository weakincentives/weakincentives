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

"""Tests for budget tracking in the Claude Agent SDK adapter."""

from __future__ import annotations

from tests.adapters.claude_agent_sdk.conftest import (
    MockResultMessage,
    SimpleOutput,
    sdk_patches,
    setup_mock_query,
)
from weakincentives.adapters.claude_agent_sdk import ClaudeAgentSDKAdapter
from weakincentives.prompt import Prompt
from weakincentives.runtime.session import Session


class TestBudgetTracking:
    def test_records_usage_to_budget_tracker(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        from weakincentives.budget import Budget, BudgetTracker

        setup_mock_query(
            [
                MockResultMessage(
                    result="Done",
                    usage={"input_tokens": 100, "output_tokens": 50},
                    structured_output=None,
                )
            ]
        )

        budget = Budget(max_total_tokens=1000)
        budget_tracker = BudgetTracker(budget)

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            adapter.evaluate(
                simple_prompt, session=session, budget_tracker=budget_tracker
            )

        assert budget_tracker.consumed.input_tokens == 100
        assert budget_tracker.consumed.output_tokens == 50

    def test_creates_budget_tracker_from_budget(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """Test that passing budget without budget_tracker creates one."""
        from weakincentives.budget import Budget

        setup_mock_query(
            [
                MockResultMessage(
                    result="Done",
                    usage={"input_tokens": 100, "output_tokens": 50},
                    structured_output=None,
                )
            ]
        )

        budget = Budget(max_total_tokens=1000)

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            response = adapter.evaluate(simple_prompt, session=session, budget=budget)

        assert response.text == "Done"

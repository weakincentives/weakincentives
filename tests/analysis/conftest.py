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

"""Shared test infrastructure for analysis tests."""

from __future__ import annotations

from weakincentives.adapters.core import PromptResponse, ProviderAdapter
from weakincentives.analysis import AnalysisBundle
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.deadlines import Deadline
from weakincentives.prompt import Prompt
from weakincentives.runtime.run_context import RunContext
from weakincentives.runtime.session.protocols import SessionProtocol


class MockAnalysisAdapter(ProviderAdapter[AnalysisBundle]):
    """Mock adapter that returns a fixed AnalysisBundle."""

    def __init__(
        self,
        *,
        response: PromptResponse[AnalysisBundle] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._response = response or PromptResponse(
            prompt_name="analysis-agent",
            text="# Analysis Report\n\n**Objective:** test\n",
            output=AnalysisBundle(report="# Analysis Report\n\n**Objective:** test\n"),
        )
        self._error = error
        self._call_count = 0

    def evaluate(
        self,
        prompt: Prompt[AnalysisBundle],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
        heartbeat: object = None,
        run_context: RunContext | None = None,
    ) -> PromptResponse[AnalysisBundle]:
        del budget, heartbeat, deadline, budget_tracker, session, run_context
        self._call_count += 1

        with prompt.resources:
            if self._error is not None:
                raise self._error
            return self._response

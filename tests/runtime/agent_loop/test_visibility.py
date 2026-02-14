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

"""Tests for visibility override accumulation and budget tracker behavior."""

from __future__ import annotations

from collections.abc import Mapping

from weakincentives.budget import Budget
from weakincentives.prompt import (
    SectionPath,
    SectionVisibility,
)
from weakincentives.runtime.agent_loop import (
    AgentLoopConfig,
    AgentLoopRequest,
    AgentLoopResult,
)
from weakincentives.runtime.mailbox import InMemoryMailbox
from weakincentives.runtime.session import VisibilityOverrides

from .conftest import (
    MockAdapter,
    SampleLoop,
    SampleOutput,
    SampleRequest,
)


def test_visibility_overrides_accumulate_in_session() -> None:
    """AgentLoop accumulates visibility overrides in session state."""
    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        visibility_requests: list[Mapping[SectionPath, SectionVisibility]] = [
            {("section1",): SectionVisibility.FULL},
            {("section2",): SectionVisibility.FULL},
        ]
        adapter = MockAdapter(visibility_requests=visibility_requests)
        loop = SampleLoop(adapter=adapter, requests=requests)

        request = AgentLoopRequest(request=SampleRequest(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Check session state has accumulated overrides
        assert loop.session_created is not None
        overrides = loop.session_created[VisibilityOverrides].latest()
        assert overrides is not None
        assert overrides.get(("section1",)) == SectionVisibility.FULL
        assert overrides.get(("section2",)) == SectionVisibility.FULL
    finally:
        requests.close()
        results.close()


def test_same_budget_tracker_used_across_visibility_retries() -> None:
    """Same BudgetTracker is used across visibility expansion retries."""
    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        budget = Budget(max_total_tokens=1000)
        config = AgentLoopConfig(budget=budget)
        visibility_requests: list[Mapping[SectionPath, SectionVisibility]] = [
            {("section1",): SectionVisibility.FULL},
            {("section2",): SectionVisibility.FULL},
        ]
        adapter = MockAdapter(visibility_requests=visibility_requests)
        loop = SampleLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(request=SampleRequest(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Called 3 times: 2 visibility expansions + 1 success
        assert adapter._call_count == 3
        # Same BudgetTracker should be used for all calls
        assert len(adapter._budget_trackers) == 3
        assert all(t is adapter._budget_trackers[0] for t in adapter._budget_trackers)
        # And it should have the correct budget
        assert adapter._budget_trackers[0] is not None
        assert adapter._budget_trackers[0].budget is budget
    finally:
        requests.close()
        results.close()


def test_no_budget_tracker_when_no_budget() -> None:
    """No BudgetTracker is created when no budget is set."""
    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        adapter = MockAdapter()
        loop = SampleLoop(adapter=adapter, requests=requests)

        request = AgentLoopRequest(request=SampleRequest(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        assert adapter._last_budget_tracker is None
    finally:
        requests.close()
        results.close()

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

"""Tests for AnalysisLoop."""

from __future__ import annotations

from pathlib import Path

from weakincentives.analysis import (
    AnalysisBundle,
    AnalysisLoop,
    AnalysisLoopConfig,
    AnalysisPromptOverrides,
    AnalysisRequest,
)
from weakincentives.runtime.agent_loop import AgentLoopRequest, AgentLoopResult
from weakincentives.runtime.mailbox import InMemoryMailbox

from .conftest import MockAnalysisAdapter


class TestAnalysisLoop:
    """Tests for AnalysisLoop core behavior."""

    def test_execute_produces_analysis_bundle(self) -> None:
        """Execute returns an AnalysisBundle with a report."""
        adapter = MockAnalysisAdapter()
        requests: InMemoryMailbox[
            AgentLoopRequest[AnalysisRequest], AgentLoopResult[AnalysisBundle]
        ] = InMemoryMailbox(name="analysis-requests")

        loop = AnalysisLoop(adapter=adapter, requests=requests)

        request = AnalysisRequest(
            objective="Why did this fail?",
            bundles=(Path("/tmp/bundle-001.zip"),),
        )

        response, _session = loop.execute(request)
        assert response.output is not None
        assert "Analysis Report" in response.output.report

    def test_session_tagged_as_analysis(self) -> None:
        """Session is tagged with loop=analysis."""
        adapter = MockAnalysisAdapter()
        requests: InMemoryMailbox[
            AgentLoopRequest[AnalysisRequest], AgentLoopResult[AnalysisBundle]
        ] = InMemoryMailbox(name="analysis-requests")

        loop = AnalysisLoop(adapter=adapter, requests=requests)

        request = AnalysisRequest(
            objective="test objective",
            bundles=(Path("/tmp/bundle.zip"),),
        )

        _, session = loop.execute(request)
        assert session.tags["loop"] == "analysis"
        assert session.tags["objective"] == "test objective"

    def test_custom_config(self) -> None:
        """AnalysisLoop accepts custom AnalysisLoopConfig."""
        adapter = MockAnalysisAdapter()
        requests: InMemoryMailbox[
            AgentLoopRequest[AnalysisRequest], AgentLoopResult[AnalysisBundle]
        ] = InMemoryMailbox(name="analysis-requests")

        config = AnalysisLoopConfig(
            output_dir=Path("/tmp/custom-output"),
            overrides=AnalysisPromptOverrides(
                methodology="Focus on security patterns.",
            ),
        )

        loop = AnalysisLoop(adapter=adapter, requests=requests, config=config)

        request = AnalysisRequest(
            objective="Security analysis",
            bundles=(Path("/tmp/bundle.zip"),),
        )

        response, _ = loop.execute(request)
        assert response.output is not None
        assert adapter._call_count == 1

    def test_mailbox_driven_processing(self) -> None:
        """AnalysisLoop processes requests from its mailbox."""
        adapter = MockAnalysisAdapter()
        requests: InMemoryMailbox[
            AgentLoopRequest[AnalysisRequest], AgentLoopResult[AnalysisBundle]
        ] = InMemoryMailbox(name="analysis-requests")
        responses: InMemoryMailbox[AgentLoopResult[AnalysisBundle], None] = (
            InMemoryMailbox(name="analysis-responses")
        )

        loop = AnalysisLoop(adapter=adapter, requests=requests, worker_id="test")

        analysis_request = AnalysisRequest(
            objective="batch analysis",
            bundles=(Path("/tmp/b1.zip"), Path("/tmp/b2.zip")),
        )

        requests.send(
            AgentLoopRequest(request=analysis_request),
            reply_to=responses,
        )

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Check response was sent
        msgs = responses.receive(max_messages=10, wait_time_seconds=0)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.success is True
        assert result.output is not None
        assert "Analysis Report" in result.output.report

    def test_error_handling(self) -> None:
        """Adapter errors are propagated correctly."""
        adapter = MockAnalysisAdapter(error=RuntimeError("LLM unavailable"))
        requests: InMemoryMailbox[
            AgentLoopRequest[AnalysisRequest], AgentLoopResult[AnalysisBundle]
        ] = InMemoryMailbox(name="analysis-requests")
        responses: InMemoryMailbox[AgentLoopResult[AnalysisBundle], None] = (
            InMemoryMailbox(name="analysis-responses")
        )

        loop = AnalysisLoop(adapter=adapter, requests=requests, worker_id="test")

        analysis_request = AnalysisRequest(
            objective="test",
            bundles=(Path("/tmp/bundle.zip"),),
        )

        requests.send(
            AgentLoopRequest(request=analysis_request),
            reply_to=responses,
        )

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = responses.receive(max_messages=10, wait_time_seconds=0)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.success is False
        assert result.error is not None
        assert "LLM unavailable" in result.error

    def test_default_config(self) -> None:
        """AnalysisLoop uses default config when none provided."""
        adapter = MockAnalysisAdapter()
        requests: InMemoryMailbox[
            AgentLoopRequest[AnalysisRequest], AgentLoopResult[AnalysisBundle]
        ] = InMemoryMailbox(name="analysis-requests")

        loop = AnalysisLoop(adapter=adapter, requests=requests)
        assert loop._analysis_config.output_dir == Path("./analysis-bundles/")
        assert loop._analysis_config.overrides is None

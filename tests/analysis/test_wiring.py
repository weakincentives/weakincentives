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

"""Tests for the connect_analysis wiring helper."""

from __future__ import annotations

from pathlib import Path

from weakincentives.analysis import (
    AnalysisBudget,
    AnalysisForwarder,
    AnalysisLoop,
    AnalysisLoopConfig,
    AnalysisPromptOverrides,
    CompletionNotification,
    connect_analysis,
)
from weakincentives.runtime.mailbox import InMemoryMailbox

from .conftest import MockAnalysisAdapter


class TestConnectAnalysis:
    """Tests for connect_analysis helper."""

    def test_returns_forwarder_and_loop(self) -> None:
        """Returns a tuple of (AnalysisForwarder, AnalysisLoop)."""
        notifications: InMemoryMailbox[CompletionNotification, None] = InMemoryMailbox(
            name="notifications"
        )
        adapter = MockAnalysisAdapter()

        forwarder, analysis = connect_analysis(
            notifications=notifications,
            adapter=adapter,
            objective="test objective",
        )

        assert isinstance(forwarder, AnalysisForwarder)
        assert isinstance(analysis, AnalysisLoop)

    def test_custom_sample_rate(self) -> None:
        """Sample rate is passed to forwarder config."""
        notifications: InMemoryMailbox[CompletionNotification, None] = InMemoryMailbox(
            name="notifications"
        )
        adapter = MockAnalysisAdapter()

        forwarder, _ = connect_analysis(
            notifications=notifications,
            adapter=adapter,
            objective="test",
            sample_rate=0.5,
        )

        assert forwarder._config.sample_rate == 0.5

    def test_custom_budget(self) -> None:
        """Custom budget is passed to forwarder config."""
        notifications: InMemoryMailbox[CompletionNotification, None] = InMemoryMailbox(
            name="notifications"
        )
        adapter = MockAnalysisAdapter()

        budget = AnalysisBudget(max_requests=10)
        forwarder, _ = connect_analysis(
            notifications=notifications,
            adapter=adapter,
            objective="test",
            budget=budget,
        )

        assert forwarder._config.budget.max_requests == 10

    def test_custom_analysis_config(self) -> None:
        """Analysis config is passed to the AnalysisLoop."""
        notifications: InMemoryMailbox[CompletionNotification, None] = InMemoryMailbox(
            name="notifications"
        )
        adapter = MockAnalysisAdapter()

        analysis_config = AnalysisLoopConfig(
            output_dir=Path("/tmp/custom"),
            overrides=AnalysisPromptOverrides(methodology="custom"),
        )

        _, analysis = connect_analysis(
            notifications=notifications,
            adapter=adapter,
            objective="test",
            analysis_config=analysis_config,
        )

        assert analysis._analysis_config.output_dir == Path("/tmp/custom")

    def test_always_forward_failures_default(self) -> None:
        """always_forward_failures defaults to True."""
        notifications: InMemoryMailbox[CompletionNotification, None] = InMemoryMailbox(
            name="notifications"
        )
        adapter = MockAnalysisAdapter()

        forwarder, _ = connect_analysis(
            notifications=notifications,
            adapter=adapter,
            objective="test",
        )

        assert forwarder._config.always_forward_failures is True

    def test_always_forward_failures_override(self) -> None:
        """always_forward_failures can be set to False."""
        notifications: InMemoryMailbox[CompletionNotification, None] = InMemoryMailbox(
            name="notifications"
        )
        adapter = MockAnalysisAdapter()

        forwarder, _ = connect_analysis(
            notifications=notifications,
            adapter=adapter,
            objective="test",
            always_forward_failures=False,
        )

        assert forwarder._config.always_forward_failures is False

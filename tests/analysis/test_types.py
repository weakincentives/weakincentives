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

"""Tests for analysis data types."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

from weakincentives.analysis import (
    AnalysisBudget,
    AnalysisBundle,
    AnalysisForwarderConfig,
    AnalysisLoopConfig,
    AnalysisPromptOverrides,
    AnalysisRequest,
    CompletionNotification,
    EvalContext,
)


class TestCompletionNotification:
    """Tests for CompletionNotification."""

    def test_agent_loop_notification(self) -> None:
        """Agent loop notification has correct defaults."""
        n = CompletionNotification(
            source="agent_loop",
            bundle_path=Path("/tmp/bundle.zip"),
            request_id=uuid4(),
            success=True,
        )
        assert n.source == "agent_loop"
        assert n.success is True
        assert n.passed is None
        assert n.score is None
        assert isinstance(n.completed_at, datetime)

    def test_eval_loop_notification(self) -> None:
        """Eval loop notification includes score and passed."""
        n = CompletionNotification(
            source="eval_loop",
            bundle_path=Path("/tmp/bundle.zip"),
            request_id=uuid4(),
            success=True,
            passed=True,
            score=0.95,
        )
        assert n.source == "eval_loop"
        assert n.passed is True
        assert n.score == 0.95

    def test_failed_notification(self) -> None:
        """Failed notification has success=False."""
        n = CompletionNotification(
            source="agent_loop",
            bundle_path=Path("/tmp/bundle.zip"),
            request_id=uuid4(),
            success=False,
        )
        assert n.success is False

    def test_frozen(self) -> None:
        """CompletionNotification is immutable."""
        n = CompletionNotification(
            source="agent_loop",
            bundle_path=Path("/tmp/bundle.zip"),
            request_id=uuid4(),
            success=True,
        )
        try:
            n.success = False  # type: ignore[misc]
            raise AssertionError("Expected FrozenInstanceError")  # pragma: no cover
        except AttributeError:
            pass


class TestEvalContext:
    """Tests for EvalContext."""

    def test_defaults(self) -> None:
        """EvalContext has None defaults."""
        ctx = EvalContext()
        assert ctx.experiment_name is None
        assert ctx.pass_rate is None
        assert ctx.sample_count is None

    def test_with_values(self) -> None:
        """EvalContext stores provided values."""
        ctx = EvalContext(
            experiment_name="baseline",
            pass_rate=0.85,
            sample_count=100,
        )
        assert ctx.experiment_name == "baseline"
        assert ctx.pass_rate == 0.85
        assert ctx.sample_count == 100


class TestAnalysisRequest:
    """Tests for AnalysisRequest."""

    def test_minimal(self) -> None:
        """Minimal request with just objective and bundles."""
        r = AnalysisRequest(
            objective="Why are tests failing?",
            bundles=(Path("/tmp/bundle-1.zip"),),
        )
        assert r.objective == "Why are tests failing?"
        assert len(r.bundles) == 1
        assert r.source == "manual"
        assert r.eval_context is None
        assert isinstance(r.request_id, UUID)

    def test_with_eval_context(self) -> None:
        """Request with eval context metadata."""
        ctx = EvalContext(experiment_name="v2", pass_rate=0.7, sample_count=50)
        r = AnalysisRequest(
            objective="Analyze failures",
            bundles=(Path("/tmp/b1.zip"), Path("/tmp/b2.zip")),
            source="eval_loop",
            eval_context=ctx,
        )
        assert r.source == "eval_loop"
        assert r.eval_context is not None
        assert r.eval_context.experiment_name == "v2"

    def test_multiple_bundles(self) -> None:
        """Request can reference multiple bundles."""
        bundles = tuple(Path(f"/tmp/bundle-{i}.zip") for i in range(5))
        r = AnalysisRequest(objective="batch analysis", bundles=bundles)
        assert len(r.bundles) == 5


class TestAnalysisBundle:
    """Tests for AnalysisBundle."""

    def test_report_field(self) -> None:
        """AnalysisBundle stores report text."""
        b = AnalysisBundle(report="# Analysis\n\nFindings here.")
        assert "Analysis" in b.report

    def test_frozen(self) -> None:
        """AnalysisBundle is immutable."""
        b = AnalysisBundle(report="test")
        try:
            b.report = "modified"  # type: ignore[misc]
            raise AssertionError("Expected FrozenInstanceError")  # pragma: no cover
        except AttributeError:
            pass


class TestAnalysisBudget:
    """Tests for AnalysisBudget."""

    def test_defaults(self) -> None:
        """Budget has sensible defaults."""
        b = AnalysisBudget()
        assert b.max_requests == 100
        assert b.reset_interval == timedelta(hours=1)

    def test_custom_values(self) -> None:
        """Budget accepts custom values."""
        b = AnalysisBudget(max_requests=50, reset_interval=timedelta(minutes=30))
        assert b.max_requests == 50
        assert b.reset_interval == timedelta(minutes=30)


class TestAnalysisForwarderConfig:
    """Tests for AnalysisForwarderConfig."""

    def test_defaults(self) -> None:
        """Config has sensible defaults."""
        c = AnalysisForwarderConfig(objective="analyze")
        assert c.objective == "analyze"
        assert c.sample_rate == 0.1
        assert c.always_forward_failures is True
        assert isinstance(c.budget, AnalysisBudget)

    def test_custom_budget(self) -> None:
        """Config accepts custom budget."""
        budget = AnalysisBudget(max_requests=10)
        c = AnalysisForwarderConfig(objective="test", budget=budget)
        assert c.budget.max_requests == 10


class TestAnalysisPromptOverrides:
    """Tests for AnalysisPromptOverrides."""

    def test_defaults(self) -> None:
        """All overrides default to None."""
        o = AnalysisPromptOverrides()
        assert o.methodology is None
        assert o.output_format is None
        assert o.evidence_gathering is None

    def test_partial_override(self) -> None:
        """Can override just one section."""
        o = AnalysisPromptOverrides(methodology="Focus on security.")
        assert o.methodology == "Focus on security."
        assert o.output_format is None


class TestAnalysisLoopConfig:
    """Tests for AnalysisLoopConfig."""

    def test_defaults(self) -> None:
        """Config has sensible defaults."""
        c = AnalysisLoopConfig()
        assert c.output_dir == Path("./analysis-bundles/")
        assert c.include_source_bundles is True
        assert c.max_source_bundle_size == 50_000_000
        assert c.overrides is None

    def test_custom_output_dir(self) -> None:
        """Config accepts custom output directory."""
        c = AnalysisLoopConfig(output_dir=Path("/tmp/analysis"))
        assert c.output_dir == Path("/tmp/analysis")

    def test_with_overrides(self) -> None:
        """Config accepts prompt overrides."""
        overrides = AnalysisPromptOverrides(methodology="custom")
        c = AnalysisLoopConfig(overrides=overrides)
        assert c.overrides is not None
        assert c.overrides.methodology == "custom"

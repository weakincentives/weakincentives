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

"""Tests for Observation and Feedback data types including rendering."""

from __future__ import annotations

import pytest

from weakincentives.prompt import (
    Feedback,
    Observation,
)

# =============================================================================
# Observation Tests
# =============================================================================


class TestObservation:
    """Tests for the Observation data type."""

    def test_creates_basic_observation(self) -> None:
        obs = Observation(category="Pattern", description="Repeated failures")

        assert obs.category == "Pattern"
        assert obs.description == "Repeated failures"
        assert obs.evidence is None

    def test_creates_observation_with_evidence(self) -> None:
        obs = Observation(
            category="Loop",
            description="Same file read 3 times",
            evidence="file.txt",
        )

        assert obs.evidence == "file.txt"

    def test_observation_is_frozen(self) -> None:
        obs = Observation(category="Test", description="Test")

        with pytest.raises(AttributeError):
            obs.category = "Changed"  # type: ignore[misc]


# =============================================================================
# Feedback Tests
# =============================================================================


class TestFeedback:
    """Tests for the Feedback data type."""

    def test_creates_basic_feedback(self) -> None:
        feedback = Feedback(provider_name="Test", summary="All good")

        assert feedback.provider_name == "Test"
        assert feedback.summary == "All good"
        assert feedback.observations == ()
        assert feedback.suggestions == ()
        assert feedback.severity == "info"
        assert feedback.call_index == 0

    def test_creates_feedback_with_observations(self) -> None:
        obs = Observation(category="Pattern", description="Loop detected")
        feedback = Feedback(
            provider_name="Loop",
            summary="Possible loop",
            observations=(obs,),
        )

        assert len(feedback.observations) == 1
        assert feedback.observations[0].category == "Pattern"

    def test_creates_feedback_with_suggestions(self) -> None:
        feedback = Feedback(
            provider_name="Time",
            summary="Running low on time",
            suggestions=("Wrap up soon", "Summarize progress"),
        )

        assert len(feedback.suggestions) == 2

    def test_severity_levels(self) -> None:
        info = Feedback(provider_name="A", summary="Info", severity="info")
        caution = Feedback(provider_name="B", summary="Caution", severity="caution")
        warning = Feedback(provider_name="C", summary="Warning", severity="warning")

        assert info.severity == "info"
        assert caution.severity == "caution"
        assert warning.severity == "warning"


class TestFeedbackRender:
    """Tests for Feedback.render() method."""

    def test_render_basic_feedback(self) -> None:
        feedback = Feedback(provider_name="Test", summary="Status check")
        rendered = feedback.render()

        assert "<feedback provider='Test'>" in rendered
        assert "</feedback>" in rendered
        assert "Status check" in rendered

    def test_render_with_observations(self) -> None:
        obs = Observation(category="Pattern", description="Loop detected")
        feedback = Feedback(
            provider_name="Loop",
            summary="Possible loop",
            observations=(obs,),
        )
        rendered = feedback.render()

        assert "• Pattern: Loop detected" in rendered

    def test_render_with_suggestions(self) -> None:
        feedback = Feedback(
            provider_name="Time",
            summary="Low time",
            suggestions=("Wrap up", "Summarize"),
        )
        rendered = feedback.render()

        assert "→ Wrap up" in rendered
        assert "→ Summarize" in rendered

    def test_render_full_feedback(self) -> None:
        obs1 = Observation(category="Files", description="10 files read")
        obs2 = Observation(category="Time", description="5 minutes elapsed")
        feedback = Feedback(
            provider_name="Progress",
            summary="Making progress",
            observations=(obs1, obs2),
            suggestions=("Continue current approach",),
            severity="info",
        )
        rendered = feedback.render()

        assert "<feedback provider='Progress'>" in rendered
        assert "</feedback>" in rendered
        assert "Making progress" in rendered
        assert "• Files: 10 files read" in rendered
        assert "• Time: 5 minutes elapsed" in rendered
        assert "→ Continue current approach" in rendered

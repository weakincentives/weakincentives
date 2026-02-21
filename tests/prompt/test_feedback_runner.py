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

"""Tests for run_feedback_providers, PromptTemplate integration, and session storage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from weakincentives.prompt import (
    DeadlineFeedback,
    Feedback,
    FeedbackContext,
    FeedbackProviderConfig,
    FeedbackTrigger,
    Prompt,
    PromptTemplate,
    run_feedback_providers,
)
from weakincentives.runtime import InProcessDispatcher, Session
from weakincentives.runtime.events import ToolInvoked


def make_session() -> Session:
    """Create a session for testing."""
    return Session(dispatcher=InProcessDispatcher())


def make_prompt() -> Prompt[None]:
    """Create a minimal prompt for testing."""
    template: PromptTemplate[None] = PromptTemplate(
        ns="test", key="test-prompt", name="test"
    )
    return Prompt(template)


def make_tool_invoked(name: str = "test_tool") -> ToolInvoked:
    """Create a ToolInvoked event for testing."""
    return ToolInvoked(
        prompt_name="test",
        adapter="test",
        name=name,
        params=None,
        result=None,
        session_id=None,
        created_at=datetime.now(UTC),
    )


@dataclass(frozen=True)
class MockFeedbackProvider:
    """Test provider with configurable behavior."""

    should_run_return: bool = True
    feedback_summary: str = "Mock feedback"
    provider_name: str = "Mock"

    @property
    def name(self) -> str:
        return self.provider_name

    def should_run(self, *, context: FeedbackContext) -> bool:
        return self.should_run_return

    def provide(self, *, context: FeedbackContext) -> Feedback:
        return Feedback(provider_name=self.name, summary=self.feedback_summary)


# =============================================================================
# run_feedback_providers Tests
# =============================================================================


class TestRunFeedbackProviders:
    """Tests for the run_feedback_providers runner function."""

    def _make_context(self, tool_calls: int = 0) -> FeedbackContext:
        session = make_session()
        prompt = make_prompt()

        for i in range(tool_calls):
            session.dispatcher.dispatch(make_tool_invoked(f"tool_{i}"))

        return FeedbackContext(session=session, prompt=prompt)

    def test_returns_none_when_no_providers(self) -> None:
        context = self._make_context(tool_calls=5)

        result = run_feedback_providers(providers=(), context=context)

        assert result is None

    def test_returns_none_when_trigger_not_met(self) -> None:
        context = self._make_context(tool_calls=2)
        config = FeedbackProviderConfig(
            provider=MockFeedbackProvider(),
            trigger=FeedbackTrigger(every_n_calls=10),
        )

        result = run_feedback_providers(providers=(config,), context=context)

        assert result is None

    def test_returns_none_when_should_run_false(self) -> None:
        context = self._make_context(tool_calls=5)
        config = FeedbackProviderConfig(
            provider=MockFeedbackProvider(should_run_return=False),
            trigger=FeedbackTrigger(every_n_calls=3),
        )

        result = run_feedback_providers(providers=(config,), context=context)

        assert result is None

    def test_returns_rendered_feedback_when_triggered(self) -> None:
        context = self._make_context(tool_calls=5)
        config = FeedbackProviderConfig(
            provider=MockFeedbackProvider(feedback_summary="Test feedback"),
            trigger=FeedbackTrigger(every_n_calls=3),
        )

        result = run_feedback_providers(providers=(config,), context=context)

        assert result is not None
        assert "Test feedback" in result
        assert "<feedback provider='Mock'>" in result

    def test_stores_feedback_in_session(self) -> None:
        context = self._make_context(tool_calls=5)
        config = FeedbackProviderConfig(
            provider=MockFeedbackProvider(),
            trigger=FeedbackTrigger(every_n_calls=3),
        )

        run_feedback_providers(providers=(config,), context=context)

        latest = context.session[Feedback].latest()
        assert latest is not None
        assert latest.provider_name == "Mock"
        assert latest.call_index == 5  # Updated to current count

    def test_all_matching_providers_collected(self) -> None:
        context = self._make_context(tool_calls=5)
        configs = (
            FeedbackProviderConfig(
                provider=MockFeedbackProvider(feedback_summary="First"),
                trigger=FeedbackTrigger(every_n_calls=3),
            ),
            FeedbackProviderConfig(
                provider=MockFeedbackProvider(feedback_summary="Second"),
                trigger=FeedbackTrigger(every_n_calls=3),
            ),
        )

        result = run_feedback_providers(providers=configs, context=context)

        assert result is not None
        assert "First" in result
        assert "Second" in result
        # Verify both are rendered as separate blocks
        assert result.count("<feedback provider='Mock'>") == 2
        assert result.count("</feedback>") == 2

    def test_all_matching_feedback_stored_in_session(self) -> None:
        context = self._make_context(tool_calls=5)
        configs = (
            FeedbackProviderConfig(
                provider=MockFeedbackProvider(feedback_summary="First"),
                trigger=FeedbackTrigger(every_n_calls=3),
            ),
            FeedbackProviderConfig(
                provider=MockFeedbackProvider(feedback_summary="Second"),
                trigger=FeedbackTrigger(every_n_calls=3),
            ),
        )

        run_feedback_providers(providers=configs, context=context)

        all_feedback = context.session[Feedback].all()
        assert len(all_feedback) == 2
        summaries = {fb.summary for fb in all_feedback}
        assert "First" in summaries
        assert "Second" in summaries

    def test_only_matching_providers_included(self) -> None:
        context = self._make_context(tool_calls=5)
        configs = (
            FeedbackProviderConfig(
                provider=MockFeedbackProvider(feedback_summary="Matches"),
                trigger=FeedbackTrigger(every_n_calls=3),
            ),
            FeedbackProviderConfig(
                provider=MockFeedbackProvider(
                    feedback_summary="Skipped", should_run_return=False
                ),
                trigger=FeedbackTrigger(every_n_calls=3),
            ),
            FeedbackProviderConfig(
                provider=MockFeedbackProvider(feedback_summary="No trigger"),
                trigger=FeedbackTrigger(every_n_calls=100),  # Won't trigger
            ),
        )

        result = run_feedback_providers(providers=configs, context=context)

        assert result is not None
        assert "Matches" in result
        assert "Skipped" not in result
        assert "No trigger" not in result

    def test_providers_maintain_independent_trigger_cadences(self) -> None:
        """Each provider tracks its own feedback history for triggers."""
        session = make_session()
        prompt = make_prompt()

        # Provider A triggers every 3 calls, Provider B triggers every 5 calls
        configs = (
            FeedbackProviderConfig(
                provider=MockFeedbackProvider(
                    provider_name="ProviderA", feedback_summary="From A"
                ),
                trigger=FeedbackTrigger(every_n_calls=3),
            ),
            FeedbackProviderConfig(
                provider=MockFeedbackProvider(
                    provider_name="ProviderB", feedback_summary="From B"
                ),
                trigger=FeedbackTrigger(every_n_calls=5),
            ),
        )

        # Simulate tool calls and run feedback collection
        def run_at_call_count(n: int) -> str | None:
            # Add tool calls to reach count n
            while len(session[ToolInvoked].all()) < n:
                session.dispatcher.dispatch(make_tool_invoked("tool"))
            context = FeedbackContext(session=session, prompt=prompt)
            return run_feedback_providers(providers=configs, context=context)

        # At call 3: A triggers (3 calls since start), B doesn't (only 3, needs 5)
        result = run_at_call_count(3)
        assert result is not None
        assert "From A" in result
        assert "From B" not in result

        # At call 5: B triggers (5 calls since start), A doesn't (only 2 since last)
        result = run_at_call_count(5)
        assert result is not None
        assert "From A" not in result
        assert "From B" in result

        # At call 6: A triggers (3 calls since last at 3), B doesn't (1 since last)
        result = run_at_call_count(6)
        assert result is not None
        assert "From A" in result
        assert "From B" not in result

        # At call 9: A triggers (3 calls since last at 6), B doesn't (4 since last)
        result = run_at_call_count(9)
        assert result is not None
        assert "From A" in result
        assert "From B" not in result

        # At call 10: B triggers (5 calls since last at 5), A doesn't (1 since last)
        result = run_at_call_count(10)
        assert result is not None
        assert "From A" not in result
        assert "From B" in result


# =============================================================================
# PromptTemplate Integration Tests
# =============================================================================


class TestPromptTemplateFeedbackProviders:
    """Tests for feedback providers on PromptTemplate."""

    def test_creates_template_without_providers(self) -> None:
        template: PromptTemplate[None] = PromptTemplate(
            ns="test", key="test", name="test"
        )
        assert template.feedback_providers == ()

    def test_creates_template_with_providers(self) -> None:
        provider = DeadlineFeedback()
        config = FeedbackProviderConfig(
            provider=provider, trigger=FeedbackTrigger(every_n_seconds=30)
        )
        template: PromptTemplate[None] = PromptTemplate(
            ns="test", key="test", name="test", feedback_providers=[config]
        )

        assert len(template.feedback_providers) == 1
        assert template.feedback_providers[0].provider is provider

    def test_providers_converted_to_tuple(self) -> None:
        configs = [
            FeedbackProviderConfig(
                provider=DeadlineFeedback(),
                trigger=FeedbackTrigger(every_n_seconds=30),
            ),
            FeedbackProviderConfig(
                provider=MockFeedbackProvider(),
                trigger=FeedbackTrigger(every_n_calls=10),
            ),
        ]
        template: PromptTemplate[None] = PromptTemplate(
            ns="test", key="test", name="test", feedback_providers=configs
        )

        assert isinstance(template.feedback_providers, tuple)
        assert len(template.feedback_providers) == 2


class TestPromptFeedbackProviders:
    """Tests for feedback providers on Prompt."""

    def test_prompt_exposes_template_providers(self) -> None:
        provider = DeadlineFeedback()
        config = FeedbackProviderConfig(
            provider=provider, trigger=FeedbackTrigger(every_n_seconds=30)
        )
        template: PromptTemplate[None] = PromptTemplate(
            ns="test", key="test", name="test", feedback_providers=[config]
        )
        prompt: Prompt[None] = Prompt(template)

        assert len(prompt.feedback_providers) == 1
        assert prompt.feedback_providers[0].provider is provider

    def test_prompt_providers_returns_tuple(self) -> None:
        template: PromptTemplate[None] = PromptTemplate(
            ns="test", key="test", name="test"
        )
        prompt: Prompt[None] = Prompt(template)

        assert isinstance(prompt.feedback_providers, tuple)


# =============================================================================
# Feedback Session Storage Tests
# =============================================================================


class TestFeedbackSessionStorage:
    """Tests for Feedback storage in session slices."""

    def test_feedback_stored_in_session_slice(self) -> None:
        session = make_session()

        session[Feedback].append(Feedback(provider_name="A", summary="First"))

        stored = session[Feedback].latest()
        assert stored is not None
        assert stored.provider_name == "A"

    def test_multiple_feedback_stored(self) -> None:
        session = make_session()

        session[Feedback].append(Feedback(provider_name="A", summary="First"))
        session[Feedback].append(Feedback(provider_name="B", summary="Second"))

        all_feedback = session[Feedback].all()
        assert len(all_feedback) == 2

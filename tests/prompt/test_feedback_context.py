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

"""Tests for FeedbackContext and its helper methods."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from weakincentives.deadlines import Deadline
from weakincentives.prompt import (
    Feedback,
    FeedbackContext,
    Prompt,
    PromptResources,
    PromptTemplate,
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


# =============================================================================
# FeedbackContext Tests
# =============================================================================


class TestFeedbackContext:
    """Tests for the FeedbackContext class."""

    def test_creates_context_with_basic_fields(self) -> None:
        session = make_session()
        prompt = make_prompt()

        context = FeedbackContext(session=session, prompt=prompt)

        assert context.session is session
        assert context.prompt is prompt
        assert context.deadline is None

    def test_creates_context_with_deadline(self) -> None:
        session = make_session()
        prompt = make_prompt()
        deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(hours=1))

        context = FeedbackContext(session=session, prompt=prompt, deadline=deadline)

        assert context.deadline is deadline

    def test_resources_returns_prompt_resources(self) -> None:
        session = make_session()
        prompt = make_prompt()
        context = FeedbackContext(session=session, prompt=prompt)

        assert isinstance(context.resources, PromptResources)

    def test_filesystem_returns_none_when_not_registered(self) -> None:
        session = make_session()
        prompt = make_prompt()
        context = FeedbackContext(session=session, prompt=prompt)

        with prompt.resources:
            assert context.filesystem is None

    def test_last_feedback_returns_none_when_empty(self) -> None:
        session = make_session()
        prompt = make_prompt()
        context = FeedbackContext(session=session, prompt=prompt)

        assert context.last_feedback is None

    def test_last_feedback_returns_most_recent(self) -> None:
        session = make_session()
        prompt = make_prompt()

        session[Feedback].append(
            Feedback(provider_name="A", summary="First", prompt_name="test")
        )
        session[Feedback].append(
            Feedback(provider_name="B", summary="Second", prompt_name="test")
        )

        context = FeedbackContext(session=session, prompt=prompt)

        assert context.last_feedback is not None
        assert context.last_feedback.summary == "Second"

    def test_tool_call_count_returns_zero_when_empty(self) -> None:
        session = make_session()
        prompt = make_prompt()
        context = FeedbackContext(session=session, prompt=prompt)

        assert context.tool_call_count == 0

    def test_tool_call_count_returns_correct_count(self) -> None:
        session = make_session()
        prompt = make_prompt()

        for i in range(3):
            session.dispatcher.dispatch(make_tool_invoked(f"tool_{i}"))

        context = FeedbackContext(session=session, prompt=prompt)

        assert context.tool_call_count == 3

    def test_tool_calls_since_last_feedback_all_when_no_feedback(self) -> None:
        session = make_session()
        prompt = make_prompt()

        for i in range(5):
            session.dispatcher.dispatch(make_tool_invoked(f"tool_{i}"))

        context = FeedbackContext(session=session, prompt=prompt)

        assert context.tool_calls_since_last_feedback() == 5

    def test_tool_calls_since_last_feedback_counts_after_last(self) -> None:
        session = make_session()
        prompt = make_prompt()

        # 3 tool calls, then feedback, then 2 more
        for i in range(3):
            session.dispatcher.dispatch(make_tool_invoked(f"tool_{i}"))
        session[Feedback].append(
            Feedback(
                provider_name="A", summary="Test", call_index=3, prompt_name="test"
            )
        )
        for i in range(2):
            session.dispatcher.dispatch(make_tool_invoked(f"more_{i}"))

        context = FeedbackContext(session=session, prompt=prompt)

        # 5 total - 3 at last feedback = 2 since
        assert context.tool_calls_since_last_feedback() == 2

    def test_last_feedback_for_provider_returns_none_when_no_feedback(self) -> None:
        session = make_session()
        prompt = make_prompt()
        context = FeedbackContext(session=session, prompt=prompt)

        assert context.last_feedback_for_provider("A") is None

    def test_last_feedback_for_provider_filters_by_provider(self) -> None:
        session = make_session()
        prompt = make_prompt()

        session[Feedback].append(
            Feedback(provider_name="A", summary="From A", prompt_name="test")
        )
        session[Feedback].append(
            Feedback(provider_name="B", summary="From B", prompt_name="test")
        )
        session[Feedback].append(
            Feedback(provider_name="A", summary="From A again", prompt_name="test")
        )

        context = FeedbackContext(session=session, prompt=prompt)

        # Should get the most recent from provider A
        last_a = context.last_feedback_for_provider("A")
        assert last_a is not None
        assert last_a.summary == "From A again"

        # Should get the only one from provider B
        last_b = context.last_feedback_for_provider("B")
        assert last_b is not None
        assert last_b.summary == "From B"

        # Provider C has no feedback
        assert context.last_feedback_for_provider("C") is None

    def test_tool_calls_since_last_feedback_for_provider_all_when_none(self) -> None:
        session = make_session()
        prompt = make_prompt()

        for i in range(5):
            session.dispatcher.dispatch(make_tool_invoked(f"tool_{i}"))

        context = FeedbackContext(session=session, prompt=prompt)

        # No feedback from any provider, so all tool calls count
        assert context.tool_calls_since_last_feedback_for_provider("A") == 5

    def test_tool_calls_since_last_feedback_for_provider_independent(self) -> None:
        session = make_session()
        prompt = make_prompt()

        # 3 tool calls
        for i in range(3):
            session.dispatcher.dispatch(make_tool_invoked(f"tool_{i}"))

        # Provider A gives feedback at call 3
        session[Feedback].append(
            Feedback(
                provider_name="A", summary="From A", call_index=3, prompt_name="test"
            )
        )

        # 2 more tool calls
        for i in range(2):
            session.dispatcher.dispatch(make_tool_invoked(f"more_{i}"))

        context = FeedbackContext(session=session, prompt=prompt)

        # Provider A: 5 total - 3 at last feedback = 2 since
        assert context.tool_calls_since_last_feedback_for_provider("A") == 2
        # Provider B: no feedback, so all 5 tool calls count
        assert context.tool_calls_since_last_feedback_for_provider("B") == 5

    def test_recent_tool_calls_returns_last_n(self) -> None:
        session = make_session()
        prompt = make_prompt()

        for i in range(5):
            session.dispatcher.dispatch(make_tool_invoked(f"tool_{i}"))

        context = FeedbackContext(session=session, prompt=prompt)
        recent = context.recent_tool_calls(3)

        assert len(recent) == 3
        assert recent[0].name == "tool_2"
        assert recent[1].name == "tool_3"
        assert recent[2].name == "tool_4"

    def test_recent_tool_calls_returns_all_when_fewer_than_n(self) -> None:
        session = make_session()
        prompt = make_prompt()

        for i in range(2):
            session.dispatcher.dispatch(make_tool_invoked(f"tool_{i}"))

        context = FeedbackContext(session=session, prompt=prompt)
        recent = context.recent_tool_calls(5)

        assert len(recent) == 2

    def test_recent_tool_calls_returns_empty_when_n_is_zero(self) -> None:
        session = make_session()
        prompt = make_prompt()

        for i in range(3):
            session.dispatcher.dispatch(make_tool_invoked(f"tool_{i}"))

        context = FeedbackContext(session=session, prompt=prompt)
        recent = context.recent_tool_calls(0)

        assert len(recent) == 0

    def test_recent_tool_calls_returns_empty_when_no_tool_invoked_events(self) -> None:
        session = make_session()
        prompt = make_prompt()

        context = FeedbackContext(session=session, prompt=prompt)
        recent = context.recent_tool_calls(5)

        assert len(recent) == 0

    def test_tool_call_count_scoped_to_current_prompt(self) -> None:
        """Tool call count only includes events for the current prompt."""
        session = make_session()
        prompt = make_prompt()  # name="test"

        # Add tool calls from different prompts
        session.dispatcher.dispatch(make_tool_invoked("tool_1"))  # prompt_name="test"
        session.dispatcher.dispatch(make_tool_invoked("tool_2"))  # prompt_name="test"
        session.dispatcher.dispatch(
            ToolInvoked(
                prompt_name="other_prompt",  # Different prompt
                adapter="test",
                name="other_tool",
                params=None,
                result=None,
                session_id=None,
                created_at=datetime.now(UTC),
            )
        )
        session.dispatcher.dispatch(make_tool_invoked("tool_3"))  # prompt_name="test"

        context = FeedbackContext(session=session, prompt=prompt)

        # Only 3 calls for "test" prompt, not the 4th from "other_prompt"
        assert context.tool_call_count == 3

    def test_recent_tool_calls_scoped_to_current_prompt(self) -> None:
        """Recent tool calls only includes events for the current prompt."""
        session = make_session()
        prompt = make_prompt()

        session.dispatcher.dispatch(make_tool_invoked("tool_1"))
        session.dispatcher.dispatch(
            ToolInvoked(
                prompt_name="other_prompt",
                adapter="test",
                name="other_tool",
                params=None,
                result=None,
                session_id=None,
                created_at=datetime.now(UTC),
            )
        )
        session.dispatcher.dispatch(make_tool_invoked("tool_2"))

        context = FeedbackContext(session=session, prompt=prompt)
        recent = context.recent_tool_calls(5)

        # Only 2 calls for "test" prompt
        assert len(recent) == 2
        assert recent[0].name == "tool_1"
        assert recent[1].name == "tool_2"

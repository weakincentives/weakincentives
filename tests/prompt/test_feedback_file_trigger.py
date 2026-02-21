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

"""Tests for FileCreatedTrigger, StaticFeedbackProvider, and related functionality."""

from __future__ import annotations

import pytest

from weakincentives.prompt import (
    FeedbackContext,
    FeedbackProviderConfig,
    FeedbackTrigger,
    FileCreatedTrigger,
    FileCreatedTriggerState,
    Prompt,
    PromptTemplate,
    StaticFeedbackProvider,
    run_feedback_providers,
)
from weakincentives.prompt.feedback import _should_trigger
from weakincentives.runtime import InProcessDispatcher, Session


def make_session() -> Session:
    """Create a session for testing."""
    return Session(dispatcher=InProcessDispatcher())


def make_prompt() -> Prompt[None]:
    """Create a minimal prompt for testing."""
    template: PromptTemplate[None] = PromptTemplate(
        ns="test", key="test-prompt", name="test"
    )
    return Prompt(template)


# =============================================================================
# FileCreatedTrigger Tests
# =============================================================================


class TestFileCreatedTrigger:
    """Tests for FileCreatedTrigger configuration."""

    def test_creates_trigger_with_filename(self) -> None:
        trigger = FileCreatedTrigger(filename="AGENTS.md")

        assert trigger.filename == "AGENTS.md"

    def test_trigger_is_frozen(self) -> None:
        trigger = FileCreatedTrigger(filename="test.txt")

        with pytest.raises(AttributeError):
            trigger.filename = "changed.txt"  # type: ignore[misc]


class TestFileCreatedTriggerState:
    """Tests for FileCreatedTriggerState session slice."""

    def test_creates_empty_state(self) -> None:
        state = FileCreatedTriggerState()

        assert state.fired_filenames == frozenset()

    def test_creates_state_with_fired_filenames(self) -> None:
        state = FileCreatedTriggerState(fired_filenames=frozenset({"a.md", "b.md"}))

        assert "a.md" in state.fired_filenames
        assert "b.md" in state.fired_filenames

    def test_state_stored_in_session(self) -> None:
        session = make_session()
        state = FileCreatedTriggerState(fired_filenames=frozenset({"test.md"}))
        session[FileCreatedTriggerState].seed(state)

        retrieved = session[FileCreatedTriggerState].latest()
        assert retrieved is not None
        assert "test.md" in retrieved.fired_filenames


class TestFeedbackTriggerWithFileCreated:
    """Tests for FeedbackTrigger with on_file_created."""

    def test_creates_trigger_with_file_created(self) -> None:
        trigger = FeedbackTrigger(
            on_file_created=FileCreatedTrigger(filename="AGENTS.md")
        )

        assert trigger.on_file_created is not None
        assert trigger.on_file_created.filename == "AGENTS.md"
        assert trigger.every_n_calls is None
        assert trigger.every_n_seconds is None

    def test_creates_trigger_with_combined_conditions(self) -> None:
        trigger = FeedbackTrigger(
            every_n_calls=10,
            on_file_created=FileCreatedTrigger(filename="test.md"),
        )

        assert trigger.every_n_calls == 10
        assert trigger.on_file_created is not None


class TestShouldTriggerWithFileCreated:
    """Tests for _should_trigger with file creation triggers."""

    def _make_context_with_filesystem(
        self,
        files: dict[str, str] | None = None,
        fired_filenames: frozenset[str] | None = None,
    ) -> FeedbackContext:
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
        from weakincentives.filesystem import Filesystem

        session = make_session()
        fs = InMemoryFilesystem()

        # Create files in filesystem
        if files:
            for path, content in files.items():
                fs.write(path, content)

        # Set up fired state if provided
        if fired_filenames:
            session[FileCreatedTriggerState].seed(
                FileCreatedTriggerState(fired_filenames=fired_filenames)
            )

        # Create template and bind filesystem resource
        template: PromptTemplate[None] = PromptTemplate(
            ns="test",
            key="test-prompt",
            name="test",
        )
        prompt: Prompt[None] = Prompt(template)
        prompt = prompt.bind(resources={Filesystem: fs})

        # We need to enter the resource context for the filesystem to be available
        prompt.resources.__enter__()

        return FeedbackContext(session=session, prompt=prompt)

    def test_file_trigger_fires_when_file_exists(self) -> None:
        trigger = FeedbackTrigger(
            on_file_created=FileCreatedTrigger(filename="AGENTS.md")
        )
        context = self._make_context_with_filesystem(files={"AGENTS.md": "content"})

        assert _should_trigger(trigger, context, "TestProvider") is True

    def test_file_trigger_does_not_fire_when_file_missing(self) -> None:
        trigger = FeedbackTrigger(
            on_file_created=FileCreatedTrigger(filename="AGENTS.md")
        )
        context = self._make_context_with_filesystem(files={})

        assert _should_trigger(trigger, context, "TestProvider") is False

    def test_file_trigger_does_not_fire_when_already_fired(self) -> None:
        trigger = FeedbackTrigger(
            on_file_created=FileCreatedTrigger(filename="AGENTS.md")
        )
        context = self._make_context_with_filesystem(
            files={"AGENTS.md": "content"},
            fired_filenames=frozenset({"AGENTS.md"}),
        )

        assert _should_trigger(trigger, context, "TestProvider") is False

    def test_file_trigger_does_not_fire_without_filesystem(self) -> None:
        trigger = FeedbackTrigger(
            on_file_created=FileCreatedTrigger(filename="AGENTS.md")
        )
        # Context without filesystem
        session = make_session()
        prompt = make_prompt()
        context = FeedbackContext(session=session, prompt=prompt)

        with prompt.resources:
            assert _should_trigger(trigger, context, "TestProvider") is False


class TestRunFeedbackProvidersWithFileCreated:
    """Tests for run_feedback_providers with file creation triggers."""

    def _make_context_with_filesystem(
        self, files: dict[str, str] | None = None
    ) -> FeedbackContext:
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
        from weakincentives.filesystem import Filesystem

        session = make_session()
        fs = InMemoryFilesystem()

        if files:
            for path, content in files.items():
                fs.write(path, content)

        template: PromptTemplate[None] = PromptTemplate(
            ns="test",
            key="test-prompt",
            name="test",
        )
        prompt: Prompt[None] = Prompt(template)
        prompt = prompt.bind(resources={Filesystem: fs})
        prompt.resources.__enter__()

        return FeedbackContext(session=session, prompt=prompt)

    def test_marks_file_trigger_as_fired(self) -> None:
        context = self._make_context_with_filesystem(files={"AGENTS.md": "content"})
        config = FeedbackProviderConfig(
            provider=StaticFeedbackProvider(feedback="File detected"),
            trigger=FeedbackTrigger(
                on_file_created=FileCreatedTrigger(filename="AGENTS.md")
            ),
        )

        result = run_feedback_providers(providers=(config,), context=context)

        assert result is not None
        assert "File detected" in result

        # Check trigger is marked as fired
        state = context.session[FileCreatedTriggerState].latest()
        assert state is not None
        assert "AGENTS.md" in state.fired_filenames

    def test_file_trigger_fires_only_once(self) -> None:
        context = self._make_context_with_filesystem(files={"AGENTS.md": "content"})
        config = FeedbackProviderConfig(
            provider=StaticFeedbackProvider(feedback="File detected"),
            trigger=FeedbackTrigger(
                on_file_created=FileCreatedTrigger(filename="AGENTS.md")
            ),
        )

        # First call should trigger
        result1 = run_feedback_providers(providers=(config,), context=context)
        assert result1 is not None

        # Second call should NOT trigger (already fired)
        result2 = run_feedback_providers(providers=(config,), context=context)
        assert result2 is None

    def test_multiple_file_triggers_accumulate_state(self) -> None:
        context = self._make_context_with_filesystem(
            files={"AGENTS.md": "content", "README.md": "readme"}
        )
        config1 = FeedbackProviderConfig(
            provider=StaticFeedbackProvider(feedback="AGENTS.md detected"),
            trigger=FeedbackTrigger(
                on_file_created=FileCreatedTrigger(filename="AGENTS.md")
            ),
        )
        config2 = FeedbackProviderConfig(
            provider=StaticFeedbackProvider(feedback="README.md detected"),
            trigger=FeedbackTrigger(
                on_file_created=FileCreatedTrigger(filename="README.md")
            ),
        )

        # First trigger fires for AGENTS.md
        result1 = run_feedback_providers(providers=(config1,), context=context)
        assert result1 is not None
        assert "AGENTS.md" in result1

        # Second trigger fires for README.md (existing state gets updated)
        result2 = run_feedback_providers(providers=(config2,), context=context)
        assert result2 is not None
        assert "README.md" in result2

        # Verify both filenames are tracked in state
        state = context.session[FileCreatedTriggerState].latest()
        assert state is not None
        assert "AGENTS.md" in state.fired_filenames
        assert "README.md" in state.fired_filenames


# =============================================================================
# StaticFeedbackProvider Tests
# =============================================================================


class TestStaticFeedbackProvider:
    """Tests for the built-in StaticFeedbackProvider."""

    def _make_context(self) -> FeedbackContext:
        session = make_session()
        prompt = make_prompt()
        return FeedbackContext(session=session, prompt=prompt)

    def test_name_property(self) -> None:
        provider = StaticFeedbackProvider(feedback="Test message")
        assert provider.name == "Static"

    def test_should_run_always_returns_true(self) -> None:
        provider = StaticFeedbackProvider(feedback="Test message")
        context = self._make_context()

        assert provider.should_run(context=context) is True

    def test_provide_returns_static_feedback(self) -> None:
        provider = StaticFeedbackProvider(
            feedback="AGENTS.md detected. Follow the conventions within."
        )
        context = self._make_context()

        feedback = provider.provide(context=context)

        assert feedback.provider_name == "Static"
        assert feedback.summary == "AGENTS.md detected. Follow the conventions within."
        assert feedback.severity == "info"

    def test_feedback_renders_correctly(self) -> None:
        provider = StaticFeedbackProvider(feedback="Custom guidance message")
        context = self._make_context()

        feedback = provider.provide(context=context)
        rendered = feedback.render()

        assert "<feedback provider='Static'>" in rendered
        assert "</feedback>" in rendered
        assert "Custom guidance message" in rendered

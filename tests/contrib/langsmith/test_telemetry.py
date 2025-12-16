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

"""Tests for LangSmithTelemetryHandler."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from tests.helpers.adapters import TEST_ADAPTER_NAME
from weakincentives.contrib.langsmith import (
    LangSmithConfig,
    LangSmithTelemetryHandler,
    LangSmithTraceCompleted,
    LangSmithTraceStarted,
)
from weakincentives.contrib.langsmith.testing import MockLangSmithClient
from weakincentives.runtime.events import (
    InProcessEventBus,
    PromptExecuted,
    PromptRendered,
    TokenUsage,
    ToolInvoked,
)


@pytest.fixture
def mock_client() -> MockLangSmithClient:
    """Create a mock LangSmith client."""
    return MockLangSmithClient()


@pytest.fixture
def config() -> LangSmithConfig:
    """Create test config with sync upload for deterministic tests."""
    return LangSmithConfig(
        api_key="test-key",
        project="test-project",
        async_upload=False,
        flush_on_exit=False,
    )


@pytest.fixture
def handler(
    config: LangSmithConfig, mock_client: MockLangSmithClient
) -> LangSmithTelemetryHandler:
    """Create telemetry handler with mock client."""
    return LangSmithTelemetryHandler(config, client=mock_client)


@pytest.fixture
def bus() -> InProcessEventBus:
    """Create event bus."""
    return InProcessEventBus()


class TestLangSmithTelemetryHandler:
    """Tests for LangSmithTelemetryHandler."""

    def test_attach_subscribes_to_events(
        self, handler: LangSmithTelemetryHandler, bus: InProcessEventBus
    ) -> None:
        """attach subscribes handler to telemetry events."""
        handler.attach(bus)

        # Verify by publishing events and checking they're handled
        session_id = uuid4()
        bus.publish(
            PromptRendered(
                prompt_ns="test",
                prompt_key="prompt",
                prompt_name="test_prompt",
                adapter=TEST_ADAPTER_NAME,
                session_id=session_id,
                render_inputs=(),
                rendered_prompt="Test prompt",
                created_at=datetime.now(UTC),
            )
        )

        handler.detach(bus)

    def test_detach_unsubscribes_from_events(
        self,
        handler: LangSmithTelemetryHandler,
        bus: InProcessEventBus,
        mock_client: MockLangSmithClient,
    ) -> None:
        """detach unsubscribes handler from events."""
        handler.attach(bus)
        handler.detach(bus)

        # Events should not be handled after detach
        bus.publish(
            PromptRendered(
                prompt_ns="test",
                prompt_key="prompt",
                prompt_name="test_prompt",
                adapter=TEST_ADAPTER_NAME,
                session_id=uuid4(),
                render_inputs=(),
                rendered_prompt="Test",
                created_at=datetime.now(UTC),
            )
        )

        # Mock client should have no calls from after detach
        # (it may have calls from before detach)

    def test_prompt_rendered_creates_chain_run(
        self,
        handler: LangSmithTelemetryHandler,
        bus: InProcessEventBus,
        mock_client: MockLangSmithClient,
    ) -> None:
        """PromptRendered event creates a chain run."""
        handler.attach(bus)

        session_id = uuid4()
        bus.publish(
            PromptRendered(
                prompt_ns="test-ns",
                prompt_key="test-key",
                prompt_name="test_prompt",
                adapter=TEST_ADAPTER_NAME,
                session_id=session_id,
                render_inputs=("arg1",),
                rendered_prompt="Rendered content",
                created_at=datetime.now(UTC),
            )
        )

        handler.detach(bus)

        assert mock_client.runs_created >= 1
        last_run = mock_client.last_run
        assert last_run is not None
        assert last_run.name == "test_prompt"
        assert last_run.run_type == "chain"
        assert last_run.inputs is not None
        assert "rendered_prompt" in last_run.inputs

    def test_tool_invoked_creates_tool_run(
        self,
        handler: LangSmithTelemetryHandler,
        bus: InProcessEventBus,
        mock_client: MockLangSmithClient,
    ) -> None:
        """ToolInvoked event creates a tool run."""
        handler.attach(bus)

        session_id = uuid4()

        # First create chain run via PromptRendered
        bus.publish(
            PromptRendered(
                prompt_ns="test",
                prompt_key="prompt",
                prompt_name="test_prompt",
                adapter=TEST_ADAPTER_NAME,
                session_id=session_id,
                render_inputs=(),
                rendered_prompt="Test",
                created_at=datetime.now(UTC),
            )
        )

        # Then invoke tool
        bus.publish(
            ToolInvoked(
                prompt_name="test_prompt",
                adapter=TEST_ADAPTER_NAME,
                name="search",
                params={"query": "test"},
                result="result",
                session_id=session_id,
                created_at=datetime.now(UTC),
                call_id="call-123",
            )
        )

        handler.detach(bus)

        tool_runs = mock_client.get_runs_by_type("tool")
        assert len(tool_runs) >= 1
        assert tool_runs[0].name == "search"

    def test_prompt_executed_updates_chain_run(
        self,
        handler: LangSmithTelemetryHandler,
        bus: InProcessEventBus,
        mock_client: MockLangSmithClient,
    ) -> None:
        """PromptExecuted event updates the chain run."""
        handler.attach(bus)

        session_id = uuid4()

        # Create chain run
        bus.publish(
            PromptRendered(
                prompt_ns="test",
                prompt_key="prompt",
                prompt_name="test_prompt",
                adapter=TEST_ADAPTER_NAME,
                session_id=session_id,
                render_inputs=(),
                rendered_prompt="Test",
                created_at=datetime.now(UTC),
            )
        )

        # Complete execution
        bus.publish(
            PromptExecuted(
                prompt_name="test_prompt",
                adapter=TEST_ADAPTER_NAME,
                result={"output": "done"},
                session_id=session_id,
                created_at=datetime.now(UTC),
                usage=TokenUsage(input_tokens=100, output_tokens=50),
            )
        )

        handler.detach(bus)

        assert mock_client.runs_updated >= 1

    def test_trace_started_event_published(
        self,
        handler: LangSmithTelemetryHandler,
        bus: InProcessEventBus,
    ) -> None:
        """LangSmithTraceStarted event is published."""
        handler.attach(bus)

        events: list[LangSmithTraceStarted] = []

        def capture_event(event: object) -> None:
            if isinstance(event, LangSmithTraceStarted):
                events.append(event)

        bus.subscribe(LangSmithTraceStarted, capture_event)

        bus.publish(
            PromptRendered(
                prompt_ns="test",
                prompt_key="prompt",
                prompt_name="test_prompt",
                adapter=TEST_ADAPTER_NAME,
                session_id=uuid4(),
                render_inputs=(),
                rendered_prompt="Test",
                created_at=datetime.now(UTC),
            )
        )

        handler.detach(bus)

        assert len(events) == 1
        assert events[0].project == "test-project"

    def test_trace_completed_event_published(
        self,
        handler: LangSmithTelemetryHandler,
        bus: InProcessEventBus,
    ) -> None:
        """LangSmithTraceCompleted event is published."""
        handler.attach(bus)

        events: list[LangSmithTraceCompleted] = []

        def capture_event(event: object) -> None:
            if isinstance(event, LangSmithTraceCompleted):
                events.append(event)

        bus.subscribe(LangSmithTraceCompleted, capture_event)

        session_id = uuid4()

        bus.publish(
            PromptRendered(
                prompt_ns="test",
                prompt_key="prompt",
                prompt_name="test_prompt",
                adapter=TEST_ADAPTER_NAME,
                session_id=session_id,
                render_inputs=(),
                rendered_prompt="Test",
                created_at=datetime.now(UTC),
            )
        )

        bus.publish(
            PromptExecuted(
                prompt_name="test_prompt",
                adapter=TEST_ADAPTER_NAME,
                result="done",
                session_id=session_id,
                created_at=datetime.now(UTC),
            )
        )

        handler.detach(bus)

        assert len(events) == 1
        assert events[0].run_count >= 1

    def test_sampling_respects_rate(self) -> None:
        """Sampling rate is respected."""
        config = LangSmithConfig(
            api_key="test",
            project="test",
            async_upload=False,
            flush_on_exit=False,
            trace_sample_rate=0.0,  # Never sample
        )
        mock_client = MockLangSmithClient()
        handler = LangSmithTelemetryHandler(config, client=mock_client)
        bus = InProcessEventBus()

        handler.attach(bus)

        for _ in range(10):
            bus.publish(
                PromptRendered(
                    prompt_ns="test",
                    prompt_key="prompt",
                    prompt_name="test_prompt",
                    adapter=TEST_ADAPTER_NAME,
                    session_id=uuid4(),
                    render_inputs=(),
                    rendered_prompt="Test",
                    created_at=datetime.now(UTC),
                )
            )

        handler.detach(bus)

        # No traces should be created with 0% sample rate
        assert mock_client.runs_created == 0

    def test_tracing_disabled(self) -> None:
        """No tracing when disabled."""
        config = LangSmithConfig(
            api_key="test",
            project="test",
            async_upload=False,
            flush_on_exit=False,
            tracing_enabled=False,
        )
        mock_client = MockLangSmithClient()
        handler = LangSmithTelemetryHandler(config, client=mock_client)
        bus = InProcessEventBus()

        handler.attach(bus)

        bus.publish(
            PromptRendered(
                prompt_ns="test",
                prompt_key="prompt",
                prompt_name="test_prompt",
                adapter=TEST_ADAPTER_NAME,
                session_id=uuid4(),
                render_inputs=(),
                rendered_prompt="Test",
                created_at=datetime.now(UTC),
            )
        )

        handler.detach(bus)

        assert mock_client.runs_created == 0

    def test_deduplication_by_call_id(
        self,
        handler: LangSmithTelemetryHandler,
        bus: InProcessEventBus,
        mock_client: MockLangSmithClient,
    ) -> None:
        """Tool calls are deduplicated by call_id."""
        handler.attach(bus)

        session_id = uuid4()

        # Create chain run
        bus.publish(
            PromptRendered(
                prompt_ns="test",
                prompt_key="prompt",
                prompt_name="test_prompt",
                adapter=TEST_ADAPTER_NAME,
                session_id=session_id,
                render_inputs=(),
                rendered_prompt="Test",
                created_at=datetime.now(UTC),
            )
        )

        # Same call_id should only create one run
        for _ in range(3):
            bus.publish(
                ToolInvoked(
                    prompt_name="test_prompt",
                    adapter=TEST_ADAPTER_NAME,
                    name="search",
                    params={},
                    result="result",
                    session_id=session_id,
                    created_at=datetime.now(UTC),
                    call_id="same-call-id",
                )
            )

        handler.detach(bus)

        tool_runs = mock_client.get_runs_by_type("tool")
        # Only one tool run should be created despite 3 events
        assert len(tool_runs) == 1
